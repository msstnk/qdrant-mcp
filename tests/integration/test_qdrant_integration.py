"""Integration tests for OpenAI embeddings and Qdrant operations."""

from __future__ import annotations

import os
import unittest
import uuid
from pathlib import Path

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from qdrant_mcp.server import (
    CHUNK_HEADER_MAX_CHARS,
    DEFAULT_CHUNK_HEADER_MODEL,
    generate_chunk_header,
)


def load_dotenv(dotenv_path: Path) -> None:
    """Load key=value pairs into environment variables when not already set."""

    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class QdrantIntegrationTest(unittest.TestCase):
    """Exercises vector insert, query, facet, and delete against Qdrant."""

    @classmethod
    def setUpClass(cls) -> None:
        qdrant_url = os.getenv("QDRANT_URL")
        if not qdrant_url:
            raise unittest.SkipTest("QDRANT_URL is not set")

        cls.collection = f"it_{uuid.uuid4().hex[:12]}"
        cls.qdrant = QdrantClient(
            url=qdrant_url,
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        cls.embeddings = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        cls.qdrant.create_collection(
            collection_name=cls.collection,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        cls.qdrant.create_payload_index(
            collection_name=cls.collection,
            field_name="category",
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )
        cls.qdrant.create_payload_index(
            collection_name=cls.collection,
            field_name="path",
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "qdrant") and hasattr(cls, "collection"):
            cls.qdrant.delete_collection(collection_name=cls.collection)

    def test_openai_embeddings_and_qdrant_flow(self) -> None:
        """Embeds sample texts and verifies indexed retrieval and deletion flow."""

        response = self.embeddings.embeddings.create(
            model="text-embedding-3-small",
            input=[
                "Qdrant payload index for category and path",
                "Delete points by path in Qdrant",
            ],
        )
        self.assertEqual(2, len(response.data))

        points = [
            PointStruct(
                id=1,
                vector=response.data[0].embedding,
                payload={"category": "qdrant", "path": "/docs/a.md", "text": "alpha"},
            ),
            PointStruct(
                id=2,
                vector=response.data[1].embedding,
                payload={"category": "qdrant", "path": "/docs/b.md", "text": "beta"},
            ),
        ]
        self.qdrant.upsert(collection_name=self.collection, points=points)

        hits = self.qdrant.query_points(
            collection_name=self.collection,
            query=response.data[0].embedding,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="category",
                        match=MatchValue(value="qdrant"),
                    )
                ]
            ),
            limit=5,
            with_payload=True,
        ).points
        self.assertGreaterEqual(len(hits), 1)

        categories = self.qdrant.facet(
            collection_name=self.collection,
            key="category",
            limit=10,
        ).hits
        self.assertEqual(["qdrant"], [str(hit.value) for hit in categories])

        paths = self.qdrant.facet(
            collection_name=self.collection,
            key="path",
            facet_filter=Filter(
                must=[
                    FieldCondition(
                        key="category",
                        match=MatchValue(value="qdrant"),
                    )
                ]
            ),
            limit=10,
        ).hits
        self.assertEqual(
            {"/docs/a.md", "/docs/b.md"},
            {str(hit.value) for hit in paths},
        )

        self.qdrant.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[FieldCondition(key="path", match=MatchValue(value="/docs/a.md"))]
            ),
            wait=True,
        )
        remaining = self.qdrant.facet(
            collection_name=self.collection,
            key="path",
            facet_filter=Filter(
                must=[
                    FieldCondition(
                        key="category",
                        match=MatchValue(value="qdrant"),
                    )
                ]
            ),
            limit=10,
        ).hits
        self.assertEqual({"/docs/b.md"}, {str(hit.value) for hit in remaining})

    def test_openai_responses_api_chunk_header(self) -> None:
        """Verifies Responses API can generate a normalized chunk header."""

        header = generate_chunk_header(
            openai_client=self.embeddings,
            markdown="# Qdrant Guide\n\nPayload filtering and facet examples.",
            model=DEFAULT_CHUNK_HEADER_MODEL,
        )

        self.assertTrue(header.endswith(": "))
        self.assertLessEqual(len(header), CHUNK_HEADER_MAX_CHARS)


if __name__ == "__main__":
    unittest.main()
