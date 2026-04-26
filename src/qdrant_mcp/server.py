"""MCP server for document ingestion and vector search backed by Qdrant."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from markitdown import MarkItDown
from mcp.server.fastmcp import FastMCP
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Condition,
    Distance,
    FieldCondition,
    FilterSelector,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

DEFAULT_COLLECTION = "docs"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHUNK_HEADER_MODEL = "gpt-5.4-mini"
CHUNK_HEADER_SOURCE_BYTES = 4096
CHUNK_HEADER_MAX_CHARS = 64
INDEXED_PAYLOAD_FIELDS = ("category", "path")


@dataclass(frozen=True)
class ChunkRecord:
    """A chunk extracted from a source document."""

    chunk_index: int
    text: str


@dataclass(frozen=True)
class IngestConfig:
    """Configuration for chunking and embedding during file ingestion."""

    category: str
    chunk_size: int
    overlap_ratio: float
    embedding_model: str
    chunk_header_mode: Literal["enabled", "disabled"]
    chunk_header_model: str


class ChunkHeaderOutput(BaseModel):
    """Structured output schema for generated chunk headers."""

    chunk_header: str = Field(min_length=1, max_length=CHUNK_HEADER_MAX_CHARS)


class IngestOptions(BaseModel):
    """Optional ingest parameters for chunking and embedding."""

    chunk_size: int = 1200
    overlap_ratio: float = 0.15
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chunk_header_mode: Literal["enabled", "disabled"] = "enabled"


def chunk_text(text: str, chunk_size: int, overlap_ratio: float) -> list[ChunkRecord]:
    """Split text into overlapping chunks and skip whitespace-only chunks."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if not 0 <= overlap_ratio < 1:
        raise ValueError("overlap_ratio must be in range [0, 1)")

    overlap_size = int(chunk_size * overlap_ratio)
    step = max(1, chunk_size - overlap_size)

    records: list[ChunkRecord] = []
    cursor = 0
    chunk_index = 0
    text_length = len(text)

    while cursor < text_length:
        chunk = text[cursor : cursor + chunk_size]
        if chunk.strip():
            records.append(ChunkRecord(chunk_index=chunk_index, text=chunk))
            chunk_index += 1
        cursor += step

    return records


def document_to_markdown(markitdown: MarkItDown, path: Path) -> str:
    """Convert a document file to markdown text with MarkItDown."""

    result = markitdown.convert(str(path))
    markdown = getattr(result, "text_content", None)
    if not isinstance(markdown, str) or not markdown.strip():
        raise ValueError(f"MarkItDown produced empty markdown for: {path}")
    return markdown


def first_n_utf8_bytes(text: str, max_bytes: int) -> str:
    """Return a UTF-8 safe string clipped to the first max_bytes bytes."""

    if max_bytes < 1:
        raise ValueError("max_bytes must be >= 1")
    return text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


def normalize_chunk_header(header: str) -> str:
    """Normalize chunk header format to a single-line '<title>: ' prefix."""

    normalized = " ".join(header.split())
    if not normalized:
        raise ValueError("chunk header must not be empty")
    if not normalized.endswith(":"):
        normalized = f"{normalized}:"
    normalized = f"{normalized} "
    if len(normalized) > CHUNK_HEADER_MAX_CHARS:
        normalized = normalized[:CHUNK_HEADER_MAX_CHARS]
    return normalized


def generate_chunk_header(
    openai_client: OpenAI,
    markdown: str,
    model: str,
) -> str:
    """Generate a short chunk header from the leading markdown bytes."""

    source_excerpt = first_n_utf8_bytes(markdown, CHUNK_HEADER_SOURCE_BYTES)
    if not source_excerpt.strip():
        raise ValueError("cannot generate chunk header from empty markdown")

    response = openai_client.responses.parse(
        model=model,
        text_format=ChunkHeaderOutput,
        temperature=0,
        input=[
            {
                "role": "system",
                "content": (
                    "Create a concise chunk header prefix for a document. "
                    "Return JSON only and keep chunk_header within 64 characters. "
                    "Use a format like 'Document Title: '."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Document markdown excerpt (first 4096 bytes):\n"
                    f"{source_excerpt}"
                ),
            },
        ],
    )
    parsed = response.output_parsed
    if parsed is None:
        raise ValueError("response parsing failed for chunk header")
    return normalize_chunk_header(parsed.chunk_header)


def embedding_dimension_for_model(model: str) -> int:
    """Return embedding vector size for a configured model name."""

    if model == "text-embedding-3-small":
        return 1536
    if model == "text-embedding-3-large":
        return 4096
    raise ValueError(
        f"Unknown embedding dimension for model '{model}'. "
        "Update embedding_dimension_for_model() to add this model."
    )


def build_point_id(path: str, chunk_index: int, category: str) -> int:
    """Create a stable integer point id from path, category, and chunk index."""

    digest = hashlib.md5(f"{path}:{category}:{chunk_index}".encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


def ensure_collection(
    qdrant: QdrantClient,
    collection: str,
    embedding_size: int,
) -> None:
    """Create the collection when it does not already exist."""

    existing_collections = qdrant.get_collections().collections
    if any(info.name == collection for info in existing_collections):
        return

    qdrant.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
    )


def ensure_payload_indexes(qdrant: QdrantClient, collection: str) -> None:
    """Create keyword payload indexes for fields used in filter and facet queries."""

    payload_schema = qdrant.get_collection(collection).payload_schema
    for field_name in INDEXED_PAYLOAD_FIELDS:
        if field_name in payload_schema:
            continue
        qdrant.create_payload_index(
            collection_name=collection,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )


def build_qdrant_filter(category: str | None, path: str | None) -> Filter | None:
    """Build an AND filter for optional category and path exact matches."""

    conditions: list[Condition] = []

    if category:
        conditions.append(
            FieldCondition(key="category", match=MatchValue(value=category))
        )
    if path:
        conditions.append(FieldCondition(key="path", match=MatchValue(value=path)))

    if not conditions:
        return None

    return Filter(must=conditions)


def build_points_for_file(
    markitdown: MarkItDown,
    openai_client: OpenAI,
    path: Path,
    config: IngestConfig,
) -> list[PointStruct]:
    """Convert one file to chunk vectors and build Qdrant points."""

    markdown = document_to_markdown(markitdown=markitdown, path=path)
    chunk_header = ""
    if config.chunk_header_mode == "enabled":
        chunk_header = generate_chunk_header(
            openai_client=openai_client,
            markdown=markdown,
            model=config.chunk_header_model,
        )

    records = chunk_text(
        text=markdown,
        chunk_size=config.chunk_size,
        overlap_ratio=config.overlap_ratio,
    )
    if not records:
        raise ValueError("no chunks generated")

    chunk_texts = [f"{chunk_header}{record.text}" for record in records]
    response = openai_client.embeddings.create(
        model=config.embedding_model,
        input=chunk_texts,
    )

    return [
        PointStruct(
            id=build_point_id(
                path=str(path),
                chunk_index=record.chunk_index,
                category=config.category,
            ),
            vector=embedded.embedding,
            payload={
                "path": str(path),
                "category": config.category,
                "chunk_index": record.chunk_index,
                "chunk_size": config.chunk_size,
                "overlap_ratio": config.overlap_ratio,
                "chunk_header": chunk_header,
                "text": chunk_text,
            },
        )
        for record, chunk_text, embedded in zip(records, chunk_texts, response.data)
    ]


def register_ingest_documents_tool(
    app: FastMCP,
    markitdown: MarkItDown,
    qdrant: QdrantClient,
    openai_client: OpenAI,
    collection: str,
) -> None:
    """Register the ingest tool on the MCP app."""
    chunk_header_model = os.getenv(
        "CHUNK_HEADER_MODEL",
        DEFAULT_CHUNK_HEADER_MODEL,
    ).strip()
    if not chunk_header_model:
        chunk_header_model = DEFAULT_CHUNK_HEADER_MODEL

    @app.tool()
    def ingest_documents(
        paths: list[str],
        category: str,
        options: IngestOptions | None = None,
    ) -> dict[str, Any]:
        """
        Ingest documents into Qdrant: convert to markdown, chunk, embed, and upsert.

        Guidelines:
        - Before ingesting, check existing categories via `list_category`.
        - Reuse an existing category when possible; otherwise ask before creating one.
        - Use `chunk_size`, `overlap_ratio`, and `chunk_header_mode` defaults unless
          the user explicitly requests overrides.
        """
        if not paths:
            raise ValueError("paths must contain at least one file path")

        effective_options = options or IngestOptions()
        embedding_size = embedding_dimension_for_model(
            effective_options.embedding_model
        )
        ensure_collection(
            qdrant=qdrant,
            collection=collection,
            embedding_size=embedding_size,
        )
        ensure_payload_indexes(qdrant=qdrant, collection=collection)
        ingest_config = IngestConfig(
            category=category,
            chunk_size=effective_options.chunk_size,
            overlap_ratio=effective_options.overlap_ratio,
            embedding_model=effective_options.embedding_model,
            chunk_header_mode=effective_options.chunk_header_mode,
            chunk_header_model=chunk_header_model,
        )

        upsert_points: list[PointStruct] = []
        ingested_files = 0
        failed_files: list[dict[str, str]] = []

        for raw_path in paths:
            path = Path(raw_path).expanduser().resolve()

            if not path.exists() or not path.is_file():
                failed_files.append({"path": raw_path, "error": "file not found"})
                continue

            try:
                file_points = build_points_for_file(
                    markitdown=markitdown,
                    openai_client=openai_client,
                    path=path,
                    config=ingest_config,
                )
                upsert_points.extend(file_points)
                ingested_files += 1
            except (OpenAIError, OSError, TypeError, ValueError) as exc:
                failed_files.append({"path": str(path), "error": str(exc)})

        if upsert_points:
            qdrant.upsert(collection_name=collection, points=upsert_points)

        return {
            "collection": collection,
            "embedding_model": effective_options.embedding_model,
            "ingested_files": ingested_files,
            "ingested_points": len(upsert_points),
            "failed_files": failed_files,
        }


def register_search_documents_tool(
    app: FastMCP,
    qdrant: QdrantClient,
    embeddings_client: OpenAI,
    collection: str,
) -> None:
    """Register the search tool on the MCP app."""

    @app.tool()
    def search_documents(
        query: str,
        top_k: int = 5,
        category: str | None = None,
        path: str | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> dict[str, Any]:
        """
        Embed query text and return top-k chunk matches from Qdrant.
        """
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        vector = (
            embeddings_client.embeddings.create(
                model=embedding_model,
                input=[query],
            )
            .data[0]
            .embedding
        )

        query_filter = build_qdrant_filter(category=category, path=path)
        hits = qdrant.query_points(
            collection_name=collection,
            query=vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        ).points

        results = [
            {
                "score": hit.score,
                "path": (hit.payload or {}).get("path"),
                "category": (hit.payload or {}).get("category"),
                "chunk_index": (hit.payload or {}).get("chunk_index"),
                "text": (hit.payload or {}).get("text"),
            }
            for hit in hits
        ]

        return {
            "collection": collection,
            "embedding_model": embedding_model,
            "query": query,
            "count": len(results),
            "results": results,
        }


def register_delete_documents_by_path_tool(
    app: FastMCP,
    qdrant: QdrantClient,
    collection: str,
) -> None:
    """Register the delete-by-path tool on the MCP app."""

    @app.tool()
    def delete_documents_by_path(
        path: str,
        category: str | None = None,
    ) -> dict[str, Any]:
        """
        Delete points from the configured collection by exact path match.
        """
        normalized_path = path.strip()
        if not normalized_path:
            raise ValueError("path must not be empty")

        query_filter = build_qdrant_filter(category=category, path=normalized_path)
        if query_filter is None:
            raise ValueError("path must not be empty")

        result = qdrant.delete(
            collection_name=collection,
            points_selector=FilterSelector(filter=query_filter),
            wait=True,
        )

        return {
            "collection": collection,
            "path": normalized_path,
            "category": category,
            "status": str(result.status),
            "operation_id": result.operation_id,
        }


def register_list_category_tool(
    app: FastMCP,
    qdrant: QdrantClient,
    collection: str,
) -> None:
    """Register the category listing tool on the MCP app."""

    @app.tool()
    def list_category(limit: int = 100) -> dict[str, Any]:
        """
        List existing categories in the configured collection.
        """
        if limit < 1:
            raise ValueError("limit must be >= 1")

        hits = qdrant.facet(
            collection_name=collection,
            key="category",
            limit=limit,
            exact=False,
        ).hits

        categories = [str(hit.value) for hit in hits]
        return {
            "collection": collection,
            "count": len(categories),
            "categories": categories,
        }


def register_list_path_tool(
    app: FastMCP,
    qdrant: QdrantClient,
    collection: str,
) -> None:
    """Register the path listing tool on the MCP app."""

    @app.tool()
    def list_path(category: str, limit: int = 1000) -> dict[str, Any]:
        """
        List existing file paths for a category in the configured collection.
        """
        normalized_category = category.strip()
        if not normalized_category:
            raise ValueError("category must not be empty")
        if limit < 1:
            raise ValueError("limit must be >= 1")

        category_filter = build_qdrant_filter(category=normalized_category, path=None)
        hits = qdrant.facet(
            collection_name=collection,
            key="path",
            facet_filter=category_filter,
            limit=limit,
            exact=False,
        ).hits

        paths = [str(hit.value) for hit in hits]
        return {
            "collection": collection,
            "category": normalized_category,
            "count": len(paths),
            "paths": paths,
        }


def create_app() -> FastMCP:
    """Create and configure the MCP application and all registered tools."""

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection = os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    app = FastMCP("qdrant-mcp")
    markitdown = MarkItDown()
    qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    embeddings_client = OpenAI()

    register_ingest_documents_tool(
        app=app,
        markitdown=markitdown,
        qdrant=qdrant,
        openai_client=embeddings_client,
        collection=collection,
    )
    register_search_documents_tool(
        app=app,
        qdrant=qdrant,
        embeddings_client=embeddings_client,
        collection=collection,
    )
    register_delete_documents_by_path_tool(
        app=app,
        qdrant=qdrant,
        collection=collection,
    )
    register_list_category_tool(
        app=app,
        qdrant=qdrant,
        collection=collection,
    )
    register_list_path_tool(
        app=app,
        qdrant=qdrant,
        collection=collection,
    )

    return app


def main() -> None:
    """Run the MCP server process."""

    app = create_app()
    app.run()


if __name__ == "__main__":
    main()
