# qdrant-mcp

MCP server for document ingestion and semantic search on Qdrant.

## Overview

`qdrant-mcp` provides tools to:

- ingest local documents into a Qdrant collection
- generate embeddings with OpenAI
- run vector search with optional metadata filters

## Features

- `ingest_documents`
  - converts files such as `docx`, `pptx`, and `pdf` to Markdown via MarkItDown
  - splits content into chunks using `chunk_size` and `overlap_ratio`
  - embeds chunks with OpenAI Embeddings (`text-embedding-3-small` by default)
  - upserts chunk text and metadata into Qdrant
- `search_documents`
  - embeds query text with the same embeddings API
  - retrieves top `k` matches from Qdrant
  - supports filtering by `category` and `path`

## Requirements

- Python 3.11+
- `uv`
- Qdrant (for example, `http://localhost:6333`)
- `OPENAI_API_KEY`

## Setup

```bash
uv sync
```

## Run inside the Codex CLI

```toml
[mcp_servers.qdrant-mcp]
command = "uv"
args = ["run", "qdrant-mcp"]
cwd = "/sandbox/qdrant-mcp"

env = {
  OPENAI_API_KEY = "sk-...",
  QDRANT_URL = "http://127.0.0.1:6333",
  QDRANT_API_KEY = "QDRANT_API_KEY",
  QDRANT_COLLECTION = "codex_collection",
  CHUNK_HEADER_MODEL = "gpt-5.4-mini"
}
```

## Testing

Set `OPENAI_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY` in `.env`, then run:

```bash
uv run python -m unittest tests/integration/test_qdrant_integration.py
```

## MCP Tools

### `ingest_documents`

Parameters:

- `paths: list[str]`
- `category: str`
- `chunk_size: int = 1200`
- `overlap_ratio: float = 0.15`
- `embedding_model: str = "text-embedding-3-small"`
- `chunk_header_mode: Literal["enabled", "disabled"] = "enabled"`

Returns:

- `collection`
- `embedding_model`
- `ingested_files`
- `ingested_points`
- `failed_files`

### `search_documents`

Parameters:

- `query: str`
- `top_k: int = 5`
- `category: str | None = None`
- `path: str | None = None`
- `embedding_model: str = "text-embedding-3-small"`

Returns:

- `collection`
- `embedding_model`
- `query`
- `count`
- `results` (`score`, `path`, `category`, `chunk_index`, `text`)

### `delete_documents_by_path`

Parameters:

- `path: str`
- `category: str | None = None`

Returns:

- `collection`
- `path`
- `category`
- `status`
- `operation_id`

### `list_category`

Parameters:

- `limit: int = 100`

Returns:

- `collection`
- `count`
- `categories`

### `list_path`

Parameters:

- `category: str`
- `limit: int = 1000`

Returns:

- `collection`
- `category`
- `count`
- `paths`

## Notes

- If the target collection does not exist, it is created automatically on first ingestion.
- If payload indexes for `category` and `path` do not exist, they are created during ingestion.
- By default, ingestion prepends a generated `Chunk-Header` (max 64 chars), derived from the first 4096 bytes, to every chunk.
- The Chunk-Header model is read from `CHUNK_HEADER_MODEL` when `chunk_header_mode` is `enabled` (default: `gpt-5.4-mini`).
- The collection name is configured only through `QDRANT_COLLECTION` (not via MCP tool parameters).
- With `text-embedding-3-small`, the vector size is `1536`.

## License

See [LICENSE](LICENSE).
