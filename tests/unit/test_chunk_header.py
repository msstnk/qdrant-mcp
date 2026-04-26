"""Unit tests for chunk header generation helpers."""
from types import SimpleNamespace

from qdrant_mcp.server import (
    CHUNK_HEADER_MAX_CHARS,
    CHUNK_HEADER_SOURCE_BYTES,
    first_n_utf8_bytes,
    generate_chunk_header,
    normalize_chunk_header,
)


def build_fake_openai(
    chunk_header: str,
) -> tuple[SimpleNamespace, dict[str, str | None]]:
    """Build a fake OpenAI client and mutable state for parse input assertions."""

    state: dict[str, str | None] = {"excerpt_content": None}

    def parse(**kwargs):
        """Record parse kwargs and return the expected structured response."""

        state["excerpt_content"] = kwargs["input"][1]["content"]
        return SimpleNamespace(output_parsed=SimpleNamespace(chunk_header=chunk_header))

    client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    return client, state


def test_first_n_utf8_bytes_respects_byte_limit() -> None:
    """Truncates safely by UTF-8 bytes without breaking a multibyte character."""

    assert first_n_utf8_bytes("AあB", 4) == "Aあ"


def test_normalize_chunk_header_appends_suffix_and_clips() -> None:
    """Adds the ': ' suffix and enforces the max header length."""

    normalized = normalize_chunk_header("FortiGate 7.6.6 Command Line Reference")
    assert normalized.endswith(": ")

    clipped = normalize_chunk_header("x" * (CHUNK_HEADER_MAX_CHARS + 10))
    assert len(clipped) == CHUNK_HEADER_MAX_CHARS


def test_generate_chunk_header_uses_first_4096_bytes() -> None:
    """Sends only the first 4096 bytes of markdown to the Responses API prompt."""

    fake_client, state = build_fake_openai(
        chunk_header="FortiGate 7.6.6 Command Line Reference: "
    )
    markdown = "A" * (CHUNK_HEADER_SOURCE_BYTES + 1000)

    header = generate_chunk_header(
        openai_client=fake_client,  # type: ignore[arg-type]
        markdown=markdown,
        model="gpt-4.1-mini",
    )

    assert header == "FortiGate 7.6.6 Command Line Reference: "
    excerpt = state["excerpt_content"]
    assert excerpt is not None
    prefix = "Document markdown excerpt (first 4096 bytes):\n"
    assert excerpt.startswith(prefix)
    source_excerpt = excerpt.removeprefix(prefix)
    assert len(source_excerpt.encode("utf-8")) == CHUNK_HEADER_SOURCE_BYTES
