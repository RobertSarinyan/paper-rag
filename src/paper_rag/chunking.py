"""Split PDF pages into overlapping, source-traceable token windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .pdf import PageText


@dataclass(frozen=True)
class Chunk:
    document_id: str
    chunk_id: int
    page_number: int
    start_token: int  # Position within this page, before special model tokens.
    end_token: int  # Exclusive.
    start_char: int  # Offset into the extracted page text.
    end_char: int  # Exclusive.
    text: str


def chunk_pages(
    pages: list[PageText],
    tokenizer: Any,
    document_id: str,
    *,
    chunk_size: int = 180,
    overlap: int = 40,
) -> list[Chunk]:
    """Make chunks without crossing page boundaries or rewriting source text.

    Token offsets locate each window in the original page string. Slicing that
    string preserves exact source wording for later citations.
    """
    if not document_id.strip():
        raise ValueError("document_id must not be empty")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    chunks: list[Chunk] = []
    step = chunk_size - overlap

    for page in pages:
        encoded = tokenizer(
            page.text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,  # Full pages are expected to exceed the model limit.
        )
        offsets = encoded.get("offset_mapping")
        if offsets is None:
            raise ValueError("The tokenizer must provide character offsets")

        start = 0
        while start < len(offsets):
            end = min(start + chunk_size, len(offsets))
            start_char = offsets[start][0]
            end_char = offsets[end - 1][1]
            text = page.text[start_char:end_char]
            if text.strip():
                chunks.append(
                    Chunk(
                        document_id=document_id,
                        chunk_id=len(chunks),
                        page_number=page.page_number,
                        start_token=start,
                        end_token=end,
                        start_char=start_char,
                        end_char=end_char,
                        text=text,
                    )
                )
            if end == len(offsets):
                break
            start += step

    if not chunks:
        raise ValueError("No text chunks could be produced")
    return chunks
