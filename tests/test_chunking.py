import re

from paper_rag.chunking import chunk_pages
from paper_rag.pdf import PageText


class WordOffsetTokenizer:
    """Small tokenizer fixture that exposes exact character offsets."""

    def __call__(self, text, **_kwargs):
        return {"offset_mapping": [match.span() for match in re.finditer(r"\S+", text)]}


def test_chunks_keep_source_text_page_and_overlap():
    pages = [
        PageText(1, "one two three four five six"),
        PageText(2, "alpha beta"),
    ]
    chunks = chunk_pages(pages, WordOffsetTokenizer(), "sample", chunk_size=4, overlap=2)

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "three four five six",
        "alpha beta",
    ]
    assert [chunk.page_number for chunk in chunks] == [1, 1, 2]
    assert [chunk.chunk_id for chunk in chunks] == [0, 1, 2]
    for chunk in chunks:
        page_text = pages[chunk.page_number - 1].text
        assert page_text[chunk.start_char : chunk.end_char] == chunk.text
