import json

import pytest

from paper_rag.llm import (
    InvalidModelResponseError,
    SourceContext,
    build_prompt,
    parse_generated_answer,
)


def make_source() -> SourceContext:
    return SourceContext(
        source_id="S1",
        document_id="paper",
        source_filename="paper.pdf",
        page_number=4,
        chunk_id=16,
        score=0.685,
        text="The values are scaled by the square root of the key dimension.",
    )


def test_prompt_preserves_question_and_source_metadata():
    prompt = build_prompt("Why scale attention?", [make_source()])
    payload = json.loads(prompt.split("\n", maxsplit=1)[1])

    assert payload["question"] == "Why scale attention?"
    assert payload["sources"][0]["source_id"] == "S1"
    assert payload["sources"][0]["page"] == 4
    assert payload["sources"][0]["chunk"] == 16


def test_response_validation_rejects_an_unknown_citation():
    response = json.dumps(
        {
            "answer": "The passage explains it [S9].",
            "answerable": True,
            "cited_source_ids": ["S9"],
        }
    )

    with pytest.raises(InvalidModelResponseError, match="unknown source"):
        parse_generated_answer(response, ["S1"])
