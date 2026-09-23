"""Run retrieval followed by a source-grounded Gemini answer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from .config import ConfigurationError, Settings, load_settings
from .console import configure_console_output
from .embeddings import EmbeddingModel
from .llm import (
    GeneratedAnswer,
    GeminiAnswerGenerator,
    GeminiRequestError,
    InvalidModelResponseError,
    SourceContext,
)
from .retrieval import retrieve
from .vector_store import PreparedIndex, load_index


class AnswerGenerator(Protocol):
    model_name: str

    def generate(
        self, question: str, sources: Sequence[SourceContext]
    ) -> GeneratedAnswer: ...


@dataclass(frozen=True)
class RagResult:
    generated: GeneratedAnswer
    retrieved_sources: tuple[SourceContext, ...]
    cited_sources: tuple[SourceContext, ...]


def answer_question(
    prepared: PreparedIndex,
    question: str,
    settings: Settings,
    *,
    top_k: int = 4,
    embedder: EmbeddingModel | None = None,
    generator: AnswerGenerator | None = None,
) -> RagResult:
    """Retrieve evidence, generate an answer, and resolve its citations."""
    matches = retrieve(prepared, question, top_k=top_k, embedder=embedder)
    sources = tuple(
        SourceContext(
            source_id=f"S{match.rank}",
            document_id=match.chunk.document_id,
            source_filename=prepared.source_filename,
            page_number=match.chunk.page_number,
            chunk_id=match.chunk.chunk_id,
            score=match.score,
            text=match.chunk.text,
        )
        for match in matches
    )

    answer_generator = generator or GeminiAnswerGenerator(
        api_key=settings.gemini_api_key,
        model_name=settings.gemini_model,
    )
    generated = answer_generator.generate(question, sources)
    sources_by_id = {source.source_id: source for source in sources}
    cited = tuple(sources_by_id[source_id] for source_id in generated.cited_source_ids)
    return RagResult(
        generated=generated,
        retrieved_sources=sources,
        cited_sources=cited,
    )


def _print_source(source: SourceContext) -> None:
    readable_text = " ".join(source.text.split())
    print(
        f"[{source.source_id}] score={source.score:.4f} "
        f"| page={source.page_number} | chunk={source.chunk_id}"
    )
    print(readable_text)


def main() -> None:
    configure_console_output()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path, help="Directory containing the saved index")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")
    parser.add_argument(
        "--show-retrieved",
        action="store_true",
        help="Print every retrieved chunk after the cited sources",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if not args.question.strip():
        parser.error("question must not be empty")

    try:
        settings = load_settings()
        prepared = load_index(args.index)
        result = answer_question(
            prepared,
            args.question,
            settings,
            top_k=args.top_k,
        )
    except (
        ConfigurationError,
        FileNotFoundError,
        GeminiRequestError,
        InvalidModelResponseError,
    ) as exc:
        parser.exit(1, f"Error: {exc}\n")

    print(f"Document: {prepared.source_filename}")
    print(f"Embedding model: {prepared.embedding_model}")
    print(f"LLM model: {settings.gemini_model}")
    print(f"Question: {args.question}")
    print(f"Answer supported: {'yes' if result.generated.answerable else 'no'}")
    print("\nAnswer:")
    print(result.generated.answer)

    print("\nCited sources:")
    if result.cited_sources:
        for position, source in enumerate(result.cited_sources):
            if position:
                print()
            _print_source(source)
    else:
        print("None")

    if args.show_retrieved:
        print("\nAll retrieved sources:")
        for position, source in enumerate(result.retrieved_sources):
            if position:
                print()
            _print_source(source)


if __name__ == "__main__":
    main()
