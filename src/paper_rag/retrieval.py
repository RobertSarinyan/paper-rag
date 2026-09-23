"""Retrieve the most relevant PDF chunks for a question, without an LLM."""

from __future__ import annotations

import argparse
from pathlib import Path

from .console import configure_console_output
from .embeddings import EmbeddingModel
from .vector_store import PreparedIndex, SearchResult, load_index, search_index


def retrieve(
    prepared: PreparedIndex,
    question: str,
    *,
    top_k: int = 4,
    embedder: EmbeddingModel | None = None,
) -> list[SearchResult]:
    """Embed one question and map its nearest vectors back to source chunks."""
    if not question.strip():
        raise ValueError("Question must not be empty")
    model = embedder or EmbeddingModel(prepared.embedding_model)
    if model.model_name != prepared.embedding_model:
        raise ValueError(
            "The query embedding model must match the model used to build the index"
        )
    query_vector = model.encode_query(question)
    return search_index(prepared, query_vector, top_k=top_k)


def main() -> None:
    configure_console_output()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path, help="Directory containing the saved index")
    parser.add_argument("question", help="Question to search for")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to show")
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if not args.question.strip():
        parser.error("question must not be empty")

    prepared = load_index(args.index)
    results = retrieve(prepared, args.question, top_k=args.top_k)

    print(f"Document: {prepared.source_filename}")
    print(f"Embedding model: {prepared.embedding_model}")
    print(f"Question: {args.question}")
    print(f"Top matches: {len(results)}")

    for result in results:
        chunk = result.chunk
        readable_text = " ".join(chunk.text.split())
        print(
            f"\nResult {result.rank} | score={result.score:.4f} "
            f"| page={chunk.page_number} | chunk={chunk.chunk_id}"
        )
        print(readable_text)


if __name__ == "__main__":
    main()
