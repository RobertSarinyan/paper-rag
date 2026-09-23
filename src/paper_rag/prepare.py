"""Run stages 1-4: PDF extraction, chunking, embeddings, and indexing."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from .chunking import chunk_pages
from .console import configure_console_output
from .embeddings import DEFAULT_MODEL, EmbeddingModel
from .pdf import extract_pages
from .vector_store import build_index, load_index, save_index


def main() -> None:
    configure_console_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="A text-based PDF to prepare")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunk-size", type=int, default=180)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument(
        "--stop-after", choices=("chunks", "embeddings", "index"), default="index"
    )
    parser.add_argument("--output", type=Path, help="Index directory")
    args = parser.parse_args()

    pages = extract_pages(args.pdf)
    print(f"PDF: {args.pdf.name}; pages: {len(pages)}")
    print(f"Extractable pages: {sum(bool(page.text.strip()) for page in pages)}")

    embedder = EmbeddingModel(args.model)
    special_tokens = embedder.tokenizer.num_special_tokens_to_add(pair=False)
    if args.chunk_size + special_tokens > embedder.max_sequence_length:
        parser.error(
            f"chunk size must be at most {embedder.max_sequence_length - special_tokens} "
            f"for {args.model}"
        )
    chunks = chunk_pages(
        pages,
        embedder.tokenizer,
        args.pdf.stem,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(f"Chunks: {len(chunks)}; size: {args.chunk_size}; overlap: {args.overlap}")
    first = chunks[0]
    print(f"First chunk: id={first.chunk_id}, page={first.page_number}")
    print(f"Preview: {first.text[:160].replace(chr(10), ' ')}")
    if args.stop_after == "chunks":
        return

    vectors = embedder.encode_documents([chunk.text for chunk in chunks])
    lengths = np.linalg.norm(vectors, axis=1)
    print(f"Embeddings: shape={vectors.shape}, dtype={vectors.dtype}")
    print(f"Vector norms: min={lengths.min():.4f}, max={lengths.max():.4f}")
    if args.stop_after == "embeddings":
        return

    prepared = build_index(
        chunks,
        vectors,
        source_filename=args.pdf.name,
        source_sha256=hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
        embedding_model=args.model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    destination = args.output or Path("data/indexes") / args.pdf.stem
    save_index(prepared, destination)
    loaded = load_index(destination)
    print(f"FAISS index: {loaded.index.ntotal} vectors, dimension {loaded.index.d}")
    print(f"Saved and reloaded: {destination}")


if __name__ == "__main__":
    main()
