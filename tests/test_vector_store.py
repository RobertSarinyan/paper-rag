import numpy as np

from paper_rag.chunking import Chunk
from paper_rag.vector_store import build_index, load_index, save_index


def test_index_round_trip_preserves_vectors_and_source_metadata(tmp_path):
    chunks = [
        Chunk("paper", 0, 2, 0, 2, 0, 10, "first text"),
        Chunk("paper", 1, 3, 0, 2, 0, 11, "second text"),
    ]
    vectors = np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    prepared = build_index(
        chunks,
        vectors,
        source_filename="paper.pdf",
        source_sha256="a" * 64,
        embedding_model="test-model",
        chunk_size=180,
        overlap=40,
    )
    save_index(prepared, tmp_path)
    loaded = load_index(tmp_path)

    assert loaded.index.ntotal == len(chunks)
    assert loaded.index.d == 2
    assert loaded.chunks == chunks
    assert loaded.source_filename == "paper.pdf"
    assert loaded.source_sha256 == "a" * 64
    np.testing.assert_allclose(loaded.index.reconstruct(0), [1.0, 0.0])
    np.testing.assert_allclose(loaded.index.reconstruct(1), [0.0, 1.0])
