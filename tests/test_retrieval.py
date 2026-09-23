import numpy as np

from paper_rag.chunking import Chunk
from paper_rag.vector_store import build_index, search_index


def test_search_ranks_chunks_and_returns_source_metadata():
    chunks = [
        Chunk("paper", 0, 2, 0, 2, 0, 12, "unrelated text"),
        Chunk("paper", 1, 7, 0, 2, 0, 13, "relevant text"),
        Chunk("paper", 2, 9, 0, 2, 0, 12, "weaker match"),
    ]
    vectors = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.8, 0.6],
        ],
        dtype=np.float32,
    )
    prepared = build_index(
        chunks,
        vectors,
        source_filename="paper.pdf",
        source_sha256="a" * 64,
        embedding_model="test-model",
        chunk_size=180,
        overlap=40,
    )

    results = search_index(prepared, np.array([5.0, 0.0]), top_k=2)

    assert [result.chunk.chunk_id for result in results] == [1, 2]
    assert [result.chunk.page_number for result in results] == [7, 9]
    assert [result.rank for result in results] == [1, 2]
    assert results[0].score == 1.0
    assert np.isclose(results[1].score, 0.8)
