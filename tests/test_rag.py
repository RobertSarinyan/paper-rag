import numpy as np

from paper_rag.chunking import Chunk
from paper_rag.config import Settings
from paper_rag.llm import GeneratedAnswer
from paper_rag.rag import answer_question
from paper_rag.vector_store import build_index


class FakeEmbedder:
    model_name = "test-model"

    def encode_query(self, question: str) -> np.ndarray:
        return np.array([[1.0, 0.0]], dtype=np.float32)


class FakeGenerator:
    model_name = "fake-model"

    def __init__(self) -> None:
        self.sources = ()

    def generate(self, question, sources):
        self.sources = tuple(sources)
        return GeneratedAnswer(
            answer="The retrieved passage supports the answer [S1].",
            answerable=True,
            cited_source_ids=("S1",),
        )


def test_rag_maps_model_citations_back_to_retrieved_chunks():
    chunks = [
        Chunk("paper", 0, 2, 0, 2, 0, 10, "relevant text"),
        Chunk("paper", 1, 7, 0, 2, 0, 12, "unrelated text"),
    ]
    prepared = build_index(
        chunks,
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        source_filename="paper.pdf",
        source_sha256="a" * 64,
        embedding_model="test-model",
        chunk_size=180,
        overlap=40,
    )
    generator = FakeGenerator()

    result = answer_question(
        prepared,
        "What is relevant?",
        Settings(gemini_api_key="unused", gemini_model="fake-model"),
        top_k=2,
        embedder=FakeEmbedder(),
        generator=generator,
    )

    assert generator.sources[0].source_id == "S1"
    assert generator.sources[0].chunk_id == 0
    assert result.generated.answerable is True
    assert result.cited_sources == (result.retrieved_sources[0],)
