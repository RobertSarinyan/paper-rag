"""Encode text with a local sentence embedding model."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.tokenizer = self.model.tokenizer
        self.max_sequence_length = self.model.max_seq_length
        if not self.tokenizer.is_fast:
            raise ValueError("The embedding model needs a fast tokenizer for source offsets")

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """Return one normalized float32 vector per chunk."""
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Provide at least one nonempty text per chunk")
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def encode_query(self, question: str) -> np.ndarray:
        """Return one normalized float32 query vector with shape (1, dimension)."""
        if not question.strip():
            raise ValueError("Question must not be empty")
        vector = self.model.encode_query(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vector, dtype=np.float32)
