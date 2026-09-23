"""Local document storage and reusable operations for the API."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from uuid import uuid4

from .chunking import chunk_pages
from .embeddings import DEFAULT_MODEL, EmbeddingModel
from .pdf import extract_pages
from .schemas import DocumentResponse
from .vector_store import PreparedIndex, build_index, load_index, save_index


class DocumentService:
    def __init__(self, index_root: Path) -> None:
        self.index_root = index_root.resolve()
        self.models: dict[str, EmbeddingModel] = {}
        # A single local model is reused. Serialize embedding operations in this MVP.
        self.model_lock = RLock()

    def embedder(self, name: str = DEFAULT_MODEL) -> EmbeddingModel:
        with self.model_lock:
            if name not in self.models:
                self.models[name] = EmbeddingModel(name)
            return self.models[name]

    @staticmethod
    def describe(prepared: PreparedIndex) -> DocumentResponse:
        return DocumentResponse(
            document_id=prepared.document_id,
            source_filename=prepared.source_filename,
            chunk_count=len(prepared.chunks),
            embedding_model=prepared.embedding_model,
        )

    def get(self, document_id: str) -> PreparedIndex:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", document_id):
            raise FileNotFoundError("Unknown document")
        directory = (self.index_root / document_id).resolve()
        if directory.parent != self.index_root or not directory.is_dir():
            raise FileNotFoundError("Unknown document")
        prepared = load_index(directory)
        if prepared.document_id != document_id:
            raise ValueError("Stored document ID does not match its directory")
        return prepared

    def list_documents(self) -> list[DocumentResponse]:
        if not self.index_root.exists():
            return []
        return [
            self.describe(self.get(path.name))
            for path in sorted(self.index_root.iterdir())
            if path.is_dir() and not path.name.startswith(".")
        ]

    def ingest(self, content: bytes, filename: str) -> DocumentResponse:
        """Publish an index only after both its vectors and metadata are written."""
        document_id = uuid4().hex
        self.index_root.mkdir(parents=True, exist_ok=True)
        # Temporary files stay on the same filesystem as the final index for rename.
        with TemporaryDirectory(prefix=".ingest-", dir=self.index_root) as temporary:
            staging = Path(temporary)
            pdf = staging / "source.pdf"
            pdf.write_bytes(content)
            pages = extract_pages(pdf)
            with self.model_lock:
                model = self.embedder()
                chunks = chunk_pages(pages, model.tokenizer, document_id)
                vectors = model.encode_documents([chunk.text for chunk in chunks])
            prepared = build_index(
                chunks, vectors, source_filename=filename,
                source_sha256=hashlib.sha256(content).hexdigest(),
                embedding_model=model.model_name, chunk_size=180, overlap=40,
            )
            destination = staging / "index"
            save_index(prepared, destination)
            destination.rename(self.index_root / document_id)
        return self.describe(prepared)
