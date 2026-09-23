"""Exercise HTTP contracts and real PDF/FAISS persistence without network calls."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from paper_rag import api, service
from paper_rag.chunking import Chunk
from paper_rag.config import ConfigurationError, Settings
from paper_rag.llm import GeneratedAnswer, GeminiRequestError, SourceContext
from paper_rag.rag import RagResult
from paper_rag.vector_store import build_index, save_index


class FakeEmbedder:
    model_name = "test-model"
    tokenizer = None

    def encode_documents(self, texts):
        return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)


@pytest.fixture
def app(tmp_path, monkeypatch):
    application = api.create_app(tmp_path / "indexes", tmp_path / ".env")
    monkeypatch.setattr(application.state.documents, "embedder", lambda *args: FakeEmbedder())

    def small_chunks(pages, tokenizer, document_id):
        # Actual PDF extraction and FAISS writing remain active in this test.
        text = pages[0].text
        return [Chunk(document_id, 0, 1, 0, 2, 0, len(text), text)]

    monkeypatch.setattr(service, "chunk_pages", small_chunks)
    monkeypatch.setattr(api, "load_settings", lambda *args: Settings("unused", "fake-gemini"))
    return application


def upload_sample(client):
    sample = Path(__file__).resolve().parents[1] / "data/samples/1706.03762v7.pdf"
    with sample.open("rb") as file:
        return client.post("/upload", files={"file": ("paper.pdf", file, "application/pdf")})


def test_upload_persists_and_same_filenames_do_not_collide(app):
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        first = upload_sample(client)
        second = upload_sample(client)
        assert first.status_code == second.status_code == 201
        assert first.json()["document_id"] != second.json()["document_id"]
        assert first.json()["chunk_count"] == 1
        assert len(client.get("/documents").json()) == 2
    # A new application instance reads the indexes from disk after a restart.
    restarted = api.create_app(app.state.documents.index_root)
    with TestClient(restarted) as client:
        assert len(client.get("/documents").json()) == 2


def test_existing_cli_index_with_dotted_id_is_available(app, monkeypatch):
    document_id = "1706.03762v7"
    prepared = build_index(
        [Chunk(document_id, 0, 1, 0, 1, 0, 8, "Evidence")],
        np.array([[1.0, 0.0]], dtype=np.float32),
        source_filename="paper.pdf", source_sha256="a" * 64,
        embedding_model="test-model", chunk_size=180, overlap=40,
    )
    save_index(prepared, app.state.documents.index_root / document_id)
    monkeypatch.setattr(api, "answer_question", lambda *args, **kwargs: RagResult(GeneratedAnswer("Not found", False, ()), (), ()))
    with TestClient(app) as client:
        assert client.get("/documents").json()[0]["document_id"] == document_id
        assert client.post("/ask", json={"document_id": document_id, "question": "Why?"}).status_code == 200


@pytest.mark.parametrize("question,top_k", [("   ", 4), ("Why?", 0), ("Why?", 21), ("Why?", 1.5)])
def test_invalid_ask_input_is_rejected(app, question, top_k):
    with TestClient(app) as client:
        assert client.post("/ask", json={"document_id": "paper", "question": question, "top_k": top_k}).status_code == 422


def test_missing_and_unsafe_document_ids(app):
    with TestClient(app) as client:
        for document_id, expected in [("missing", 404), ("../secret", 422)]:
            assert client.post("/ask", json={"document_id": document_id, "question": "Why?"}).status_code == expected


def test_invalid_uploads_leave_no_indexes(app, monkeypatch):
    with TestClient(app) as client:
        for name, content, expected in [
            ("file.txt", b"text", 415), ("file.pdf", b"", 400),
            ("file.pdf", b"not pdf", 400), ("file.pdf", b"%PDF-1.7\nbroken", 400),
        ]:
            response = client.post("/upload", files={"file": (name, content)})
            assert response.status_code == expected
        monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 8)
        assert client.post("/upload", files={"file": ("large.pdf", b"%PDF-" + b"x" * 10)}).status_code == 413
        assert client.get("/documents").json() == []
        assert list(app.state.documents.index_root.iterdir()) == []


@pytest.mark.parametrize("answerable", [True, False])
def test_ask_returns_answer_and_traceable_sources(app, monkeypatch, answerable):
    prepared = SimpleNamespace(document_id="paper", embedding_model="test-model")
    monkeypatch.setattr(app.state.documents, "get", lambda _: prepared)
    source = SourceContext("S1", "paper", "paper.pdf", 4, 16, 0.685, "Evidence")

    def fake_answer(index, question, settings, **kwargs):
        assert index is prepared
        assert kwargs["top_k"] == 2
        return RagResult(
            GeneratedAnswer("Supported [S1]" if answerable else "Not found", answerable, ("S1",) if answerable else ()),
            (source,), (source,) if answerable else (),
        )

    monkeypatch.setattr(api, "answer_question", fake_answer)
    with TestClient(app) as client:
        response = client.post("/ask", json={"document_id": "paper", "question": " Why? ", "top_k": 2})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answerable"] is answerable
    assert payload["question"] == "Why?"
    assert payload["retrieved_sources"][0]["page_number"] == 4
    assert len(payload["cited_sources"]) == int(answerable)


def test_upstream_error_is_sanitized(app, monkeypatch):
    monkeypatch.setattr(app.state.documents, "get", lambda _: SimpleNamespace(embedding_model="test-model"))

    def fail(*args, **kwargs):
        raise GeminiRequestError("private upstream details")

    monkeypatch.setattr(api, "answer_question", fail)
    with TestClient(app) as client:
        response = client.post("/ask", json={"document_id": "paper", "question": "Why?"})
    assert response.status_code == 502
    assert "private" not in response.text


def test_missing_configuration_returns_503(app, monkeypatch):
    monkeypatch.setattr(app.state.documents, "get", lambda _: object())

    def missing(*args):
        raise ConfigurationError("Missing key")

    monkeypatch.setattr(api, "load_settings", missing)
    with TestClient(app) as client:
        assert client.post("/ask", json={"document_id": "paper", "question": "Why?"}).status_code == 503
