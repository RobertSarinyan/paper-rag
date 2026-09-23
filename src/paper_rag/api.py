"""FastAPI application for document ingestion and grounded questions."""

from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, UploadFile
from pypdf.errors import PyPdfError

from .config import ConfigurationError, load_settings
from .llm import GeminiRequestError, InvalidModelResponseError
from .rag import answer_question
from .schemas import AskRequest, AskResponse, DocumentResponse, SourceResponse
from .service import DocumentService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def create_app(index_root: Path | None = None, env_file: Path | None = None) -> FastAPI:
    application = FastAPI(title="Paper RAG", version="0.1.0")
    application.state.documents = DocumentService(index_root or PROJECT_ROOT / "data/indexes")
    settings_file = env_file or PROJECT_ROOT / ".env"

    @application.get("/health")
    def health() -> dict[str, str]:
        """Liveness only: this does not call Gemini or load the embedding model."""
        return {"status": "ok"}

    @application.get("/documents", response_model=list[DocumentResponse])
    def documents():
        return application.state.documents.list_documents()

    @application.post("/upload", response_model=DocumentResponse, status_code=201)
    def upload(file: UploadFile):
        """Upload one text PDF (up to 20 MiB); wait until its index is ready."""
        filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(415, "Upload a .pdf file")
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "PDF must be no larger than 20 MiB")
        if not content or not content.lstrip().startswith(b"%PDF-"):
            raise HTTPException(400, "The file does not contain a PDF header")
        try:
            return application.state.documents.ingest(content, filename)
        except (PyPdfError, ValueError) as exc:
            raise HTTPException(400, "Cannot process this PDF. Use an unencrypted PDF with extractable text.") from exc

    @application.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest):
        """Answer from one indexed document and return the supporting passages."""
        service = application.state.documents
        try:
            prepared = service.get(request.document_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, "Document not found; upload it or check GET /documents") from exc
        try:
            settings = load_settings(settings_file)
        except ConfigurationError as exc:
            raise HTTPException(503, "Configure GEMINI_API_KEY and GEMINI_MODEL in the server's .env") from exc
        try:
            # The synchronous route runs in FastAPI's thread pool. For this local
            # MVP, serialize model use (including the answer) to keep memory bounded.
            with service.model_lock:
                result = answer_question(
                    prepared, request.question, settings, top_k=request.top_k,
                    embedder=service.embedder(prepared.embedding_model),
                )
        except (GeminiRequestError, InvalidModelResponseError, httpx.HTTPError) as exc:
            # Do not return upstream error bodies, which can contain sensitive data.
            raise HTTPException(502, "Gemini could not produce a valid answer. Check model availability, API access, and quota, then retry.") from exc
        return AskResponse(
            document_id=prepared.document_id, question=request.question,
            answer=result.generated.answer, answerable=result.generated.answerable,
            model=settings.gemini_model,
            cited_sources=[SourceResponse.model_validate(source) for source in result.cited_sources],
            retrieved_sources=[SourceResponse.model_validate(source) for source in result.retrieved_sources],
        )

    return application


app = create_app()
