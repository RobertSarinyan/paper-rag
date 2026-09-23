# Paper RAG

Paper RAG is a local research-document assistant that answers questions from
uploaded PDFs and shows the passages used as evidence. It extracts text with
page metadata, splits it into overlapping token windows, creates local sentence
embeddings, searches them with FAISS, and gives the retrieved passages to
Gemini for a grounded answer.

The project is intentionally small enough to study end to end while covering
software engineering, data processing, information retrieval, LLM integration,
API design, testing, and reproducible packaging.

## How it works

```text
PDF
 │
 ├─ pypdf extracts text and page numbers
 │
 ├─ Sentence Transformers splits and embeds the text
 │
 ├─ FAISS stores normalized vectors and retrieves similar chunks
 │
 ├─ Gemini receives only the question and retrieved chunks
 │
 └─ FastAPI returns the answer, citations, and retrieved evidence
```

The embedding model and FAISS run locally. Gemini is used only for final answer
generation. The API rejects model citations that do not refer to a retrieved
source ID.

## Features

- Text extraction from text-based PDFs with one-based page numbers
- Overlapping, page-aware chunks that preserve original source text
- Local `sentence-transformers/all-MiniLM-L6-v2` embeddings
- Exact cosine-similarity search using a normalized FAISS inner-product index
- Persistent vector indexes with document hashes and chunk metadata
- Source-grounded Gemini answers with validated citations
- Explicit refusal when retrieved passages do not support an answer
- FastAPI endpoints for uploads, document listing, and questions
- A reproducible, CPU-only Docker image and Docker Compose workflow
- Input, file-size, file-type, path, and upstream-error handling
- Unit and API tests that do not require Gemini network calls
- Optional live check that exercises the complete API and Gemini pipeline

## Repository layout

```text
paper-rag/
├── data/
│   ├── samples/              # Example PDFs
│   └── indexes/              # Generated FAISS indexes; ignored by Git
├── scripts/
│   └── check_api.py          # Live end-to-end API check
├── src/paper_rag/
│   ├── api.py                # FastAPI application
│   ├── chunking.py           # Token-window construction
│   ├── config.py             # Environment configuration
│   ├── embeddings.py         # Local embedding model
│   ├── llm.py                # Gemini client and response validation
│   ├── pdf.py                # PDF text extraction
│   ├── prepare.py            # PDF-to-index command
│   ├── rag.py                # Retrieval and generation pipeline
│   ├── retrieval.py          # Semantic-search command
│   ├── schemas.py            # API request and response models
│   ├── service.py            # Document ingestion and persistence
│   └── vector_store.py       # FAISS persistence and search
├── tests/
├── .dockerignore
├── .env.example
├── .gitignore
├── compose.yaml
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

## Requirements

- Python 3.11 or newer
- Internet access on the first embedding-model load
- A Gemini API key for answer generation

For the container workflow, install Docker Desktop and make sure its engine is
running. A host Python installation is only needed for the local workflow.

The embedding dependencies include PyTorch, so installation and the first model
load can take several minutes. FAISS is installed from PyPI; the embedding model
is downloaded from Hugging Face and cached locally.

## Installation

Clone the repository, then create a virtual environment:

```bash
git clone https://github.com/RobertSarinyan/paper-rag.git
cd paper-rag
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Git Bash, macOS, or Linux:

```bash
python -m venv .venv
source .venv/bin/activate  # Git Bash on Windows: source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs the package in editable mode with its development
dependencies. Runtime dependencies are declared once in `pyproject.toml`.

## Configuration

Create your private environment file from the example.

PowerShell:

```powershell
Copy-Item .env.example .env
```

Bash:

```bash
cp .env.example .env
```

Then set:

```dotenv
GEMINI_API_KEY=your-api-key
GEMINI_MODEL=gemini-3.6-flash
```

`.env` is ignored by Git. Never commit an API key.

## Prepare a PDF

Build a persistent index for the attention paper:

```bash
paper-rag-prepare data/samples/1706.03762v7.pdf
```

Equivalent module command:

```bash
python -m paper_rag.prepare data/samples/1706.03762v7.pdf
```

With the current model and defaults, expect 15 extracted pages, 75 chunks,
384-dimensional vectors, and these files:

```text
data/indexes/1706.03762v7/
├── metadata.json
└── vectors.faiss
```

Prepare the budget report in the same way:

```bash
paper-rag-prepare data/samples/59822_MBR.pdf
```

It currently produces 28 chunks from 7 pages. Counts can change slightly with
tokenizer-library updates.

## Inspect retrieval without Gemini

```bash
paper-rag-retrieve data/indexes/1706.03762v7 \
  "Why are attention dot products divided by the square root of the key dimension?"
```

This embeds the question locally and prints the four closest chunks, their
cosine-similarity scores, pages, and chunk IDs.

## Ask from the terminal

```bash
paper-rag-ask data/indexes/1706.03762v7 \
  "Why are attention dot products divided by the square root of the key dimension?" \
  --top-k 4 --show-retrieved
```

A supported answer should cite page 4, usually chunk 16. An unrelated question,
such as `What was the weather in Yerevan when this paper was written?`, should
return `Answer supported: no`.

## Run the API

For development:

```bash
python -m uvicorn paper_rag.api:app --reload
```

The server listens at `http://127.0.0.1:8000`. Open the interactive API at:

- `http://127.0.0.1:8000/docs`

The root path `/` is not an application page and returns `404`; use `/docs`.

## Run with Docker

Docker packages the application, Python, and its dependencies into one image.
Docker Compose starts that image with the API port, environment variables, and
persistent storage already connected. You do not need to activate `.venv` for
this workflow.

Make sure Docker Desktop is open and its engine is running. From the repository
root, build the image:

```bash
docker compose build
```

The first build is large because it installs the CPU version of PyTorch and the
embedding stack. Later builds reuse Docker's cache unless relevant files change.

Start the API in the background:

```bash
docker compose up -d
docker compose ps
```

When the service becomes healthy, open:

- `http://localhost:8000/docs`

View application output or follow it continuously:

```bash
docker compose logs api
docker compose logs -f api
```

Stop and remove the container and Compose network:

```bash
docker compose down
```

The generated indexes remain in `data/indexes/` on the host. The downloaded
Hugging Face model remains in the named Docker volume `huggingface-cache`, so
stopping the container does not delete either one. To start the existing image
again, run `docker compose up -d`. Rebuild with `docker compose build` after
changing application source, dependencies, or the Dockerfile.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Verify that the API process responds |
| `GET` | `/documents` | List persisted document indexes |
| `POST` | `/upload` | Upload and synchronously index one PDF, up to 20 MiB |
| `POST` | `/ask` | Ask a question about one indexed document |

Upload a PDF with `curl`:

```bash
curl -F "file=@data/samples/1706.03762v7.pdf;type=application/pdf" \
  http://127.0.0.1:8000/upload
```

The response contains a generated `document_id`. Use it in `/ask`:

```bash
curl -H "Content-Type: application/json" \
  -d '{
    "document_id": "replace-with-document-id",
    "question": "What is the purpose of multi-head attention?",
    "top_k": 4
  }' \
  http://127.0.0.1:8000/ask
```

An answer response includes:

- `answer` and the model's `answerable` assessment
- `cited_sources`, the passages cited in the answer
- `retrieved_sources`, every passage supplied to Gemini
- Page numbers, chunk IDs, similarity scores, and source text

The exact text depends on Gemini, but the response shape is similar to:

```json
{
  "document_id": "1706.03762v7-...",
  "question": "What is the purpose of multi-head attention?",
  "answer": "Multi-head attention lets the model attend to information from different representation subspaces.",
  "answerable": true,
  "model": "gemini-3.6-flash",
  "cited_sources": [
    {
      "source_id": "S1",
      "document_id": "1706.03762v7-...",
      "source_filename": "1706.03762v7.pdf",
      "page_number": 4,
      "chunk_id": 17,
      "score": 0.62,
      "text": "..."
    }
  ],
  "retrieved_sources": [
    {
      "source_id": "S1",
      "document_id": "1706.03762v7-...",
      "source_filename": "1706.03762v7.pdf",
      "page_number": 4,
      "chunk_id": 17,
      "score": 0.62,
      "text": "..."
    }
  ]
}
```

Common response codes are `201` for a completed upload, `422` for invalid input,
`404` for an unknown document, `413` for an oversized file, and `502` when
Gemini cannot produce a valid response.

## Testing

Run the deterministic test suite:

```bash
python -m pytest -q
```

These tests use temporary indexes and fake answer generators; they do not spend
Gemini quota. They cover chunk provenance, FAISS persistence, retrieval,
citation validation, PDF upload behavior, request validation, error sanitizing,
and indexes surviving an application restart.

With the API running, execute the optional live check:

```bash
python scripts/check_api.py --check-refusal
```

This uploads the attention paper, makes one supported and one unsupported Gemini
request, checks citations and HTTP validation, and leaves the new index in
`data/indexes/` for inspection.

## Design decisions

- Each index represents one PDF and stores both vectors and source metadata.
- Embeddings are normalized, so FAISS inner product equals cosine similarity.
- Chunks never cross page boundaries, keeping citations understandable.
- Retrieved PDF text is treated as untrusted quoted data in the LLM prompt.
- Gemini returns structured JSON; the application verifies every citation ID.
- Upload processing is synchronous for a transparent local MVP.
- Embedding and generation are serialized to keep local memory use bounded.

## Limitations

- Scanned PDFs require OCR, which is not included.
- PDF extraction can produce imperfect spacing or mathematical symbols.
- FAISS uses exact search, which is suitable for this MVP's small collections.
- There is no authentication, authorization, rate limiting, or multi-user data
  isolation.
- Uploads and model requests run synchronously and can take time.
- The `answerable` flag is an LLM assessment and is not a correctness guarantee;
  users should inspect cited text.
- Index metadata is trusted local data and should not be loaded from untrusted
  sources.

## Technology

Python, FastAPI, Pydantic, pypdf, Sentence Transformers, PyTorch, NumPy, FAISS,
Google GenAI, Uvicorn, HTTPX, pytest, and Docker Compose.
