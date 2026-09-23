FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

ARG TORCH_VERSION=2.14.0

RUN python -m pip install --upgrade pip \
    && python -m pip install "torch==${TORCH_VERSION}" \
        --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install .

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/indexes "${HF_HOME}" \
    && chown -R appuser:appuser /app/data "${HF_HOME}"

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "paper_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]