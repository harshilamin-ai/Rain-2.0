# ── Build stage ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="matchmaker-api"
LABEL description="Three-stage candidate matching: Knowledge Graph + ChromaDB + Mistral-7B"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # LLM backend: auto | ollama | hf | none
    LLM_BACKEND=auto \
    # Ollama endpoint (override in docker-compose / k8s)
    OLLAMA_HOST=http://ollama:11434 \
    OLLAMA_MODEL=mistral \
    # HuggingFace token (optional)
    HF_API_TOKEN="" \
    LLM_TIMEOUT=30

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source
COPY . .

# Pre-download the sentence-transformer model at build time
# so the container starts instantly (no network needed at runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
