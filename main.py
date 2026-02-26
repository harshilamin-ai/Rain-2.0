"""
main.py
───────
FastAPI entry point for the Matchmaker API.

Endpoints
  POST /match          – full 3-stage pipeline
  GET  /health         – liveness probe
  GET  /ready          – readiness probe (checks embedding model loaded)
  GET  /docs           – Swagger UI (auto-generated)
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schemas import MatchRequest, MatchResult
from matcher import run_matching

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)

# ── Startup: warm-up embedding model ──────────────────────────────────────────
_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ready
    log.info("Warming up sentence-transformer embedding model …")
    try:
        from vector_store import _embed_fn
        _embed_fn(["warm-up"])   # triggers model download / load
        log.info("Embedding model ready.")
    except Exception as e:
        log.warning(f"Embedding model warm-up failed: {e}")
    _ready = True
    yield
    log.info("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Matchmaker API",
    description=(
        "Three-stage candidate matching pipeline:\n"
        "1. **Knowledge Graph** (NetworkX) — typed structural scoring\n"
        "2. **ChromaDB** — semantic vector retrieval\n"
        "3. **Mistral-7B** — natural-language reasoning\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware ──────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Process-Time-Ms"] = str(elapsed)
    return response


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"], summary="Liveness probe")
async def health():
    return {"status": "ok"}


@app.get("/ready", tags=["ops"], summary="Readiness probe")
async def ready():
    if not _ready:
        raise HTTPException(status_code=503, detail="Service not ready yet")
    return {"status": "ready"}


@app.post(
    "/match",
    response_model=List[MatchResult],
    tags=["matching"],
    summary="Run the three-stage matching pipeline",
    responses={
        200: {"description": "Ranked list of matched candidates"},
        422: {"description": "Validation error — check request schema"},
        500: {"description": "Internal server error"},
    },
)
async def match_candidates(request: MatchRequest):
    """
    Submit a user profile + objective + network of candidates.
    Returns candidates ranked by a blended Knowledge-Graph + Semantic score,
    with an LLM-generated natural-language reason for each match.
    """
    if not request.network_profiles:
        return []

    try:
        results = await run_matching(request)
        return results
    except Exception as e:
        log.exception("Matching pipeline error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Matching failed: {str(e)}",
        )


# ── Dev entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
