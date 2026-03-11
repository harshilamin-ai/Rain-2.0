"""
matcher.py — LLM reasoning for top 3 only
"""

import asyncio
import logging
from typing import List

from schemas import MatchRequest, MatchResult
from knowledge_graph import kg_filter_and_score
from vector_store import get_retrieval_scores
from llm_reasoner import generate_reason, _fallback_reason

log = logging.getLogger(__name__)

KG_WEIGHT           = float(0.35)
CHROMA_WEIGHT       = float(0.65)
MIN_SCORE_THRESHOLD = 0.0

LLM_CONCURRENCY = 1   # Ollama CPU handles 1 at a time
LLM_TOP_N       = 3   # ← Only top 3 get Mistral reasoning
_llm_semaphore  = asyncio.Semaphore(LLM_CONCURRENCY)


async def run_matching(request: MatchRequest) -> List[MatchResult]:
    candidates     = request.network_profiles
    user_profile   = request.user_profile
    user_objective = request.user_objective

    if not candidates:
        return []

    # ── Stage 1: Knowledge Graph ──────────────────────────────────────────────
    log.info("Stage 1: Running knowledge graph scoring …")
    kg_results = kg_filter_and_score(user_profile, user_objective, candidates)

    # ── Stage 2: ChromaDB semantic retrieval ──────────────────────────────────
    log.info("Stage 2: Running ChromaDB semantic retrieval …")
    chroma_results = get_retrieval_scores(user_profile, user_objective, candidates)

    # ── Pre-score everyone to find the top 3 ─────────────────────────────────
    def get_final_score(c) -> float:
        kg_score, _     = kg_results.get(c.profile_id, (0.0, []))
        chroma_score, _ = chroma_results.get(c.profile_id, (0.0, None))
        return round(KG_WEIGHT * kg_score + CHROMA_WEIGHT * chroma_score, 2)

    sorted_candidates = sorted(candidates, key=get_final_score, reverse=True)
    top3_ids = {c.profile_id for c in sorted_candidates[:LLM_TOP_N]}

    log.info(f"Stage 3: Mistral reasoning for top {LLM_TOP_N} candidates only …")
    for c in sorted_candidates[:LLM_TOP_N]:
        log.info(f"  → {c.name} (score: {get_final_score(c)})")

    # ── Stage 3: Reason top 3 with Mistral, rest get instant fallback ─────────
    async def process_candidate(c) -> MatchResult:
        pid               = c.profile_id
        kg_score, signals = kg_results.get(pid, (0.0, []))
        chroma_score, rank = chroma_results.get(pid, (0.0, None))
        final_score       = round(KG_WEIGHT * kg_score + CHROMA_WEIGHT * chroma_score, 2)

        if pid in top3_ids:
            async with _llm_semaphore:
                reason = await generate_reason(
                    user_profile, user_objective, c, signals, kg_score, chroma_score
                )
        else:
            reason = _fallback_reason(user_objective, c, signals)

        return MatchResult(
            profile_id=pid,
            name=c.name,
            score=final_score,
            reason=reason,
            kg_signals=signals,
            retrieval_rank=rank,
        )

    tasks   = [process_candidate(c) for c in candidates]
    results: List[MatchResult] = await asyncio.gather(*tasks)

    results = [r for r in results if r.score >= MIN_SCORE_THRESHOLD]
    results.sort(key=lambda r: r.score, reverse=True)

    log.info(f"Matching complete. {len(results)} candidates scored.")
    return results