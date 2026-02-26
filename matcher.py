"""
matcher.py
──────────
Orchestrates the three-stage pipeline:

  Stage 1 – Knowledge Graph  → structural scores + typed signals
  Stage 2 – ChromaDB         → semantic similarity scores + retrieval rank
  Stage 3 – Mistral 7B       → natural-language reasoning per candidate

Final score = weighted blend of KG score and ChromaDB score.
  final = (KG_WEIGHT × kg_score) + (CHROMA_WEIGHT × chroma_score)
"""

import asyncio
import logging
from typing import List

from schemas import MatchRequest, MatchResult
from knowledge_graph import kg_filter_and_score
from vector_store import get_retrieval_scores
from llm_reasoner import generate_reason

log = logging.getLogger(__name__)

KG_WEIGHT = float(0.45)        # Knowledge graph contributes 45%
CHROMA_WEIGHT = float(0.55)    # Semantic retrieval contributes 55%
MIN_SCORE_THRESHOLD = 0.0      # Set > 0 to filter out weak matches


async def run_matching(request: MatchRequest) -> List[MatchResult]:
    candidates = request.network_profiles
    user_profile = request.user_profile
    user_objective = request.user_objective

    if not candidates:
        return []

    # ── Stage 1: Knowledge Graph ───────────────────────────────────────────────
    log.info("Stage 1: Running knowledge graph scoring …")
    kg_results = kg_filter_and_score(user_profile, user_objective, candidates)
    # {profile_id: (kg_score, [signals])}

    # ── Stage 2: ChromaDB semantic retrieval ───────────────────────────────────
    log.info("Stage 2: Running ChromaDB semantic retrieval …")
    chroma_results = get_retrieval_scores(user_profile, user_objective, candidates)
    # {profile_id: (chroma_score_0_to_100, rank)}

    # ── Stage 3: LLM reasoning (parallelised) ─────────────────────────────────
    log.info("Stage 3: Generating LLM reasons …")

    async def process_candidate(c) -> MatchResult:
        pid = c.profile_id
        kg_score, signals = kg_results.get(pid, (0.0, []))
        chroma_score, rank = chroma_results.get(pid, (0.0, None))

        final_score = round(
            KG_WEIGHT * kg_score + CHROMA_WEIGHT * chroma_score, 2
        )

        reason = await generate_reason(
            user_profile, user_objective, c, signals, kg_score, chroma_score
        )

        return MatchResult(
            profile_id=pid,
            name=c.name,
            score=final_score,
            reason=reason,
            kg_signals=signals,
            retrieval_rank=rank,
        )

    tasks = [process_candidate(c) for c in candidates]
    results: List[MatchResult] = await asyncio.gather(*tasks)

    # Sort by score descending, filter by threshold
    results = [r for r in results if r.score >= MIN_SCORE_THRESHOLD]
    results.sort(key=lambda r: r.score, reverse=True)

    log.info(f"Matching complete. {len(results)} candidates scored.")
    return results
