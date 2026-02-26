"""
llm_reasoner.py
───────────────
Calls Mistral-7B to generate a one-sentence natural-language reason
for each candidate match.

Strategy (in priority order):
  1. Ollama local server  (OLLAMA_HOST env var, default http://localhost:11434)
  2. HuggingFace Inference API  (HF_API_TOKEN env var)
  3. Fallback deterministic summary  (no LLM required — always works)

Set LLM_BACKEND=ollama | hf | auto (default: auto)
"""

import os
import json
import logging
from typing import List, Optional

import httpx

from schemas import NetworkProfile, UserObjective, UserProfileInfo

log = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
LLM_BACKEND = os.getenv("LLM_BACKEND", "auto").lower()  # auto | ollama | hf | none
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))

HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _build_prompt(
    user_profile: UserProfileInfo,
    user_objective: UserObjective,
    candidate: NetworkProfile,
    kg_signals: List[str],
    kg_score: float,
    chroma_score: float,
) -> str:
    user_skills = ", ".join(sk.skill for sk in (user_profile.top_skills or []))
    target_titles = ", ".join(
        t for tp in user_objective.target_profiles for t in tp.titles
    )
    signals_text = "; ".join(kg_signals) if kg_signals else "none"

    return f"""<s>[INST]
You are an AI recruitment assistant. Given the context below, write a single concise sentence
(max 25 words) explaining why this candidate is a good match for the user's objective.
Be specific. Do not repeat the candidate's name in the reason.

USER CONTEXT
  Goal: {user_objective.primary_goal}
  Seeking: {target_titles}
  User skills: {user_skills}
  Success signals: {', '.join(user_objective.success_signals or [])}

CANDIDATE
  Title: {candidate.title}
  Company: {candidate.company or 'N/A'}
  Industry: {candidate.industry or 'N/A'}
  Skills: {', '.join(candidate.skills or [])}
  Summary: {candidate.summary or 'N/A'}

MATCH SIGNALS (from knowledge graph): {signals_text}
KG Score: {kg_score:.1f}/100   Semantic Score: {chroma_score:.1f}/100

Respond with ONLY the reason sentence, nothing else.
[/INST]"""


# ── Backend implementations ────────────────────────────────────────────────────

async def _call_ollama(prompt: str) -> Optional[str]:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 60},
    }
    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            r = await client.post(f"{OLLAMA_HOST}/api/generate", json=payload)
            r.raise_for_status()
            return r.json().get("response", "").strip()
    except Exception as e:
        log.warning(f"Ollama call failed: {e}")
        return None


async def _call_hf(prompt: str) -> Optional[str]:
    if not HF_API_TOKEN:
        return None
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 60, "temperature": 0.3, "return_full_text": False},
    }
    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            r = await client.post(
                f"https://api-inference.huggingface.co/models/{HF_MODEL}",
                headers=headers,
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data:
                return data[0].get("generated_text", "").strip()
    except Exception as e:
        log.warning(f"HuggingFace call failed: {e}")
    return None


def _fallback_reason(
    candidate: NetworkProfile,
    kg_signals: List[str],
    kg_score: float,
    chroma_score: float,
) -> str:
    """Deterministic reason when no LLM is available."""
    if kg_signals:
        top = kg_signals[0]
        return f"Strong match based on {top.lower()} with a combined alignment score of {((kg_score + chroma_score) / 2):.0f}/100."
    return (
        f"Candidate aligns semantically with the target profile "
    )


# ── Public interface ───────────────────────────────────────────────────────────

async def generate_reason(
    user_profile: UserProfileInfo,
    user_objective: UserObjective,
    candidate: NetworkProfile,
    kg_signals: List[str],
    kg_score: float,
    chroma_score: float,
) -> str:
    """
    Generate a match reason using the configured backend.
    Always returns a string (falls back to deterministic if LLM unavailable).
    """
    backend = LLM_BACKEND

    # Check Ollama availability once on first call (simple heuristic)
    if backend in ("auto", "ollama"):
        result = await _call_ollama(
            _build_prompt(user_profile, user_objective, candidate, kg_signals, kg_score, chroma_score)
        )
        if result:
            return result
        if backend == "ollama":
            log.warning("Ollama backend selected but unavailable; using fallback.")
            return _fallback_reason(candidate, kg_signals, kg_score, chroma_score)

    if backend in ("auto", "hf"):
        result = await _call_hf(
            _build_prompt(user_profile, user_objective, candidate, kg_signals, kg_score, chroma_score)
        )
        if result:
            return result

    return _fallback_reason(candidate, kg_signals, kg_score, chroma_score)
