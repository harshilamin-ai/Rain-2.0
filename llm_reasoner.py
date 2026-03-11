"""
llm_reasoner.py
"""

import os
import logging
from typing import List, Optional

import httpx

from schemas import NetworkProfile, UserObjective, UserProfileInfo

log = logging.getLogger(__name__)

OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
LLM_BACKEND  = os.getenv("LLM_BACKEND", "auto").lower()
LLM_TIMEOUT  = int(os.getenv("LLM_TIMEOUT", "30"))
HF_MODEL     = "mistralai/Mistral-7B-Instruct-v0.2"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

def _build_prompt(
    user_profile: UserProfileInfo,
    user_objective: UserObjective,
    candidate: NetworkProfile,
    kg_signals: List[str],
) -> str:
    user_role      = user_profile.current_role.title
    user_company   = user_profile.current_role.company or ""
    user_skills    = ", ".join(sk.skill for sk in (user_profile.top_skills or []))
    user_solutions = ", ".join(user_profile.solutions_offered or [])
    target_titles  = ", ".join(t for tp in user_objective.target_profiles for t in tp.titles)
    signals_text   = "; ".join(kg_signals) if kg_signals else "none"
    why_targets    = ", ".join(tp.why for tp in user_objective.target_profiles if tp.why)

    return f"""<s>[INST]
You are a professional networking strategist. Your job is NOT to say two people "match" — your job is to explain the TRANSACTIONAL VALUE of a connection.

Answer this one question in a single sentence:
"What specific business outcome or opportunity does this connection unlock for the user?"

Think in terms of:
- What door does this person open for the user?
- What can the user sell, offer, or deliver to this person that solves a real problem?
- What becomes possible for the user's business BECAUSE of this connection?
- What is the ROI of spending time reaching out to this person?

RULES:
- Lead with the VALUE or OUTCOME, not the person's job title
- Be deal-focused and outcome-specific
- Do NOT say "strong match", "good fit", "aligns with", or "relevant experience"
- Do NOT mention scores, algorithms, or ranking systems
- Do NOT repeat the user's goal verbatim
- Do NOT start with "This candidate" or repeat their name
- Max 30 words, one sentence only

USER CONTEXT
  Role        : {user_role} at {user_company}
  Expertise   : {user_skills}
  Offering    : {user_solutions}
  Objective   : {user_objective.primary_goal}
  Target type : {target_titles} — because: {why_targets}
  Success looks like: {', '.join(user_objective.success_signals or [])}

CANDIDATE
  Title       : {candidate.title}
  Company     : {candidate.company or 'N/A'}
  Industry    : {candidate.industry or 'N/A'}
  Skills      : {', '.join(candidate.skills or [])}
  Summary     : {candidate.summary or 'N/A'}

SIGNALS THAT TRIGGERED THIS MATCH: {signals_text}

Respond with ONLY the single sentence. No preamble, no explanation, no punctuation beyond the sentence itself.
[/INST]"""


async def _call_ollama(prompt: str) -> Optional[str]:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": 80},
    }
    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            r = await client.post(f"{OLLAMA_HOST}/api/generate", json=payload)
            r.raise_for_status()
            text = r.json().get("response", "").strip()
            for prefix in ["Reason:", "Answer:", "Response:", "Result:", "-"]:
                if text.startswith(prefix):
                    text = text[len(prefix):].strip()
            return text if text else None
    except Exception as e:
        log.warning(f"Ollama call failed: {e}")
        return None


async def _call_hf(prompt: str) -> Optional[str]:
    if not HF_API_TOKEN:
        return None
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 80, "temperature": 0.4, "return_full_text": False},
    }
    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            r = await client.post(
                f"https://api-inference.huggingface.co/models/{HF_MODEL}",
                headers=headers, json=payload,
            )
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data:
                return data[0].get("generated_text", "").strip()
    except Exception as e:
        log.warning(f"HuggingFace call failed: {e}")
    return None


def _fallback_reason(
    user_objective: UserObjective,
    candidate: NetworkProfile,
    kg_signals: List[str],
) -> str:
    industry      = candidate.industry or "their sector"
    title         = candidate.title
    goal          = user_objective.primary_goal.lower() \
                        .replace("find ", "").replace("connect with ", "")
    skill_signals = [s for s in kg_signals if "skill" in s.lower()]
    title_signals = [s for s in kg_signals if "title" in s.lower()]

    if title_signals and skill_signals:
        skill = skill_signals[0].replace("Shared skill: ", "")
        return (
            f"Direct access to a {industry} leader with proven {skill} expertise — "
            f"a decision-maker positioned to act on your advisory offering."
        )
    elif skill_signals:
        skill = skill_signals[0].replace("Shared skill: ", "")
        return (
            f"Opens a direct line to a {industry} professional with hands-on {skill} experience, "
            f"positioning your services where the problem already exists."
        )
    elif title_signals:
        return (
            f"Provides access to a {title} in {industry} who has the authority "
            f"to commission the exact services you offer."
        )
    else:
        return (
            f"Connects you with a {industry} professional whose role creates a direct need "
            f"for your {goal} expertise."
        )


async def generate_reason(
    user_profile: UserProfileInfo,
    user_objective: UserObjective,
    candidate: NetworkProfile,
    kg_signals: List[str],
    kg_score: float,
    chroma_score: float,
) -> str:
    backend = LLM_BACKEND

    if backend in ("auto", "ollama"):
        result = await _call_ollama(
            _build_prompt(user_profile, user_objective, candidate, kg_signals)
        )
        if result:
            return result
        if backend == "ollama":
            log.warning("Ollama unavailable — using deterministic fallback.")
            return _fallback_reason(user_objective, candidate, kg_signals)

    if backend in ("auto", "hf"):
        result = await _call_hf(
            _build_prompt(user_profile, user_objective, candidate, kg_signals)
        )
        if result:
            return result

    return _fallback_reason(user_objective, candidate, kg_signals)