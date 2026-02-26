"""
vector_store.py
───────────────
Uses ChromaDB (in-memory, ephemeral per request) to embed candidate profiles
and retrieve the most semantically similar ones to the user's objective.

Embedding model: sentence-transformers/all-MiniLM-L6-v2 (local, no API key).
Each request creates a fresh ephemeral collection so the service stays stateless.
"""

import uuid
from typing import Dict, List, Tuple

import chromadb

from schemas import UserProfileInfo, UserObjective, NetworkProfile

# ── Singleton embedding function (loaded once at startup) ──────────────────────
from sentence_transformers import SentenceTransformer
from chromadb import EmbeddingFunction, Documents, Embeddings

_EMBED_MODEL = "all-MiniLM-L6-v2"
_st_model = SentenceTransformer(_EMBED_MODEL)

class STEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return _st_model.encode(list(input)).tolist()

_embed_fn = STEmbeddingFunction()


def _build_candidate_document(c: NetworkProfile) -> str:
    """Convert a candidate profile to a rich text document for embedding."""
    parts = [
        f"Name: {c.name}",
        f"Title: {c.title}",
    ]
    if c.company:
        parts.append(f"Company: {c.company}")
    if c.industry:
        parts.append(f"Industry: {c.industry}")
    if c.skills:
        parts.append(f"Skills: {', '.join(c.skills)}")
    if c.summary:
        parts.append(f"Summary: {c.summary}")
    return ". ".join(parts)


def _build_query_document(
    user_profile: UserProfileInfo, user_objective: UserObjective
) -> str:
    """Synthesise the user's intent into a query string for retrieval."""
    parts = [
        f"Goal: {user_objective.primary_goal}",
    ]
    for tp in user_objective.target_profiles:
        parts.append(f"Seeking: {', '.join(tp.titles)} — {tp.why or ''}")
    if user_objective.success_signals:
        parts.append(f"Success signals: {', '.join(user_objective.success_signals)}")
    if user_profile.top_skills:
        skills_text = ", ".join(sk.skill for sk in user_profile.top_skills)
        parts.append(f"User skills: {skills_text}")
    if user_profile.solutions_offered:
        parts.append(f"Solutions offered: {', '.join(user_profile.solutions_offered)}")
    return ". ".join(parts)


def retrieve_ranked_candidates(
    user_profile: UserProfileInfo,
    user_objective: UserObjective,
    candidates: List[NetworkProfile],
    top_k: int = 5,
) -> List[Tuple[str, float, int]]:
    """
    Embed all candidates, query with user intent, return ranked list.

    Returns
    -------
    List of (profile_id, cosine_distance, retrieval_rank) sorted by relevance.
    Distance is ChromaDB's cosine distance (lower = closer).
    """
    if not candidates:
        return []

    # Ephemeral in-memory client — new UUID collection per request → stateless
    client = chromadb.EphemeralClient()
    collection_name = f"matches_{uuid.uuid4().hex[:8]}"
    collection = client.create_collection(
        name=collection_name,
        embedding_function=_embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Index all candidates
    docs = [_build_candidate_document(c) for c in candidates]
    ids = [c.profile_id for c in candidates]
    metadatas = [{"name": c.name, "profile_id": c.profile_id} for c in candidates]

    collection.add(documents=docs, ids=ids, metadatas=metadatas)

    # Query
    query_text = _build_query_document(user_profile, user_objective)
    n_results = min(top_k, len(candidates))
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["distances", "metadatas"],
    )

    ranked: List[Tuple[str, float, int]] = []
    for rank, (pid, dist) in enumerate(
        zip(results["ids"][0], results["distances"][0]), start=1
    ):
        ranked.append((pid, dist, rank))

    return ranked


def get_retrieval_scores(
    user_profile: UserProfileInfo,
    user_objective: UserObjective,
    candidates: List[NetworkProfile],
) -> Dict[str, Tuple[float, int]]:
    """
    Convenience wrapper.

    Returns {profile_id: (similarity_score_0_to_1, rank)}
    where similarity = 1 - cosine_distance.
    """
    ranked = retrieve_ranked_candidates(user_profile, user_objective, candidates)
    return {
        pid: (round((1 - dist) * 100, 4), rank)  # convert to 0-100 scale
        for pid, dist, rank in ranked
    }
