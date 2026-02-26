"""
knowledge_graph.py
──────────────────
Builds a typed property graph over user + candidate data.

Nodes
  • USER          – the requesting user
  • CANDIDATE     – each network profile
  • SKILL         – unique skill string
  • TITLE         – job title / role keyword
  • INDUSTRY      – industry tag
  • GOAL          – objective / signal keyword

Edges (directed)
  USER      –[HAS_SKILL]→     SKILL
  USER      –[SEEKS_TITLE]→   TITLE
  USER      –[HAS_GOAL]→      GOAL
  CANDIDATE –[HAS_SKILL]→     SKILL
  CANDIDATE –[HAS_TITLE]→     TITLE
  CANDIDATE –[IN_INDUSTRY]→   INDUSTRY

Scoring: structural overlap between USER intent nodes and CANDIDATE nodes.
"""

import re
import networkx as nx
from typing import Dict, List, Tuple

from schemas import UserProfileInfo, UserObjective, NetworkProfile


def _normalise(text: str) -> str:
    return re.sub(r"\s+", "_", text.strip().lower())


def build_graph(
    user_profile: UserProfileInfo,
    user_objective: UserObjective,
    candidates: List[NetworkProfile],
) -> nx.DiGraph:
    G = nx.DiGraph()

    # ── User node ──────────────────────────────────────────────────────────────
    user_id = f"user::{user_objective.person_id}"
    G.add_node(user_id, node_type="USER", label=user_profile.current_role.title)

    # User skills
    for sk in user_profile.top_skills or []:
        sk_node = f"skill::{_normalise(sk.skill)}"
        G.add_node(sk_node, node_type="SKILL", label=sk.skill)
        G.add_edge(user_id, sk_node, rel="HAS_SKILL")

    # Sought titles from target_profiles
    for tp in user_objective.target_profiles:
        for t in tp.titles:
            t_node = f"title::{_normalise(t)}"
            G.add_node(t_node, node_type="TITLE", label=t)
            G.add_edge(user_id, t_node, rel="SEEKS_TITLE", why=tp.why or "")

    # Success signals as GOAL nodes
    for sig in user_objective.success_signals or []:
        g_node = f"goal::{_normalise(sig)}"
        G.add_node(g_node, node_type="GOAL", label=sig)
        G.add_edge(user_id, g_node, rel="HAS_GOAL")

    # ── Candidate nodes ────────────────────────────────────────────────────────
    for c in candidates:
        c_id = f"candidate::{c.profile_id}"
        G.add_node(
            c_id,
            node_type="CANDIDATE",
            label=c.name,
            title=c.title,
            company=c.company or "",
            industry=c.industry or "",
        )

        # Skills
        for sk in c.skills or []:
            sk_node = f"skill::{_normalise(sk)}"
            if not G.has_node(sk_node):
                G.add_node(sk_node, node_type="SKILL", label=sk)
            G.add_edge(c_id, sk_node, rel="HAS_SKILL")

        # Title keywords (each word as a possible title match)
        for word in c.title.split():
            if len(word) > 3:
                t_node = f"title::{_normalise(word)}"
                if not G.has_node(t_node):
                    G.add_node(t_node, node_type="TITLE", label=word)
                G.add_edge(c_id, t_node, rel="HAS_TITLE")

        # Full title node
        full_t_node = f"title::{_normalise(c.title)}"
        G.add_node(full_t_node, node_type="TITLE", label=c.title)
        G.add_edge(c_id, full_t_node, rel="HAS_TITLE")

        # Industry
        if c.industry:
            ind_node = f"industry::{_normalise(c.industry)}"
            if not G.has_node(ind_node):
                G.add_node(ind_node, node_type="INDUSTRY", label=c.industry)
            G.add_edge(c_id, ind_node, rel="IN_INDUSTRY")

    return G


def score_candidate_kg(
    G: nx.DiGraph,
    user_id: str,
    candidate_id: str,
) -> Tuple[float, List[str]]:
    """
    Returns (score 0-100, list of matched signal descriptions).

    Scoring breakdown
    -----------------
    Shared SKILL nodes    : 15 pts each  (max 45)
    SEEKS_TITLE hit       : 20 pts each  (max 40)
    GOAL / signal overlap : 10 pts each  (max 20)
    ───────────────────────────────────────────────
    Raw cap               : 100
    """
    signals: List[str] = []
    score = 0.0

    user_neighbours = {n: G[user_id][n]["rel"] for n in G.successors(user_id)}
    cand_neighbours = {n: G[candidate_id][n]["rel"] for n in G.successors(candidate_id)}

    # Skill overlap
    user_skills = {n for n, r in user_neighbours.items() if r == "HAS_SKILL"}
    cand_skills = {n for n, r in cand_neighbours.items() if r == "HAS_SKILL"}
    shared_skills = user_skills & cand_skills
    for s in shared_skills:
        label = G.nodes[s].get("label", s)
        signals.append(f"Shared skill: {label}")
        score += 15

    # Title match: user SEEKS_TITLE → candidate HAS_TITLE
    user_titles = {n for n, r in user_neighbours.items() if r == "SEEKS_TITLE"}
    cand_titles = {n for n, r in cand_neighbours.items() if r == "HAS_TITLE"}
    matched_titles = user_titles & cand_titles
    for t in matched_titles:
        label = G.nodes[t].get("label", t)
        signals.append(f"Title match: {label}")
        score += 20

    # Partial title match (token-level)
    for ut in user_titles:
        ut_label = G.nodes[ut].get("label", "").lower()
        for ct in cand_titles:
            ct_label = G.nodes[ct].get("label", "").lower()
            if ut not in matched_titles and (ut_label in ct_label or ct_label in ut_label):
                signals.append(f"Partial title match: {ut_label} ↔ {ct_label}")
                score += 10
                matched_titles.add(ut)  # avoid double-count

    # Goal signals (keyword overlap in candidate skills / title)
    user_goals = {n for n, r in user_neighbours.items() if r == "HAS_GOAL"}
    cand_all_nodes = set(cand_skills) | set(cand_titles)
    for g in user_goals:
        g_label = G.nodes[g].get("label", "").lower()
        for cn in cand_all_nodes:
            cn_label = G.nodes[cn].get("label", "").lower()
            if g_label in cn_label or cn_label in g_label:
                signals.append(f"Goal signal match: {g_label}")
                score += 10
                break

    return min(score, 100.0), signals


def kg_filter_and_score(
    user_profile: UserProfileInfo,
    user_objective: UserObjective,
    candidates: List[NetworkProfile],
) -> Dict[str, Tuple[float, List[str]]]:
    """Build graph and return {profile_id: (score, signals)} for all candidates."""
    G = build_graph(user_profile, user_objective, candidates)
    user_id = f"user::{user_objective.person_id}"
    results: Dict[str, Tuple[float, List[str]]] = {}
    for c in candidates:
        c_id = f"candidate::{c.profile_id}"
        score, signals = score_candidate_kg(G, user_id, c_id)
        results[c.profile_id] = (score, signals)
    return results
