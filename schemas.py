from pydantic import BaseModel, Field
from typing import List, Optional


# ── Input Schemas ──────────────────────────────────────────────────────────────

class Role(BaseModel):
    title: str
    company: Optional[str] = None
    location: Optional[str] = None


class Skill(BaseModel):
    skill: str
    applied_in: Optional[str] = None


class UserProfileInfo(BaseModel):
    current_role: Role
    previous_roles: Optional[List[Role]] = []
    top_skills: Optional[List[Skill]] = []
    solutions_offered: Optional[List[str]] = []
    career_highlights: Optional[List[str]] = []


class TargetProfile(BaseModel):
    type: str
    titles: List[str]
    why: Optional[str] = None


class UserObjective(BaseModel):
    person_id: str
    primary_goal: str
    secondary_goals: Optional[List[str]] = []
    target_profiles: List[TargetProfile]
    exclude: Optional[List[str]] = []
    success_signals: Optional[List[str]] = []


class NetworkProfile(BaseModel):
    profile_id: str
    name: str
    title: str
    company: Optional[str] = None
    industry: Optional[str] = None
    skills: Optional[List[str]] = []
    summary: Optional[str] = None


class MatchRequest(BaseModel):
    user_profile: UserProfileInfo
    user_objective: UserObjective
    network_profiles: List[NetworkProfile]


# ── Output Schemas ─────────────────────────────────────────────────────────────

class MatchResult(BaseModel):
    profile_id: str
    name: str
    score: float
    reason: str
    kg_signals: Optional[List[str]] = Field(default=[], description="Knowledge graph matched signals")
    retrieval_rank: Optional[int] = Field(default=None, description="ChromaDB retrieval rank")
