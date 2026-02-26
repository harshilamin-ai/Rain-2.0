# Matchmaker API

A production-ready, stateless candidate-matching API built on a **three-stage AI pipeline**:

```
┌──────────────┐     ┌───────────────────┐     ┌──────────────────────┐
│   Request    │────▶│  Stage 1          │────▶│  Stage 2             │
│  (JSON body) │     │  Knowledge Graph  │     │  ChromaDB Retrieval  │
└──────────────┘     │  (NetworkX)       │     │  (all-MiniLM-L6-v2)  │
                     │                   │     └──────────┬───────────┘
                     │ Typed nodes:      │                │
                     │  USER, CANDIDATE  │     ┌──────────▼───────────┐
                     │  SKILL, TITLE     │     │  Stage 3             │
                     │  INDUSTRY, GOAL   │────▶│  Mistral-7B Reasoning│
                     └───────────────────┘     │  (Ollama / HF / Auto)│
                                               └──────────┬───────────┘
                                                          │
                                               ┌──────────▼───────────┐
                                               │  Ranked MatchResults  │
                                               │  score + reason +     │
                                               │  kg_signals + rank    │
                                               └───────────────────────┘
```

---

## Architecture

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Web framework | FastAPI + Uvicorn | Async REST API, auto OpenAPI docs |
| Knowledge Graph | NetworkX DiGraph | Structural candidate filtering with typed edges |
| Vector Store | ChromaDB (ephemeral) | Semantic similarity via cosine distance |
| Embeddings | `all-MiniLM-L6-v2` | Local, no API key, runs at startup |
| LLM Reasoning | Mistral-7B via Ollama | Natural-language match explanations |
| Deployment | Docker + Kubernetes | Horizontal scaling, GPU-ready |

### Scoring Formula
```
final_score = 0.45 × KG_score + 0.55 × ChromaDB_score
```

| Score Range | Meaning |
|-------------|---------|
| 70–100 | Very strong semantic match |
| 40–70 | Relevant / contextual fit |
| 20–40 | Weak alignment |
| < 20 | Likely irrelevant |

---

## Quick Start

### Option A — Docker Compose (recommended)

```bash
# 1. Clone / copy project
git clone <your-repo>
cd matchmaker-api

# 2. Copy env file
cp .env.example .env

# 3. Start everything (Ollama + Mistral pull + API)
docker compose up --build

# API available at http://localhost:8000
# Swagger UI at  http://localhost:8000/docs
```

> First run pulls the `mistral` model (~4 GB). Subsequent starts use cached volume.

### Option B — Local Python (no Docker)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Start Ollama separately for LLM reasoning
ollama serve &
ollama pull mistral

# 3. Run API
cp .env.example .env
python main.py
```

### Option C — Kubernetes

```bash
# Build and push image to your registry
docker build -t your-registry/matchmaker-api:latest .
docker push your-registry/matchmaker-api:latest

# Update image reference in k8s.yaml, then apply
kubectl apply -f k8s.yaml

# Check status
kubectl get pods -n matchmaker
```

---

## API Reference

### `POST /match`

**Request body:**
```json
{
  "user_profile": {
    "current_role": { "title": "Managing Partner", "company": "TestCo", "location": "London" },
    "previous_roles": [{ "title": "Cybersecurity Consultant", "company": "SecureWorks" }],
    "top_skills": [{ "skill": "Cybersecurity", "applied_in": "Governance" }],
    "solutions_offered": ["Cybersecurity Advisory"],
    "career_highlights": ["Led enterprise cybersecurity programs"]
  },
  "user_objective": {
    "person_id": "u1",
    "primary_goal": "Find cybersecurity leaders",
    "secondary_goals": [],
    "target_profiles": [{ "type": "Professional", "titles": ["CISO"], "why": "Decision maker" }],
    "exclude": [],
    "success_signals": ["Leadership experience"]
  },
  "network_profiles": [
    {
      "profile_id": "p1",
      "name": "Alice Smith",
      "title": "Chief Information Security Officer",
      "company": "FinBank",
      "industry": "Financial Services",
      "skills": ["Cybersecurity", "Governance"],
      "summary": "Leads security strategy for a global bank"
    }
  ]
}
```

**Response:**
```json
[
  {
    "profile_id": "p1",
    "name": "Alice Smith",
    "score": 72.45,
    "reason": "Experienced CISO with cybersecurity governance skills directly matching the advisory focus.",
    "kg_signals": ["Shared skill: Cybersecurity", "Title match: CISO"],
    "retrieval_rank": 1
  }
]
```

### `GET /health`
Liveness probe → `{"status": "ok"}`

### `GET /ready`
Readiness probe → `{"status": "ready"}` (503 until embedding model is loaded)

---

## LLM Backend Configuration

Set `LLM_BACKEND` in `.env`:

| Value | Behaviour |
|-------|-----------|
| `auto` | Try Ollama → HuggingFace → deterministic fallback |
| `ollama` | Ollama only (fails gracefully to fallback) |
| `hf` | HuggingFace Inference API (requires `HF_API_TOKEN`) |
| `none` | Skip LLM — deterministic reason always used |

---

## File Structure

```
matchmaker-api/
├── main.py            # FastAPI app, routes, lifespan
├── matcher.py         # 3-stage pipeline orchestrator
├── knowledge_graph.py # NetworkX KG builder + structural scorer
├── vector_store.py    # ChromaDB ephemeral collection + retrieval
├── llm_reasoner.py    # Mistral-7B via Ollama / HuggingFace
├── schemas.py         # Pydantic request/response models
├── requirements.txt
├── Dockerfile
├── docker-compose.yml # API + Ollama sidecar
├── k8s.yaml           # Kubernetes manifests + HPA
└── .env.example
```

---

## GPU Acceleration

To use a GPU for faster Mistral-7B inference, uncomment the GPU section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Requires NVIDIA Container Toolkit installed on the host.
