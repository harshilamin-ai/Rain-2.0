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

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Matchmaker API — Architecture Diagram</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #e6edf3; min-height: 100vh; }

  .page { max-width: 1100px; margin: 0 auto; padding: 32px 24px; }

  h1 { text-align: center; font-size: 22px; font-weight: 700; color: #58a6ff; margin-bottom: 4px; letter-spacing: 1px; }
  .subtitle { text-align: center; font-size: 13px; color: #8b949e; margin-bottom: 32px; }

  /* ── FLOW ── */
  .flow { display: flex; flex-direction: column; gap: 0; }

  /* ── REQUEST ROW ── */
  .row { display: flex; align-items: center; justify-content: center; gap: 12px; margin-bottom: 0; }

  /* ── ARROW ── */
  .arrow-down { display: flex; justify-content: center; align-items: center; height: 44px; }
  .arrow-down svg { filter: drop-shadow(0 0 6px #58a6ff88); }

  .arrow-right { display: flex; align-items: center; }
  .arrow-right svg { filter: drop-shadow(0 0 4px #58a6ff66); }

  /* ── BOX BASE ── */
  .box {
    border-radius: 10px;
    padding: 14px 18px;
    border: 1px solid;
    position: relative;
    min-width: 140px;
    text-align: center;
  }

  .box .tag {
    position: absolute;
    top: -10px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1.2px;
    padding: 2px 8px;
    border-radius: 20px;
    white-space: nowrap;
  }

  .box .icon { font-size: 24px; display: block; margin-bottom: 4px; }
  .box .title { font-size: 13px; font-weight: 700; }
  .box .sub   { font-size: 11px; opacity: 0.75; margin-top: 3px; }

  /* colours */
  .box-client   { background: #161b22; border-color: #30363d; }
  .box-client .tag { background: #30363d; color: #8b949e; }

  .box-api      { background: #0d2340; border-color: #58a6ff; box-shadow: 0 0 18px #58a6ff33; }
  .box-api .tag { background: #58a6ff; color: #0d1117; }

  .box-kg       { background: #1a1a2e; border-color: #9b59b6; box-shadow: 0 0 14px #9b59b622; }
  .box-kg .tag  { background: #9b59b6; color: #fff; }

  .box-chroma   { background: #0d2b1a; border-color: #2ecc71; box-shadow: 0 0 14px #2ecc7122; }
  .box-chroma .tag { background: #2ecc71; color: #0d1117; }

  .box-llm      { background: #2b1a0d; border-color: #e67e22; box-shadow: 0 0 14px #e67e2222; }
  .box-llm .tag { background: #e67e22; color: #0d1117; }

  .box-out      { background: #1a1a1a; border-color: #f1c40f; box-shadow: 0 0 14px #f1c40f22; }
  .box-out .tag { background: #f1c40f; color: #0d1117; }

  .box-ollama   { background: #1a0d0d; border-color: #e74c3c; box-shadow: 0 0 14px #e74c3c22; }
  .box-ollama .tag { background: #e74c3c; color: #fff; }

  /* ── PIPELINE STAGES ── */
  .stages { display: flex; gap: 0; align-items: stretch; justify-content: center; }
  .stage-wrap { display: flex; flex-direction: column; align-items: center; }

  /* ── SCORE FORMULA ── */
  .formula-box {
    background: linear-gradient(135deg, #0d1f3c, #1a2a4a);
    border: 1px solid #58a6ff;
    border-radius: 10px;
    padding: 14px 28px;
    text-align: center;
    margin: 0 auto;
    max-width: 480px;
  }
  .formula-box .label { font-size: 11px; color: #8b949e; margin-bottom: 6px; letter-spacing: 1px; text-transform: uppercase; }
  .formula-box .formula { font-size: 16px; font-weight: 700; color: #58a6ff; font-family: monospace; }
  .formula-box .weights { display: flex; gap: 24px; justify-content: center; margin-top: 8px; }
  .formula-box .w { font-size: 11px; color: #8b949e; }
  .formula-box .w span { color: #e6edf3; font-weight: 600; }

  /* ── OUTPUT FIELDS ── */
  .outputs { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; margin-top: 0; }
  .output-chip {
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 12px;
    font-weight: 600;
    text-align: center;
    min-width: 130px;
    border: 1px solid;
  }
  .chip-score   { background: #0d2340; border-color: #58a6ff; color: #58a6ff; }
  .chip-reason  { background: #2b1a0d; border-color: #e67e22; color: #e67e22; }
  .chip-signals { background: #1a1a2e; border-color: #9b59b6; color: #9b59b6; }
  .chip-rank    { background: #0d2b1a; border-color: #2ecc71; color: #2ecc71; }

  /* ── TIERS ── */
  .tiers { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }
  .tier {
    border-radius: 8px;
    padding: 10px 18px;
    font-size: 12px;
    font-weight: 700;
    text-align: center;
    min-width: 150px;
    border: 1px solid;
  }
  .tier-hot  { background: #2b1a0d; border-color: #e74c3c; color: #e74c3c; }
  .tier-warm { background: #2b2508; border-color: #f1c40f; color: #f1c40f; }
  .tier-cold { background: #0d1a2b; border-color: #3498db; color: #3498db; }
  .tier .tscore { font-size: 10px; font-weight: 400; opacity: 0.8; margin-top: 3px; }

  /* ── SECTION LABELS ── */
  .section-label {
    text-align: center;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #8b949e;
    margin: 8px 0 12px;
  }

  /* ── DIVIDER ── */
  .divider {
    border: none;
    border-top: 1px dashed #30363d;
    margin: 20px 0;
  }

  /* ── TECH LEGEND ── */
  .legend { display: flex; gap: 16px; justify-content: center; flex-wrap: wrap; margin-top: 24px; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-size: 11px; color: #8b949e; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; }

  /* ── ENDPOINT TABLE ── */
  .endpoints { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }
  .ep {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 11px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .ep .method { font-weight: 700; font-family: monospace; font-size: 10px; padding: 2px 6px; border-radius: 4px; }
  .ep .path   { font-family: monospace; color: #58a6ff; }
  .ep .desc   { color: #8b949e; }
  .post { background: #2ecc7133; color: #2ecc71; }
  .get  { background: #58a6ff33; color: #58a6ff; }

  /* horizontal connector line */
  .hline { width: 20px; height: 2px; background: linear-gradient(90deg, #58a6ff55, #58a6ff); flex-shrink: 0; }
  .hline-kg     { background: linear-gradient(90deg, #9b59b655, #9b59b6); }
  .hline-chroma { background: linear-gradient(90deg, #2ecc7155, #2ecc71); }
  .hline-llm    { background: linear-gradient(90deg, #e67e2255, #e67e22); }
</style>
</head>
<body>
<div class="page">

  <h1>⚡ MATCHMAKER API — SYSTEM ARCHITECTURE</h1>
  <p class="subtitle">Knowledge Graph + ChromaDB + Mistral-7B · Three-Stage AI Matching Pipeline</p>

  <div class="flow">

    <!-- ── CLIENT ── -->
    <div class="section-label">Input</div>
    <div class="row">
      <div class="box box-client" style="min-width:200px">
        <span class="tag">CLIENT</span>
        <span class="icon">🖥️</span>
        <div class="title">Swagger UI / HTTP Client</div>
        <div class="sub">POST /match · POST /recommend</div>
      </div>
    </div>

    <div class="arrow-down">
      <svg width="24" height="40" viewBox="0 0 24 40"><line x1="12" y1="0" x2="12" y2="32" stroke="#58a6ff" stroke-width="2"/><polygon points="5,28 12,40 19,28" fill="#58a6ff"/></svg>
    </div>

    <!-- ── FASTAPI ── -->
    <div class="row">
      <div class="box box-api" style="min-width:340px; padding: 16px 24px;">
        <span class="tag">FASTAPI + UVICORN</span>
        <span class="icon">🚀</span>
        <div class="title">main.py — REST API Gateway</div>
        <div class="sub">Async · CORS · Request timing middleware · Lifespan warm-up</div>
        <div style="margin-top:12px">
          <div class="endpoints">
            <div class="ep"><span class="method post">POST</span><span class="path">/match</span></div>
            <div class="ep"><span class="method post">POST</span><span class="path">/recommend</span></div>
            <div class="ep"><span class="method get">GET</span><span class="path">/health</span></div>
            <div class="ep"><span class="method get">GET</span><span class="path">/ready</span></div>
          </div>
        </div>
      </div>
    </div>

    <div class="arrow-down">
      <svg width="24" height="40" viewBox="0 0 24 40"><line x1="12" y1="0" x2="12" y2="32" stroke="#58a6ff" stroke-width="2"/><polygon points="5,28 12,40 19,28" fill="#58a6ff"/></svg>
    </div>

    <!-- ── SCHEMAS ── -->
    <div class="row">
      <div class="box box-client" style="min-width:340px;">
        <span class="tag">PYDANTIC V2 · schemas.py</span>
        <span class="icon">🛡️</span>
        <div class="title">Request Validation & Parsing</div>
        <div class="sub">UserProfile · UserObjective · NetworkProfile · MatchResult</div>
      </div>
    </div>

    <div class="arrow-down">
      <svg width="24" height="40" viewBox="0 0 24 40"><line x1="12" y1="0" x2="12" y2="32" stroke="#58a6ff" stroke-width="2"/><polygon points="5,28 12,40 19,28" fill="#58a6ff"/></svg>
    </div>

    <!-- ── MATCHER ORCHESTRATOR ── -->
    <div class="row">
      <div class="box box-api" style="min-width:340px; padding: 16px 24px;">
        <span class="tag">ORCHESTRATOR · matcher.py</span>
        <span class="icon">🎯</span>
        <div class="title">3-Stage Pipeline Runner</div>
        <div class="sub">asyncio.gather · parallel LLM calls · score blending · sort & return</div>
      </div>
    </div>

    <div class="arrow-down">
      <svg width="24" height="44" viewBox="0 0 24 44"><line x1="12" y1="0" x2="12" y2="36" stroke="#58a6ff" stroke-width="2"/><polygon points="5,32 12,44 19,32" fill="#58a6ff"/></svg>
    </div>

    <!-- ── 3 STAGES ── -->
    <div class="section-label">Three-Stage AI Pipeline</div>
    <div class="stages">

      <!-- STAGE 1 -->
      <div class="stage-wrap">
        <div class="box box-kg" style="width:280px; padding: 18px 16px;">
          <span class="tag">STAGE 1</span>
          <span class="icon">🕸️</span>
          <div class="title">Knowledge Graph</div>
          <div class="sub" style="margin-bottom:10px">knowledge_graph.py · NetworkX DiGraph</div>
          <div style="text-align:left; font-size:11px; color:#c39bd3; line-height:1.8">
            <div>🔵 Nodes: USER, CANDIDATE</div>
            <div>🟣 Nodes: SKILL, TITLE, GOAL</div>
            <div>🟠 Nodes: INDUSTRY</div>
            <div style="margin-top:6px; color:#8b949e">Edges:</div>
            <div>→ HAS_SKILL · SEEKS_TITLE</div>
            <div>→ HAS_GOAL · IN_INDUSTRY</div>
            <div style="margin-top:8px; color:#c39bd3">Score weights:</div>
            <div>Shared Skill: +15 pts</div>
            <div>Title Match: +20 pts</div>
            <div>Goal Signal: +10 pts</div>
          </div>
        </div>
        <div style="font-size:10px; color:#9b59b6; margin-top:6px; font-weight:600">STRUCTURAL SCORE (0–100)</div>
      </div>

      <!-- arrow -->
      <div style="display:flex; align-items:center; padding: 0 8px; margin-top: -20px;">
        <svg width="32" height="24" viewBox="0 0 32 24"><line x1="0" y1="12" x2="24" y2="12" stroke="#58a6ff" stroke-width="2"/><polygon points="20,5 32,12 20,19" fill="#58a6ff"/></svg>
      </div>

      <!-- STAGE 2 -->
      <div class="stage-wrap">
        <div class="box box-chroma" style="width:280px; padding: 18px 16px;">
          <span class="tag">STAGE 2</span>
          <span class="icon">🔍</span>
          <div class="title">Semantic Retrieval</div>
          <div class="sub" style="margin-bottom:10px">vector_store.py · ChromaDB</div>
          <div style="text-align:left; font-size:11px; color:#82e0aa; line-height:1.8">
            <div>📦 EphemeralClient (stateless)</div>
            <div>🧠 Embedding model:</div>
            <div style="padding-left:12px">all-MiniLM-L6-v2</div>
            <div>📐 Distance: Cosine similarity</div>
            <div>🔎 HNSW index per request</div>
            <div style="margin-top:8px; color:#82e0aa">Understands:</div>
            <div>CISO = "cybersecurity leader"</div>
            <div>Synonyms &amp; context</div>
          </div>
        </div>
        <div style="font-size:10px; color:#2ecc71; margin-top:6px; font-weight:600">SEMANTIC SCORE (0–100) + RANK</div>
      </div>

      <!-- arrow -->
      <div style="display:flex; align-items:center; padding: 0 8px; margin-top: -20px;">
        <svg width="32" height="24" viewBox="0 0 32 24"><line x1="0" y1="12" x2="24" y2="12" stroke="#58a6ff" stroke-width="2"/><polygon points="20,5 32,12 20,19" fill="#58a6ff"/></svg>
      </div>

      <!-- STAGE 3 -->
      <div class="stage-wrap">
        <div class="box box-llm" style="width:280px; padding: 18px 16px;">
          <span class="tag">STAGE 3</span>
          <span class="icon">🤖</span>
          <div class="title">LLM Reasoning</div>
          <div class="sub" style="margin-bottom:10px">llm_reasoner.py · Mistral-7B</div>
          <div style="text-align:left; font-size:11px; color:#f0b27a; line-height:1.8">
            <div>🔄 Backend priority:</div>
            <div style="padding-left:12px">1. Ollama (local)</div>
            <div style="padding-left:12px">2. HuggingFace API</div>
            <div style="padding-left:12px">3. Deterministic fallback</div>
            <div style="margin-top:8px">📝 Input to Mistral:</div>
            <div>Candidate profile</div>
            <div>KG signals + both scores</div>
            <div style="margin-top:6px">💬 Output: 1 sentence</div>
            <div>natural-language reason</div>
          </div>
        </div>
        <div style="font-size:10px; color:#e67e22; margin-top:6px; font-weight:600">NATURAL LANGUAGE REASON</div>
      </div>

    </div>

    <!-- ── OLLAMA SIDECAR ── -->
    <div style="display:flex; justify-content:flex-end; margin-top: -160px; margin-bottom: 80px; padding-right: 40px;">
      <div style="display:flex; flex-direction:column; align-items:center;">
        <div class="box box-ollama" style="min-width:160px;">
          <span class="tag">LOCAL LLM SERVER</span>
          <span class="icon">🦙</span>
          <div class="title">Ollama</div>
          <div class="sub">mistral:latest</div>
          <div class="sub">localhost:11434</div>
        </div>
        <div style="font-size:10px; color:#e74c3c; margin-top:4px;">~4GB · CPU/GPU</div>
      </div>
    </div>

    <!-- ── SCORE FORMULA ── -->
    <div class="arrow-down">
      <svg width="24" height="40" viewBox="0 0 24 40"><line x1="12" y1="0" x2="12" y2="32" stroke="#f1c40f" stroke-width="2"/><polygon points="5,28 12,40 19,28" fill="#f1c40f"/></svg>
    </div>

    <div class="row">
      <div class="formula-box">
        <div class="label">Score Blending Formula</div>
        <div class="formula">final_score = 0.45 × KG + 0.55 × ChromaDB</div>
        <div class="weights">
          <div class="w">KG weight: <span>45%</span> (structural)</div>
          <div class="w">ChromaDB weight: <span>55%</span> (semantic)</div>
        </div>
      </div>
    </div>

    <div class="arrow-down">
      <svg width="24" height="40" viewBox="0 0 24 40"><line x1="12" y1="0" x2="12" y2="32" stroke="#f1c40f" stroke-width="2"/><polygon points="5,28 12,40 19,28" fill="#f1c40f"/></svg>
    </div>

    <!-- ── OUTPUT FIELDS ── -->
    <div class="section-label">Output Per Candidate</div>
    <div class="row">
      <div class="outputs">
        <div class="output-chip chip-score">📊 score<br/><span style="font-size:10px;font-weight:400">0–100 blended</span></div>
        <div class="output-chip chip-reason">💬 reason<br/><span style="font-size:10px;font-weight:400">Mistral sentence</span></div>
        <div class="output-chip chip-signals">🕸️ kg_signals<br/><span style="font-size:10px;font-weight:400">named graph edges</span></div>
        <div class="output-chip chip-rank">🔢 retrieval_rank<br/><span style="font-size:10px;font-weight:400">ChromaDB position</span></div>
      </div>
    </div>

    <div class="arrow-down">
      <svg width="24" height="40" viewBox="0 0 24 40"><line x1="12" y1="0" x2="12" y2="32" stroke="#f1c40f" stroke-width="2"/><polygon points="5,28 12,40 19,28" fill="#f1c40f"/></svg>
    </div>

    <!-- ── RECOMMENDATION TIERS ── -->
    <div class="section-label">Recommendation Tiers · /recommend endpoint</div>
    <div class="row">
      <div class="tiers">
        <div class="tier tier-hot">
          🔥 HOT LEADS
          <div class="tscore">score ≥ 70 · Reach out now</div>
        </div>
        <div class="tier tier-warm">
          ✅ WARM LEADS
          <div class="tscore">score 40–70 · Worth connecting</div>
        </div>
        <div class="tier tier-cold">
          ❄️ WEAK MATCHES
          <div class="tscore">score &lt; 40 · Skip for now</div>
        </div>
      </div>
    </div>

    <!-- ── DEPLOYMENT ── -->
    <hr class="divider"/>
    <div class="section-label">Deployment</div>
    <div class="row" style="gap:16px; flex-wrap:wrap;">

      <div class="box box-client" style="min-width:160px;">
        <span class="tag">LOCAL</span>
        <span class="icon">💻</span>
        <div class="title">Python 3.11 + venv</div>
        <div class="sub">python main.py</div>
      </div>

      <div class="box box-api" style="min-width:180px;">
        <span class="tag">DOCKER</span>
        <span class="icon">🐳</span>
        <div class="title">Docker Compose</div>
        <div class="sub">API + Ollama sidecar</div>
        <div class="sub">Auto model pull</div>
      </div>

      <div class="box box-chroma" style="min-width:180px;">
        <span class="tag">KUBERNETES</span>
        <span class="icon">☸️</span>
        <div class="title">k8s.yaml</div>
        <div class="sub">HPA: 2→10 pods</div>
        <div class="sub">GPU-ready StatefulSet</div>
      </div>

    </div>

  </div>

  <!-- ── LEGEND ── -->
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#9b59b6"></div>Knowledge Graph (NetworkX)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#2ecc71"></div>ChromaDB + Embeddings</div>
    <div class="legend-item"><div class="legend-dot" style="background:#e67e22"></div>Mistral-7B (Ollama)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#58a6ff"></div>FastAPI Layer</div>
    <div class="legend-item"><div class="legend-dot" style="background:#f1c40f"></div>Output / Scoring</div>
  </div>

</div>
</body>
</html>
