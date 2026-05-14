# Hospital Chatbot — MVP1

A bilingual (Thai + English) agentic RAG chatbot for **Walailuk Hospital**. Patients ask questions about hours, services, departments, insurance, appointments, and more. The bot answers from your own documents first, then falls back to a targeted web search if the local knowledge base doesn't have the answer.

Channels: **LINE** · **Facebook Messenger** · **REST API** · **CLI terminal**

---

## Tech Stack

| Layer | Tool | Why it helps |
|---|---|---|
| **LLM (routing)** | `gpt-4o-mini` | Fast, cheap — used only for classify/judge/expand calls |
| **LLM (answer)** | `gpt-4.1` | Higher quality for the user-facing reply |
| **Embeddings** | `text-embedding-3-large` | OpenAI's highest-quality embedding; larger vector space means better semantic match for medical Thai/English mixed content |
| **Vector DB** | ChromaDB (file-based) | Zero-infrastructure; persists to disk; single collection with metadata filter |
| **Reranker** | Cohere `rerank-multilingual-v3.0` | Cross-encoder reranking over the top ~15 candidates — biggest quality lift for Thai content where pure vector similarity can miss nuance |
| **Workflow** | LangGraph (CRAG) | Stateful graph with conditional edges; auto-short-circuits when RAG is sufficient |
| **Web fallback** | Tavily | Targeted web search scoped to the hospital name — activated only when local KB fails |
| **API** | FastAPI + Uvicorn | Thin async server; LINE/Facebook webhooks already wired |
| **Deploy** | Google Cloud Run | Serverless; `gcloud builds submit` handles ARM→x86 cross-compile on Apple Silicon |

---

## How the Agent Works

The heart of the system is a **CRAG (Corrective RAG)** graph built with LangGraph. Every user message passes through a fixed set of nodes; the path taken depends on what is found at each step.

### Workflow diagram

```
User message
     │
     ▼
┌─────────────┐
│  supervisor │── smalltalk ──────────────────────────────────────────► END
└──────┬──────┘── off_topic ─────────────────────────────────────────► END
       │ rag
       ▼
┌─────────────┐   multi-query expansion (3 paraphrases, Thai↔EN)
│   retrieve  │   → embed all variants → query ChromaDB (k=5 each)
└──────┬──────┘   → pool unique chunks → Cohere rerank → top 5
       ▼
┌──────────────┐       ✓ relevant
│ grade_chunks │ ──────────────────────────────────────► draft_answer ──┐
└──────┬───────┘                                                        │
       │ ✗ not relevant                                          (ENABLE_REFLECTION?)
       ▼                                                    yes ─► reflect ─► END
┌──────────────┐                                            no  ──────────► END
│  web_search  │── (Tavily, hospital-scoped) ──► answer_web ─────────────► ↑
└──────────────┘
```

### Node descriptions

| Node | Role | Model |
|---|---|---|
| `supervisor` | Routes to `rag` / `smalltalk` / `off_topic` | `ROUTER_MODEL` |
| `retrieve` | Expands query → embeds → queries single collection → Cohere rerank | `ROUTER_MODEL` for expansion |
| `grade_chunks` | One LLM call: are these chunks sufficient to answer? | `ROUTER_MODEL` |
| `draft_answer` / `answer_web` | Generates the user-facing reply from context | `ANSWER_MODEL` |
| `web_search` | Tavily search, prefixed with hospital name (Thai or English) | — |
| `reflect` | Quality-reviews the draft against 3 criteria (reliability, politeness, logic). Only active when `ENABLE_REFLECTION=true` | `ANSWER_MODEL` |
| `smalltalk` | Short friendly reply for greetings / thanks | `ANSWER_MODEL` |
| `off_topic` | Fixed bilingual response for non-hospital questions | — |

### How speed is optimized

**Model split:** The cheap `gpt-4o-mini` handles all routing, grading, and query expansion. The better `gpt-4.1` is only called for the one reply the user actually reads. This cuts the cost of each inference step without downgrading the final answer quality.

**LLM calls capped at 3–4:** The old multi-stage design ran up to ~10 LLM calls per question (answer → evaluate → try next source → answer again…). CRAG collapses this: one grade call decides whether the chunks are good; if yes, one answer call; if not, one web search then one answer call. The optional `reflect` step adds one more, but it is off by default.

**Cached multi-query expansion:** The same FAQ is asked many times by different users. `_expand_queries_cached` wraps the expansion call in `lru_cache(maxsize=500)` — once a question variant set is computed, subsequent identical questions skip the LLM call entirely.

```python
@lru_cache(maxsize=500)
def _expand_queries_cached(question: str) -> tuple[str, ...]:
    return _expand_queries_uncached(question)
```

**Large embedding + Cohere reranker pipeline:** `text-embedding-3-large` produces a higher-dimensional vector space that captures Thai/English mixed semantics more faithfully than smaller models. But vector similarity alone can be fooled by superficial lexical overlap. After pooling up to `max(k×3, 12)` candidates from all query variants, the Cohere cross-encoder reranker re-scores each candidate against the original question text — not just its embedding — and returns the top `k`. This two-stage approach keeps retrieval fast (vector search) while making the final selection accurate (cross-encoder).

```
multi-query expand (3 variants)
    → embed each variant (text-embedding-3-large)
    → query ChromaDB, deduplicate → ~12–15 unique chunks
    → Cohere rerank-multilingual-v3.0 → top 5
    → grade_chunks (1 LLM call)
```

**Hard timeout on every OpenAI call:** `OPENAI_TIMEOUT_S=60` is set on every API call. On Cloud Run, a hung OpenAI/Tavily call would hold the instance until the 120-second Cloud Run timeout, blocking all concurrent requests. The 60-second cap kills it early and returns the bilingual fallback message instead.

**Reflection is off by default:** `ENABLE_REFLECTION=false` skips the polish step — cutting one LLM call and 1–3 seconds of latency on every answer path. Enable it when quality matters more than speed.

---

## Real Conversations — CRAG Traces

These are real traces from `logs/cli/2026-05.log`. Each shows the exact path taken through the graph and which nodes were active.

---

### Trace 1 — Smalltalk (no retrieval)

**Question:** `Hello`

```
  ┌─ ROUTING  →  SMALLTALK

  ┌─ SMALLTALK  (no retrieval needed)

  ┌─ FINAL ANSWER
  │  Hello! How can I assist you today? Feel free to ask any
  │  questions about the hospital.
```

**Path:** `supervisor → smalltalk → END`
**LLM calls:** 2 (supervisor + smalltalk reply)

The supervisor classified this as a greeting and short-circuited the entire retrieval pipeline. No embedding, no ChromaDB query, no grading.

---

### Trace 2 — RAG hit (answer found in knowledge base)

**Question:** `What is hospital open time?`

```
  ┌─ ROUTING  →  RAG

  ┌─ RETRIEVE (knowledge base)  —  3 chunk(s) found
  │  ── chunk 1 (url) ──────────────────────────────────────
  │  เวลาทำการ
  │  จ – ศ เวลา 8.00-20.00 น.
  │  ส – อา และวันหยุดนักขัตฤกษ์ เวลา 8.00-16.00 น.
  │  แผนกอุบัติเหตุและฉุกเฉินเปิดบริการทุกวัน 24 ชั่วโมง
  │  ── chunk 2 (md) ───────────────────────────────────────
  │  Our ER is staffed 24/7 and accepts walk-ins…
  │  ── chunk 3 (url) ──────────────────────────────────────
  │  วันจันทร์ ถึง วันศุกร์  เวลา 8.30-20.00 น. …

  ┌─ GRADE CHUNKS  →  ✓ relevant

  ┌─ DRAFT ANSWER  (knowledge base)
  │  สวัสดีค่ะ ยินดีต้อนรับสู่โรงพยาบาลวลัยลักษณ์ …
  │
  │  - วันจันทร์ – วันศุกร์ : 8.00 – 20.00 น.
  │  - วันเสาร์, วันอาทิตย์ และวันหยุดนักขัตฤกษ์ : 8.00 – 16.00 น.
  │
  │  แผนกอุบัติเหตุและฉุกเฉิน เปิดให้บริการทุกวัน ตลอด 24 ชั่วโมง

  ┌─ REFLECTION  (✎ Revised)

  ┌─ FINAL ANSWER
  │  สวัสดีค่ะ ยินดีต้อนรับสู่โรงพยาบาลวลัยลักษณ์ พร้อมให้บริการคุณค่ะ 😊
  │
  │  เวลาทำการของโรงพยาบาลวลัยลักษณ์ มีดังนี้ค่ะ
  │
  │  - วันจันทร์ – วันศุกร์ : 8.00 – 20.00 น.
  │  - วันเสาร์, วันอาทิตย์ และวันหยุดนักขัตฤกษ์ : 8.00 – 16.00 น.
  │
  │  แผนกอุบัติเหตุและฉุกเฉิน เปิดให้บริการทุกวัน ตลอด 24 ชั่วโมงค่ะ
  │  หากมีข้อสงสัยเพิ่มเติม สามารถติดต่อได้ที่ 0 7547 9999
```

**Path:** `supervisor → retrieve → grade_chunks (✓) → draft_answer → reflect → END`
**LLM calls:** 4 (supervisor + expansion + grade + answer + reflect)

The question was English but the knowledge base had Thai content. Multi-query expansion generated Thai paraphrases of the question, which matched the Thai URL chunks. `grade_chunks` saw useful hours data and passed them straight through. `reflect` lightly revised the draft for warmth.

---

### Trace 3 — Web fallback (not in knowledge base)

**Question:** `มีหมอฟันนทั้งหมดกี่คน` *(How many dentists are there?)*

```
  ┌─ ROUTING  →  RAG

  ┌─ RETRIEVE (knowledge base)  —  3 chunk(s) found
  │  ── chunk 1 (md) ── FAQ placeholder (emergency/visitor/parking)
  │  ── chunk 2 (url) ── hospital contact info + hours
  │  ── chunk 3 (url) ── department hours / vaccine schedule

  ┌─ GRADE CHUNKS  →  ✗ not relevant — trying web search

  ┌─ WEB SEARCH  (Tavily fallback)
  │  query   : รพ วลัยลักษณ์  มีหมอฟันนทั้งหมดกี่คน
  │  results : 3 snippet(s)

  ┌─ DRAFT ANSWER  (web search)
  │  คลินิกทันตกรรมมีทีมทันตแพทย์ผู้เชี่ยวชาญมากกว่า 7 ท่านค่ะ
  │  หากต้องการข้อมูลเพิ่มเติมติดต่อ 0 7547 9999 กด 0

  ┌─ REFLECTION  (✎ Revised)

  ┌─ FINAL ANSWER
  │  ทางคลินิกทันตกรรมของโรงพยาบาลฯ มีทันตแพทย์ผู้เชี่ยวชาญมากกว่า 7 ท่านค่ะ
  │  สามารถสอบถามเพิ่มเติมได้ที่เบอร์ 0 7547 9999 กด 0 ในวันและเวลาราชการค่ะ
```

**Path:** `supervisor → retrieve → grade_chunks (✗) → web_search → answer_web → reflect → END`
**LLM calls:** 4 (supervisor + expansion + grade + answer + reflect)

Three chunks were retrieved but none contained dentist headcount. `grade_chunks` correctly rejected them. Tavily was queried with the Thai prefix `รพ วลัยลักษณ์` automatically added because the question language was Thai. The web result returned the dentist count and the system composed a polished reply.

---

## Getting Started

### Prerequisites

- Python 3.11
- `OPENAI_API_KEY` (required)
- `TAVILY_API_KEY` (for web fallback; omit to skip)
- `COHERE_API_KEY` (for reranker; omit to use vector-only retrieval)

### Install and run locally

```bash
# 1. Configure env
cp .env.example .env
#    Edit .env — at minimum set OPENAI_API_KEY

# 2. Create virtualenv
python3.11 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your hospital documents
#    - PDF / MD / TXT files → data/raw/
#    - Web pages (one URL per line) → data/urls.txt

# 5. Build the vector index
python main.py ingest

# 6. Try it in the terminal
python main.py chat

# 7. Start the API server
python main.py serve
#    → http://localhost:8000/health
#    → POST http://localhost:8000/chat   {"message": "What are OPD hours?"}
```

### Connect LINE / Facebook locally

```bash
# Expose local server via ngrok
ngrok http 8000

# Paste the ngrok HTTPS URL into developer consoles:
#   LINE:     https://<ngrok>/webhook/line
#   Facebook: https://<ngrok>/webhook/facebook
```

### Docker (local)

```bash
docker build -t hospital-chatbot .
docker run -p 8000:8000 --env-file .env hospital-chatbot
```

---

## Google Cloud Run — Build and Deploy

### One-time setup

```bash
# Install gcloud CLI: https://cloud.google.com/sdk/docs/install
gcloud auth login
gcloud config set project phupha-chatbot

# Enable APIs
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

# Create Artifact Registry repo
gcloud artifacts repositories create chatbot-repo \
  --repository-format=docker \
  --location=asia-southeast1
```

### Environment variables for Cloud Run

Copy your `.env` values into `.env.yaml` (YAML format — never commit this file):

```yaml
# .env.yaml
OPENAI_API_KEY: "sk-..."

ROUTER_MODEL: "gpt-4o-mini"
ANSWER_MODEL: "gpt-4.1"

EMBED_MODEL: "text-embedding-3-large"
OPENAI_TIMEOUT_S: "60"

RERANKER: "cohere"           # remove line to disable
COHERE_API_KEY: "..."
COHERE_RERANK_MODEL: "rerank-multilingual-v3.0"

TAVILY_API_KEY: "tvly-..."

CHAT_API_KEY: "set-a-long-random-string"

LINE_CHANNEL_SECRET: "..."
LINE_CHANNEL_ACCESS_TOKEN: "..."
FB_PAGE_ACCESS_TOKEN: "..."
FB_VERIFY_TOKEN: "..."

ENABLE_REFLECTION: "false"
```

### Ingest before deploying

The vector index (`data/chroma/`) is built locally and baked into the Docker image. Run ingest first so the image includes the latest knowledge base:

```bash
source .venv/bin/activate
python main.py ingest
```

### Build image on Google Cloud

`gcloud builds submit` builds on Google's servers — no ARM/x86 issues on Apple Silicon:

```bash
gcloud builds submit \
  --tag asia-southeast1-docker.pkg.dev/phupha-chatbot/chatbot-repo/hospital-chatbot \
  --project phupha-chatbot
```

### Deploy to Cloud Run

```bash
gcloud run deploy hospital-chatbot \
  --image asia-southeast1-docker.pkg.dev/phupha-chatbot/chatbot-repo/hospital-chatbot \
  --platform managed \
  --region asia-southeast1 \
  --min-instances 1 \
  --max-instances 3 \
  --memory 1Gi \
  --cpu 1 \
  --timeout 120 \
  --port 8000 \
  --allow-unauthenticated \
  --project phupha-chatbot \
  --env-vars-file .env.yaml
```

> `--memory 1Gi` — ChromaDB + embeddings + LangGraph state + the OpenAI client all share Cloud Run's RAM-backed tmpfs. 512Mi OOMs on the first concurrent request.  
> `--timeout 120` — kills any stuck OpenAI/Tavily call so it doesn't hold the instance.

Cloud Run prints the service URL on completion:
```
Service URL: https://hospital-chatbot-xxxx-as.a.run.app
```

### Verify deployment

```bash
curl https://hospital-chatbot-xxxx-as.a.run.app/health
# {"ok":true}

curl -X POST https://hospital-chatbot-xxxx-as.a.run.app/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <CHAT_API_KEY>" \
  -d '{"message": "What are the OPD hours?"}'
```

### Webhook URLs (paste into developer consoles)

| Platform | Webhook URL |
|---|---|
| LINE | `https://<service-url>/webhook/line` |
| Facebook | `https://<service-url>/webhook/facebook` |

### Re-deploy cheatsheet

| What changed | Steps |
|---|---|
| Documents (`data/raw/`, `data/urls.txt`) | `python main.py ingest` → build → deploy |
| Code (`app/`, `server.py`) | build → deploy |
| Env vars only (`.env.yaml`) | deploy only (skip build) |

**Env-only update (fastest):**
```bash
gcloud run deploy hospital-chatbot \
  --image asia-southeast1-docker.pkg.dev/phupha-chatbot/chatbot-repo/hospital-chatbot \
  --platform managed \
  --region asia-southeast1 \
  --project phupha-chatbot \
  --env-vars-file .env.yaml
```

### View live logs

```bash
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=hospital-chatbot" \
  --project phupha-chatbot \
  --limit 50 \
  --format "value(textPayload)"
```

---

## Configuration Reference

| Env var | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | — | **required** |
| `ROUTER_MODEL` | `gpt-4o-mini` | supervisor, grade_chunks, query expansion — keep cheap |
| `ANSWER_MODEL` | `gpt-4.1` | user-facing answer + reflect — upgrade here for quality |
| `EMBED_MODEL` | `text-embedding-3-large` | changing this requires re-ingest |
| `OPENAI_TIMEOUT_S` | `60` | hard cap on every OpenAI/embedding call |
| `VECTOR_DB` | `chroma` | file-based; swap to `qdrant` for multi-instance |
| `CHROMA_DIR` | `<repo>/data/chroma` | auto-resolved; override only for external mounts |
| `RERANKER` | `` (off) | set to `cohere` to enable |
| `COHERE_API_KEY` | — | required when `RERANKER=cohere` |
| `COHERE_RERANK_MODEL` | `rerank-multilingual-v3.0` | works for Thai + English |
| `TAVILY_API_KEY` | — | omit to disable web-search fallback |
| `ENABLE_REFLECTION` | `false` | `true` adds one more LLM call to polish every answer |
| `CHAT_API_KEY` | — | when set, `/chat` requires header `X-API-Key: <value>` |
| `LOGS_DIR` | `<repo>/logs` | pretty per-user trace files |
| `LINE_CHANNEL_SECRET` / `LINE_CHANNEL_ACCESS_TOKEN` | — | LINE webhook |
| `FB_PAGE_ACCESS_TOKEN` / `FB_VERIFY_TOKEN` | — | Facebook Messenger webhook |

---

## API Reference

```http
POST /chat
Content-Type: application/json
X-API-Key: <CHAT_API_KEY>          ← required only when CHAT_API_KEY is set

{"message": "What are the OPD hours?"}
```

```json
{"answer": "OPD is open Mon–Fri 08:00–20:00 and Sat–Sun 08:00–16:00."}
```

```http
GET /health
→ {"ok": true}
```

Webhooks for LINE and Facebook are wired in `server.py`. Set credentials in `.env` and point the developer consoles at your service URL.

---

## Project Structure

```
.
├── main.py               CLI entry: ingest / chat / serve
├── server.py             FastAPI + LINE/Facebook webhooks
├── requirements.txt
├── Dockerfile
├── .env.example          env template
├── app/
│   ├── config.py         all env vars, repo-rooted absolute paths
│   ├── rag.py            ingest, embed, multi-query retrieve, Cohere rerank
│   ├── agents.py         supervisor, grade_chunks, draft_answer, reflect,
│   │                     web_search, smalltalk, off_topic, no_data
│   ├── graph.py          LangGraph CRAG graph + ask()
│   └── tracer.py         per-user pretty log file + JSON stdout for Cloud Logging
├── data/
│   ├── raw/              your documents (PDF / MD / TXT) — replace sample
│   ├── urls.txt          one URL per line; text extracted via trafilatura
│   └── chroma/           vector index (auto-created; gitignored)
│       └── hospital-kb   single collection; source_type metadata = "md" | "url"
└── logs/
    └── <user_id>/
        └── YYYY-MM.log   pretty conversation trace (also mirrored as JSON to Cloud Logging)
```
