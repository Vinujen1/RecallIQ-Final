# RecallIQ

**Find the official Canadian recall notice that matches the product you own.**
ARTE 4110U Deep Learning Final Project (Group 5).

A consumer describes a product in plain words ("Bell youth bike helmet", "peanut butter ice
cream salmonella", "heater model XJ-500"). RecallIQ ranks Government of Canada recall notices
with four interchangeable retrieval models. It compares any UPC, model, lot or best-before
code exactly against each notice, and can write a short summary grounded only in the retrieved
notices. Every result links to the official notice on `recalls-rappels.canada.ca`.

> RecallIQ points to official notices. It never decides that a product is recalled or safe:
> a match is not confirmation, and no match does not mean a product is safe.

## Features

| Feature | What it does | Where |
|---|---|---|
| **Four retrieval models** | BM25, pretrained MiniLM (Model A), contrastively fine-tuned MiniLM (Model B), and Model B trained with hard negatives (Model C), selectable per search | `ml/retrieval/`, `POST /search` |
| **Deterministic identifier checks** | Extracts UPC / model / lot / best-before codes from the query and each notice and compares them **exactly** (MATCH / MISMATCH / UNKNOWN). Embeddings treat "XJ-500" and "XJ-600" as near-identical; recalls do not | `ml/verification/identifiers.py` |
| **GPT-powered RAG summary** | GPT-4o-mini summarises the top notices; the answer is shown only if it passes deterministic grounding checks, otherwise a fixed template answer is shown with the reason | `ml/rag/`, `POST /explain` |
| **Search filters** | Domain (food / consumer product) and official category, applied before ranking | `POST /search` |
| **Model Lab** | Training curves, proxy metrics with bootstrap confidence intervals, and a live four-model comparison | `/model-lab` |
| **Leakage-aware evaluation** | Grouped 60/20/20 split frozen after 81 human pair decisions; annotation tooling; Recall@5, MRR@10, nDCG@10 | `ml/evaluation/` |

## Retrieval models

All models search the same fixed index of **1,000 Government of Canada recall notices**
(a seeded, stratified sample of 10,586 eligible food and consumer-product recalls).

| API name | Model | How it ranks |
|---|---|---|
| `bm25` | BM25 baseline | Okapi BM25 over title + product + issue + category (shared keywords) |
| `minilm` | **Model A**: `sentence-transformers/all-MiniLM-L6-v2` (pinned revision) | Cosine similarity of mean-pooled 384-d embeddings |
| `minilm_finetuned` | **Model B**: MiniLM fine-tuned with a contrastive loss | Same as A, with weights trained on recall query pairs |
| `minilm_hard_negative` | **Model C**: Model B's recipe plus one mined hard negative per query | Same as B; trained to separate look-alike notices |

BM25 scores and cosine similarities are on different scales. The API labels each response
with a `score_type`, and the UI never shows scores as probabilities.

## Deep learning methodology

- **Training data.** (query, recall) pairs generated deterministically from training-split
  notices only (brand/product/issue templates and consumer-style questions). Validation
  pairs come only from validation-split notices. Leakage checks refuse overlapping groups.
  See [`docs/TRAINING.md`](docs/TRAINING.md).
- **Loss.** MultipleNegativesRanking (InfoNCE): for a batch of B queries, the logits are
  `scale · cos(q_i, c_j)` over the B positives (plus M hard negatives for Model C). Cross-entropy
  pushes each query towards its own positive and away from every other candidate. Candidates
  from the query's own product group are masked, so near-duplicate notices are not treated as
  negatives.
- **Hard negatives (Model C).** Training-split notices a person judged `DISTINCT` come first
  (123 training pairs start with one). Further look-alikes are mined from BM25 results and the
  positive's nearest MiniLM neighbours. Anything that might be the same product (same group,
  brand group, or product-key Jaccard ≥ 0.5) is excluded, so a re-notice is never pushed away.
  One of the top 3 is sampled each step.
- **Model selection.** Checkpoints are selected on validation (proxy MRR@10) every epoch. The
  test split stays sealed and is evaluated once, at the end.
- **Reproducibility.** Seeds, configurations, git commit, library versions and per-epoch
  history are stored with each run (`experiments/runs/`). Figures and tables are generated
  from those files (`python -m reports.figures`), never written by hand.

## Results so far (proxy evaluation; not human judgments)

The numbers below come from **391 generated validation queries** (deterministic templates over
validation notices) on the 1,000-notice index. Source: [`experiments/reports/tables.md`](experiments/reports/tables.md).

| System | Recall@5 | MRR@10 | nDCG@10 | False match @1 |
|---|---:|---:|---:|---:|
| BM25 | 0.9680 | 0.9702 | 0.9627 | 4.9 % |
| Model A (pretrained MiniLM) | 0.9463 | 0.9367 | 0.9305 | 9.0 % |
| Model B (fine-tuned) | 0.9706 | 0.9743 | 0.9640 | 4.1 % |
| Model C (hard negatives) | 0.9719 | 0.9769 | 0.9670 | 3.6 % |

**What this supports.** Fine-tuning clearly improves the pretrained model. B − A ΔMRR@10 is
+0.038, with a 95 % paired bootstrap CI of [+0.022, +0.056].

**What it does not support (yet):**
- **C over B.** C is only marginally better: +0.0023 MRR@10 averaged over 3 training seeds
  (sd 0.0013, [`seeds.md`](experiments/reports/seeds.md)), on a proxy close to its ceiling.
- **Fine-tuned models over BM25.** The difference is within noise.
- **Real consumer queries.** Generated queries share wording with the notices, so they favour
  lexical matching and overstate accuracy on real consumer language.

Human-written evaluation queries are being collected (`data/annotations/`). Results from
single-annotator queries are reported separately from independently reviewed ones. The
held-out test evaluation has **not** been run. Learning curves:
`experiments/reports/loss_model_b.svg`, `loss_model_c.svg`, `validation_loss_compare.svg`.

## RAG with grounding safeguards

`POST /explain` retrieves notices with the selected model, applies the identifier checks, and
selects up to 3 notices as evidence. It then asks GPT-4o-mini (configurable) for a short
consumer-facing summary in a strict JSON format with cited recall IDs. The answer is shown only
if **all** deterministic guards pass:

- every code, number, recall ID, capitalised name and quoted phrase appears in the evidence;
- no links except the official evidence URLs (citations are attached by the server, never
  written by the model);
- no claim that a product is safe or not recalled, except as a hedge ("does not mean it is
  safe");
- no claim that a code matches unless the deterministic verifier found a MATCH;
- no echoes of prompt-injection text. The query and notices are sent as JSON data, never as
  instructions.

With no results, or only notices whose codes conflict, the model is not called. Any API error,
timeout, refusal or rejected answer falls back to a deterministic template (still HTTP 200)
with a `fallback_reason`. Latency and token usage are returned and logged. Search works fully
without OpenAI. Details: [`docs/RAG.md`](docs/RAG.md).

## Architecture

```
Browser ──► frontend/ (Next.js 16, :3000) ──HTTP/JSON──► backend/ (FastAPI, :8000)
                                                            │  read-only SQLite + model artifacts
data/raw/HCRSAMOpenData.csv ──► ml/recall_data ──► data/processed/recalliq.db
                                ml/training ──► ml/artifacts/models/*   (checkpoints)
                                ml/retrieval ──► ml/artifacts/embeddings/* (per-model indexes)
query ─► {BM25 | A | B | C} ─► top-k ─► identifier checks ─► [/explain: evidence ─► GPT ─► guards]
```

```
RecallIQ/
├── backend/            FastAPI service: config, routers (health, recalls, search/explain), schemas
├── frontend/           Next.js App Router UI: /, /search, /recalls/[id], /model-lab, /how-it-works
├── ml/                 Offline Python package (installed editable; also imported by the API)
│   ├── recall_data/    CSV → cleaned, grouped, sampled corpus → SQLite
│   ├── retrieval/      BM25, MiniLM encoder, embedding indexes, SearchEngine
│   ├── training/       Pair generation, hard-negative mining, contrastive trainer
│   ├── evaluation/     Splits, split review, annotation tools, metrics, proxy evaluation
│   ├── verification/   Deterministic UPC / model / lot / best-before matching
│   ├── rag/            Evidence selection, OpenAI adapter, grounding guards
│   ├── reports/        Generates experiments/reports from run files
│   └── configs/        Training configurations (Models B and C, seeds 4110–4112)
├── data/               raw/ + processed/ (git-ignored), annotations/ (templates), reviews/ (split review evidence)
├── experiments/        Run configs + histories (runs/) and generated tables/figures (reports/)
├── docs/               Detailed documentation (see below)
└── tests/              pytest suites for backend, ml, evaluation, training, verification, rag, reports
```

## Setup

Requirements: Python 3.12, Node.js 20+ (developed on 26.x), Git. The commands below use Git
Bash on Windows. In PowerShell, activate with `.venv\Scripts\Activate.ps1`.

### 1. Python environment

```bash
cd backend
python -m venv .venv
source .venv/Scripts/activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt -r ../ml/requirements.txt
pip install -e ../ml                 # makes recall_data, retrieval, training, rag, ... importable
cp .env.example .env                 # then edit; never commit .env
```

`ml/requirements.txt` installs PyTorch (CPU build on Windows; a large download).

### 2. Data and model artifacts

None of these are committed (they are large or licensed separately). The API degrades
gracefully: BM25 needs only the database, and each neural model is disabled (HTTP 503 for that
model only) until its files exist.

| Artifact | Path | How to produce |
|---|---|---|
| Official CSV | `data/raw/HCRSAMOpenData.csv` | Download *Recalls and Safety Alerts* open data |
| Database (BM25) | `data/processed/recalliq.db` | `cd ml && python -m recall_data.build` (seed 4110, 1,000 notices) |
| Model A index | `ml/artifacts/embeddings/all-MiniLM-L6-v2/` | `python -m retrieval.embed` (downloads ~92 MB once) |
| Model B / C checkpoints | `ml/artifacts/models/model-{b,c}-v1/checkpoints/best/` | `python -m training.train --config configs/model_b.json --run-name model-b-v1` (and `model_c`) |
| Model B / C indexes | `ml/artifacts/embeddings/model-{b,c}-v1/` | `python -m retrieval.finetuned --label minilm_finetuned --checkpoint artifacts/models/model-b-v1/checkpoints/best --out artifacts/embeddings/model-b-v1` (and `minilm_hard_negative` / model-c) |

Each index records the checkpoint's weight hash and the corpus fingerprint. A stale or
mismatched index is refused, not silently used. All paths can be overridden in
`backend/.env`. The full reproduction sequence is in [`docs/CHECKPOINT.md`](docs/CHECKPOINT.md).

### 3. Run the backend

```bash
cd backend
uvicorn app.main:app --reload --port 8000      # docs at http://localhost:8000/docs
python -m app.llm_check                        # optional: verify the OpenAI key and model
```

AI summaries need `RECALLIQ_LLM_PROVIDER=openai` and `OPENAI_API_KEY` in `backend/.env`. Never
put them in the frontend. Without them, `/explain` uses the deterministic template.

### 4. Run the frontend

```bash
cd frontend
npm install
cp .env.example .env.local           # NEXT_PUBLIC_API_BASE_URL (default http://localhost:8000)
npm run dev                          # http://localhost:3000
```

The first MiniLM request loads the model (about 15–30 s on CPU).

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/recalls?limit&offset` | Newest-first page of notices |
| GET | `/recalls/{id}` | One notice by official recall ID |
| GET | `/categories` | Official categories with counts |
| GET | `/search/models` | Which of the four models are servable, and why not |
| POST | `/search` | `{"query", "model", "top_k", "domain", "category"}` → ranked notices + identifier checks |
| POST | `/explain` | Same body (`top_k` ≤ 5) → grounded summary, cited evidence, official URLs, disclaimer, generation stats |

```bash
curl -s -X POST http://127.0.0.1:8000/search -H "Content-Type: application/json" \
     -d '{"query": "heater model XJ-500 overheating", "model": "minilm_hard_negative", "top_k": 5}'
```

Full schemas: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and the interactive docs at `/docs`.
On Windows, prefer `127.0.0.1` in clients (`localhost` tries IPv6 first).

## Tests and checks

```bash
# Python (repo root, venv active). pytest.ini puts backend/ and ml/ on the import path.
python -m pytest

# Frontend
cd frontend
npx tsc --noEmit && npm run lint && npm run build
npm run test:unit                    # pure helpers
npm run test:e2e                     # Playwright, desktop + mobile; needs the backend on :8000
```

| Suite | Covers |
|---|---|
| `tests/backend/` | Every endpoint: validation, 404/422/503 degradation, filters, fine-tuned models, `/explain` with mocked OpenAI |
| `tests/ml/` | Data pipeline (loading, cleaning, grouping, sampling, SQLite) and retrieval (BM25, embeddings, stale-index detection) |
| `tests/training/` | Pair generation, leakage guards, loss, trainer, model variants |
| `tests/evaluation/` | Metrics vs hand-computed values, splits, split review, annotations, single-annotator mode, evaluation runner |
| `tests/verification/` | Identifier extraction and exact matching |
| `tests/rag/` | Template, grounding guards, hallucination/injection cases, every OpenAI failure path |
| `tests/reports/` | Multi-seed report pairing (only finished runs with matching seeds) |
| `frontend/tests/` | Helpers (unit); user journeys, API failure states, mobile layout (e2e) |

Tests never touch real data, models or OpenAI unless stated. A synthetic CSV and a
deterministic fake encoder are built in a temporary folder. An autouse fixture blocks any real
OpenAI client. Tests marked for the real database or trained checkpoints are skipped until
those artifacts exist.

## Documentation

| Document | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Components, design decisions, API schemas, data model |
| [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md) | Filtering, cleaning, grouping and sampling of the official CSV |
| [`docs/RETRIEVAL.md`](docs/RETRIEVAL.md) | BM25 and MiniLM search, score semantics, index integrity |
| [`docs/TRAINING.md`](docs/TRAINING.md) | Pair generation, contrastive loss, hard-negative mining, Models B and C |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Frozen split, annotation workflow, review modes, metrics, test-set rules |
| [`docs/QUERY_WRITING_GUIDE.md`](docs/QUERY_WRITING_GUIDE.md) | Protocol for writing and judging human evaluation queries |
| [`docs/RAG.md`](docs/RAG.md) | RAG flow, safeguards, configuration, observability, limitations |
| [`docs/CHECKPOINT.md`](docs/CHECKPOINT.md) | Evidence pack for the technical checkpoint and the full reproduction sequence |
| [`docs/PROGRESS.md`](docs/PROGRESS.md) | Phase-by-phase development log and verification history |
| [`frontend/DESIGN.md`](frontend/DESIGN.md) | Approved visual design system and trust rules for the UI |
| [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md) | Member contributions and AI-use disclosure |

## Data source and academic integrity

Government of Canada *Recalls and Safety Alerts* open data, downloaded as a CSV. RecallIQ does
not scrape government webpages; record URLs point to the official notices.

AI tool use is disclosed in [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md). Reported evaluation
results must rest on queries and relevance judgments written by people. The 29 draft
queries that exist so far were produced with AI assistance (ChatGPT), are not human-reviewed, are kept
local, and are not used in any metric (see `CONTRIBUTIONS.md`). Queries judged by a single annotator are labelled
`single_annotator`, evaluated only on request, and reported separately from independently
reviewed queries ([`docs/EVALUATION.md`](docs/EVALUATION.md) §3a).
