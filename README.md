# RecallIQ

**Canadian product recall intelligence platform** — ARTE 4110U Deep Learning Final Project.

RecallIQ will ingest public Canadian product recall data, apply deep-learning models to
classify and analyse recalls, and present the results through a web interface.

> **Status:** Phase 1B (Canadian recall dataset integration) — see
> [`docs/PROGRESS.md`](docs/PROGRESS.md). The API serves 1,000 real Government of Canada
> recall records from SQLite. No ML models, RAG, or application pages exist yet.

## Architecture

```
RecallIQ/
├── frontend/   Next.js (App Router) + TypeScript + Tailwind CSS + ESLint
├── backend/    FastAPI (Python) — read-only REST API over the recall SQLite database
├── ml/         recall_data/ pandas pipeline (Phase 1B); model training later
├── data/       raw/ official CSV, processed/ JSONL + SQLite (contents git-ignored)
├── docs/       Architecture notes and phase progress log
└── tests/      Automated tests (pytest for backend; more later)
```

```
Browser ──► frontend (Next.js :3000) ──HTTP/JSON──► backend (FastAPI :8000)
                                                        │ read-only
data/raw/HCRSAMOpenData.csv ──► ml/recall_data ──► data/processed/recalliq.db
```

More detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and
[`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md) (filtering, sampling, product families).

## Prerequisites

| Tool    | Version used in development |
| ------- | --------------------------- |
| Node.js | 26.x (npm 11.x)             |
| Python  | 3.12                        |
| Git     | 2.47                        |
| Docker  | Installed, **not required** yet |

## Getting started

Commands below are for Git Bash on Windows. In PowerShell, replace
`source .venv/Scripts/activate` with `.venv\Scripts\Activate.ps1`.

### 1. Python environment (backend + data pipeline)

```bash
cd backend
python -m venv .venv
source .venv/Scripts/activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt -r ../ml/requirements.txt
cp .env.example .env                   # optional; defaults work for local dev
```

### 2. Build the recall dataset

Place the official Government of Canada *Recalls and Safety Alerts* CSV at
`data/raw/HCRSAMOpenData.csv` (it is git-ignored and never modified), then:

```bash
cd ml
python -m recall_data.build            # --size 1000 --seed 4110 are the defaults
```

Outputs in `data/processed/`: `recalls_eligible.jsonl`, `recalls_sample.jsonl`,
`recalliq.db`, `manifest.json`. Re-running with the same CSV and seed gives identical
output. Stop the API first on Windows (an open database file cannot be replaced).

### 3. Backend (FastAPI)

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

| Endpoint | Description |
| --- | --- |
| `GET /health` | Liveness check |
| `GET /recalls?limit=20&offset=0` | Newest-first page (`limit` 1–100) with `total` |
| `GET /recalls/{id}` | One recall by official NID (404 if not in the dataset) |
| `GET /categories` | Official categories with record counts |

`/recalls*` and `/categories` return 503 until the dataset has been built.
Interactive API docs: <http://localhost:8000/docs>

> **Port 8000 already in use?** Run `uvicorn app.main:app --reload --port 8001` and set
> `NEXT_PUBLIC_API_BASE_URL=http://localhost:8001` in `frontend/.env.local`.

### 4. Frontend (Next.js)

```bash
cd frontend
npm install
cp .env.example .env.local             # optional for now
npm run dev
```

Open <http://localhost:3000>.

### 5. Tests and checks

```bash
# from the repo root, with the backend venv active
python -m pytest        # pipeline + API tests; real-data checks run once the DB is built

# frontend
cd frontend
npm run lint
npm run build
```

## Configuration

Environment variables are documented in `backend/.env.example` and
`frontend/.env.example`. Real `.env` files are git-ignored — never commit secrets.
Backend variables use the `RECALLIQ_` prefix (`RECALLIQ_DATABASE_PATH` overrides the
SQLite location).

## Data source

Government of Canada — Recalls and Safety Alerts open data, supplied by the project owner
as a downloaded CSV. RecallIQ does not scrape government webpages. Record URLs point to
the official pages on `recalls-rappels.canada.ca`.
