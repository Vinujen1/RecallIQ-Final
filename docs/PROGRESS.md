# RecallIQ — Progress Log

Tracks completed development phases. Each phase is small, verified, and reviewed before
the next begins.

| Phase | Title                  | Status      | Date       |
| ----- | ---------------------- | ----------- | ---------- |
| 1A    | Project initialization | ✅ Complete | 2026-11-15 |
| 1B    | Canadian recall dataset integration | ✅ Complete | 2026-11-15 |

---

## Phase 1A — Project initialization

**Goal:** a runnable, empty skeleton for frontend, backend, ML, data, docs and tests.

### Delivered
- `frontend/` — Next.js 16 (App Router, `src/` dir), TypeScript, Tailwind CSS v4, ESLint.
  Default starter page kept; only page metadata changed to "RecallIQ".
- `backend/` — FastAPI app factory (`app/main.py`), env-based settings
  (`app/config.py`, `RECALLIQ_` prefix), `GET /health` router, CORS limited to the
  frontend origin.
- `tests/backend/test_health.py` — pytest tests for `/health` (200, 405, 404).
- `ml/`, `data/raw`, `data/processed` — placeholders only.
- Root `.gitignore` extended for Node, env files, datasets and model artifacts;
  root `README.md`; `docs/ARCHITECTURE.md`; this file.

### Verification (2026-11-15)
| Check | Result |
| --- | --- |
| `python -m pytest` | PASS — 3 passed |
| Live `GET /health` (uvicorn) | PASS — 200, `{"status":"ok",...}` |
| CORS: `localhost:3000` allowed, foreign origin rejected | PASS |
| `GET /docs` (OpenAPI UI) | PASS — 200 |
| `npm run lint` | PASS — 0 problems |
| `npm run build` | PASS |
| `npm run dev` → `GET /` | PASS — 200, title "RecallIQ" |

### Known issues / notes
- On the development machine, port 8000 was already taken by an unrelated local app, so
  the live check ran on port 8001. See README for the override.
- pytest emits a `StarletteDeprecationWarning` about `httpx` in `TestClient`; harmless
  for now, revisit when upgrading test dependencies.
- `npm audit` reports 7 moderate-severity advisories in transitive dev dependencies
  (`minimatch` via ESLint tooling) from the create-next-app template. Not force-fixed,
  to avoid breaking changes; revisit when upgrading ESLint.

### Explicitly out of scope
ML models, RAG, application pages, databases, cloud services, deployment.

---

## Phase 1B — Canadian recall dataset integration

**Goal:** turn the official Government of Canada recalls CSV into a clean, reproducible
1,000-record dataset and serve it from SQLite through the API.
Full method: [`DATA_PIPELINE.md`](DATA_PIPELINE.md).

### Delivered
- `ml/recall_data/` — pandas pipeline: `load` (schema + NID validation) → `clean`
  (organization filter, null/date normalization) → `grouping` (exact-title groups,
  heuristic product families with review flags) → `sample` (seeded stratified selection)
  → `storage` (JSONL + atomically rebuilt SQLite) → `build` (CLI + manifest).
- `backend/app/` — `db.py` (read-only SQLite connection, 503 when missing),
  `schemas.py` (Pydantic response models), `routers/recalls.py`
  (`GET /recalls`, `GET /recalls/{id}`, `GET /categories`). `/health` unchanged.
- Tests: `tests/conftest.py` (synthetic fixture CSV), `tests/ml/test_recall_data.py`,
  `tests/backend/test_recalls_api.py`.
- Docs: `docs/DATA_PIPELINE.md`; README, ARCHITECTURE, data/ml/tests READMEs updated.

### Results (input SHA-256 `30ad9683…aeae84d`, seed 4110)
| Metric | Value |
| --- | ---: |
| Raw records | 34,384 |
| Eligible (CFIA food + Consumer product safety) | 10,586 (5,348 food / 5,238 consumer) |
| Distinct exact-title groups (eligible) | 9,889 — 435 groups have duplicates |
| Selected records | 1,000 (476 food / 524 consumer) |
| Product family confidence (selected) | high 542 · medium 272 · low 186 |
| Flagged `needs_review` (selected) | 458 |
| Unparseable dates (eligible) | 0 |

### Verification (2026-11-15)
| Check | Result |
| --- | --- |
| `python -m recall_data.build` | PASS — outputs written |
| Re-run with same seed | PASS — byte-identical `recalls_sample.jsonl` |
| SQLite `PRAGMA integrity_check` | PASS — `ok` |
| `python -m pytest` | PASS — 50 passed (incl. 2 real-data checks) |
| Live API (uvicorn :8001): `/health`, `/recalls`, `/recalls/{id}`, `/categories` | PASS |
| Live API invalid input (`limit=0`, `limit=101`, `offset=-1`, `id=abc`) | PASS — 422; unknown id → 404 |
| Independent review (`ecc:python-reviewer` agent) | 0 high; 2 medium fixed (64-bit overflow → 500, Windows locked-file rebuild) |
| Frontend | Unchanged this phase |

### Known issues / notes
- The raw CSV was found at the repo root (not git-ignored) and was moved, unmodified
  (hash verified), to `data/raw/HCRSAMOpenData.csv`.
- Product families are heuristic: 458/1,000 selected records need manual review.
- Sparse source fields stay `null` (never filled in): of the 1,000 selected records,
  748 have no Product, 811 no "What you should do", 263 no Issue, 522 no Recall class
  (consumer-product recalls rarely carry a class).
- Whitespace normalization collapses newlines; the current CSV has none in text fields.

### Explicitly out of scope
New frontend pages, PostgreSQL/Supabase, RAG, MiniLM training, scraping, deployment.

---

## Next phase
_To be defined by the project owner._
