# RecallIQ — Progress Log

Tracks completed development phases. Each phase is small, verified, and reviewed before
the next begins.

| Phase | Title                  | Status      | Date       |
| ----- | ---------------------- | ----------- | ---------- |
| 1A    | Project initialization | ✅ Complete | 2026-11-15 |
| 1B    | Canadian recall dataset integration | ✅ Complete | 2026-11-15 |
| 2A    | BM25 + pretrained MiniLM retrieval | ✅ Complete | 2026-11-15 |

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

## Phase 2A — BM25 and pretrained MiniLM retrieval

**Goal:** lexical and semantic search over the existing 1,000-record corpus, served by
`POST /search`, without any training. Details: [`RETRIEVAL.md`](RETRIEVAL.md).

### Delivered
- `ml/retrieval/` — `corpus` (deterministic `search_text` + fingerprint), `bm25` (Okapi),
  `semantic` (MiniLM encoder, embedding store, cosine), `search` (shared interface,
  `bm25` | `minilm`), `loader`, `embed` (CLI), `smoke` (CLI).
- `ml/recall_data/family_review.py` — exports the 458 `needs_review` records to CSV.
- `ml/pyproject.toml` — `pip install -e ml` makes `recall_data`/`retrieval` importable.
- Backend: `app/search_service.py` (cached engine, 503 on unavailable index),
  `app/routers/search.py` (`POST /search`), search schemas, `embeddings_dir` setting,
  CORS now allows POST.
- Tests: `tests/ml/test_retrieval.py`, `tests/backend/test_search_api.py`, `FakeEncoder`
  in `tests/conftest.py`.

### Results
| Item | Value |
| --- | --- |
| Model | `sentence-transformers/all-MiniLM-L6-v2` @ `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (pretrained, no fine-tuning) |
| Embeddings | (1000, 384) float32, unit-norm; `embeddings.npy` 1,536,128 bytes |
| Reproducibility | re-running `retrieval.embed` gives a bitwise-identical file (same environment) |
| Retrieval quality | **not measured** — no independent relevance judgments yet |

### Verification (2026-11-15)
| Check | Result |
| --- | --- |
| `python -m pytest` | PASS — 106 passed, 0 failed, 0 skipped (incl. real-embedding + real-API tests) |
| `python -m retrieval.embed` (x3) | PASS — identical `embeddings.npy` hash each run |
| `python -m retrieval.smoke` | PASS — 6 queries, both models return results with official recall ids |
| Live API (uvicorn :8001) `POST /search` bm25 + minilm | PASS — 200, labelled `score_type`; invalid input → 422 |
| Independent reviews | `ecc:mle-reviewer`: 0 high; 4 medium + 1 low fixed (500→503 paths, code-level revision pin, model-load failure cache, matrix validation, server faults no longer 422) |

### Known issues / notes
- The brief referenced an existing `search_text` field; none existed. It is derived in
  `retrieval/corpus.py` from verbatim fields (Phase 1B schema unchanged).
- No held-out test split exists yet; Phase 2A tuned nothing on data.
- BM25 shows surface-token matches in the smoke test ("can" → "6 Can Mini Fridge");
  intentionally not tuned to those queries.
- 458 heuristic product families remain unverified; an informal spot check found clear
  errors (see RETRIEVAL.md). They are excluded from `search_text`.
- First MiniLM request loads the model (~14 s on this machine); later requests ~16 ms
  median locally (informal, 5 requests).
- The initial unfiltered model download put ~977 MB in the user-level Hugging Face cache
  (`~/.cache/huggingface`); only ~92 MB is needed. Downloads are now filtered; the extra
  files were left in place (outside the project) pending the owner's decision.

### Explicitly out of scope
Fine-tuning, hard negatives, RAG/LLM calls, frontend changes, evaluation metrics.

---

## Next phase
_To be defined by the project owner._
