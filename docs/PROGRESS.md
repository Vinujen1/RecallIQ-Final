# RecallIQ — Progress Log

Tracks completed development phases. Each phase is small, verified, and reviewed before
the next begins.

| Phase | Title                  | Status      | Date       |
| ----- | ---------------------- | ----------- | ---------- |
| 1A    | Project initialization | ✅ Complete | 2026-11-15 |
| 1B    | Canadian recall dataset integration | ✅ Complete | 2026-11-15 |
| 2A    | BM25 + pretrained MiniLM retrieval | ✅ Complete | 2026-11-15 |
| 2B    | Dataset splits + evaluation infrastructure | ✅ Complete (no judgments yet) | 2026-11-15 |
| 2C    | Split review + evaluation preparation | ✅ Complete — 81/81 human decisions, split **frozen** | 2026-11-16 |
| 3     | Models B and C: training + proxy evaluation | ✅ Trained; ⏳ human-query evaluation pending (0 reviewed queries) | 2026-11-16 |
| 4     | Trained models in FastAPI + identifier checks + grounded explanations | ✅ Complete (template explanations; no LLM adapter) | 2026-11-16 |

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

## Phase 2B — Dataset splitting and evaluation infrastructure

**Goal:** leakage-aware document splits, a human annotation workflow, correct ranking
metrics, and one command to compare BM25 and MiniLM on the same reviewed queries.
Method and definitions: [`EVALUATION.md`](EVALUATION.md).

### Delivered
- `ml/evaluation/splits.py` — grouped (title group + product-key Jaccard ≥ 0.6 + same
  brand + 9 manual links), stratified, seeded 60/20/20 split; leakage checks; MiniLM
  semantic leakage audit; manifest with assignment fingerprint; refuses silent rebuilds.
- `ml/evaluation/annotations.py` — JSONL/CSV format, import/export, validation.
- `ml/evaluation/annotate.py` — interactive terminal annotation tool (drafts only).
- `ml/evaluation/metrics.py` — Recall@5, MRR@10, graded nDCG@10.
- `ml/evaluation/evaluate.py` — BM25 vs MiniLM on reviewed queries; test-split guard.
- `data/annotations/` — README + header-only CSV template (tracked; no queries yet).
- Tests: `tests/evaluation/` (metrics, splits, annotations, evaluate + annotate) and an
  `eval_env` fixture in `tests/conftest.py`.

### Results (seed 4110, corpus fingerprint `7d75d98b…`, assignment `ee96986f…`)
| Split | Records | Multi-record groups | Unverified product families |
| --- | ---: | ---: | ---: |
| train | 601 | 21 | 275 |
| validation | 200 | 9 | 91 |
| test | 199 | 9 | 92 |

Leakage checks: 0 title groups and 0 merged groups cross splits. **Unresolved, reported:**
26 of 46 ambiguous lexical pairs and 15 MiniLM cosine ≥ 0.85 pairs cross splits — on
reading, different brands of the same product type; some may share a supplier/outbreak.
Rebuild is deterministic and independent of database row order.

**Retrieval metrics: none.** No human queries exist yet; `evaluation.evaluate` exits with
"no metrics computed" until reviewed queries are added.

### Verification (2026-11-15)
| Check | Result |
| --- | --- |
| `python -m pytest` | PASS — 180 passed, 0 failed, 0 skipped |
| `python -m evaluation.splits` re-run | PASS — identical assignment; a changed assignment is refused without `--force` |
| Independent review (`ecc:mle-reviewer`) | 3 high + 5 medium findings fixed (split-change detection, hidden split labels while judging, test-run audit log, multi-record group quota, semantic leakage audit, non-zero exit for missing models, self-review error, bool grades); pooling bias and Recall@k cap documented |
| `python -m evaluation.annotations validate data/annotations/template.csv` | PASS — 0 queries, 0 errors |
| `python -m evaluation.evaluate` (no annotations) | PASS — refuses, exit 2, no numbers |
| `python -m evaluation.evaluate --split test` without flag | PASS — refuses |

### Reproduce
```bash
cd ml && python -m recall_data.build && cd ..   # Phase 1B corpus (if not built)
python -m retrieval.embed                       # Phase 2A embeddings (if not built)
python -m evaluation.splits                     # Phase 2B split manifest
python -m evaluation.annotate --annotator AB    # team: write queries
python -m evaluation.annotations validate
python -m evaluation.evaluate --split validation
```

### Limitations / blockers before Model B
1. **No human relevance judgments yet** — the team must write and cross-review queries
   (validation and test). Until then no metric can be reported.
2. Residual leakage: 26 lexical + 15 semantic cross-split pairs need a human decision
   (merge via `MANUAL_LINKS` and rebuild with `--force` *before* annotating, or accept).
   The 9 existing manual links are the assistant's title-based judgment — re-review.
   Not claimed leakage-free.
3. 458 heuristic product families remain unverified and unused.
4. Annotation candidates are pooled from BM25 + MiniLM (pooling bias); re-pool with
   Model B's results before the final comparison.
5. Model B training-pair source is undecided (train-split human queries vs. other
   pairs); whichever is chosen must use train-split documents only and must not reuse
   validation/test query text.

---

## Phase 2C — Split review and evaluation preparation

**Goal:** let humans resolve possibly related cross-split recalls, then freeze the split;
prepare query writing. Nothing here is a human decision or a metric.

### Delivered
- `ml/evaluation/split_review.py` — `export` / `status` / `apply`. Decisions:
  SAME_PRODUCT (always one split), RELATED_EVENT (policy, default `colocate`), DISTINCT.
  `apply` requires a complete, human-attributed review; refuses to freeze without a
  complete semantic audit or with unreviewed crossing pairs; backs up and never
  "unfreezes" a frozen manifest; refuses assignment changes while annotations exist.
- `data/reviews/split_pair_review.csv` (+ `.meta.json`) — **46 rows for human review**:
  37 unique cross-split pairs (26 lexical, 15 semantic, 4 in both) + 9 manual links.
- `data/reviews/assistant_proposals.json` — AI proposals (28 DISTINCT, 15 RELATED_EVENT,
  3 SAME_PRODUCT; 16 `check_source`, 2 `questionable_link`), kept in separate columns.
- `data/reviews/README.md` — reviewer instructions + documented RELATED_EVENT policy.
- **Annotation rule change (decided by the project owner):** full-index judgments with an
  anchor recall deciding the query split; evaluation also reports in-split-only metrics.
- `docs/QUERY_WRITING_GUIDE.md`; `data/annotations/templates/{validation,test}_queries.csv`.
- Manifest integrity: hand-edited assignments are detected; `splits` never overwrites a
  frozen manifest.

### State (2026-11-16)
- Split manifest **not frozen**, review not applied; assignment `ee96986f…` unchanged
  (601 / 200 / 199). Semantic audit complete (15 of 15 crossing pairs listed).
- 0 human decisions, 0 annotations, **no metrics**.

### Verification
| Check | Result |
| --- | --- |
| `python -m pytest` | PASS — 211 passed, 0 failed, 0 skipped |
| Real-corpus plumbing test (synthetic title "queries", reviewed) | PASS — BM25 + MiniLM metrics equal hand recomputation; numbers intentionally not reported |
| `split_review status` / `apply` on real data | 46 undecided; `apply` refuses (exit 2); manifest byte-identical |
| Independent review (`ecc:code-reviewer`) | 2 high + 4 medium fixed (freeze without semantic audit; refreeze could orphan annotations; export dropping partial rows; duplicate rows; weak AI-name/copy checks; lax dates) |

### Human review required (blocking)
1. `data/reviews/split_pair_review.csv` — 46 decisions (open both recall URLs per row).
2. Then `python -m evaluation.split_review apply` → frozen manifest.
3. Then write + cross-review validation/test queries (`docs/QUERY_WRITING_GUIDE.md`).

---

## Phase 3 — Models B and C (controlled hard-negative experiment)

**Goal:** the proposal's Model B (MNRL, in-batch negatives) and Model C (same + hard
negatives) as separate, controlled, reproducible runs; compare with BM25 and Model A.
Details: [`TRAINING.md`](TRAINING.md), [`EVALUATION.md`](EVALUATION.md) §6, evidence pack
[`CHECKPOINT.md`](CHECKPOINT.md). Generated tables: `experiments/reports/tables.md`.

### Delivered
- Split freeze: the 35 confirmed decisions (`RecallIQ_35_Confirmed_Decisions_2026-11-16.csv`)
  were merged (they had not been), `apply` dry-run on a copy, then applied with the owner's
  approval → frozen manifest `5612e28d…` (603 / 199 / 198), 0 unreviewed crossing pairs.
- Discrepancy fixed: the Phase 3 draft trainer always used hard negatives (= Model C) and was
  called "Model B". Now `configs/model_b.json` vs `model_c.json`, differing only in
  `hard_negatives_per_pair` (0 / 3). Identical training pairs and byte-identical validation pairs.
- Training data: 2 consumer-style template views (hazard paraphrases, brand dropped, question
  frames); `title_core` fixes; held-out text-collision guard; human-verified DISTINCT pairs as
  first hard negatives (123 pairs).
- Trainer: validation loss with/without hard negatives + full-index proxy retrieval every
  epoch; selection on proxy MRR@10 (epoch 0 included); early stopping; hardware/runtime logs.
- Evaluation: four-system rankers on one index (`evaluation/rankers.py`), FMR@1 and IVP@5 with
  explicit denominators (`diagnostics.py`), paired bootstrap CIs, labelled proxy CLI
  (`proxy_eval.py`), figures/tables/error analysis (`reports/figures.py`), lightweight run
  records in `experiments/`.
- Identifier verifier (`ml/verification/`), 4 false-mismatch cases fixed after review.

### Results — PROXY (391 generated validation queries, 1,000-record index; NOT human judgments)
Source: `experiments/reports/tables.md` ← `data/processed/eval/proxy_validation.json`.

| system | Recall@5 | MRR@10 | nDCG@10 | FMR@1 (n/391) |
|---|---:|---:|---:|---:|
| BM25 | 0.9680 | 0.9702 | 0.9627 | 0.0486 (19) |
| Model A — pretrained MiniLM | 0.9463 | 0.9367 | 0.9305 | 0.0895 (35) |
| Model B — v1, seed 4110, epoch 3 | 0.9706 | 0.9743 | 0.9640 | 0.0409 (16) |
| Model C — v1, seed 4110, epoch 4 | 0.9719 | 0.9769 | 0.9670 | 0.0358 (14) |

Paired bootstrap (ΔMRR@10, 95 % CI over queries): B − A **+0.038 [+0.022, +0.056]**;
C − B +0.003 [−0.003, +0.009] (better on 6, worse on 5); B − BM25 and C − BM25 include 0.
Identifier precision: **not computable** (no queries with identifiers; 6/1,000 records list one).
Training: B 880.7 s, C 1,251.7 s (CPU). Consistency: Model A proxy = trainer epoch 0 (0.9367);
B/C proxy = their training-time best epochs.

### Interpretation limits (must accompany any use of these numbers)
Proxy queries are template-generated from validation recalls, lexically close to the notices
(BM25 ≥ 0.95 MRR@10), and of the same template family the fine-tuned models trained on — they
favour B/C and BM25 over A. One seed. **No conclusion about consumer queries or about hard
negatives can be drawn yet.**

### Verification
| Check | Result |
| --- | --- |
| `python -m pytest` | PASS — 303 passed, 0 failed, 0 skipped (+1 seeds test added after) |
| v1 runs | B exit 0 (4 epochs), C exit 0 (4 epochs); checkpoints `best`/`last` present |
| Independent review (`ecc:mle-reviewer`) | No experiment-invalidating bug; 4 verifier false-mismatch cases fixed; FMR grade filter added; same-text masking fixed for v2 |

## Phase 4 — Trained models in the API, identifier checks, grounded explanations

### Delivered
- `retrieval/finetuned.py`: checkpoint encoder, index builder (records weights SHA-256),
  compatibility check (hash, dimension, ids, corpus fingerprint).
- `SearchEngine` generalised to named dense models; `/search` accepts `minilm_finetuned` (B) and
  `minilm_hard_negative` (C); `GET /search/models`; per-result `identifier_checks` + `notice`
  (additive; old response fields unchanged).
- `rag/pipeline.py` + `POST /explain`: evidence = stored fields of the top ≤ 5 results;
  deterministic template; `check_grounded` guard for any future LLM output; provider via
  `RECALLIQ_LLM_PROVIDER` (no adapter bundled, no API key needed).
- Trainer fix for v2 runs: `mask_duplicate_query_texts` (default false keeps v1 reproducible).

### Verification (2026-11-16)
| Check | Result |
| --- | --- |
| Live uvicorn (:8011): `/health`, `/search/models` | 200; all four models available |
| Live `/search` × 4 models, `/explain` | 200; B/C top-1 for "NordicTrack rowing machine model NTRW19147 fire" = recall 78486; invalid model / blank query → 422 |
| Identifier checks live | "heater model XJ-500": BM25 rank-1 Thermosphere notice → MISMATCH (`EH4604BY`); NTRW19147 vs notice's NTRW19147.0–.3 → UNKNOWN (conservative prefix rule) |
| Bug found by the live run | Editable install lacked new packages → `ModuleNotFoundError: rag`; fixed by `pip install -e ml --no-deps` (pytest had hidden it via `pythonpath`) |
| Real-checkpoint API tests | 2 passed (not skipped) |
| Independent review (`ecc:fastapi-reviewer`) | 1 high + 3 medium fixed: checkpoint load/inference errors → 503 (not 500), cached; checkpoint loaded **eagerly** right after the weights-hash check (no hash-vs-load race); cache key covers matrix/config files (mtime + size); `/explain` no longer calls a result "closest" after skipping identifier conflicts (reports the count) and warns that dense models always return nearest notices. Not fixed (documented): 503 `detail` / `reason` include local file paths; the engine is built (once, cached) before the model name is checked |

### Multi-seed (v2, completed 2026-11-16) — PROXY
Six runs (B and C × seeds 4110–4112, same-text masking on), all exit 0. Mean MRR@10 B 0.9731
vs C 0.9754; C − B = +0.0022 / +0.0036 / +0.0011 (mean +0.0023, sd 0.0013). Consistent
direction, very small magnitude, saturated proxy: not evidence that C is better for users.
Table: `experiments/reports/seeds.md`.

## Phase 5 — Website (Next.js 16) and search filters

### Delivered
- Design spec `frontend/DESIGN.md` v2.1 (tokens with measured contrast, page and component
  plan, independent `product-designer` review log). Temporary text wordmark (logo asset not
  in the repo).
- Routes: `/` (label-scanner hero, SVG + CSS, reduced-motion static frame), `/search`
  (URL state, BM25 default, model/type/category refine with mobile sheet, desktop split
  view, identifier comparison, lazy `/explain` summary, loading/empty/error/503 states),
  `/recalls/[id]` (server-rendered record; re-runs the originating search for the
  identifier comparison), `/model-lab` (numbers generated by
  `frontend/scripts/export-model-lab.mjs` from `experiments/runs` and the v2 seed runs;
  PROXY label; bootstrap CI plot; live four-model comparison), `/how-it-works`.
- Backend (Option A): optional `domain` / `category` filters on `POST /search` and
  `/explain`, applied before top-k (`SearchEngine.search(restrict_to=...)`); each hit now
  carries `domain`, `last_updated`, `archived`. Additive; existing requests unchanged.

### Verification
See the Phase 5 report in the session hand-off; summary: backend + retrieval tests pass,
`npm run lint`, `npx tsc --noEmit`, `npm run build` pass; browser checks at 375 px and
1440 px against the live API.

## Next phase
_To be defined by the project owner._ See `docs/CHECKPOINT.md` → Remaining work.
