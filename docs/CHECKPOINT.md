# Technical checkpoint (Nov 20, 2026) — evidence pack and remaining work

This is **not** the checkpoint PDF. The course requires students to write the report text;
this file lists the technical facts, where every number comes from, how to reproduce it, and
what is still missing. Numbers: `experiments/reports/tables.md` (generated from run files).

## Suggested PDF outline (guidelines §5.2, max 3 pages) → evidence to cite
| Section | Evidence in the repo |
|---|---|
| 1. Problem | Proposal; README |
| 2. Data | `docs/DATA_PIPELINE.md` (34,384 raw → 10,586 eligible → 1,000 selected, seed 4110); frozen split in `docs/EVALUATION.md` §2 (603/199/198, 0 crossing same-product/related-event pairs, 81 human pair decisions); sample inputs from `experiments/runs/model-b-v1/sample_train_pairs.jsonl` (generated queries — label them as such) |
| 3. Approach | Architecture below; `docs/TRAINING.md` (loss, how negatives enter, B vs C configs, mining/filtering) |
| 4. Preliminary results | `experiments/reports/tables.md` (config, per-epoch history, proxy table); figures `loss_model_b.svg`, `loss_model_c.svg`, `validation_loss_compare.svg`, `proxy_mrr_compare.svg` — **label proxy numbers as generated-query diagnostics** |
| 5. Remaining work, risks, timeline | "Remaining work" below |
| 6. Contributions | `CONTRIBUTIONS.md` (each member completes their own rows) |

## Architecture (as implemented)
```
Gov. of Canada recalls CSV (34,384) ─► recall_data (filter food+consumer, clean, group, sample 1,000)
        └─► SQLite recalliq.db ─► retrieval.corpus (search_text) ─► fixed 1,000-record index
evaluation.splits + human split review (81 decisions) ─► FROZEN 60/20/20 grouped split
train docs ─► generated (query, recall) pairs ─┬─► Model B: MiniLM + MNRL, in-batch negatives
             + mined/human-verified negatives ─┴─► Model C: MiniLM + MNRL + 1 hard negative
query ─► {BM25 | MiniLM (A) | Model B | Model C} ─► top-10 recalls ─► identifier verifier
        (UPC/model/lot/best-before: match / mismatch / unknown) ─► FastAPI ─► Next.js UI
validation: generated-query proxy (now) | reviewed human queries (pending); test: sealed
```

## Reproduce (from `ml/`, Python 3.12, `pip install -e . -r requirements.txt`)
```bash
python -m recall_data.build                    # corpus (needs data/raw/HCRSAMOpenData.csv)
python -m retrieval.embed                      # Model A embeddings
python -m evaluation.split_review status       # must print: Manifest frozen: True
python -m training.train --config configs/model_b.json --run-name model-b-v1
python -m training.train --config configs/model_c.json --run-name model-c-v1
python -m evaluation.proxy_eval --pairs artifacts/models/model-b-v1/validation_pairs.jsonl \
    --checkpoint model_b=artifacts/models/model-b-v1/checkpoints/best \
    --checkpoint model_c=artifacts/models/model-c-v1/checkpoints/best
python -m reports.figures --run model_b=artifacts/models/model-b-v1 \
    --run model_c=artifacts/models/model-c-v1 \
    --proxy ../data/processed/eval/proxy_validation.json --out ../experiments/reports \
    --export-runs ../experiments/runs
# serve B and C in the API: an index built by each exact checkpoint
python -m retrieval.finetuned --label minilm_finetuned \
    --checkpoint artifacts/models/model-b-v1/checkpoints/best --out artifacts/embeddings/model-b-v1
python -m retrieval.finetuned --label minilm_hard_negative \
    --checkpoint artifacts/models/model-c-v1/checkpoints/best --out artifacts/embeddings/model-c-v1
pip install -e . --no-deps                      # after pulling: registers rag/verification/reports
cd .. && python -m pytest                       # full test suite
```

### Live demo (from `backend/`)
```bash
uvicorn app.main:app --port 8000
curl localhost:8000/search/models
curl -X POST localhost:8000/search -H "Content-Type: application/json" \
     -d '{"query": "heater model XJ-500 overheating", "model": "minilm_hard_negative", "top_k": 5}'
curl -X POST localhost:8000/explain -H "Content-Type: application/json" \
     -d '{"query": "NordicTrack rowing machine model NTRW19147 fire", "model": "minilm_finetuned"}'
```
Verified 2026-11-16 on port 8011: all four models 200; good demo cases — the XJ-500 query
(BM25 ranks a Thermosphere heater first; the verifier marks it MISMATCH, `EH4604BY`) and the
NordicTrack query (top-1 recall 78486 for B and C; NTRW19147 vs listed NTRW19147.0–.3 →
UNKNOWN, not an over-claimed MATCH). First MiniLM request loads the model (~15–30 s on CPU).

## What the checkpoint can honestly claim
- Working end-to-end pipeline on the real data: data → frozen split → training (B and C) →
  validation every epoch → checkpointing → evaluation of all four systems on one index →
  served through FastAPI with identifier checks and evidence-only explanations.
- Learning curves for Models B and C (real CPU runs): `experiments/reports/loss_model_*.svg`.
- **Proxy** numbers (391 generated validation queries — say so in the PDF):
  BM25 0.9702 / Model A 0.9367 / Model B 0.9743 / Model C 0.9769 MRR@10
  (full table incl. Recall@5, nDCG@10, FMR@1: `experiments/reports/tables.md`).
- Fine-tuning beats the pretrained model on this proxy: B − A ΔMRR@10 +0.038, 95 % bootstrap
  CI [+0.022, +0.056]. C − B and fine-tuned − BM25 differences are within noise.
- Identifier verifier implemented, unit-tested and live in the API; identifier precision
  **not computable** yet (no identifier-bearing queries; 6/1,000 records list identifiers).

## Cannot be claimed yet
- Any retrieval quality on consumer-written queries (0 reviewed human queries).
- That hard negatives meaningfully help: across 3 seeds C beat B on proxy MRR@10 every
  time, but only by +0.0023 on average (sd 0.0013) on a saturated proxy (`experiments/reports/seeds.md`).
- Identifier-verification precision; RAG explanations (not started).

## Remaining work — priority order (owners to be agreed by the team)
| # | Task | Needs | Effort |
|---|---|---|---|
| 1 | **Write + cross-review ≥ 30 validation and ≥ 30 test queries** (`docs/QUERY_WRITING_GUIDE.md`) | all 3 members; a different member reviews each query | queries land in a split via their anchor (~20 % each val/test) → free writing needs ~150 queries for 30/30; see decision below |
| 2 | `evaluation.evaluate --split validation` with the B and C checkpoints; move checkpoint selection to human validation queries | item 1 | ~15 min CPU |
| 3 | Re-pool: reviewers judge Model B/C top-10 for every query (pooling bias) | item 1 | 1–2 h people |
| 4 | ~~Seeds 4110–4112 for B and C, v2~~ — **done** (`experiments/reports/seeds.md`) | — | — |
| 5 | Fetch official notice pages (UPC / lot / best-before) as verifier evidence — proposal pre-processing step not done yet | network, polite rate limit | ~2 h |
| 6 | RAG: foundation done (`POST /explain`, template + grounding guard). Remaining: an LLM adapter (optional, env-configured) whose output must pass `check_grounded`; evaluate explanations by hand on ~20 queries | items 2, 5 | 0.5–1 day |
| 7 | ~~Demo website~~ — **built** (Phase 5: `/`, `/search`, `/recalls/[id]`, `/model-lab`, `/how-it-works`; see `docs/PROGRESS.md`) | — | — |
| 8 | Final test evaluation **once**, after model selection and judgments are complete | items 1–4 | 10 min |
| 9 | CONTRIBUTIONS.md + commits from every member's own account | everyone | ongoing |

**Decision for the team (query efficiency):** the current guide forbids picking a recall and
describing it. An anchor-first protocol (the tool shows a random validation/test recall; the
writer describes it as a consumer would, without copying, then marks all relevant recalls)
reaches 30/30 with ~60 queries, at the cost of queries biased toward known recalls. Changing
the protocol is the team's call; document it either way.

## Implementation deviations from the proposal (state them in the PDF)
| Proposal | Implementation | Why / impact |
|---|---|---|
| Streamlit demo | Next.js frontend + FastAPI backend (since Phase 1A) | The demo goal (live inference on held-out inputs, ranked notices, verification, official links) can be met by FastAPI `/search` + the Next.js page; a Streamlit rewrite adds no evaluation value. A thin Streamlit client over the same API is the fallback if the UI is not ready. |
| Fetch full notice pages; extract UPC/lot/expiry | CSV fields only so far | Identifier precision not computable; remaining work #5 |
| Split by product family | Grouped by title group / product-key Jaccard / brand / human review | Heuristic product family proved unreliable (458/1,000 unverified) |
| "Generate training queries" | Deterministic templates (no LLM) | Reproducible, but lexically easy (TRAINING.md) |
| 34,299 records / 10,560 eligible | 34,384 / 10,586 | Newer CSV download |
