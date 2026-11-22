# ml/ — module guide

The offline Python package behind RecallIQ: data preparation, retrieval, training,
evaluation, identifier verification and RAG. It is installed in editable mode
(`pip install -e ml`), so the FastAPI backend imports **the same code** the experiments use.
Queries and documents are therefore always encoded identically.

```bash
cd backend && source .venv/Scripts/activate
pip install -r ../ml/requirements.txt && pip install -e ../ml
cd ../ml                     # run every `python -m ...` command below from ml/
```

Large outputs go to `ml/artifacts/` (git-ignored). Small, reviewable run records are exported
to `../experiments/`.

## Pipeline at a glance

```
recall_data ──► SQLite corpus ──► evaluation.splits (+ split_review: human-frozen split)
                     │                         │
                     ▼                         ▼
             retrieval (BM25, A)        training (B, C) ──► retrieval.finetuned (B/C indexes)
                     │                         │
                     └──────► evaluation (metrics, proxy_eval, evaluate) ──► reports
API request path: retrieval.search ──► verification.identifiers ──► rag.pipeline (+ rag.llm, rag.guards)
```

## Packages

| Package | Module | Responsibility |
|---|---|---|
| `recall_data/` | `load.py` | Read the official CSV as text; validate columns and unique numeric recall IDs |
| | `clean.py` | Keep food (CFIA) and consumer-product recalls; normalise whitespace, nulls, dates |
| | `grouping.py` | Exact-title groups and heuristic product families (with review flags) |
| | `sample.py` | Deduplicate by title; seeded, stratified 1,000-notice sample |
| | `storage.py` / `build.py` | JSONL + SQLite export (atomic replace); CLI and `manifest.json` |
| | `family_review.py` | Export product families that need manual review |
| `retrieval/` | `corpus.py` | Load notices from SQLite; build `search_text`; corpus fingerprint |
| | `bm25.py` | Okapi BM25 index and tokenizer |
| | `semantic.py` | Pinned MiniLM encoder, embedding save/load with integrity checks, cosine scores |
| | `finetuned.py` | Build and validate an index for a fine-tuned checkpoint (Models B, C) |
| | `search.py` / `loader.py` | `SearchEngine` over all four models; builds from disk, disabling unavailable models with a reason |
| | `embed.py` / `smoke.py` | CLIs: Model A embeddings; realistic smoke queries |
| `training/` | `config.py` | Typed, validated training configuration (`configs/*.json`) |
| | `data.py` | Generated (query, recall) pairs, leakage guards, hard-negative mining |
| | `model.py` | `MeanPoolEncoder`: BERT encoder + attention-masked mean pooling + L2 normalisation |
| | `trainer.py` | InfoNCE loss with group masking, training loop, per-epoch validation, checkpointing |
| | `train.py` | CLI: `python -m training.train --config configs/model_b.json --run-name model-b-v1` |
| `evaluation/` | `splits.py` | Grouped, stratified, seeded 60/20/20 split and leakage audits |
| | `split_review.py` | Human review of cross-split pairs; freezes the split manifest |
| | `annotations.py` / `annotate.py` | Annotation format and validation; interactive query/judgment tool |
| | `metrics.py` | Recall@k, MRR@k, graded nDCG@k |
| | `rankers.py` | One ranking interface for all four systems over the same index (deterministic ties) |
| | `diagnostics.py` | False-match rate @1 and identifier-verification precision, with explicit numerators/denominators |
| | `evaluate.py` | Evaluate on human-annotated queries (test split guarded) |
| | `proxy_eval.py` | Evaluate all models on generated validation queries (proxy) |
| `verification/` | `identifiers.py` | Exact UPC / model / lot / best-before extraction and matching |
| `rag/` | `pipeline.py` | Evidence selection, template answer, LLM orchestration and fallback |
| | `llm.py` | OpenAI adapter: structured context, strict JSON output, error mapping, usage |
| | `guards.py` | Deterministic acceptance checks for generated answers |
| `reports/` | `figures.py` / `seeds.py` | Generate `experiments/reports/` tables and SVG figures from run files |
| `configs/` | `model_b*.json`, `model_c*.json` | Training configurations (v1 and seeds 4110–4112) |

## Common commands

```bash
python -m recall_data.build                       # corpus (needs data/raw/HCRSAMOpenData.csv)
python -m retrieval.embed                         # Model A embeddings
python -m retrieval.smoke                         # sanity-check search output
python -m evaluation.split_review status          # must print: Manifest frozen: True
python -m training.train --config configs/model_b.json --run-name model-b-v1
python -m evaluation.evaluate --split validation  # human queries (add --include-single-annotator to include those)
python -m reports.figures --help                  # regenerate experiment tables/figures
```

The complete, ordered reproduction sequence is in [`../docs/CHECKPOINT.md`](../docs/CHECKPOINT.md).
Method documentation: [`DATA_PIPELINE`](../docs/DATA_PIPELINE.md) ·
[`RETRIEVAL`](../docs/RETRIEVAL.md) · [`TRAINING`](../docs/TRAINING.md) ·
[`EVALUATION`](../docs/EVALUATION.md) · [`RAG`](../docs/RAG.md).
