# ml/

Offline data processing, model training, and evaluation for RecallIQ.

## `recall_data/` — dataset pipeline (Phase 1B)

```bash
cd ml
python -m recall_data.build [--input PATH] [--output-dir DIR] [--size 1000] [--seed 4110]
```

| Module | Responsibility |
| --- | --- |
| `load.py` | Read the CSV as text; validate columns and unique numeric NIDs |
| `clean.py` | Keep CFIA + Consumer product safety; normalize whitespace, nulls, dates |
| `grouping.py` | Exact-title `title_group`; heuristic `product_family` with review flags |
| `sample.py` | Deduplicate by title, stratify, seeded proportional sample |
| `storage.py` | JSONL + SQLite export (atomic replace) |
| `build.py` | CLI orchestration and `manifest.json` |

Method and rationale: `docs/DATA_PIPELINE.md`. Dependencies: `ml/requirements.txt`.

## Later phases
Model training/evaluation (with fixed seeds, a documented baseline, and leak-free
splits) will live alongside this package. Trained artifacts go to `ml/artifacts/`
(git-ignored).
