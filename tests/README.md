# tests/

Run everything from the repo root with the project venv active:

```bash
python -m pytest
```

`pytest.ini` adds `backend/` and `ml/` to the import path.

| Path | Covers |
| --- | --- |
| `conftest.py` | Shared fixtures; builds a small **synthetic** CSV with the official column names (temp dir only) |
| `backend/test_health.py` | `GET /health` |
| `backend/test_recalls_api.py` | `/recalls` pagination + validation, `/recalls/{id}`, `/categories`, 503 without DB |
| `ml/test_recall_data.py` | loading/validation, filtering, normalization, title groups, product families, sampling, SQLite import |
| `ml/test_retrieval.py` | search_text, BM25 ranking/ties, empty queries, invalid models, embedding save/load/validation, stale-index detection, revision pin |
| `backend/test_search_api.py` | `POST /search` responses, score labels, 422 validation, 503 degradation, id mapping |

| `evaluation/test_metrics.py` | Recall@5, MRR@10, graded nDCG@10 vs hand-computed values; invalid inputs |
| `evaluation/test_splits.py` | split reproducibility, order independence, grouping rules, leakage detection, overwrite guard, semantic audit |
| `evaluation/test_annotations.py` | JSONL/CSV import, duplicate labels, unknown ids, split mismatch, self-review, stale split fingerprints |
| `evaluation/test_evaluate_and_annotate.py` | evaluation runner (reviewed-only, test guard, exit codes, audit log) and the interactive annotation tool |

Unit tests use a deterministic `FakeEncoder` (no model download). Tests marked for the
**real** dataset/embeddings are skipped until `python -m recall_data.build` and
`python -m retrieval.embed` have been run.
