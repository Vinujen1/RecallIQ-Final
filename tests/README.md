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

Two tests check the **real** built dataset and are skipped until
`python -m recall_data.build` has been run.
