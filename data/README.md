# data/

- `raw/` — original datasets exactly as downloaded (never edited by hand or by code).
- `processed/` — outputs produced by `ml/recall_data` (see `docs/DATA_PIPELINE.md`).

All dataset files are git-ignored.

## Dataset register

| File | Source | Obtained | SHA-256 |
| --- | --- | --- | --- |
| `raw/HCRSAMOpenData.csv` | Government of Canada — Recalls and Safety Alerts open data (Health Canada / CFIA / Transport Canada), supplied by the project owner | Downloaded 2026-11-14; added to project 2026-11-15 | `30ad96831dd5d99cc16d63e94fb7ba43c558c464abab8ad73c62cd931aeae84d` |

Licence: Open Government Licence – Canada (confirm against the download page when citing
in the report).

## Processed outputs

| File | Content |
| --- | --- |
| `processed/recalls_eligible.jsonl` | all cleaned food + consumer-product recalls |
| `processed/recalls_sample.jsonl` | the seeded 1,000-record sample |
| `processed/recalliq.db` | SQLite database served by the API (sample only) |
| `processed/manifest.json` | counts, seed, input hash, per-stratum allocation |
