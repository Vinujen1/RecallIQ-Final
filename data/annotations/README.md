# data/annotations/ — human relevance judgments

This folder is **tracked in git** (team work product). It must only ever contain queries
and labels written by real team members — never generated or copied text.

| File | Purpose |
| --- | --- |
| `queries.jsonl` | Canonical annotation file (created by the tools; one query per line) |
| `template.csv` | Empty spreadsheet template (one row per query × relevant recall) |
| `templates/validation_queries.csv`, `templates/test_queries.csv` | Empty templates, one per evaluation split |

**Read first:** [`docs/QUERY_WRITING_GUIDE.md`](../../docs/QUERY_WRITING_GUIDE.md).
Do not start until the split manifest is frozen (`python -m evaluation.split_review status`).

## Record format (JSONL)

| Field | Required | Meaning |
| --- | --- | --- |
| `query_id` | yes | Unique id, e.g. `AB-0001` (annotator initials + number) |
| `query` | yes | The consumer-written question, ≤ 500 characters |
| `split` | yes | `validation`, `test` or `train` — must equal the split of the **anchor** recall |
| `relevant` | yes | **All** relevant recalls among the 1,000 (any split): `{"recall_id": <official NID>, "grade": 1-3}`; `grade` optional (default 1) |
| `anchor_recall_id` | no | The recall the consumer most likely means; default = highest grade, first listed |
| `split_fingerprint` | set by tools | Split assignment the query was labelled against |
| `reviewer_status` | yes | `draft` → `reviewed` (or `rejected`), or `draft` → `single_annotator` when working alone; by default only `reviewed` queries are evaluated |
| `annotator` | yes | Initials of the person who wrote the query |
| `reviewer` | for `reviewed` | Initials of a **different** person who checked the labels; must stay **empty** for `single_annotator` |
| `notes`, `created_at` | no | Free text; ISO timestamp |

Grades: **1** relevant · **2** highly relevant · **3** exactly the recall the user means.
Recalls not listed are treated as not relevant.

## Writing good queries
1. Write the query **before** looking at search results, in everyday words
   ("is the peanut butter I bought safe?"), not copied from a recall title.
2. Use `python -m evaluation.annotate --annotator <INITIALS>`; mark every relevant recall
   you can find (use `/find` to look beyond the suggested candidates).
3. Mark relevant recalls from any split; list the one the consumer most likely means first
   (it becomes the anchor and decides the query's split).
4. Do not use the `product_family` field as a guide — 458 of those labels are unverified.

## Reviewing
Export to a spreadsheet, check each judgment, set `reviewer_status` and `reviewer`, then
re-import (validation runs automatically and nothing is written if there are errors):

```bash
python -m evaluation.annotations export --out review.csv
python -m evaluation.annotations import review.csv --out data/annotations/queries.jsonl
python -m evaluation.annotations validate
```

## Working alone (single-annotator)
If no second person is available, re-check your own drafts and finalize them:

```bash
python -m evaluation.annotations finalize data/annotations/vd.jsonl --annotator VD
python -m evaluation.annotations finalize data/annotations/vd.jsonl --annotator VD --ids VD-0001 VD-0002
```

This changes only **your own `draft`** queries to `single_annotator` and never fills in a
reviewer. These queries are evaluated only with `evaluate --include-single-annotator`, and
the report shows them separately from independently reviewed queries. A teammate can still
review them later: export, set `reviewed` + their initials, and re-import. See
[`docs/EVALUATION.md` § 3a](../../docs/EVALUATION.md).
