# RecallIQ — Splits and Evaluation (Phase 2B–3)

Code: `ml/evaluation/`. **No retrieval metrics on human queries exist yet**: there are no
reviewed human relevance judgments so far, and `evaluation.evaluate` refuses to produce
numbers without them. The only numbers so far are the clearly labelled **proxy** results on
generated validation queries (§6), which are a pipeline/selection diagnostic, not results.

## 1. Two different things: the index vs. the splits

| | What | Used for |
| --- | --- | --- |
| **Retrieval candidate index** | All 1,000 records, fixed | Every model always searches all 1,000 (fair comparison) |
| **Document splits** | Partition of the same 1,000 ids into train / validation / test | Which documents may appear in *training pairs* (Model B, later) vs. in *validation/test judgments* |

**Judgments cover the full index (Phase 2C decision).** Annotators mark every relevant
recall among all 1,000, whatever its split. A query's split is the split of its **anchor**
recall (explicit `anchor_recall_id`, else the highest-grade recall listed first), so
validation/test queries still target documents held out from training. Evaluation reports
metrics on all judgments **and** on in-split judgments only
(`mean_in_split_judgments_only`), plus the number of judgments outside the query's split,
because relevant recalls from train may be easier for a model fine-tuned on train
documents. (Phase 2B required all relevant recalls in one split; that silently turned
correct cross-split results into false negatives and was replaced.)

Model B training data may only use train-split documents and train-split queries;
validation is for model selection; **test is used once, for the final report**
(`--confirm-final-test`, every run audit-logged).

**Split review (Phase 2C).** Before any query is written, team members review the
potentially related cross-split pairs and the manual links, then apply and **freeze** the
split — see [`data/reviews/README.md`](../data/reviews/README.md). **Applied and frozen on
2026-11-16** (81/81 human decisions); the split below is the frozen one.

## 2. Split construction — `python -m evaluation.splits`

Seed **4110**, targets 60 / 20 / 20. Records in one **group** always share a split.

| Rule | Catches |
| --- | --- |
| identical `title_group` | exact duplicate titles |
| product-key Jaccard ≥ 0.6, transitive (title tokens; hazard text and "brand/certain/recalled…" removed) | the same product re-notified (e.g. two Nestle peanut-butter ice-cream notices, FRIGG pacifiers) |
| same explicit "`<Brand>` brand" in the title, within a domain | sibling products / one recall event (Ruffles ×2, Wang Korea ×3, Alasko ×3) |
| 9 manual links (`MANUAL_LINKS`, reviewed from titles) | same company/event with no brand phrase (Mannarich Food, Farm Boy, Bed Handles re-announcement, Amana, Evenflo, Taylor Farms, pistachio + cantaloupe outbreaks, beef E. coli) |

Jaccard thresholds were chosen by reading candidate pairs at each band (grouping only — no
model scores). Over-merging is the safe direction: it costs balance, not validity.

**Stratification:** `domain::primary official category`. The heuristic `product_family`
is **not used** (458 unverified; see RETRIEVAL.md). Category-level similarity across splits
(e.g. cheddar from two different brands) is intended — it is not leakage.

**Assignment:** groups shuffled once with the seed, placed largest first. Multi-record
groups get their own 60/20/20 quota so validation and test also contain sibling products;
then each group goes to the split with the largest per-stratum deficit (then global).
The result does not depend on database row order (tested).

### Frozen split (2026-11-16)
Assignment fingerprint `5612e28d…0987`, corpus fingerprint `7d75d98b…`, `frozen: true`,
review policy `colocate` (RELATED_EVENT pairs kept together). Source: manifest `counts`,
`checks`. (Pre-review assignment `ee96986f…` of 2026-11-15 is superseded; 504 records
changed split when the reviewed groups were applied — no annotations existed yet.)

| | train | validation | test |
| --- | ---: | ---: | ---: |
| Records | 603 (60.3%) | 199 (19.9%) | 198 (19.8%) |
| Multi-record groups (43 total, sizes 2–5) | 24 | 10 | 9 |
| Records with unverified product family | 273 | 95 | 90 |

| Leakage check | Result |
| --- | --- |
| Every record assigned exactly once | ✅ |
| Title groups / merged groups crossing splits | 0 / 0 |
| SAME_PRODUCT or colocated RELATED_EVENT pairs crossing splits | 0 |
| Candidate pairs crossing splits without a human decision | 0 |
| Lexical (Jaccard 0.4–0.6) pairs crossing splits | 26 of 47 — all human-reviewed DISTINCT |
| Semantic audit (MiniLM cosine ≥ 0.85) pairs crossing splits | 17 — all human-reviewed DISTINCT; audit complete |

### Residual leakage risks — reported, not eliminated
- Review covered only lexical candidates (Jaccard 0.4–0.6), MiniLM pairs with cosine ≥ 0.85
  and the manual links. Related recalls below those thresholds were never shown to a person.
- DISTINCT decisions were made by one reviewer (VD), partly using AI-gathered evidence
  (`evidence_audit.csv`, `candidate_evidence_35.csv`); no second independent reviewer.
- Crossing DISTINCT pairs are still similar products (different brands of shrimp, sesame,
  cheddar…). That is intended category-level similarity, but a model fine-tuned on one can
  find the other easier — relevant for interpreting validation/test gains.
- Lexical + embedding audits cannot see relations that are not in the text (shared
  supplier, same outbreak described differently).
- The heuristic `find_violations` check verifies the assignment against the groups; it
  cannot find relations the grouping rules missed — that is what the audits are for.
- The dataset is **not** claimed to be leakage-free.

Outputs (git-ignored, regenerable): `data/processed/splits/recall_splits_v1.json` (seed,
targets, assignment + corpus fingerprints, source-dataset hash, groups with rationale,
manual links, ambiguous pairs, semantic audit, per-record assignment) and
`recall_splits_v1.csv` (`recall_id,split,group_id`). Re-running with a different seed or
rules is **refused** unless `--force` is given, because annotations are stamped with the
assignment fingerprint.

## 3. Annotation format and workflow

Full field reference: [`data/annotations/README.md`](../data/annotations/README.md).
JSONL, one query per line (placeholders shown — no example queries are shipped):

```json
{"query_id": "<INITIALS>-0001", "query": "<written by a team member>", "split": "validation",
 "relevant": [{"recall_id": <NID>, "grade": 2}, {"recall_id": <NID>}],
 "reviewer_status": "draft", "annotator": "<INITIALS>", "reviewer": "",
 "notes": "", "created_at": "...", "split_fingerprint": "<assignment fingerprint>"}
```

1. **Write** — `python -m evaluation.annotate --annotator AB --out data/annotations/ab.jsonl`
   (one file per person avoids id clashes). Type the query first; the tool then shows
   BM25 + MiniLM candidates **without split labels**; `/find` searches further, `/show ID`
   reads a record; enter `ID` or `ID:GRADE`. All relevant recalls must fall in one split;
   the split and the manifest fingerprint are stamped on save. Status = `draft`.
2. **Review** — a different person exports to CSV, checks every judgment, sets
   `reviewer_status=reviewed` and `reviewer`, and re-imports:
   `python -m evaluation.annotations export data/annotations/ab.jsonl --out ab_review.csv`
   `python -m evaluation.annotations import ab_review.csv --out data/annotations/queries.jsonl --merge`
3. **Validate** — `python -m evaluation.annotations validate` (also runs on import).

Validation **errors**: duplicate `query_id`; empty or > 500-char query; unknown split or
status; duplicate relevance labels; non-integer / out-of-range grade; recall ids not in the
1,000-record corpus; a query split that differs from its anchor recall's split, or an
`anchor_recall_id` that is not among its relevant recalls; `reviewed` without a reviewer
or **reviewed by the annotator**; a `split_fingerprint` from a different split
assignment; conflicting CSV rows; a split manifest whose assignments were edited by hand.
Query-writing guidance: [`QUERY_WRITING_GUIDE.md`](QUERY_WRITING_GUIDE.md); empty
templates: `data/annotations/templates/{validation,test}_queries.csv`.
**Warnings**: missing fingerprint stamp, repeated query text, query nearly copying a
relevant recall's title. Nothing is written when there are errors.

### 3a. Single-annotator mode (one person writes and judges)

Use this when only one team member is available to write and label queries. Nothing is
presented as independently reviewed.

| Status | Who checked the labels | Evaluated |
|---|---|---|
| `draft` | nobody yet | never |
| `reviewed` | a **different** person (`reviewer` ≠ `annotator`) | by default |
| `single_annotator` | only the author; `reviewer` must be **empty** | only with `--include-single-annotator` |
| `rejected` | — | never |

1. **Create** queries as usual (from `ml/`):
   `python -m evaluation.annotate --annotator VD --out ../data/annotations/vd.jsonl`
   Type each query before looking at results, find every relevant recall (`/find`,
   `/show`), and grade it 1–3. Queries are saved as `draft`.
2. **Self-check, then finalize.** Re-read each draft later (export it to CSV to look over
   it if you like) and fix or reject anything doubtful. Then mark your own drafts:
   `python -m evaluation.annotations finalize ../data/annotations/vd.jsonl --annotator VD`
   (add `--ids VD-0001 VD-0002` to finalize only some). The command changes only drafts
   whose `annotator` matches, never sets a reviewer, refuses unknown or other people's
   ids, re-validates, and rewrites the file atomically, leaving all other fields as they were.
3. **Evaluate on validation** (model selection):
   `python -m evaluation.evaluate --split validation --include-single-annotator`
   Without the flag, behaviour is unchanged: only `reviewed` queries are used. The error
   message tells you when single-annotator queries were left out.
4. **Final test** stays sealed: it still needs `--confirm-final-test`, and every run is
   logged to `test_runs.jsonl` together with the annotation composition.

Validation rules: a `single_annotator` query needs `annotator` initials and an empty
`reviewer`. A `reviewed` query whose reviewer is the annotator, however the initials are
spelled (`VD`, `vd`, `V D`), is still an **error**, so a self-check cannot be passed
off as an independent review. Training (`ml/training`) keeps using `reviewed` train
queries only.

**What the report records** (`annotation_composition` in `eval_<split>.json`, also printed):
`review_mode` (`independent` / `single_annotator` / `mixed`),
`independently_reviewed_queries`, `single_annotator_queries`, `total_queries_evaluated`,
`included_statuses`, `excluded_in_split` (e.g. drafts left out), `annotators`, `reviewers`,
and the query ids for each status. Each model also gets `mean_by_review_status`, so a mixed
run never shows only a blended number. Whenever single-annotator queries are included, a
`LIMITATION` warning is added to `warnings`.

**Limitations: why these metrics are preliminary.**
- *No inter-annotator check.* One person's view of what is "relevant" and which grade fits
  is never cross-checked, so agreement (e.g. Cohen's κ) cannot be measured and labelling
  mistakes or personal interpretations go uncaught.
- *Author bias.* The person who wrote the query knows what they meant. Their judgments can
  be more lenient or narrower than a typical consumer's would be.
- *Query style.* All queries reflect one person's wording, so they cover less of the
  variety in how real consumers phrase questions.
- *Familiarity with the systems.* An annotator who has seen BM25/MiniLM results while
  building the project may write queries that suit them, even unintentionally (this adds
  to the pooling bias below).
Report single-annotator results as **preliminary, single-annotator relevance judgments**
alongside the query count. Do not describe them as independently validated. If a teammate
becomes available, have them review a sample (`export` → set `reviewed` + their initials →
`import`) and report both numbers.

**Pooling bias:** candidates come from BM25 + MiniLM, so judgments favour documents those
systems retrieve; unjudged documents count as non-relevant. Before the final Model B
comparison, re-pool: have reviewers judge Model B's top 10 for every query as well.

## 4. Metrics — `evaluation/metrics.py`

For a query with relevant set R (grades g ≥ 1) and a ranked list:

| Metric | Definition |
| --- | --- |
| Recall@5 | \|R ∩ top-5\| / \|R\| — fraction of all relevant recalls retrieved (max 5/\|R\| when \|R\| > 5) |
| MRR@10 | 1 / rank of the first relevant recall in the top 10, else 0 |
| nDCG@10 | DCG@10 / IDCG@10, gain 2^g − 1, discount log2(rank + 1); IDCG from all judged relevant, truncated at 10 |

Means over evaluated queries are reported with `num_queries` (and the review-status
composition, see § 3a); < 30 queries triggers a
"may be noise" warning. Hand-computed unit tests cover binary and graded nDCG (e.g. binary
example = 0.693426), multiple relevant documents, cutoffs and invalid inputs.

## 5. Commands

```bash
python -m evaluation.split_review export                 # refresh the pair review file (keeps decisions)
python -m evaluation.split_review status                 # progress / validation of human decisions
python -m evaluation.split_review apply                  # apply decisions, rebuild + FREEZE the split
python -m evaluation.splits                              # build split manifest (refuses changes without --force; never overwrites a frozen one)
python -m evaluation.annotate --annotator AB --out data/annotations/ab.jsonl
python -m evaluation.annotations validate data/annotations/queries.jsonl
python -m evaluation.annotations export data/annotations/ab.jsonl --out ab_review.csv
python -m evaluation.annotations import ab_review.csv --out data/annotations/queries.jsonl --merge
python -m evaluation.annotations finalize data/annotations/vd.jsonl --annotator VD   # working alone: draft -> single_annotator
python -m evaluation.evaluate --split validation         # BM25 vs MiniLM, same queries + 1,000-doc index
python -m evaluation.evaluate --split validation --include-single-annotator   # also single-annotator queries (reported separately)
python -m evaluation.evaluate --split test --confirm-final-test   # final report only (audit-logged)
```

`evaluate` exits 2 when it cannot run (no reviewed queries, invalid annotations, stale
manifest) and 3 when a requested model is unavailable. Reports:
`data/processed/eval/eval_<split>.json` with per-query rankings and provenance (annotation
hash, split-manifest hash + assignment fingerprint, seed, corpus fingerprint, MiniLM
revision, git commit); final-test runs are appended to `data/processed/eval/test_runs.jsonl`.

## 6. Phase 3 additions — four systems, proposal metrics, proxy diagnostic

### Compared systems (all rank the same fixed 1,000-record index; `evaluation/rankers.py`)
| Label | System |
|---|---|
| `bm25` | Baseline 0 — Okapi BM25 over `search_text` (no training) |
| `minilm` | Model A — pretrained all-MiniLM-L6-v2, stored Phase 2A embeddings |
| `model_b` | Model B — fine-tuned, in-batch negatives only (`--checkpoint model_b=<run>/checkpoints/best`) |
| `model_c` | Model C — fine-tuned with hard negatives (`--checkpoint model_c=<run>/checkpoints/best`) |

```bash
python -m evaluation.evaluate --split validation \
    --checkpoint model_b=ml/artifacts/models/model-b-v1/checkpoints/best \
    --checkpoint model_c=ml/artifacts/models/model-c-v1/checkpoints/best
```
The report adds, per system: `false_match_rate@1`, `identifier_precision@5`, `error_rows`
(per query: first relevant rank + top-3 titles), and `comparisons`
(`model_b_vs_minilm`, `model_c_vs_model_b`, `model_c_vs_bm25`: queries whose first relevant
rank improved / worsened — the hard-negative help/hurt examples — plus
`bootstrap_mrr@10`: mean paired ΔMRR@10 with a 95 % percentile bootstrap CI over queries,
10,000 resamples, seed 4110). The CI covers query sampling only, **not training-seed
variance**; use the multi-seed runs (`python -m reports.seeds`) for that.

### Exact definitions (`evaluation/diagnostics.py`)
| Metric | Numerator | Denominator |
|---|---|---|
| **FMR@1** (false-match rate) | evaluated queries whose rank-1 result is not judged relevant (empty ranking counts) | all evaluated queries |
| **IVP@5** (identifier-verification precision) | top-5 results the verifier marks CONFIRMED_MATCH **and** judged relevant | top-5 results marked CONFIRMED_MATCH, over queries that name an identifier |

Unjudged documents count as non-relevant, so FMR@1 is an upper bound under incomplete
judgments (FMR@1 = 1 − P@1). **IVP is reported as `null` with a reason when its denominator is
0** — never as 0 or 1. Also reported: queries with identifiers, CONFIRMED_MISMATCH count and
how many of those were judged relevant (wrongly rejected).

**Current limitation:** identifiers can be extracted from only 6 of the 1,000 indexed records
(model numbers only; 0 UPC, 0 lot, 0 best-before) because the corpus has the CSV fields, not
the full notice pages the proposal planned to fetch. IVP will be `null` until notice pages
(with UPC/lot tables) are added as evidence and human queries name identifiers.

### Proxy diagnostic (`python -m evaluation.proxy_eval`) — NOT human judgments
Template-generated queries of the 199 **validation** documents (title-copy view excluded;
relevant = source recall + its split group; refuses pairs from other splits), all four
systems, same metrics + FMR@1, broken down by query view. Output
`data/processed/eval/proxy_validation.json`, labelled `PROXY - generated validation queries`.
It is the checkpoint-selection signal and a pipeline check. It is lexically biased (BM25
≥ 0.95 MRR@10 on every view) and the fine-tuned models were trained on the same templates
(different documents), so it **favours BM25 and Models B/C over Model A** and must not be
presented as retrieval quality on consumer queries.

### Test split
Untouched in Phase 3. `evaluate --split test` still requires `--confirm-final-test` and is
audit-logged; `proxy_eval` refuses non-validation pairs; training never builds test pairs.
