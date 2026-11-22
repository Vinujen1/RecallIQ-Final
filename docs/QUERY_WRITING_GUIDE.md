# Writing evaluation queries — team guide

RecallIQ is evaluated on questions that real Canadians might ask about product safety.
Your queries are the ground truth for every reported number, so they must come from
**you**, not from recall titles, search results or AI tools.

Before you start: the split manifest must be **frozen** (see `data/reviews/README.md`).
The annotation tool warns if it is not.

## 1. Think like a consumer, not a database
Picture a person with a product in their kitchen, closet or garage. They know what they
bought and what worries them; they rarely know the brand spelling, the lot number or the
regulatory wording.

| Usually known to a consumer | Usually NOT known |
| --- | --- |
| product type ("baby monitor", "frozen dumplings") | official recall title wording |
| roughly where/when bought | lot codes, UPCs, NIDs |
| symptom or worry ("smells like burning", "my kid is allergic to eggs") | hazard taxonomy ("undeclared allergen", "Class 1") |
| sometimes the brand, often misspelled or partial | the recalling company's legal name |

## 2. Vary the query styles
Aim for a mix across the team (rough targets, not quotas):

| Style | Pattern (fill in yourself) |
| --- | --- |
| Product + worry | "[product] [concern]" |
| Question | "is the [product] I bought [problem]?" |
| Symptom-led | "[product] [what happened]" |
| Allergy / dietary | "[allergen] in [food type]" |
| Brand-partial | "[partial or misspelled brand] [product type]" |
| Child safety | "[child product] [risk a parent fears]" |
| Vague | "[broad category] recall [season/place]" |

Write the pattern in your own words; do not copy these brackets literally. Mix short
(2–4 words) and longer natural questions. English only for now.

## 3. Do / Don't

**Do**
- Write the query **before** looking at any results (the tool enforces this order).
- Write about products and worries people actually have; your own experience is fine.
- Mark **every** relevant recall you can find — use `/find` with different words, not only
  the suggested candidates. Missing relevant recalls makes every model look worse.
- Use grades: 3 = exactly what the person means, 2 = clearly relevant, 1 = somewhat relevant.
- Put the recall the person most likely means **first** (or grade it highest): it becomes
  the query's *anchor* and decides the query's split.

**Don't**
- Don't copy or lightly edit a recall title (the validator warns on near-copies).
- Don't write queries by reading through the recall list and picking one.
- Don't use ChatGPT/Claude or any AI tool to write or rephrase queries.
- Don't use the `product_family` field as a guide (458 values are unverified).
- Don't write a query just because you know a model gets it right or wrong.
- Don't reuse the smoke-test queries from `ml/retrieval/smoke.py`.

## 4. Validation vs. test queries
- The tool assigns the split from the anchor recall; you do not pick it.
- **Test queries are sealed.** Whoever writes or reviews test queries should not look at
  model results on them, and nobody tunes anything on them. Evaluate test once, at the
  end (`--confirm-final-test`; every run is logged).
- Validation queries are for comparing and choosing models during development.
- Rough target before reporting metrics: ≥ 30 reviewed queries per split (fewer triggers a
  "may be noise" warning), spread over food and consumer products.

## 5. Workflow
```bash
# 1. write (one file per person)
python -m evaluation.annotate --annotator AB --out data/annotations/ab.jsonl
# 2. a DIFFERENT person reviews: export, check every judgment, set reviewer_status=reviewed + reviewer
python -m evaluation.annotations export data/annotations/ab.jsonl --out ab_review.csv
python -m evaluation.annotations import ab_review.csv --out data/annotations/queries.jsonl --merge
# 3. check
python -m evaluation.annotations validate data/annotations/queries.jsonl
```
Spreadsheet alternative: fill `data/annotations/templates/validation_queries.csv` or
`test_queries.csv` (one row per query × relevant recall) and import them the same way.
The importer rejects errors and writes nothing until they are fixed.
