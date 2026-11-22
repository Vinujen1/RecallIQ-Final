# RecallIQ — Models B and C: contrastive fine-tuning of MiniLM (Phase 3)

Code: `ml/training/`. Configs: `ml/configs/model_b.json`, `ml/configs/model_c.json`.
Tests: `tests/training/` (synthetic data and a tiny random BERT only).
Real training requires a **frozen** split manifest (frozen 2026-11-16, assignment `5612e28d…`).
Results live in run folders; tables/figures are generated from them (`python -m reports.figures`)
— numbers are not typed into docs without a source file.

> Earlier draft (2026-11-15) had ONE trainer that always used hard negatives and called it
> "Model B". That was actually the proposal's Model C. Both variants are now explicit.

## The controlled experiment (proposal: Model B vs Model C)

| | Model B | Model C |
|---|---|---|
| Backbone | `sentence-transformers/all-MiniLM-L6-v2` @ pinned revision (~22.7M params, 384-d, mean pooling, L2-norm) | same |
| Loss | Multiple Negatives Ranking (InfoNCE), **in-batch negatives only** | same loss, **+ one hard negative per query** |
| Training (query, positive) pairs | 1,821 pairs, 603 train documents | identical set and order |
| Optimizer / schedule / seed / batch / epochs / early stopping | AdamW lr 2e-5, wd 0.01, 10 % warmup + linear decay, clip 1.0, seed 4110, batch 32, ≤ 4 epochs, patience 2 | identical |
| Validation pairs | 593 pairs, fingerprint `18a6003d…` (always mined with 3 hard negatives) | **byte-identical** |
| Checkpoint selection | proxy MRR@10 (below), epoch 0 included | identical |

The two config files differ **only** in `hard_negatives_per_pair` (0 vs 3) and the label
(tested: `test_shipped_b_and_c_configs_differ_only_in_hard_negatives`).
**Unavoidable difference:** Model C encodes up to 2B candidate documents per step instead of
B, so it costs more compute per step; the number of optimizer steps is the same. Dropout masks
come from the global torch RNG, and C's extra forward passes consume it differently, so the two
runs see different dropout noise despite the same seed (batch order uses a separate seeded
generator and is identical).

### How negatives enter the loss
For a batch of B queries qᵢ with positives pᵢ, candidates are `c = [p₁ … p_B]` (Model B) or
`c = [p₁ … p_B, n₁ … n_B]` (Model C, nᵢ = one of query i's mined negatives, re-sampled each
step from up to 3 with a seeded RNG):

    logitsᵢⱼ = 20 · cos(qᵢ, cⱼ)          (scale 20 = temperature 0.05)
    loss     = mean_i CrossEntropy(logitsᵢ, target = i)

Every other query's positive is a negative for qᵢ (in-batch); in Model C every hard negative
nⱼ — including other queries' hard negatives — also sits in qᵢ's softmax denominator.
Candidates from the same split group as pᵢ (same document via another query view, same
product, re-notice or colocated related event) are masked to −∞, so they are never pushed
away. Code: `training/trainer.py::contrastive_loss`, `_batch_loss`.

### Hard-negative mining and filtering (Model C; train documents only)
1. **Human-verified first:** recalls a person judged `DISTINCT` from the positive in the split
   review (`data/reviews/split_pair_review.csv`; 21 pairs have both recalls in train) —
   123 training pairs start with such a negative.
2. Then BM25 top-10 for the query interleaved with the positive's 10 nearest pretrained-MiniLM
   neighbours (same-brand / same-category / same-hazard look-alikes in practice).
3. Excluded: the positive, any judged-relevant recall, anything in the positive's split group
   (SAME_PRODUCT / colocated RELATED_EVENT / title group / brand group), and candidates with
   product-key Jaccard ≥ 0.5 to the positive (possible re-notice = false negative).
Mined negatives from step 2 are **filter-verified, not human-verified**. Residual
false-negative risk: e.g. the same food recalled in different years with different wording.

## Training examples (weak supervision — never evaluation data)
Generated deterministically from each train document's own fields (no LLM), labelled `pseudo:*`:

| view | shape | purpose |
|---|---|---|
| `title_core` | title minus "Archived -", "Expanded recall:", "recalled due to …"; "`<Co.>` recalls `<X>`" → `<X>` | anchor |
| `generic_issue` | title core without brand + official issue | brand-free |
| `product_issue` | product field + issue | |
| `consumer_question` | brand dropped, product shortened, issue in everyday words (`HAZARD_PLAIN`, e.g. *Extraneous Material → found pieces of plastic or metal in it*), one of 5 question frames chosen by a stable hash | consumer wording |
| `brand_product` | brand + last 3 product words | brand-partial queries |

Guards: train pairs use train documents only (`assert_no_leakage`); building pairs for `test`
raises; train pairs whose normalised text equals any validation pair or any human
validation/test query are dropped (4 dropped on the real data); reviewed human `train`
queries are added when they exist (none yet).

**Why this is not enough:** the templates reuse title/product tokens, so they are lexically
easy (BM25 MRR@10 ≥ 0.95 on every generated validation view). They teach domain vocabulary,
but cannot show whether a model generalises to how consumers write. Only reviewed human
queries can (`docs/QUERY_WRITING_GUIDE.md`).

## Validation every epoch and checkpoint selection
Logged per epoch (epoch 0 = the untouched pretrained model) in `history.json`: mean training
loss; validation loss **with** one hard negative and **in-batch only**, on the same validation
pairs; proxy Recall@5 / MRR@10 / nDCG@10 of the validation generated queries (`title_core`
excluded; relevant = source recall + its split group) ranked against the **full 1,000-record
index**. `checkpoints/best` = highest proxy MRR@10 — chosen because it is neutral between B and
C, whereas each loss favours the variant trained on it. Early stopping after 2 epochs without
improvement. The proxy is near its ceiling; switch selection to reviewed human validation
queries once ≥ 30 exist.

## Commands (from `ml/`)
```bash
python -m training.train --config configs/model_b.json --dry-run           # pair statistics
python -m training.train --config configs/model_b.json --run-name smoke-b --max-steps 6
python -m training.train --config configs/model_b.json --run-name model-b-v1
python -m training.train --config configs/model_c.json --run-name model-c-v1
python -m evaluation.proxy_eval --pairs artifacts/models/model-b-v1/validation_pairs.jsonl \
    --checkpoint model_b=artifacts/models/model-b-v1/checkpoints/best \
    --checkpoint model_c=artifacts/models/model-c-v1/checkpoints/best
python -m reports.figures --run model_b=artifacts/models/model-b-v1 \
    --run model_c=artifacts/models/model-c-v1 \
    --proxy ../data/processed/eval/proxy_validation.json --out artifacts/reports
```
Run folder (git-ignored; weights ~90 MB per checkpoint): `config.json`, `run_info.json`
(environment, hardware, git commit + dirty flag, split/corpus/pair fingerprints, loss),
`train_pairs.jsonl`, `validation_pairs.jsonl`, `history.json`, `summary.json`,
`checkpoints/{best,last}/`. Lightweight copies (no weights) are kept in `experiments/`.

## Compute
CPU only: PyTorch 2.15.0+cpu, AMD Ryzen 5 2600 (6 torch threads), 16 GB RAM. A GTX 1660 (6 GB)
is present but the installed PyTorch has no CUDA build; not needed at this scale. Measured
epoch times are in `run_info.json` / `history.json`. Colab alternative: clone, `pip install -e
ml -r ml/requirements.txt`, copy `data/processed/` (SQLite DB + frozen manifest) and
`ml/artifacts/embeddings/`, then run the same commands.

## Known limitations
- Generated training queries are lexically close to the notices (see above).
- **Resolved for v2:** v1 loss masking was by split group only. Different recalls can receive
  identical generated text (v1: 3 texts, 6 of 1,821 train pairs, e.g. "cheddar cheese
  listeria"); in one batch each positive became a false in-batch negative for the other.
  Fix: config flag `mask_duplicate_query_texts` (`trainer.duplicate_text_mask`), **false in
  the v1 configs** so `model-b-v1` / `model-c-v1` stay reproducible and keep their labels,
  **true in all `*_v2_seed*.json` configs**. Measured effect at seed 4110 (Model B v1 vs v2):
  identical best epoch (3) and proxy MRR@10 (0.9743); validation loss differs only in the 4th
  decimal. Regression tests: `tests/training/test_model_variants.py`.

## Multi-seed experiment (v2: same-text masking on; seeds 4110, 4111, 4112)
```bash
# from ml/ — six runs, ~15 min (B) / ~21 min (C) each on this CPU when the machine is idle
for s in 4110 4111 4112; do for m in b c; do
  python -m training.train --config configs/model_${m}_v2_seed${s}.json --run-name model-${m}-v2-s${s}
done; done
python -m reports.seeds --runs-dir artifacts/models --prefix v2 --out ../experiments/reports/seeds.md
```
Each B/C pair differs only in `hard_negatives_per_pair` (tested). Results:
`experiments/reports/seeds.md` (generated). Single-seed evidence (v1) is **not sufficient**
to claim a B-vs-C difference.
- Hard negatives beyond the 21 human-reviewed train pairs are filter-verified only.
- Multi-seed result (v2, completed 2026-11-16; `experiments/reports/seeds.md`, PROXY):
  mean MRR@10 B 0.9731 (sd 0.0013) vs C 0.9754 (sd 0.0010); C − B per seed +0.0022,
  +0.0036, +0.0011 (mean +0.0023, sd 0.0013). C was higher on all three seeds, but the
  gap is small (~1.8 sd of the paired difference with n = 3) on a proxy near its ceiling:
  a consistent direction, not evidence of a meaningful improvement. Recall@5 per-seed
  differences change sign (−0.0026, +0.0064, +0.0026).
- Runs were made from an uncommitted working tree (`git_dirty: true` in run_info.json);
  re-run from the tagged commit for the final submission.
