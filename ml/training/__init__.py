"""RecallIQ Phase 3: contrastive fine-tuning of MiniLM ("Model B").

Leakage rules enforced in code (see docs/TRAINING.md):
- training pairs use ONLY train-split documents (positives and negatives);
- validation loss uses ONLY validation-split documents;
- the test split is never read here;
- a frozen split manifest is required for real data.
"""
