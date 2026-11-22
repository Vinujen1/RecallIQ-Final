"""RecallIQ retrieval: BM25 (lexical) and pretrained MiniLM (semantic) search over the
1,000-record recall corpus built by recall_data.

No fine-tuning happens here (Phase 2A). See docs/RETRIEVAL.md.
"""

MINILM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# Exact Hugging Face commit used for Phase 2A embeddings (resolved 2026-11-15). Pinning in
# code makes rebuilds on a fresh clone use the same weights and tokenizer.
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
