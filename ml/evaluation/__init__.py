"""RecallIQ evaluation infrastructure (Phase 2B): grouped document splits, human
relevance-annotation format, retrieval metrics, and a BM25 / MiniLM evaluation runner.

Key distinction: splits partition the 1,000 documents for *training/evaluation pairing*
(which documents may appear in training pairs vs. in validation/test judgments). Retrieval
ALWAYS searches the complete, fixed 1,000-record candidate index. See docs/EVALUATION.md.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO_ROOT / "data" / "processed" / "recalliq.db"
DEFAULT_SPLIT_DIR = REPO_ROOT / "data" / "processed" / "splits"
DEFAULT_MANIFEST = DEFAULT_SPLIT_DIR / "recall_splits_v1.json"
DEFAULT_ANNOTATIONS = REPO_ROOT / "data" / "annotations" / "queries.jsonl"
DEFAULT_REPORT_DIR = REPO_ROOT / "data" / "processed" / "eval"

SPLITS: tuple[str, ...] = ("train", "validation", "test")
