"""Compute and save pretrained MiniLM embeddings for every recall in the SQLite corpus.

Usage (with the project venv active, from any directory):
    python -m retrieval.embed [--db PATH] [--out DIR] [--revision REV] [--batch-size 64]

The model revision is resolved to an exact Hugging Face commit hash and recorded, and the
backend loads that same revision to encode queries.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

from retrieval import MINILM_MODEL_ID, MINILM_REVISION
from retrieval.corpus import SEARCH_TEXT_VERSION, corpus_fingerprint, load_corpus
from retrieval.semantic import MiniLMEncoder, save_embeddings

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO_ROOT / "data" / "processed" / "recalliq.db"
DEFAULT_OUT = REPO_ROOT / "ml" / "artifacts" / "embeddings" / "all-MiniLM-L6-v2"


# Only what the PyTorch backend needs (~92 MB). Without this filter snapshot_download also
# fetches ONNX/OpenVINO/TF/Rust exports (~977 MB total for this repo).
MODEL_FILE_PATTERNS = ["*.json", "*.txt", "model.safetensors", "1_Pooling/*"]


def resolve_revision(model_id: str, revision: str) -> str:
    """Return the exact commit hash for `revision` (downloads the needed files if absent)."""
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(model_id, revision=revision, allow_patterns=MODEL_FILE_PATTERNS)
    )
    return path.name  # snapshot folders are named by commit hash


def run(db_path: Path, out_dir: Path, revision: str, batch_size: int) -> dict:
    corpus = load_corpus(db_path)
    commit = resolve_revision(MINILM_MODEL_ID, revision)
    encoder = MiniLMEncoder(revision=commit)

    matrix = encoder.encode([record.search_text for record in corpus], batch_size=batch_size)
    norms = np.linalg.norm(matrix, axis=1)

    metadata = {
        "model_id": MINILM_MODEL_ID,
        "model_revision": commit,
        "dimension": int(matrix.shape[1]),
        "count": int(matrix.shape[0]),
        "dtype": str(matrix.dtype),
        "normalized": True,
        "similarity": "cosine",
        "search_text_version": SEARCH_TEXT_VERSION,
        "corpus_fingerprint": corpus_fingerprint(corpus),
        "source_database": db_path.name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "sentence_transformers": version("sentence-transformers"),
            "torch": version("torch"),
            "device": "cpu",
        },
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
    }
    save_embeddings(out_dir, [r.recall_id for r in corpus], matrix, metadata)
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build MiniLM embeddings for RecallIQ.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--revision", default=MINILM_REVISION, help="HF commit (default: pinned Phase 2A commit)"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    metadata = run(args.db, args.out, args.revision, args.batch_size)
    print(json.dumps(metadata, indent=2))
    print(f"\nEmbeddings written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
