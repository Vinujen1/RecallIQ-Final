"""Fine-tuned MiniLM checkpoints (Models B and C) as search models.

Each served model = a checkpoint directory (training output, `checkpoints/best`) + an
embedding index built WITH THAT EXACT CHECKPOINT (same on-disk format as Phase 2A):

    python -m retrieval.finetuned --label minilm_finetuned \\
        --checkpoint ml/artifacts/models/model-b-v1/checkpoints/best \\
        --out ml/artifacts/embeddings/model-b-v1

metadata.json records the checkpoint's weight SHA-256, dimension and corpus fingerprint;
`check_compatible` refuses an index whose weights hash, dimension or corpus does not match,
so queries and documents are always encoded by the same network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from retrieval.corpus import SEARCH_TEXT_VERSION, corpus_fingerprint, load_corpus
from retrieval.semantic import (
    EmbeddingIndex,
    ModelUnavailableError,
    StaleIndexError,
    save_embeddings,
)

FINETUNED_MODELS: dict[str, str] = {
    "minilm_finetuned": "Model B - MiniLM fine-tuned with MNRL, in-batch negatives only",
    "minilm_hard_negative": "Model C - MiniLM fine-tuned with MNRL + hard negatives",
}
WEIGHTS_FILE = "model.safetensors"
ENCODER_META = "recalliq_encoder.json"


def checkpoint_sha256(directory: Path) -> str:
    """Hash of the weights file: identifies the exact network that produced an index."""
    weights = directory / WEIGHTS_FILE
    if not weights.is_file():
        raise FileNotFoundError(f"No {WEIGHTS_FILE} in checkpoint {directory}")
    digest = hashlib.sha256()
    with weights.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_dimension(directory: Path) -> int:
    config = json.loads((directory / "config.json").read_text("utf-8"))
    return int(config["hidden_size"])


def check_compatible(index: EmbeddingIndex, checkpoint: Path) -> None:
    """Index must have been built by this checkpoint (weights hash) with matching size."""
    for name in (WEIGHTS_FILE, "config.json", ENCODER_META):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"Checkpoint incomplete: {checkpoint / name} missing")
    expected = index.metadata.get("checkpoint_sha256")
    if not expected:
        raise StaleIndexError("Embedding metadata has no checkpoint_sha256 (not a fine-tuned index)")
    if expected != checkpoint_sha256(checkpoint):
        raise StaleIndexError(
            f"Embeddings were built with a different checkpoint than {checkpoint}; "
            "rebuild them with `python -m retrieval.finetuned`")
    if index.dimension != checkpoint_dimension(checkpoint):
        raise StaleIndexError(
            f"Index dimension {index.dimension} != checkpoint dimension "
            f"{checkpoint_dimension(checkpoint)}")


class CheckpointEncoder:
    """Lazily loads a fine-tuned MeanPoolEncoder (CPU); fails fast after a load error."""

    RETRY_AFTER_SECONDS = 60.0

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._model = None
        self._lock = threading.Lock()
        self._failure: tuple[float, str] | None = None

    def _load(self):
        with self._lock:
            if self._model is None and self._failure is not None:
                failed_at, message = self._failure
                if time.monotonic() - failed_at < self.RETRY_AFTER_SECONDS:
                    raise ModelUnavailableError(message)
            if self._model is None:
                try:
                    from training.model import MeanPoolEncoder

                    self._model = MeanPoolEncoder.load(self.directory)
                    self._model.eval()
                except Exception as error:  # noqa: BLE001 - safetensors/torch/transformers
                    # errors (corrupt file, size mismatch, MemoryError...) must become a 503
                    # with a cached reason, never an opaque 500 retried on every request
                    message = f"Could not load checkpoint {self.directory}: {error}"
                    self._failure = (time.monotonic(), message)
                    raise ModelUnavailableError(message) from error
        return self._model

    def load(self) -> "CheckpointEncoder":
        """Load now (the server does this right after the weights-hash check)."""
        self._load()
        return self

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        model = self._load()
        try:
            return model.encode(texts, batch_size=batch_size)
        except Exception as error:  # noqa: BLE001 - inference failure -> 503, not 500
            raise ModelUnavailableError(f"Encoding with {self.directory} failed: {error}") from error


def build_index(db_path: Path, checkpoint: Path, out_dir: Path, label: str) -> dict:
    import torch
    import transformers

    corpus = load_corpus(db_path)
    matrix = CheckpointEncoder(checkpoint).encode([r.search_text for r in corpus])
    meta = json.loads((checkpoint / ENCODER_META).read_text("utf-8"))
    norms = np.linalg.norm(matrix, axis=1)
    metadata = {
        "model_id": f"recalliq:{label}",
        "label": label,
        "description": FINETUNED_MODELS.get(label, ""),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "checkpoint_epoch": meta.get("epoch"),
        "base_model": meta.get("base_model"),
        "base_revision": meta.get("base_revision"),
        "pooling": meta.get("pooling"),
        "max_seq_length": meta.get("max_seq_length"),
        "dimension": int(matrix.shape[1]),
        "count": int(matrix.shape[0]),
        "dtype": str(matrix.dtype),
        "normalized": True,
        "similarity": "cosine",
        "search_text_version": SEARCH_TEXT_VERSION,
        "corpus_fingerprint": corpus_fingerprint(corpus),
        "source_database": db_path.name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "device": "cpu"},
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
    }
    save_embeddings(out_dir, [r.recall_id for r in corpus], matrix, metadata)
    return metadata


def main(argv: list[str] | None = None) -> None:
    from retrieval.embed import DEFAULT_DB

    parser = argparse.ArgumentParser(description="Embed the corpus with a fine-tuned checkpoint.")
    parser.add_argument("--label", required=True, choices=sorted(FINETUNED_MODELS))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args(argv)
    print(json.dumps(build_index(args.db, args.checkpoint, args.out, args.label), indent=2))


if __name__ == "__main__":
    main()
