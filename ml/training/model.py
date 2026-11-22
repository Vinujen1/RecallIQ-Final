"""Bi-encoder in plain PyTorch + transformers: mean pooling over token states, L2-normalised.

This is the all-MiniLM-L6-v2 architecture (BERT encoder -> mean pooling -> normalise), so
the pretrained weights load unchanged and the fine-tuned model is used exactly like the
pretrained one at search time (cosine similarity of unit vectors).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

ENCODER_META = "recalliq_encoder.json"


class MeanPoolEncoder(nn.Module):
    """Text -> one unit-length vector per text (the shared bi-encoder for queries and notices).

    Queries and recall notices go through the SAME network ("bi-encoder"), so notices can be
    encoded once offline and a query is ranked by a single matrix product at search time.
    `encode` is the protocol retrieval uses; `forward` keeps gradients for training.
    """

    def __init__(self, model: nn.Module, tokenizer, max_seq_length: int = 128) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    @classmethod
    def from_pretrained(cls, name_or_path: str | Path, revision: str | None = None,
                        max_seq_length: int = 128) -> "MeanPoolEncoder":
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(name_or_path), revision=revision)
        model = AutoModel.from_pretrained(str(name_or_path), revision=revision)
        return cls(model, tokenizer, max_seq_length)

    @classmethod
    def load(cls, directory: Path) -> "MeanPoolEncoder":
        """Reload a checkpoint written by `save` (weights, tokenizer, pooling settings)."""
        meta = json.loads((directory / ENCODER_META).read_text("utf-8"))
        return cls.from_pretrained(directory, max_seq_length=meta["max_seq_length"])

    def save(self, directory: Path, extra: dict | None = None) -> None:
        """Write a self-describing checkpoint. The pooling/normalisation settings are stored
        next to the weights so an index is never built with a mismatched encoder."""
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        meta = {"pooling": "mean", "normalize": True, "max_seq_length": self.max_seq_length,
                **(extra or {})}
        (directory / ENCODER_META).write_text(json.dumps(meta, indent=2) + "\n", "utf-8")

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(self, texts: list[str]) -> torch.Tensor:
        """(B,) texts -> (B, H) unit vectors, differentiable (used by the trainer).

        Mean pooling must ignore padding: texts in a batch are padded to the longest one, and
        averaging the padding positions would make a text's vector depend on its batch-mates.
        The attention mask zeroes those positions and the sum is divided by the number of
        real tokens. L2 normalisation makes the dot product equal to cosine similarity, which
        is what both the loss and the search index use."""
        batch = self.tokenizer(texts, padding=True, truncation=True,
                               max_length=self.max_seq_length, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        states = self.model(**batch).last_hidden_state                  # (B, T, H)
        mask = batch["attention_mask"].unsqueeze(-1).to(states.dtype)    # (B, T, 1)
        pooled = (states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)    # (B, H)
        return nn.functional.normalize(pooled, p=2, dim=-1)

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encoder protocol used by retrieval (unit vectors, float32)."""
        was_training = self.training
        self.eval()
        try:
            chunks = [self(texts[i:i + batch_size]).cpu().numpy()
                      for i in range(0, len(texts), batch_size)]
        finally:
            self.train(was_training)
        dim = self.model.config.hidden_size
        return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros((0, dim), np.float32)
