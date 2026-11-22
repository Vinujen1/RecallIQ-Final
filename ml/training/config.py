"""Training configuration: one JSON-serialisable dataclass, saved with every run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from retrieval import MINILM_MODEL_ID, MINILM_REVISION


PSEUDO_QUERY_VIEWS = ("title_core", "generic_issue", "product_issue",
                      "consumer_question", "brand_product")
SELECTION_METRICS = ("proxy_mrr@10", "validation_loss")


@dataclass(frozen=True)
class TrainingConfig:
    base_model: str = MINILM_MODEL_ID
    base_revision: str | None = MINILM_REVISION
    seed: int = 4110
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    # MultipleNegativesRanking / InfoNCE: logits = scale * cosine (scale 20 = temperature 0.05)
    scale: float = 20.0
    max_seq_length: int = 128
    # hard negatives
    hard_negatives_per_pair: int = 3
    bm25_candidates: int = 10
    semantic_candidates: int = 10
    # a candidate negative this lexically close to the positive may be a false negative
    false_negative_jaccard: float = 0.5
    # pseudo-query views generated from document fields (see data.pseudo_queries)
    pseudo_query_views: tuple[str, ...] = ("title_core", "generic_issue", "product_issue")
    include_human_train_queries: bool = True
    # Human split-review DISTINCT pairs (both in the pair's split) are used as the first
    # hard negatives: verified non-matches. Ignored when hard_negatives_per_pair == 0.
    use_reviewed_distinct_negatives: bool = True
    # Validation pairs are ALWAYS mined with this many hard negatives, whatever the training
    # variant, so Model B and Model C are validated on byte-identical validation pairs.
    validation_hard_negatives: int = 3
    # Checkpoint selection: "proxy_mrr@10" = MRR@10 of validation pseudo-queries against the
    # full 1,000-record index (generated, NOT human judgments); "validation_loss" = legacy.
    selection_metric: str = "proxy_mrr@10"
    # stop after this many epochs without improving the selection metric (0 = never)
    early_stopping_patience: int = 0
    # Also mask in-batch positives of OTHER queries with identical (normalised) text.
    # False for the v1 runs (reproducibility); True from v2 on (fix, 2026-11-16).
    mask_duplicate_query_texts: bool = False
    # free-text label recorded in run_info.json, e.g. "Model B (in-batch negatives only)"
    experiment_label: str = ""

    def to_json(self) -> dict:
        data = asdict(self)
        data["pseudo_query_views"] = list(self.pseudo_query_views)
        return data

    @classmethod
    def from_json(cls, data: dict) -> "TrainingConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown training config keys: {sorted(unknown)}")
        if "pseudo_query_views" in data:
            data = {**data, "pseudo_query_views": tuple(data["pseudo_query_views"])}
        config = cls(**data)
        config.validate()
        return config

    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        return cls.from_json(json.loads(path.read_text("utf-8")))

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 2:
            raise ValueError("epochs must be >= 1 and batch_size >= 2 (in-batch negatives)")
        if not 0 < self.learning_rate < 1 or self.scale <= 0:
            raise ValueError("learning_rate must be in (0, 1) and scale > 0")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.hard_negatives_per_pair < 0 or self.validation_hard_negatives < 0:
            raise ValueError("hard_negatives_per_pair and validation_hard_negatives must be >= 0")
        if self.selection_metric not in SELECTION_METRICS:
            raise ValueError(f"selection_metric must be one of {SELECTION_METRICS}")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be >= 0")
        unknown = set(self.pseudo_query_views) - set(PSEUDO_QUERY_VIEWS)
        if unknown:
            raise ValueError(f"unknown pseudo-query views: {sorted(unknown)}")
