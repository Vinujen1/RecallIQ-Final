"""Contrastive fine-tuning loop (PyTorch) with in-batch + hard negatives.

Loss (MultipleNegativesRanking / InfoNCE): for a batch of B queries q_i with positives p_i
and one hard negative per query n_i (when available),

    logits_ij = scale * cos(q_i, c_j),   c = [p_1..p_B, n_1..n_M]
    loss      = mean_i CrossEntropy(logits_i, target = i)

Candidates from the same split group as q_i's positive (same document, product, event or
title group) other than p_i itself are masked out, so a second view of the same recall —
or a sibling recall — in the batch is not punished as a negative.
"""

from __future__ import annotations

import json
import math
import platform
import random
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from training.config import TrainingConfig
from training.data import TrainingPair, pairs_fingerprint
from training.model import MeanPoolEncoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def contrastive_loss(
    queries: torch.Tensor, candidates: torch.Tensor, query_groups: list[str],
    candidate_groups: list[str], scale: float, extra_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """queries (B, H), candidates (B + M, H): the first B candidates are the positives.

    `extra_mask` (B, B + M, bool): further candidates to exclude from qi's softmax (e.g.
    positives of other queries with the same text); the diagonal is never masked."""
    batch = queries.shape[0]
    logits = scale * queries @ candidates.T                                  # (B, B + M)
    same = torch.tensor([[qg == cg for cg in candidate_groups] for qg in query_groups],
                        device=logits.device)
    if extra_mask is not None:
        same = same | extra_mask.to(logits.device)
    same[torch.arange(batch), torch.arange(batch)] = False                   # keep own positive
    logits = logits.masked_fill(same, float("-inf"))
    return torch.nn.functional.cross_entropy(logits, torch.arange(batch, device=logits.device))


@dataclass
class TrainingResult:
    history: dict = field(default_factory=dict)
    best_epoch: int | None = None
    best_validation_loss: float | None = None
    run_dir: Path | None = None


def batches_per_epoch(num_pairs: int, batch_size: int) -> int:
    """Exactly the number of batches `_batches` yields (a trailing batch of 1 is skipped)."""
    full, rest = divmod(num_pairs, batch_size)
    return full + (1 if rest >= 2 else 0)


def _batches(pairs: list[TrainingPair], batch_size: int, generator: torch.Generator | None):
    order = (torch.randperm(len(pairs), generator=generator).tolist() if generator is not None
             else list(range(len(pairs))))
    for start in range(0, len(order), batch_size):
        chunk = [pairs[i] for i in order[start:start + batch_size]]
        if len(chunk) >= 2:  # a single query has no in-batch negatives
            yield chunk


def duplicate_text_mask(queries: list[str], num_candidates: int) -> torch.Tensor:
    """(B, num_candidates) bool: True where candidate j < B is the positive of ANOTHER query
    with the same normalised text as query i (identical generated text for two recalls
    would otherwise make each positive a false in-batch negative for the other)."""
    from training.data import normalize_query

    norm = [normalize_query(q) for q in queries]
    mask = torch.zeros((len(queries), num_candidates), dtype=torch.bool)
    for i, a in enumerate(norm):
        for j, b in enumerate(norm):
            if i != j and a == b:
                mask[i, j] = True
    return mask


def _batch_loss(model, chunk, texts, group_of, scale, rng: random.Random | None,
                mask_duplicate_texts: bool = False) -> torch.Tensor:
    negatives = []
    for p in chunk:
        if p.negative_ids:
            negatives.append(rng.choice(p.negative_ids) if rng else p.negative_ids[0])
    candidate_ids = [p.positive_id for p in chunk] + negatives
    q = model([p.query for p in chunk])
    c = model([texts[rid] for rid in candidate_ids])
    extra = (duplicate_text_mask([p.query for p in chunk], len(candidate_ids))
             if mask_duplicate_texts else None)
    return contrastive_loss(q, c, [group_of[p.positive_id] for p in chunk],
                            [group_of[rid] for rid in candidate_ids], scale, extra)


@torch.no_grad()
def evaluate_loss(model, pairs, texts, group_of, config: TrainingConfig,
                  hard_negatives: bool = True) -> float:
    """Deterministic validation loss: fixed order, first hard negative (or none when
    `hard_negatives` is False: in-batch negatives only)."""
    model.eval()
    if not hard_negatives:
        pairs = [TrainingPair(p.query, p.positive_id, (), p.source, p.query_id) for p in pairs]
    losses, weights = [], []
    for chunk in _batches(pairs, config.batch_size, None):
        losses.append(_batch_loss(model, chunk, texts, group_of, config.scale, None,
                                  config.mask_duplicate_query_texts).item())
        weights.append(len(chunk))
    model.train()
    return float(np.average(losses, weights=weights)) if losses else float("nan")


ProxyQuery = tuple[str, str, dict[int, int]]  # (query_id, text, {recall_id: grade})


@torch.no_grad()
def proxy_retrieval(model, index_texts: dict[int, str], queries: list[ProxyQuery],
                    depth: int = 10) -> dict:
    """Rank the FULL index for each (generated) validation query; return mean metrics.

    These queries are generated from validation documents, not written or judged by
    people: the numbers are a model-selection signal only, never a reported result.
    """
    from evaluation.metrics import evaluate_run

    ids = sorted(index_texts)
    docs = torch.from_numpy(model.encode([index_texts[i] for i in ids]))
    q = torch.from_numpy(model.encode([text for _, text, _ in queries]))
    scores = q @ docs.T
    order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :depth].tolist()
    run = {qid: [ids[j] for j in row] for (qid, _, _), row in zip(queries, order)}
    return evaluate_run(run, {qid: rel for qid, _, rel in queries})["mean"]


def save_checkpoint(model: MeanPoolEncoder, dest: Path, extra: dict, attempts: int = 5) -> None:
    """Write into a fresh temp dir, then swap it in. Overwriting model files in place fails
    on Windows when a scanner still holds the previous file; a failed save never leaves a
    half-written checkpoint at `dest`."""
    import shutil

    tmp = dest.with_name(dest.name + ".tmp")
    old = dest.with_name(dest.name + ".old")
    for path in (tmp, old):
        shutil.rmtree(path, ignore_errors=True)
    model.save(tmp, extra)
    for attempt in range(attempts):
        try:
            if dest.exists():
                dest.rename(old)
            tmp.rename(dest)
            break
        except PermissionError:
            if old.exists() and not dest.exists():
                old.rename(dest)  # never lose the previous checkpoint
            if attempt == attempts - 1:
                raise
            time.sleep(0.5 * (attempt + 1))
    shutil.rmtree(old, ignore_errors=True)


def _environment() -> dict:
    try:
        root = Path(__file__).resolve().parents[2]
        git = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True,  # noqa: E731
                                        text=True, check=False, timeout=10).stdout.strip()
        commit = git("rev-parse", "HEAD") or None
        dirty = bool(git("status", "--porcelain")) if commit else None
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    import transformers

    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "platform": platform.platform(),
            "git_commit": commit, "git_dirty": dirty}


def _hardware(device) -> dict:
    import os

    info = {"device": str(device), "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(), "processor": platform.processor(),
            "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def train(
    model: MeanPoolEncoder,
    train_pairs: list[TrainingPair],
    validation_pairs: list[TrainingPair],
    texts: dict[int, str],
    group_of: dict[int, str],
    config: TrainingConfig,
    run_dir: Path,
    data_provenance: dict | None = None,
    max_steps: int | None = None,
    proxy_queries: list[ProxyQuery] | None = None,
) -> TrainingResult:
    """Fine-tune `model` in place; writes config, history and best/last checkpoints.

    Every epoch (and epoch 0 = the untouched base model) records the validation loss with
    and without hard negatives on the SAME validation pairs and, when `proxy_queries` are
    given, full-index retrieval metrics on them. `texts` must then hold the full index.
    Checkpoint selection follows `config.selection_metric`; without proxy queries it falls
    back to the validation loss and says so in run_info.json / summary.json.
    """
    from transformers import get_linear_schedule_with_warmup

    config.validate()
    set_seed(config.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(config.seed)
    rng = random.Random(config.seed)
    selection = config.selection_metric
    if selection == "proxy_mrr@10" and not proxy_queries:
        selection = "validation_loss"

    steps_per_epoch = batches_per_epoch(len(train_pairs), config.batch_size)
    if steps_per_epoch == 0:
        raise ValueError("not enough training pairs for one batch")
    total_steps = steps_per_epoch * config.epochs
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, math.ceil(config.warmup_ratio * total_steps), total_steps)

    hard = config.hard_negatives_per_pair > 0
    info = {"experiment_label": config.experiment_label,
            "config": config.to_json(), "environment": _environment(),
            "hardware": _hardware(model.device),
            "data": {**(data_provenance or {}),
                     "train_pairs": len(train_pairs), "validation_pairs": len(validation_pairs),
                     "train_pairs_with_hard_negatives": sum(1 for p in train_pairs if p.negative_ids),
                     "proxy_validation_queries": len(proxy_queries or []),
                     "train_pairs_fingerprint": pairs_fingerprint(train_pairs),
                     "validation_pairs_fingerprint": pairs_fingerprint(validation_pairs)},
            "device": str(model.device), "max_steps": max_steps,
            "scheduled_steps": total_steps, "truncated_smoke_run": max_steps is not None,
            "selection_metric_requested": config.selection_metric,
            "selection_metric_used": selection,
            "model_selection": ("highest proxy MRR@10 (validation-split GENERATED queries vs the "
                                "full index)" if selection == "proxy_mrr@10" else
                                "lowest validation loss (in-batch + hard negative) on "
                                "validation-split PSEUDO pairs") +
                               "; epoch 0 (base model) included; not evidence of retrieval "
                               "quality on consumer queries",
            "loss": ("MultipleNegativesRanking (InfoNCE): in-batch negatives + 1 hard negative "
                     "per query (sampled each step)" if hard else
                     "MultipleNegativesRanking (InfoNCE): in-batch negatives only") +
                    f", scale {config.scale}, same-group candidates masked"
                    + (", same-text positives masked" if config.mask_duplicate_query_texts else "")}
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n", "utf-8")

    def validate_epoch() -> dict:
        started = time.monotonic()
        row = {"validation_loss": evaluate_loss(model, validation_pairs, texts, group_of, config),
               "validation_loss_in_batch_only": evaluate_loss(
                   model, validation_pairs, texts, group_of, config, hard_negatives=False)}
        if proxy_queries:
            row["proxy"] = proxy_retrieval(model, texts, proxy_queries)
        row["validation_seconds"] = round(time.monotonic() - started, 1)
        return row

    def score(row: dict) -> float:  # higher is better
        return row["proxy"]["mrr@10"] if selection == "proxy_mrr@10" else -row["validation_loss"]

    history = {"selection_metric": selection, "step_loss": [], "epochs": []}
    run_started = time.monotonic()
    initial = validate_epoch()
    history["epochs"].append({"epoch": 0, "train_loss": None, **initial})
    result = TrainingResult(history=history, run_dir=run_dir)
    # the base model is a candidate: "best" can never be worse than not fine-tuning at all
    best, since_best = score(initial), 0
    result.best_epoch, result.best_validation_loss = 0, initial["validation_loss"]
    save_checkpoint(model, run_dir / "checkpoints" / "best",
                    {"epoch": 0, **initial, "base_model": config.base_model,
                     "base_revision": config.base_revision})
    step, stopped_early = 0, False
    model.train()
    for epoch in range(1, config.epochs + 1):
        started, epoch_losses = time.monotonic(), []
        for chunk in _batches(train_pairs, config.batch_size, generator):
            loss = _batch_loss(model, chunk, texts, group_of, config.scale, rng,
                               config.mask_duplicate_query_texts)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            step += 1
            epoch_losses.append(loss.item())
            history["step_loss"].append({"step": step, "epoch": epoch, "loss": loss.item(),
                                         "lr": scheduler.get_last_lr()[0]})
            if max_steps is not None and step >= max_steps:
                break
        train_seconds = round(time.monotonic() - started, 1)
        row = validate_epoch()
        history["epochs"].append({
            "epoch": epoch, "train_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
            **row, "steps": len(epoch_losses), "seconds": train_seconds})
        extra = {"epoch": epoch, **row, "base_model": config.base_model,
                 "base_revision": config.base_revision}
        save_checkpoint(model, run_dir / "checkpoints" / "last", extra)
        if score(row) > best:
            best, since_best = score(row), 0
            result.best_epoch, result.best_validation_loss = epoch, row["validation_loss"]
            save_checkpoint(model, run_dir / "checkpoints" / "best", extra)
        else:
            since_best += 1
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", "utf-8")
        print(json.dumps({"epoch": epoch, **{k: v for k, v in history["epochs"][-1].items()
                                             if k != "epoch"}}), flush=True)
        if max_steps is not None and step >= max_steps:
            break
        if config.early_stopping_patience and since_best >= config.early_stopping_patience:
            stopped_early = True
            break
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", "utf-8")
    summary = {"experiment_label": config.experiment_label,
               "selection_metric": selection, "best_epoch": result.best_epoch,
               "best_selection_score": best if selection == "proxy_mrr@10" else -best,
               "best_validation_loss": result.best_validation_loss,
               "initial_validation_loss": initial["validation_loss"],
               "initial_proxy": initial.get("proxy"),
               "epochs_run": len(history["epochs"]) - 1, "stopped_early": stopped_early,
               "total_steps": step,
               "wall_clock_seconds": round(time.monotonic() - run_started, 1)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    return result
