"""CLI: build train/validation pairs and fine-tune MiniLM.

    python -m training.train --dry-run          # pairs + statistics only (no training)
    python -m training.train --run-name ft-v1   # requires a FROZEN split manifest

Outputs (git-ignored): ml/artifacts/models/<run-name>/ with config.json, run_info.json,
train_pairs.jsonl, validation_pairs.jsonl, history.json, summary.json and
checkpoints/{best,last}/. The test split is never read.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

from evaluation import DEFAULT_ANNOTATIONS, DEFAULT_DB, DEFAULT_MANIFEST, REPO_ROOT
from training.config import TrainingConfig
from training.data import (
    TrainingPair,
    build_pairs,
    drop_text_collisions,
    group_lookup,
    heldout_query_collisions,
    normalize_query,
    require_frozen,
    reviewed_distinct_pairs,
    summarize,
    write_pairs,
)

DEFAULT_RUNS = REPO_ROOT / "ml" / "artifacts" / "models"
DEFAULT_REVIEW = REPO_ROOT / "data" / "reviews" / "split_pair_review.csv"


PROXY_EXCLUDED_SOURCES = ("pseudo:title_core",)  # near-verbatim titles: trivially matched


def proxy_queries_from(pairs: list[TrainingPair], group_of: dict[int, str],
                       exclude_sources: tuple[str, ...] = PROXY_EXCLUDED_SOURCES,
                       ) -> list[tuple[str, str, dict[int, int]]]:
    """Validation pseudo-queries as (query_id, text, qrels) for full-index proxy retrieval.

    Relevant = the source document plus the other members of its split group (same product,
    re-notice or colocated related event). Identical texts generated for different documents
    are merged into one query with the union of their relevant sets. GENERATED labels: a
    selection signal only, never reported as human-judged results.
    """
    members: dict[str, list[int]] = {}
    for rid, group in group_of.items():
        members.setdefault(group, []).append(rid)
    merged: dict[str, tuple[str, str, dict[int, int]]] = {}
    for pair in pairs:
        key = normalize_query(pair.query)
        if not key or pair.source in exclude_sources:
            continue
        qid, text, rel = merged.setdefault(key, (pair.query_id, pair.query, {}))
        for rid in members[group_of[pair.positive_id]]:
            rel[rid] = 1
    return sorted(merged.values())


def _products(db_path: Path) -> dict[int, str | None]:
    connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return dict(connection.execute("SELECT id, product FROM recalls").fetchall())
    finally:
        connection.close()


def _doc_embeddings(embeddings_dir: Path, corpus) -> dict[int, np.ndarray] | None:
    from retrieval.semantic import StaleIndexError, load_embeddings

    try:
        index = load_embeddings(embeddings_dir)
        index.check_matches(corpus)
    except (FileNotFoundError, StaleIndexError) as error:
        print(f"warning: no semantic hard negatives ({error})", file=sys.stderr)
        return None
    return {rid: index.matrix[i] for i, rid in enumerate(index.recall_ids)}


def _annotations(path: Path, split_of, titles, fingerprint) -> tuple[list, set[str]]:
    """(reviewed train queries, normalised texts of ALL validation/test queries)."""
    from evaluation.annotations import read_any, validate

    if not path.is_file():
        return [], set()
    report = validate(read_any(path), split_of, titles, fingerprint)
    if not report.ok:
        raise RuntimeError(f"annotation errors; first: {report.errors[0]}")
    collisions = heldout_query_collisions(report.valid)
    if collisions:
        raise RuntimeError(f"train queries repeat a validation/test query: {collisions[:5]}")
    held_out = {normalize_query(a.query) for a in report.valid if a.split in ("validation", "test")}
    return report.reviewed("train"), held_out


def _verified_distinct(review_path: Path, manifest: dict) -> dict[int, set[int]]:
    """DISTINCT decisions from the applied human split review (empty if not applied)."""
    from evaluation.split_review import read_review

    if not (manifest.get("review") or {}).get("applied") or not review_path.is_file():
        return {}
    return reviewed_distinct_pairs(read_review(review_path))


def prepare(db_path: Path, manifest_path: Path, embeddings_dir: Path, annotations_path: Path,
            config: TrainingConfig, require_frozen_split: bool = True,
            allow_no_semantic: bool = False, review_path: Path = DEFAULT_REVIEW):
    from evaluation.annotations import corpus_context
    from retrieval.corpus import load_corpus

    split_of, titles, manifest = corpus_context(db_path, manifest_path)
    if require_frozen_split:
        require_frozen(manifest)
    corpus = load_corpus(db_path)
    products = _products(db_path)
    embeddings = _doc_embeddings(embeddings_dir, corpus)
    if embeddings is None and config.semantic_candidates > 0 and not allow_no_semantic:
        raise RuntimeError("semantic hard negatives need current MiniLM embeddings "
                           "(`python -m retrieval.embed`); pass --allow-no-semantic-negatives "
                           "to train with BM25 negatives only")
    human, held_out_texts = _annotations(annotations_path, split_of, titles,
                                         manifest.get("assignment_fingerprint"))
    distinct = _verified_distinct(review_path, manifest)
    train_pairs = build_pairs(corpus, manifest, "train", config, products, embeddings, human,
                              verified_distinct=distinct)
    # identical validation pairs for every training variant (Model B and Model C)
    validation_pairs = build_pairs(corpus, manifest, "validation", config, products, embeddings,
                                   verified_distinct=distinct,
                                   num_negatives=config.validation_hard_negatives)
    held_out_texts |= {normalize_query(p.query) for p in validation_pairs}
    train_pairs, dropped = drop_text_collisions(train_pairs, held_out_texts)
    verified_used = sum(1 for p in train_pairs if p.negative_ids
                        and p.negative_ids[0] in distinct.get(p.positive_id, ()))
    provenance = {"split_manifest": str(manifest_path), "split_frozen": bool(manifest.get("frozen")),
                  "assignment_fingerprint": manifest.get("assignment_fingerprint"),
                  "corpus_fingerprint": manifest.get("corpus_fingerprint"),
                  "human_train_queries": len(human),
                  "semantic_hard_negatives": embeddings is not None,
                  "reviewed_distinct_pairs_available": sum(len(v) for v in distinct.values()) // 2,
                  "train_pairs_with_human_verified_negative": verified_used,
                  "train_pairs_dropped_for_heldout_text_collision": dropped}
    return corpus, manifest, train_pairs, validation_pairs, provenance


def main(argv: list[str] | None = None) -> int:
    from retrieval.embed import DEFAULT_OUT as DEFAULT_EMBEDDINGS

    parser = argparse.ArgumentParser(description="Fine-tune MiniLM with contrastive learning.")
    parser.add_argument("--config", type=Path, help="JSON TrainingConfig overrides")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--run-name", default="minilm-ft")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and summarise pairs only; allowed on an unfrozen split")
    parser.add_argument("--max-steps", type=int, help="stop early (smoke runs)")
    parser.add_argument("--allow-no-semantic-negatives", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = TrainingConfig.load(args.config) if args.config else TrainingConfig()
        config.validate()
        corpus, manifest, train_pairs, validation_pairs, provenance = prepare(
            args.db, args.manifest, args.embeddings, args.annotations, config,
            require_frozen_split=not args.dry_run,
            allow_no_semantic=args.allow_no_semantic_negatives)
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    stats = {"experiment_label": config.experiment_label,
             "train": summarize(train_pairs), "validation": summarize(validation_pairs),
             "provenance": provenance}
    print(json.dumps(stats, indent=2))
    if args.dry_run:
        if not provenance["split_frozen"]:
            print("\nDRY RUN on an UNFROZEN split: statistics only, no training.")
        return 0

    from training.model import MeanPoolEncoder
    from training.trainer import train

    run_dir = args.runs_dir / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        print(f"error: {run_dir} already exists; choose another --run-name", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(config.to_json(), indent=2) + "\n", "utf-8")
    write_pairs(train_pairs, run_dir / "train_pairs.jsonl")
    write_pairs(validation_pairs, run_dir / "validation_pairs.jsonl")
    model = MeanPoolEncoder.from_pretrained(config.base_model, config.base_revision,
                                            config.max_seq_length)
    texts = {r.recall_id: r.search_text for r in corpus}  # full 1,000-record index
    group_of = group_lookup(manifest, corpus)
    proxy = proxy_queries_from(validation_pairs, group_of)
    result = train(model, train_pairs, validation_pairs, texts, group_of,
                   config, run_dir, provenance, args.max_steps, proxy_queries=proxy)
    print(json.dumps({"run_dir": str(run_dir), "best_epoch": result.best_epoch,
                      "best_validation_loss": result.best_validation_loss}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
