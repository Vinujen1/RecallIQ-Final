"""PROXY comparison on GENERATED validation queries (not human judgments).

    python -m evaluation.proxy_eval --pairs ml/artifacts/models/model-b-v1/validation_pairs.jsonl \\
        --checkpoint model_b=ml/artifacts/models/model-b-v1/checkpoints/best \\
        --checkpoint model_c=ml/artifacts/models/model-c-v1/checkpoints/best

Queries are the template-generated pseudo-queries of VALIDATION documents (the same ones
used for checkpoint selection, `title_core` excluded); relevant = the source recall plus its
split group. They are lexically close to the recall text, so the comparison favours lexical
matching and saturates near the top. It is a pipeline/selection diagnostic for the technical
checkpoint and must be reported as "proxy (generated queries)", never as a result on
consumer queries. The real comparison is `evaluation.evaluate` on reviewed human queries.
The test split is never touched (the pairs file holds validation documents only; checked).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from evaluation import DEFAULT_DB, DEFAULT_MANIFEST, DEFAULT_REPORT_DIR
from evaluation.diagnostics import error_rows, paired_bootstrap, false_match_rate, rank_changes
from evaluation.metrics import METRICS, evaluate_run
from evaluation.rankers import build_rankers, parse_checkpoint
from evaluation.splits import load_manifest, split_lookup
from retrieval.embed import DEFAULT_OUT as DEFAULT_EMBEDDINGS
from retrieval.loader import build_engine
from training.data import TrainingPair, group_lookup
from training.train import proxy_queries_from

LABEL = "PROXY - generated validation queries, NOT human relevance judgments"


def load_pairs(path: Path) -> list[TrainingPair]:
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    return [TrainingPair(r["query"], r["positive_id"], tuple(r["negative_ids"]), r["source"],
                         r["query_id"]) for r in rows]


def run_proxy(pairs: list[TrainingPair], engine, manifest: dict, rankers: dict) -> dict:
    split_of = split_lookup(manifest)
    bad = sorted({p.positive_id for p in pairs if split_of.get(p.positive_id) != "validation"})
    if bad:
        raise ValueError(f"proxy pairs must come from validation documents only; got {bad[:5]}")
    queries = proxy_queries_from(pairs, group_lookup(manifest, engine.corpus))
    texts = {q: t for q, t, _ in queries}
    qrels = {q: rel for q, _, rel in queries}
    source = {p.query_id: p.source for p in pairs}
    titles = {r.recall_id: r.title for r in engine.corpus}
    results = {}
    for name, ranker in rankers.items():
        run = {q: ranker(t, 10) for q, t in texts.items()}
        scored = evaluate_run(run, qrels)
        by_view = {}
        for view in sorted({source[q] for q in qrels}):
            subset = {q: r for q, r in qrels.items() if source[q] == view}
            by_view[view] = {"num_queries": len(subset), **evaluate_run(run, subset)["mean"]}
        results[name] = {"mean": scored["mean"], "by_view": by_view,
                         "false_match_rate@1": false_match_rate(run, qrels),
                         "per_query": scored["per_query"], "rankings": run,
                         "error_rows": error_rows(run, texts, qrels, titles)}
    comparisons = {}
    for a, b in (("minilm", "model_b"), ("model_b", "model_c")):
        if a in results and b in results:
            comparisons[f"{b}_vs_{a}"] = {
                **rank_changes(results[a]["error_rows"], results[b]["error_rows"]),
                "bootstrap_mrr@10": paired_bootstrap(results[a]["per_query"], results[b]["per_query"])}
    for a, b in (("bm25", "model_b"), ("bm25", "model_c")):
        if a in results and b in results:
            comparisons[f"{b}_vs_{a}"] = {"bootstrap_mrr@10": paired_bootstrap(
                results[a]["per_query"], results[b]["per_query"])}
    return {"label": LABEL, "num_queries": len(queries), "corpus_size": len(engine.corpus),
            "metrics": list(METRICS), "results": results, "comparisons": comparisons}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=LABEL)
    parser.add_argument("--pairs", type=Path, required=True,
                        help="validation_pairs.jsonl written by training.train")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_DIR / "proxy_validation.json")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    engine = build_engine(args.db, args.embeddings)
    checkpoints = [parse_checkpoint(spec) for spec in args.checkpoint]
    rankers = build_rankers(engine, ["bm25", "minilm"], checkpoints)
    try:
        report = run_proxy(load_pairs(args.pairs), engine, manifest, rankers)
    except ValueError as error:
        print(f"Proxy evaluation not run: {error}")
        return 2
    report["provenance"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pairs_file": str(args.pairs),
        "pairs_sha256": hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
        "split_assignment_fingerprint": manifest.get("assignment_fingerprint"),
        "split_frozen": bool(manifest.get("frozen")),
        "corpus_fingerprint": manifest.get("corpus_fingerprint"),
        "checkpoints": {label: str(path) for label, path in checkpoints},
    }
    print(f"*** {LABEL} ***")
    print(f"queries={report['num_queries']}  corpus={report['corpus_size']}")
    print(f"{'model':10} " + " ".join(f"{m:>9}" for m in report["metrics"]) + "     FMR@1")
    for name, result in report["results"].items():
        print(f"{name:10} " + " ".join(f"{result['mean'][m]:9.4f}" for m in report["metrics"])
              + f"  {result['false_match_rate@1']['value']:8.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(f"Report: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
