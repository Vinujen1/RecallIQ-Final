"""Evaluate BM25, pretrained MiniLM and fine-tuned checkpoints on the same reviewed human
queries and the same 1,000-record index.

    python -m evaluation.evaluate --split validation
    python -m evaluation.evaluate --split validation \
        --checkpoint model_b=ml/artifacts/models/model-b-v1/checkpoints/best \
        --checkpoint model_c=ml/artifacts/models/model-c-v1/checkpoints/best
    python -m evaluation.evaluate --split test --confirm-final-test     # final report only

Only queries with reviewer_status == "reviewed" in the chosen split are used. Retrieval
searches the full 1,000-record index for every model. No metrics are produced when no
reviewed queries exist — nothing is estimated or filled in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from evaluation import (
    DEFAULT_ANNOTATIONS,
    DEFAULT_DB,
    DEFAULT_MANIFEST,
    DEFAULT_REPORT_DIR,
    SPLITS,
)
from evaluation.annotations import corpus_context, read_any, validate
from evaluation.diagnostics import error_rows, paired_bootstrap, false_match_rate, identifier_precision, rank_changes
from evaluation.metrics import METRICS, evaluate_run
from evaluation.rankers import Ranker, build_rankers, parse_checkpoint
from evaluation.splits import manifest_sha256
from retrieval.embed import DEFAULT_OUT as DEFAULT_EMBEDDINGS
from retrieval.loader import build_engine
from retrieval.search import SearchEngine, SearchUnavailableError

RANK_DEPTH = 10  # deepest cutoff used by the metrics (MRR@10, nDCG@10)
SMALL_SAMPLE = 30
TEST_RUN_LOG = DEFAULT_REPORT_DIR / "test_runs.jsonl"


class EvaluationError(RuntimeError):
    pass


def _git_commit() -> str | None:
    import subprocess

    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=DEFAULT_REPORT_DIR.parents[2], timeout=10, check=True)
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _log_test_run(report: dict, log: Path | None = None) -> int:
    """Append an audit line for a final-test evaluation; return the number of earlier runs."""
    log = log or TEST_RUN_LOG
    previous = len(log.read_text("utf-8").splitlines()) if log.is_file() else 0
    log.parent.mkdir(parents=True, exist_ok=True)
    entry = {"provenance": report.get("provenance"), "num_queries": report["num_queries"],
             "means": {m: r.get("mean") for m, r in report["results"].items()}}
    with log.open("a", encoding="utf-8") as handle:
        print(json.dumps(entry), file=handle)
    return previous


def run_evaluation(
    annotations_path: Path,
    split: str,
    models: list[str],
    engine: SearchEngine,
    split_of: dict[int, str],
    titles: dict[int, str],
    confirm_final_test: bool = False,
    assignment_fingerprint: str | None = None,
    extra_rankers: dict[str, Ranker] | None = None,
) -> dict:
    if split not in SPLITS:
        raise EvaluationError(f"split must be one of {SPLITS}")
    if split == "test" and not confirm_final_test:
        raise EvaluationError(
            "The test split is the held-out final evaluation. Re-run with "
            "--confirm-final-test only for the final report, never for model selection."
        )
    if split == "train":
        raise EvaluationError("Queries in the train split are for training, not evaluation.")

    report = validate(read_any(annotations_path), split_of, titles, assignment_fingerprint)
    if not report.ok:
        raise EvaluationError(
            f"{len(report.errors)} annotation errors; run `python -m evaluation.annotations "
            f"validate` first. First error: {report.errors[0]}"
        )
    queries = report.reviewed(split)
    if not queries:
        raise EvaluationError(
            f"No reviewed queries in the {split} split: no metrics computed. Write queries "
            "with `python -m evaluation.annotate` and have a second person review them."
        )

    qrels = {a.query_id: a.qrels for a in queries}
    # Breakdown: judgments restricted to the query's own split (always non-empty because
    # the anchor recall is in the query's split). Relevant recalls from train may be
    # easier for a model fine-tuned on train documents, so both views are reported.
    qrels_in_split = {
        a.query_id: {d: g for d, g in a.qrels.items() if split_of[d] == a.split} for a in queries
    }
    outside = sum(len(qrels[q]) - len(qrels_in_split[q]) for q in qrels)
    rankers = build_rankers(engine, models, [])
    rankers.update(extra_rankers or {})
    texts = {a.query_id: a.query for a in queries}
    evidence = {r.recall_id: r.search_text for r in engine.corpus}
    results: dict[str, dict] = {}
    for model, ranker in rankers.items():
        run = {}
        try:
            for a in queries:
                run[a.query_id] = ranker(a.query, RANK_DEPTH)
        except SearchUnavailableError as error:
            results[model] = {"error": str(error)}
            continue
        scored = evaluate_run(run, qrels)
        scored["mean_in_split_judgments_only"] = evaluate_run(run, qrels_in_split)["mean"]
        scored["false_match_rate@1"] = false_match_rate(run, qrels)
        scored["identifier_precision@5"] = identifier_precision(run, texts, qrels, evidence)
        scored["rankings"] = run
        scored["error_rows"] = error_rows(run, texts, qrels, titles)
        results[model] = scored

    comparisons = {}
    for a, b in (("minilm", "model_b"), ("model_b", "model_c"), ("bm25", "model_c")):
        if "error_rows" in results.get(a, {}) and "error_rows" in results.get(b, {}):
            comparisons[f"{b}_vs_{a}"] = {
                **rank_changes(results[a]["error_rows"], results[b]["error_rows"]),
                "bootstrap_mrr@10": paired_bootstrap(results[a]["per_query"], results[b]["per_query"])}

    warnings = list(report.warnings)
    if len(queries) < SMALL_SAMPLE:
        warnings.append(
            f"Only {len(queries)} reviewed queries: differences between models may be noise."
        )
    return {
        "split": split,
        "final_test": split == "test",
        "num_queries": len(queries),
        "num_relevant_judgments": sum(len(a.relevant) for a in queries),
        "relevant_judgments_outside_query_split": outside,
        "corpus_size": len(engine.corpus),
        "metrics": list(METRICS),
        "results": results,
        "comparisons": comparisons,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate BM25 and MiniLM on reviewed queries.")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--models", nargs="+", choices=["bm25", "minilm"], default=["bm25", "minilm"])
    parser.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH",
                        help="fine-tuned encoder checkpoint to compare (repeatable), e.g. "
                             "model_b=ml/artifacts/models/model-b-v1/checkpoints/best")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--out", type=Path, default=None, help="JSON report path")
    args = parser.parse_args(argv)

    try:
        split_of, titles, manifest = corpus_context(args.db, args.manifest)
        engine = build_engine(args.db, args.embeddings)
        checkpoints = [parse_checkpoint(spec) for spec in args.checkpoint]
        report = run_evaluation(args.annotations, args.split, args.models, engine,
                                split_of, titles, args.confirm_final_test,
                                manifest.get("assignment_fingerprint"),
                                extra_rankers=build_rankers(engine, [], checkpoints))
    except (EvaluationError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Evaluation not run: {error}")
        return 2

    report["provenance"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "annotations_file": str(args.annotations),
        "annotations_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
        "split_manifest_sha256": manifest_sha256(args.manifest),
        "split_seed": manifest["seed"],
        "split_assignment_fingerprint": manifest.get("assignment_fingerprint"),
        "git_commit": _git_commit(),
        "models": args.models,
        "checkpoints": {label: str(path) for label, path in
                        (parse_checkpoint(spec) for spec in args.checkpoint)},
        "rank_depth": RANK_DEPTH,
        "corpus_fingerprint": manifest["corpus_fingerprint"],
        "minilm_revision": (engine.embeddings.metadata.get("model_revision")
                            if engine.embeddings else None),
    }
    if not manifest.get("frozen"):
        report["warnings"].append("Split manifest is not frozen (split review pending).")
    if args.split == "test":
        print("*** FINAL TEST SPLIT — report only; do not use these numbers to choose models. ***")
        previous = _log_test_run(report)
        if previous:
            print(f"WARNING: the test split has now been evaluated {previous + 1} times "
                  f"(see {TEST_RUN_LOG}). Repeated test runs invite selection on test.")
    print(f"split={report['split']}  queries={report['num_queries']}  "
          f"judgments={report['num_relevant_judgments']}  corpus={report['corpus_size']}")
    print(f"{'model':10} " + " ".join(f"{m:>9}" for m in report["metrics"]) + "     FMR@1   IVP@5")
    for model, result in report["results"].items():
        if "error" in result:
            print(f"{model:10} unavailable: {result['error']}")
            continue
        ivp = result["identifier_precision@5"]["value"]
        print(f"{model:10} " + " ".join(f"{result['mean'][m]:9.4f}" for m in report["metrics"])
              + f"  {result['false_match_rate@1']['value']:8.4f}"
              + (f"  {ivp:6.4f}" if ivp is not None else "     n/a"))
    for warning in report["warnings"]:
        print(f"WARNING {warning}")

    out = args.out or DEFAULT_REPORT_DIR / f"eval_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nReport: {out.resolve()}")
    unavailable = [m for m, r in report["results"].items() if "error" in r]
    if unavailable:
        print(f"Incomplete comparison: {unavailable} unavailable (exit 3).")
        return 3
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
