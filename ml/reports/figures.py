"""Checkpoint figures and tables from REAL run artifacts (no numbers are typed in here).

    python -m reports.figures --run model_b=artifacts/models/model-b-v1 \\
        --run model_c=artifacts/models/model-c-v1 \\
        --proxy ../data/processed/eval/proxy_validation.json --out artifacts/reports

Writes, per run, `loss_<label>.svg` (+ `.csv`): mean training loss per epoch and the two
validation losses (with / without a hard negative, same validation pairs); and, across
runs, `validation_loss_compare.svg`, `proxy_mrr_compare.svg` and `tables.md` (training
configuration, per-epoch history, proxy comparison). Every value is read from
history.json / summary.json / run_info.json / the proxy report, and each output names its
source files. Dependency-free SVG (no matplotlib).

Colours: categorical slots 1-3 of the validated default palette (blue, orange, aqua), in
fixed order; series also differ by dash pattern, so identity is never colour-alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from html import escape
from pathlib import Path

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")     # categorical slots 1-3 (light surface)
DASHES = ("", "6 4", "2 3")
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#5f5e58", "#e4e3de", "#ffffff"
W, H, L, R, T, B = 640, 360, 64, 150, 56, 52     # size + margins (right margin: end labels)


def _range(values: list[float]) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.08 or abs(hi) * 0.1 or 0.1
    return (max(0.0, lo - pad) if lo >= 0 else lo - pad), hi + pad


def line_chart(title: str, x_label: str, y_label: str,
               series: list[tuple[str, list[tuple[float, float]]]], source: str) -> str:
    """series: [(name, [(x, y), ...])]; one shared y-axis; legend + end labels."""
    points = [p for _, pts in series for p in pts]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x0 = min(xs)
    x1 = max(xs) if max(xs) > x0 else x0 + 1
    y0, y1 = _range(ys)

    def sx(x: float) -> float:
        return L + (x - x0) / (x1 - x0) * (W - L - R)

    def sy(y: float) -> float:
        return T + (1 - (y - y0) / (y1 - y0)) * (H - T - B)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'font-family="Helvetica, Arial, sans-serif" role="img" aria-label="{escape(title)}">',
           f'<title>{escape(title)}</title><desc>Source: {escape(source)}</desc>',
           f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
           f'<text x="{L}" y="22" font-size="15" font-weight="600" fill="{INK}">{escape(title)}</text>']
    for i in range(5):  # recessive horizontal grid + y ticks
        v = y0 + (y1 - y0) * i / 4
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{GRID}"/>')
        out.append(f'<text x="{L - 8}" y="{sy(v) + 4:.1f}" font-size="11" fill="{MUTED}" '
                   f'text-anchor="end">{v:.3g}</text>')
    if len(set(xs)) <= 12:
        for x in sorted(set(xs)):
            out.append(f'<text x="{sx(x):.1f}" y="{H - B + 18}" font-size="11" fill="{MUTED}" '
                       f'text-anchor="middle">{x:g}</text>')
    out.append(f'<line x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}" stroke="{MUTED}"/>')
    out.append(f'<text x="{(L + W - R) / 2}" y="{H - 12}" font-size="12" fill="{MUTED}" '
               f'text-anchor="middle">{escape(x_label)}</text>')
    out.append(f'<text transform="translate(16 {(T + H - B) / 2}) rotate(-90)" font-size="12" '
               f'fill="{MUTED}" text-anchor="middle">{escape(y_label)}</text>')
    legend_x = L
    for i, (name, pts) in enumerate(series):
        color, dash = SERIES[i % 3], DASHES[i % 3]
        path = " ".join(f"{'M' if j == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for j, (x, y) in enumerate(pts))
        out.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" '
                   f'stroke-dasharray="{dash}" stroke-linejoin="round"/>')
        if len(pts) <= 12:
            out += [f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}" '
                    f'stroke="{SURFACE}" stroke-width="2"/>' for x, y in pts]
        lx, ly = pts[-1]
        out.append(f'<text x="{sx(lx) + 8:.1f}" y="{sy(ly) + 4:.1f}" font-size="11" fill="{INK}">'
                   f'{escape(name)} {ly:.3g}</text>')
        out.append(f'<line x1="{legend_x}" x2="{legend_x + 22}" y1="38" y2="38" stroke="{color}" '
                   f'stroke-width="2" stroke-dasharray="{dash}"/>')
        out.append(f'<text x="{legend_x + 28}" y="42" font-size="11" fill="{INK}">{escape(name)}</text>')
        legend_x += 40 + 7 * len(name)
    out.append("</svg>")
    return "\n".join(out)


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def load_run(directory: Path) -> dict:
    def read(name: str) -> dict:
        return json.loads((directory / name).read_text("utf-8"))
    return {"dir": directory, "history": read("history.json"), "summary": read("summary.json"),
            "info": read("run_info.json")}


def run_figures(label: str, run: dict, out: Path) -> None:
    epochs = run["history"]["epochs"]
    train = [(e["epoch"], e["train_loss"]) for e in epochs if e["train_loss"] is not None]
    val = [(e["epoch"], e["validation_loss"]) for e in epochs]
    val_ib = [(e["epoch"], e["validation_loss_in_batch_only"]) for e in epochs
              if "validation_loss_in_batch_only" in e]
    series = [("train", train), ("val +hard neg", val)] + ([("val in-batch", val_ib)] if val_ib else [])
    (out / f"loss_{label}.svg").write_text(line_chart(
        f"{label}: contrastive loss per epoch (epoch 0 = pretrained)", "epoch",
        "MNRL / InfoNCE loss", series, str(run["dir"] / "history.json")), "utf-8")
    _write_csv(out / f"loss_{label}.csv",
               ["epoch", "train_loss", "validation_loss_hard_neg", "validation_loss_in_batch_only",
                "proxy_mrr@10"],
               [[e["epoch"], e["train_loss"], e["validation_loss"],
                 e.get("validation_loss_in_batch_only"), (e.get("proxy") or {}).get("mrr@10")]
                for e in epochs])


def compare_figures(runs: dict[str, dict], out: Path) -> None:
    sources = ", ".join(str(r["dir"] / "history.json") for r in runs.values())
    (out / "validation_loss_compare.svg").write_text(line_chart(
        "Validation loss (same pairs, +1 hard negative)", "epoch", "loss",
        [(label, [(e["epoch"], e["validation_loss"]) for e in r["history"]["epochs"]])
         for label, r in runs.items()], sources), "utf-8")
    proxy = [(label, [(e["epoch"], e["proxy"]["mrr@10"]) for e in r["history"]["epochs"] if "proxy" in e])
             for label, r in runs.items()]
    if all(pts for _, pts in proxy):
        (out / "proxy_mrr_compare.svg").write_text(line_chart(
            "Proxy MRR@10 (GENERATED validation queries)", "epoch", "MRR@10", proxy, sources), "utf-8")


def _fmt(value) -> str:
    return "" if value is None else f"{value:.4f}"


def tables(runs: dict[str, dict], proxy: dict | None, proxy_path: Path | None) -> str:
    lines = ["# Checkpoint tables (generated by `python -m reports.figures`; do not edit by hand)", ""]
    labels = list(runs)
    keys = ["experiment_label", "seed", "epochs", "batch_size", "learning_rate", "weight_decay",
            "warmup_ratio", "scale", "max_seq_length", "hard_negatives_per_pair",
            "validation_hard_negatives", "selection_metric", "early_stopping_patience"]
    lines += ["## Training configuration and provenance", "",
              "| setting | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for key in keys:
        lines.append(f"| {key} | " + " | ".join(str(runs[x]["info"]["config"].get(key)) for x in labels) + " |")
    rows = (
        ("train pairs", lambda i, s: i["data"]["train_pairs"]),
        ("train pairs with hard negatives", lambda i, s: i["data"].get("train_pairs_with_hard_negatives")),
        ("train pairs with human-verified negative",
         lambda i, s: i["data"].get("train_pairs_with_human_verified_negative")),
        ("validation pairs", lambda i, s: i["data"]["validation_pairs"]),
        ("validation pairs fingerprint", lambda i, s: i["data"]["validation_pairs_fingerprint"][:12]),
        ("proxy validation queries", lambda i, s: i["data"].get("proxy_validation_queries")),
        ("split assignment fingerprint", lambda i, s: (i["data"].get("assignment_fingerprint") or "")[:12]),
        ("device / threads", lambda i, s: f'{i["device"]} / {i.get("hardware", {}).get("torch_threads")}'),
        ("torch / transformers", lambda i, s: f'{i["environment"]["torch"]} / {i["environment"]["transformers"]}'),
        ("git commit (dirty)", lambda i, s: f'{(i["environment"]["git_commit"] or "")[:7]} ({i["environment"]["git_dirty"]})'),
        ("epochs run (early stop)", lambda i, s: f'{s["epochs_run"]} ({s["stopped_early"]})'),
        ("best epoch (selection metric)", lambda i, s: f'{s["best_epoch"]} ({s["selection_metric"]})'),
        ("wall clock (s)", lambda i, s: s.get("wall_clock_seconds")),
    )
    for name, getter in rows:
        lines.append(f"| {name} | " + " | ".join(
            str(getter(runs[x]["info"], runs[x]["summary"])) for x in labels) + " |")
    for label, run in runs.items():
        lines += ["", f"## Per-epoch history: {label}  (source: `{run['dir'].as_posix()}/history.json`)", "",
                  "| epoch | train loss | val loss (+hard neg) | val loss (in-batch) | proxy R@5 | "
                  "proxy MRR@10 | proxy nDCG@10 | train s |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for e in run["history"]["epochs"]:
            p = e.get("proxy") or {}
            lines.append(f'| {e["epoch"]} | {_fmt(e["train_loss"])} | {_fmt(e["validation_loss"])} | '
                         f'{_fmt(e.get("validation_loss_in_batch_only"))} | {_fmt(p.get("recall@5"))} | '
                         f'{_fmt(p.get("mrr@10"))} | {_fmt(p.get("ndcg@10"))} | {e.get("seconds", "")} |')
    if proxy:
        first = next(iter(proxy["results"].values()))
        lines += ["", f"## {proxy['label']}", "",
                  f"Source: `{proxy_path.as_posix() if proxy_path else ''}`; {proxy['num_queries']} "
                  f"generated validation queries; index = {proxy['corpus_size']} recalls. "
                  "Not a result on consumer queries.", "",
                  "| system | Recall@5 | MRR@10 | nDCG@10 | FMR@1 (n/N) |", "|---|---:|---:|---:|---:|"]
        for name, r in proxy["results"].items():
            f = r["false_match_rate@1"]
            lines.append(f'| {name} | {r["mean"]["recall@5"]:.4f} | {r["mean"]["mrr@10"]:.4f} | '
                         f'{r["mean"]["ndcg@10"]:.4f} | {f["value"]:.4f} ({f["numerator"]}/{f["denominator"]}) |')
        views = sorted(first["by_view"])
        lines += ["", "Proxy MRR@10 by generated query view:", "",
                  "| system | " + " | ".join(f"{v.split(':')[-1]} (n={first['by_view'][v]['num_queries']})"
                                            for v in views) + " |",
                  "|---|" + "---:|" * len(views)]
        for name, r in proxy["results"].items():
            lines.append(f"| {name} | " + " | ".join(f'{r["by_view"][v]["mrr@10"]:.4f}' for v in views) + " |")
        lines += ["", "Paired comparisons (same queries; 95 % bootstrap CI over queries, "
                  "10,000 resamples, seed 4110; does not include training-seed variance):", "",
                  "| comparison | mean ΔMRR@10 | 95 % CI | better / worse / tied | first-relevant rank improved / worsened |",
                  "|---|---:|---|---|---|"]
        for key, change in proxy.get("comparisons", {}).items():
            boot = change.get("bootstrap_mrr@10", {})
            ranks = (f"{len(change['improved'])} / {len(change['worsened'])}"
                     if "improved" in change else "")
            lines.append(f"| {key} | {boot.get('mean_difference', 0):+.4f} | "
                         f"[{boot['ci95'][0]:+.4f}, {boot['ci95'][1]:+.4f}] | "
                         f"{boot['better']} / {boot['worse']} / {boot['ties']} | {ranks} |")
    if proxy:
        lines += error_analysis(proxy)
    return "\n".join(lines) + "\n"


def error_analysis(proxy: dict, limit: int = 6) -> list[str]:
    """Concrete cases from the proxy report: hard-negative help/hurt, common failures."""
    res = proxy["results"]
    lines = ["", "## Error analysis (generated validation queries; titles verbatim from the index)"]
    change = proxy.get("comparisons", {}).get("model_c_vs_model_b", {})
    rows = {m: {r["query_id"]: r for r in res[m].get("error_rows", [])} for m in res}
    for kind in ("improved", "worsened"):
        cases = change.get(kind, [])
        lines += ["", f"### Model C vs Model B: first relevant rank {kind} ({len(cases)} queries)", "",
                  "| query | B rank | C rank | relevant recall (title) |", "|---|---:|---:|---|"]
        for c in cases[:limit]:
            row = rows.get("model_c", {}).get(c["query_id"])
            rel = row["relevant"][0]["title"] if row else ""
            lines.append(f"| {c['query']} | {c['from'] or '>10'} | {c['to'] or '>10'} | {rel} |")
    if all(m in rows for m in ("bm25", "minilm", "model_b", "model_c")):
        missed = [q for q in rows["model_c"] if all(rows[m][q]["first_relevant_rank"] != 1 for m in rows)]
        lines += ["", f"### Rank-1 false match for ALL four systems ({len(missed)} queries)", "",
                  "| query | relevant recall | Model C rank-1 result (wrong) | C rank of relevant |",
                  "|---|---|---|---:|"]
        for q in missed[:limit]:
            r = rows["model_c"][q]
            lines.append(f"| {r['query']} | {r['relevant'][0]['title']} | "
                         f"{r['top3'][0]['title'] if r['top3'] else ''} | {r['first_relevant_rank'] or '>10'} |")
        wins = [q for q in rows["model_b"] if rows["minilm"][q]["first_relevant_rank"] != 1
                and rows["model_b"][q]["first_relevant_rank"] == 1]
        lines += ["", f"### Fine-tuning success: rank 1 for Model B but not pretrained MiniLM "
                  f"({len(wins)} queries)", "",
                  "| query | MiniLM rank | MiniLM rank-1 result | relevant recall |", "|---|---:|---|---|"]
        for q in wins[:limit]:
            a = rows["minilm"][q]
            lines.append(f"| {a['query']} | {a['first_relevant_rank'] or '>10'} | "
                         f"{a['top3'][0]['title'] if a['top3'] else ''} | {a['relevant'][0]['title']} |")
    return lines


RECORD_FILES =("config.json", "run_info.json", "history.json", "summary.json")


def export_records(runs: dict[str, dict], dest: Path, proxy_path: Path | None) -> None:
    """Small, git-trackable copies of each run (no weights, no full pair files): the JSON
    records plus the first 20 training pairs as a sample. Large files stay in ml/artifacts."""
    import shutil

    for run in runs.values():
        target = dest / run["dir"].name
        target.mkdir(parents=True, exist_ok=True)
        for name in RECORD_FILES:
            shutil.copy2(run["dir"] / name, target / name)
        lines = (run["dir"] / "train_pairs.jsonl").read_text("utf-8").splitlines()[:20]
        (target / "sample_train_pairs.jsonl").write_text("\n".join(lines) + "\n", "utf-8")
    if proxy_path:
        report = json.loads(proxy_path.read_text("utf-8"))
        for result in report["results"].values():   # keep means + per-query; drop bulky rankings
            result.pop("rankings", None)
        (dest / "proxy_validation.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", "utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Checkpoint figures/tables from run artifacts.")
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=RUN_DIR")
    parser.add_argument("--proxy", type=Path, help="proxy_validation.json from evaluation.proxy_eval")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--export-runs", type=Path,
                        help="copy lightweight run records (no weights) here, e.g. ../experiments/runs")
    args = parser.parse_args(argv)
    runs = {}
    for spec in args.run:
        label, _, path = spec.partition("=")
        runs[label] = load_run(Path(path))
    args.out.mkdir(parents=True, exist_ok=True)
    for label, run in runs.items():
        run_figures(label, run, args.out)
    compare_figures(runs, args.out)
    proxy = json.loads(args.proxy.read_text("utf-8")) if args.proxy else None
    (args.out / "tables.md").write_text(tables(runs, proxy, args.proxy), "utf-8")
    if args.export_runs:
        export_records(runs, args.export_runs, args.proxy)
    print(f"Wrote figures and tables to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
