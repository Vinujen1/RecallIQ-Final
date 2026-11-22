"""Command-line entry point: raw CSV -> cleaned JSONL + SQLite.

Usage (from the ml/ directory, with the project venv active):
    python -m recall_data.build
    python -m recall_data.build --input ../data/raw/HCRSAMOpenData.csv --size 1000 --seed 4110
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from recall_data import PIPELINE_VERSION
from recall_data.clean import filter_eligible, normalize, unparseable_date_count
from recall_data.grouping import add_product_families, add_title_groups
from recall_data.load import file_sha256, load_raw
from recall_data.sample import DEFAULT_SAMPLE_SIZE, DEFAULT_SEED, stratified_sample
from recall_data.storage import to_records, write_json, write_jsonl, write_sqlite

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "HCRSAMOpenData.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed"


def run(
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    raw = load_raw(input_path)
    eligible_raw = filter_eligible(raw)
    if eligible_raw.empty:
        raise ValueError(
            f"No eligible records (organizations CFIA / Consumer product safety) in {input_path}"
        )
    cleaned = normalize(eligible_raw)
    grouped = add_product_families(add_title_groups(cleaned))
    sample, strata = stratified_sample(grouped, size=size, seed=seed)

    eligible_records = to_records(grouped)
    sample_records = to_records(sample)

    manifest: dict[str, object] = {
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_file": input_path.name,
        "input_sha256": file_sha256(input_path),
        "seed": seed,
        "target_size": size,
        "raw_records": int(len(raw)),
        "raw_by_organization": dict(Counter(raw["Organization"]).most_common()),
        "eligible_records": int(len(grouped)),
        "eligible_by_domain": dict(Counter(grouped["domain"]).most_common()),
        "eligible_title_groups": int(grouped["title_group"].nunique()),
        "eligible_duplicate_title_groups": int(
            (grouped.drop_duplicates("title_group")["title_group_size"] > 1).sum()
        ),
        "eligible_unparseable_dates": unparseable_date_count(eligible_raw),
        "selected_records": int(len(sample)),
        "selected_by_domain": dict(Counter(sample["domain"]).most_common()),
        "selected_family_confidence": dict(Counter(sample["family_confidence"]).most_common()),
        "selected_needs_review": int(sample["needs_review"].sum()),
        "selected_missing": {
            field: int(sample[field].isna().sum())
            for field in ("product", "issue", "what_you_should_do", "recall_class", "last_updated")
        },
        "strata": strata,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(eligible_records, output_dir / "recalls_eligible.jsonl")
    write_jsonl(sample_records, output_dir / "recalls_sample.jsonl")
    write_sqlite(
        sample_records,
        output_dir / "recalliq.db",
        metadata={k: v for k, v in manifest.items() if k != "strata"},
    )
    write_json(manifest, output_dir / "manifest.json")  # written last: marks a complete build
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the RecallIQ recall dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    manifest = run(args.input, args.output_dir, args.size, args.seed)
    summary = {k: v for k, v in manifest.items() if k != "strata"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nOutputs written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
