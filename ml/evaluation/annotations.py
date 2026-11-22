"""Human relevance annotations: format, import (JSONL / CSV), validation, export.

    python -m evaluation.annotations validate data/annotations/queries.jsonl
    python -m evaluation.annotations import reviewed.csv --out data/annotations/queries.jsonl
    python -m evaluation.annotations export data/annotations/queries.jsonl --out review.csv

JSONL — one query per line:
  {"query_id": "AB-0001", "query": "<consumer-written text>", "split": "validation",
   "relevant": [{"recall_id": 50609, "grade": 2}, {"recall_id": 50874}],
   "reviewer_status": "draft|reviewed|rejected", "annotator": "AB", "reviewer": "CD",
   "notes": "", "created_at": "2026-11-15T12:00:00+00:00", "anchor_recall_id": 50609}
CSV — one row per (query, relevant recall) with columns CSV_COLUMNS.
Grades: 1 = relevant (default), 2 = highly relevant, 3 = exact recall the user means.
Relevant recalls may come from ANY split (judgments cover the full 1,000-record index);
the query's `split` must equal the split of its anchor recall (`anchor_recall_id`, or by
default the highest-grade recall listed first). (Ids above are placeholders.)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from evaluation import DEFAULT_ANNOTATIONS, DEFAULT_DB, DEFAULT_MANIFEST, SPLITS

GRADES = (1, 2, 3)
STATUSES = ("draft", "reviewed", "rejected")
MAX_QUERY_LENGTH = 500
CSV_COLUMNS = (
    "query_id", "query", "split", "recall_id", "grade", "reviewer_status", "annotator",
    "reviewer", "notes", "created_at", "split_fingerprint", "anchor_recall_id",
)
_QUERY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass
class Judgment:
    recall_id: int
    grade: int = 1


@dataclass
class Annotation:
    query_id: str
    query: str
    split: str
    relevant: list[Judgment]
    reviewer_status: str = "draft"
    annotator: str = ""
    reviewer: str = ""
    notes: str = ""
    created_at: str = ""
    split_fingerprint: str = ""
    anchor_recall_id: int | None = None

    @property
    def qrels(self) -> dict[int, int]:
        return {j.recall_id: j.grade for j in self.relevant}

    def to_json(self) -> dict:
        data = asdict(self)
        data["relevant"] = [asdict(j) for j in self.relevant]
        return data


@dataclass
class ValidationReport:
    valid: list[Annotation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def reviewed(self, split: str | None = None) -> list[Annotation]:
        return [a for a in self.valid
                if a.reviewer_status == "reviewed" and (split is None or a.split == split)]


# --- reading ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    raws = []
    for number, line in enumerate(path.read_text("utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path.name}:{number}: invalid JSON ({error.msg})") from error
        if not isinstance(raw, dict):
            raise ValueError(f"{path.name}:{number}: each line must be a JSON object")
        raws.append(raw)
    return raws


def read_csv(path: Path) -> list[dict]:
    """Group long-format rows into one raw record per query_id (order preserved)."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in ("query_id", "query", "split", "recall_id") if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path.name}: missing CSV columns {missing}")
        grouped: dict[str, dict] = {}
        for number, row in enumerate(reader, start=2):
            qid = (row.get("query_id") or "").strip()
            meta = {k: (row.get(k) or "").strip() for k in
                    ("query", "split", "reviewer_status", "annotator", "reviewer", "notes", "created_at",
                     "split_fingerprint", "anchor_recall_id")}
            if qid not in grouped:
                grouped[qid] = {"query_id": qid, **meta, "relevant": [], "_rows": [number]}
            else:
                record = grouped[qid]
                record["_rows"].append(number)
                for key, value in meta.items():
                    if value and record[key] and value != record[key]:
                        record.setdefault("_conflicts", []).append(f"row {number}: {key} differs")
                    elif value and not record[key]:
                        record[key] = value
            recall_id = (row.get("recall_id") or "").strip()
            if recall_id:
                grouped[qid]["relevant"].append(
                    {"recall_id": recall_id, "grade": (row.get("grade") or "").strip() or 1}
                )
    return list(grouped.values())


def read_any(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    if path.suffix.lower() == ".csv":
        return read_csv(path)
    raise ValueError("Annotation files must be .jsonl or .csv")


# --- validation ------------------------------------------------------------------------

def _initials(name: str) -> str:
    return "".join(name.split()).casefold()


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.casefold()) if len(t) > 1}


def _as_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def validate(
    raws: list[dict], split_of: dict[int, str], titles: dict[int, str] | None = None,
    assignment_fingerprint: str | None = None,
) -> ValidationReport:
    """Check every record against the corpus ids and the split manifest.

    Errors make a query unusable; warnings flag quality/leakage concerns for reviewers.
    """
    report = ValidationReport()
    titles = titles or {}
    seen_ids: set[str] = set()
    seen_text: dict[str, str] = {}

    for index, raw in enumerate(raws, start=1):
        qid = str(raw.get("query_id") or "").strip()
        where = f"query {qid or f'#{index}'}"
        errors: list[str] = [f"{where}: {c}" for c in raw.get("_conflicts", [])]

        if not _QUERY_ID.match(qid):
            errors.append(f"{where}: query_id must be 1-64 chars of letters, digits, _ . -")
        elif qid in seen_ids:
            errors.append(f"{where}: duplicate query_id")
        seen_ids.add(qid)

        query = " ".join(str(raw.get("query") or "").split())
        if not query:
            errors.append(f"{where}: query text is empty")
        elif len(query) > MAX_QUERY_LENGTH:
            errors.append(f"{where}: query longer than {MAX_QUERY_LENGTH} characters")

        split = str(raw.get("split") or "").strip()
        if split not in SPLITS:
            errors.append(f"{where}: split must be one of {SPLITS}")
        status = str(raw.get("reviewer_status") or "draft").strip()
        if status not in STATUSES:
            errors.append(f"{where}: reviewer_status must be one of {STATUSES}")
        reviewer = str(raw.get("reviewer") or "").strip()
        annotator = str(raw.get("annotator") or "").strip()
        if status == "reviewed" and not reviewer:
            errors.append(f"{where}: reviewed queries need a reviewer")
        elif status == "reviewed" and _initials(reviewer) == _initials(annotator):
            errors.append(f"{where}: reviewer must be a different person than the annotator")
        stamp = str(raw.get("split_fingerprint") or "").strip()
        if assignment_fingerprint and stamp and stamp != assignment_fingerprint:
            errors.append(f"{where}: labelled against a different split assignment "
                          "(manifest was rebuilt); re-check this query's split and labels")

        relevant_raw = raw.get("relevant") or []
        if not isinstance(relevant_raw, list):
            errors.append(f"{where}: relevant must be a list")
            relevant_raw = []
        if not relevant_raw and status != "rejected":
            errors.append(f"{where}: at least one relevant recall_id is required")

        judgments: list[Judgment] = []
        seen_docs: set[int] = set()
        for item in relevant_raw:
            item = item if isinstance(item, dict) else {"recall_id": item}
            rid = _as_int(item.get("recall_id"))
            grade = _as_int(item.get("grade", 1))
            if rid is None:
                errors.append(f"{where}: recall_id {item.get('recall_id')!r} is not an integer")
                continue
            if rid in seen_docs:
                errors.append(f"{where}: duplicate relevance label for recall {rid}")
                continue
            seen_docs.add(rid)
            if grade not in GRADES:
                errors.append(f"{where}: grade for recall {rid} must be one of {GRADES}")
                continue
            if rid not in split_of:
                errors.append(f"{where}: recall {rid} is not in the 1,000-record corpus")
                continue
            judgments.append(Judgment(rid, grade))

        # Full-index judgments: relevant recalls may come from any split. The query's split
        # is the split of its ANCHOR recall (explicit, else highest grade, first listed),
        # so validation/test queries still target documents held out from training.
        anchor: int | None = None
        if judgments and not errors:
            explicit = raw.get("anchor_recall_id")
            if explicit not in (None, ""):
                anchor = _as_int(explicit)
                if anchor is None or anchor not in {j.recall_id for j in judgments}:
                    errors.append(f"{where}: anchor_recall_id must be one of its relevant recalls")
            else:
                anchor = choose_anchor(judgments)
            if anchor is not None and not errors and split in SPLITS and split_of[anchor] != split:
                errors.append(
                    f"{where}: anchor recall {anchor} belongs to the {split_of[anchor]} split, "
                    f"but the query is labelled {split}"
                )

        if errors:
            report.errors.extend(errors)
            continue

        if assignment_fingerprint and not stamp:
            report.warnings.append(f"{where}: no split_fingerprint stamp (cannot prove it was "
                                   "labelled against the current split manifest)")
        normalized = query.casefold()
        if normalized in seen_text:
            report.warnings.append(f"{where}: same query text as {seen_text[normalized]}")
        seen_text.setdefault(normalized, qid)
        q_tokens = _tokens(query)
        for j in judgments:
            t_tokens = _tokens(titles.get(j.recall_id, ""))
            if q_tokens and t_tokens and len(q_tokens & t_tokens) / len(q_tokens | t_tokens) >= 0.8:
                report.warnings.append(
                    f"{where}: query nearly copies the title of recall {j.recall_id}; "
                    "consumer queries should be written independently"
                )

        report.valid.append(Annotation(
            query_id=qid, query=query, split=split, relevant=judgments,
            reviewer_status=status, annotator=annotator, reviewer=reviewer,
            notes=str(raw.get("notes") or ""), created_at=str(raw.get("created_at") or ""),
            split_fingerprint=stamp or (assignment_fingerprint or ""),
            anchor_recall_id=anchor,
        ))
    return report


def choose_anchor(judgments: list[Judgment]) -> int:
    """Highest grade wins; ties go to the first listed recall."""
    best = max(j.grade for j in judgments)
    return next(j.recall_id for j in judgments if j.grade == best)


# --- writing ---------------------------------------------------------------------------

def write_jsonl(annotations: list[Annotation], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            for a in annotations:
                handle.write(json.dumps(a.to_json(), ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def append_jsonl(annotation: Annotation, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(annotation.to_json(), ensure_ascii=False) + "\n")


def write_csv(annotations: list[Annotation], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for a in annotations:
            for j in a.relevant or [Judgment(0)]:
                writer.writerow({
                    "query_id": a.query_id, "query": a.query, "split": a.split,
                    "recall_id": j.recall_id or "", "grade": j.grade if j.recall_id else "",
                    "reviewer_status": a.reviewer_status, "annotator": a.annotator,
                    "reviewer": a.reviewer, "notes": a.notes, "created_at": a.created_at,
                    "split_fingerprint": a.split_fingerprint,
                    "anchor_recall_id": a.anchor_recall_id or "",
                })


# --- corpus context --------------------------------------------------------------------

def corpus_context(
    db_path: Path = DEFAULT_DB, manifest_path: Path = DEFAULT_MANIFEST
) -> tuple[dict[int, str], dict[int, str], dict]:
    """(split_of, titles, manifest), verifying the manifest matches the current corpus."""
    from evaluation.splits import load_manifest, split_lookup
    from retrieval.corpus import corpus_fingerprint, load_corpus

    from evaluation.splits import assignment_fingerprint

    manifest = load_manifest(manifest_path)
    if manifest.get("assignment_fingerprint") and assignment_fingerprint(
            split_lookup(manifest)) != manifest["assignment_fingerprint"]:
        raise RuntimeError("Split manifest assignments were edited by hand (fingerprint "
                           "mismatch). Rebuild it with the split tools instead.")
    corpus = load_corpus(db_path)
    if manifest["corpus_fingerprint"] != corpus_fingerprint(corpus):
        raise RuntimeError(
            "Split manifest was built for a different corpus. Rebuild it with "
            "`python -m evaluation.splits` (existing annotations must then be re-validated)."
        )
    return split_lookup(manifest), {r.recall_id: r.title for r in corpus}, manifest


def _print_report(report: ValidationReport) -> None:
    counts = {s: {st: 0 for st in STATUSES} for s in SPLITS}
    for a in report.valid:
        counts[a.split][a.reviewer_status] += 1
    print(f"valid queries: {len(report.valid)} | errors: {len(report.errors)} | "
          f"warnings: {len(report.warnings)}")
    for split in SPLITS:
        print(f"  {split:10} " + "  ".join(f"{st}={counts[split][st]}" for st in STATUSES))
    for line in report.errors:
        print(f"ERROR   {line}")
    for line in report.warnings:
        print(f"WARNING {line}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate / import / export annotations.")
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("validate", help="validate a .jsonl or .csv annotation file")
    v.add_argument("path", type=Path, nargs="?", default=DEFAULT_ANNOTATIONS)
    i = sub.add_parser("import", help="validate a .jsonl/.csv file and write canonical JSONL")
    i.add_argument("path", type=Path)
    i.add_argument("--out", type=Path, default=DEFAULT_ANNOTATIONS)
    i.add_argument("--merge", action="store_true", help="merge into --out (ids must not clash)")
    e = sub.add_parser("export", help="export JSONL to long-format CSV for spreadsheet review")
    e.add_argument("path", type=Path, nargs="?", default=DEFAULT_ANNOTATIONS)
    e.add_argument("--out", type=Path, required=True)
    for p in (v, i, e):
        p.add_argument("--db", type=Path, default=DEFAULT_DB)
        p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)

    try:
        split_of, titles, manifest = corpus_context(args.db, args.manifest)
        raws = read_any(args.path)
        if args.command == "import" and args.merge and args.out.is_file():
            raws = read_jsonl(args.out) + raws
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Cannot read annotations: {error}")
        return 2
    report = validate(raws, split_of, titles, manifest.get("assignment_fingerprint"))
    _print_report(report)
    if not report.ok:
        print("\nNothing written: fix the errors above and re-run.")
        return 1
    if args.command == "import":
        write_jsonl(report.valid, args.out)
        print(f"\nWrote {len(report.valid)} queries to {args.out.resolve()}")
    elif args.command == "export":
        write_csv(report.valid, args.out)
        print(f"\nWrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
