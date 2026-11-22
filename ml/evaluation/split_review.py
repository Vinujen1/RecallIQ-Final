"""Human review of potentially related recall pairs that sit in different splits.

    python -m evaluation.split_review export     # write/refresh the review CSV
    python -m evaluation.split_review status     # validate decisions, show progress
    python -m evaluation.split_review apply      # apply decisions, rebuild + freeze splits

Decisions (one per pair, entered by a HUMAN reviewer):
  SAME_PRODUCT   the same product (re-notice, size/flavour variant, re-announcement)
                 -> always kept in one split (hard constraint)
  RELATED_EVENT  different products from one recall event / company / outbreak
                 -> policy `colocate` (default): kept in one split;
                    policy `allow-cross-split`: may stay apart, recorded as accepted risk
  DISTINCT       unrelated beyond product type -> no constraint; removed from residual risk

Nothing is merged on embedding similarity alone: the semantic audit only nominates pairs.
Assistant proposals are stored in separate `assistant_*` columns and never count as a
decision. See data/reviews/README.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from evaluation import DEFAULT_DB, DEFAULT_MANIFEST, REPO_ROOT
from evaluation.splits import (
    MANUAL_LINKS,
    _atomic_write,
    build_manifest,
    jaccard,
    load_manifest,
    product_key_tokens,
    split_lookup,
    write_manifest,
)
from retrieval.corpus import corpus_fingerprint, load_corpus

DECISIONS = ("SAME_PRODUCT", "RELATED_EVENT", "DISTINCT")
POLICIES = ("colocate", "allow-cross-split")
DEFAULT_POLICY = "colocate"
REVIEW_DIR = REPO_ROOT / "data" / "reviews"
DEFAULT_REVIEW = REVIEW_DIR / "split_pair_review.csv"
DEFAULT_PROPOSALS = REVIEW_DIR / "assistant_proposals.json"
DEFAULT_ANNOTATIONS = REPO_ROOT / "data" / "annotations"
MIN_RATIONALE = 10
# Reviewer names that indicate an automated "review". This is an ATTRIBUTION check, not
# authentication: it stops accidental AI attribution, it cannot prove who typed a row.
# Short tokens such as "ai" are deliberately absent so real names (e.g. "Ai Nguyen") pass.
AI_REVIEWER_TOKENS = frozenset({
    "assistant", "claude", "chatgpt", "gpt", "llm", "copilot", "gemini", "openai", "anthropic",
})
COPY_SIMILARITY = 0.8
COLUMNS = (
    "pair_id", "source", "recall_id_a", "recall_id_b", "split_a", "split_b",
    "title_a", "title_b", "category_a", "category_b", "url_a", "url_b",
    "jaccard", "cosine",
    "assistant_proposal", "assistant_flag", "assistant_rationale",
    "decision", "reviewer", "rationale", "reviewed_at", "corpus_fingerprint",
)
HUMAN_COLUMNS = ("decision", "reviewer", "rationale", "reviewed_at", "corpus_fingerprint")


def pair_id(a: int, b: int) -> str:
    lo, hi = sorted((int(a), int(b)))
    return f"P-{lo}-{hi}"


# --- collecting pairs ------------------------------------------------------------------

def collect_pairs(
    manifest: dict, semantic_candidates: list[tuple[int, int]] | None = None,
) -> dict[str, dict]:
    """Pairs that need a human decision, plus all manual links.

    Default: ambiguous (lexical) + semantic-audit pairs that currently CROSS splits.
    With `semantic_candidates` (every MiniLM pair >= threshold, see `all_semantic_pairs`)
    the scope is ALL candidates whatever their split, excluding pairs already in one merged
    group (those always share a split). Reviewing that set once means a later rebuild
    cannot expose new unreviewed crossing candidates.
    """
    split_of = split_lookup(manifest)
    pairs: dict[str, dict] = {}
    all_candidates = semantic_candidates is not None
    group_of = {rid: g["group_id"] for g in manifest.get("groups", []) for rid in g["recall_ids"]}

    def needs_review(a: int, b: int) -> bool:
        if all_candidates:
            return group_of.get(a) is None or group_of.get(a) != group_of.get(b)
        return split_of[a] != split_of[b]

    def add(a: int, b: int, source: str) -> None:
        key = pair_id(a, b)
        lo, hi = sorted((int(a), int(b)))
        entry = pairs.setdefault(key, {"a": lo, "b": hi, "sources": []})
        if source not in entry["sources"]:
            entry["sources"].append(source)

    for p in manifest.get("ambiguous_pairs", []):
        a, b = p["recall_ids"]
        if needs_review(a, b):
            add(a, b, "lexical_ambiguous")
    semantic = semantic_candidates if all_candidates else [
        tuple(p["recall_ids"]) for p in manifest.get("semantic_audit", {}).get("top_pairs", [])]
    for a, b in semantic:
        if needs_review(a, b):
            add(a, b, "semantic_audit")
    for a, b, _why in manifest.get("grouping", {}).get("manual_links", []):
        if a in split_of and b in split_of:  # links may name ids absent from this corpus
            add(a, b, "manual_link")
    return pairs


def _record_info(db_path: Path) -> dict[int, dict]:
    connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = connection.execute("SELECT id, title, category, url FROM recalls").fetchall()
    finally:
        connection.close()
    return {rid: {"title": t, "category": c or "", "url": u} for rid, t, c, u in rows}


def _cosines(embeddings_dir: Path | None, db_path: Path):
    if embeddings_dir is None:
        return None
    from retrieval.semantic import StaleIndexError, load_embeddings

    try:
        index = load_embeddings(embeddings_dir)
        index.check_matches(load_corpus(db_path))
    except (FileNotFoundError, StaleIndexError):
        return None
    row = {rid: i for i, rid in enumerate(index.recall_ids)}
    return lambda a, b: float(index.matrix[row[a]] @ index.matrix[row[b]])


def all_semantic_pairs(
    embeddings_dir: Path | None, db_path: Path, threshold: float,
) -> list[tuple[int, int]]:
    """Every pair with MiniLM cosine >= threshold, whatever its split.

    Raises instead of returning a partial list: an all-candidates export without the
    semantic half would look complete while missing pairs.
    """
    import numpy as np

    from retrieval.semantic import StaleIndexError, load_embeddings

    if embeddings_dir is None:
        raise RuntimeError("--all-candidates needs the MiniLM embeddings (--embeddings)")
    try:
        index = load_embeddings(embeddings_dir)
        index.check_matches(load_corpus(db_path))
    except (FileNotFoundError, StaleIndexError) as error:
        raise RuntimeError(f"--all-candidates needs current MiniLM embeddings: {error}") from error
    sims = index.matrix @ index.matrix.T
    rows, cols = np.where(np.triu(sims, k=1) >= threshold)
    ids = index.recall_ids
    return sorted((min(ids[r], ids[c]), max(ids[r], ids[c])) for r, c in zip(rows, cols))


# --- reading / writing the review file -------------------------------------------------

def read_review(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Review file not found: {path}. Run `python -m evaluation.split_review export`."
        )
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}")
        rows = [{k: (v or "").strip() for k, v in row.items() if k} for row in reader]
    duplicates = [k for k, n in Counter(r["pair_id"] for r in rows).items() if n > 1]
    if duplicates:
        raise ValueError(f"{path.name}: duplicate pair_id rows {duplicates[:5]}; keep one row per pair")
    return rows


def write_review(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
            writer.writeheader()
            writer.writerows({c: row.get(c, "") for c in COLUMNS} for row in rows)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _load_proposals(path: Path | None) -> dict:
    if not path or not path.is_file():
        return {}
    return json.loads(path.read_text("utf-8")).get("proposals", {})


def export(
    manifest: dict, db_path: Path, out: Path, embeddings_dir: Path | None = None,
    proposals_path: Path | None = DEFAULT_PROPOSALS, all_candidates: bool = False,
) -> dict:
    """Write/refresh the review CSV. Existing human decisions are never discarded.

    `all_candidates=True` lists every lexical/semantic candidate pair not already in one
    merged group (not only those crossing splits now) — see `collect_pairs`.
    """
    info = _record_info(db_path)
    split_of = split_lookup(manifest)
    cosine = _cosines(embeddings_dir, db_path)
    proposals = _load_proposals(proposals_path)
    threshold = manifest.get("semantic_audit", {}).get("threshold", 0.85)
    candidates = (all_semantic_pairs(embeddings_dir, db_path, threshold)
                  if all_candidates else None)

    existing = {row["pair_id"]: row for row in read_review(out)} if out.is_file() else {}
    rows: list[dict] = []
    for key, pair in sorted(collect_pairs(manifest, candidates).items()):
        a, b = pair["a"], pair["b"]
        proposal = proposals.get(f"{a}-{b}", {})
        key_j = jaccard(product_key_tokens(info[a]["title"]), product_key_tokens(info[b]["title"]))
        row = {
            "pair_id": key, "source": "+".join(pair["sources"]),
            "recall_id_a": a, "recall_id_b": b,
            "split_a": split_of[a], "split_b": split_of[b],
            "title_a": info[a]["title"], "title_b": info[b]["title"],
            "category_a": info[a]["category"], "category_b": info[b]["category"],
            "url_a": info[a]["url"], "url_b": info[b]["url"],
            "jaccard": f"{key_j:.3f}",
            "cosine": f"{cosine(a, b):.3f}" if cosine else "",
            "assistant_proposal": proposal.get("decision", ""),
            "assistant_flag": proposal.get("flag", ""),
            "assistant_rationale": proposal.get("rationale", ""),
        }
        previous = existing.pop(key, None)
        for column in HUMAN_COLUMNS:
            row[column] = previous.get(column, "") if previous else ""
        rows.append(row)
    # keep rows with ANY human input that are no longer flagged (e.g. now co-located):
    # decisions and partial reviews are records and are never discarded
    rows += [row for row in existing.values() if any(row.get(c) for c in HUMAN_COLUMNS)]
    write_review(rows, out)

    meta = {
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus_fingerprint": manifest["corpus_fingerprint"],
        "assignment_fingerprint": manifest.get("assignment_fingerprint"),
        "scope": "all_candidates" if all_candidates else "cross_split",
        "pairs": len(rows),
        "by_source": dict(Counter(s for r in rows for s in r["source"].split("+"))),
        "decided": sum(1 for r in rows if r["decision"]),
        "assistant_proposals": {
            "file": proposals_path.name if proposals_path and proposals_path.is_file() else None,
            "status": "AI-generated suggestions from titles/categories only — NOT human verified",
        },
    }
    _atomic_write(out.with_suffix(".meta.json"), json.dumps(meta, indent=2) + "\n")
    return meta


# --- validation ------------------------------------------------------------------------

def _is_ai_name(name: str) -> bool:
    return bool(set(re.findall(r"[a-z]+", name.casefold())) & AI_REVIEWER_TOKENS)


def _date_problem(text: str) -> str | None:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return ("reviewed_at must be YYYY-MM-DD (spreadsheets may rewrite dates, e.g. "
                "11/15/2026 — format the column as text)")
    try:
        day = date.fromisoformat(text)
    except ValueError:
        return "reviewed_at is not a real calendar date"
    if day > date.today() + timedelta(days=1):  # 1-day allowance for reviewer time zones
        return "reviewed_at is in the future"
    return None


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


def _copies(rationale: str, assistant_text: str) -> bool:
    a, b = _words(rationale), _words(assistant_text)
    return bool(a and b) and len(a & b) / len(a | b) >= COPY_SIMILARITY


def validate_review(
    rows: list[dict], corpus_ids: set[int], fingerprint: str, require_complete: bool = True,
    proposals: dict | None = None,
) -> tuple[list[dict], list[str], list[str]]:
    """Return (decided rows, errors, warnings).

    `proposals` (from assistant_proposals.json) is the source of truth for the assistant's
    text — the CSV's assistant_* columns could have been edited.
    """
    proposals = proposals or {}
    decided, errors, warnings = [], [], []
    seen: dict[str, str] = {}
    for row in rows:
        where = row.get("pair_id") or "?"
        try:
            a, b = int(row["recall_id_a"]), int(row["recall_id_b"])
        except ValueError:
            errors.append(f"{where}: recall ids must be integers")
            continue
        if a not in corpus_ids or b not in corpus_ids:
            errors.append(f"{where}: recall id not in the current corpus")
            continue
        if row["pair_id"] != pair_id(a, b):
            errors.append(f"{where}: pair_id does not match its recall ids")
            continue

        decision = row["decision"].upper()
        if not decision:
            (errors if require_complete else warnings).append(f"{where}: no decision yet")
            continue
        problems = []
        if decision not in DECISIONS:
            problems.append(f"decision must be one of {DECISIONS}")
        if not row["reviewer"]:
            problems.append("reviewer is required")
        elif _is_ai_name(row["reviewer"]):
            problems.append("reviewer must be a human team member (AI/assistant names are not accepted)")
        assistant_text = proposals.get(f"{min(a, b)}-{max(a, b)}", {}).get(
            "rationale", row.get("assistant_rationale", ""))
        if len(row["rationale"]) < MIN_RATIONALE:
            problems.append(f"rationale of at least {MIN_RATIONALE} characters is required")
        elif _copies(row["rationale"], assistant_text) or _copies(
                row["rationale"], row.get("assistant_rationale", "")):
            problems.append("rationale (nearly) copies the assistant proposal; write your own")
        date_problem = _date_problem(row["reviewed_at"])
        if date_problem:
            problems.append(date_problem)
        if row["corpus_fingerprint"] != fingerprint:
            problems.append("corpus_fingerprint missing or from a different corpus")
        if row["pair_id"] in seen and seen[row["pair_id"]] != decision:
            problems.append("conflicting decisions for the same pair")
        seen[row["pair_id"]] = decision
        if problems:
            errors += [f"{where}: {p}" for p in problems]
            continue
        decided.append({**row, "decision": decision, "a": a, "b": b})
    return decided, errors, warnings


# --- applying --------------------------------------------------------------------------

def effective_links(
    decided: list[dict], policy: str
) -> tuple[tuple[tuple[int, int, str], ...], list[tuple[int, int]]]:
    """Links to enforce, and manual links dropped by review."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    by_pair = {pair_id(r["a"], r["b"]): r for r in decided}

    def colocate(decision: str) -> bool:
        return decision == "SAME_PRODUCT" or (decision == "RELATED_EVENT" and policy == "colocate")

    links: list[tuple[int, int, str]] = []
    dropped: list[tuple[int, int]] = []
    for a, b, why in MANUAL_LINKS:
        review = by_pair.get(pair_id(a, b))
        if review is None or colocate(review["decision"]):
            note = f" [human: {review['decision']} by {review['reviewer']}]" if review else " [unreviewed]"
            links.append((a, b, why + note))
        else:
            dropped.append((a, b))
    manual = {pair_id(a, b) for a, b, _ in MANUAL_LINKS}
    for key, r in sorted(by_pair.items()):
        if key not in manual and colocate(r["decision"]):
            links.append((r["a"], r["b"],
                          f"human review {r['decision']} by {r['reviewer']}: {r['rationale']}"))
    return tuple(links), dropped


def _annotation_count(annotations_dir: Path) -> int:
    if not annotations_dir.is_dir():
        return 0
    return sum(
        sum(1 for line in path.read_text("utf-8-sig").splitlines() if line.strip())
        for path in annotations_dir.rglob("*.jsonl")
    )


def apply(
    review_path: Path, db_path: Path, manifest_path: Path, embeddings_dir: Path | None,
    policy: str = DEFAULT_POLICY, refreeze: bool = False,
    require_semantic_audit: bool = True,
    annotations_dir: Path = DEFAULT_ANNOTATIONS,
    allow_annotation_invalidation: bool = False,
    proposals_path: Path | None = DEFAULT_PROPOSALS,
) -> tuple[dict, list[str]]:
    """Apply complete human decisions; freezes only when (a) no unreviewed cross-split pairs
    remain after the rebuild and (b) the semantic audit ran and listed every crossing pair.
    Returns (manifest, messages)."""
    current = load_manifest(manifest_path)
    if current.get("frozen") and not refreeze:
        raise RuntimeError("Split manifest is already frozen; pass --refreeze to rebuild it.")
    corpus = load_corpus(db_path)
    fingerprint = corpus_fingerprint(corpus)
    decided, errors, _ = validate_review(
        read_review(review_path), {r.recall_id for r in corpus}, fingerprint,
        require_complete=True, proposals=_load_proposals(proposals_path))
    if errors:
        raise ValueError(f"{len(errors)} review problems, e.g. {errors[0]}")

    links, dropped = effective_links(decided, policy)
    try:
        shown = str(review_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        shown = str(review_path)
    review_info = {
        "applied": True,
        "file": shown,
        "file_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
        "related_event_policy": policy,
        "decisions": dict(Counter(r["decision"] for r in decided)),
        "reviewers": sorted({r["reviewer"] for r in decided}),
        "manual_links_dropped_by_review": [list(p) for p in dropped],
        "applied_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest = build_manifest(db_path, current["seed"], embeddings_dir=embeddings_dir,
                              manual_links=links, review=review_info, frozen=False)

    split_of = split_lookup(manifest)
    must_colocate = [r for r in decided if r["decision"] == "SAME_PRODUCT"
                     or (r["decision"] == "RELATED_EVENT" and policy == "colocate")]
    broken = [r["pair_id"] for r in must_colocate if split_of[r["a"]] != split_of[r["b"]]]
    if broken:
        raise RuntimeError(f"Reviewed pairs still cross splits: {broken}")

    reviewed_keys = {pair_id(r["a"], r["b"]) for r in decided}
    unreviewed = [k for k in collect_pairs(manifest) if k not in reviewed_keys]
    accepted = [r["pair_id"] for r in decided if r["decision"] == "RELATED_EVENT"
                and policy == "allow-cross-split" and split_of[r["a"]] != split_of[r["b"]]]
    audit = manifest.get("semantic_audit", {})
    listed = len(audit.get("top_pairs", []))
    crossing = audit.get("pairs_crossing_splits") or 0
    checks = manifest["checks"]
    checks["same_product_pairs_crossing_splits"] = 0
    checks["unreviewed_crossing_pairs"] = len(unreviewed)
    checks["accepted_related_event_pairs_crossing"] = len(accepted)
    checks["semantic_audit_available"] = bool(audit.get("available"))
    checks["semantic_audit_complete"] = bool(audit.get("available")) and listed >= crossing
    checks["review_scope"] = ("lexical ambiguous pairs + semantic-audit pairs (cosine >= "
                              f"{audit.get('threshold', 'n/a')}) + manual links; relations "
                              "below these thresholds were not reviewed")

    blockers = []
    if unreviewed:
        blockers.append(f"the rebuild produced {len(unreviewed)} cross-split pairs without a "
                        "decision (run `export`, review them, then `apply` again)")
    if require_semantic_audit and not checks["semantic_audit_available"]:
        blockers.append("the MiniLM semantic leakage audit did not run (build embeddings with "
                        "`python -m retrieval.embed`, or pass --skip-semantic-audit knowingly)")
    elif require_semantic_audit and not checks["semantic_audit_complete"]:
        blockers.append(f"the semantic audit listed only {listed} of {crossing} crossing pairs")

    changed = current.get("assignment_fingerprint") != manifest["assignment_fingerprint"]
    existing_annotations = _annotation_count(annotations_dir)
    if changed and existing_annotations and not allow_annotation_invalidation:
        raise RuntimeError(
            f"{existing_annotations} annotation records exist and this would change the split "
            "assignment; nothing written. Pass --allow-annotation-invalidation only if those "
            "annotations will be re-validated/re-labelled."
        )
    if refreeze and blockers:
        raise RuntimeError("Refusing to replace the frozen manifest with an unfrozen one: "
                           + "; ".join(blockers))

    messages = []
    if blockers:
        messages.append("NOT frozen: " + "; ".join(blockers) + ".")
    else:
        manifest["frozen"] = True
        manifest["frozen_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        messages.append("Split manifest FROZEN.")
    if current.get("frozen"):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = manifest_path.with_name(f"{manifest_path.stem}.frozen-{stamp}.json")
        backup.write_bytes(manifest_path.read_bytes())
        messages.append(f"Previous frozen manifest backed up to {backup.name}.")
    if changed:
        messages.append("Split assignment changed: any existing annotations must be re-validated.")
    write_manifest(manifest, manifest_path)
    return manifest, messages


# --- CLI -------------------------------------------------------------------------------

def _print_status(rows: list[dict], decided: list[dict], errors: list[str],
                  undecided: list[str]) -> None:
    by_source = Counter(s for r in rows for s in r["source"].split("+"))
    print(f"pairs: {len(rows)} | decided (valid): {len(decided)} | undecided: {len(undecided)} "
          f"| errors: {len(errors)}")
    print("by source:  " + ", ".join(f"{k}={v}" for k, v in sorted(by_source.items())))
    decisions = Counter(r["decision"] for r in decided)
    print("decisions:  " + (", ".join(f"{k}={v}" for k, v in sorted(decisions.items())) or "none"))
    print("reviewers:  " + (", ".join(sorted({r["reviewer"] for r in decided})) or "none"))
    for line in errors:
        print(f"ERROR   {line}")


def main(argv: list[str] | None = None) -> int:
    from retrieval.embed import DEFAULT_OUT as DEFAULT_EMBEDDINGS

    parser = argparse.ArgumentParser(description="Human review of cross-split recall pairs.")
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {name: sub.add_parser(name) for name in ("export", "status", "apply")}
    for p in commands.values():
        p.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
        p.add_argument("--db", type=Path, default=DEFAULT_DB)
        p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        p.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
        p.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS,
                       help="AI-assisted pair proposals (used to flag unverified rows)")
    commands["export"].add_argument(
        "--all-candidates", action="store_true",
        help="list every lexical/semantic candidate pair not already grouped, whatever its "
             "current split (one review round instead of one per rebuild)")
    commands["apply"].add_argument("--related-event-policy", choices=POLICIES,
                                   default=DEFAULT_POLICY)
    commands["apply"].add_argument("--refreeze", action="store_true")
    commands["apply"].add_argument("--skip-semantic-audit", action="store_true",
                                   help="freeze even if the MiniLM leakage audit cannot run (weaker)")
    commands["apply"].add_argument("--allow-annotation-invalidation", action="store_true")
    # Where existing relevance annotations live: apply refuses to change the split while any
    # exist (they would silently become invalid). Configurable so tests never read real data.
    commands["apply"].add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS)
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
        if args.command == "export":
            meta = export(manifest, args.db, args.review, args.embeddings, args.proposals,
                          all_candidates=args.all_candidates)
            print(json.dumps(meta, indent=2))
            print(f"\nReview file: {args.review.resolve()}")
            return 0
        corpus = load_corpus(args.db)
        if args.command == "status":
            rows = read_review(args.review)
            fingerprint = corpus_fingerprint(corpus)
            decided, errors, undecided = validate_review(
                rows, {r.recall_id for r in corpus}, fingerprint, require_complete=False,
                proposals=_load_proposals(args.proposals))
            _print_status(rows, decided, errors, undecided)
            print(f"\nManifest frozen: {manifest.get('frozen', False)}")
            print(f"corpus_fingerprint to enter in each reviewed row: {fingerprint}")
            return 1 if errors else 0
        result, messages = apply(
            args.review, args.db, args.manifest, args.embeddings, args.related_event_policy,
            args.refreeze, require_semantic_audit=not args.skip_semantic_audit,
            annotations_dir=args.annotations_dir,
            allow_annotation_invalidation=args.allow_annotation_invalidation,
            proposals_path=args.proposals)
        summary = {k: result[k] for k in ("counts", "frozen", "assignment_fingerprint", "checks")}
        print(json.dumps(summary, indent=2))
        for message in messages:
            print(message)
        return 0 if result["frozen"] else 4
    except (OSError, ValueError, RuntimeError) as error:  # OSError: e.g. CSV open in Excel
        print(f"Not done: {error}")
        return 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
