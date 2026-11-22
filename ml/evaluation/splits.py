"""Deterministic grouped + stratified train/validation/test partition of the corpus.

    python -m evaluation.splits [--db PATH] [--out JSON] [--seed 4110]

Grouping (the leakage unit) — records in one group always share a split:
  1. identical `title_group` (exact duplicate titles);
  2. same-product near-duplicates: token Jaccard >= MERGE_JACCARD between title-derived
     product keys (hazard clauses, "brand", "certain", ... removed), merged transitively;
  3. same explicit brand in the title ("<Brand> brand ...") within a domain — sibling
     products / the same recall event (e.g. two Ruffles chip flavours) stay together.
Pairs with AMBIGUOUS_JACCARD <= J < MERGE_JACCARD are NOT merged; they are listed and
reported when they land in different splits (residual leakage risk).

Stratification uses the official primary category (source data), never the heuristic
`product_family` — 458 of those are unverified and are not used for any decision here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from evaluation import DEFAULT_DB, DEFAULT_MANIFEST, SPLITS
from recall_data.grouping import product_text
from retrieval.corpus import corpus_fingerprint, load_corpus

SPLIT_VERSION = "1"
DEFAULT_SEED = 4110
DEFAULT_TARGETS: dict[str, float] = {"train": 0.6, "validation": 0.2, "test": 0.2}
MERGE_JACCARD = 0.6
AMBIGUOUS_JACCARD = 0.4

KEY_STOPWORDS = frozenset(
    "certain various brand brands product products recalled recall recalls archived "
    "the of and in with by from for an".split()
)
_ARCHIVED = re.compile(r"^archived\s*-\s*", re.IGNORECASE)
_WORD = re.compile(r"\w+", re.UNICODE)
_BRAND = re.compile(
    r"^(?:archived\s*-\s*)?(?:(?:undeclared|non-permitted) .+? in |consumption of )?"
    r"(?:certain |various )?(.+?)\s+brands?\b",
    re.IGNORECASE,
)
NOT_BRANDS = frozenset({"various", "certain", "several", "multiple", "some", "other", "all"})

# Ambiguous pairs that a reviewer judged to be the same company / recall event but that no
# lexical rule above captures. Merge-only (can never create cross-split leakage).
# Reviewed 2026-11-15 by the assistant from titles only — re-review as a team.
MANUAL_LINKS: tuple[tuple[int, int, str], ...] = (
    (54424, 54606, "same company (Mannarich Food / Ocean Chinese Food Products), same event"),
    (48883, 49109, "same hazard wording for beef products, E. coli O157:H7 — likely one event"),
    (44269, 44279, "same retailer (Farm Boy), same hazard (glass fragments in milk products)"),
    # Found by the semantic audit / lexical ambiguous list after the first split build:
    (51591, 54283, "same recall: Bed Handles Inc. notice and its 'Expanded Recall' re-announcement"),
    (75034, 75109, "same product: Taylor Farms Mexican Style Street Corn chopped kits"),
    (74037, 81901, "same manufacturer brand (Amana) packaged terminal / through-the-wall AC units"),
    (23298, 25346, "same manufacturer (Evenflo) high chairs"),
    (78316, 78040, "pistachio products, Salmonella — likely the same outbreak"),
    (74591, 74675, "cantaloupe products, Salmonella — likely the same outbreak"),
)


def extract_brand(title: str) -> str | None:
    """Brand named as '<Brand> brand ...' in the title, normalized; None if absent."""
    match = _BRAND.match(title)
    if not match:
        return None
    brand = re.sub(r"[^\w ]", "", match.group(1).casefold()).strip()
    brand = " ".join(brand.split())
    if not brand or brand in NOT_BRANDS or len(brand) > 40:
        return None
    return brand


@dataclass(frozen=True)
class SplitRecord:
    """The fields of one notice that decide its group and stratum."""
    recall_id: int
    title: str
    title_group: str
    domain: str
    primary_category: str
    needs_review: bool
    key_tokens: frozenset[str]


@dataclass
class Group:
    """Notices that must land in the same split (same product / re-notice / related event),
    with the human-readable reasons they were joined."""
    group_id: str
    recall_ids: list[int]
    rationale: list[str]
    stratum: str


def product_key_tokens(title: str) -> frozenset[str]:
    """Tokens describing the product: hazard text and generic recall words removed."""
    text = product_text(_ARCHIVED.sub("", title), None).casefold()
    return frozenset(t for t in _WORD.findall(text) if len(t) > 1 and t not in KEY_STOPWORDS)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """|A ∩ B| / |A ∪ B| of product-key tokens; 0 for an empty side (no evidence of overlap)."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_split_records(db_path: Path) -> list[SplitRecord]:
    """Read the corpus read-only and derive each notice's product-key tokens."""
    connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT r.id, r.title, r.title_group, r.domain, r.needs_review, "
            "  (SELECT category FROM recall_categories c WHERE c.recall_id = r.id "
            "   ORDER BY position LIMIT 1) "
            "FROM recalls r ORDER BY r.id"
        ).fetchall()
    finally:
        connection.close()
    return [
        SplitRecord(
            recall_id=rid, title=title, title_group=tg, domain=domain,
            primary_category=primary or "Uncategorized", needs_review=bool(review),
            key_tokens=product_key_tokens(title),
        )
        for rid, title, tg, domain, review, primary in rows
    ]


class _UnionFind:
    def __init__(self, items: list[int]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def build_groups(
    records: list[SplitRecord],
    merge_threshold: float = MERGE_JACCARD,
    ambiguous_threshold: float = AMBIGUOUS_JACCARD,
    manual_links: tuple[tuple[int, int, str], ...] = MANUAL_LINKS,
) -> tuple[list[Group], list[dict]]:
    """Union-find over notices that would leak information if split apart.

    Splitting by record is not enough: the same product is often recalled twice with slightly
    different titles, and a model trained on one notice would "recognise" its twin in the
    test split. Notices are merged when they share a title group, have product-key Jaccard
    >= `merge_threshold`, share a brand, or are manually linked. Pairs between the two
    thresholds are returned as `ambiguous` for human review instead of being guessed."""
    by_id = {r.recall_id: r for r in records}
    uf = _UnionFind(sorted(by_id))
    reasons: dict[int, list[str]] = defaultdict(list)  # keyed by member id
    ambiguous: list[dict] = []

    by_title_group: dict[str, list[int]] = defaultdict(list)
    for r in records:
        by_title_group[r.title_group].append(r.recall_id)
    for tg, ids in by_title_group.items():
        for other in ids[1:]:
            uf.union(ids[0], other)
            reasons[other].append(f"same title_group {tg} as #{ids[0]}")

    by_brand: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in sorted(records, key=lambda r: r.recall_id):
        brand = extract_brand(r.title)
        if brand:
            by_brand[(r.domain, brand)].append(r.recall_id)
    for (_, brand), ids in by_brand.items():
        for other in ids[1:]:
            uf.union(ids[0], other)
            reasons[other].append(f"same brand '{brand}' as #{ids[0]}")

    for a, b, why in manual_links:
        if a in by_id and b in by_id:
            uf.union(a, b)
            reasons[b].append(f"manual link with #{a}: {why}")

    ordered = sorted(records, key=lambda r: r.recall_id)
    for a, b in itertools.combinations(ordered, 2):
        if a.domain != b.domain:
            continue
        j = jaccard(a.key_tokens, b.key_tokens)
        if j >= merge_threshold:
            uf.union(a.recall_id, b.recall_id)
            reasons[b.recall_id].append(f"product-key Jaccard {j:.2f} with #{a.recall_id}")
        elif j >= ambiguous_threshold:
            ambiguous.append({"recall_ids": [a.recall_id, b.recall_id], "jaccard": round(j, 3),
                              "titles": [a.title, b.title]})

    members: dict[int, list[int]] = defaultdict(list)
    for rid in sorted(by_id):
        members[uf.find(rid)].append(rid)

    groups = []
    for root, ids in sorted(members.items()):
        strata = Counter(f"{by_id[i].domain}::{by_id[i].primary_category}" for i in ids)
        stratum = sorted(strata.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        rationale = [reason for i in ids for reason in reasons[i]] or ["singleton"]
        groups.append(Group(f"g{root}", ids, rationale, stratum))
    return groups, ambiguous


def assign_splits(
    groups: list[Group], seed: int = DEFAULT_SEED, targets: dict[str, float] = DEFAULT_TARGETS
) -> dict[str, str]:
    """Greedy grouped assignment that tracks per-stratum and global target counts.

    Deterministic: groups are shuffled once with `seed`, then placed largest first; each
    goes to the split with the largest stratum deficit, then global deficit, then
    split order.
    """
    if set(targets) != set(SPLITS) or abs(sum(targets.values()) - 1.0) > 1e-9:
        raise ValueError(f"targets must cover {SPLITS} and sum to 1")
    rng = np.random.default_rng(seed)
    ordered = sorted(groups, key=lambda g: g.group_id)
    ordered = [ordered[i] for i in rng.permutation(len(ordered))]
    ordered.sort(key=lambda g: -len(g.recall_ids))  # stable: ties keep shuffled order

    total = sum(len(g.recall_ids) for g in groups)
    stratum_total = Counter()
    for g in groups:
        stratum_total[g.stratum] += len(g.recall_ids)
    stratum_count: dict[str, Counter] = defaultdict(Counter)
    global_count: Counter = Counter()
    # Multi-record groups get their own 60/20/20 quota (by record count) so validation and
    # test also contain sibling/near-duplicate products — otherwise the largest-deficit
    # rule would send every multi-record group to train.
    multi_total = sum(len(g.recall_ids) for g in groups if len(g.recall_ids) > 1)
    multi_count: Counter = Counter()

    assignment: dict[str, str] = {}
    for g in ordered:
        size = len(g.recall_ids)

        def priority(split: str) -> tuple[float, float, float, int]:
            multi_deficit = targets[split] * multi_total - multi_count[split] if size > 1 else 0.0
            stratum_deficit = targets[split] * stratum_total[g.stratum] - stratum_count[g.stratum][split]
            global_deficit = targets[split] * total - global_count[split]
            return (multi_deficit, stratum_deficit, global_deficit, -SPLITS.index(split))

        split = max(SPLITS, key=priority)
        assignment[g.group_id] = split
        stratum_count[g.stratum][split] += size
        global_count[split] += size
        if size > 1:
            multi_count[split] += size
    return assignment


def assignment_fingerprint(split_of: dict[int, str]) -> str:
    """Hash of the exact id -> split mapping; annotations are stamped with it."""
    digest = hashlib.sha256()
    for rid in sorted(split_of):
        digest.update(f"{rid}:{split_of[rid]}\n".encode())
    return digest.hexdigest()


def semantic_cross_split_pairs(
    db_path: Path, embeddings_dir: Path, split_of: dict[int, str],
    threshold: float = 0.85, limit: int = 50,
) -> dict:
    """Independent leakage audit: most similar cross-split pairs by MiniLM cosine.

    Lexical grouping cannot see the same product described differently; this lists
    candidates for MANUAL review (they are not merged automatically).
    """
    from retrieval.semantic import StaleIndexError, load_embeddings

    try:
        index = load_embeddings(embeddings_dir)
        index.check_matches(load_corpus(db_path))
    except (FileNotFoundError, StaleIndexError) as error:
        return {"available": False, "reason": str(error)}
    sims = index.matrix @ index.matrix.T
    ids = index.recall_ids
    splits = np.array([split_of[i] for i in ids])
    rows, cols = np.where(np.triu(sims, k=1) >= threshold)
    pairs = [
        (float(sims[r, c]), ids[r], ids[c]) for r, c in zip(rows, cols)
        if splits[r] != splits[c]
    ]
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    return {
        "available": True,
        "threshold": threshold,
        "model_revision": index.metadata.get("model_revision"),
        "pairs_crossing_splits": len(pairs),
        "top_pairs": [
            {"cosine": round(s, 4), "recall_ids": [a, b], "splits": [split_of[a], split_of[b]]}
            for s, a, b in pairs[:limit]
        ],
    }


def find_violations(
    split_of: dict[int, str], records: list[SplitRecord], groups: list[Group]
) -> list[str]:
    """Hard leakage checks: every record once; no title group / merged group across splits."""
    problems = []
    ids = [r.recall_id for r in records]
    if sorted(split_of) != sorted(ids):
        problems.append("split assignment does not cover every corpus record exactly once")
    if any(s not in SPLITS for s in split_of.values()):
        problems.append("unknown split name in assignment")
    by_tg: dict[str, set[str]] = defaultdict(set)
    for r in records:
        if r.recall_id in split_of:
            by_tg[r.title_group].add(split_of[r.recall_id])
    problems += [f"title_group {tg} spans splits {sorted(s)}" for tg, s in by_tg.items() if len(s) > 1]
    for g in groups:
        spans = {split_of.get(i) for i in g.recall_ids}
        if len(spans) > 1:
            problems.append(f"group {g.group_id} spans splits {sorted(map(str, spans))}")
    return problems


def build_manifest(
    db_path: Path = DEFAULT_DB, seed: int = DEFAULT_SEED,
    targets: dict[str, float] = DEFAULT_TARGETS, embeddings_dir: Path | None = None,
    manual_links: tuple[tuple[int, int, str], ...] = MANUAL_LINKS,
    review: dict | None = None, frozen: bool = False,
) -> dict:
    """`manual_links` / `review` / `frozen` are supplied by `evaluation.split_review apply`
    once human review decisions exist; plain `evaluation.splits` uses the defaults."""
    records = load_split_records(db_path)
    if not records:
        raise ValueError("Corpus is empty")
    groups, ambiguous = build_groups(records, manual_links=manual_links)
    group_split = assign_splits(groups, seed, targets)
    split_of = {i: group_split[g.group_id] for g in groups for i in g.recall_ids}
    violations = find_violations(split_of, records, groups)
    if violations:
        raise RuntimeError(f"Split leakage checks failed: {violations[:5]}")

    by_id = {r.recall_id: r for r in records}
    counts = Counter(split_of.values())
    n = len(records)
    for pair in ambiguous:
        pair["splits"] = [split_of[i] for i in pair["recall_ids"]]
    group_of = {i: g.group_id for g in groups for i in g.recall_ids}
    ambiguous = [p for p in ambiguous  # drop pairs later joined by brand/manual rules
                 if group_of[p["recall_ids"][0]] != group_of[p["recall_ids"][1]]]
    crossing = [p for p in ambiguous if len(set(p["splits"])) > 1]

    strata: dict[str, dict[str, int]] = defaultdict(lambda: {s: 0 for s in SPLITS})
    for rid, split in split_of.items():
        r = by_id[rid]
        strata[f"{r.domain}::{r.primary_category}"][split] += 1

    multi = [g for g in groups if len(g.recall_ids) > 1]
    semantic = (
        semantic_cross_split_pairs(db_path, embeddings_dir, split_of)
        if embeddings_dir is not None else {"available": False, "reason": "not requested"}
    )
    for pair in semantic.get("top_pairs", []):
        pair["titles"] = [by_id[i].title for i in pair["recall_ids"]]
    return {
        "split_version": SPLIT_VERSION,
        "frozen": frozen,
        "review": review or {"applied": False,
                             "note": "no human split-review decisions applied yet"},
        "seed": seed,
        "targets": targets,
        "assignment_fingerprint": assignment_fingerprint(split_of),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus_fingerprint": corpus_fingerprint(load_corpus(db_path)),
        "source_dataset": _dataset_metadata(db_path),
        "grouping": {
            "unit": "same-product groups (title_group + product-key Jaccard >= "
                    f"{MERGE_JACCARD} + same explicit brand, transitive); stratified by "
                    "domain::primary official category",
            "merge_jaccard": MERGE_JACCARD,
            "ambiguous_jaccard": AMBIGUOUS_JACCARD,
            "manual_links": [list(link) for link in manual_links],
            "uses_heuristic_product_family": False,
        },
        "counts": {s: counts[s] for s in SPLITS},
        "achieved_fractions": {s: round(counts[s] / n, 4) for s in SPLITS},
        "groups_total": len(groups),
        "multi_record_groups": len(multi),
        "multi_record_groups_per_split": {
            s: sum(1 for g in multi if group_split[g.group_id] == s) for s in SPLITS
        },
        "largest_group_size": max(len(g.recall_ids) for g in groups),
        "checks": {
            "records_assigned_once": True,
            "title_groups_crossing_splits": 0,
            "merged_groups_crossing_splits": 0,
            "ambiguous_pairs_total": len(ambiguous),
            "ambiguous_pairs_crossing_splits": len(crossing),
            "semantic_pairs_crossing_splits": semantic.get("pairs_crossing_splits"),
        },
        "semantic_audit": semantic,
        "unverified_product_family_records": {
            s: sum(1 for rid, sp in split_of.items() if sp == s and by_id[rid].needs_review)
            for s in SPLITS
        },
        "residual_leakage_risks": [
            f"{len(crossing)} ambiguous near-duplicate pairs (Jaccard {AMBIGUOUS_JACCARD}-"
            f"{MERGE_JACCARD}) fall in different splits; review `ambiguous_pairs`.",
            "Grouping is lexical: the same product described with different words, or one "
            "company/recall event without an explicit '<Brand> brand' phrase (e.g. "
            "'Mannarich Food ...' notices), is not detected.",
            "Heuristic product families (458 unverified) were not used; category-level "
            "similarity across splits is intended, not leakage.",
        ],
        "strata": dict(sorted(strata.items())),
        "groups": [
            {"group_id": g.group_id, "split": group_split[g.group_id], "stratum": g.stratum,
             "recall_ids": g.recall_ids, "rationale": g.rationale}
            for g in multi
        ],
        "ambiguous_pairs": sorted(crossing, key=lambda p: -p["jaccard"])
        + sorted([p for p in ambiguous if p not in crossing], key=lambda p: -p["jaccard"]),
        "assignments": {str(rid): split_of[rid] for rid in sorted(split_of)},
    }


def _dataset_metadata(db_path: Path) -> dict:
    connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
    except sqlite3.Error:
        rows = {}
    finally:
        connection.close()
    return {k: rows.get(k) for k in ("input_sha256", "seed", "pipeline_version")}


def write_manifest(manifest: dict, path: Path = DEFAULT_MANIFEST) -> None:
    """Write the manifest (JSON) and a flat recall_id,split,group_id CSV, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    group_of = {i: g["group_id"] for g in manifest["groups"] for i in g["recall_ids"]}
    lines = ["recall_id,split,group_id"] + [
        f"{rid},{split},{group_of.get(int(rid), 'g' + rid)}"
        for rid, split in manifest["assignments"].items()
    ]
    _atomic_write(path.with_suffix(".csv"), "\n".join(lines) + "\n")


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    """Read the split manifest, with a hint on how to create it if missing."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Split manifest not found: {path}. Run `python -m evaluation.splits` first."
        )
    return json.loads(path.read_text("utf-8"))


def split_lookup(manifest: dict) -> dict[int, str]:
    """recall_id -> "train" | "validation" | "test"."""
    return {int(rid): split for rid, split in manifest["assignments"].items()}


def manifest_sha256(path: Path) -> str:
    """Hash of the exact manifest file, recorded with results to prove which split was used."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    from retrieval.embed import DEFAULT_OUT as DEFAULT_EMBEDDINGS

    parser = argparse.ArgumentParser(description="Build grouped train/validation/test splits.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS,
                        help="MiniLM embeddings for the semantic leakage audit")
    parser.add_argument("--force", action="store_true",
                        help="overwrite a manifest whose assignments differ (invalidates annotations)")
    args = parser.parse_args(argv)

    manifest = build_manifest(args.db, args.seed, embeddings_dir=args.embeddings)
    if args.out.is_file() and load_manifest(args.out).get("frozen"):
        print("Refusing to overwrite: this split manifest is FROZEN (human review decisions "
              "applied). Change splits only through `python -m evaluation.split_review apply "
              "--refreeze` after updating the review file.")
        return 1
    if args.out.is_file() and not args.force:
        previous = load_manifest(args.out)
        if previous.get("assignment_fingerprint") != manifest["assignment_fingerprint"]:
            print("Refusing to overwrite: the new split assignment differs from the existing "
                  "manifest. Annotations are stamped with the old assignment; re-run with "
                  "--force only if you accept re-validating/re-labelling them.")
            return 1
    write_manifest(manifest, args.out)
    summary = {k: manifest[k] for k in (
        "seed", "assignment_fingerprint", "counts", "achieved_fractions", "groups_total",
        "multi_record_groups", "multi_record_groups_per_split", "largest_group_size",
        "checks", "unverified_product_family_records",
    )}
    print(json.dumps(summary, indent=2))
    print(f"\nManifest: {args.out.resolve()}\nCSV:      {args.out.with_suffix('.csv').resolve()}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
