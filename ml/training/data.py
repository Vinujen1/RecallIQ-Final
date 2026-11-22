"""Training examples for contrastive fine-tuning, built from ONE split's documents.

Sources of (query, positive) pairs
1. Pseudo-queries derived deterministically from a document's own fields (weak
   supervision, no model or LLM involved): the title without recall boilerplate, the same
   without the brand plus the hazard, and product + hazard. They are NOT consumer queries
   and are labelled `pseudo:*`; they never count as evaluation data.
2. Reviewed human queries whose split is `train` (from data/annotations), when present.
   Only relevant recalls inside the train split are used.

Hard negatives come from the SAME split only: BM25 top-k for the query plus the positive's
nearest MiniLM neighbours (precomputed document embeddings). Candidates that may actually be
relevant are excluded: same split group (same product / event / title group), product-key
Jaccard >= `false_negative_jaccard` with the positive, and any judged-relevant recall.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evaluation.splits import jaccard, product_key_tokens, split_lookup
from retrieval.bm25 import BM25Index, tokenize
from retrieval.corpus import CorpusRecord

SPLITS_FOR_TRAINING = ("train", "validation")  # the test split is never used here


class LeakageError(RuntimeError):
    """A training/validation example references a document outside its split."""


@dataclass(frozen=True)
class TrainingPair:
    query: str
    positive_id: int
    negative_ids: tuple[int, ...]
    source: str          # pseudo:title_core | pseudo:generic_issue | pseudo:product_issue | human
    query_id: str


# --- pseudo-queries ----------------------------------------------------------------------

# "Archived - ", also with a mis-encoded dash ("Archived � ") present in the source CSV
_ARCHIVED = re.compile(r"^\s*archived\s*[^\w\s]+\s*", re.I)
_PREFIX = re.compile(r"^\s*(?:expanded|updated|update(?:d)?\s+to)?\s*recall\s*(?:notice)?\s*:\s*", re.I)
_COMPANY_RECALLS = re.compile(r"^.+?\brecalls\s+(?!due\b|of\b)(.{6,})$", re.I)
_WARNS = re.compile(r"^.*?\bwarns?\s+(?:consumers\s+)?(?:that\s+)?", re.I)
_TAIL = re.compile(
    r"\s+(?:recalled|recalls?|may\s+(?:contain|be|pose|have)|due\s+to|pose[sd]?|"
    r"previously\s+available|sold\s+(?:at|in|online))\b.*$", re.I)
_LEAD = re.compile(r"^(?:certain|various|several|some)\s+", re.I)
_BRAND = re.compile(r"^.*\bbrands?\b\s+(?:and\s+)?", re.I)


def title_core(title: str) -> str:
    text = _ARCHIVED.sub("", title)
    text = _PREFIX.sub("", text)
    text = _WARNS.sub("", text) if re.search(r"\bwarns?\b", text, re.I) else text
    company = _COMPANY_RECALLS.match(text)  # "<Company> recalls <Product> ..." -> product
    if company:
        text = company.group(1)
    text = _TAIL.sub("", text)
    text = _LEAD.sub("", text)
    return re.sub(r"\s+", " ", text).strip(" .,:;-")


def _issue(record: CorpusRecord) -> str:
    return (record.issue or "").strip()


# Consumer wording for the official `issue` vocabulary (written for this project; a person
# rarely says "Extraneous Material" or "Physical hazard"). Unlisted issues fall back to the
# lower-cased official term.
HAZARD_PLAIN: dict[str, tuple[str, ...]] = {
    "milk": ("milk allergy", "has milk not on the label", "dairy allergy"),
    "egg": ("egg allergy", "has egg not on the label"),
    "peanut": ("peanut allergy", "contains peanuts"),
    "tree nut": ("nut allergy", "contains nuts"),
    "sesame seeds": ("sesame allergy",),
    "soy": ("soy allergy",),
    "wheat": ("wheat allergy", "celiac"),
    "gluten": ("gluten", "celiac safe"),
    "mustard": ("mustard allergy",),
    "sulphites": ("sulphite sensitivity", "sulfites"),
    "salmonella": ("salmonella", "food poisoning"),
    "listeria": ("listeria", "food poisoning"),
    "e. coli o157:h7": ("e coli", "food poisoning"),
    "norovirus": ("norovirus", "made people sick"),
    "microbial hazard": ("bacteria", "contaminated"),
    "microbial contamination": ("bacteria", "contaminated"),
    "marine biotoxin": ("shellfish poisoning", "toxin"),
    "extraneous material": ("found pieces of plastic or metal in it", "foreign object inside"),
    "physical hazard": ("could hurt someone", "safety problem"),
    "fire hazard": ("could catch fire", "overheating"),
    "flammability hazard": ("catches fire easily", "not flame resistant"),
    "electrical hazard": ("electric shock", "sparks"),
    "burn hazard": ("could burn you", "gets too hot"),
    "choking hazard": ("choking risk for kids", "small parts"),
    "strangulation hazard": ("strangulation risk", "cord danger for kids"),
    "fall hazard": ("could fall", "tips over"),
    "laceration hazard": ("could cut you", "sharp edges"),
    "injury hazard": ("could injure", "safety problem"),
    "entrapment hazard": ("child could get trapped",),
    "chemical hazard": ("harmful chemicals", "toxic"),
    "chemical": ("harmful chemicals",),
    "labelling and packaging": ("wrong label", "labelling problem"),
    "non harmful (quality or spoilage)": ("tastes off", "spoiled"),
}
# Question frames; {p} = product without brand, {h} = consumer hazard wording.
_FRAMES_WITH_HAZARD = ("{p} {h}", "is the {p} i bought safe {h}", "{h} {p} recall",
                       "worried about my {p} {h}", "{p} recall canada {h}")
_FRAMES_NO_HAZARD = ("was {p} recalled", "{p} recall", "is my {p} safe")


def _pick(options: tuple[str, ...], key: str) -> str:
    """Deterministic, seed-free choice (stable across Python runs, unlike hash())."""
    return options[int(hashlib.md5(key.encode()).hexdigest(), 16) % len(options)]


def plain_hazard(issue: str, key: str) -> str:
    """Consumer wording for an official issue ("Fire hazard - Burn hazard" -> two parts)."""
    parts = [part.strip() for part in re.split(r"\s+-\s+", issue) if part.strip()]
    return " and ".join(_pick(HAZARD_PLAIN.get(part.casefold(), (part.casefold(),)), f"{key}:{part}")
                        for part in parts[:2])


def brand_of(core: str) -> str:
    """The text before an explicit "brand" word in the title core, else ''."""
    match = re.match(r"^(.*?)\bbrands?\b", core, re.I)
    return match.group(1).strip(" ,") if match else ""


def pseudo_queries(record: CorpusRecord, product: str | None, views: tuple[str, ...]) -> list[tuple[str, str]]:
    """[(view, query)] for one document; empty / duplicate / one-word views are dropped.

    `consumer_question` and `brand_product` are template-generated consumer-style
    paraphrases (brand dropped / product shortened, hazard in everyday words, question
    frame). They are weak supervision like the other views: generated, never reviewed,
    never evaluation data.
    """
    core = title_core(record.title)
    generic = _BRAND.sub("", core).strip()
    issue = _issue(record)
    short = " ".join((generic or core).split()[:6]).casefold()
    key = str(record.recall_id)
    if issue and short:
        consumer = _pick(_FRAMES_WITH_HAZARD, key).format(p=short, h=plain_hazard(issue, key))
    else:
        consumer = _pick(_FRAMES_NO_HAZARD, key).format(p=short) if short else ""
    brand = brand_of(core)
    tail = " ".join(generic.split()[-3:]).casefold() if generic else ""
    candidates = {
        "title_core": core,
        "generic_issue": f"{generic} {issue}".strip() if generic and generic != core else "",
        "product_issue": f"{product} {issue}".strip() if product and issue else "",
        "consumer_question": consumer,
        "brand_product": f"{brand} {tail}".strip() if brand and tail else "",
    }
    out, seen = [], set()
    for view in views:
        query = candidates.get(view, "")
        key = query.casefold()
        if len(tokenize(query)) >= 2 and key not in seen and key != record.search_text.casefold():
            seen.add(key)
            out.append((view, query))
    return out


# --- split guards ------------------------------------------------------------------------

def require_frozen(manifest: dict) -> None:
    if not manifest.get("frozen"):
        raise RuntimeError(
            "The split manifest is not frozen. Finish the split review "
            "(`python -m evaluation.split_review apply`) before training on real data."
        )


def group_lookup(manifest: dict, records: list[CorpusRecord]) -> dict[int, str]:
    """recall_id -> group key; records outside multi-record groups are their own group."""
    group_of = {r.recall_id: f"r{r.recall_id}" for r in records}
    for group in manifest.get("groups", []):
        for rid in group["recall_ids"]:
            group_of[rid] = group["group_id"]
    return group_of


def assert_no_leakage(pairs: list[TrainingPair], split_of: dict[int, str], split: str) -> None:
    for pair in pairs:
        for rid in (pair.positive_id, *pair.negative_ids):
            if split_of.get(rid) != split:
                raise LeakageError(
                    f"{pair.query_id}: recall {rid} is in split {split_of.get(rid)!r}, "
                    f"not {split!r}")


# --- building examples -------------------------------------------------------------------

def heldout_query_collisions(annotations: list) -> list[str]:
    """Train query ids whose text (case/space-normalised) equals a validation/test query —
    training on it would leak an evaluation query."""
    norm = lambda text: " ".join(tokenize(text))  # noqa: E731
    held_out = {norm(a.query) for a in annotations if a.split in ("validation", "test")}
    return sorted(a.query_id for a in annotations
                  if a.split == "train" and norm(a.query) in held_out)


def _human_queries(annotations: list, split: str, split_of: dict[int, str]) -> list[tuple[str, str, list[int], set[int]]]:
    """[(query_id, text, in-split positives (anchor first), all judged ids)] for reviewed
    queries of `split`. Relevant recalls outside the split are dropped as positives (they are
    held-out documents) but still excluded from the negatives."""
    out = []
    for a in annotations:
        if a.split != split or a.reviewer_status != "reviewed":
            continue
        judged = {j.recall_id for j in a.relevant}
        ordered = sorted(a.relevant, key=lambda j: (j.recall_id != a.anchor_recall_id, -j.grade))
        positives = [j.recall_id for j in ordered if split_of.get(j.recall_id) == split]
        if positives:
            out.append((a.query_id, a.query, positives, judged))
    return out


def build_pairs(
    records: list[CorpusRecord],
    manifest: dict,
    split: str,
    config,
    products: dict[int, str | None] | None = None,
    doc_embeddings: dict[int, np.ndarray] | None = None,
    annotations: list | None = None,
    verified_distinct: dict[int, set[int]] | None = None,
    num_negatives: int | None = None,
) -> list[TrainingPair]:
    """Deterministic (query, positive, hard negatives) examples using only `split` documents.

    `verified_distinct`: recall id -> recall ids a human reviewer judged DISTINCT (split
    review); those inside the split come first in the negative list (group rule still
    applies). `num_negatives` overrides `config.hard_negatives_per_pair` (validation pairs
    use `config.validation_hard_negatives` for every training variant).
    """
    wanted = config.hard_negatives_per_pair if num_negatives is None else num_negatives
    verified_distinct = verified_distinct if config.use_reviewed_distinct_negatives else None
    if split not in SPLITS_FOR_TRAINING:
        raise ValueError(f"training examples may only be built for {SPLITS_FOR_TRAINING}")
    split_of = split_lookup(manifest)
    docs = [r for r in records if split_of.get(r.recall_id) == split]
    if not docs:
        raise ValueError(f"no documents in the {split} split")
    ids = [r.recall_id for r in docs]
    index_of = {rid: i for i, rid in enumerate(ids)}
    by_id = {r.recall_id: r for r in docs}
    group_of = group_lookup(manifest, records)
    keys = {r.recall_id: product_key_tokens(r.title) for r in docs}
    bm25 = BM25Index([r.search_text for r in docs])
    products = products or {}

    sem = None
    if doc_embeddings is not None and config.semantic_candidates > 0:
        matrix = np.stack([doc_embeddings[rid] for rid in ids]).astype(np.float32)
        sem = matrix @ matrix.T

    def negatives(query: str, positive: int, judged: set[int]) -> tuple[int, ...]:
        if wanted == 0:
            return ()

        def allowed(rid: int) -> bool:
            return (rid != positive and rid not in judged
                    and group_of[rid] != group_of[positive]
                    and jaccard(keys[rid], keys[positive]) < config.false_negative_jaccard)

        # human-verified non-matches: the Jaccard filter is skipped (a person judged the
        # pair DISTINCT); the group and judged-relevant rules still apply
        verified = sorted(rid for rid in (verified_distinct or {}).get(positive, ())
                          if rid in by_id and rid != positive and rid not in judged
                          and group_of[rid] != group_of[positive])

        scores = bm25.scores(query)
        lexical = [ids[i] for i in np.argsort(-scores, kind="stable")[: config.bm25_candidates + 5]
                   if scores[i] > 0]
        semantic = []
        if sem is not None:
            row = sem[index_of[positive]]
            semantic = [ids[i] for i in np.argsort(-row, kind="stable")[: config.semantic_candidates + 5]]
        chosen: list[int] = verified[:wanted]
        for rid in _interleave(lexical[: config.bm25_candidates], semantic[: config.semantic_candidates]):
            if len(chosen) >= wanted:
                break
            if allowed(rid) and rid not in chosen:
                chosen.append(rid)
        return tuple(chosen)

    pairs: list[TrainingPair] = []
    for record in docs:
        for view, query in pseudo_queries(record, products.get(record.recall_id), config.pseudo_query_views):
            pairs.append(TrainingPair(query, record.recall_id,
                                      negatives(query, record.recall_id, {record.recall_id}),
                                      f"pseudo:{view}", f"{split}-{record.recall_id}-{view}"))
    if annotations and config.include_human_train_queries:
        for query_id, text, positives, judged in _human_queries(annotations, split, split_of):
            for positive in positives:
                if positive in by_id:
                    pairs.append(TrainingPair(text, positive, negatives(text, positive, judged),
                                              "human", query_id))
    assert_no_leakage(pairs, split_of, split)
    return pairs


def _interleave(a: list[int], b: list[int]) -> list[int]:
    out = []
    for i in range(max(len(a), len(b))):
        out += ([a[i]] if i < len(a) else []) + ([b[i]] if i < len(b) else [])
    return out


def normalize_query(text: str) -> str:
    return " ".join(tokenize(text))


def drop_text_collisions(pairs: list[TrainingPair], held_out_texts: set[str]) -> tuple[list[TrainingPair], int]:
    """Remove training pairs whose normalised query text equals a held-out (validation or
    test, generated or human) query, so no evaluation query text is ever trained on."""
    kept = [p for p in pairs if normalize_query(p.query) not in held_out_texts]
    return kept, len(pairs) - len(kept)


def reviewed_distinct_pairs(review_rows: list[dict]) -> dict[int, set[int]]:
    """recall id -> partner ids from human split-review rows decided DISTINCT."""
    out: dict[int, set[int]] = {}
    for row in review_rows:
        if (row.get("decision") or "").strip().upper() == "DISTINCT" and (row.get("reviewer") or "").strip():
            a, b = int(row["recall_id_a"]), int(row["recall_id_b"])
            out.setdefault(a, set()).add(b)
            out.setdefault(b, set()).add(a)
    return out


def pairs_fingerprint(pairs: list[TrainingPair]) -> str:
    digest = hashlib.sha256()
    for p in pairs:
        digest.update(json.dumps([p.query, p.positive_id, list(p.negative_ids), p.source],
                                 ensure_ascii=False).encode())
    return digest.hexdigest()


def write_pairs(pairs: list[TrainingPair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for p in pairs:
            handle.write(json.dumps({"query_id": p.query_id, "query": p.query,
                                     "positive_id": p.positive_id,
                                     "negative_ids": list(p.negative_ids), "source": p.source},
                                    ensure_ascii=False) + "\n")


def summarize(pairs: list[TrainingPair]) -> dict:
    from collections import Counter

    return {
        "pairs": len(pairs),
        "documents": len({p.positive_id for p in pairs}),
        "by_source": dict(sorted(Counter(p.source for p in pairs).items())),
        "pairs_with_hard_negatives": sum(1 for p in pairs if p.negative_ids),
        "pairs_without_hard_negative": sum(1 for p in pairs if not p.negative_ids),
        "fingerprint": pairs_fingerprint(pairs),
    }
