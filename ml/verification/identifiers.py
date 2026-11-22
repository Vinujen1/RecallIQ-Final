"""Deterministic product-identifier verification (post-retrieval, no embeddings).

Embeddings treat "XJ-500" and "XJ-600" as near-identical; recalls do not. After retrieval,
this module extracts identifiers from the consumer's text and from the recall evidence and
compares them EXACTLY (after normalisation):

| type        | extracted from                                                           | normalised to |
| ----------- | ------------------------------------------------------------------------ | ------------- |
| upc         | 12-digit UPC-A / 13-digit EAN-13 (spaces/hyphens allowed), valid check digit | 13 digits |
| model       | code(s) after "model / item / sku / style / part / cat. no." (lists allowed) | upper-case, no separators |
| lot         | code(s) after "lot / batch (code / no. / #)"                              | upper-case, no separators |
| best_before | date after "best before / BB / exp(iry) / use by" (ISO, CFIA "2024 JA 15", "January 15, 2024") | YYYY-MM-DD |

Per type the verdict is
- CONFIRMED_MATCH    - both sides have identifiers of this type and at least one is equal;
- CONFIRMED_MISMATCH - both sides have identifiers of this type and none is equal;
- UNKNOWN            - either side has none (missing is NEVER treated as a mismatch).
Overall: any mismatch -> CONFIRMED_MISMATCH (e.g. right UPC, different lot = not that lot);
else any match -> CONFIRMED_MATCH; else UNKNOWN.

Confirmed mismatches are deliberately conservative: a model/lot code that is a prefix of
the other or appears in the evidence with different spacing, and a date checked against a
range or list of dates, give UNKNOWN (see `_ambiguous`). A 12/13-digit number written as a
lot/model code is not also counted as a UPC.

Codes written in a query WITHOUT a keyword (e.g. "my xj500 heater") are "loose": they can
confirm a model match but never produce a mismatch, because they may not be identifiers.

A verdict only says whether identifiers agree with the retrieved notice. It never means a
product is safe: a notice can be missing from the index, or may not list every identifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

TYPES = ("upc", "model", "lot", "best_before")
NOT_A_SAFETY_GUARANTEE = (
    "Identifier checks compare your text with the retrieved recall notice only. No match, "
    "or no retrieved recall, does not mean the product is safe; check the official notice "
    "at recalls-rappels.canada.ca.")


class Verdict(str, Enum):
    MATCH = "confirmed_match"
    MISMATCH = "confirmed_mismatch"
    UNKNOWN = "missing_or_unknown"


@dataclass
class Identifiers:
    upc: set[str] = field(default_factory=set)
    model: set[str] = field(default_factory=set)
    lot: set[str] = field(default_factory=set)
    best_before: set[str] = field(default_factory=set)
    loose: set[str] = field(default_factory=set)  # keyword-less codes (queries only)

    def get(self, kind: str) -> set[str]:
        return getattr(self, kind)


@dataclass
class TypeResult:
    verdict: Verdict
    query_values: list[str]
    evidence_values: list[str]
    matched: list[str]


@dataclass
class VerificationResult:
    verdict: Verdict
    by_type: dict[str, TypeResult]
    note: str = NOT_A_SAFETY_GUARANTEE

    def to_json(self) -> dict:
        return {"verdict": self.verdict.value, "note": self.note,
                "by_type": {k: {"verdict": v.verdict.value, "query": v.query_values,
                                "evidence": v.evidence_values, "matched": v.matched}
                            for k, v in self.by_type.items()}}


# --- UPC -------------------------------------------------------------------------------

_UPC_CANDIDATE = re.compile(r"(?<![\w-])(\d(?:[ -]?\d){11,12})(?![\w-])")


def _check_digit_ok(digits: str) -> bool:
    """GS1 check digit for 12 (UPC-A) or 13 (EAN-13) digit codes."""
    body, check = digits[:-1], int(digits[-1])
    weights = (3, 1) if len(digits) == 12 else (1, 3)
    total = sum(int(d) * weights[i % 2] for i, d in enumerate(body))
    return (10 - total % 10) % 10 == check


def normalize_upc(text: str) -> str | None:
    digits = re.sub(r"[ -]", "", text)
    if len(digits) not in (12, 13) or not digits.isdigit() or not _check_digit_ok(digits):
        return None
    return digits.zfill(13)  # UPC-A 012345678905 == EAN-13 0012345678905


def extract_upcs(text: str) -> set[str]:
    return {upc for m in _UPC_CANDIDATE.finditer(text) if (upc := normalize_upc(m.group(1)))}


# --- model / lot codes --------------------------------------------------------------------

_CODE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9\-./]*[A-Za-z0-9])?")
_MODEL_KEY = re.compile(
    r"\b(?:models?|items?|sku|style|part|cat(?:alogue|alog)?\.?)\s*"
    r"(?:no\.?|nos\.?|numbers?|#)?\s*[:#]?\s*", re.I)
_LOT_KEY = re.compile(r"\b(?:lots?|batch(?:es)?)\s*(?:codes?|no\.?|numbers?|#)?\s*[:#]?\s*", re.I)
_LIST_SEP = re.compile(r"\s*(?:,|/|;|&|\band\b|\bor\b)\s*", re.I)
_LOOSE = re.compile(r"\b(?=[A-Za-z0-9\-]*\d)(?=[A-Za-z0-9\-]*[A-Za-z])[A-Za-z0-9][A-Za-z0-9\-]{2,}[A-Za-z0-9]\b")
# tokens that look like codes but are hazard names / units, never product identifiers
_NOT_CODES = {"O157", "O157H7", "H7", "COVID19", "MP3", "MP4", "USB3", "4K", "3D"}


def normalize_code(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _codes_after(key: re.Pattern, text: str) -> set[str]:
    """Codes following a keyword; comma/and-separated lists are followed while each item
    contains a digit (so "model 12A and 14B" gives both, "model 12A and blue" gives one)."""
    out = set()
    for m in key.finditer(text):
        rest, position = text[m.end():], 0
        while True:
            code = _CODE.match(rest, position)
            if not code:
                break
            value = normalize_code(code.group(0))
            if len(value) < 2 or not any(c.isdigit() for c in value):
                break
            out.add(value)
            sep = _LIST_SEP.match(rest, code.end())
            if not sep or sep.end() == code.end():
                break
            position = sep.end()
    return out - _NOT_CODES


# --- dates ---------------------------------------------------------------------------------

_MONTHS = {"JA": 1, "FE": 2, "MR": 3, "AL": 4, "MA": 5, "JN": 6, "JL": 7, "AU": 8, "SE": 9,
           "OC": 10, "NO": 11, "DE": 12}  # CFIA two-letter month codes
_FULL_MONTHS = {m: i for i, m in enumerate(
    ("january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"), start=1)}
_DATE_KEY = r"\b(?:best\s+before|bb|exp(?:iry|iration)?(?:\s+date)?|use\s+by)\b\s*(?:date)?\s*[:\-]?\s*"
_DATE_FORMS = (
    (re.compile(_DATE_KEY + r"(\d{4})-(\d{2})-(\d{2})\b", re.I), lambda y, m, d: (y, int(m), d)),
    (re.compile(_DATE_KEY + r"(\d{4})\s*([A-Z]{2})\s*(\d{1,2})\b", re.I),
     lambda y, m, d: (y, _MONTHS.get(m.upper()), d)),
    (re.compile(_DATE_KEY + r"([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})\b", re.I),
     lambda m, d, y: (y, _FULL_MONTHS.get(m.casefold()), d)),
)


def extract_dates(text: str) -> set[str]:
    """Only unambiguous date formats; dd/mm vs mm/dd numeric dates are deliberately skipped."""
    out = set()
    for pattern, order in _DATE_FORMS:
        for m in pattern.finditer(text):
            year, month, day = order(*m.groups())
            if not month:
                continue
            try:
                out.add(date(int(year), int(month), int(day)).isoformat())
            except ValueError:
                continue
    return out


# --- public API ----------------------------------------------------------------------------

def extract(text: str, loose: bool = False) -> Identifiers:
    """Identifiers in `text`. `loose=True` (queries) also collects keyword-less codes."""
    text = text or ""
    model, lot = _codes_after(_MODEL_KEY, text), _codes_after(_LOT_KEY, text)
    # a 12/13-digit number written as a lot or model code is not also a UPC
    upcs = {u for u in extract_upcs(text) if u.lstrip("0") not in {c.lstrip("0") for c in model | lot}}
    ids = Identifiers(upc=upcs, model=model, lot=lot, best_before=extract_dates(text))
    if loose:
        tagged = ids.model | ids.lot
        ids.loose = {c for m in _LOOSE.finditer(text)
                     if (c := normalize_code(m.group(0))) not in tagged and c not in _NOT_CODES}
    return ids


_RANGE_WORDS = re.compile(r"\b(?:to|through|thru|until|between)\b", re.I)
_ANY_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{4}\s*[A-Z]{2}\s*\d{1,2}\b|"
                       r"\b[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}\b")


def _ambiguous(kind: str, query: set[str], evidence: set[str], evidence_text: str) -> bool:
    """True when a non-equal pair could still be the same identifier, so the verdict must
    be UNKNOWN rather than a confirmed mismatch:
    - model/lot: one code is a prefix of the other ("500" vs "500S"), or the query code
      appears in the evidence once spaces/separators are removed ("24123A" in "24 123 A");
    - best_before: the evidence lists several dates or a range ("2024 JA 15 to 2024 JA 20").
    """
    if kind in ("model", "lot"):
        flat = normalize_code(evidence_text)
        return any(c in flat for c in query) or any(
            a.startswith(b) or b.startswith(a) for a in query for b in evidence)
    if kind == "best_before":  # any second date-shaped token, keyword or not, = a list
        return (len(_ANY_DATE.findall(evidence_text)) > 1
                or bool(_RANGE_WORDS.search(evidence_text)))
    return False


def verify(query_text: str, evidence_text: str) -> VerificationResult:
    """Compare identifiers in a consumer query with those in a recall's evidence text."""
    query, evidence = extract(query_text, loose=True), extract(evidence_text)
    by_type: dict[str, TypeResult] = {}
    for kind in TYPES:
        q, e = query.get(kind), evidence.get(kind)
        loose = query.loose if kind == "model" else set()  # may confirm, never refute
        matched = (q | loose) & e
        if matched:
            verdict = Verdict.MATCH
        elif q and e and not _ambiguous(kind, q, e, evidence_text):
            verdict = Verdict.MISMATCH
        else:
            verdict = Verdict.UNKNOWN
        by_type[kind] = TypeResult(verdict, sorted(q | loose), sorted(e), sorted(matched))
    verdicts = {r.verdict for r in by_type.values()}
    overall = (Verdict.MISMATCH if Verdict.MISMATCH in verdicts
               else Verdict.MATCH if Verdict.MATCH in verdicts else Verdict.UNKNOWN)
    return VerificationResult(overall, by_type)


def has_identifier(query_text: str) -> bool:
    """True when the query names at least one keyword-tagged identifier or a valid UPC."""
    ids = extract(query_text)
    return any(ids.get(k) for k in TYPES)
