"""Deterministic checks applied to every generated answer before it is shown.

An LLM answer is accepted only if ALL of these pass; otherwise the pipeline shows the
deterministic template instead (see pipeline.explain). The checks are conservative on
purpose: a false rejection costs one template answer, a false acceptance could tell a
consumer something the official notice does not say.

- check_grounded      cited recall ids come from the evidence; every code/number-like token
                      (UPC, model, lot, year, recall id) is a WHOLE token of the evidence
                      ("123" is not grounded by "12345").
- names               every capitalised / all-caps word that does not start a sentence
                      (brand, product, lot code...) is a whole evidence token or on a small
                      allow-list ("Health Canada", "Class"...).
- quoted spans        text inside double quotes is copied verbatim from the evidence.
- URLs / domains      none, unless it is an evidence URL or part of one (citations are
                      attached server-side from cited ids, never written by the model).
- safety claims       never says or implies a product is safe / fine / harmless / not
                      recalled / not subject to a recall. The only allowed form is a hedge
                      whose object is the claim ("this does not mean it is safe"): hedges
                      are removed clause by clause and any claim left over rejects.
- identifier claims   never says a UPC / model / lot / code matches unless the deterministic
                      identifier verifier found a MATCH for a cited notice.
- injection echoes    no "system prompt" / "ignore previous instructions" style text.

These are lexical checks: they bound what an answer can assert (identifiers, names, links,
safety) but cannot prove that every paraphrased sentence is entailed by the evidence.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any, Iterable

MAX_ANSWER_CHARS = 4000

_CODE_LIKE = re.compile(r"\b(?=[\w-]*\d)[\w-]{3,}\b")
_TOKEN = re.compile(r"[\w-]+")
_URL = re.compile(r"(?:https?://|www\.)[^\s)\]>\"']+", re.I)
_BARE_DOMAIN = re.compile(
    r"\b(?:[a-z0-9-]+\.)+(?:com|ca|org|net|io|gov|info|co|us|uk|app|dev|xyz|biz|me|ly|link|"
    r"site|online)\b(?:/[^\s)\]>\"']*)?", re.I)
_QUOTED = re.compile(r"\"([^\"]{3,300})\"|“([^”]{3,300})”")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAUSE_SPLIT = re.compile(r"[;:—]|\s-\s|(?<=[.!?])\s+|\n+")
# A capitalised word preceded by whitespace that does not follow a sentence/clause start.
# `\s` in the look-behind makes the whitespace run start right after the previous word, so
# ". \n\nSince" is still seen as following "." (a sentence start, not a name).
_NAME_WORD = re.compile(r"(?<![.!?:;\n\"“(*\-•\s])\s+([A-Z][\w-]*)")
_NAME_ALLOW = {"canada", "canadian", "canadians", "health", "government", "food", "inspection",
               "agency", "cfia", "recalliq", "recall", "recalls", "upc", "lot", "model",
               "class", "official", "notice", "notices", "consumer", "product", "products",
               "safety", "i"}

_CLAIM_OBJECT = (r"(?:recalled|affected|impacted|included|covered|listed|subject|part|"
                 r"involved|concerned)")
_SAFETY_CLAIM = re.compile(
    r"\b(?:safe|fine|okay|ok|harmless|unaffected|risk[- ]free|"
    r"(?:(?:is|are|was|were|be|been)\s+not|isn'?t|aren'?t|wasn'?t|weren'?t|not\s+been)\s+"
    r"(?:\w+\s+)?" + _CLAIM_OBJECT + r"|"
    r"(?:not|n't)\s+(?:appear|seem|look)\w*\s+to\s+be\s+(?:\w+\s+)?" + _CLAIM_OBJECT + r"|"
    r"not\s+(?:subject|part)\s+(?:to|of)|not\s+(?:included|covered|listed)\s+in|"
    r"no\s+recalls?|keep\s+using|nothing\s+to\s+worry|no\s+need\s+to\s+worry)\b", re.I)
# A hedge whose object is the claim: "does not mean ... safe", "cannot confirm ... recalled".
# Removed to the end of its clause (a comma ends it), then the clause is re-checked, so
# "it is safe, this does not mean otherwise" and "cannot confirm anything, so it is safe"
# are still rejected.
_SAFETY_HEDGE = re.compile(
    r"\b(?:not|n't|never|cannot|can't|unable\s+to)\s+(?:\w+\s+){0,2}?(?:mean|means|guarantee|"
    r"guarantees|prove|proves|confirm|confirms|indicate|indicates|imply|implies|establish|"
    r"determine)\b[^.;,]{0,80}", re.I)

_IDENTIFIER_CLAIM = re.compile(
    r"\b(?:upc|barcode|lot|batch|model|item|sku|code|identifier)s?\b[^.]{0,80}?"
    r"\b(?:match|matches|matched|matching|same\s+as|identical)\b|"
    r"\b(?:match|matches|matched|matching)\b[^.]{0,80}?\b(?:upc|barcode|lot|batch|model|"
    r"sku|code|identifier)s?\b", re.I)
_NEGATION = re.compile(r"\b(?:not|n't|no|never|differ|differs|different|cannot|can't|"
                       r"could\s+not|unable|mismatch)\b", re.I)

_INJECTION_ECHO = re.compile(
    r"system\s+prompt|ignore\s+(?:all\s+|any\s+)?(?:previous|prior|above)\s+instructions|"
    r"developer\s+message|as\s+an\s+ai\s+language\s+model", re.I)


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def evidence_corpus(evidence: Iterable[Any]) -> str:
    """Every stored field of every evidence item, normalised, as one searchable string."""
    return _norm(" ".join(" ".join(str(v) for v in asdict(e).values() if v) for e in evidence))


def evidence_tokens(evidence: Iterable[Any]) -> set[str]:
    """Whole tokens of the evidence (casefolded), plus the parts of hyphenated tokens."""
    tokens = set(_TOKEN.findall(evidence_corpus(evidence)))
    return tokens | {part for t in tokens if "-" in t for part in t.split("-") if part}


def check_grounded(text: str, cited: list[int], evidence: list[Any]) -> str | None:
    """None if `text` is grounded in `evidence`, else why not: every cited recall id must be
    in the evidence, and every code/number-like token (UPC, model, lot, year...) in the text
    must be a whole token of the evidence fields."""
    allowed = {e.recall_id for e in evidence}
    extra = [c for c in cited if c not in allowed]
    if extra:
        return f"cites recall ids not in the evidence: {extra}"
    tokens = evidence_tokens(evidence)
    unknown = sorted({t for t in _CODE_LIKE.findall(text) if t.casefold() not in tokens})
    if unknown:
        return f"mentions codes/numbers not present in the evidence: {unknown[:5]}"
    return None


def unsupported_names(text: str, evidence: list[Any]) -> list[str]:
    """Capitalised / all-caps words (not starting a sentence) that are not evidence tokens."""
    tokens = evidence_tokens(evidence) | _NAME_ALLOW
    words = {m.group(1).strip("-") for m in _NAME_WORD.finditer(text)}
    return sorted(w for w in words if w and w.casefold() not in tokens
                  and w.casefold().removesuffix("s") not in tokens)


_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december")
_MONTH = r"(january|february|march|april|may|june|july|august|september|october|november|december)"
_WORD_DATE = re.compile(
    rf"\b{_MONTH}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b"      # October 1, 2012
    rf"|\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_MONTH},?\s+(\d{{4}})\b"     # 1 October 2012
    rf"|\b{_MONTH}\s+(\d{{4}})\b", re.I)                              # October 2012


def resolve_word_dates(text: str, evidence: list[Any]) -> tuple[str, str | None]:
    """Replace dates written in words with their ISO form, if the evidence has that date.

    Notices store ISO dates (2012-10-01); a model naturally writes "October 1, 2012". That
    is a faithful rewording, so it must not trip the name check on "October". Each such date
    must still match an evidence date exactly (a month-year must match an evidence date's
    month), otherwise the answer is rejected as an invented date. A word-date copied verbatim
    from the evidence (a few notices write dates in words) is left as is. Returns
    (text, reason)."""
    tokens = evidence_tokens(evidence)
    corpus = evidence_corpus(evidence)
    problem: list[str] = []

    def iso(match: re.Match) -> str:
        if _norm(match.group(0)) in corpus:
            return match.group(0)
        g = match.groups()
        if g[0]:
            year, month, day = g[2], g[0], g[1]
        elif g[3]:
            year, month, day = g[5], g[4], g[3]
        else:
            year, month, day = g[7], g[6], None
        stem = f"{year}-{_MONTHS.index(month.casefold()) + 1:02d}"
        if day:
            value = f"{stem}-{int(day):02d}"
            known = value if value in tokens else None
        else:  # month-year: stands for an evidence date in that month (substituted whole)
            known = next((t for t in sorted(tokens) if t.startswith(stem + "-")), None)
        if known is None:
            problem.append(match.group(0))
            return match.group(0)
        return known

    resolved = _WORD_DATE.sub(iso, text)
    if problem:
        return text, f"mentions dates not present in the evidence: {problem[:3]}"
    return resolved, None


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def unsupported_safety_claim(text: str) -> str | None:
    """The first clause that says or implies a product is safe / not recalled once hedges
    ("does not mean it is safe") are removed, or None."""
    for clause in _CLAUSE_SPLIT.split(text):
        if clause and clause.strip() and _SAFETY_CLAIM.search(_SAFETY_HEDGE.sub(" ", clause)):
            return clause.strip()
    return None


def unsupported_identifier_claim(text: str, cited: list[int], evidence: list[Any]) -> str | None:
    """A sentence claiming a code match when no cited notice has a deterministic MATCH."""
    cited_set = set(cited)
    if any(e.identifier_verdict == "MATCH" and e.recall_id in cited_set for e in evidence):
        return None
    for sentence in _sentences(text):
        if _IDENTIFIER_CLAIM.search(sentence) and not _NEGATION.search(sentence):
            return sentence.strip()
    return None


def validate_answer(text: str, cited: list[int], evidence: list[Any], *,
                    evidence_sufficient: bool = True) -> str | None:
    """None if a generated answer may be shown, else the (log-only) reason it was rejected."""
    if not text or not text.strip():
        return "empty answer"
    if len(text) > MAX_ANSWER_CHARS:
        return f"answer longer than {MAX_ANSWER_CHARS} characters"
    if evidence_sufficient and not cited:
        return "answer cites no evidence"
    if _INJECTION_ECHO.search(text):
        return "answer echoes prompt/instruction text"
    # Checks below run on the text with verified word-dates in ISO form (see resolve_word_dates).
    text, reason = resolve_word_dates(text, evidence)
    if reason:
        return reason
    reason = check_grounded(text, cited, evidence)
    if reason:
        return reason
    urls = {e.url.rstrip("/").casefold() for e in evidence}
    for url in _URL.findall(text):
        if url.rstrip(".,;:/").casefold() not in urls:
            return f"contains a URL that is not an evidence source: {url[:80]}"
    for domain in _BARE_DOMAIN.findall(_URL.sub(" ", text)):
        bare = domain.rstrip(".,;:/").casefold()
        if not any(bare in url for url in urls):
            return f"contains a domain that is not an evidence source: {domain[:80]}"
    names = unsupported_names(text, evidence)
    if names:
        return f"mentions names/codes not present in the evidence: {names[:5]}"
    corpus = evidence_corpus(evidence)
    for match in _QUOTED.finditer(text):
        quoted = _norm(match.group(1) or match.group(2))
        if quoted not in corpus:
            return f"quotes text not found in the evidence: {quoted[:80]!r}"
    claim = unsupported_safety_claim(text)
    if claim:
        return f"unsupported safety claim: {claim[:120]!r}"
    claim = unsupported_identifier_claim(text, cited, evidence)
    if claim:
        return f"claims an identifier match the verifier did not confirm: {claim[:120]!r}"
    return None
