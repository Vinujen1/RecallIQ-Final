"""Grounded explanation pipeline (proposal: retrieval-augmented explanations).

    consumer query -> retrieval (any search model) -> identifier verification
                   -> evidence selection -> grounded explanation -> official source links

Design rules (enforced here, tested in tests/rag/):
- Evidence = only fields stored for the retrieved recall records (title, product, issue,
  category, recall class, last updated, official URL). Nothing is fetched or inferred.
- Retrieved facts and the generated explanation are separate fields; the explanation is
  labelled with its generator ("template" or "llm:<provider>").
- Missing fields are said to be missing ("not listed in the indexed record"), never filled.
- A result is never presented as proof that the consumer's product is recalled; no result
  is never presented as proof that it is safe.
- The default generator is a deterministic template (no network, no API key). An LLM
  provider is selected with RECALLIQ_LLM_PROVIDER; none is bundled yet, so any value other
  than "none" falls back to the template and says why. Any future LLM output must pass
  `check_grounded` (cites only evidence recall ids; no codes/numbers absent from evidence)
  or the template is used instead. API keys are only ever read from the environment.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field

MAX_EVIDENCE = 3
DISCLAIMER = (
    "RecallIQ matches your description to official Government of Canada recall notices by "
    "text similarity. A match is not confirmation that your specific product is recalled, "
    "and no match does not mean it is safe. Check the brand, product name, UPC, lot code and "
    "dates against the official notice before acting.")


@dataclass
class EvidenceItem:
    rank: int
    recall_id: int
    title: str
    url: str
    product: str | None = None
    issue: str | None = None
    category: str | None = None
    recall_class: str | None = None
    last_updated: str | None = None
    identifier_verdict: str = "UNKNOWN"     # MATCH / MISMATCH / UNKNOWN for this record
    identifier_detail: dict = field(default_factory=dict)


@dataclass
class Explanation:
    generator: str                  # "template" | "llm:<provider>"
    text: str
    cited_recall_ids: list[int]
    fallback_reason: str | None = None


@dataclass
class GroundedAnswer:
    query: str
    model: str
    evidence_status: str            # "no_results" | "identifier_mismatch" | "evidence_found"
    retrieved_facts: list[EvidenceItem]
    explanation: Explanation
    sources: list[str]
    skipped_identifier_mismatches: int = 0
    disclaimer: str = DISCLAIMER

    def to_json(self) -> dict:
        return asdict(self)


def select_evidence(hits: list[dict], max_items: int = MAX_EVIDENCE) -> list[EvidenceItem]:
    """Top results in rank order, skipping records whose identifiers contradict the query
    (confirmed mismatches are kept only when nothing else is left)."""
    items = [EvidenceItem(**{k: h[k] for k in EvidenceItem.__dataclass_fields__ if k in h})
             for h in hits]
    agreeing = [i for i in items if i.identifier_verdict != "MISMATCH"]
    return (agreeing or items)[:max_items]


def _field(value: str | None) -> str:
    return value if value else "not listed in the indexed record"


def template_explanation(evidence: list[EvidenceItem], status: str, skipped: int = 0,
                         dense: bool = False) -> Explanation:
    """Deterministic, evidence-only wording: every value comes from a stored field."""
    if status == "no_results":
        return Explanation("template", (
            "No recall notice in the RecallIQ index (1,000 Canadian food and consumer-product "
            "recalls) matched this description. The index is a sample, so also search the "
            "official database at recalls-rappels.canada.ca."), [])
    lines = []
    top = evidence[0]
    if status == "identifier_mismatch":
        lines.append("The closest notices list identifiers that differ from the ones in your "
                     "description, so they may concern a different product, model or lot.")
    if skipped:
        lines.append(f"{skipped} higher-ranked notice(s) were set aside because their "
                     "identifiers conflict with the ones in your description.")
    label = ("Highest-ranked notice without conflicting identifiers" if skipped
             else "Highest-ranked official notice")
    lines.append(f"{label}: \"{top.title}\" (recall {top.recall_id}). "
                 f"Reason given: {_field(top.issue)}. Product: {_field(top.product)}. "
                 f"Category: {_field(top.category)}. Recall class: {_field(top.recall_class)}. "
                 f"Last updated: {_field(top.last_updated)}.")
    if top.identifier_verdict == "MATCH":
        lines.append("An identifier in your description exactly matches one in this record.")
    elif top.identifier_verdict == "UNKNOWN":
        lines.append("Identifiers could not be compared (missing from your description or "
                     "from the indexed record).")
    others = [f"\"{e.title}\" (recall {e.recall_id})" for e in evidence[1:]]
    if others:
        lines.append("Other possibly related notices: " + "; ".join(others) + ".")
    if dense:
        lines.append("Semantic models always return their nearest notices, even when none is "
                     "about your product.")
    lines.append("Read the official notice for affected sizes, codes and what to do.")
    return Explanation("template", " ".join(lines), [e.recall_id for e in evidence])


_CODE_LIKE = re.compile(r"\b(?=[\w-]*\d)[\w-]{3,}\b")


def check_grounded(text: str, cited: list[int], evidence: list[EvidenceItem]) -> str | None:
    """None if `text` is grounded in `evidence`, else why not: every cited recall id must be
    in the evidence, and every code/number-like token (UPC, model, lot, year...) in the text
    must appear in the evidence fields."""
    allowed = {e.recall_id for e in evidence}
    extra = [c for c in cited if c not in allowed]
    if extra:
        return f"cites recall ids not in the evidence: {extra}"
    corpus = " ".join(" ".join(str(v) for v in asdict(e).values() if v) for e in evidence).casefold()
    unknown = sorted({t for t in _CODE_LIKE.findall(text) if t.casefold() not in corpus})
    if unknown:
        return f"mentions codes/numbers not present in the evidence: {unknown[:5]}"
    return None


def configured_provider() -> str:
    return (os.environ.get("RECALLIQ_LLM_PROVIDER") or "none").strip().lower()


def explain(query: str, model: str, hits: list[dict], provider: str | None = None) -> GroundedAnswer:
    """`hits`: ranked dicts with EvidenceItem fields (rank, recall_id, title, url, product,
    issue, category, recall_class, last_updated, identifier_verdict, identifier_detail)."""
    evidence = select_evidence(hits)
    shown = {e.recall_id for e in evidence}
    first_shown = min((e.rank for e in evidence), default=0)
    skipped = sum(1 for h in hits if h.get("identifier_verdict") == "MISMATCH"
                  and h["recall_id"] not in shown and h["rank"] < first_shown)
    if not evidence:
        status = "no_results"
    elif all(e.identifier_verdict == "MISMATCH" for e in evidence):
        status = "identifier_mismatch"
    else:
        status = "evidence_found"
    explanation = template_explanation(evidence, status, skipped, dense=model != "bm25")
    provider = provider or configured_provider()
    if provider != "none":
        # No LLM adapter is bundled (no paid calls for the checkpoint). A future adapter must
        # return (text, cited ids) that pass check_grounded; otherwise this template stays.
        explanation.fallback_reason = (f"LLM provider {provider!r} is not implemented; "
                                       "deterministic template used")
    return GroundedAnswer(query=query, model=model, evidence_status=status,
                          retrieved_facts=evidence, explanation=explanation,
                          sources=[e.url for e in evidence],
                          skipped_identifier_mismatches=skipped)
