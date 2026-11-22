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
- The default generator is a deterministic template (no network, no API key). With
  provider "openai" (rag/llm.py) and sufficient evidence, an OpenAI model writes the answer
  from the same evidence; it is shown only if it passes rag.guards.validate_answer
  (grounded codes/ids, verbatim quotes, no URLs, no safety or unverified identifier claims).
  Any API error, timeout, refusal or failed check falls back to the template and says why.
- The LLM is never called without evidence (no results, or only identifier-conflicting
  notices): the deterministic template handles those cases.
- API keys are only ever read from configuration / the environment.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from rag.guards import check_grounded, validate_answer  # noqa: F401  (check_grounded re-exported)
from rag.llm import PROMPT_VERSION, LLMConfig, LLMError, generate

logger = logging.getLogger("recalliq.rag")

MAX_EVIDENCE = 3
DISCLAIMER = (
    "RecallIQ matches your description to official Government of Canada recall notices by "
    "text similarity. A match is not confirmation that your specific product is recalled, "
    "and no match does not mean it is safe. Check the brand, product name, UPC, lot code and "
    "dates against the official notice before acting.")


@dataclass
class EvidenceItem:
    """One retrieved notice as evidence: stored fields copied verbatim (None = not listed)
    plus its deterministic identifier verdict for this query."""
    rank: int
    recall_id: int
    title: str
    url: str
    product: str | None = None
    issue: str | None = None
    category: str | None = None
    recall_class: str | None = None
    last_updated: str | None = None
    what_you_should_do: str | None = None   # official "what you should do" text, verbatim
    identifier_verdict: str = "UNKNOWN"     # MATCH / MISMATCH / UNKNOWN for this record
    identifier_detail: dict = field(default_factory=dict)


@dataclass
class GenerationStats:
    """What happened in the LLM step (also logged). Tokens/latency are None when no call
    was made or the provider did not report them."""
    provider: str
    model: str | None
    outcome: str                    # accepted | rejected | error:<code> | skipped:<status>
    prompt_version: str = PROMPT_VERSION
    latency_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class Explanation:
    """The text shown to the consumer and how it was produced (template or LLM)."""
    generator: str                  # "template" | "llm:<provider>"
    text: str
    cited_recall_ids: list[int]
    fallback_reason: str | None = None
    evidence_sufficient: bool | None = None   # the LLM's own judgement; None for the template
    generation: GenerationStats | None = None


@dataclass
class GroundedAnswer:
    """Everything POST /explain returns: evidence, explanation, official sources, disclaimer.
    Retrieved facts and generated text are separate fields, so the UI never mixes them."""
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


FALLBACK_MESSAGES = {
    "not_configured": "AI answers are switched on but no OpenAI API key is configured",
    "timeout": "The AI service took too long to respond",
    "rate_limited": "The AI service is busy right now",
    "auth": "The AI service rejected the server's credentials",
    "model_unavailable": "The configured AI model is not available",
    "rejected": "The AI-written answer did not pass RecallIQ's evidence checks",
}
_TEMPLATE_USED = "; the fixed template answer is shown instead."


def _fallback_message(code: str) -> str:
    return FALLBACK_MESSAGES.get(code, "The AI service is unavailable right now") + _TEMPLATE_USED


def configured_provider() -> str:
    """Provider named in the environment ("none" when unset); used by scripts/tests."""
    return LLMConfig.from_env().provider


def _log(stats: GenerationStats, query: str, reason: str | None = None) -> None:
    """One INFO line per generation. The query text is never logged (only its length).

    A reason is "<fixed category>: <excerpt>". Guard excerpts quote the model's answer, which
    can repeat what the consumer typed (a code, a name), so INFO gets the category only and
    the excerpt goes to DEBUG (RECALLIQ_LOG_LEVEL=DEBUG when diagnosing)."""
    category, _, excerpt = (reason or "").partition(": ")
    logger.info("rag_generation provider=%s model=%s outcome=%s latency_ms=%s "
                "prompt_tokens=%s completion_tokens=%s total_tokens=%s query_chars=%d%s",
                stats.provider, stats.model, stats.outcome, stats.latency_ms,
                stats.prompt_tokens, stats.completion_tokens, stats.total_tokens, len(query),
                f" reason={category!r}" if category else "")
    if excerpt:
        logger.debug("rag_generation reason_detail=%r", excerpt[:300])


def _llm_explanation(query: str, evidence: list[EvidenceItem], template: Explanation,
                     config: LLMConfig, client: Any) -> Explanation:
    """LLM answer if it is generated and passes every guard, else the template + reason."""
    stats = GenerationStats(provider=config.provider, model=config.model, outcome="")
    try:
        result = generate(query, evidence, config, client=client)
    except LLMError as error:
        stats.outcome = f"error:{error.code}"
        stats.latency_ms = error.latency_ms
        for name, value in error.tokens.items():
            setattr(stats, name, value)
        _log(stats, query, error.detail or None)
        template.fallback_reason = _fallback_message(error.code)
        template.generation = stats
        return template

    except Exception:  # never let the AI step turn /explain into a 500
        logger.exception("rag_generation unexpected failure")
        stats.outcome = "error:unexpected"
        template.fallback_reason = _fallback_message("unexpected")
        template.generation = stats
        return template

    stats.model, stats.latency_ms = result.model, result.latency_ms
    stats.prompt_tokens, stats.completion_tokens, stats.total_tokens = (
        result.prompt_tokens, result.completion_tokens, result.total_tokens)
    try:
        reason = validate_answer(result.text, result.cited_recall_ids, evidence,
                                 evidence_sufficient=result.evidence_sufficient)
    except Exception:  # a guard bug must reject, never accept or crash
        logger.exception("rag_generation guard failure")
        reason = "guard error"
    if reason:
        stats.outcome = "rejected"
        _log(stats, query, reason)
        template.fallback_reason = _fallback_message("rejected")
        template.generation = stats
        return template
    stats.outcome = "accepted"
    _log(stats, query)
    order = {e.recall_id: i for i, e in enumerate(evidence)}
    cited = sorted(result.cited_recall_ids, key=order.__getitem__)  # all validated above
    return Explanation(generator=f"llm:{config.provider}", text=result.text,
                       cited_recall_ids=cited, evidence_sufficient=result.evidence_sufficient,
                       generation=stats)


def explain(query: str, model: str, hits: list[dict], provider: str | None = None, *,
            llm_config: LLMConfig | None = None, max_evidence: int = MAX_EVIDENCE,
            llm_client: Any = None) -> GroundedAnswer:
    """`hits`: ranked dicts with EvidenceItem fields (rank, recall_id, title, url, product,
    issue, category, recall_class, last_updated, what_you_should_do, identifier_verdict,
    identifier_detail). `provider` overrides the configured provider ("none" = template);
    `llm_config` defaults to the environment (LLMConfig.from_env)."""
    evidence = select_evidence(hits, max_evidence)
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
    config = llm_config or LLMConfig.from_env()
    provider = (provider or config.provider or "none").strip().lower()
    if provider == "openai":
        config = LLMConfig(**{**asdict(config), "provider": "openai"})
        if status == "evidence_found":
            explanation = _llm_explanation(query, evidence, explanation, config, llm_client)
        else:  # no grounded evidence to write from: never ask the model
            stats = GenerationStats(provider=provider, model=config.model,
                                    outcome=f"skipped:{status}")
            _log(stats, query)
            explanation.generation = stats
    elif provider != "none":
        explanation.fallback_reason = (f"LLM provider {provider!r} is not implemented; "
                                       "deterministic template used")
    return GroundedAnswer(query=query, model=model, evidence_status=status,
                          retrieved_facts=evidence, explanation=explanation,
                          sources=[e.url for e in evidence],
                          skipped_identifier_mismatches=skipped)
