"""Response schemas for the recalls API."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Domain = Literal["food", "consumer_product"]
Confidence = Literal["high", "medium", "low"]


class RecallSummary(BaseModel):
    """One official recall notice as listed in pages and search results."""

    id: int = Field(description="Official Government of Canada recall NID")
    title: str
    url: str = Field(description="Official recalls-rappels.canada.ca page")
    domain: Domain
    organization: str
    category: str | None
    categories: list[str]
    recall_class: str | None
    last_updated: date | None
    archived: bool
    product_family: str | None


class RecallDetail(RecallSummary):
    """Full stored record for one notice, including RecallIQ's own grouping fields (labelled as such in the UI)."""

    product: str | None
    issue: str | None
    what_you_should_do: str | None
    title_group: str
    title_group_size: int = Field(ge=1)
    family_source: str
    family_confidence: Confidence
    needs_review: bool
    review_reason: str | None


class RecallPage(BaseModel):
    """Newest-first page of notices with the total count for pagination."""

    items: list[RecallSummary]
    total: int = Field(ge=0)
    limit: int
    offset: int


class CategoryCount(BaseModel):
    """An official category and how many indexed notices list it."""

    category: str
    domain: Domain
    count: int = Field(ge=1)


class CategoryList(BaseModel):
    """All official categories in the index (feeds the category filter)."""

    items: list[CategoryCount]
    total: int = Field(ge=0)


SearchModel = Literal["bm25", "minilm", "minilm_finetuned", "minilm_hard_negative"]
ScoreType = Literal["bm25", "cosine_similarity"]
IdentifierVerdict = Literal["MATCH", "MISMATCH", "UNKNOWN"]


class SearchRequest(BaseModel):
    """Consumer query, retrieval model and optional filters (applied before ranking)."""

    query: str = Field(min_length=1, max_length=500, description="Free-text consumer query")
    model: SearchModel = Field(default="bm25", description="Retrieval model")
    top_k: int = Field(default=10, ge=1, le=50)
    domain: Domain | None = Field(default=None, description="Only search this domain")
    category: str | None = Field(
        default=None, max_length=200,
        description="Only search recalls listing this category (see GET /categories)")

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("query must contain non-whitespace characters")
        return cleaned

    @field_validator("category")
    @classmethod
    def category_blank_is_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = " ".join(value.split())
        return cleaned or None


class SearchHit(BaseModel):
    """One ranked notice. Interpret `score` only with the response's `score_type`."""

    rank: int = Field(ge=1)
    recall_id: int = Field(description="Official Government of Canada recall NID")
    title: str
    category: str | None
    issue: str | None
    url: str = Field(description="Official recalls-rappels.canada.ca page")
    score: float = Field(description="Interpret using the response's score_type")
    domain: Domain
    last_updated: date | None = Field(
        description="Date the official notice was last updated (not necessarily the recall date)")
    archived: bool = Field(description="The official notice is archived")


class IdentifierTypeCheck(BaseModel):
    """Exact comparison for one identifier type (UPC, model, lot or best-before)."""

    verdict: IdentifierVerdict
    query_values: list[str] = Field(description="Normalised identifiers found in the query")
    evidence_values: list[str] = Field(description="Normalised identifiers in the recall record")
    matched: list[str]


class IdentifierCheck(BaseModel):
    """Deterministic identifier verdict for one result."""

    recall_id: int
    verdict: IdentifierVerdict = Field(
        description="MATCH / MISMATCH only when both the query and the official recall record "
                    "contain identifiers of the same type; otherwise UNKNOWN")
    by_type: dict[str, IdentifierTypeCheck] = Field(description="upc, model, lot, best_before")


class SearchResponse(BaseModel):
    """Ranked notices plus identifier checks, aligned by position with `results`."""

    query: str
    model: SearchModel
    score_type: ScoreType
    score_description: str
    top_k: int
    domain: Domain | None = Field(default=None, description="Domain filter applied, if any")
    category: str | None = Field(default=None, description="Category filter applied, if any")
    results: list[SearchHit]
    identifiers_in_query: bool = Field(
        default=False, description="Whether the query names a UPC / model / lot / best-before")
    identifier_checks: list[IdentifierCheck] = Field(
        default_factory=list,
        description="Deterministic exact identifier comparison per result (same order)")
    notice: str = Field(default="", description="How to interpret the results")


class ExplainRequest(SearchRequest):
    """Same as a search request; `top_k` is capped at 5 because the results become RAG evidence."""

    top_k: int | None = Field(
        default=None, ge=1, le=5,
        description="Results retrieved and considered as evidence (default: RECALLIQ_RAG_TOP_K)")


class EvidenceFact(BaseModel):
    """Fields copied verbatim from the indexed official recall record (null = not listed)."""
    rank: int
    recall_id: int
    title: str
    url: str
    product: str | None
    issue: str | None
    category: str | None
    recall_class: str | None
    last_updated: str | None
    what_you_should_do: str | None = Field(
        default=None, description="Official 'what you should do' text, verbatim (null = not listed)")
    identifier_verdict: IdentifierVerdict


class GenerationStats(BaseModel):
    """Observability for the LLM step. Never contains the API key or the prompt."""
    provider: str
    model: str | None
    outcome: str = Field(description="accepted | rejected | error:<code> | skipped:<evidence_status>")
    prompt_version: str
    latency_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class GeneratedExplanation(BaseModel):
    generator: str = Field(description='"template" (deterministic) or "llm:<provider>"')
    text: str = Field(description="Generated from the evidence above; not an official statement")
    cited_recall_ids: list[int] = Field(description="Evidence recall ids the text relies on")
    fallback_reason: str | None = Field(
        default=None, description="Why the deterministic template is shown instead of an AI answer")
    evidence_sufficient: bool | None = Field(
        default=None, description="AI answers only: whether the model judged the notices to "
                                  "concern the product asked about")
    generation: GenerationStats | None = Field(
        default=None, description="Present when an LLM provider is configured")


class ExplainResponse(BaseModel):
    query: str
    model: SearchModel
    evidence_status: Literal["no_results", "identifier_mismatch", "evidence_found"]
    retrieved_facts: list[EvidenceFact]
    skipped_identifier_mismatches: int = Field(
        default=0, description="Higher-ranked results set aside: identifiers conflict with the query")
    explanation: GeneratedExplanation
    sources: list[str] = Field(description="Official recall notice URLs used as evidence")
    disclaimer: str


class ModelStatus(BaseModel):
    """Whether one retrieval model can be served right now, and why not."""

    model: SearchModel
    available: bool
    score_type: ScoreType
    description: str
    reason: str | None = None


class ModelList(BaseModel):
    """Availability of all four retrieval models (drives the model selector)."""

    models: list[ModelStatus]
