"""POST /search — BM25, pretrained MiniLM (Model A) or fine-tuned MiniLM (Models B/C)
retrieval over the recall corpus, with deterministic identifier checks per result.
GET /search/models — which models are servable right now.
POST /explain — search + identifier checks + evidence selection + grounded explanation
(deterministic template by default; GPT answer checked by guards when
RECALLIQ_LLM_PROVIDER=openai; see ml/rag/pipeline.py and docs/RAG.md)."""

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings, get_settings
from app.db import get_connection
from app.schemas import (
    ExplainRequest,
    ExplainResponse,
    IdentifierCheck,
    IdentifierTypeCheck,
    ModelList,
    ModelStatus,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from app.search_service import get_search_engine
from retrieval.search import (
    SCORE_DESCRIPTIONS,
    SCORE_TYPES,
    SearchEngine,
    SearchUnavailableError,
)
from rag.pipeline import explain as grounded_explain
from verification.identifiers import Verdict, has_identifier, verify

router = APIRouter(tags=["search"])

_VERDICT = {Verdict.MATCH: "MATCH", Verdict.MISMATCH: "MISMATCH", Verdict.UNKNOWN: "UNKNOWN"}
SEARCH_NOTICE = (
    "Results are recall notices ranked by text similarity, not a determination that your "
    "product is recalled. Confirm the brand, product, UPC, lot code and dates on the official "
    "notice (url). Identifier checks compare your text with the indexed recall record only; "
    "UNKNOWN means it does not list a comparable identifier. No result does not mean a "
    "product is safe.")


def _identifier_check(query: str, recall_id: int, evidence: str) -> IdentifierCheck:
    result = verify(query, evidence)
    return IdentifierCheck(
        recall_id=recall_id, verdict=_VERDICT[result.verdict],
        by_type={kind: IdentifierTypeCheck(verdict=_VERDICT[r.verdict], query_values=r.query_values,
                                           evidence_values=r.evidence_values, matched=r.matched)
                 for kind, r in result.by_type.items()})


@router.get("/search/models", response_model=ModelList)
def search_models(engine: Annotated[SearchEngine, Depends(get_search_engine)]) -> ModelList:
    return ModelList(models=[
        ModelStatus(model=name, available=reason is None, score_type=SCORE_TYPES[name],
                    description=SCORE_DESCRIPTIONS[name], reason=reason)
        for name, reason in engine.available_models().items()])


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    engine: Annotated[SearchEngine, Depends(get_search_engine)],
    connection: Annotated[sqlite3.Connection, Depends(get_connection)],
) -> SearchResponse:
    return _search(request, engine, connection)


@router.post("/explain", response_model=ExplainResponse)
def explain(
    request: ExplainRequest,
    engine: Annotated[SearchEngine, Depends(get_search_engine)],
    connection: Annotated[sqlite3.Connection, Depends(get_connection)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ExplainResponse:
    request = request.model_copy(update={"top_k": request.top_k or settings.rag_top_k})
    found = _search(request, engine, connection)
    checks = {c.recall_id: c for c in found.identifier_checks}
    ids = [hit.recall_id for hit in found.results]
    rows = {}
    if ids:
        marks = ",".join("?" * len(ids))
        rows = {r["id"]: r for r in connection.execute(
            f"SELECT id, product, recall_class, last_updated, what_you_should_do "
            f"FROM recalls WHERE id IN ({marks})", ids)}
    hits = []
    for hit in found.results:
        row = rows.get(hit.recall_id)
        hits.append({"rank": hit.rank, "recall_id": hit.recall_id, "title": hit.title,
                     "url": hit.url, "issue": hit.issue, "category": hit.category,
                     "product": row["product"] if row else None,
                     "recall_class": row["recall_class"] if row else None,
                     "last_updated": row["last_updated"] if row else None,
                     "what_you_should_do": row["what_you_should_do"] if row else None,
                     "identifier_verdict": checks[hit.recall_id].verdict})
    # Sync endpoint: FastAPI runs it in a worker thread, so the blocking OpenAI call (bounded
    # by RECALLIQ_LLM_TIMEOUT_SECONDS) does not block the event loop.
    answer = grounded_explain(request.query, request.model, hits,
                              llm_config=settings.llm_config,
                              max_evidence=settings.rag_max_evidence).to_json()
    for fact in answer["retrieved_facts"]:
        fact.pop("identifier_detail", None)
    return ExplainResponse.model_validate(answer)


def _allowed_ids(request: SearchRequest, connection: sqlite3.Connection) -> set[int] | None:
    """Recall ids passing the optional domain/category filters (None = no filter)."""
    if request.domain is None and request.category is None:
        return None
    sql, params = "SELECT r.id FROM recalls r WHERE 1 = 1", []
    if request.domain is not None:
        sql += " AND r.domain = ?"
        params.append(request.domain)
    if request.category is not None:
        sql += (" AND EXISTS (SELECT 1 FROM recall_categories rc "
                "WHERE rc.recall_id = r.id AND rc.category = ?)")
        params.append(request.category)
    return {row[0] for row in connection.execute(sql, params)}


def _metadata(ids: list[int], connection: sqlite3.Connection) -> dict[int, sqlite3.Row]:
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    return {row["id"]: row for row in connection.execute(
        f"SELECT id, domain, last_updated, archived FROM recalls WHERE id IN ({marks})", ids)}


def _search(request: SearchRequest, engine: SearchEngine,
            connection: sqlite3.Connection) -> SearchResponse:
    allowed = _allowed_ids(request, connection)
    try:
        hits = [] if allowed is not None and not allowed else engine.search(
            request.query, model=request.model, top_k=request.top_k, restrict_to=allowed)
    except SearchUnavailableError as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    except ValueError as error:  # defensive: request schema already validates these
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error

    meta = _metadata([hit.recall_id for hit in hits], connection)
    return SearchResponse(
        query=request.query,
        model=request.model,
        score_type=SCORE_TYPES[request.model],
        score_description=SCORE_DESCRIPTIONS[request.model],
        top_k=request.top_k,
        domain=request.domain,
        category=request.category,
        results=[_hit(hit, meta) for hit in hits],
        identifiers_in_query=has_identifier(request.query),
        identifier_checks=[_identifier_check(request.query, hit.recall_id,
                                             engine.record(hit.recall_id).search_text)
                           for hit in hits],
        notice=SEARCH_NOTICE,
    )


def _hit(hit, meta: dict[int, sqlite3.Row]) -> SearchHit:
    row = meta.get(hit.recall_id)
    if row is None:  # engine and database out of sync: a server-side state problem
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Search index and recall database are out of sync")
    return SearchHit(rank=hit.rank, recall_id=hit.recall_id, title=hit.title,
                     category=hit.category, issue=hit.issue, url=hit.url, score=hit.score,
                     domain=row["domain"], last_updated=row["last_updated"],
                     archived=bool(row["archived"]))
