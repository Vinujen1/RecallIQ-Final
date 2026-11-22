"""POST /search — BM25, pretrained MiniLM (Model A) or fine-tuned MiniLM (Models B/C)
retrieval over the recall corpus, with deterministic identifier checks per result.
GET /search/models — which models are servable right now.
POST /explain — search + identifier checks + evidence selection + grounded explanation
(deterministic template by default; see ml/rag/pipeline.py)."""

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

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
) -> SearchResponse:
    return _search(request, engine)


@router.post("/explain", response_model=ExplainResponse)
def explain(
    request: ExplainRequest,
    engine: Annotated[SearchEngine, Depends(get_search_engine)],
    connection: Annotated[sqlite3.Connection, Depends(get_connection)],
) -> ExplainResponse:
    found = _search(request, engine)
    checks = {c.recall_id: c for c in found.identifier_checks}
    ids = [hit.recall_id for hit in found.results]
    rows = {}
    if ids:
        marks = ",".join("?" * len(ids))
        rows = {r["id"]: r for r in connection.execute(
            f"SELECT id, product, recall_class, last_updated FROM recalls WHERE id IN ({marks})", ids)}
    hits = []
    for hit in found.results:
        row = rows.get(hit.recall_id)
        hits.append({"rank": hit.rank, "recall_id": hit.recall_id, "title": hit.title,
                     "url": hit.url, "issue": hit.issue, "category": hit.category,
                     "product": row["product"] if row else None,
                     "recall_class": row["recall_class"] if row else None,
                     "last_updated": row["last_updated"] if row else None,
                     "identifier_verdict": checks[hit.recall_id].verdict})
    answer = grounded_explain(request.query, request.model, hits).to_json()
    for fact in answer["retrieved_facts"]:
        fact.pop("identifier_detail", None)
    return ExplainResponse.model_validate(answer)


def _search(request: SearchRequest, engine: SearchEngine) -> SearchResponse:
    try:
        hits = engine.search(request.query, model=request.model, top_k=request.top_k)
    except SearchUnavailableError as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    except ValueError as error:  # defensive: request schema already validates these
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error

    return SearchResponse(
        query=request.query,
        model=request.model,
        score_type=SCORE_TYPES[request.model],
        score_description=SCORE_DESCRIPTIONS[request.model],
        top_k=request.top_k,
        results=[SearchHit.model_validate(hit, from_attributes=True) for hit in hits],
        identifiers_in_query=has_identifier(request.query),
        identifier_checks=[_identifier_check(request.query, hit.recall_id,
                                             engine.record(hit.recall_id).search_text)
                           for hit in hits],
        notice=SEARCH_NOTICE,
    )
