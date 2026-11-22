"""POST /search — BM25 or pretrained MiniLM retrieval over the recall corpus."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas import SearchHit, SearchRequest, SearchResponse
from app.search_service import get_search_engine
from retrieval.search import (
    SCORE_DESCRIPTIONS,
    SCORE_TYPES,
    SearchEngine,
    SearchUnavailableError,
)

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    engine: Annotated[SearchEngine, Depends(get_search_engine)],
) -> SearchResponse:
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
    )
