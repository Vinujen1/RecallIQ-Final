"""Response schemas for the recalls API."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Domain = Literal["food", "consumer_product"]
Confidence = Literal["high", "medium", "low"]


class RecallSummary(BaseModel):
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
    items: list[RecallSummary]
    total: int = Field(ge=0)
    limit: int
    offset: int


class CategoryCount(BaseModel):
    category: str
    domain: Domain
    count: int = Field(ge=1)


class CategoryList(BaseModel):
    items: list[CategoryCount]
    total: int = Field(ge=0)


SearchModel = Literal["bm25", "minilm"]
ScoreType = Literal["bm25", "cosine_similarity"]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500, description="Free-text consumer query")
    model: SearchModel = Field(default="bm25", description="Retrieval model")
    top_k: int = Field(default=10, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("query must contain non-whitespace characters")
        return cleaned


class SearchHit(BaseModel):
    rank: int = Field(ge=1)
    recall_id: int = Field(description="Official Government of Canada recall NID")
    title: str
    category: str | None
    issue: str | None
    url: str = Field(description="Official recalls-rappels.canada.ca page")
    score: float = Field(description="Interpret using the response's score_type")


class SearchResponse(BaseModel):
    query: str
    model: SearchModel
    score_type: ScoreType
    score_description: str
    top_k: int
    results: list[SearchHit]
