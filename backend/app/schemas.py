"""Response schemas for the recalls API."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

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
