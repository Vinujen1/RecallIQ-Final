"""Filter to food and consumer-product recalls and normalize values without inventing data."""

from __future__ import annotations

import re
from datetime import date

import pandas as pd

# Eligibility is decided by the dataset's own Organization values (inspected 2026-11-15):
#   "CFIA"                    -> Canadian Food Inspection Agency food recalls
#   "Consumer product safety" -> Health Canada consumer product recalls
# Excluded: TC (vehicles), Medical devices, Drugs and health products, Marketed health
# products, Controlled substances and cannabis, HC, and the mixed
# "Communications and Public Affairs Branch" feed (mostly drugs/health products).
ELIGIBLE_ORGANIZATIONS: dict[str, str] = {
    "CFIA": "food",
    "Consumer product safety": "consumer_product",
}

# Multi-valued Category cells join individual categories with " - ".
CATEGORY_SEPARATOR = " - "

# Values the source uses as "no value" placeholders. Only exact matches are nulled.
PLACEHOLDER_VALUES = frozenset({"--"})

_WHITESPACE = re.compile(r"\s+")


def normalize_text(value: object) -> str | None:
    """Trim and collapse whitespace; drop zero-width spaces; empty -> None.

    Words, punctuation and casing are never changed.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).replace("\u200b", "").replace("\xa0", " ")
    text = _WHITESPACE.sub(" ", text).strip()
    if not text or text in PLACEHOLDER_VALUES:
        return None
    return text


def parse_date(value: object) -> str | None:
    """Return an ISO yyyy-mm-dd string, or None if missing/unparseable (never guessed)."""
    text = normalize_text(value)
    if text is None:
        return None
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return None


def parse_archived(value: object) -> bool:
    text = normalize_text(value)
    if text == "1":
        return True
    if text == "0":
        return False
    raise ValueError(f"Unexpected Archived value: {value!r}")


def split_categories(category: str | None) -> list[str]:
    if category is None:
        return []
    parts: list[str] = []
    for part in category.split(CATEGORY_SEPARATOR):
        part = part.strip()
        if part and part not in parts:
            parts.append(part)
    return parts


def filter_eligible(raw: pd.DataFrame) -> pd.DataFrame:
    organization = raw["Organization"].map(normalize_text)
    return raw.loc[organization.isin(ELIGIBLE_ORGANIZATIONS)].copy()


def normalize(eligible: pd.DataFrame) -> pd.DataFrame:
    """Map raw government columns to RecallIQ's snake_case schema."""
    out = pd.DataFrame(index=eligible.index)
    out["id"] = eligible["NID"].map(lambda v: int(str(v).strip()))
    out["title"] = eligible["Title"].map(normalize_text)
    out["url"] = eligible["URL"].map(normalize_text)
    out["organization"] = eligible["Organization"].map(normalize_text)
    out["domain"] = out["organization"].map(ELIGIBLE_ORGANIZATIONS)
    out["category"] = eligible["Category"].map(normalize_text)
    out["categories"] = out["category"].map(split_categories)
    out["product"] = eligible["Product"].map(normalize_text)
    out["issue"] = eligible["Issue"].map(normalize_text)
    out["what_you_should_do"] = eligible["What you should do"].map(normalize_text)
    out["recall_class"] = eligible["Recall class"].map(normalize_text)
    out["last_updated"] = eligible["Last updated"].map(parse_date)
    out["archived"] = eligible["Archived"].map(parse_archived)

    if out["title"].isna().any() or out["url"].isna().any():
        raise ValueError("Eligible records must have a title and URL")

    # pandas 3 infers a string dtype for text columns and turns None into NaN; store
    # missing values uniformly as None so downstream code and exports see one sentinel.
    out = out.astype(object)
    out = out.where(out.notna(), None)
    return out.sort_values("id").reset_index(drop=True)


def unparseable_date_count(eligible: pd.DataFrame) -> int:
    """Non-empty date cells that could not be parsed (reported, then stored as None)."""
    present = eligible["Last updated"].map(normalize_text).notna()
    parsed = eligible["Last updated"].map(parse_date).notna()
    return int((present & ~parsed).sum())
