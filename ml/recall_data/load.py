"""Read the raw government CSV exactly as published and validate its structure."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

# Column names exactly as they appear in HCRSAMOpenData.csv (inspected 2026-11-15).
RAW_COLUMNS: tuple[str, ...] = (
    "NID",
    "Title",
    "URL",
    "Organization",
    "Product",
    "Issue",
    "What you should do",
    "Category",
    "Recall class",
    "Last updated",
    "Archived",
)


class DataValidationError(ValueError):
    """Raised when the input file does not match the expected government schema."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(path: Path) -> pd.DataFrame:
    """Load every column as text, without letting pandas guess types or invent NaNs.

    Empty cells stay as "" here; normalization to None happens in clean.py so that the
    raw frame remains a faithful copy of the file.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Raw recall CSV not found: {path}")

    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    ).astype(object)

    missing = [column for column in RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(f"CSV is missing expected columns: {missing}")

    validate_ids(frame)
    return frame


def validate_ids(frame: pd.DataFrame) -> None:
    """NID is the official recall identifier; it must be numeric and unique."""
    ids = frame["NID"].astype(str).str.strip()
    non_numeric = ids[~ids.str.fullmatch(r"\d+")]
    if not non_numeric.empty:
        raise DataValidationError(f"Non-numeric NID values: {non_numeric.head(5).tolist()}")

    duplicated = ids[ids.duplicated(keep=False)]
    if not duplicated.empty:
        raise DataValidationError(
            f"Duplicate NID values ({duplicated.nunique()} ids): {sorted(set(duplicated))[:5]}"
        )
