"""Seeded, stratified selection of representative recall records.

Strategy (documented in docs/DATA_PIPELINE.md):
1. Deduplicate by exact title: keep one representative per title_group (latest
   last_updated, then highest id), so repeated notices don't crowd the sample.
2. Stratify by (domain, primary official category), where primary = first category.
3. Allocate the target size proportionally to stratum sizes using the largest-remainder
   method (deterministic tie-break by stratum name).
4. Draw within each stratum with a NumPy Generator seeded once.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_SEED = 4110  # ARTE 4110U


def representatives(records: pd.DataFrame) -> pd.DataFrame:
    ordered = records.assign(_date=records["last_updated"].fillna("")).sort_values(
        ["title_group", "_date", "id"], ascending=[True, False, False]
    )
    reps = ordered.drop_duplicates("title_group", keep="first").drop(columns="_date")
    return reps.sort_values("id").reset_index(drop=True)


def stratum_key(row: pd.Series) -> str:
    primary = row["categories"][0] if row["categories"] else "Uncategorized"
    return f"{row['domain']}::{primary}"


def allocate(stratum_sizes: dict[str, int], target: int) -> dict[str, int]:
    """Largest-remainder proportional allocation; never exceeds a stratum's size."""
    total = sum(stratum_sizes.values())
    if target >= total:
        return dict(stratum_sizes)

    quotas = {key: size * target / total for key, size in stratum_sizes.items()}
    allocation = {key: math.floor(quota) for key, quota in quotas.items()}
    remaining = target - sum(allocation.values())
    by_remainder = sorted(quotas, key=lambda key: (-(quotas[key] - allocation[key]), key))
    for key in by_remainder[:remaining]:
        allocation[key] += 1
    return allocation


def stratified_sample(
    records: pd.DataFrame, size: int = DEFAULT_SAMPLE_SIZE, seed: int = DEFAULT_SEED
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    if size <= 0:
        raise ValueError("Sample size must be positive")

    pool = representatives(records)
    pool = pool.assign(_stratum=pool.apply(stratum_key, axis=1))
    sizes = pool["_stratum"].value_counts().sort_index().to_dict()
    allocation = allocate(sizes, size)

    rng = np.random.default_rng(seed)
    parts = []
    for stratum in sorted(allocation):
        k = allocation[stratum]
        if k == 0:
            continue
        members = pool[pool["_stratum"] == stratum].sort_values("id")
        chosen = rng.choice(len(members), size=k, replace=False)
        parts.append(members.iloc[np.sort(chosen)])

    sample = pd.concat(parts).drop(columns="_stratum").sort_values("id").reset_index(drop=True)
    report = {s: {"available": int(sizes[s]), "selected": int(allocation[s])} for s in sizes}
    return sample, report
