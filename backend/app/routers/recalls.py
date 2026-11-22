"""Read-only endpoints serving recall records from SQLite."""

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from app.db import get_connection
from app.schemas import CategoryCount, CategoryList, RecallDetail, RecallPage, RecallSummary

router = APIRouter(tags=["recalls"])

Connection = Annotated[sqlite3.Connection, Depends(get_connection)]

MAX_PAGE_SIZE = 100
MAX_OFFSET = 1_000_000
# SQLite INTEGER is signed 64-bit; larger path ids would raise OverflowError (HTTP 500).
MAX_RECALL_ID = 2**63 - 1

# Newest first; NULL dates last; id breaks ties so pagination is stable.
_ORDER = "ORDER BY last_updated IS NULL, last_updated DESC, id DESC"


def _categories_for(connection: sqlite3.Connection, ids: list[int]) -> dict[int, list[str]]:
    if not ids:
        return {}
    placeholders = ", ".join("?" for _ in ids)
    rows = connection.execute(
        f"SELECT recall_id, category FROM recall_categories "
        f"WHERE recall_id IN ({placeholders}) ORDER BY recall_id, position",
        ids,
    ).fetchall()
    result: dict[int, list[str]] = {recall_id: [] for recall_id in ids}
    for row in rows:
        result[row["recall_id"]].append(row["category"])
    return result


@router.get("/recalls", response_model=RecallPage)
def list_recalls(
    connection: Connection,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 20,
    offset: Annotated[int, Query(ge=0, le=MAX_OFFSET)] = 0,
) -> RecallPage:
    total = connection.execute("SELECT COUNT(*) FROM recalls").fetchone()[0]
    rows = connection.execute(
        f"SELECT * FROM recalls {_ORDER} LIMIT ? OFFSET ?", (limit, offset)
    ).fetchall()
    categories = _categories_for(connection, [row["id"] for row in rows])
    items = [
        RecallSummary.model_validate({**dict(row), "categories": categories[row["id"]]})
        for row in rows
    ]
    return RecallPage(items=items, total=total, limit=limit, offset=offset)


@router.get("/recalls/{recall_id}", response_model=RecallDetail)
def get_recall(
    connection: Connection,
    recall_id: Annotated[int, Path(ge=1, le=MAX_RECALL_ID)],
) -> RecallDetail:
    row = connection.execute("SELECT * FROM recalls WHERE id = ?", (recall_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recall not found")
    categories = _categories_for(connection, [recall_id])[recall_id]
    return RecallDetail.model_validate({**dict(row), "categories": categories})


@router.get("/categories", response_model=CategoryList)
def list_categories(connection: Connection) -> CategoryList:
    rows = connection.execute(
        "SELECT rc.category, r.domain, COUNT(*) AS count "
        "FROM recall_categories rc JOIN recalls r ON r.id = rc.recall_id "
        "GROUP BY rc.category, r.domain ORDER BY count DESC, rc.category"
    ).fetchall()
    items = [CategoryCount.model_validate(dict(row)) for row in rows]
    return CategoryList(items=items, total=len(items))
