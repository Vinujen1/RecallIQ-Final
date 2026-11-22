"""Read-only SQLite access for the recall database built by ml/recall_data."""

from collections.abc import Iterator
import sqlite3

from fastapi import Depends, HTTPException, status

from app.config import Settings, get_settings


def get_connection(settings: Settings = Depends(get_settings)) -> Iterator[sqlite3.Connection]:
    path = settings.database_path
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recall database is not available. Run the data pipeline first.",
        )

    # mode=ro: the API can never modify the dataset. check_same_thread=False because
    # FastAPI may run the dependency and the endpoint on different threadpool threads;
    # each request still gets its own connection.
    connection = sqlite3.connect(
        f"{path.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False
    )
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()
