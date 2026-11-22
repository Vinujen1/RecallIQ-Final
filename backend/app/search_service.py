"""Builds and caches the retrieval engine used by POST /search."""

import sqlite3
import threading
from pathlib import Path

from fastapi import Depends, HTTPException, status

from app.config import Settings, get_settings
from retrieval.loader import build_engine
from retrieval.search import SearchEngine

_lock = threading.Lock()
_cache: dict[tuple, SearchEngine] = {}


def _version_key(db_path: Path, embeddings_dir: Path) -> tuple:
    """Rebuild the engine when the database or embedding metadata file changes."""
    metadata = embeddings_dir / "metadata.json"
    return (
        str(db_path.resolve()),
        db_path.stat().st_mtime_ns,
        str(embeddings_dir.resolve()),
        metadata.stat().st_mtime_ns if metadata.is_file() else None,
    )


def _unavailable(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)


def get_search_engine(settings: Settings = Depends(get_settings)) -> SearchEngine:
    if not settings.database_path.is_file():
        raise _unavailable("Recall database is not available. Run the data pipeline first.")
    try:
        key = _version_key(settings.database_path, settings.embeddings_dir)
        with _lock:
            engine = _cache.get(key)
            if engine is None:
                _cache.clear()  # keep only the current version in memory
                engine = build_engine(settings.database_path, settings.embeddings_dir)
                _cache[key] = engine
    except (OSError, sqlite3.Error, ValueError) as error:
        # e.g. DB removed mid-request, missing table, empty corpus: a server-side state
        # problem, reported as 503 rather than an opaque 500.
        raise _unavailable(f"Search index could not be built: {error}") from error
    return engine
