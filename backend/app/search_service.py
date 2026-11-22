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


def _mtime(path: Path) -> tuple[int, int] | None:
    """(mtime, size) of a file, None if absent."""
    if not path.is_file():
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _version_key(db_path: Path, embeddings_dir: Path,
                 finetuned: dict[str, tuple[Path, Path]] | None = None) -> tuple:
    """Rebuild the engine when the database, an index's metadata or a checkpoint changes."""
    key = [str(db_path.resolve()), db_path.stat().st_mtime_ns,
           str(embeddings_dir.resolve()), _mtime(embeddings_dir / "metadata.json"),
           _mtime(embeddings_dir / "embeddings.npy")]
    for name, (checkpoint, index_dir) in sorted((finetuned or {}).items()):
        key += [name, str(checkpoint)]
        key += [_mtime(checkpoint / f) for f in ("model.safetensors", "config.json",
                                                  "recalliq_encoder.json")]
        key += [str(index_dir)] + [_mtime(index_dir / f) for f in
                                   ("metadata.json", "embeddings.npy", "recall_ids.json")]
    return tuple(key)


def _unavailable(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)


def get_search_engine(settings: Settings = Depends(get_settings)) -> SearchEngine:
    if not settings.database_path.is_file():
        raise _unavailable("Recall database is not available. Run the data pipeline first.")
    try:
        key = _version_key(settings.database_path, settings.embeddings_dir,
                           settings.finetuned_models)
        with _lock:
            engine = _cache.get(key)
            if engine is None:
                _cache.clear()  # keep only the current version in memory
                engine = build_engine(settings.database_path, settings.embeddings_dir,
                                      finetuned=settings.finetuned_models)
                _cache[key] = engine
    except (OSError, sqlite3.Error, ValueError) as error:
        # e.g. DB removed mid-request, missing table, empty corpus: a server-side state
        # problem, reported as 503 rather than an opaque 500.
        raise _unavailable(f"Search index could not be built: {error}") from error
    return engine
