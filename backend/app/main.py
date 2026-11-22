"""FastAPI entry point for the RecallIQ backend.

Run from the backend/ directory:
    uvicorn app.main:app --reload --port 8000
"""

import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import health, recalls, search


def configure_logging(level: str) -> None:
    """Send `recalliq.*` records (e.g. RAG accept/reject lines) to stderr.

    uvicorn configures only its own loggers, so without this they were silently dropped.
    Idempotent (one handler, replaced on re-call). Records still propagate, so pytest's
    caplog and any root handler see them; uvicorn's default config adds no root handler,
    so lines are not duplicated."""
    log = logging.getLogger("recalliq")
    for handler in [h for h in log.handlers if getattr(h, "_recalliq", False)]:
        log.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    handler._recalliq = True  # type: ignore[attr-defined]
    log.addHandler(handler)
    log.setLevel(level)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(title=settings.app_name, version=settings.app_version)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(recalls.router)
    app.include_router(search.router)
    return app


app = create_app()
