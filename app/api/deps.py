"""FastAPI dependencies."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from app.core.logging import get_correlation_id, new_correlation_id
from app.db.base import SessionLocal


def get_db() -> Iterator[Session]:
    """One session per request, always closed.

    Commits are explicit in the service layer; an auto-commit here would hide
    the transaction boundary that idempotency depends on.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def correlation_id(request: Request) -> str:
    """Read the id stamped on the scope by CorrelationIdMiddleware."""
    from_scope = request.scope.get("state", {}).get("correlation_id")
    return from_scope or get_correlation_id() or new_correlation_id()
