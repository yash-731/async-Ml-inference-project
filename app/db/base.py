"""SQLAlchemy engine, session factory, and declarative base.

Synchronous SQLAlchemy on purpose. The API's route handlers are declared with
`def` (not `async def`), so FastAPI runs them in a thread pool and blocking DB
calls never stall the event loop. The same session code is then reusable
verbatim inside the Celery worker, which is a plain synchronous process.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(
    settings.database_url,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_pre_ping=True,  # survives Postgres restarts / idle connection drops
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,  # attributes stay readable after commit()
    future=True,
)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for non-request code (the Celery worker)."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
