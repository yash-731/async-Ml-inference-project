"""Shared test fixtures.

Environment variables are set before any `app.*` import, because settings are
parsed once at import time.

Integration tests run against real PostgreSQL rather than SQLite. Swapping the
engine would mean testing different JSONB, enum, and unique-constraint
semantics than production uses, which is exactly where this kind of bug hides.
When no database is reachable, those tests skip with a clear message instead of
erroring, so the unit suite still runs on a laptop with no Docker.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from collections.abc import Iterator

import pytest

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://forecast:forecast@localhost:5432/forecast_test",
)
_STORAGE_DIR = tempfile.mkdtemp(prefix="forecast-test-uploads-")

os.environ.update(
    {
        "ENVIRONMENT": "test",
        "LOG_LEVEL": "WARNING",
        "JSON_LOGS": "false",
        "DATABASE_URL": TEST_DATABASE_URL,
        "REDIS_URL": "redis://localhost:6379/15",
        # In-memory limiter: these tests must not need a live Redis.
        "RATE_LIMIT_STORAGE_URI": "memory://",
        "UPLOAD_RATE_LIMIT": "1000/minute",
        "STORAGE_DIR": _STORAGE_DIR,
        "MIN_ROWS": "14",
        "FORECAST_HORIZON": "14",
    }
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402

from app.config import settings  # noqa: E402


def make_sales_csv(
    periods: int = 90,
    start: str = "2025-01-01",
    trend: float = 0.5,
    weekend_bump: float = 25.0,
    noise: float = 3.0,
    seed: int = 7,
    date_column: str | None = None,
    value_column: str | None = None,
) -> bytes:
    """Synthetic daily sales with a known trend and weekend seasonality."""
    index = pd.date_range(start, periods=periods, freq="D")
    rng = np.random.default_rng(seed)
    values = (
        100.0
        + np.arange(periods) * trend
        + np.where(index.dayofweek >= 5, weekend_bump, 0.0)
        + rng.normal(0.0, noise, periods)
    )
    frame = pd.DataFrame(
        {
            (date_column or settings.date_column): index.date,
            (value_column or settings.value_column): values.round(2),
        }
    )
    return frame.to_csv(index=False).encode("utf-8")


@pytest.fixture
def sales_csv() -> bytes:
    return make_sales_csv()


@pytest.fixture
def unique_csv() -> bytes:
    """A CSV whose bytes differ per test, so the idempotency key differs too."""
    return make_sales_csv(seed=uuid.uuid4().int % 10_000)


# --------------------------------------------------------------------------
# Database fixtures (integration only)
# --------------------------------------------------------------------------


def _ensure_test_database() -> str | None:
    """Create the test database if absent. Returns a skip reason on failure."""
    url = settings.database_url
    admin_url, _, db_name = url.rpartition("/")
    admin_engine = create_engine(f"{admin_url}/postgres", isolation_level="AUTOCOMMIT")
    try:
        with admin_engine.connect() as conn:
            exists = conn.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": db_name}
            )
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{db_name}"'))
    except Exception as exc:
        return f"PostgreSQL is not reachable at {admin_url}/postgres ({type(exc).__name__})"
    finally:
        admin_engine.dispose()
    return None


@pytest.fixture(scope="session")
def migrated_database() -> str:
    """Bring the test database to head with Alembic.

    Running the real migrations rather than `Base.metadata.create_all` means
    the tests also prove the migrations describe the models they are supposed
    to, instead of silently drifting apart.
    """
    skip_reason = _ensure_test_database()
    if skip_reason:
        # Skipping is the right behaviour on a laptop with no Docker, and the
        # wrong behaviour in CI, where a silent skip would mean the whole
        # integration suite never ran. REQUIRE_DATABASE=1 makes it loud.
        if os.environ.get("REQUIRE_DATABASE", "").lower() in {"1", "true", "yes"}:
            pytest.fail(f"REQUIRE_DATABASE is set but {skip_reason}", pytrace=False)
        pytest.skip(skip_reason)

    from alembic.config import Config

    from alembic import command

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", settings.database_url)
    command.downgrade(config, "base")
    command.upgrade(config, "head")
    return settings.database_url


@pytest.fixture(scope="session")
def db_engine(migrated_database: str):
    engine = create_engine(migrated_database, pool_pre_ping=True)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine) -> Iterator[Session]:
    """A session wrapped in a transaction that is always rolled back.

    `join_transaction_mode="create_savepoint"` lets application code call
    commit() for real, while the outer transaction still discards everything
    at the end of the test. No table truncation, no cross-test leakage.
    """
    connection = db_engine.connect()
    transaction = connection.begin()
    factory = sessionmaker(
        bind=connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    session = factory()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def recorded_enqueue() -> tuple[list[dict], object]:
    """Stand-in for the Celery enqueue hook.

    The API's contract is "a task was handed to the broker", not "a forecast
    ran". Keeping Celery out of the API tests makes them fast and
    deterministic; the task itself is tested directly.
    """
    calls: list[dict] = []

    def enqueue(job_id, correlation_id):
        task_id = uuid.uuid4().hex
        calls.append({"job_id": job_id, "correlation_id": correlation_id, "task_id": task_id})
        return task_id

    return calls, enqueue


@pytest.fixture
def client(db_session: Session):
    """TestClient with the request-scoped DB session replaced by the test one."""
    from fastapi.testclient import TestClient

    from app.api.deps import get_db
    from app.core.security import limiter
    from app.main import app

    app.dependency_overrides[get_db] = lambda: db_session
    limiter.reset()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    limiter.reset()


@pytest.fixture
def api_enqueue_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Patch the service's default enqueue so API tests never touch a broker."""
    from app.services import job_service

    calls: list[dict] = []

    def fake_enqueue(job_id, correlation_id):
        task_id = uuid.uuid4().hex
        calls.append({"job_id": job_id, "correlation_id": correlation_id, "task_id": task_id})
        return task_id

    monkeypatch.setattr(job_service, "_enqueue_forecast", fake_enqueue)
    return calls
