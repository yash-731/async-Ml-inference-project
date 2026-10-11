"""FastAPI application wiring."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api.routes import jobs
from app.config import settings
from app.core.errors import CsvValidationError
from app.core.logging import CorrelationIdMiddleware, configure_logging
from app.core.security import RateLimitExceeded, limiter, rate_limit_handler
from app.db.base import engine
from app.services.forecasting import available_models

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging(level=settings.log_level, json_logs=settings.json_logs)
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "app.startup",
        environment=settings.environment,
        models=available_models(),
        storage_dir=str(settings.storage_dir),
    )
    yield
    log.info("app.shutdown")


app = FastAPI(
    title="Async Forecasting Service",
    description=(
        "Upload a CSV of historical values, a Celery worker forecasts it in the "
        "background, and you poll for the result."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
app.add_middleware(CorrelationIdMiddleware)
app.include_router(jobs.router)


@app.exception_handler(CsvValidationError)
async def csv_validation_handler(request: Request, exc: Exception) -> JSONResponse:
    """Safety net so validation errors raised outside a route still map to 422."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc), "error_type": getattr(exc, "error_type", None)},
    )


@app.get("/health", tags=["ops"], summary="Liveness check")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}


@app.get("/ready", tags=["ops"], summary="Readiness check")
def ready() -> JSONResponse:
    """Kept separate from /health deliberately.

    A liveness probe that touched Postgres would restart a perfectly healthy
    API container during a brief database outage. Readiness only removes it
    from the load balancer.
    """
    checks: dict[str, str] = {}

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {type(exc).__name__}"

    try:
        import redis

        redis.Redis.from_url(settings.redis_url, socket_connect_timeout=2).ping()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {type(exc).__name__}"

    healthy = all(value == "ok" for value in checks.values())
    return JSONResponse(
        status_code=status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "ready" if healthy else "degraded", "checks": checks},
    )
