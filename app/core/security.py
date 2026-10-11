"""Rate limiting for the upload endpoint.

Redis-backed rather than in-process: the counter has to be shared, otherwise
running two API containers silently doubles the real limit.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import settings

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=settings.limiter_storage_uri,
    enabled=settings.rate_limit_enabled,
    headers_enabled=True,
)


async def rate_limit_handler(request: Request, exc: Exception) -> JSONResponse:
    detail = getattr(exc, "detail", "rate limit exceeded")
    return JSONResponse(
        status_code=429,
        content={
            "detail": f"Rate limit exceeded: {detail}",
            "error_type": "rate_limit_exceeded",
        },
        headers={"Retry-After": "60"},
    )


__all__ = ["RateLimitExceeded", "limiter", "rate_limit_handler"]
