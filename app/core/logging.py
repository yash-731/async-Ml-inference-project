"""Structured logging and request correlation.

Every log line is JSON with a `correlation_id`. The API generates it (or
honours an inbound `X-Request-ID`), stores it on the job row, and passes it to
the Celery task, which binds it again -- so one grep pulls the full path of a
request from HTTP ingress through to forecast completion.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextvars import ContextVar

import structlog
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

CORRELATION_ID_HEADER = "X-Request-ID"

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def new_correlation_id() -> str:
    return uuid.uuid4().hex


def set_correlation_id(correlation_id: str) -> None:
    """Bind the id to both the plain contextvar and structlog's context."""
    _correlation_id.set(correlation_id)
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)


def get_correlation_id() -> str | None:
    return _correlation_id.get()


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()
    _correlation_id.set(None)


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    """Route structlog *and* stdlib logging through one renderer.

    Without the stdlib bridge, uvicorn and Celery would keep emitting plain
    text while our own lines were JSON -- useless for a log aggregator.
    """
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    renderer = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processor=renderer,
    )

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    # Let these propagate to root instead of writing their own format.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "celery", "celery.app.trace"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True

    logging.getLogger("sqlalchemy.engine").setLevel("WARNING")


class CorrelationIdMiddleware:
    """Pure ASGI middleware.

    Deliberately not BaseHTTPMiddleware: that runs the downstream app in a
    separate task, which makes contextvar propagation subtle. Raw ASGI keeps
    the context in one task and lets us stamp the response header cheaply.
    """

    def __init__(self, app: ASGIApp, header_name: str = CORRELATION_ID_HEADER) -> None:
        self.app = app
        self.header_name = header_name

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        inbound = Headers(scope=scope).get(self.header_name)
        correlation_id = inbound or new_correlation_id()

        clear_context()
        set_correlation_id(correlation_id)
        structlog.contextvars.bind_contextvars(
            method=scope.get("method"),
            path=scope.get("path"),
        )
        scope.setdefault("state", {})["correlation_id"] = correlation_id

        log = structlog.get_logger("api.access")
        started = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message)[self.header_name] = correlation_id
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            log.info(
                "request.completed",
                status_code=status_code,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            clear_context()
