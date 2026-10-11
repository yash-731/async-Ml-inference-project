"""Celery application and worker-side logging hooks."""

from __future__ import annotations

from typing import Any

import structlog
from celery import Celery
from celery.signals import setup_logging, task_postrun, task_prerun

from app.config import settings
from app.core.logging import (
    clear_context,
    configure_logging,
    new_correlation_id,
    set_correlation_id,
)

celery_app = Celery(
    "ml_inference",
    broker=settings.broker_url,
    backend=settings.result_backend,
    include=["app.worker.tasks"],
)

celery_app.conf.update(
    task_default_queue=settings.task_queue,
    # Reject anything that is not JSON. Celery's pickle serializer turns the
    # broker into a remote code execution path if it is ever reachable.
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Redis is the broker, and nothing more. Job state and results live in
    # PostgreSQL, and no code path in this service ever reads Celery's result
    # backend -- the API answers GET /jobs/{id} from the jobs table.
    #
    # Leaving results on was actively harmful: apply_async touches the result
    # backend on every publish, and when Redis is down that connection retries
    # for about 108 seconds (measured), holding the HTTP request open and
    # burning a thread per upload. Ignoring results removes that failure mode
    # entirely rather than merely shortening it. It also makes
    # task_track_started pointless, since the `running` status the API reports
    # is written to the jobs table by the task itself.
    task_ignore_result=True,
    # Acknowledge only after the task finishes: if the worker is killed
    # mid-forecast the message is redelivered rather than lost. The price is
    # at-least-once delivery, which is why run_forecast is idempotent.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # One message in flight per worker process. With long tasks, the default
    # of 4 lets one process hoard work while its siblings sit idle.
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=200,  # caps the damage from any slow leak
    task_soft_time_limit=settings.task_soft_time_limit,
    task_time_limit=settings.task_time_limit,
    result_expires=60 * 60 * 24,
    # The worker may wait patiently for a broker to come back.
    broker_connection_retry_on_startup=True,
    # The API may not. A publish that cannot reach the broker must fail in
    # seconds, not minutes: the job row is already committed, so the caller
    # gets a retryable 503 and the row stays requeueable. One quick retry
    # still absorbs a momentary blip.
    broker_connection_timeout=2,
    broker_transport_options={
        "socket_connect_timeout": 2,
        "socket_timeout": 2,
        "retry_policy": {"timeout": 3.0},
    },
    task_publish_retry=True,
    task_publish_retry_policy={
        "max_retries": 2,
        "interval_start": 0,
        "interval_step": 0.2,
        "interval_max": 0.5,
    },
    # Defence in depth: if someone turns results back on for debugging, the
    # backend must not reintroduce the two-minute stall described above.
    result_backend_always_retry=False,
    redis_socket_connect_timeout=2,
    redis_socket_timeout=2,
)


@setup_logging.connect
def _setup_worker_logging(**_: Any) -> None:
    """Stop Celery replacing our handlers with its own text format."""
    configure_logging(level=settings.log_level, json_logs=settings.json_logs)


@task_prerun.connect
def _bind_task_context(
    task_id: str | None = None,
    task: Any = None,
    kwargs: dict[str, Any] | None = None,
    **_: Any,
) -> None:
    """Re-bind the API's correlation id inside the worker process.

    This is the join that makes a single id searchable across both services.
    """
    clear_context()
    correlation_id = (kwargs or {}).get("correlation_id") or new_correlation_id()
    set_correlation_id(correlation_id)
    structlog.contextvars.bind_contextvars(
        celery_task_id=task_id,
        task_name=getattr(task, "name", None),
    )


@task_postrun.connect
def _clear_task_context(**_: Any) -> None:
    clear_context()
