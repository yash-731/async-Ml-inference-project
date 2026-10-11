"""The forecasting task, plus its retry and dead-letter policy.

Failure policy, stated once so it is not re-decided ad hoc:

  PermanentJobError      -> status `failed`, no retry. The data is wrong and
                            no number of attempts will change that.
  Known transient errors -> exponential backoff with jitter, up to
                            max_retries, then status `dead_letter`.
  Anything unexpected    -> status `dead_letter` immediately, no retry.
                            An unknown exception is probably a bug, and
                            retrying a deterministic bug just burns the queue
                            three times and delays the alert.
  Timeout                -> status `dead_letter`, no retry. A forecast that
                            blew the time limit will blow it again.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from kombu.exceptions import OperationalError as KombuOperationalError
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy.exc import OperationalError as SQLOperationalError

from app.config import settings
from app.core.errors import PermanentJobError, TransientJobError
from app.db.base import session_scope
from app.db.models import Job, JobStatus
from app.services.csv_validation import load_series_from_path
from app.services.forecasting import get_model
from app.worker.celery_app import celery_app

log = structlog.get_logger(__name__)

# Only these are retried. Everything else is treated as a bug (see module docstring).
RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    TransientJobError,
    SQLOperationalError,
    KombuOperationalError,
    RedisConnectionError,
)

ERROR_MESSAGE_LIMIT = 2000


def _now() -> datetime:
    return datetime.now(UTC)


def _truncate(message: str) -> str:
    return message[:ERROR_MESSAGE_LIMIT]


def _update_job(job_id: str, **fields: Any) -> bool:
    """Apply field updates in a short, self-contained transaction.

    Short on purpose: the forecast itself runs with no database transaction
    open, so a slow model never holds a connection from the pool.
    """
    try:
        with session_scope() as session:
            job = session.get(Job, uuid.UUID(job_id))
            if job is None:
                return False
            for key, value in fields.items():
                setattr(job, key, value)
            return True
    except Exception:
        log.exception("task.job_update_failed", job_id=job_id)
        return False


class JobTask(Task):
    """Base task owning the terminal-state transitions Celery drives.

    Putting these in `on_retry`/`on_failure` rather than in the task body means
    they still fire for failures the body never sees, such as the worker being
    killed or a retry being scheduled by `autoretry_for`.
    """

    def on_retry(
        self,
        exc: BaseException,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        job_id = (kwargs or {}).get("job_id")
        if not job_id:
            return
        attempts = self.request.retries + 1
        log.warning(
            "task.retry_scheduled",
            job_id=job_id,
            attempt=attempts,
            max_retries=self.max_retries,
            error=str(exc),
            error_class=type(exc).__name__,
        )
        _update_job(
            job_id,
            retry_count=attempts,
            error_type=getattr(exc, "error_type", type(exc).__name__),
            error_message=_truncate(str(exc)),
        )

    def on_failure(
        self,
        exc: BaseException,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        """Last stop: retries are exhausted or the error was never retryable.

        This is the dead-letter path. It is a status transition rather than a
        separate Celery queue because with one worker and one queue, a
        dedicated dead-letter consumer would have nothing to do that a
        `WHERE status = 'dead_letter'` query does not already do -- and the
        row keeps the error attached to the job the user is polling.
        """
        job_id = (kwargs or {}).get("job_id")
        if not job_id:
            log.error("task.failed_without_job_id", celery_task_id=task_id, error=str(exc))
            return

        log.error(
            "task.dead_lettered",
            job_id=job_id,
            celery_task_id=task_id,
            retries=self.request.retries,
            error=str(exc),
            error_class=type(exc).__name__,
        )
        _dead_letter(job_id, exc, retries=self.request.retries)


def _dead_letter(job_id: str, exc: BaseException, retries: int) -> None:
    """Move a job to dead_letter unless it already reached a terminal state."""
    try:
        with session_scope() as session:
            job = session.get(Job, uuid.UUID(job_id))
            if job is None or job.is_terminal:
                return
            job.status = JobStatus.dead_letter
            job.error_type = getattr(exc, "error_type", type(exc).__name__)
            job.error_message = _truncate(str(exc) or type(exc).__name__)
            job.retry_count = retries
            job.finished_at = _now()
    except Exception:
        log.exception("task.dead_letter_failed", job_id=job_id)


def _compute_forecast(storage_path: str, model_name: str, horizon: int) -> dict[str, Any]:
    """Pure pipeline step: file in, result dict out. No ORM, no Celery."""
    series = load_series_from_path(storage_path)
    model = get_model(model_name)
    return model.fit_predict(series, horizon).to_dict()


@celery_app.task(
    base=JobTask,
    bind=True,
    name="forecast.run",
    queue=settings.task_queue,
    autoretry_for=RETRYABLE_ERRORS,
    retry_backoff=settings.task_retry_backoff,
    retry_backoff_max=settings.task_retry_backoff_max,
    retry_jitter=True,  # spreads a thundering herd of simultaneous retries
    max_retries=settings.task_max_retries,
)
def run_forecast(self: Task, *, job_id: str, correlation_id: str | None = None) -> dict[str, Any]:
    """Run the forecast for a job and persist the outcome.

    Idempotent by design: `task_acks_late` makes redelivery possible, so an
    already-successful job short-circuits instead of recomputing and
    overwriting a result the user may already have read.
    """
    bound = log.bind(job_id=job_id, attempt=self.request.retries + 1)

    with session_scope() as session:
        job = session.get(Job, uuid.UUID(job_id))
        if job is None:
            # Nothing to retry toward. Raising here would just dead-letter a
            # task whose job row no longer exists.
            bound.error("task.job_not_found")
            return {"job_id": job_id, "status": "not_found"}

        if job.status is JobStatus.success:
            bound.info("task.redelivery_ignored", status=job.status.value)
            return {"job_id": job_id, "status": job.status.value, "skipped": True}

        job.status = JobStatus.running
        job.started_at = job.started_at or _now()
        job.celery_task_id = self.request.id
        job.retry_count = self.request.retries
        # Snapshot what the computation needs so the transaction can close.
        storage_path = job.storage_path
        model_name = job.model_name
        horizon = job.horizon

    bound.info("task.started", model=model_name, horizon=horizon)

    try:
        result = _compute_forecast(storage_path, model_name, horizon)

    except PermanentJobError as exc:
        # A business failure, not an infrastructure one. The task completed
        # successfully at the thing it was asked to do: decide the outcome.
        bound.warning(
            "task.permanent_failure",
            error=str(exc),
            error_type=exc.error_type,
        )
        _update_job(
            job_id,
            status=JobStatus.failed,
            error_type=exc.error_type,
            error_message=_truncate(str(exc)),
            finished_at=_now(),
        )
        return {"job_id": job_id, "status": JobStatus.failed.value, "error": str(exc)}

    except SoftTimeLimitExceeded as exc:
        bound.error("task.timeout", soft_limit=settings.task_soft_time_limit)
        _dead_letter(job_id, exc, retries=self.request.retries)
        return {"job_id": job_id, "status": JobStatus.dead_letter.value, "error": "timeout"}

    # Retryable errors are left to propagate: `autoretry_for` schedules the
    # backoff and `on_retry`/`on_failure` record it. Unexpected exceptions
    # also propagate, straight to `on_failure` and the dead-letter status.

    _update_job(
        job_id,
        status=JobStatus.success,
        result=result,
        error_type=None,
        error_message=None,
        finished_at=_now(),
    )
    bound.info(
        "task.succeeded",
        points=len(result.get("points", [])),
        mae=result.get("metrics", {}).get("mae"),
    )
    return {"job_id": job_id, "status": JobStatus.success.value}
