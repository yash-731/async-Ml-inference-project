"""Job submission and lookup.

Holds the two rules that matter most and that neither the route nor the task
should be free to re-interpret:

1. The same bytes never produce two jobs (idempotency).
2. The task is enqueued only after the row is committed, so the worker can
   never pick up a job id that does not exist yet.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import structlog
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import settings
from app.core.errors import TransientJobError
from app.db.models import Job, JobStatus
from app.services.csv_validation import validate_csv_bytes

log = structlog.get_logger(__name__)

EnqueueFn = Callable[[uuid.UUID, str], str]


@dataclass(frozen=True)
class JobSubmission:
    job: Job
    duplicate: bool


def compute_idempotency_key(data: bytes) -> str:
    """SHA-256 of the raw upload.

    Content-addressed, so it needs no client cooperation -- unlike an
    Idempotency-Key header, which a naive client simply would not send.
    """
    return hashlib.sha256(data).hexdigest()


def _storage_path_for(key: str) -> Path:
    # Named by content hash: a duplicate upload maps to the same file, and a
    # file orphaned by a failed insert is harmless rather than a stale result.
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return settings.storage_dir / f"{key}.csv"


def _write_upload(key: str, data: bytes) -> Path:
    path = _storage_path_for(key)
    if path.exists() and path.stat().st_size == len(data):
        return path
    tmp = path.with_suffix(".csv.part")
    tmp.write_bytes(data)
    tmp.replace(path)  # atomic: the worker never sees a half-written file
    return path


def _enqueue_forecast(job_id: uuid.UUID, correlation_id: str) -> str:
    """Default enqueue hook. Imported lazily to keep Celery out of the API's
    import graph (and out of the way of tests that stub this function)."""
    from app.worker.tasks import run_forecast

    try:
        async_result = run_forecast.apply_async(
            kwargs={"job_id": str(job_id), "correlation_id": correlation_id},
            queue=settings.task_queue,
        )
    except Exception as exc:  # broker unreachable
        raise TransientJobError(f"Could not enqueue job: {exc}") from exc
    return str(async_result.id)


def get_job(session: Session, job_id: uuid.UUID) -> Job | None:
    return session.get(Job, job_id)


def list_jobs(
    session: Session,
    *,
    page: int = 1,
    page_size: int = 20,
    status: JobStatus | None = None,
) -> tuple[list[Job], int]:
    """Offset pagination with a matching COUNT.

    Offsets degrade on very large tables; keyset pagination on (created_at, id)
    is the upgrade path. At this scale the simpler contract wins.
    """
    filters = [Job.status == status] if status is not None else []

    total = session.scalar(select(func.count()).select_from(Job).where(*filters)) or 0
    rows = (
        session.scalars(
            select(Job)
            .where(*filters)
            .order_by(Job.created_at.desc(), Job.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        .unique()
        .all()
    )
    return list(rows), total


def submit_job(
    session: Session,
    *,
    data: bytes,
    filename: str,
    correlation_id: str,
    model_name: str | None = None,
    horizon: int | None = None,
    enqueue: EnqueueFn | None = None,
) -> JobSubmission:
    """Validate, persist, and enqueue a forecasting job."""
    enqueue = enqueue or _enqueue_forecast
    key = compute_idempotency_key(data)

    existing = session.scalar(select(Job).where(Job.idempotency_key == key))
    if existing is not None:
        log.info("job.duplicate_upload", job_id=str(existing.id), status=existing.status.value)
        return JobSubmission(job=existing, duplicate=True)

    # Cheap structural check first: a bad file should never reach the queue.
    metadata = validate_csv_bytes(data)
    storage_path = _write_upload(key, data)

    job = Job(
        id=uuid.uuid4(),
        idempotency_key=key,
        status=JobStatus.pending,
        model_name=model_name or settings.default_model,
        horizon=horizon or settings.forecast_horizon,
        original_filename=filename[:255],
        storage_path=str(storage_path),
        row_count=metadata.row_count,
        max_retries=settings.task_max_retries,
        correlation_id=correlation_id,
    )
    session.add(job)

    try:
        session.commit()
    except IntegrityError:
        # Lost a race against a concurrent identical upload. The unique
        # constraint did its job; return the winner instead of erroring.
        session.rollback()
        winner = session.scalar(select(Job).where(Job.idempotency_key == key))
        if winner is None:
            raise
        log.info("job.duplicate_race_lost", job_id=str(winner.id))
        return JobSubmission(job=winner, duplicate=True)

    # Enqueue strictly after the commit. The reverse order lets the worker
    # start before the row is visible. The residual risk -- committed but
    # never queued, if the broker is down -- is visible as a `pending` job
    # with no celery_task_id, which a sweeper task could requeue later.
    task_id = enqueue(job.id, correlation_id)
    job.celery_task_id = task_id
    session.commit()

    log.info(
        "job.submitted",
        job_id=str(job.id),
        celery_task_id=task_id,
        rows=metadata.row_count,
        gaps=metadata.gap_count,
        model=job.model_name,
    )
    return JobSubmission(job=job, duplicate=False)
