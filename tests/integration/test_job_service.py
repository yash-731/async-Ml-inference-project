"""Tests for the two ordering guarantees in job submission.

These are the rules that are easy to get wrong and expensive to debug in
production, so they are asserted directly rather than only through the API.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.config import settings
from app.core.errors import CsvValidationError, TransientJobError
from app.db.models import Job, JobStatus
from app.services import job_service
from tests.conftest import make_sales_csv

pytestmark = pytest.mark.integration


def submit(session, data: bytes, enqueue, **kwargs):
    return job_service.submit_job(
        session,
        data=data,
        filename=kwargs.pop("filename", "sales.csv"),
        correlation_id=kwargs.pop("correlation_id", "cid-service-test"),
        enqueue=enqueue,
        **kwargs,
    )


def test_a_job_is_created_with_the_configured_defaults(
    db_session, recorded_enqueue, unique_csv
) -> None:
    calls, enqueue = recorded_enqueue

    submission = submit(db_session, unique_csv, enqueue)

    assert submission.duplicate is False
    job = submission.job
    assert job.status is JobStatus.pending
    assert job.model_name == settings.default_model
    assert job.horizon == settings.forecast_horizon
    assert job.max_retries == settings.task_max_retries
    assert job.celery_task_id == calls[0]["task_id"]


def test_the_row_is_committed_before_the_task_is_enqueued(db_session, unique_csv) -> None:
    """Order matters: a worker that starts before the commit sees no job.

    The enqueue hook queries for the row, so this fails loudly if the two
    steps are ever swapped.
    """
    observed: dict[str, bool] = {}

    def enqueue(job_id, correlation_id):
        observed["row_visible"] = db_session.get(Job, job_id) is not None
        return "task-ordering-1"

    submit(db_session, unique_csv, enqueue)

    assert observed["row_visible"] is True


def test_a_broker_outage_leaves_a_recoverable_job(db_session, unique_csv) -> None:
    """The accepted tradeoff of committing first, made explicit.

    If the broker is down the job is already persisted, so it survives as a
    `pending` row with no task id -- exactly the shape a sweeper would look
    for to requeue it. Nothing is silently lost.
    """

    def failing_enqueue(job_id, correlation_id):
        raise TransientJobError("broker unreachable")

    with pytest.raises(TransientJobError):
        submit(db_session, unique_csv, failing_enqueue)

    key = job_service.compute_idempotency_key(unique_csv)
    job = db_session.scalar(select(Job).where(Job.idempotency_key == key))
    assert job is not None
    assert job.status is JobStatus.pending
    assert job.celery_task_id is None


def test_resubmitting_the_same_bytes_returns_the_original_job(
    db_session, recorded_enqueue, unique_csv
) -> None:
    calls, enqueue = recorded_enqueue
    first = submit(db_session, unique_csv, enqueue)

    second = submit(db_session, unique_csv, enqueue)

    assert second.duplicate is True
    assert second.job.id == first.job.id
    assert len(calls) == 1


def test_losing_the_idempotency_race_returns_the_winning_job(
    db_session, recorded_enqueue, unique_csv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unique constraint, not the pre-check, is the real guarantee.

    Simulates the interleaving where the duplicate lookup misses because the
    competing insert lands immediately after it runs. The INSERT then violates
    the unique index, and we return the winner instead of erroring.
    """
    calls, enqueue = recorded_enqueue
    winner = submit(db_session, unique_csv, enqueue).job

    real_scalar = db_session.scalar
    first_lookup = {"pending": True}

    def blind_first_lookup(statement, *args, **kwargs):
        if first_lookup["pending"]:
            first_lookup["pending"] = False
            return None  # the pre-check misses
        return real_scalar(statement, *args, **kwargs)

    monkeypatch.setattr(db_session, "scalar", blind_first_lookup)

    submission = submit(db_session, unique_csv, enqueue)

    assert submission.duplicate is True
    assert submission.job.id == winner.id
    assert len(calls) == 1  # still only one task, despite two submissions


def test_an_invalid_file_is_rejected_before_anything_is_persisted(
    db_session, recorded_enqueue
) -> None:
    calls, enqueue = recorded_enqueue

    with pytest.raises(CsvValidationError):
        submit(db_session, b"date,amount\n2025-01-01,1\n", enqueue)

    assert db_session.scalar(select(Job.id)) is None
    assert calls == []


def test_listing_paginates_and_counts_independently(db_session, recorded_enqueue) -> None:
    _calls, enqueue = recorded_enqueue
    for seed in range(4):
        submit(db_session, make_sales_csv(seed=5000 + seed), enqueue)

    page_one, total = job_service.list_jobs(db_session, page=1, page_size=3)
    page_two, total_again = job_service.list_jobs(db_session, page=2, page_size=3)

    assert total == total_again == 4
    assert len(page_one) == 3
    assert len(page_two) == 1
    assert {job.id for job in page_one}.isdisjoint({job.id for job in page_two})


def test_listing_can_filter_by_status(db_session, recorded_enqueue) -> None:
    _calls, enqueue = recorded_enqueue
    first = submit(db_session, make_sales_csv(seed=6001), enqueue).job
    submit(db_session, make_sales_csv(seed=6002), enqueue)
    first.status = JobStatus.success
    db_session.commit()

    done, done_total = job_service.list_jobs(db_session, status=JobStatus.success)
    _pending, pending_total = job_service.list_jobs(db_session, status=JobStatus.pending)

    assert done_total == 1
    assert done[0].id == first.id
    assert pending_total == 1


def test_a_long_filename_is_truncated_to_fit_the_column(
    db_session, recorded_enqueue, unique_csv
) -> None:
    _calls, enqueue = recorded_enqueue

    job = submit(db_session, unique_csv, enqueue, filename="a" * 400 + ".csv").job

    assert len(job.original_filename) == 255
