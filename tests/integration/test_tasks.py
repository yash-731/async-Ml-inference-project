"""Tests for the Celery task: success, failure classification, retries, and
the dead-letter path.

The task is executed with `.apply()`, which runs it inline but still drives
Celery's real machinery, including the `on_retry` and `on_failure` hooks that
own the terminal state transitions.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager

import pytest

from app.config import settings
from app.core.errors import PermanentJobError, TransientJobError
from app.db.models import Job, JobStatus
from app.services import job_service
from app.worker import tasks
from app.worker.tasks import run_forecast

pytestmark = pytest.mark.integration


@pytest.fixture
def worker_session(db_session, monkeypatch: pytest.MonkeyPatch):
    """Point the task's session_scope at the test transaction.

    The worker normally opens its own sessions, which would escape the
    rollback that keeps these tests isolated.
    """

    @contextmanager
    def scope():
        try:
            yield db_session
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise

    monkeypatch.setattr(tasks, "session_scope", scope)
    return db_session


@pytest.fixture
def job(db_session, recorded_enqueue, unique_csv) -> Job:
    _calls, enqueue = recorded_enqueue
    submission = job_service.submit_job(
        db_session,
        data=unique_csv,
        filename="sales.csv",
        correlation_id="cid-under-test",
        enqueue=enqueue,
    )
    return submission.job


def run(job_id: uuid.UUID | str):
    return run_forecast.apply(kwargs={"job_id": str(job_id), "correlation_id": "cid-under-test"})


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_a_successful_run_persists_the_forecast(worker_session, job) -> None:
    outcome = run(job.id)

    assert outcome.successful()
    worker_session.refresh(job)
    assert job.status is JobStatus.success
    assert job.result is not None
    assert len(job.result["points"]) == job.horizon
    assert job.result["model_name"] == settings.default_model
    assert job.result["metrics"]["mae"] >= 0
    assert job.started_at is not None
    assert job.finished_at is not None
    assert job.error_message is None
    assert job.error_type is None


def test_the_result_is_shaped_for_the_api(worker_session, job) -> None:
    run(job.id)
    worker_session.refresh(job)

    point = job.result["points"][0]
    assert set(point) == {"date", "yhat", "yhat_lower", "yhat_upper"}
    assert point["yhat_lower"] <= point["yhat"] <= point["yhat_upper"]


def test_the_celery_task_id_is_recorded(worker_session, job) -> None:
    run(job.id)
    worker_session.refresh(job)

    assert job.celery_task_id is not None


# --------------------------------------------------------------------------
# Idempotency under redelivery
# --------------------------------------------------------------------------


def test_redelivery_of_a_successful_job_is_ignored(worker_session, job) -> None:
    # `task_acks_late` makes redelivery possible, so this guard is what stops
    # a second delivery from recomputing and overwriting a published result.
    run(job.id)
    worker_session.refresh(job)
    first_result = job.result
    first_finished_at = job.finished_at

    second = run(job.id)

    worker_session.refresh(job)
    assert second.result["skipped"] is True
    assert job.result == first_result
    assert job.finished_at == first_finished_at


def test_a_missing_job_is_not_retried(worker_session) -> None:
    # Returning instead of raising: there is no job row to retry toward, so a
    # dead-letter entry would have nothing to attach to.
    outcome = run(uuid.uuid4())

    assert outcome.successful()
    assert outcome.result["status"] == "not_found"


# --------------------------------------------------------------------------
# Permanent failures: no retry
# --------------------------------------------------------------------------


def test_corrupt_stored_data_fails_permanently(worker_session, job) -> None:
    with open(job.storage_path, "wb") as handle:
        handle.write(b"date,value\n2025-01-01,not-a-number\n")

    outcome = run(job.id)

    worker_session.refresh(job)
    # The task itself completed: it correctly decided the job cannot succeed.
    assert outcome.successful()
    assert job.status is JobStatus.failed
    assert job.error_type == "csv_validation_error"
    assert job.retry_count == 0
    assert job.finished_at is not None
    assert job.result is None


def test_a_missing_stored_file_fails_permanently(worker_session, job) -> None:
    import os

    os.remove(job.storage_path)

    run(job.id)

    worker_session.refresh(job)
    assert job.status is JobStatus.failed
    assert "missing" in job.error_message.lower()


def test_an_unknown_model_fails_permanently(worker_session, job) -> None:
    job.model_name = "prophet_not_installed"
    worker_session.commit()

    run(job.id)

    worker_session.refresh(job)
    assert job.status is JobStatus.failed
    assert job.error_type == "model_not_found"
    assert job.retry_count == 0


def test_a_permanent_error_is_never_dead_lettered(
    worker_session, job, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tasks,
        "_compute_forecast",
        lambda *a, **k: (_ for _ in ()).throw(PermanentJobError("bad input")),
    )

    run(job.id)

    worker_session.refresh(job)
    # `failed` means "we know why"; `dead_letter` means "a human must look".
    assert job.status is JobStatus.failed
    assert job.status is not JobStatus.dead_letter


# --------------------------------------------------------------------------
# Transient failures: retry, then dead-letter
# --------------------------------------------------------------------------


def test_the_retry_policy_is_configured(worker_session) -> None:
    assert TransientJobError in run_forecast.autoretry_for
    assert run_forecast.max_retries == settings.task_max_retries
    assert run_forecast.retry_backoff == settings.task_retry_backoff
    assert run_forecast.retry_backoff_max == settings.task_retry_backoff_max
    assert run_forecast.retry_jitter is True
    # Late acknowledgement is what makes redelivery, and therefore the
    # idempotency guard above, necessary.
    assert run_forecast.acks_late is True


def test_a_transient_error_is_never_recorded_as_a_permanent_one(
    worker_session, job, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tasks,
        "_compute_forecast",
        lambda *a, **k: (_ for _ in ()).throw(TransientJobError("broker blip")),
    )

    run(job.id)

    worker_session.refresh(job)
    assert job.status is not JobStatus.failed
    assert job.status is not JobStatus.success


def test_on_retry_records_the_attempt(worker_session, job) -> None:
    run_forecast.push_request(retries=1, id="task-retry-1")
    try:
        run_forecast.on_retry(
            TransientJobError("database unavailable"),
            "task-retry-1",
            (),
            {"job_id": str(job.id)},
            None,
        )
    finally:
        run_forecast.pop_request()

    worker_session.refresh(job)
    assert job.retry_count == 2  # attempt number, not retries remaining
    assert job.error_type == "transient_error"
    assert "database unavailable" in job.error_message
    # Still not terminal: a retry is scheduled, so the job is not finished.
    assert job.status is not JobStatus.dead_letter
    assert job.finished_at is None


def test_on_failure_dead_letters_the_job(worker_session, job) -> None:
    run_forecast.push_request(retries=settings.task_max_retries, id="task-failed-1")
    try:
        run_forecast.on_failure(
            TransientJobError("redis down for 10 minutes"),
            "task-failed-1",
            (),
            {"job_id": str(job.id)},
            None,
        )
    finally:
        run_forecast.pop_request()

    worker_session.refresh(job)
    assert job.status is JobStatus.dead_letter
    assert job.error_type == "transient_error"
    assert job.retry_count == settings.task_max_retries
    assert job.finished_at is not None


def test_an_unexpected_exception_dead_letters_immediately(worker_session, job) -> None:
    # Not retried: an unknown exception is probably a deterministic bug, and
    # three more attempts would only delay the alert.
    assert not isinstance(KeyError("x"), run_forecast.autoretry_for)

    tasks._dead_letter(str(job.id), KeyError("unexpected"), retries=0)

    worker_session.refresh(job)
    assert job.status is JobStatus.dead_letter
    assert job.error_type == "KeyError"
    assert job.retry_count == 0


def test_dead_lettering_never_overwrites_a_finished_job(worker_session, job) -> None:
    run(job.id)
    worker_session.refresh(job)
    assert job.status is JobStatus.success

    tasks._dead_letter(str(job.id), TransientJobError("late failure"), retries=3)

    worker_session.refresh(job)
    assert job.status is JobStatus.success
    assert job.result is not None


def test_dead_lettering_an_unknown_job_is_harmless(worker_session) -> None:
    tasks._dead_letter(str(uuid.uuid4()), TransientJobError("nobody home"), retries=1)


def test_a_long_error_message_is_truncated(worker_session, job) -> None:
    tasks._dead_letter(str(job.id), TransientJobError("x" * 5000), retries=1)

    worker_session.refresh(job)
    assert len(job.error_message) == tasks.ERROR_MESSAGE_LIMIT
