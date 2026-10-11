"""Integration tests for the HTTP API against a real PostgreSQL database.

Celery is stubbed out: the endpoint's contract is "the job is persisted and a
task was handed to the broker", not "a forecast has run". The task itself is
covered in test_tasks.py.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from app.config import settings
from app.db.models import Job, JobStatus
from tests.conftest import make_sales_csv

pytestmark = pytest.mark.integration


def upload(client, data: bytes, filename: str = "sales.csv", content_type: str = "text/csv", **kw):
    return client.post(
        "/jobs",
        files={"file": (filename, data, content_type)},
        **kw,
    )


# --------------------------------------------------------------------------
# POST /jobs
# --------------------------------------------------------------------------


def test_valid_upload_is_accepted_and_queued(client, api_enqueue_calls, sales_csv) -> None:
    response = upload(client, sales_csv)

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == JobStatus.pending.value
    assert body["duplicate"] is False
    uuid.UUID(body["id"])  # a parseable job id is the whole point of the 202
    assert response.headers["Location"] == f"/jobs/{body['id']}"
    assert len(api_enqueue_calls) == 1
    assert api_enqueue_calls[0]["job_id"] == uuid.UUID(body["id"])


def test_accepted_job_is_persisted_with_metadata(
    client, api_enqueue_calls, db_session, sales_csv
) -> None:
    body = upload(client, sales_csv).json()

    job = db_session.get(Job, uuid.UUID(body["id"]))
    assert job is not None
    assert job.status is JobStatus.pending
    assert job.original_filename == "sales.csv"
    assert job.row_count == 90
    assert job.model_name == settings.default_model
    assert job.horizon == settings.forecast_horizon
    assert job.max_retries == settings.task_max_retries
    assert job.celery_task_id == api_enqueue_calls[0]["task_id"]
    assert job.result is None


def test_the_same_file_is_never_processed_twice(client, api_enqueue_calls, sales_csv) -> None:
    first = upload(client, sales_csv)
    second = upload(client, sales_csv)

    assert first.status_code == 202
    # 200, not 202: nothing new was created.
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert second.json()["id"] == first.json()["id"]
    # The decisive assertion: one enqueue, not two.
    assert len(api_enqueue_calls) == 1


def test_a_different_file_creates_a_different_job(client, api_enqueue_calls) -> None:
    first = upload(client, make_sales_csv(seed=101))
    second = upload(client, make_sales_csv(seed=202))

    assert second.status_code == 202
    assert first.json()["id"] != second.json()["id"]
    assert len(api_enqueue_calls) == 2


def test_horizon_and_model_can_be_overridden(client, api_enqueue_calls, db_session) -> None:
    response = client.post(
        "/jobs",
        files={"file": ("sales.csv", make_sales_csv(seed=303), "text/csv")},
        params={"horizon": 30, "model": "seasonal_naive_trend"},
    )

    assert response.status_code == 202
    job = db_session.get(Job, uuid.UUID(response.json()["id"]))
    assert job.horizon == 30


def test_a_csv_missing_the_value_column_is_rejected(client, api_enqueue_calls) -> None:
    data = b"date,amount\n2025-01-01,5\n2025-01-02,6\n"

    response = upload(client, data, filename="wrong.csv")

    assert response.status_code == 422
    assert settings.value_column in response.json()["detail"]
    # Nothing reaches the queue, which is the point of validating synchronously.
    assert api_enqueue_calls == []


def test_a_too_short_series_is_rejected(client, api_enqueue_calls) -> None:
    response = upload(client, make_sales_csv(periods=5, seed=404))

    assert response.status_code == 422
    assert str(settings.min_rows) in response.json()["detail"]
    assert api_enqueue_calls == []


def test_an_empty_file_is_rejected(client, api_enqueue_calls) -> None:
    assert upload(client, b"").status_code == 422
    assert api_enqueue_calls == []


def test_a_non_csv_upload_is_rejected(client, api_enqueue_calls) -> None:
    response = upload(client, b"\x00\x01binary", filename="image.png", content_type="image/png")

    assert response.status_code == 415
    assert api_enqueue_calls == []


def test_an_oversized_upload_is_rejected(
    client, api_enqueue_calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "max_upload_bytes", 128)

    response = upload(client, make_sales_csv(periods=90, seed=505))

    assert response.status_code == 413
    assert api_enqueue_calls == []


def test_a_missing_file_field_is_a_validation_error(client) -> None:
    assert client.post("/jobs").status_code == 422


# --------------------------------------------------------------------------
# Correlation IDs
# --------------------------------------------------------------------------


def test_an_inbound_request_id_is_echoed_and_stored(
    client, api_enqueue_calls, db_session, sales_csv
) -> None:
    inbound = "trace-me-0123456789"

    response = upload(client, sales_csv, headers={"X-Request-ID": inbound})

    assert response.headers["X-Request-ID"] == inbound
    body = response.json()
    assert body["correlation_id"] == inbound
    # Stored on the row, so the worker can log under the same id.
    job = db_session.get(Job, uuid.UUID(body["id"]))
    assert job.correlation_id == inbound
    assert api_enqueue_calls[0]["correlation_id"] == inbound


def test_a_request_id_is_generated_when_absent(client, api_enqueue_calls, sales_csv) -> None:
    response = upload(client, sales_csv)

    generated = response.headers["X-Request-ID"]
    assert generated
    assert response.json()["correlation_id"] == generated


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------


def test_upload_endpoint_is_rate_limited(
    client, api_enqueue_calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "upload_rate_limit", "2/minute")

    codes = [upload(client, make_sales_csv(seed=900 + i)).status_code for i in range(3)]

    assert codes[:2] == [202, 202]
    assert codes[2] == 429
    assert len(api_enqueue_calls) == 2


def test_rate_limit_response_explains_itself(
    client, api_enqueue_calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "upload_rate_limit", "1/minute")

    upload(client, make_sales_csv(seed=950))
    blocked = upload(client, make_sales_csv(seed=951))

    assert blocked.status_code == 429
    assert blocked.json()["error_type"] == "rate_limit_exceeded"
    assert blocked.headers["Retry-After"] == "60"


def test_reads_are_not_rate_limited(
    client, api_enqueue_calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The limit protects the expensive path. Polling must stay cheap, or
    # clients get punished for using the async API as designed.
    monkeypatch.setattr(settings, "upload_rate_limit", "1/minute")

    codes = [client.get("/jobs").status_code for _ in range(10)]

    assert codes == [200] * 10


# --------------------------------------------------------------------------
# GET /jobs/{job_id}
# --------------------------------------------------------------------------


def test_polling_a_pending_job(client, api_enqueue_calls, sales_csv) -> None:
    job_id = upload(client, sales_csv).json()["id"]

    response = client.get(f"/jobs/{job_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == job_id
    assert body["status"] == JobStatus.pending.value
    assert body["result"] is None
    assert body["error_message"] is None
    assert body["retry_count"] == 0
    assert body["finished_at"] is None


def test_polling_a_finished_job_returns_the_result(
    client, api_enqueue_calls, db_session, sales_csv
) -> None:
    job_id = upload(client, sales_csv).json()["id"]
    job = db_session.get(Job, uuid.UUID(job_id))
    job.status = JobStatus.success
    job.result = {"model_name": "seasonal_naive_trend", "points": [{"date": "2025-04-01"}]}
    db_session.commit()

    body = client.get(f"/jobs/{job_id}").json()

    assert body["status"] == "success"
    assert body["result"]["points"][0]["date"] == "2025-04-01"


def test_polling_a_failed_job_returns_the_error(
    client, api_enqueue_calls, db_session, sales_csv
) -> None:
    job_id = upload(client, sales_csv).json()["id"]
    job = db_session.get(Job, uuid.UUID(job_id))
    job.status = JobStatus.failed
    job.error_type = "csv_validation_error"
    job.error_message = "Column 'value' has 3 value(s) that are not numeric."
    db_session.commit()

    body = client.get(f"/jobs/{job_id}").json()

    assert body["status"] == "failed"
    assert body["error_type"] == "csv_validation_error"
    assert "not numeric" in body["error_message"]
    assert body["result"] is None


def test_an_unknown_job_is_a_404(client) -> None:
    response = client.get(f"/jobs/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Job not found."


def test_a_malformed_job_id_is_a_422(client) -> None:
    assert client.get("/jobs/not-a-uuid").status_code == 422


# --------------------------------------------------------------------------
# GET /jobs
# --------------------------------------------------------------------------


def test_listing_is_paginated(client, api_enqueue_calls) -> None:
    for seed in range(5):
        upload(client, make_sales_csv(seed=1000 + seed))

    first = client.get("/jobs", params={"page": 1, "page_size": 2}).json()
    second = client.get("/jobs", params={"page": 2, "page_size": 2}).json()
    third = client.get("/jobs", params={"page": 3, "page_size": 2}).json()

    assert first["total"] == second["total"] == 5
    assert first["pages"] == 3
    assert [len(p["items"]) for p in (first, second, third)] == [2, 2, 1]
    ids = [item["id"] for page in (first, second, third) for item in page["items"]]
    assert len(set(ids)) == 5  # no row appears on two pages


def test_listing_is_newest_first(client, api_enqueue_calls, db_session) -> None:
    created = [upload(client, make_sales_csv(seed=1100 + i)).json()["id"] for i in range(3)]

    # Stated explicitly because the ordering depends on it: created_at uses
    # clock_timestamp(), so rows inserted in one transaction still get
    # distinct stamps. With now() these would tie and the order below would
    # collapse to an arbitrary tie-break on a random UUID.
    stamps = [db_session.get(Job, uuid.UUID(job_id)).created_at for job_id in created]
    assert len(set(stamps)) == 3

    items = client.get("/jobs").json()["items"]

    assert [item["id"] for item in items] == list(reversed(created))


def test_listing_can_filter_by_status(client, api_enqueue_calls, db_session) -> None:
    ids = [upload(client, make_sales_csv(seed=1200 + i)).json()["id"] for i in range(3)]
    job = db_session.get(Job, uuid.UUID(ids[0]))
    job.status = JobStatus.dead_letter
    db_session.commit()

    dead = client.get("/jobs", params={"status": "dead_letter"}).json()
    pending = client.get("/jobs", params={"status": "pending"}).json()

    assert dead["total"] == 1
    assert dead["items"][0]["id"] == ids[0]
    assert pending["total"] == 2


def test_an_empty_listing_is_well_formed(client) -> None:
    body = client.get("/jobs").json()

    assert body == {
        "items": [],
        "total": 0,
        "page": 1,
        "page_size": settings.default_page_size,
        "pages": 0,
    }


def test_an_unknown_status_filter_is_rejected(client) -> None:
    assert client.get("/jobs", params={"status": "exploded"}).status_code == 422


def test_page_size_is_capped(client) -> None:
    # Without a cap, one client can ask for the entire table in one query.
    response = client.get("/jobs", params={"page_size": settings.max_page_size + 1})

    assert response.status_code == 422


def test_only_committed_jobs_are_listed(client, api_enqueue_calls, db_session, sales_csv) -> None:
    upload(client, sales_csv)

    assert db_session.scalar(select(Job.id)) is not None
    assert client.get("/jobs").json()["total"] == 1


# --------------------------------------------------------------------------
# Ops endpoints
# --------------------------------------------------------------------------


def test_health_needs_no_dependencies(client) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_checks_the_database(client) -> None:
    body = client.get("/ready").json()

    assert body["checks"]["database"] == "ok"
    # Redis may or may not be up where this runs; the database check is the
    # one this test can assert on.
    assert "redis" in body["checks"]


def test_openapi_schema_documents_the_endpoints(client) -> None:
    schema = client.get("/openapi.json").json()

    assert "/jobs" in schema["paths"]
    assert "/jobs/{job_id}" in schema["paths"]
    assert "202" in schema["paths"]["/jobs"]["post"]["responses"]
