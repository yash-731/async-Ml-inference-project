# Async Forecasting Service

Upload a CSV of historical values, a background worker forecasts it, and you poll for the
result. FastAPI on the front, Celery and Redis in the middle, PostgreSQL as the system of
record.

The forecasting model is deliberately a simple statistical baseline. The point of the project
is everything around it: the asynchronous job lifecycle, idempotent submission, a retry and
dead-letter policy that distinguishes bad input from broken infrastructure, and request
tracing that survives the hop from the API process into the worker process.

---

## What it does

```
POST /jobs          -> 202 Accepted + job_id      (returns immediately)
GET  /jobs/{id}     -> pending | running | success | failed | dead_letter
GET  /jobs          -> paginated history
```

A CSV needs a `date` column and a `value` column, and at least 14 dated rows. The service
validates structure synchronously so a malformed file gets a `422` on upload instead of a
job that fails half a minute later.

---

## Architecture

```mermaid
flowchart LR
    client([Client])

    subgraph api["API container"]
        fastapi["FastAPI<br/>uvicorn"]
    end

    subgraph worker["Worker container"]
        celery["Celery worker<br/>concurrency 2"]
    end

    redis[("Redis<br/>broker only")]
    postgres[("PostgreSQL<br/>job metadata + results")]
    volume[("Shared volume<br/>uploaded CSVs")]

    client -->|"POST /jobs (multipart)"| fastapi
    client -->|"GET /jobs/{id}"| fastapi

    fastapi -->|"1. validate + INSERT"| postgres
    fastapi -->|"2. write CSV"| volume
    fastapi -->|"3. enqueue after commit"| redis

    redis -->|"deliver task"| celery
    celery -->|"read CSV"| volume
    celery -->|"UPDATE status + result"| postgres

    fastapi -->|"read status + result"| postgres
```

Both containers run the **same image** with different commands, so the API and the worker can
never disagree about what a task payload looks like.

### Request lifecycle

```mermaid
sequenceDiagram
    participant C as Client
    participant A as FastAPI
    participant P as PostgreSQL
    participant R as Redis
    participant W as Celery worker

    C->>A: POST /jobs (sales.csv)
    A->>A: generate correlation_id
    A->>A: validate CSV structure
    A->>P: SELECT by idempotency_key (sha256 of bytes)
    alt already submitted
        P-->>A: existing job
        A-->>C: 200 OK (duplicate: true, same job_id)
    else new
        A->>P: INSERT job (pending) + COMMIT
        A->>R: enqueue forecast.run
        A->>P: UPDATE celery_task_id
        A-->>C: 202 Accepted (job_id)
    end

    R->>W: deliver task (carries correlation_id)
    W->>P: status = running
    W->>W: fit model, predict horizon
    alt success
        W->>P: status = success, result = {...}
    else bad data
        W->>P: status = failed (no retry)
    else transient error
        W->>R: retry with exponential backoff
        Note over W,R: after max_retries -> status = dead_letter
    end

    C->>A: GET /jobs/{id}
    A->>P: SELECT
    A-->>C: 200 OK (status + result)
```

---

## Quick start

Requires Docker and Docker Compose.

```bash
docker compose up --build
```

That brings up PostgreSQL, Redis, a one-shot migration job, the API, and one worker. The API
listens on `http://localhost:8000`; interactive docs are at `http://localhost:8000/docs`.

### Submit a job

```bash
# Generate 90 days of sample data
python - <<'PY' > sales.csv
import datetime as dt, math, random
random.seed(7)
print("date,value")
start = dt.date(2025, 1, 1)
for i in range(90):
    d = start + dt.timedelta(days=i)
    weekend = 25 if d.weekday() >= 5 else 0
    print(f"{d},{round(100 + i*0.5 + weekend + random.gauss(0,3), 2)}")
PY

curl -i -X POST http://localhost:8000/jobs \
  -H "X-Request-ID: my-trace-id" \
  -F "file=@sales.csv"
```

```http
HTTP/1.1 202 Accepted
Location: /jobs/6f1c5e0a-...-9b2d
X-Request-ID: my-trace-id

{
  "id": "6f1c5e0a-...-9b2d",
  "status": "pending",
  "correlation_id": "my-trace-id",
  "created_at": "2026-10-08T12:00:00Z",
  "duplicate": false
}
```

### Poll for the result

```bash
curl -s http://localhost:8000/jobs/6f1c5e0a-...-9b2d | jq
```

```json
{
  "id": "6f1c5e0a-...-9b2d",
  "status": "success",
  "model_name": "seasonal_naive_trend",
  "horizon": 14,
  "row_count": 90,
  "retry_count": 0,
  "result": {
    "model_name": "seasonal_naive_trend",
    "horizon": 14,
    "trained_rows": 90,
    "metrics": {
      "mae": 2.0213,
      "rmse": 2.4861,
      "mape": 1.6057,
      "residual_std": 2.4986,
      "trend_per_day": 0.529095,
      "weekly_seasonality": true
    },
    "points": [
      { "date": "2025-04-01", "yhat": 145.61, "yhat_lower": 140.71, "yhat_upper": 150.5 }
    ]
  },
  "error_type": null,
  "error_message": null
}
```

Submitting the same file again returns `200 OK` with `"duplicate": true` and the original
`job_id`. Nothing is re-queued.

### Other endpoints

| Method | Path             | Notes                                                           |
| ------ | ---------------- | --------------------------------------------------------------- |
| `POST` | `/jobs`          | Rate limited (10/minute per IP by default). `?horizon=`, `?model=` |
| `GET`  | `/jobs/{job_id}` | `404` if unknown, `422` if the id is not a UUID                  |
| `GET`  | `/jobs`          | `?page=`, `?page_size=` (max 100), `?status=`                    |
| `GET`  | `/health`        | Liveness. No dependencies touched                                |
| `GET`  | `/ready`         | Readiness. Checks PostgreSQL and Redis, `503` when degraded      |
| `GET`  | `/docs`          | OpenAPI UI                                                       |

---

## Running without Docker

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env

# PostgreSQL and Redis must be reachable at the URLs in .env
alembic upgrade head

uvicorn app.main:app --reload
celery -A app.worker.celery_app:celery_app worker --loglevel=INFO --queues=forecasts
```

---

## Tests

```bash
pytest                      # everything
pytest -m "not integration" # unit only: no database, no broker, no network
pytest -m integration       # needs PostgreSQL
```

Unit tests cover the forecasting mathematics, CSV validation, and the content-hash
idempotency key. Integration tests run against a real PostgreSQL database and cover every
endpoint, the task's success and failure paths, retry bookkeeping, and the dead-letter
transition.

Integration tests **skip with a clear message** when no database is reachable, so the unit
suite still runs on a machine without Docker. In CI, `REQUIRE_DATABASE=1` turns that skip into
a failure, because a silent skip there would mean the suite never ran.

The test database is created automatically if it does not exist. Point the suite anywhere
with `TEST_DATABASE_URL`, which is handy if you already have PostgreSQL on the default port
and want tests on a throwaway cluster instead:

```bash
TEST_DATABASE_URL="postgresql+psycopg://forecast:forecast@127.0.0.1:55432/forecast_test" \
  REQUIRE_DATABASE=1 pytest
```

### CI

`.github/workflows/ci.yml` runs three jobs on every push:

- **lint** — `ruff check`, `ruff format --check`, `mypy app`, and `docker compose config`
- **test** — the full suite against PostgreSQL 16 and Redis 7 service containers, after
  verifying migrations apply to an empty database
- **build** — builds the Docker image and imports both entrypoints inside it

---

## Project layout

```
app/
  api/routes/jobs.py      HTTP layer: parse, validate, delegate, shape responses
  services/
    job_service.py        submission rules: idempotency, commit-then-enqueue
    csv_validation.py     parsing and series preparation (shared by API and worker)
    forecasting.py        pure model code and the model registry
  worker/
    celery_app.py         Celery config and logging hooks
    tasks.py              the task, its retry policy, and the dead-letter path
  core/
    logging.py            structlog setup and correlation-ID middleware
    errors.py             the permanent/transient error taxonomy
    security.py           rate limiting
  db/models.py            the jobs table
alembic/versions/         migrations
tests/unit/               fast, no I/O
tests/integration/        real PostgreSQL
```

The layering is deliberate: `job_service` can be tested without HTTP, and `forecasting` can be
tested without a database, a broker, or a filesystem.

---

## Plugging in a real model

`app/services/forecasting.py` defines a protocol and a registry. A new model needs one class
and one call:

```python
from app.services.forecasting import ForecastPoint, ForecastResult, register_model


class ProphetModel:
    name = "prophet"

    def fit_predict(self, series: pd.Series, horizon: int) -> ForecastResult:
        ...
        return ForecastResult(
            model_name=self.name,
            horizon=horizon,
            trained_rows=len(series),
            metrics={"mae": mae},
            points=[ForecastPoint(date=..., yhat=..., yhat_lower=..., yhat_upper=...)],
        )


register_model(ProphetModel())
```

Select it per request with `POST /jobs?model=prophet`, or change `DEFAULT_MODEL`. Nothing in
the API, the task, or the database schema changes. An unknown model name is a *permanent*
error, so such a job fails immediately rather than retrying three times.

---

## Design decisions and tradeoffs

### Why asynchronous at all

Forecasting is CPU-bound and unbounded in duration. Doing it inside the request would hold a
connection open, put the response time at the mercy of the slowest upload, and make a client
timeout indistinguishable from a server failure. Returning `202` with a job id turns an
unpredictable request into two predictable ones.

**Tradeoff:** the client has to poll. For a UI, server-sent events or a webhook would be
kinder; polling was chosen because it needs no connection state and no retry logic on the
callback side.

### PostgreSQL is the system of record, Redis is only transport

Celery can store results in Redis, so the database looks redundant at first. It is not:
Redis results expire, are awkward to query, and cannot answer "show me every job that
dead-lettered this week". Job state lives in PostgreSQL, where it is durable, indexed, and
queryable; Redis just moves messages.

Because nothing reads Celery's result backend, the service sets `task_ignore_result = True`.
That is not tidiness. `apply_async` touches the result backend on every publish, and with a
backend configured but unreachable, a single upload blocked for **108 seconds** of backend
reconnect attempts before returning — measured, not theorised. Under a Redis outage that
would hold a thread per upload and time out every client. Ignoring results removes the
failure mode instead of merely shortening it, and bounded broker timeouts turn the remaining
publish failure into a ~4 second `503`.

**Tradeoff:** Celery no longer knows a task's outcome, so Flower and `celery inspect` cannot
show results and `AsyncResult.get()` is useless here. That is acceptable precisely because
the `jobs` table is the thing the API reads. A backend is still configured, so flipping the
flag for local debugging is a one-line change.

### Idempotency is enforced by a unique constraint, not by a lookup

The idempotency key is the SHA-256 of the uploaded bytes, so it needs no client cooperation.
The duplicate check before insert is only there for a clean response; the `UNIQUE` index on
`idempotency_key` is what actually makes submission safe. Two concurrent identical uploads
race, one insert fails with an integrity error, and the loser returns the winner's job.
There is a test that simulates exactly that interleaving.

**Tradeoff:** hashing content means a file with one changed cell is a different job, which is
correct here but would be wrong if clients legitimately retried with slightly different data.
It also means the whole upload is buffered to hash it, which is why there is a size cap.

### The task is enqueued *after* the row is committed

The reverse order allows a worker to pick up a job id that is not yet visible to other
transactions. Committing first closes that window.

**Tradeoff:** this is a dual-write, and the remaining failure mode is a committed job whose
enqueue failed. That is deliberate and visible: the row stays `pending` with
`celery_task_id = NULL`, which is exactly the shape a periodic sweeper would query to requeue.
The API returns `503` with `Retry-After` in that case rather than a `500`, because it is worth
retrying. A full transactional outbox would close the gap properly and is overkill at this
size.

### Two kinds of failure, deliberately kept apart

| Status        | Meaning                                       | Retried? |
| ------------- | --------------------------------------------- | -------- |
| `failed`      | The input is wrong. No attempt will fix it.   | No       |
| `dead_letter` | Retries were exhausted, or something unknown broke. A human should look. | Yes, then given up |

Retries are reserved for errors known to be transient: broker and database connection errors,
and an explicit `TransientJobError`. Anything unexpected goes straight to `dead_letter`
without retrying, on the grounds that an unknown exception is probably a deterministic bug,
and three more attempts would only burn the queue and delay the alert. A soft time limit is
treated the same way: a forecast that blew the limit will blow it again.

Backoff is exponential with jitter, so a dependency recovering from an outage does not get
hit by every retry at the same instant.

**Tradeoff:** dead-lettering is a status transition, not a separate Celery queue. With one
worker and one queue, a dedicated dead-letter consumer would do nothing that
`WHERE status = 'dead_letter'` does not already do, and keeping the error on the job row means
the user polling that job sees why it failed. At higher volume, or with multiple consumers, a
real dead-letter queue starts to earn its keep.

### Late acknowledgement forces the task to be idempotent

`task_acks_late = True` means a task is acknowledged only after it finishes, so a worker
killed mid-forecast gets its message redelivered rather than losing it. The price is
at-least-once delivery, so `run_forecast` short-circuits when the job is already `success`
rather than recomputing and overwriting a result the client may already have read.

`worker_prefetch_multiplier = 1` goes with it: with long tasks, the default of 4 lets one
process hoard messages while its siblings idle.

### Synchronous SQLAlchemy behind an async framework

Route handlers are declared `def`, not `async def`, so FastAPI runs them in a thread pool and
blocking database calls never stall the event loop. The same session code is then reusable
verbatim in the Celery worker, which is an ordinary synchronous process.

**Tradeoff:** a thread per in-flight request instead of a coroutine, so this caps out lower
than a fully async stack. For a workload whose expensive part is in a different process
entirely, one codepath for both services is worth more than the extra concurrency.

### Uploads are content-addressed files on a shared volume

Each CSV is stored as `<sha256>.csv`, written to a temp name and atomically renamed, so the
worker never reads a half-written file. Content addressing means a duplicate upload maps to
the same file for free, and a file orphaned by a failed insert is harmless rather than stale.

**Tradeoff:** a Docker volume shared between API and worker is the one piece here that does not
scale past a single host. S3 or similar is the production answer, which is why the path is
stored in a `storage_path` column rather than being derived — swapping the backend touches
one module.

### Correlation IDs are stored, not just logged

The API accepts an inbound `X-Request-ID` or generates one, binds it into structlog's context,
echoes it on the response, **stores it on the job row**, and passes it in the task's kwargs.
A `task_prerun` signal re-binds it inside the worker. One id therefore covers the HTTP request
and the forecast that happens seconds later in another process, and every log line is JSON
with that id attached. Logging alone would not survive the process hop; storing it on the row
is what makes the trace reconstructable after the fact.

### Smaller calls worth defending

- **Rate limiting in Redis, not in process.** An in-memory counter silently doubles the real
  limit as soon as you run two API containers. Only `POST /jobs` is limited; polling stays
  cheap, or clients get punished for using the async API as intended.
- **`VARCHAR` + `CHECK` instead of a native PostgreSQL enum.** Keeps database-level validation
  without the `ALTER TYPE` dance that a native enum forces on every future migration that
  adds a status.
- **`clock_timestamp()`, not `now()`, for `created_at`.** PostgreSQL's `now()` is fixed at
  transaction start, so rows inserted in one transaction share a timestamp and the
  newest-first listing silently degrades to tie-breaking on a random UUID. An insert-time
  audit column should record when the row was actually written.
- **Offset pagination.** Simple and honest at this size. Offsets degrade on large tables;
  keyset pagination on `(created_at, id)` is the upgrade path.
- **Migrations as a one-shot compose service.** The API and worker both wait for it to exit
  successfully, so neither ever talks to a database whose schema is behind the code. Running
  migrations in the API's entrypoint would race between replicas.
- **Tests run against PostgreSQL, and through Alembic.** SQLite would mean testing different
  JSONB, enum, and constraint semantics than production uses. Applying the real migrations in
  the test fixture, instead of `create_all`, also catches migrations drifting from the models.
- **`200` on a duplicate upload, `202` on a new one.** Both return the same job id to poll;
  the status code says plainly whether anything was created.

---

## Known limitations

- **No authentication.** Every endpoint is open, and rate limiting is per IP. Real deployment
  needs an API key or OAuth, and a per-tenant limit.
- **No sweeper for orphaned jobs.** A job committed while the broker was down stays `pending`
  forever. Celery Beat plus a periodic requeue of `pending AND celery_task_id IS NULL AND
  created_at < now() - interval` would fix it.
- **Single host storage.** The shared Docker volume does not survive scaling the worker onto a
  second machine.
- **No metrics.** Structured logs are there, but queue depth, task duration, and failure rate
  belong in Prometheus, with Flower for ad-hoc queue inspection.
- **Daily cadence assumed.** The series is reindexed onto a daily grid and short gaps are
  interpolated; weekly or hourly data would need the frequency inferred instead.
- **Full result inlined.** A long horizon returns every point in one response. A large forecast
  would want the points paginated or offered as a separate download.
