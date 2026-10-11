"""Job endpoints: submit a CSV, poll status, list history."""

from __future__ import annotations

from uuid import UUID

import structlog
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from app.api.deps import correlation_id, get_db
from app.config import settings
from app.core.errors import CsvValidationError, TransientJobError
from app.core.security import limiter
from app.db.models import JobStatus
from app.schemas.job import (
    ErrorResponse,
    JobCreatedResponse,
    JobDetailResponse,
    JobSummary,
    PaginatedJobs,
)
from app.services import job_service

router = APIRouter(prefix="/jobs", tags=["jobs"])
log = structlog.get_logger(__name__)

CSV_CONTENT_TYPES = frozenset(
    {
        "text/csv",
        "application/csv",
        "text/plain",
        "application/vnd.ms-excel",
        "application/octet-stream",
    }
)
CHUNK_SIZE = 64 * 1024


def _read_upload(upload: UploadFile, limit: int) -> bytes:
    """Read the body in chunks, refusing anything over the cap.

    Streaming the read means an oversized upload is rejected as soon as it
    crosses the limit, instead of after the whole body is buffered in memory.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = upload.file.read(CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds the {limit} byte limit.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


@router.post(
    "",
    response_model=JobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a CSV for asynchronous forecasting",
    responses={
        200: {
            "model": JobCreatedResponse,
            "description": "Duplicate upload; the existing job is returned",
        },
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Not a CSV upload"},
        422: {"model": ErrorResponse, "description": "CSV failed validation"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        503: {"model": ErrorResponse, "description": "Queue unavailable; retry"},
    },
)
# Callable, not a bare string: the limit is read per request, so it can be
# changed by configuration (and overridden in tests) without a code change.
@limiter.limit(lambda: settings.upload_rate_limit)
def create_job(
    request: Request,
    response: Response,
    file: UploadFile = File(..., description="CSV containing date and value columns"),
    horizon: int | None = Query(default=None, ge=1, le=365),
    model: str | None = Query(default=None, description="Registered model name"),
    session: Session = Depends(get_db),
    cid: str = Depends(correlation_id),
) -> JobCreatedResponse:
    filename = file.filename or "upload.csv"
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    looks_like_csv = filename.lower().endswith(".csv")
    if content_type and content_type not in CSV_CONTENT_TYPES and not looks_like_csv:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Expected a CSV upload but received content type {content_type}.",
        )

    data = _read_upload(file, settings.max_upload_bytes)

    try:
        submission = job_service.submit_job(
            session,
            data=data,
            filename=filename,
            correlation_id=cid,
            model_name=model,
            horizon=horizon,
        )
    except CsvValidationError as exc:
        # 422 rather than 400: the request itself is well-formed, but its
        # content cannot be processed.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except TransientJobError as exc:
        # The broker is unreachable. The job row is already committed, so it
        # stays visible as `pending` with no task id and can be requeued.
        # 503 tells the client this is worth retrying; a 500 would not.
        log.error("job.enqueue_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The job queue is temporarily unavailable. Please retry.",
            headers={"Retry-After": "10"},
        ) from exc

    job = submission.job
    # 200 on replay states plainly that nothing new was created; 202 means
    # accepted and queued. Both carry the same job id to poll.
    response.status_code = status.HTTP_200_OK if submission.duplicate else status.HTTP_202_ACCEPTED
    response.headers["Location"] = f"/jobs/{job.id}"

    return JobCreatedResponse(
        id=job.id,
        status=job.status,
        correlation_id=job.correlation_id,
        created_at=job.created_at,
        duplicate=submission.duplicate,
    )


@router.get("", response_model=PaginatedJobs, summary="List jobs, newest first")
def list_jobs(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=settings.default_page_size, ge=1, le=settings.max_page_size),
    job_status: JobStatus | None = Query(default=None, alias="status"),
    session: Session = Depends(get_db),
) -> PaginatedJobs:
    rows, total = job_service.list_jobs(
        session,
        page=page,
        page_size=page_size,
        status=job_status,
    )
    return PaginatedJobs(
        items=[JobSummary.model_validate(row) for row in rows],
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size,
    )


@router.get(
    "/{job_id}",
    response_model=JobDetailResponse,
    summary="Poll a job for its status and result",
    responses={404: {"model": ErrorResponse, "description": "No such job"}},
)
def get_job(job_id: UUID, session: Session = Depends(get_db)) -> JobDetailResponse:
    job = job_service.get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return JobDetailResponse.model_validate(job)
