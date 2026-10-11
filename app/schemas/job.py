"""API request/response models.

`result` is typed as a free-form object rather than a strict schema: the whole
point of the model registry is that a future Prophet/XGBoost model can return
richer output without a breaking API change. The shape is documented in the
README instead of enforced here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import JobStatus


class _ORMModel(BaseModel):
    # protected_namespaces=() silences pydantic's warning about `model_name`,
    # which collides with its reserved `model_` prefix.
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class JobCreatedResponse(_ORMModel):
    id: UUID
    status: JobStatus
    correlation_id: str
    created_at: datetime
    duplicate: bool = Field(
        default=False,
        description="True when this upload matched an existing job and was not re-queued.",
    )


class JobDetailResponse(_ORMModel):
    id: UUID
    status: JobStatus
    model_name: str
    horizon: int
    original_filename: str
    row_count: int | None
    result: dict[str, Any] | None
    error_type: str | None
    error_message: str | None
    retry_count: int
    max_retries: int
    correlation_id: str
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class JobSummary(_ORMModel):
    id: UUID
    status: JobStatus
    model_name: str
    original_filename: str
    row_count: int | None
    created_at: datetime
    finished_at: datetime | None


class PaginatedJobs(BaseModel):
    items: list[JobSummary]
    total: int
    page: int
    page_size: int
    pages: int


class ErrorResponse(BaseModel):
    detail: str
    error_type: str | None = None
