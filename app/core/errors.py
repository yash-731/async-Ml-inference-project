"""Error taxonomy shared by the API and the worker.

The split between permanent and transient is the whole basis of the retry
policy, so it lives in one place rather than being re-decided at each
call site.
"""

from __future__ import annotations


class JobError(Exception):
    """Base class for domain errors attached to a job."""

    error_type = "job_error"


class PermanentJobError(JobError):
    """The input will never produce a result. Retrying is wasted work."""

    error_type = "permanent_error"


class TransientJobError(JobError):
    """A dependency was briefly unavailable. Retrying may well succeed."""

    error_type = "transient_error"


class CsvValidationError(PermanentJobError):
    """The uploaded CSV is structurally unusable.

    Raised during the synchronous upload check (becomes a 422) and again in the
    worker if the file somehow degrades between submission and execution.
    """

    error_type = "csv_validation_error"


class ModelNotFoundError(PermanentJobError):
    error_type = "model_not_found"
