"""CSV parsing, validation, and time-series preparation.

Structural validation runs synchronously during upload so a malformed file
gets an immediate 422 instead of a job that fails 30 seconds later. The worker
reuses the exact same loader, so there is no second, divergent definition of
"valid input".
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.config import settings
from app.core.errors import CsvValidationError

MAX_GAP_RATIO = 0.5


@dataclass(frozen=True)
class SeriesMetadata:
    row_count: int
    start_date: str
    end_date: str
    gap_count: int


def _read_frame(buffer: io.BytesIO | Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(buffer)
    except pd.errors.EmptyDataError as exc:
        raise CsvValidationError("The file is empty or has no header row.") from exc
    except pd.errors.ParserError as exc:
        raise CsvValidationError(f"The file is not parseable as CSV: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise CsvValidationError("The file is not valid UTF-8 text.") from exc
    return frame


def _frame_to_series(frame: pd.DataFrame) -> pd.Series:
    date_column = settings.date_column
    value_column = settings.value_column

    frame.columns = [str(c).strip() for c in frame.columns]
    missing = [c for c in (date_column, value_column) if c not in frame.columns]
    if missing:
        found = ", ".join(frame.columns) or "none"
        raise CsvValidationError(
            f"Missing required column(s): {', '.join(missing)}. Columns found: {found}."
        )

    if frame.empty:
        raise CsvValidationError("The file has a header row but no data rows.")

    dates = pd.to_datetime(frame[date_column], errors="coerce")
    bad_dates = int(dates.isna().sum())
    if bad_dates:
        raise CsvValidationError(
            f"Column '{date_column}' has {bad_dates} value(s) that are not valid dates."
        )

    values = pd.to_numeric(frame[value_column], errors="coerce")
    bad_values = int(values.isna().sum())
    if bad_values:
        raise CsvValidationError(
            f"Column '{value_column}' has {bad_values} value(s) that are not numeric."
        )

    series = pd.Series(
        values.to_numpy(dtype="float64"),
        index=pd.DatetimeIndex(dates).normalize(),
        name=value_column,
    ).sort_index()

    # Repeated dates: last row wins. Silently summing them would be a guess
    # about the caller's intent that we have no basis for.
    series = series[~series.index.duplicated(keep="last")]

    if len(series) < settings.min_rows:
        raise CsvValidationError(
            f"Need at least {settings.min_rows} dated rows to forecast; got {len(series)}."
        )
    return series


def prepare_daily_series(series: pd.Series) -> tuple[pd.Series, int]:
    """Reindex onto a gap-free daily grid.

    The baseline model assumes a regular daily cadence. Short gaps are
    interpolated; a mostly-empty calendar is rejected rather than invented.
    """
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
    reindexed = series.reindex(full_index)
    gap_count = int(reindexed.isna().sum())

    if gap_count and gap_count / len(full_index) > MAX_GAP_RATIO:
        raise CsvValidationError(
            f"Series spans {len(full_index)} days but only {len(series)} are present; "
            "too sparse to interpolate reliably."
        )

    filled = reindexed.interpolate(method="linear", limit_direction="both")
    return filled, gap_count


def validate_csv_bytes(data: bytes) -> SeriesMetadata:
    """Validate an uploaded file. Raises CsvValidationError on bad input."""
    if not data.strip():
        raise CsvValidationError("The uploaded file is empty.")

    series = _frame_to_series(_read_frame(io.BytesIO(data)))
    prepared, gap_count = prepare_daily_series(series)
    return SeriesMetadata(
        row_count=len(series),
        start_date=prepared.index.min().date().isoformat(),
        end_date=prepared.index.max().date().isoformat(),
        gap_count=gap_count,
    )


def load_series_from_path(path: str | Path) -> pd.Series:
    """Load a stored CSV into a model-ready daily series (worker side)."""
    file_path = Path(path)
    if not file_path.exists():
        raise CsvValidationError(f"Stored upload is missing at {file_path}.")

    series = _frame_to_series(_read_frame(file_path))
    prepared, _ = prepare_daily_series(series)
    return prepared
