"""Unit tests for upload validation and series preparation."""

from __future__ import annotations

import pandas as pd
import pytest

from app.config import settings
from app.core.errors import CsvValidationError
from app.services.csv_validation import (
    load_series_from_path,
    prepare_daily_series,
    validate_csv_bytes,
)
from tests.conftest import make_sales_csv


def csv_from_rows(rows: list[tuple[str, object]]) -> bytes:
    header = f"{settings.date_column},{settings.value_column}\n"
    body = "".join(f"{d},{v}\n" for d, v in rows)
    return (header + body).encode("utf-8")


def dated_rows(count: int, start: str = "2025-01-01", value: float = 10.0) -> list[tuple]:
    index = pd.date_range(start, periods=count, freq="D")
    return [(d.date().isoformat(), value + i) for i, d in enumerate(index)]


def test_accepts_a_well_formed_file() -> None:
    metadata = validate_csv_bytes(make_sales_csv(periods=90, start="2025-01-01"))

    assert metadata.row_count == 90
    assert metadata.start_date == "2025-01-01"
    assert metadata.end_date == "2025-03-31"
    assert metadata.gap_count == 0


def test_rejects_an_empty_upload() -> None:
    with pytest.raises(CsvValidationError, match="empty"):
        validate_csv_bytes(b"")


def test_rejects_a_header_with_no_rows() -> None:
    with pytest.raises(CsvValidationError, match="no data rows"):
        validate_csv_bytes(csv_from_rows([]))


def test_rejects_a_missing_value_column() -> None:
    data = b"date,amount\n2025-01-01,5\n2025-01-02,6\n"

    with pytest.raises(CsvValidationError) as exc_info:
        validate_csv_bytes(data)

    message = str(exc_info.value)
    assert settings.value_column in message
    # The error names what was actually found, so the user can self-serve.
    assert "amount" in message


def test_rejects_unparseable_dates() -> None:
    rows = dated_rows(20)
    rows[3] = ("not-a-date", 11.0)

    with pytest.raises(CsvValidationError, match="not valid dates"):
        validate_csv_bytes(csv_from_rows(rows))


def test_rejects_non_numeric_values() -> None:
    rows = dated_rows(20)
    rows[5] = (rows[5][0], "n/a")

    with pytest.raises(CsvValidationError, match="not numeric"):
        validate_csv_bytes(csv_from_rows(rows))


def test_rejects_a_series_shorter_than_the_minimum() -> None:
    with pytest.raises(CsvValidationError, match=f"at least {settings.min_rows}"):
        validate_csv_bytes(csv_from_rows(dated_rows(settings.min_rows - 1)))


def test_accepts_exactly_the_minimum_rows() -> None:
    metadata = validate_csv_bytes(csv_from_rows(dated_rows(settings.min_rows)))

    assert metadata.row_count == settings.min_rows


def test_duplicate_dates_collapse_to_the_last_row() -> None:
    rows = dated_rows(settings.min_rows + 2)
    rows.append((rows[0][0], 999.0))  # repeat the first date with a new value

    metadata = validate_csv_bytes(csv_from_rows(rows))

    assert metadata.row_count == settings.min_rows + 2


def test_unsorted_input_is_ordered_by_date() -> None:
    rows = list(reversed(dated_rows(30, start="2025-02-01")))

    metadata = validate_csv_bytes(csv_from_rows(rows))

    assert metadata.start_date == "2025-02-01"
    assert metadata.end_date == "2025-03-02"


def test_short_gaps_are_interpolated_and_counted() -> None:
    rows = dated_rows(30)
    del rows[10]  # drop one calendar day

    metadata = validate_csv_bytes(csv_from_rows(rows))

    assert metadata.row_count == 29
    assert metadata.gap_count == 1


def test_rejects_a_series_that_is_mostly_gaps() -> None:
    # 15 points spread over a year: interpolating would invent the data.
    index = pd.date_range("2025-01-01", periods=15, freq="25D")
    rows = [(d.date().isoformat(), 10.0) for d in index]

    with pytest.raises(CsvValidationError, match="too sparse"):
        validate_csv_bytes(csv_from_rows(rows))


def test_rejects_binary_junk() -> None:
    with pytest.raises(CsvValidationError):
        validate_csv_bytes(b"\x00\x01\x02\x89PNG\r\n\x1a\n\xff\xd8\xff")


def test_prepare_daily_series_fills_the_calendar() -> None:
    index = pd.DatetimeIndex(["2025-01-01", "2025-01-02", "2025-01-05"])
    series = pd.Series([10.0, 20.0, 50.0], index=index)

    prepared, gaps = prepare_daily_series(series)

    assert len(prepared) == 5
    assert gaps == 2
    assert not prepared.isna().any()
    assert prepared.loc["2025-01-03"] == pytest.approx(30.0)


def test_load_series_from_path_round_trips(tmp_path) -> None:
    path = tmp_path / "sales.csv"
    path.write_bytes(make_sales_csv(periods=40))

    series = load_series_from_path(path)

    assert len(series) == 40
    assert series.index.is_monotonic_increasing


def test_load_series_from_path_reports_a_missing_file(tmp_path) -> None:
    # Permanent, not transient: the stored upload is gone, so retrying the
    # same job can never succeed.
    with pytest.raises(CsvValidationError, match="missing"):
        load_series_from_path(tmp_path / "absent.csv")


def test_column_names_are_whitespace_tolerant() -> None:
    header = f" {settings.date_column} , {settings.value_column} \n"
    body = "".join(f"{d},{v}\n" for d, v in dated_rows(20))

    metadata = validate_csv_bytes((header + body).encode("utf-8"))

    assert metadata.row_count == 20
