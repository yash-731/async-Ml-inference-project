"""Unit tests for the pieces of job submission that need no database."""

from __future__ import annotations

from app.config import settings
from app.services.job_service import (
    _storage_path_for,
    _write_upload,
    compute_idempotency_key,
)
from tests.conftest import make_sales_csv


def test_identical_bytes_produce_the_same_key() -> None:
    data = make_sales_csv(seed=1)

    assert compute_idempotency_key(data) == compute_idempotency_key(bytes(data))


def test_different_bytes_produce_different_keys() -> None:
    assert compute_idempotency_key(make_sales_csv(seed=1)) != compute_idempotency_key(
        make_sales_csv(seed=2)
    )


def test_a_single_changed_cell_changes_the_key() -> None:
    original = make_sales_csv(periods=20)
    altered = original.replace(b"100", b"101", 1)

    assert original != altered
    assert compute_idempotency_key(original) != compute_idempotency_key(altered)


def test_key_is_a_sha256_hex_digest() -> None:
    key = compute_idempotency_key(b"anything")

    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


def test_storage_is_content_addressed() -> None:
    data = make_sales_csv(seed=11)
    key = compute_idempotency_key(data)

    path = _storage_path_for(key)

    assert path.parent == settings.storage_dir
    assert path.name == f"{key}.csv"


def test_write_upload_persists_the_exact_bytes() -> None:
    data = make_sales_csv(seed=12)
    key = compute_idempotency_key(data)

    path = _write_upload(key, data)

    assert path.exists()
    assert path.read_bytes() == data


def test_write_upload_leaves_no_partial_files() -> None:
    data = make_sales_csv(seed=13)
    key = compute_idempotency_key(data)

    _write_upload(key, data)

    # The temp file is renamed into place, so the worker never sees a
    # half-written CSV.
    assert not list(settings.storage_dir.glob("*.part"))


def test_rewriting_the_same_content_is_a_no_op() -> None:
    data = make_sales_csv(seed=14)
    key = compute_idempotency_key(data)

    first = _write_upload(key, data)
    mtime = first.stat().st_mtime_ns
    second = _write_upload(key, data)

    assert first == second
    assert second.stat().st_mtime_ns == mtime
