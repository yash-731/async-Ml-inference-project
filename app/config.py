"""Application configuration.

Everything is read from the environment via pydantic-settings so the same image
runs locally, in docker-compose, and in CI with no code changes.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- app -------------------------------------------------------------
    app_name: str = "ml-inference-service"
    environment: str = "development"
    log_level: str = "INFO"
    json_logs: bool = True

    # --- persistence -----------------------------------------------------
    database_url: str = "postgresql+psycopg://forecast:forecast@localhost:5432/forecast"
    db_pool_size: int = 5
    db_max_overflow: int = 5

    # --- broker ----------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None
    task_queue: str = "forecasts"

    # --- uploads / validation -------------------------------------------
    storage_dir: Path = Path("./var/uploads")
    max_upload_bytes: int = 10 * 1024 * 1024
    min_rows: int = 14
    date_column: str = "date"
    value_column: str = "value"

    # --- rate limiting ---------------------------------------------------
    rate_limit_enabled: bool = True
    upload_rate_limit: str = "10/minute"
    rate_limit_storage_uri: str | None = None

    # --- forecasting -----------------------------------------------------
    forecast_horizon: int = 14
    default_model: str = "seasonal_naive_trend"

    # --- task reliability ------------------------------------------------
    task_max_retries: int = 3
    task_retry_backoff: int = 2
    task_retry_backoff_max: int = 120
    task_soft_time_limit: int = 240
    task_time_limit: int = 300

    # --- api -------------------------------------------------------------
    default_page_size: int = 20
    max_page_size: int = 100

    @property
    def broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def result_backend(self) -> str:
        return self.celery_result_backend or self.redis_url

    @property
    def limiter_storage_uri(self) -> str:
        # Redis-backed so the limit holds across multiple API replicas.
        return self.rate_limit_storage_uri or self.redis_url


@lru_cache
def get_settings() -> Settings:
    """Cached so the env is parsed once per process."""
    return Settings()


settings = get_settings()
