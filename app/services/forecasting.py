"""Forecasting models.

Pure functions over a pandas Series: no Celery, no database, no filesystem.
That is what makes the statistical logic unit-testable in milliseconds, and
what lets a real model (Prophet, XGBoost) be dropped in by registering a new
class against the same protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from app.core.errors import ModelNotFoundError, PermanentJobError

Z_95 = 1.959964
MIN_FIT_ROWS = 3
WEEKLY_PERIOD = 7


@dataclass(frozen=True)
class ForecastPoint:
    date: str
    yhat: float
    yhat_lower: float
    yhat_upper: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "yhat": self.yhat,
            "yhat_lower": self.yhat_lower,
            "yhat_upper": self.yhat_upper,
        }


@dataclass(frozen=True)
class ForecastResult:
    model_name: str
    horizon: int
    trained_rows: int
    metrics: dict[str, Any]
    points: list[ForecastPoint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe payload for the `jobs.result` JSONB column."""
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "trained_rows": self.trained_rows,
            "metrics": self.metrics,
            "points": [p.to_dict() for p in self.points],
        }


@runtime_checkable
class ForecastModel(Protocol):
    name: str

    def fit_predict(self, series: pd.Series, horizon: int) -> ForecastResult: ...


def _finite(value: float) -> float | None:
    """NaN and inf are not valid JSON; JSONB would reject them."""
    return float(value) if np.isfinite(value) else None


class SeasonalNaiveTrendModel:
    """Linear trend plus additive day-of-week seasonality.

    Chosen as the baseline because every number it produces is explainable:
    a least-squares trend line, mean residual per weekday, and a prediction
    interval from the residual standard deviation. It gives the service a real
    numeric contract to test against without pretending to be a good model.
    """

    name = "seasonal_naive_trend"

    def fit_predict(self, series: pd.Series, horizon: int) -> ForecastResult:
        if horizon < 1:
            raise PermanentJobError("Forecast horizon must be at least 1.")
        if len(series) < MIN_FIT_ROWS:
            raise PermanentJobError(
                f"Need at least {MIN_FIT_ROWS} observations to fit; got {len(series)}."
            )

        y = series.to_numpy(dtype="float64")
        n = y.size
        t = np.arange(n, dtype="float64")

        slope, intercept = np.polyfit(t, y, 1)
        trend = intercept + slope * t

        # Seasonality is learned from the de-trended residual, then centred so
        # it redistributes rather than shifts the level.
        residual = y - trend
        use_weekly = n >= 2 * WEEKLY_PERIOD
        period = WEEKLY_PERIOD if use_weekly else 1
        dow = series.index.dayofweek.to_numpy() if use_weekly else np.zeros(n, dtype="int64")

        seasonal = np.zeros(period, dtype="float64")
        for phase in range(period):
            mask = dow == phase
            if mask.any():
                seasonal[phase] = residual[mask].mean()
        seasonal -= seasonal.mean()

        fitted = trend + seasonal[dow]
        errors = y - fitted

        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))
        sigma = float(np.std(errors, ddof=1)) if n > 1 else 0.0

        nonzero = y != 0
        mape = (
            float(np.mean(np.abs(errors[nonzero] / y[nonzero])) * 100.0)
            if nonzero.any()
            else float("nan")
        )

        future_t = np.arange(n, n + horizon, dtype="float64")
        future_index = pd.date_range(
            series.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D"
        )
        future_dow = (
            future_index.dayofweek.to_numpy() if use_weekly else np.zeros(horizon, dtype="int64")
        )
        yhat = intercept + slope * future_t + seasonal[future_dow]

        margin = Z_95 * sigma
        lower = yhat - margin
        upper = yhat + margin

        # Historically non-negative series (sales, units) should not be
        # forecast negative just because the interval is wide.
        if float(np.min(y)) >= 0.0:
            lower = np.maximum(lower, 0.0)
            yhat = np.maximum(yhat, 0.0)

        points = [
            ForecastPoint(
                date=stamp.date().isoformat(),
                yhat=round(float(point), 4),
                yhat_lower=round(float(low), 4),
                yhat_upper=round(float(high), 4),
            )
            for stamp, point, low, high in zip(future_index, yhat, lower, upper, strict=True)
        ]

        return ForecastResult(
            model_name=self.name,
            horizon=horizon,
            trained_rows=n,
            metrics={
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "mape": (None if (m := _finite(mape)) is None else round(m, 4)),
                "residual_std": round(sigma, 4),
                "trend_per_day": round(float(slope), 6),
                "weekly_seasonality": bool(use_weekly),
            },
            points=points,
        )


_REGISTRY: dict[str, ForecastModel] = {}


def register_model(model: ForecastModel) -> ForecastModel:
    _REGISTRY[model.name] = model
    return model


def get_model(name: str) -> ForecastModel:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ModelNotFoundError(
            f"Unknown model {name!r}. Available: {', '.join(available_models()) or 'none'}."
        ) from None


def available_models() -> list[str]:
    return sorted(_REGISTRY)


register_model(SeasonalNaiveTrendModel())
