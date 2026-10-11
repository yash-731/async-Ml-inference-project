"""Unit tests for the forecasting logic.

These are the fast tests: pure functions, no database, no broker, no network.
"""

from __future__ import annotations

import json
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from app.core.errors import ModelNotFoundError, PermanentJobError
from app.services.forecasting import (
    ForecastResult,
    SeasonalNaiveTrendModel,
    available_models,
    get_model,
    register_model,
)


def daily_series(values: list[float] | np.ndarray, start: str = "2025-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(np.asarray(values, dtype="float64"), index=index)


@pytest.fixture
def model() -> SeasonalNaiveTrendModel:
    return SeasonalNaiveTrendModel()


def test_recovers_a_clean_linear_trend(model: SeasonalNaiveTrendModel) -> None:
    series = daily_series([10.0 + 2.0 * i for i in range(60)])

    result = model.fit_predict(series, horizon=5)

    assert result.metrics["trend_per_day"] == pytest.approx(2.0, abs=1e-6)
    assert result.metrics["mae"] == pytest.approx(0.0, abs=1e-6)
    # Next point continues the line: last observed value was 10 + 2*59 = 128.
    assert result.points[0].yhat == pytest.approx(130.0, abs=1e-3)
    assert result.points[-1].yhat == pytest.approx(138.0, abs=1e-3)


def test_learns_weekly_seasonality(model: SeasonalNaiveTrendModel) -> None:
    index = pd.date_range("2025-01-06", periods=70, freq="D")  # starts Monday
    values = 100.0 + np.where(index.dayofweek >= 5, 40.0, 0.0)
    series = pd.Series(values, index=index)

    result = model.fit_predict(series, horizon=7)

    assert result.metrics["weekly_seasonality"] is True
    by_date = {p.date: p.yhat for p in result.points}
    weekend = [v for d, v in by_date.items() if pd.Timestamp(d).dayofweek >= 5]
    weekday = [v for d, v in by_date.items() if pd.Timestamp(d).dayofweek < 5]
    assert min(weekend) > max(weekday) + 30.0


def test_short_series_skips_weekly_seasonality(model: SeasonalNaiveTrendModel) -> None:
    # Fewer than two full weeks: estimating seven weekday effects from this
    # would be fitting noise.
    series = daily_series([5.0, 7.0, 6.0, 8.0, 9.0, 7.0, 10.0, 11.0])

    result = model.fit_predict(series, horizon=3)

    assert result.metrics["weekly_seasonality"] is False


def test_forecast_dates_are_consecutive_and_start_after_history(
    model: SeasonalNaiveTrendModel,
) -> None:
    series = daily_series([float(i) for i in range(30)], start="2025-03-01")

    result = model.fit_predict(series, horizon=10)

    dates = [pd.Timestamp(p.date) for p in result.points]
    assert len(dates) == 10
    assert dates[0] == pd.Timestamp("2025-03-31")  # history ends 2025-03-30
    assert all((b - a).days == 1 for a, b in pairwise(dates))


def test_constant_series_gives_a_flat_forecast_with_no_interval(
    model: SeasonalNaiveTrendModel,
) -> None:
    series = daily_series([42.0] * 40)

    result = model.fit_predict(series, horizon=4)

    assert result.metrics["residual_std"] == pytest.approx(0.0, abs=1e-9)
    for point in result.points:
        assert point.yhat == pytest.approx(42.0, abs=1e-6)
        assert point.yhat_lower == pytest.approx(point.yhat, abs=1e-6)
        assert point.yhat_upper == pytest.approx(point.yhat, abs=1e-6)


def test_non_negative_history_is_never_forecast_negative(
    model: SeasonalNaiveTrendModel,
) -> None:
    # A steep decline would run the trend line below zero within the horizon.
    series = daily_series([100.0 - 3.0 * i for i in range(30)])

    result = model.fit_predict(series, horizon=30)

    assert min(p.yhat for p in result.points) >= 0.0
    assert min(p.yhat_lower for p in result.points) >= 0.0


def test_negative_history_keeps_negative_forecasts(model: SeasonalNaiveTrendModel) -> None:
    # Clamping is only valid for series that were never negative to begin with.
    series = daily_series([-10.0 - i for i in range(30)])

    result = model.fit_predict(series, horizon=5)

    assert all(p.yhat < 0 for p in result.points)


def test_all_zero_series_reports_mape_as_null(model: SeasonalNaiveTrendModel) -> None:
    # MAPE divides by the actual value; NaN would be invalid JSON for JSONB.
    series = daily_series([0.0] * 30)

    result = model.fit_predict(series, horizon=3)

    assert result.metrics["mape"] is None
    json.dumps(result.to_dict())


def test_result_dict_is_json_serializable(model: SeasonalNaiveTrendModel) -> None:
    series = daily_series(np.linspace(50, 80, 45))

    payload = model.fit_predict(series, horizon=7).to_dict()
    round_tripped = json.loads(json.dumps(payload))

    assert round_tripped["model_name"] == "seasonal_naive_trend"
    assert round_tripped["horizon"] == 7
    assert round_tripped["trained_rows"] == 45
    assert len(round_tripped["points"]) == 7
    assert set(round_tripped["points"][0]) == {"date", "yhat", "yhat_lower", "yhat_upper"}


@pytest.mark.parametrize("horizon", [0, -1])
def test_rejects_non_positive_horizon(model: SeasonalNaiveTrendModel, horizon: int) -> None:
    with pytest.raises(PermanentJobError, match="horizon"):
        model.fit_predict(daily_series([1.0] * 20), horizon=horizon)


def test_rejects_series_too_short_to_fit(model: SeasonalNaiveTrendModel) -> None:
    with pytest.raises(PermanentJobError, match="at least"):
        model.fit_predict(daily_series([1.0, 2.0]), horizon=3)


def test_prediction_interval_widens_with_noise(model: SeasonalNaiveTrendModel) -> None:
    rng = np.random.default_rng(0)
    base = np.linspace(100, 150, 60)
    quiet = model.fit_predict(daily_series(base + rng.normal(0, 1, 60)), horizon=5)
    noisy = model.fit_predict(daily_series(base + rng.normal(0, 20, 60)), horizon=5)

    quiet_width = quiet.points[0].yhat_upper - quiet.points[0].yhat_lower
    noisy_width = noisy.points[0].yhat_upper - noisy.points[0].yhat_lower
    assert noisy_width > quiet_width


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_default_model_is_registered() -> None:
    assert "seasonal_naive_trend" in available_models()
    assert get_model("seasonal_naive_trend").name == "seasonal_naive_trend"


def test_unknown_model_raises_a_permanent_error() -> None:
    # Permanent, not transient: no amount of retrying will register the model.
    with pytest.raises(ModelNotFoundError, match="Unknown model"):
        get_model("prophet_v9")


def test_a_new_model_can_be_plugged_in() -> None:
    class FlatModel:
        name = "test_flat"

        def fit_predict(self, series: pd.Series, horizon: int) -> ForecastResult:
            return ForecastResult(
                model_name=self.name,
                horizon=horizon,
                trained_rows=len(series),
                metrics={"mae": 0.0},
                points=[],
            )

    register_model(FlatModel())
    try:
        assert get_model("test_flat").name == "test_flat"
        assert "test_flat" in available_models()
    finally:
        from app.services import forecasting

        forecasting._REGISTRY.pop("test_flat", None)
