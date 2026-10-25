"""Temporal evaluation with exact grid checks and a weekly seasonal baseline."""

import numpy as np
import pandas as pd

from retail_forecasting.config import ForecastConfig
from retail_forecasting.evaluation.metrics import regression_metrics
from retail_forecasting.features.specification import STATIC_FEATURES
from retail_forecasting.models.forecast import build_forecaster, postprocess


def get_cutoffs(panel: pd.DataFrame, config: ForecastConfig) -> list[pd.Timestamp]:
    required = 56 + config.horizon + (config.n_windows - 1) * config.step_size
    if panel.groupby("unique_id").size().min() < required:
        raise ValueError(f"Evaluation needs at least {required} daily rows per series")
    last = panel["ds"].max() - pd.Timedelta(days=config.horizon)
    return [
        last - pd.Timedelta(days=config.step_size * i) for i in reversed(range(config.n_windows))
    ]


def seasonal_naive(panel: pd.DataFrame, cutoffs: list, horizon: int) -> pd.DataFrame:
    rows = []
    for cutoff in cutoffs:
        for uid, group in panel[panel["ds"] <= cutoff].groupby("unique_id", sort=False):
            last_week = group.sort_values("ds")["y"].tail(7).to_numpy()
            if len(last_week) != 7:
                raise ValueError("SeasonalNaive needs seven observed days")
            for step in range(1, horizon + 1):
                rows.append(
                    (
                        uid,
                        cutoff,
                        cutoff + pd.Timedelta(days=step),
                        float(last_week[(step - 1) % 7]),
                    )
                )
    return pd.DataFrame(rows, columns=["unique_id", "cutoff", "ds", "SeasonalNaive"])


def backtest(panel: pd.DataFrame, config: ForecastConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoffs = get_cutoffs(panel, config)
    forecaster = build_forecaster(config)
    cv = forecaster.cross_validation(
        df=panel,
        n_windows=config.n_windows,
        h=config.horizon,
        step_size=config.step_size,
        static_features=STATIC_FEATURES,
        refit=config.refit,
        max_horizon=None,
    )
    cv = postprocess(cv)
    keys = ["unique_id", "cutoff", "ds"]
    baseline = seasonal_naive(panel, cutoffs, config.horizon)
    if len(cv) != len(baseline) or cv.duplicated(keys).any():
        raise ValueError("Incorrect number or uniqueness of CV predictions")
    cv = baseline.merge(cv, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not cv["_merge"].eq("both").all():
        raise ValueError("CV keys differ from the exact expected forecast grid")
    cv = cv.drop(columns="_merge")
    truth = panel[["unique_id", "ds", "y", "store", "item"]].rename(columns={"y": "truth"})
    cv = cv.merge(truth, on=["unique_id", "ds"], validate="many_to_one", how="left")
    if not np.isfinite(cv[["y", "truth", "XGBoost", "SeasonalNaive"]]).all().all():
        raise ValueError("Incomplete CV or non-finite forecasts")
    if not np.allclose(cv["y"], cv["truth"], rtol=0, atol=1e-6):
        raise ValueError("CV target values do not match source observations")
    cv = cv.drop(columns="truth").sort_values(keys).reset_index(drop=True)
    cv["h"] = (cv["ds"] - cv["cutoff"]).dt.days
    summary = pd.DataFrame(
        [
            {"model": name, **regression_metrics(cv["y"], cv[name])}
            for name in ("XGBoost", "SeasonalNaive")
        ]
    )
    return cv, summary.sort_values("sMAPE_%").reset_index(drop=True)


def segment_metrics(cv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension in ("unique_id", "store", "item", "h", "cutoff"):
        for value, group in cv.groupby(dimension, sort=True):
            for name in ("XGBoost", "SeasonalNaive"):
                rows.append(
                    {
                        "dimension": dimension,
                        "value": str(value),
                        "model": name,
                        **regression_metrics(group["y"], group[name]),
                    }
                )
    return pd.DataFrame(rows)
