"""Reusable evaluation, fit/save, and load/predict operations."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from mlforecast import MLForecast

from retail_forecasting.config import ForecastConfig
from retail_forecasting.data.validation import read_sales
from retail_forecasting.evaluation.backtest import backtest, segment_metrics
from retail_forecasting.features.specification import STATIC_FEATURES
from retail_forecasting.models.forecast import build_forecaster, postprocess
from retail_forecasting.utils.metadata import metadata, sha256, write_json


def evaluate(data: Path, output: Path, config: ForecastConfig, expected_series=None) -> dict:
    panel = read_sales(data, expected_series)
    cv, summary = backtest(panel, config)
    output.mkdir(parents=True, exist_ok=True)
    cv.to_csv(output / "cv_predictions.csv", index=False)
    summary.to_csv(output / "metrics.csv", index=False)
    segment_metrics(cv).to_csv(output / "segment_metrics.csv", index=False)
    report = metadata(data, panel, config)
    report.update(
        {
            "status": "EVALUATED_NOT_DEPLOYED",
            "forecast_points_per_model": len(cv),
            "cutoffs": sorted(cv["cutoff"].dt.strftime("%Y-%m-%d").unique().tolist()),
            "metrics": summary.to_dict(orient="records"),
        }
    )
    write_json(output / "evaluation.json", report)
    return report


def train(data: Path, model_dir: Path, config: ForecastConfig, expected_series=None) -> dict:
    if model_dir.exists():
        raise FileExistsError(f"Use a new model version directory: {model_dir}")
    panel = read_sales(data, expected_series)
    if panel.groupby("unique_id").size().min() < 56:
        raise ValueError("Training requires at least 56 daily observations per series")
    forecaster = build_forecaster(config)
    forecaster.fit(panel, static_features=STATIC_FEATURES, max_horizon=None)
    preview = postprocess(forecaster.predict(config.horizon))
    if len(preview) != config.horizon * panel["unique_id"].nunique():
        raise ValueError("Incomplete training smoke prediction")
    model_dir.mkdir(parents=True, exist_ok=False)
    forecaster.save(model_dir / "forecaster")
    panel[["unique_id", "store", "item"]].drop_duplicates().to_csv(
        model_dir / "series.csv", index=False
    )
    report = metadata(data, panel, config)
    report["status"] = "SAVED_NOT_DEPLOYED"
    report["artifact_sha256"] = {
        str(p.relative_to(model_dir)): sha256(p)
        for p in sorted(model_dir.rglob("*"))
        if p.is_file()
    }
    write_json(model_dir / "metadata.json", report)
    return report


def predict(model_dir: Path, horizon: int | None = None) -> pd.DataFrame:
    """Load trusted local artifacts only (MLForecast uses Python serialization)."""
    info = json.loads((model_dir / "metadata.json").read_text())
    h = info["config"]["horizon"] if horizon is None else horizon
    if type(h) is not int or not 1 <= h <= info["config"]["horizon"]:
        raise ValueError("Requested horizon must be between 1 and the evaluated horizon")
    for relative, digest in info["artifact_sha256"].items():
        if sha256(model_dir / relative) != digest:
            raise ValueError(f"Artifact checksum mismatch: {relative}")
    forecaster = MLForecast.load(model_dir / "forecaster")
    result = postprocess(forecaster.predict(h)).rename(columns={"XGBoost": "forecast"})
    result = result.merge(
        pd.read_csv(model_dir / "series.csv"), on="unique_id", how="left", validate="many_to_one"
    )
    if len(result) != h * info["series"] or result.duplicated(["unique_id", "ds"]).any():
        raise ValueError("Invalid forecast dimensions")
    if result.isna().any().any() or not np.isfinite(result["forecast"]).all():
        raise ValueError("Invalid forecast output")
    expected = pd.date_range(pd.Timestamp(info["data_end"]) + pd.Timedelta(days=1), periods=h)
    for _, group in result.groupby("unique_id"):
        if not pd.DatetimeIndex(group["ds"].sort_values()).equals(expected):
            raise ValueError("Invalid forecast dates")
    return (
        result[["unique_id", "ds", "store", "item", "forecast"]]
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )
