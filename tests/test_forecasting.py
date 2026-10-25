import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from retail_forecasting.evaluation.backtest import backtest, get_cutoffs, seasonal_naive
from retail_forecasting.features.specification import STATIC_FEATURES
from retail_forecasting.models.forecast import build_forecaster, postprocess
from retail_forecasting.pipeline import evaluate, predict, train


def test_seasonal_naive_repeats_observed_week(panel):
    cutoff = panel["ds"].min() + pd.Timedelta(days=40)
    result = seasonal_naive(panel, [cutoff], 28)
    group = result[result["unique_id"].eq("store_1_item_1")]
    np.testing.assert_array_equal(
        group["SeasonalNaive"].iloc[:7], group["SeasonalNaive"].iloc[7:14]
    )
    changed = panel.copy()
    changed.loc[changed["ds"] > cutoff, "y"] = 999999
    pd.testing.assert_frame_equal(result, seasonal_naive(changed, [cutoff], 28))


def test_future_targets_do_not_change_first_cv_forecast(panel, config):
    first = get_cutoffs(panel, config)[0]
    original, _ = backtest(panel, config)
    changed = panel.copy()
    changed.loc[changed["ds"] > first, "y"] += 10000
    revised, _ = backtest(changed, config)
    np.testing.assert_allclose(
        original.loc[original["cutoff"].eq(first), "XGBoost"],
        revised.loc[revised["cutoff"].eq(first), "XGBoost"],
        rtol=0,
        atol=1e-8,
    )


def test_saved_forecast_equals_in_memory_forecast(raw, panel, config, tmp_path):
    source = tmp_path / "train.csv"
    raw.to_csv(source, index=False)
    model_dir = tmp_path / "version1"
    train(source, model_dir, config)
    result = predict(model_dir)
    direct = build_forecaster(config)
    direct.fit(panel, static_features=STATIC_FEATURES, max_horizon=None)
    expected = postprocess(direct.predict(config.horizon)).sort_values(["unique_id", "ds"])
    np.testing.assert_allclose(result["forecast"], expected["XGBoost"], rtol=1e-6)
    assert len(result) == 2 * config.horizon
    assert result["ds"].min() == raw["date"].max() + pd.Timedelta(days=1)
    assert result["forecast"].ge(0).all()
    with pytest.raises(FileExistsError):
        train(source, model_dir, config)
    with pytest.raises(ValueError, match="horizon"):
        predict(model_dir, config.horizon + 1)
    (model_dir / "series.csv").write_text("tampered\n")
    with pytest.raises(ValueError, match="checksum"):
        predict(model_dir)


def test_evaluation_writes_actual_predictions(raw, config, tmp_path):
    source = tmp_path / "train.csv"
    raw.to_csv(source, index=False)
    report = evaluate(source, tmp_path / "results", config)
    saved = json.loads((tmp_path / "results/evaluation.json").read_text())
    assert saved["forecast_points_per_model"] == 2 * config.horizon * config.n_windows
    assert len(report["metrics"]) == 2
    assert (tmp_path / "results/segment_metrics.csv").is_file()


def test_cli_fails_on_invalid_input(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("date,sales\n2020-01-01,10\n")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "retail_forecasting.cli",
            "evaluate",
            "--data",
            str(bad),
            "--output",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Missing input" in result.stderr
    assert not (tmp_path / "out/metrics.csv").exists()
