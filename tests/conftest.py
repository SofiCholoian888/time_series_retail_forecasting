"""Small deterministic panels keep correctness tests independent of Kaggle."""

import numpy as np
import pandas as pd
import pytest

from retail_forecasting.config import ForecastConfig
from retail_forecasting.data.validation import prepare_sales


@pytest.fixture
def raw():
    days = pd.date_range("2020-01-01", periods=180)
    return pd.DataFrame(
        [
            (day, store, 1, float(20 + 4 * store + np.sin(i * 2 * np.pi / 7) * 5 + i * 0.03))
            for store in (1, 2)
            for i, day in enumerate(days)
        ],
        columns=["date", "store", "item", "sales"],
    )


@pytest.fixture
def panel(raw):
    return prepare_sales(raw)


@pytest.fixture
def config():
    return ForecastConfig(
        horizon=14, step_size=14, n_windows=2, n_estimators=12, max_depth=3, threads=1
    )
