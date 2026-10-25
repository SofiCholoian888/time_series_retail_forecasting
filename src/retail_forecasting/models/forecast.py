"""One global XGBoost regressor with recursive multi-step forecasting."""

import numpy as np
import pandas as pd
from mlforecast import MLForecast
from mlforecast.target_transforms import LocalStandardScaler
from xgboost import XGBRegressor

from retail_forecasting.config import ForecastConfig
from retail_forecasting.features.specification import DATE_FEATURES, LAGS, lag_transforms


def build_forecaster(config: ForecastConfig) -> MLForecast:
    model = XGBRegressor(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=config.seed,
        n_jobs=config.threads,
    )
    return MLForecast(
        models={"XGBoost": model},
        freq="D",
        lags=LAGS,
        lag_transforms=lag_transforms(),
        date_features=DATE_FEATURES,
        target_transforms=[LocalStandardScaler()],
        num_threads=config.threads,
    )


def postprocess(predictions: pd.DataFrame) -> pd.DataFrame:
    """Clip final output in sales units; recursive internals retain the raw prediction."""
    out = predictions.copy()
    values = out["XGBoost"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Model returned non-finite predictions")
    out["XGBoost"] = np.maximum(values, 0.0)
    return out
