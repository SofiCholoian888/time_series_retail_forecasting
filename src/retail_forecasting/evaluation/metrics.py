"""sMAPE in percent (zero/zero contributes zero), MAE and RMSE in sales units."""

import numpy as np


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    actual, predicted = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if actual.ndim != 1 or actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("Metric inputs must be nonempty 1D arrays of equal shape")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metric inputs must be finite")
    error = actual - predicted
    denominator = np.abs(actual) + np.abs(predicted)
    relative = np.divide(
        2 * np.abs(error), denominator, out=np.zeros_like(error), where=denominator != 0
    )
    return {
        "sMAPE_%": float(100 * relative.mean()),
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
    }
