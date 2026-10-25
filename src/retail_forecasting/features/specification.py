"""The lag specification is shared by training and prediction."""

from mlforecast.lag_transforms import RollingMean

LAGS = [1, 7, 14, 28]
DATE_FEATURES = ["dayofweek", "month"]
STATIC_FEATURES = ["store", "item"]


def lag_transforms() -> dict:
    """Rolling windows follow a lag; the current target is excluded."""
    return {
        1: [RollingMean(window_size=7), RollingMean(window_size=28)],
        7: [RollingMean(window_size=7)],
    }
