import numpy as np
import pytest

from retail_forecasting.config import ForecastConfig
from retail_forecasting.evaluation.metrics import regression_metrics


def test_known_metrics_and_zero_denominator():
    scores = regression_metrics([0, 1, 3], [0, 2, 1])
    assert scores["sMAPE_%"] == pytest.approx(100 * (0 + 2 / 3 + 1) / 3)
    assert scores["MAE"] == pytest.approx(1)
    assert scores["RMSE"] == pytest.approx(np.sqrt(5 / 3))


@pytest.mark.parametrize(
    "actual,pred", [([], []), ([1], [1, 2]), ([np.nan], [1]), ([1], [np.inf]), ([[1]], [[1]])]
)
def test_invalid_metrics_rejected(actual, pred):
    with pytest.raises(ValueError):
        regression_metrics(actual, pred)


@pytest.mark.parametrize(
    "settings", [{"horizon": 0}, {"refit": "false"}, {"step_size": 1}, {"threads": -1}]
)
def test_invalid_config_rejected(settings):
    with pytest.raises(ValueError):
        ForecastConfig(**settings)
