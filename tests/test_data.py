import numpy as np
import pandas as pd
import pytest

from retail_forecasting.data.validation import prepare_sales


def test_valid_shuffled_input(raw):
    out = prepare_sales(raw.sample(frac=1, random_state=42), expected_series=2)
    assert out["unique_id"].nunique() == 2
    assert len(out) == 360
    assert out.groupby("unique_id")["ds"].apply(lambda s: s.is_monotonic_increasing).all()


@pytest.mark.parametrize(
    "column,value",
    [
        ("sales", -1),
        ("sales", np.nan),
        ("sales", np.inf),
        ("store", 1.5),
        ("item", 0),
        ("date", "not-a-date"),
        ("date", "2020-01-01 12:30:00"),
    ],
)
def test_invalid_values_rejected(raw, column, value):
    raw[column] = raw[column].astype(object)
    raw.loc[0, column] = value
    with pytest.raises((ValueError, TypeError)):
        prepare_sales(raw)


def test_duplicate_rejected(raw):
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_sales(pd.concat([raw, raw.iloc[[0]]]))


def test_missing_day_rejected(raw):
    with pytest.raises(ValueError, match="Missing daily"):
        prepare_sales(raw.drop(index=15))


def test_misaligned_series_rejected(raw):
    with pytest.raises(ValueError, match="aligned"):
        prepare_sales(raw.drop(index=0))


def test_missing_column_rejected(raw):
    with pytest.raises(ValueError, match="Missing input"):
        prepare_sales(raw.drop(columns="sales"))


def test_missing_series_rejected(raw):
    with pytest.raises(ValueError, match="Expected 500"):
        prepare_sales(raw, expected_series=500)
