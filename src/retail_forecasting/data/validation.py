"""Validate the raw daily sales contract without silently filling missing dates."""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = ["date", "store", "item", "sales"]
PANEL_COLUMNS = ["unique_id", "ds", "y", "store", "item"]


def prepare_sales(raw: pd.DataFrame, expected_series: int | None = None) -> pd.DataFrame:
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    if raw.empty:
        raise ValueError("Empty sales data")
    out = raw[RAW_COLUMNS].copy()
    if out.isna().any().any():
        raise ValueError("Null values in the input contract")
    dates = pd.to_datetime(out["date"], format="ISO8601", errors="raise")
    if not pd.api.types.is_datetime64_dtype(dates.dtype):
        raise ValueError("Dates must be timezone-free calendar dates")
    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Expected daily dates without a time component")
    out["date"] = dates
    for col in ("store", "item", "sales"):
        if pd.api.types.is_bool_dtype(out[col]):
            raise ValueError(f"Boolean {col} is not valid")
        values = pd.to_numeric(out[col], errors="raise")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite {col}")
        if col in ("store", "item"):
            if (values <= 0).any() or not values.eq(np.floor(values)).all():
                raise ValueError(f"{col} must contain positive integer IDs")
            out[col] = values.astype("int64")
        else:
            if (values < 0).any():
                raise ValueError("Negative sales")
            out[col] = values.astype("float64")
    if out.duplicated(["date", "store", "item"]).any():
        raise ValueError("Duplicate (date, store, item) observations")
    out["unique_id"] = "store_" + out["store"].astype(str) + "_item_" + out["item"].astype(str)
    out = out.rename(columns={"date": "ds", "sales": "y"})
    out = out[PANEL_COLUMNS].sort_values(["unique_id", "ds"]).reset_index(drop=True)
    count = out["unique_id"].nunique()
    if expected_series is not None and count != expected_series:
        raise ValueError(f"Expected {expected_series} series; found {count}")
    spans = out.groupby("unique_id")["ds"].agg(["min", "max", "size"])
    if spans["min"].nunique() != 1 or spans["max"].nunique() != 1:
        raise ValueError("All series must have aligned start and end dates")
    expected_days = (spans["max"] - spans["min"]).dt.days + 1
    if not spans["size"].eq(expected_days).all():
        raise ValueError("Missing daily observations; resolve gaps before training")
    return out


def read_sales(path: str | Path, expected_series: int | None = None) -> pd.DataFrame:
    return prepare_sales(pd.read_csv(path), expected_series=expected_series)
