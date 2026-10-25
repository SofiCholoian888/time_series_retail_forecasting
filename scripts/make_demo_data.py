"""Generate synthetic data for a technical smoke run, never for quality claims."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_demo(days: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=days)
    rows = []
    for store in (1, 2):
        for item in (1, 2):
            for i, day in enumerate(dates):
                sales = max(
                    0,
                    round(
                        15
                        + 4 * store
                        + 3 * item
                        + 0.035 * i
                        + 5 * np.sin(2 * np.pi * i / 7)
                        + rng.normal(0, 1.5)
                    ),
                )
                rows.append((day.date(), store, item, sales))
    return pd.DataFrame(rows, columns=["date", "store", "item", "sales"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/demo.csv"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_demo().to_csv(args.output, index=False)
    print(f"Synthetic data: {args.output} (4 series x 240 days)")
