"""Command-line interface with nonzero status on validation or training errors."""

import argparse
import logging
from pathlib import Path

from retail_forecasting.config import ForecastConfig
from retail_forecasting.pipeline import evaluate, predict, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    for name in ("evaluate", "train"):
        sub = actions.add_parser(name)
        sub.add_argument("--data", required=True, type=Path)
        sub.add_argument("--config", type=Path, default=None)
        sub.add_argument("--expected-series", type=int)
        if name == "evaluate":
            sub.add_argument("--output", required=True, type=Path)
        else:
            sub.add_argument("--model-dir", required=True, type=Path)
    sub = actions.add_parser("predict")
    sub.add_argument("--model-dir", required=True, type=Path)
    sub.add_argument("--output", required=True, type=Path)
    sub.add_argument("--horizon", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.command == "predict":
            forecast = predict(args.model_dir, args.horizon)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            forecast.to_csv(args.output, index=False)
            logging.info("Saved %s forecast rows to %s", len(forecast), args.output)
        else:
            config = ForecastConfig.load(args.config)
            if args.command == "evaluate":
                report = evaluate(args.data, args.output, config, args.expected_series)
                logging.info("Metrics: %s", report["metrics"])
            else:
                train(args.data, args.model_dir, config, args.expected_series)
                logging.info("Saved model to %s (no automatic deployment)", args.model_dir)
    except (ValueError, OSError, TypeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
