"""Record input and code identity beside each generated artifact."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from retail_forecasting.config import ForecastConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def metadata(path: Path, panel: pd.DataFrame, config: ForecastConfig) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for source in sorted(package_root.rglob("*.py")):
        digest.update(str(source.relative_to(package_root)).encode())
        digest.update(source.read_bytes())
    packages = ["numpy", "pandas", "scikit-learn", "xgboost", "mlforecast", "coreforecast"]
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "data_file": path.name,
        "data_sha256": sha256(path),
        "source_sha256": digest.hexdigest(),
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "rows": len(panel),
        "series": int(panel["unique_id"].nunique()),
        "train_start": str(panel["ds"].min().date()),
        "data_end": str(panel["ds"].max().date()),
        "config": config.to_dict(),
        "strategy": "recursive",
        "postprocessing": "clip final predictions at 0; no rounding",
    }
