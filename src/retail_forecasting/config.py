"""Explicit, serializable experiment configuration."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ForecastConfig:
    horizon: int = 28
    n_windows: int = 3
    step_size: int = 28
    refit: bool = False
    seed: int = 42
    n_estimators: int = 350
    learning_rate: float = 0.05
    max_depth: int = 6
    threads: int = 2

    def __post_init__(self) -> None:
        for name in ("horizon", "n_windows", "step_size", "n_estimators", "max_depth", "threads"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.refit) is not bool:
            raise ValueError("refit must be a JSON boolean")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if self.step_size < self.horizon:
            raise ValueError("Use non-overlapping evaluation windows: step_size >= horizon")

    @classmethod
    def load(cls, path: str | Path | None) -> "ForecastConfig":
        return cls(**json.loads(Path(path).read_text())) if path else cls()

    def to_dict(self) -> dict:
        return asdict(self)
