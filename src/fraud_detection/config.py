from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    seed: int = 42
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    output_dir: str = "artifacts/full"
    chunk_size: int = 50000
    row_limit: int | None = None
    train_fraction: float = 0.60
    tuning_fraction: float = 0.15
    calibration_fraction: float = 0.10
    review_rate: float = 0.01
    fraud_loss: float = 100.0
    review_cost: float = 1.0
    review_sensitivity: float = 0.80
    threads: int = 4
    early_stopping_rounds: int = 60
    candidates: list[dict] = field(default_factory=lambda: [{"n_estimators": 600}])
    model_candidates: dict[str, list[dict]] = field(default_factory=dict)

    def __post_init__(self):
        fractions = [self.train_fraction, self.tuning_fraction, self.calibration_fraction]
        if any(not 0 < value < 1 for value in fractions) or sum(fractions) >= 1:
            raise ValueError("Split fractions must be positive and sum to less than one")
        if not 0 < self.review_rate <= 1 or not 0 <= self.review_sensitivity <= 1:
            raise ValueError("Invalid review rate or sensitivity")
        if self.fraud_loss <= 0 or self.review_cost < 0:
            raise ValueError("Invalid business costs")
        if self.chunk_size < 1 or self.threads < 1 or not self.candidates:
            raise ValueError("Chunk size, threads and candidates must be positive")
        if self.row_limit is not None and self.row_limit < 1:
            raise ValueError("row_limit must be positive")
        if self.early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be positive")
        supported = {"logistic", "random_forest", "lightgbm", "xgboost", "catboost"}
        if set(self.model_candidates) - supported:
            raise ValueError("Unsupported model family in model_candidates")
        if any(
            not trials or any(not isinstance(p, dict) for p in trials)
            for trials in self.model_candidates.values()
        ):
            raise ValueError("Each model family requires a nonempty list of parameter dictionaries")

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path).resolve()
        values = yaml.safe_load(path.read_text()) or {}
        for key in ("raw_dir", "processed_dir", "output_dir"):
            values[key] = str((path.parent.parent / values.get(key, getattr(cls, key))).resolve())
        return cls(**values)

    def as_dict(self) -> dict:
        return asdict(self)
