import numpy as np
import pandas as pd

from fraud_detection.config import Config
from fraud_detection.data import TIME


def chronological_split(frame: pd.DataFrame, config: Config) -> dict[str, np.ndarray]:
    days = (frame[TIME] // 86400).astype("int64")
    counts = days.value_counts().sort_index()
    if len(counts) < 8:
        raise ValueError("At least eight distinct days are required for four temporal partitions")
    cumulative = counts.cumsum().to_numpy() / len(frame)
    fractions = np.cumsum(
        [
            config.train_fraction,
            config.tuning_fraction,
            config.calibration_fraction,
        ]
    )
    boundaries = []
    for i, fraction in enumerate(fractions):
        index = int(np.searchsorted(cumulative, fraction))
        index = max(index, boundaries[-1] + 1 if boundaries else 0)
        index = min(index, len(counts) - (4 - i))
        boundaries.append(index)
    cutoffs = counts.index.to_numpy()[boundaries]
    groups = np.searchsorted(cutoffs, days.to_numpy(), side="left")
    partitions = {}
    for i, name in enumerate(("train", "tuning", "calibration", "holdout")):
        indices = np.flatnonzero(groups == i)
        if not len(indices) or frame.iloc[indices]["isFraud"].nunique() != 2:
            raise ValueError(f"{name} must be nonempty and contain both classes")
        partitions[name] = indices
    return partitions
