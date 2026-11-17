from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud_detection.features import FeatureBuilder, category_values

BASELINE_NUMERIC = ["TransactionAmt", "C1", "C2", "C5", "C13", "D1", "dist1"]
BASELINE_CATEGORICAL = ["ProductCD", "card4", "card6", "P_emaildomain", "DeviceType"]


def new_booster(parameters: dict, seed: int, threads: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        random_state=seed,
        n_jobs=threads,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        metric="average_precision",
        **parameters,
    )


def baseline_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.reindex(columns=BASELINE_NUMERIC + BASELINE_CATEGORICAL).copy()
    for column in BASELINE_CATEGORICAL:
        result[column] = category_values(result[column]).astype(object)
    return result


def logistic_baseline(seed: int) -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    transform = ColumnTransformer(
        [
            ("numeric", numeric, BASELINE_NUMERIC),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=10),
                BASELINE_CATEGORICAL,
            ),
        ]
    )
    return Pipeline(
        [
            ("features", transform),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=seed, solver="liblinear"),
            ),
        ]
    )


def log_odds(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


@dataclass
class FraudModel:
    features: FeatureBuilder
    classifier: lgb.LGBMClassifier
    calibrator: LogisticRegression
    threshold: float
    review_rate: float

    def raw_probability(self, frame: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(self.features.transform(frame))[:, 1]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.calibrator.predict_proba(log_odds(self.raw_probability(frame)))[:, 1]
