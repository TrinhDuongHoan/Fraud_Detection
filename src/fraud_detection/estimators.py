from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from fraud_detection.features import FeatureBuilder
from fraud_detection.modeling import log_odds, new_booster

ENCODING = {
    "logistic": "median imputation, scaling and sparse one-hot encoding",
    "random_forest": "training-vocabulary ordinal codes with native numeric missing values",
    "lightgbm": "native categorical splits and numeric missing values",
    "xgboost": "native categorical histogram splits and numeric missing values",
    "catboost": "native categorical features with explicit missing-category tokens",
}


class TabularEstimator:
    def __init__(self, name: str, parameters: dict, seed: int, threads: int):
        self.name = name
        self.parameters = dict(parameters)
        self.seed = seed
        self.threads = threads

    def _input(self, features: pd.DataFrame):
        if self.name == "logistic":
            return self.preprocessor_.transform(features)
        if self.name == "random_forest":
            values = {
                column: features[column].cat.codes.astype("float32")
                if isinstance(features[column].dtype, pd.CategoricalDtype)
                else features[column]
                for column in features
            }
            return pd.DataFrame(values, index=features.index, copy=False)
        if self.name == "catboost":
            result = features.copy(deep=False)
            for column in self.categorical_:
                result[column] = (
                    features[column].astype("string").fillna("__UNKNOWN__").astype(object)
                )
            return result
        return features

    def fit(self, features, labels, validation=None, patience=40):
        self.columns_ = features.columns.tolist()
        self.categorical_ = [
            column for column in features if isinstance(features[column].dtype, pd.CategoricalDtype)
        ]
        self.classifier_ = self._classifier(validation is not None, patience)
        if self.name == "logistic":
            numeric = [column for column in features if column not in self.categorical_]
            self.preprocessor_ = ColumnTransformer(
                [
                    (
                        "numeric",
                        Pipeline(
                            [
                                (
                                    "imputer",
                                    SimpleImputer(strategy="median", keep_empty_features=True),
                                ),
                                ("scale", StandardScaler(with_mean=False)),
                            ]
                        ),
                        numeric,
                    ),
                    (
                        "categorical",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            min_frequency=20,
                            max_categories=128,
                            dtype=np.float32,
                        ),
                        self.categorical_,
                    ),
                ],
                sparse_threshold=1.0,
            )
            train_x = self.preprocessor_.fit_transform(features)
        else:
            train_x = self._input(features)
        fit_options = {}
        if validation is not None and self.name in {"lightgbm", "xgboost", "catboost"}:
            valid_x, valid_y = validation
            valid_x = self._input(valid_x)
            if self.name == "lightgbm":
                fit_options = {
                    "eval_X": valid_x,
                    "eval_y": valid_y,
                    "callbacks": [
                        lgb.early_stopping(patience, first_metric_only=True, verbose=False)
                    ],
                }
            elif self.name in {"xgboost", "catboost"}:
                fit_options = {"eval_set": [(valid_x, valid_y)], "verbose": False}
                if self.name == "catboost":
                    fit_options["early_stopping_rounds"] = patience
        self.classifier_.fit(train_x, labels, **fit_options)
        return self

    def _classifier(self, tuning: bool, patience: int):
        parameters = dict(self.parameters)
        if self.name == "logistic":
            return LogisticRegression(solver="liblinear", random_state=self.seed, **parameters)
        if self.name == "random_forest":
            return RandomForestClassifier(
                random_state=self.seed,
                n_jobs=self.threads,
                **parameters,
            )
        if self.name == "lightgbm":
            return new_booster(parameters, self.seed, self.threads)
        if self.name == "xgboost":
            return XGBClassifier(
                objective="binary:logistic",
                tree_method="hist",
                device="cpu",
                enable_categorical=True,
                eval_metric="aucpr",
                max_bin=128,
                random_state=self.seed,
                n_jobs=self.threads,
                early_stopping_rounds=patience if tuning else None,
                **parameters,
            )
        if self.name == "catboost":
            return CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="PRAUC",
                cat_features=self.categorical_,
                random_seed=self.seed,
                thread_count=self.threads,
                verbose=False,
                allow_writing_files=False,
                border_count=128,
                max_ctr_complexity=1,
                **parameters,
            )
        raise ValueError(f"Unknown model family: {self.name}")

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if features.columns.tolist() != self.columns_:
            raise ValueError("Feature columns or column order do not match the fitted estimator")
        return self.classifier_.predict_proba(self._input(features))

    def refit_parameters(self) -> dict:
        parameters = dict(self.parameters)
        if self.name == "lightgbm":
            parameters["n_estimators"] = max(1, self.classifier_.best_iteration_)
        elif self.name == "xgboost":
            parameters["n_estimators"] = self.classifier_.best_iteration + 1
        elif self.name == "catboost":
            parameters["iterations"] = self.classifier_.tree_count_
        return parameters

    def feature_importance(self) -> pd.DataFrame:
        if self.name == "logistic":
            names = self.preprocessor_.get_feature_names_out()
            values = np.abs(self.classifier_.coef_[0])
            method = "absolute standardized coefficient"
        elif self.name == "lightgbm":
            names, values = self.columns_, self.classifier_.booster_.feature_importance("gain")
            method = "split gain"
        else:
            names, values = self.columns_, self.classifier_.feature_importances_
            method = {
                "random_forest": "impurity reduction",
                "xgboost": "gain",
                "catboost": "prediction values change",
            }[self.name]
        return pd.DataFrame({"feature": names, "importance": values, "method": method}).sort_values(
            "importance",
            ascending=False,
        )


@dataclass
class CalibratedModel:
    features: FeatureBuilder
    estimator: TabularEstimator
    calibrator: LogisticRegression
    threshold: float
    review_rate: float

    def raw_probability(self, frame: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(self.features.transform(frame))[:, 1]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.calibrator.predict_proba(log_odds(self.raw_probability(frame)))[:, 1]
