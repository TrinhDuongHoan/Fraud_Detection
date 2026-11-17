import numpy as np
import pandas as pd

from fraud_detection.data import ID, TARGET, TEXT_COLUMNS, TIME

CATEGORICAL_CODES = {
    *(f"card{i}" for i in [1, 2, 3, 5]),
    "addr1",
    "addr2",
    *(f"id_{i:02d}" for i in [13, 14, 17, 18, 19, 20, 21, 22, 24, 25, 26, 32]),
}
FREQUENCY_COLUMNS = ["card1", "card2", "addr1", "P_emaildomain", "DeviceInfo"]


def category_values(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64").astype("string").fillna("__MISSING__")
    return series.astype("string").str.strip().str.lower().fillna("__MISSING__")


def engineer(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.drop(columns=[ID, TARGET, TIME], errors="ignore").copy()
    result["missing_count"] = result.isna().sum(axis=1).astype("float32")
    result["relative_hour"] = ((frame[TIME] // 3600) % 24).astype("float32")
    result["relative_weekday"] = ((frame[TIME] // 86400) % 7).astype("float32")
    result["amount_log1p"] = np.log1p(frame["TransactionAmt"]).astype("float32")
    result["amount_cents"] = (frame["TransactionAmt"] * 100 % 100).round().astype("float32")
    if {"P_emaildomain", "R_emaildomain"} <= set(frame.columns):
        result["email_match"] = (
            frame["P_emaildomain"].notna()
            & frame["R_emaildomain"].notna()
            & (frame["P_emaildomain"] == frame["R_emaildomain"])
        ).astype("float32")
    return result


class FeatureBuilder:
    def fit(self, frame: pd.DataFrame) -> "FeatureBuilder":
        self.raw_columns_ = [c for c in frame.columns if c not in {ID, TARGET, TIME}]
        features = engineer(frame)
        self.columns_ = [c for c in features if features[c].nunique(dropna=False) > 1]
        self.categories_ = {}
        self.frequencies_ = {}
        for column in self.columns_:
            if column in TEXT_COLUMNS | CATEGORICAL_CODES:
                values = category_values(features[column])
                self.categories_[column] = sorted(values.unique().tolist())
                if column in FREQUENCY_COLUMNS:
                    self.frequencies_[column] = values.value_counts(normalize=True).to_dict()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.raw_columns_) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing model input columns: {sorted(missing)}")
        result = engineer(frame[[ID, TIME, *self.raw_columns_]])
        encoded, frequencies = {}, {}
        for column in self.columns_:
            if column in self.categories_:
                values = category_values(result[column])
                known = values.where(values.isin(self.categories_[column]))
                encoded[column] = pd.Categorical(known, categories=self.categories_[column])
                if column in self.frequencies_:
                    frequencies[f"{column}_frequency"] = (
                        values.map(self.frequencies_[column]).fillna(0).astype("float32")
                    )
            else:
                encoded[column] = (
                    pd.to_numeric(result[column], errors="raise")
                    .replace([np.inf, -np.inf], np.nan)
                    .astype("float32")
                )
        return pd.DataFrame({**encoded, **frequencies}, index=frame.index, copy=False)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)
