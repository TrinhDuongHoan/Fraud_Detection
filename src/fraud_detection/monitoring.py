import numpy as np
import pandas as pd

NUMERIC = ["TransactionAmt", "C1", "C13", "D1"]
CATEGORICAL = ["ProductCD", "card4", "DeviceType", "P_emaildomain"]


def proportions(frame: pd.DataFrame, column: str, spec: dict) -> np.ndarray:
    values = frame[column]
    if spec["kind"] == "numeric":
        numeric = pd.to_numeric(values, errors="coerce")
        counts, _ = np.histogram(numeric.dropna(), bins=[-np.inf, *spec["cuts"], np.inf])
        counts = np.r_[counts, numeric.isna().sum()]
    else:
        values = values.astype("string").fillna("__MISSING__")
        counts = np.array([(values == item).sum() for item in spec["categories"]])
        counts = np.r_[counts, (~values.isin(spec["categories"])).sum()]
    return counts / max(len(frame), 1)


def reference_profile(frame: pd.DataFrame, scores: np.ndarray) -> dict:
    frame = frame.assign(risk_score=scores)
    profile = {"rows": len(frame), "features": {}}
    for column in [*NUMERIC, *CATEGORICAL, "risk_score"]:
        if column not in frame:
            continue
        values = frame[column]
        if column in NUMERIC or column == "risk_score":
            cuts = np.unique(values.dropna().quantile(np.linspace(0.1, 0.9, 9))).tolist()
            spec = {"kind": "numeric", "cuts": cuts}
        else:
            categories = values.astype("string").fillna("__MISSING__").value_counts().head(20)
            spec = {"kind": "categorical", "categories": categories.index.tolist()}
        spec["proportions"] = proportions(frame, column, spec).tolist()
        spec["missing_rate"] = float(values.isna().mean())
        profile["features"][column] = spec
    return profile


def drift_report(frame: pd.DataFrame, scores: np.ndarray, reference: dict) -> pd.DataFrame:
    frame = frame.assign(risk_score=scores)
    rows = []
    for column, spec in reference["features"].items():
        actual = proportions(frame, column, spec)
        expected = np.asarray(spec["proportions"])
        actual, expected = np.maximum(actual, 1e-6), np.maximum(expected, 1e-6)
        actual, expected = actual / actual.sum(), expected / expected.sum()
        psi = float(np.sum((actual - expected) * np.log(actual / expected)))
        rows.append(
            {
                "feature": column,
                "psi": psi,
                "reference_missing_rate": spec["missing_rate"],
                "current_missing_rate": float(frame[column].isna().mean()),
                "investigate": psi >= 0.2,
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False)
