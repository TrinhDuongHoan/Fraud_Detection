import json
import logging
from dataclasses import asdict
from pathlib import Path

import joblib
import pandas as pd
import pyarrow.parquet as pq

from fraud_detection.config import Config
from fraud_detection.data import ID, TARGET, TIME, csv_dtypes, validate_transactions
from fraud_detection.monitoring import drift_report
from fraud_detection.policy import ReviewState, online_review
from fraud_detection.reporting import save_json

LOGGER = logging.getLogger(__name__)


def predict(
    config: Config,
    input_path: Path | None = None,
    state_path: Path | None = None,
) -> pd.DataFrame:
    output = Path(config.output_dir)
    artifact = joblib.load(output / "model.joblib")
    source = input_path or Path(config.processed_dir) / "test.parquet"
    if source.suffix == ".parquet":
        batches = (
            batch.to_pandas()
            for batch in pq.ParquetFile(source).iter_batches(
                batch_size=config.chunk_size,
            )
        )
    elif source.suffix == ".csv":
        batches = pd.read_csv(source, dtype=csv_dtypes(source), chunksize=config.chunk_size)
    else:
        raise ValueError("Prediction input must be a joined .parquet or .csv table")
    predictions, monitoring_frames = [], []
    reference = json.loads((output / "reference.json").read_text())
    monitor_columns = [c for c in reference["features"] if c != "risk_score"]
    for batch in batches:
        validate_transactions(batch, labeled=False)
        scores = artifact.predict_proba(batch)
        result = batch[[ID, TIME]].copy()
        result["risk_score"] = scores
        predictions.append(result)
        monitoring_frames.append(batch[monitor_columns].copy())
        LOGGER.info("Scored %s transactions", len(batch))
    if not predictions:
        raise ValueError("Prediction input is empty")
    predictions = pd.concat(predictions, ignore_index=True)
    if predictions[ID].duplicated().any():
        raise ValueError("Duplicate TransactionID across prediction batches")
    state = ReviewState()
    if state_path and state_path.exists():
        state = ReviewState(**json.loads(state_path.read_text()))
    predictions["manual_review"] = online_review(
        predictions,
        predictions.risk_score.to_numpy(),
        artifact.threshold,
        artifact.review_rate,
        state,
    )
    predictions["action"] = predictions.manual_review.map({True: "manual_review", False: "approve"})
    submission = predictions[[ID, "risk_score"]].rename(columns={"risk_score": TARGET})
    if input_path is None:
        sample_path = Path(config.raw_dir) / "sample_submission.csv"
        if sample_path.exists():
            sample = pd.read_csv(sample_path, usecols=[ID])
            if config.row_limit is not None:
                sample = sample.loc[sample[ID].isin(submission[ID])]
            if set(sample[ID]) != set(submission[ID]) or sample[ID].duplicated().any():
                raise ValueError("Prediction IDs do not match sample_submission")
            submission = sample.merge(submission, on=ID, validate="one_to_one", sort=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    submission.to_csv(output / "submission.csv", index=False)
    monitoring = pd.concat(monitoring_frames, ignore_index=True)
    drift_report(monitoring, predictions.risk_score.to_numpy(), reference).to_csv(
        output / "drift_test.csv",
        index=False,
    )
    if state_path:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = state_path.with_suffix(state_path.suffix + ".tmp")
        save_json(temporary, asdict(state))
        temporary.replace(state_path)
    LOGGER.info("Saved %s predictions and submission to %s", len(predictions), output)
    return predictions
