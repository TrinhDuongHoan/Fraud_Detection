import json
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import Config
from fraud_detection.data import prepare
from fraud_detection.inference import predict
from fraud_detection.pipeline import train


def test_end_to_end_and_persistence(tmp_path, transactions):
    raw, processed, output = [tmp_path / name for name in ["raw", "processed", "output"]]
    raw.mkdir()
    train_frame = transactions.copy()
    test_frame = transactions.drop(columns="isFraud").iloc[:300].copy()
    test_frame.TransactionID += 10000
    test_frame.TransactionDT += 40 * 86400
    train_frame.to_csv(raw / "train_transaction.csv", index=False)
    test_frame.to_csv(raw / "test_transaction.csv", index=False)
    pd.DataFrame({"TransactionID": [1, 3], "id_01": [0, 1]}).to_csv(
        raw / "train_identity.csv",
        index=False,
    )
    pd.DataFrame({"TransactionID": [10001], "id-01": [4]}).to_csv(
        raw / "test_identity.csv",
        index=False,
    )
    test_frame[["TransactionID"]].iloc[::-1].assign(isFraud=0.5).to_csv(
        raw / "sample_submission.csv",
        index=False,
    )
    config = Config(
        raw_dir=str(raw),
        processed_dir=str(processed),
        output_dir=str(output),
        chunk_size=77,
        candidates=[{"n_estimators": 8, "num_leaves": 7}],
        early_stopping_rounds=3,
        threads=1,
    )
    prepare(config)
    metrics = train(config)
    assert 0 <= metrics["lightgbm_calibrated"]["average_precision"] <= 1
    predictions = predict(config, state_path=output / "state.json")
    submission = pd.read_csv(output / "submission.csv")
    assert submission.columns.tolist() == ["TransactionID", "isFraud"]
    assert submission.TransactionID.tolist() == test_frame.TransactionID.iloc[::-1].tolist()
    assert submission.isFraud.between(0, 1).all()
    saved = joblib.load(output / "model.joblib")
    test_data = pd.read_parquet(processed / "test.parquet")
    np.testing.assert_allclose(saved.predict_proba(test_data), predictions.risk_score)
    assert json.loads((output / "state.json").read_text())["day"] == 42
    with pytest.raises(ValueError, match="chronological"):
        predict(config, state_path=output / "state.json")
    manifest = json.loads((processed / "manifest.json").read_text())
    assert manifest["datasets"]["train"]["rows"] == len(transactions)
    assert (output / "model_card.md").exists()
    assert (output / "eda" / "overview.png").exists()
    prepare(replace(config, row_limit=100))
    assert len(pd.read_parquet(processed / "train.parquet")) == 100
