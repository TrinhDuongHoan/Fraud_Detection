import json
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest

from fraud_detection.comparison import (
    evaluate_models,
    prepare_comparison,
    refit_models,
    tune_models,
)
from fraud_detection.config import Config
from fraud_detection.estimators import TabularEstimator
from fraud_detection.features import FeatureBuilder
from fraud_detection.inference import predict

TRIALS = {
    "logistic": [{"C": 0.1, "max_iter": 100}],
    "random_forest": [{"n_estimators": 8, "max_depth": 4, "min_samples_leaf": 10}],
    "lightgbm": [{"n_estimators": 8, "num_leaves": 7}],
    "xgboost": [{"n_estimators": 8, "max_depth": 3}],
    "catboost": [{"iterations": 8, "depth": 3}],
}


@pytest.mark.parametrize("name", list(TRIALS))
def test_estimator_handles_unseen_categories_and_round_trip(name, transactions, tmp_path):
    builder = FeatureBuilder()
    train = builder.fit_transform(transactions.iloc[:1800])
    tuning = builder.transform(transactions.iloc[1800:2400])
    estimator = TabularEstimator(name, TRIALS[name][0], seed=42, threads=1)
    estimator.fit(
        train,
        transactions.iloc[:1800].isFraud,
        (tuning, transactions.iloc[1800:2400].isFraud),
        patience=3,
    )
    future = transactions.iloc[2400:].copy()
    future["card1"], future["ProductCD"] = 999999.0, "UNSEEN"
    transformed = builder.transform(future)
    scores = estimator.predict_proba(transformed)[:, 1]
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    path = tmp_path / f"{name}.joblib"
    joblib.dump(estimator, path)
    np.testing.assert_allclose(joblib.load(path).predict_proba(transformed)[:, 1], scores)
    assert len(estimator.feature_importance()) > 0
    with pytest.raises(ValueError, match="column order"):
        estimator.predict_proba(transformed.iloc[:, ::-1])


def test_comparison_freezes_selection_and_supports_batch_inference(tmp_path, transactions):
    processed, output = tmp_path / "processed", tmp_path / "output"
    processed.mkdir()
    transactions.to_parquet(processed / "train.parquet", index=False)
    (processed / "manifest.json").write_text(json.dumps({"row_limit": None}))
    config = Config(
        processed_dir=str(processed),
        output_dir=str(output),
        model_candidates=TRIALS,
        threads=1,
        early_stopping_rounds=3,
    )
    frame, partitions = prepare_comparison(config)
    plans = tune_models(frame, partitions, config)
    selected_before_fit = json.loads((output / "selection.json").read_text())
    assert selected_before_fit["model"] == max(
        plans, key=lambda name: plans[name]["tuning_average_precision"]
    )
    assert not selected_before_fit["holdout_used_for_selection"]
    fitted = refit_models(frame, partitions, config, plans)
    frozen = json.loads((output / "selection.json").read_text())
    table = evaluate_models(frame, partitions, config, fitted)
    assert json.loads((output / "selection.json").read_text()) == frozen
    assert set(table.model) == set(TRIALS)
    assert table.selected_on_tuning.sum() == 1
    assert (table.review_rate <= config.review_rate).all()
    assert (table.input_features == table.input_features.iloc[0]).all()
    for name in TRIALS:
        predictions = pd.read_parquet(output / "models" / name / "holdout_predictions.parquet")
        days = predictions.TransactionDT // 86400
        counts = predictions.groupby(days).cumcount() + 1
        reviews = predictions.manual_review.groupby(days).cumsum()
        assert (reviews <= np.floor(counts * config.review_rate + 1e-12)).all()
    test = transactions.drop(columns="isFraud").iloc[:300].copy()
    test.TransactionID += 10000
    test.TransactionDT += 40 * 86400
    source = processed / "test.parquet"
    test.to_parquet(source, index=False)
    predictions = predict(config, input_path=source)
    artifact = joblib.load(output / "model.joblib")
    assert artifact.estimator.name == frozen["model"]
    np.testing.assert_allclose(artifact.predict_proba(test), predictions.risk_score)
    assert (output / "comparison.md").exists()
    assert (output / "comparison.png").exists()


def test_tuning_does_not_use_holdout_labels(tmp_path, transactions):
    trials = {name: TRIALS[name] for name in ["logistic", "lightgbm"]}
    processed = tmp_path / "processed"
    processed.mkdir()
    transactions.to_parquet(processed / "train.parquet", index=False)
    (processed / "manifest.json").write_text(json.dumps({"row_limit": None}))
    config = Config(
        processed_dir=str(processed),
        output_dir=str(tmp_path / "first"),
        model_candidates=trials,
        threads=1,
        early_stopping_rounds=3,
    )
    frame, partitions = prepare_comparison(config)
    original = tune_models(frame, partitions, config)
    changed = frame.copy()
    changed.loc[partitions["holdout"], "isFraud"] = (
        1 - changed.loc[partitions["holdout"], "isFraud"]
    )
    second = replace(config, output_dir=str(tmp_path / "second"))
    (tmp_path / "second").mkdir()
    repeated = tune_models(changed, partitions, second)
    for name in trials:
        assert original[name]["parameters"] == repeated[name]["parameters"]
        assert (
            original[name]["tuning_average_precision"] == repeated[name]["tuning_average_precision"]
        )


def test_unknown_or_empty_model_configuration_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        Config(model_candidates={"unknown": [{}]})
    with pytest.raises(ValueError, match="nonempty"):
        Config(model_candidates={"lightgbm": []})


def test_run_uses_comparison_when_model_families_are_configured(monkeypatch):
    from unittest.mock import Mock

    from fraud_detection import cli

    config = Config(model_candidates=TRIALS)
    compare, legacy_train, prepare, predict = Mock(), Mock(), Mock(), Mock()
    monkeypatch.setattr("sys.argv", ["fraud", "run"])
    monkeypatch.setattr(cli.Config, "load", lambda path: config)
    monkeypatch.setattr(cli, "compare_models", compare)
    monkeypatch.setattr(cli, "train", legacy_train)
    monkeypatch.setattr(cli, "prepare", prepare)
    monkeypatch.setattr(cli, "predict", predict)
    cli.main()
    compare.assert_called_once_with(config)
    legacy_train.assert_not_called()
    prepare.assert_called_once_with(config)
    predict.assert_called_once_with(config, None, None)
