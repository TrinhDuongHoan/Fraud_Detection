import gc
import logging
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fraud_detection.config import Config
from fraud_detection.data import ID, TARGET, TIME, file_signature, validate_transactions
from fraud_detection.features import FeatureBuilder
from fraud_detection.metrics import classification_metrics
from fraud_detection.modeling import (
    FraudModel,
    baseline_frame,
    log_odds,
    logistic_baseline,
    new_booster,
)
from fraud_detection.monitoring import drift_report, reference_profile
from fraud_detection.policy import business_metrics, daily_top_k, online_review, select_threshold
from fraud_detection.reporting import eda, evaluation_plots, model_card, save_json
from fraud_detection.split import chronological_split

LOGGER = logging.getLogger(__name__)


def tune_parameters(frame, partitions, config: Config, output: Path) -> tuple[dict, dict]:
    train_rows, tune_rows = partitions["train"], partitions["tuning"]
    builder = FeatureBuilder()
    train_x = builder.fit_transform(frame.iloc[train_rows])
    tune_x = builder.transform(frame.iloc[tune_rows])
    candidates = []
    for index, parameters in enumerate(config.candidates):
        LOGGER.info("Fitting candidate %s/%s", index + 1, len(config.candidates))
        model = new_booster(parameters, config.seed, config.threads)
        model.fit(
            train_x,
            frame.iloc[train_rows][TARGET],
            eval_X=tune_x,
            eval_y=frame.iloc[tune_rows][TARGET],
            eval_metric="average_precision",
            callbacks=[
                lgb.early_stopping(
                    config.early_stopping_rounds,
                    first_metric_only=True,
                    verbose=False,
                )
            ],
        )
        score = classification_metrics(
            frame.iloc[tune_rows][TARGET],
            model.predict_proba(tune_x)[:, 1],
        )
        candidates.append(
            {
                "candidate": index,
                "best_iteration": model.best_iteration_,
                **score,
            }
        )
        LOGGER.info("Candidate %s: tuning AP %.5f", index, score["average_precision"])
        del model
        gc.collect()
    table = pd.DataFrame(candidates).sort_values(
        ["average_precision", "candidate"],
        ascending=[False, True],
    )
    table.to_csv(output / "candidate_scores.csv", index=False)
    winner = table.iloc[0]
    parameters = dict(config.candidates[int(winner.candidate)])
    parameters["n_estimators"] = max(1, int(winner.best_iteration))
    del train_x, tune_x, builder
    gc.collect()
    return parameters, {
        "candidate": int(winner.candidate),
        "tuning_average_precision": float(winner.average_precision),
    }


def evaluate_holdout(
    frame,
    partitions,
    artifact,
    calibration_scores,
    importance,
    config: Config,
    output: Path,
) -> dict:
    fit_rows = np.r_[partitions["train"], partitions["tuning"]]
    calibration = frame.iloc[partitions["calibration"]]
    holdout = frame.iloc[partitions["holdout"]]
    raw_scores = artifact.raw_probability(holdout)
    scores = artifact.calibrator.predict_proba(log_odds(raw_scores))[:, 1]
    labels = holdout[TARGET].to_numpy()
    LOGGER.info("Fitting logistic baseline")
    baseline = logistic_baseline(config.seed)
    baseline.fit(baseline_frame(frame.iloc[fit_rows]), frame.iloc[fit_rows][TARGET])
    baseline_scores = baseline.predict_proba(baseline_frame(holdout))[:, 1]
    baseline_calibration = baseline.predict_proba(baseline_frame(calibration))[:, 1]
    baseline_threshold, _ = select_threshold(calibration, baseline_calibration, config)
    baseline_reviews = online_review(
        holdout,
        baseline_scores,
        baseline_threshold,
        config.review_rate,
    )
    reviewed = online_review(holdout, scores, artifact.threshold, artifact.review_rate)
    retrospective = daily_top_k(holdout, scores, config.review_rate)
    cost_args = (config.fraud_loss, config.review_cost, config.review_sensitivity)
    metrics = {
        "prior_baseline": classification_metrics(
            labels,
            np.full(len(labels), frame.iloc[fit_rows][TARGET].mean()),
        ),
        "logistic_baseline": classification_metrics(labels, baseline_scores),
        "logistic_online_policy": business_metrics(labels, baseline_reviews, *cost_args),
        "lightgbm_raw": classification_metrics(labels, raw_scores),
        "lightgbm_calibrated": classification_metrics(labels, scores),
        "online_policy": business_metrics(labels, reviewed, *cost_args),
        "retrospective_daily_top_k": business_metrics(labels, retrospective, *cost_args),
    }
    save_json(output / "metrics.json", metrics)
    predictions = holdout[[ID, TIME, TARGET]].copy()
    predictions["risk_score"] = scores
    predictions["manual_review"] = reviewed
    predictions["day"] = (predictions[TIME] // 86400).astype(int)
    predictions.to_parquet(output / "holdout_predictions.parquet", index=False)
    daily = predictions.groupby("day").agg(
        transactions=(ID, "size"),
        frauds=(TARGET, "sum"),
        reviews=("manual_review", "sum"),
    )
    daily["review_rate"] = daily.reviews / daily.transactions
    daily["fraud_in_review"] = (
        predictions.assign(
            captured=predictions[TARGET] * predictions.manual_review,
        )
        .groupby("day")
        .captured.sum()
    )
    daily.to_csv(output / "daily_policy.csv")
    sensitivity = []
    for detection in [0.5, 0.8, 1.0]:
        for loss in [50, 100, 200]:
            sensitivity.append(
                {
                    "review_sensitivity": detection,
                    "fraud_loss": loss,
                    **business_metrics(labels, reviewed, loss, config.review_cost, detection),
                }
            )
    pd.DataFrame(sensitivity).to_csv(output / "cost_sensitivity.csv", index=False)
    reference = reference_profile(calibration, calibration_scores)
    save_json(output / "reference.json", reference)
    drift_report(holdout, scores, reference).to_csv(output / "drift_holdout.csv", index=False)
    evaluation_plots(labels, raw_scores, scores, importance, output)
    return metrics


def train(config: Config) -> dict:
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(config.processed_dir) / "train.parquet"
    frame = pd.read_parquet(source)
    validate_transactions(frame, labeled=True)
    frame = frame.sort_values([TIME, ID]).reset_index(drop=True)
    LOGGER.info("Loaded %s labeled transactions", len(frame))
    eda(frame, output / "eda")
    partitions = chronological_split(frame, config)
    splits = {}
    for name, positions in partitions.items():
        subset = frame.iloc[positions]
        splits[name] = {
            "rows": len(subset),
            "first_day": int(subset[TIME].min() // 86400),
            "last_day": int(subset[TIME].max() // 86400),
            "fraud_rate": float(subset[TARGET].mean()),
        }
    save_json(output / "splits.json", splits)
    parameters, selection = tune_parameters(frame, partitions, config, output)
    fit_rows = np.r_[partitions["train"], partitions["tuning"]]
    builder = FeatureBuilder()
    fit_x = builder.fit_transform(frame.iloc[fit_rows])
    LOGGER.info(
        "Refitting selected model on %s rows with %s trees",
        len(fit_rows),
        parameters["n_estimators"],
    )
    model = new_booster(parameters, config.seed, config.threads)
    model.fit(fit_x, frame.iloc[fit_rows][TARGET])
    importance = pd.DataFrame(
        {
            "feature": model.feature_name_,
            "gain": model.booster_.feature_importance("gain"),
        }
    ).sort_values("gain", ascending=False)
    importance.to_csv(output / "feature_importance.csv", index=False)
    del fit_x
    gc.collect()
    calibration = frame.iloc[partitions["calibration"]]
    raw_calibration = model.predict_proba(builder.transform(calibration))[:, 1]
    calibrator = LogisticRegression(C=100.0, solver="lbfgs", random_state=config.seed)
    calibrator.fit(log_odds(raw_calibration), calibration[TARGET])
    calibration_scores = calibrator.predict_proba(log_odds(raw_calibration))[:, 1]
    threshold, threshold_table = select_threshold(calibration, calibration_scores, config)
    threshold_table.to_csv(output / "threshold_search.csv", index=False)
    artifact = FraudModel(builder, model, calibrator, threshold, config.review_rate)
    joblib.dump(artifact, output / "model.joblib", compress=3)
    selection = {
        **selection,
        "parameters": parameters,
        "threshold": threshold,
        "calibration_slope": float(calibrator.coef_[0, 0]),
        "calibration_intercept": float(calibrator.intercept_[0]),
    }
    save_json(output / "selection.json", selection)
    metrics = evaluate_holdout(
        frame,
        partitions,
        artifact,
        calibration_scores,
        importance,
        config,
        output,
    )
    model_card(output, config, metrics, splits, selection)
    save_json(
        output / "run.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "config": config.as_dict(),
            "python": platform.python_version(),
            "packages": {
                name: version(name)
                for name in [
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "lightgbm",
                    "pyarrow",
                    "joblib",
                ]
            },
            "training_data": file_signature(source),
            "model": file_signature(output / "model.joblib"),
        },
    )
    LOGGER.info(
        "Holdout AP %.5f; expected savings %.2f",
        metrics["lightgbm_calibrated"]["average_precision"],
        metrics["online_policy"]["expected_savings"],
    )
    return metrics
