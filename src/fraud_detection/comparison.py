import gc
import json
import logging
import platform
import shutil
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fraud_detection.comparison_reporting import comparison_report
from fraud_detection.config import Config
from fraud_detection.data import ID, TARGET, TIME, file_signature, validate_transactions
from fraud_detection.estimators import ENCODING, CalibratedModel, TabularEstimator
from fraud_detection.features import FeatureBuilder
from fraud_detection.metrics import classification_metrics
from fraud_detection.modeling import log_odds
from fraud_detection.monitoring import drift_report, reference_profile
from fraud_detection.policy import business_metrics, daily_top_k, online_review, select_threshold
from fraud_detection.reporting import save_json
from fraud_detection.split import chronological_split

LOGGER = logging.getLogger(__name__)


def prepare_comparison(config: Config) -> tuple[pd.DataFrame, dict]:
    if not config.model_candidates:
        raise ValueError("Model comparison requires model_candidates in the configuration")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(config.processed_dir) / "manifest.json").read_text())
    if manifest["row_limit"] != config.row_limit:
        raise ValueError("Processed data row_limit differs from comparison configuration")
    frame = pd.read_parquet(Path(config.processed_dir) / "train.parquet")
    validate_transactions(frame, labeled=True)
    frame = frame.sort_values([TIME, ID]).reset_index(drop=True)
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
    LOGGER.info(
        "Comparison data: %s rows; partitions %s",
        len(frame),
        {name: len(rows) for name, rows in partitions.items()},
    )
    return frame, partitions


def tune_models(frame: pd.DataFrame, partitions: dict, config: Config) -> dict:
    output = Path(config.output_dir)
    builder = FeatureBuilder()
    train_rows, tuning_rows = partitions["train"], partitions["tuning"]
    train_x = builder.fit_transform(frame.iloc[train_rows])
    tuning_x = builder.transform(frame.iloc[tuning_rows])
    train_y, tuning_y = frame.iloc[train_rows][TARGET], frame.iloc[tuning_rows][TARGET]
    candidates, plans = [], {}
    for name, trials in config.model_candidates.items():
        best_score = -np.inf
        for index, parameters in enumerate(trials):
            LOGGER.info("Tuning %s candidate %s/%s", name, index + 1, len(trials))
            estimator = TabularEstimator(name, parameters, config.seed, config.threads)
            start = perf_counter()
            estimator.fit(train_x, train_y, (tuning_x, tuning_y), config.early_stopping_rounds)
            fit_seconds = perf_counter() - start
            scores = estimator.predict_proba(tuning_x)[:, 1]
            metrics = classification_metrics(tuning_y, scores)
            chosen_parameters = estimator.refit_parameters()
            row = {
                "model": name,
                "candidate": index,
                "fit_seconds": fit_seconds,
                "parameters": json.dumps(parameters, sort_keys=True),
                "refit_parameters": json.dumps(chosen_parameters, sort_keys=True),
                **metrics,
            }
            candidates.append(row)
            if metrics["average_precision"] > best_score:
                best_score = metrics["average_precision"]
                plans[name] = {
                    "parameters": chosen_parameters,
                    "candidate": index,
                    "tuning_average_precision": best_score,
                    "tuning_fit_seconds": fit_seconds,
                    "input_features": len(train_x.columns),
                    "encoding": ENCODING[name],
                }
            LOGGER.info(
                "%s tuning AP %.5f; fit %.1fs", name, metrics["average_precision"], fit_seconds
            )
            del estimator, scores
            gc.collect()
    pd.DataFrame(candidates).to_csv(output / "candidate_scores.csv", index=False)
    save_json(output / "model_plans.json", plans)
    winner = max(plans, key=lambda name: plans[name]["tuning_average_precision"])
    save_json(
        output / "selection.json",
        {
            "model": winner,
            "criterion": "highest_tuning_average_precision",
            "holdout_used_for_selection": False,
            **plans[winner],
        },
    )
    LOGGER.info("Selected %s on tuning before calibration and holdout evaluation", winner)
    return plans


def refit_models(frame: pd.DataFrame, partitions: dict, config: Config, plans: dict) -> dict:
    output = Path(config.output_dir)
    fit_rows = np.r_[partitions["train"], partitions["tuning"]]
    builder = FeatureBuilder()
    fit_x = builder.fit_transform(frame.iloc[fit_rows])
    calibration = frame.iloc[partitions["calibration"]]
    calibration_x = builder.transform(calibration)
    labels = frame.iloc[fit_rows][TARGET]
    fitted = {}
    for name, plan in plans.items():
        destination = output / "models" / name
        destination.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Refitting and calibrating %s on %s fitting rows", name, len(fit_rows))
        estimator = TabularEstimator(name, plan["parameters"], config.seed, config.threads)
        start = perf_counter()
        estimator.fit(fit_x, labels)
        fit_seconds = perf_counter() - start
        start = perf_counter()
        raw_scores = estimator.predict_proba(calibration_x)[:, 1]
        calibrator = LogisticRegression(C=100.0, solver="lbfgs", random_state=config.seed)
        calibrator.fit(log_odds(raw_scores), calibration[TARGET])
        scores = calibrator.predict_proba(log_odds(raw_scores))[:, 1]
        threshold, thresholds = select_threshold(calibration, scores, config)
        calibration_seconds = perf_counter() - start
        artifact = CalibratedModel(builder, estimator, calibrator, threshold, config.review_rate)
        joblib.dump(artifact, destination / "model.joblib", compress=3)
        thresholds.to_csv(destination / "threshold_search.csv", index=False)
        estimator.feature_importance().to_csv(destination / "feature_importance.csv", index=False)
        predictions = calibration[[ID, TIME, TARGET]].copy()
        predictions["raw_score"], predictions["risk_score"] = raw_scores, scores
        predictions.to_parquet(destination / "calibration_predictions.parquet", index=False)
        save_json(destination / "reference.json", reference_profile(calibration, scores))
        fitted[name] = {
            **plan,
            "fit_seconds": fit_seconds,
            "calibration_seconds": calibration_seconds,
            "threshold": threshold,
            "fit_rows": len(fit_rows),
            "calibration_slope": float(calibrator.coef_[0, 0]),
            "calibration_intercept": float(calibrator.intercept_[0]),
            "input_features": len(fit_x.columns),
        }
        save_json(destination / "selection.json", fitted[name])
        LOGGER.info("%s refit %.1fs; threshold %.6f", name, fit_seconds, threshold)
        del estimator, artifact, predictions, calibrator
        gc.collect()
    save_json(output / "fitted_models.json", fitted)
    selected = json.loads((output / "selection.json").read_text())
    selected.update(fitted[selected["model"]])
    save_json(output / "selection.json", selected)
    for filename in [
        "model.joblib",
        "reference.json",
        "threshold_search.csv",
        "feature_importance.csv",
    ]:
        shutil.copyfile(output / "models" / selected["model"] / filename, output / filename)
    return fitted


def _policy_outputs(holdout, scores, reviewed, output: Path) -> None:
    predictions = holdout[[ID, TIME, TARGET]].copy()
    predictions["risk_score"], predictions["manual_review"] = scores, reviewed
    predictions["day"] = (predictions[TIME] // 86400).astype(int)
    predictions.to_parquet(output / "holdout_predictions.parquet", index=False)
    daily = (
        predictions.assign(captured=predictions[TARGET] * predictions.manual_review)
        .groupby(
            "day",
        )
        .agg(
            transactions=(ID, "size"),
            frauds=(TARGET, "sum"),
            reviews=("manual_review", "sum"),
            fraud_in_review=("captured", "sum"),
        )
    )
    daily["review_rate"] = daily.reviews / daily.transactions
    daily.to_csv(output / "daily_policy.csv")


def evaluate_models(
    frame: pd.DataFrame, partitions: dict, config: Config, fitted: dict
) -> pd.DataFrame:
    output = Path(config.output_dir)
    selected = json.loads((output / "selection.json").read_text())
    holdout = frame.iloc[partitions["holdout"]]
    labels = holdout[TARGET].to_numpy()
    cost_args = (config.fraud_loss, config.review_cost, config.review_sensitivity)
    rows, curves = [], {}
    for name, metadata in fitted.items():
        LOGGER.info("Evaluating frozen %s on holdout", name)
        destination = output / "models" / name
        artifact = joblib.load(destination / "model.joblib")
        start = perf_counter()
        raw_scores = artifact.raw_probability(holdout)
        scores = artifact.calibrator.predict_proba(log_odds(raw_scores))[:, 1]
        predict_seconds = perf_counter() - start
        reviewed = online_review(holdout, scores, artifact.threshold, artifact.review_rate)
        metrics = {
            "raw": classification_metrics(labels, raw_scores),
            "calibrated": classification_metrics(labels, scores),
            "online_policy": business_metrics(labels, reviewed, *cost_args),
            "retrospective_daily_top_k": business_metrics(
                labels,
                daily_top_k(holdout, scores, config.review_rate),
                *cost_args,
            ),
        }
        save_json(destination / "metrics.json", metrics)
        _policy_outputs(holdout, scores, reviewed, destination)
        reference = json.loads((destination / "reference.json").read_text())
        drift_report(holdout, scores, reference).to_csv(
            destination / "drift_holdout.csv", index=False
        )
        rows.append(
            {
                "model": name,
                "selected_on_tuning": name == selected["model"],
                **metadata,
                **metrics["calibrated"],
                "raw_brier_score": metrics["raw"]["brier_score"],
                "raw_log_loss": metrics["raw"]["log_loss"],
                **metrics["online_policy"],
                "predict_seconds": predict_seconds,
                "predict_ms_per_1000": predict_seconds / len(holdout) * 1_000_000,
            }
        )
        curves[name] = scores
        if name == selected["model"]:
            save_json(
                output / "metrics.json",
                {
                    "selected_model": name,
                    "selected_raw": metrics["raw"],
                    "selected_calibrated": metrics["calibrated"],
                    "online_policy": metrics["online_policy"],
                    "retrospective_daily_top_k": metrics["retrospective_daily_top_k"],
                },
            )
            for filename in [
                "holdout_predictions.parquet",
                "daily_policy.csv",
                "drift_holdout.csv",
            ]:
                shutil.copyfile(destination / filename, output / filename)
            sensitivity = [
                {
                    "review_sensitivity": detection,
                    "fraud_loss": loss,
                    **business_metrics(labels, reviewed, loss, config.review_cost, detection),
                }
                for detection in [0.5, 0.8, 1.0]
                for loss in [50, 100, 200]
            ]
            pd.DataFrame(sensitivity).to_csv(output / "cost_sensitivity.csv", index=False)
        del artifact
        gc.collect()
    table = (
        pd.DataFrame(rows)
        .sort_values(
            ["tuning_average_precision", "model"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )
    table["parameters"] = table.parameters.map(lambda value: json.dumps(value, sort_keys=True))
    table.to_csv(output / "model_comparison.csv", index=False)
    fit_rows = np.r_[partitions["train"], partitions["tuning"]]
    prior = np.full(len(labels), frame.iloc[fit_rows][TARGET].mean())
    save_json(output / "prior_baseline.json", classification_metrics(labels, prior))
    comparison_report(table, labels, curves, selected, config)
    save_json(
        output / "run.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "config": config.as_dict(),
            "python": platform.python_version(),
            "selection": selected,
            "packages": {
                name: version(name)
                for name in [
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "lightgbm",
                    "catboost",
                ]
            },
            "xgboost": __import__("xgboost").__version__,
            "training_data": file_signature(Path(config.processed_dir) / "train.parquet"),
            "model": file_signature(output / "model.joblib"),
        },
    )
    LOGGER.info(
        "Comparison complete. Selected %s by tuning AP, not holdout results", selected["model"]
    )
    return table


def compare_models(config: Config) -> pd.DataFrame:
    frame, partitions = prepare_comparison(config)
    plans = tune_models(frame, partitions, config)
    fitted = refit_models(frame, partitions, config, plans)
    return evaluate_models(frame, partitions, config, fitted)
