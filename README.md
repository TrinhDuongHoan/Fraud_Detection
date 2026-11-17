# Fraud Detection

An end-to-end data science pipeline for IEEE-CIS transaction fraud detection, aligned with the [business problem](reports/01_problem_definition.md): score transaction risk and prioritize manual review within a daily capacity of 1% of transactions.

[Full dataset results](reports/02_results.md): holdout average precision **0.5384**, ROC-AUC **0.8966**, and review precision **83.43%** at a **0.784%** review rate.

## Analytical Notebooks

The notebooks present the analysis as a linear research workflow, with English descriptions, tables and figures from the complete dataset.

| Notebook | Analysis |
| --- | --- |
| [01 · Data Quality and EDA](notebooks/data_quality_eda.ipynb) | Business assumptions, source provenance, missingness, class imbalance, temporal variation and group profiles |
| [02 · Features and Temporal Partitions](notebooks/features_temporal_split.ipynb) | Chronological evaluation, transaction-level features, fitted encoding and leakage boundaries |
| [03 · Training and Calibration](notebooks/training_calibration.ipynb) | Direct candidate fitting, selected-model refit, sigmoid calibration, threshold selection and holdout evaluation |
| [04 · Evaluation and Inference](notebooks/evaluation_inference.ipynb) | Baselines, PR/ROC, online review allocation, batch state, expected costs, drift and unlabeled inference |

Notebooks 01–02 analyze the prepared data and CLI artifacts. Notebook 03 trains directly on the complete dataset and stores its artifact in `artifacts/notebook/full`. Notebook 04 evaluates that artifact and generates its test predictions and submission. Model training follows the cell sequence without an execution-mode flag.

`make notebooks` executes the complete sequence and saves cell outputs, including full model training and inference.

## Running the Project

The full configuration requires Python 3.11 or later, a CPU, and approximately 4–6 GB of available RAM.

```bash
make install
make test
make smoke
make run
```

`make smoke` processes the first 50,000 rows of each dataset and writes results to `artifacts/smoke`. `make run` processes the complete datasets and writes results to `artifacts/full`. Each configuration has separate processed data and model artifacts. Runtime depends on CPU performance and available memory.

The expected source files in `data/raw/` are:

```text
train_transaction.csv
train_identity.csv
test_transaction.csv
test_identity.csv
sample_submission.csv
```

Individual pipeline stages are available through the CLI:

```bash
.venv/bin/python -m fraud_detection prepare --config configs/default.yaml
.venv/bin/python -m fraud_detection eda --config configs/default.yaml
.venv/bin/python -m fraud_detection train --config configs/default.yaml
.venv/bin/python -m fraud_detection predict --config configs/default.yaml
```

## Pipeline

1. Validate schemas, duplicate keys, labels, timestamps and amounts; normalize `id-XX` to `id_XX`; and left join identity attributes on TransactionID. CSV files are processed in chunks and saved as Parquet, with SHA-256 source hashes recorded in a manifest. Raw files remain unchanged.
2. Analyze fraud prevalence, daily transaction volume, amount distributions, missingness, cardinality and constant columns, with supporting visualizations.
3. Partition complete days chronologically into approximately 60% training, 15% tuning, 10% calibration and 15% holdout. Actual proportions vary slightly because days are not split between partitions. Each partition must contain both classes.
4. Derive amount features, relative hour and weekday, missing-value counts and email-domain agreement. Category vocabularies, frequency encodings and constant-feature removal are fitted only on the fitting partition. Unseen categories become missing categorical values; unseen frequencies become zero. TransactionID and absolute transaction time are excluded from predictors.
5. Compare LightGBM configurations using tuning average precision, with early stopping on the same metric. Refit on training plus tuning with the selected iteration count. A constant-probability baseline and logistic regression on a small feature subset provide reference performance.
6. Fit sigmoid calibration and select the review threshold that maximizes expected savings on the calibration partition. Freeze the classifier, calibrator and threshold before holdout evaluation.
7. Evaluate average precision, ROC-AUC, log loss, Brier score, calibration, review precision and recall, expected costs and sensitivity to business assumptions. Report feature importance and distribution drift.
8. Score unlabeled test data in chunks and export probabilities, review actions and a submission aligned with sample_submission order. Submission files are generated locally.

## Review Policy Without Future Information

Transactions are processed in `(TransactionDT, TransactionID)` order. After `n` transactions within a day, cumulative reviews cannot exceed `floor(0.01 × n)`. A transaction is selected only when capacity remains and its probability exceeds the threshold selected on calibration data. The quota resets at the next relative day.

This rule enforces the 1% limit at every point in the event stream, but it can miss early fraud or leave capacity unused. A day with fewer than 100 transactions has zero review capacity. The reported `daily_top_k` result is a retrospective benchmark that requires all scores for a completed day; it is not an online decision policy.

Chronological batches share a persistent state file:

```bash
.venv/bin/python -m fraud_detection predict \
  --config configs/default.yaml \
  --input data/processed/new_batch.parquet \
  --state artifacts/full/review_state.json
```

Inference input contains transactions already joined with identity attributes, follows the prepared test schema, and excludes `isFraud`. Every batch must begin after the previous batch's final `(TransactionDT, TransactionID)` key. Each state belongs to a single processing stream; concurrent workers are not supported. Replaying a batch against an updated state is rejected.

Each prediction run overwrites the prediction files in `output_dir`, so retained results need separate storage. Without `--state`, the quota starts from an empty state, as expected for a standalone test file.

## Generated Artifacts

| Path within output_dir | Contents |
| --- | --- |
| `model_card.md` | Vietnamese model report covering results, assumptions and limitations |
| `eda/` | Data summary, column profiles, daily statistics and plots |
| `model.joblib` | Feature builder, LightGBM classifier, calibrator and policy settings |
| `run.json` | Configuration, package versions, and data and model hashes |
| `splits.json`, `candidate_scores.csv`, `selection.json` | Temporal partitions and model selection results |
| `metrics.json`, `evaluation.png` | Baselines, holdout metrics and evaluation plots |
| `holdout_predictions.parquet`, `daily_policy.csv` | Holdout scores and observed review allocation |
| `threshold_search.csv`, `cost_sensitivity.csv` | Threshold selection and cost-assumption analysis |
| `feature_importance.csv` | Feature importance measured by split gain |
| `reference.json`, `drift_holdout.csv`, `drift_test.csv` | Reference distributions and population stability index |
| `predictions.csv` | TransactionID, transaction time, risk_score, manual_review and action |
| `submission.csv` | TransactionID and predicted isFraud probability |

`model.joblib` is a serialized Python artifact and should only be loaded from trusted sources. `requirements-lock.txt` records the tested Python 3.14.4 environment; `requirements.txt` specifies supported dependency ranges for other Python versions.

## Project Structure

```text
configs/                  Full and smoke configurations
notebooks/                Four analysis, training and evaluation notebooks with saved outputs
scripts/run_notebooks.py  Notebook execution using the active Python environment
src/fraud_detection/
  config.py               Configuration and validation
  data.py                 Data validation, identity joins, Parquet and manifests
  features.py             Feature engineering and encoding
  split.py                Chronological partitions by day
  modeling.py             Baselines, LightGBM and model artifact
  policy.py               Online review quotas and expected costs
  metrics.py              Classification metrics
  monitoring.py           Reference distributions and PSI
  reporting.py            EDA, evaluation plots and model card
  pipeline.py             Training and evaluation orchestration
  inference.py            Batch prediction and submission generation
  cli.py                  Command-line interface
tests/                    Leakage, quota, schema and end-to-end pipeline checks
```

## Assumptions and Limitations

- The business assumptions assign a $100 loss per fraud, a $1 review cost and 80% review sensitivity. Reported savings are simulated expectations rather than measured production outcomes.
- The policy supports Approve and Manual Review. Automatic blocking is outside its scope, so the $5 false-blocking cost does not enter the calculation.
- TransactionDT has no known calendar origin. Days and hours are relative and do not establish an actual timezone or calendar weekday.
- C/D/V and identity attributes are anonymized. Production applicability depends on their availability at decision time, label delays and review-processing latency.
- The final classifier is fitted on training plus tuning, with calibration and holdout kept separate. Refitting on all labeled data would change the evaluated probabilities and threshold.
- PSI uses calibration as the reference period. A value of 0.2 is an investigation heuristic, not an automatic retraining trigger. New labels support renewed assessment of recall, costs and calibration.

Code quality and test checks are available through:

```bash
make lint
make test
```
