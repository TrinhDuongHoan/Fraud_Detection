# Fraud Detection

A reproducible fraud detection project on IEEE-CIS transaction data, combining temporal model comparison, probability calibration and risk-based manual review. The operational constraint is a review capacity of **at most 1% of transactions per day**.

The project compares **Logistic Regression, Random Forest, LightGBM, XGBoost and CatBoost**. Its evaluation connects predictive performance to review precision, fraud recall, expected costs and processing time.

## Business Objective

The system assigns a fraud probability to each transaction and prioritizes manual review without using future transaction scores or daily volume. The decision space consists of **Approve** and **Manual Review**.

| Assumption | Value |
| --- | ---: |
| Maximum daily review rate | 1% |
| Loss per missed fraud | $100 |
| Cost per manual review | $1 |
| Reviewer detection sensitivity | 80% |

Expected savings are simulated under these assumptions. Automatic blocking is outside the policy, so false-blocking costs do not enter the calculation. The original [business problem](reports/01_problem_definition.md) provides the decision context.

## Model Comparison

All five families use identical chronological rows and the same original transaction features. The full configuration allocates **two candidates per family**, with a maximum of four CPU threads. Representations differ according to estimator requirements:

| Model | Representation |
| --- | --- |
| Logistic Regression | Median imputation, numeric scaling and sparse one-hot encoding |
| Random Forest | Training-vocabulary ordinal category codes and native numeric missing values |
| LightGBM | Native categorical splits and numeric missing values |
| XGBoost | Native categorical histogram splits and numeric missing values |
| CatBoost | Native categorical features with explicit missing-category tokens |

Logistic one-hot encoding groups infrequent categories and caps each categorical output at 128 columns. Random Forest's ordinal representation is a modeling assumption. Equal candidate counts are not equal wall-clock budgets or exhaustive hyperparameter optimization.

### Evaluation Design

| Partition | Approximate share | Role |
| --- | ---: | --- |
| Training | 60% | Fit candidate models and preprocessing |
| Tuning | 15% | Select candidate parameters, boosting iterations and model family |
| Calibration | 10% | Fit a sigmoid calibrator and select a review threshold for each model |
| Holdout | 15% | Evaluate every frozen pipeline |

Boundaries follow complete relative days. Each partition contains both classes. Category vocabularies, frequency estimates and constant-feature decisions are fitted only on the corresponding fitting period.

**Model-family selection uses tuning average precision and is recorded before calibration and holdout evaluation.** Each family's selected configuration is refitted on training plus tuning. The selected artifact remains the tuning winner even when another family has a better observed holdout metric.

Boosting early stopping uses LightGBM average precision, XGBoost `aucpr` and CatBoost `PRAUC`. Candidate and family selection recomputes sklearn average precision consistently across all models. The comparison reports:

- Average precision and ROC-AUC.
- Raw and calibrated log loss and Brier score.
- Actual review rate, review precision and fraud recall.
- Expected prevention, cost and savings under the stated assumptions.
- Selected-candidate fitting time, refit time, calibration time and holdout prediction latency.

Full experiment outputs are written to `artifacts/comparison/full`; the smoke experiment uses `artifacts/comparison/smoke`. Each directory includes a comparison table, figures, model-selection record and individual model artifacts.

### Reference Experiment

The earlier LightGBM-only experiment used **590,540 labeled transactions**, with **85,430 holdout transactions** and **506,691 unlabeled test transactions**.

| Holdout metric | LightGBM reference |
| --- | ---: |
| Average precision | 0.5384 |
| ROC-AUC | 0.8966 |
| Review rate | 0.784% |
| Review precision | 83.43% |
| Fraud recall through review | 18.67% |

These are the results of the original single-family experiment, documented in the [reference report](reports/02_results.md). The comparison reuses its holdout period and is a research evaluation rather than evidence from a new production period.

## Analytical Notebooks

The notebooks contain English analytical descriptions, executable cells, tables and figures. Training follows a linear cell sequence without an execution-mode flag.

| Notebook | Analysis |
| --- | --- |
| [Data Quality and EDA](notebooks/data_quality_eda.ipynb) | Source provenance, schema consistency, missingness, class imbalance and temporal profiles |
| [Features and Temporal Partitions](notebooks/features_temporal_split.ipynb) | Transaction features, fitted encoding, unseen categories and leakage boundaries |
| [Training and Calibration](notebooks/training_calibration.ipynb) | Five-family tuning, model selection, refitting, calibration and holdout comparison |
| [Evaluation and Inference](notebooks/evaluation_inference.ipynb) | Selected-model evaluation, quota checks, expected costs, drift and test scoring |

The first two notebooks analyze prepared data and the original CLI artifacts. The training notebook creates the comparison artifacts; the evaluation notebook uses that selected model for inference. `make notebooks` executes the sequence and saves outputs, including full model training.

## Execution

Python 3.11 or later and a CPU are required. The full comparison processes models sequentially; memory use and runtime depend on the estimator and machine. The dependency lock records the tested Python 3.14.4 environment. XGBoost uses the CPU package on x86-64 platforms.

Source files belong in `data/raw/`:

```text
train_transaction.csv
train_identity.csv
test_transaction.csv
test_identity.csv
sample_submission.csv
```

```bash
make install
make test
make lint
```

The full comparison uses prepared data:

```bash
.venv/bin/python -m fraud_detection prepare --config configs/comparison.yaml
make compare
.venv/bin/python -m fraud_detection predict --config configs/comparison.yaml
```

The smoke configuration processes the first 50,000 rows per dataset:

```bash
.venv/bin/python -m fraud_detection prepare --config configs/comparison_smoke.yaml
make compare-smoke
.venv/bin/python -m fraud_detection predict --config configs/comparison_smoke.yaml
```

A complete comparison, including preparation and inference, is also available as:

```bash
.venv/bin/python -m fraud_detection run --config configs/comparison.yaml
```

`configs/default.yaml` and `configs/smoke.yaml` retain the original LightGBM-only pipeline through `make run` and `make smoke`. Each experiment configuration has a separate artifact directory; the full and smoke datasets also use separate processed directories.

## Data and Features

CSV files are validated and processed in chunks. Identity attributes are left joined one-to-one on TransactionID after normalizing test column names such as `id-01` to `id_01`. Prepared data is stored as Parquet, with source SHA-256 hashes recorded in a manifest. Raw files remain unchanged.

Derived features include missing-value counts, relative hour and weekday, log amount, fractional amount, email-domain agreement and historical frequency encodings. TransactionID, the target and absolute TransactionDT are excluded from predictors. Unseen categories become missing categorical values; their fitted frequency is zero.

TransactionDT has no known calendar origin or timezone. Relative hour and weekday features do not establish actual local hours or calendar weekdays. Anonymized C/D/V and identity attributes require confirmation of their availability at decision time before deployment.

## Online Review Policy

Events follow `(TransactionDT, TransactionID)` order. After `n` transactions within a relative day, cumulative reviews cannot exceed `floor(0.01 × n)`. A transaction is selected only when capacity remains and its calibrated probability exceeds the threshold fitted on the calibration period.

The rule enforces the quota without future daily volume. It can miss early fraud or leave capacity unused; a day with fewer than 100 transactions has no review capacity. Daily top-k uses all scores from a completed day and serves only as a retrospective benchmark.

Chronological inference batches share a persistent review state:

```bash
.venv/bin/python -m fraud_detection predict \
  --config configs/comparison.yaml \
  --input data/processed/new_batch.parquet \
  --state artifacts/comparison/full/review_state.json
```

Input is a transaction table already joined with identity attributes, matching the prepared test schema and excluding `isFraud`. A subsequent batch must begin after the preceding batch's final event key. Each state belongs to one processing stream; concurrent writers and replaying an already processed batch are unsupported.

Prediction runs overwrite output files in the configured directory. Without `--state`, review allocation begins from an empty state, suitable for a standalone test file.

## Comparison Artifacts

| Path within the comparison output directory | Contents |
| --- | --- |
| `candidate_scores.csv` | Tuning metrics, candidate parameters and fitting times |
| `model_plans.json`, `selection.json` | Per-family refit parameters and the tuning-selected family |
| `fitted_models.json` | Per-model thresholds, calibration parameters and timings |
| `model_comparison.csv`, `comparison.md` | Complete holdout comparison and experiment interpretation |
| `comparison.png`, `precision_recall.png` | Comparison figures |
| `models/<family>/` | Every calibrated model, feature importance, calibration predictions, threshold search, reference distributions and holdout outputs |
| `model.joblib`, `reference.json` | Selected artifact and its monitoring reference |
| `metrics.json`, `holdout_predictions.parquet` | Selected-model metrics and holdout predictions |
| `daily_policy.csv`, `cost_sensitivity.csv` | Daily allocation and sensitivity to cost assumptions |
| `drift_holdout.csv`, `drift_test.csv` | Population stability index against calibration references |
| `predictions.csv`, `submission.csv` | Test scores, review actions and submission probabilities |
| `run.json` | Configuration, package versions, data hash and selected-model hash |

Serialized Python artifacts should only be loaded from trusted sources. Feature-importance definitions differ between model families and are not directly comparable. Test labels are unavailable, so submission integrity and quota compliance do not imply measured test-set classification quality.

## Repository Structure

```text
configs/                         Original and comparison experiment configurations
notebooks/                       Analysis, model comparison and inference
scripts/run_notebooks.py         Notebook execution and saved outputs
src/fraud_detection/
  config.py                      Configuration validation
  data.py                        Schema checks, joins, Parquet and provenance
  features.py                    Transaction features and fitted encoding
  split.py                       Chronological day partitions
  modeling.py                    Original LightGBM pipeline and small-feature baseline
  estimators.py                  Five estimator adapters and calibrated model artifact
  comparison.py                  Shared tuning, refitting and holdout evaluation
  comparison_reporting.py        Comparison tables, plots and research report
  policy.py                      Online allocation and expected costs
  metrics.py                     Classification metrics
  monitoring.py                  Reference distributions and PSI
  reporting.py                   Original EDA and evaluation reports
  pipeline.py                    Original LightGBM orchestration
  inference.py                   Batch scoring and submission generation
  cli.py                         Pipeline commands
reports/                         Business context and experiment summaries
tests/                          Data, model, leakage, policy and integration checks
```

## Validation and Limitations

Automated checks cover unseen categories across all five estimators, serialization, consistent feature order, independence from holdout labels during tuning, frozen family selection, review quotas at every daily prefix, batch-state continuity and end-to-end inference.

Calibration-period estimates can be optimistic because the sigmoid mapping and threshold use the same labels. Label delays and reviewer-processing latency are not simulated. Refitting on all labeled data would change the evaluated probabilities and thresholds.

PSI uses calibration as the reference distribution. A value of 0.2 is an investigation heuristic rather than an automatic retraining trigger. Subsequent labeled transactions support renewed measurement of calibration, fraud recall and realized costs.
