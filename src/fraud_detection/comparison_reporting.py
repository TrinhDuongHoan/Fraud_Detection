from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve


def comparison_report(table, labels, curves, selected, config) -> None:
    output = Path(config.output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    indexed = table.set_index("model")
    indexed[["tuning_average_precision", "average_precision"]].plot.bar(
        ax=axes[0, 0],
        title="Tuning and holdout average precision",
        rot=20,
    )
    indexed[["review_precision", "review_recall"]].plot.bar(
        ax=axes[0, 1],
        title="Holdout online review quality",
        rot=20,
    )
    indexed.expected_savings.plot.bar(ax=axes[1, 0], title="Expected holdout savings (USD)", rot=20)
    indexed[["tuning_fit_seconds", "fit_seconds", "predict_seconds"]].plot.bar(
        ax=axes[1, 1],
        title="Training and prediction time (seconds)",
        rot=20,
    )
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, scores in curves.items():
        precision, recall, _ = precision_recall_curve(labels, scores)
        ax.plot(recall, precision, label=name)
    ax.axhline(np.mean(labels), color="grey", linestyle="--", label="Class prior")
    ax.set(title="Holdout precision-recall by model", xlabel="Recall", ylabel="Precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "precision_recall.png", dpi=140)
    plt.close(fig)
    lines = [
        "# Temporal Model Comparison",
        "",
        f"Selected model: **{selected['model']}**, using tuning average precision "
        f"**{selected['tuning_average_precision']:.4f}**. "
        "Family selection is recorded before calibration and holdout evaluation.",
        "",
        "All families use identical chronological rows, transaction features "
        "and fitting exclusions. "
        "Each family has the configured candidate budget. Family-specific encoding is fitted "
        "on training data; the selected configuration is refitted on training plus tuning. "
        "Each model receives a separate sigmoid calibrator and review threshold on the same "
        "calibration period. Holdout labels do not select models, configurations or thresholds.",
        "",
        "| Model | Tuning AP | Holdout AP | ROC-AUC | Brier | Review rate | Precision | Recall | "
        "Expected savings | Refit seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in table.itertuples():
        lines.append(
            f"| {row.model} | {row.tuning_average_precision:.4f} | {row.average_precision:.4f} | "
            f"{row.roc_auc:.4f} | {row.brier_score:.5f} | {row.review_rate:.3%} | "
            f"{row.review_precision:.2%} | {row.review_recall:.2%} | "
            f"${row.expected_savings:,.0f} | {row.fit_seconds:.1f} |"
        )
    lines.extend(
        [
            "",
            "![Model comparison](comparison.png)",
            "",
            "![Precision-recall curves](precision_recall.png)",
            "",
            "## Encoding and Training Budgets",
            "",
        ]
    )
    for row in table.itertuples():
        lines.append(f"- **{row.model}**: {row.encoding}. Parameters: `{row.parameters}`.")
    lines.extend(
        [
            "",
            "The original feature columns and row populations are shared, while representations "
            "differ according to estimator requirements. "
            "Logistic one-hot encoding groups infrequent categories "
            "and limits each vocabulary to 128 outputs. Random Forest category codes impose "
            "an ordinal representation. Feature-importance definitions differ between families and "
            "are not directly comparable.",
            "",
            "LightGBM stops on average precision, XGBoost on aucpr and CatBoost on PRAUC. "
            "All candidate and family selection uses sklearn average precision recomputed on the "
            "same tuning labels. Equal candidate counts and thread limits are not equal wall-clock "
            "budgets or exhaustive optimization. Timings include family-specific preprocessing "
            "inside the estimator; holdout prediction timings also include feature transformation.",
            "",
            "## Review Policy and Assumptions",
            "",
            "Online allocation uses a maximum cumulative review count of "
            f"floor({config.review_rate} × n) "
            "after n transactions in a day. Thresholds differ by model but the quota is identical. "
            "Actual review rates can be below capacity. Daily top-k is retrospective only.",
            "",
            f"Expected savings assume ${config.fraud_loss:g} per fraud, "
            f"${config.review_cost:g} per review and {config.review_sensitivity:.0%} "
            "reviewer sensitivity. These are simulated outcomes. "
            "Calibration-period threshold estimates may be optimistic. The anonymized features, "
            "label latency and reviewer latency need point-in-time validation before deployment.",
            "",
            "The holdout period is shared with the earlier LightGBM experiment. This report is a "
            "comparative research evaluation, not evidence from a new production period. "
            "The deployed artifact remains the family selected on tuning even if another family "
            "has better observed holdout results.",
            "",
            "`model_comparison.csv` contains the full metrics and timings. "
            "`models/<family>/` stores every calibrated artifact and its policy outputs. "
            "Root-level model and reference files belong to the tuning-selected family "
            "and support the existing batch inference command.",
        ]
    )
    if config.row_limit:
        lines.extend(
            ["", f"This smoke experiment uses the first {config.row_limit:,} rows per dataset."]
        )
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
