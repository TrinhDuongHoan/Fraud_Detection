import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from fraud_detection.data import ID, TARGET, TIME


def save_json(path: Path, value) -> None:
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Cannot serialize {type(item)}")

    path.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False))


def eda(frame: pd.DataFrame, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    missing = frame.isna().mean().sort_values(ascending=False)
    profile = pd.DataFrame(
        {
            "dtype": frame.dtypes.astype(str),
            "missing_rate": frame.isna().mean(),
            "unique_values": frame.nunique(),
        }
    )
    profile.to_csv(output / "feature_profile.csv", index_label="feature")
    daily = (
        frame.assign(day=(frame[TIME] // 86400).astype(int))
        .groupby("day")
        .agg(
            transactions=(ID, "size"),
            fraud_count=(TARGET, "sum"),
            fraud_rate=(TARGET, "mean"),
            amount=("TransactionAmt", "sum"),
        )
    )
    daily.to_csv(output / "daily_data.csv")
    summary = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "fraud_count": int(frame[TARGET].sum()),
        "fraud_rate": float(frame[TARGET].mean()),
        "duplicate_ids": int(frame[ID].duplicated().sum()),
        "first_day": int(daily.index.min()),
        "last_day": int(daily.index.max()),
        "missing_cells_rate": float(frame.isna().to_numpy().mean()),
        "all_missing_columns": missing[missing == 1].index.tolist(),
        "constant_columns": profile.index[profile.unique_values <= 1].tolist(),
        "amount_quantiles": frame.TransactionAmt.quantile(
            [0, 0.25, 0.5, 0.75, 0.95, 0.99, 1]
        ).to_dict(),
    }
    save_json(output / "summary.json", summary)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    daily.transactions.plot(ax=axes[0, 0], title="Transactions by relative day")
    daily.fraud_rate.plot(ax=axes[0, 1], title="Fraud rate by relative day")
    for label, name in [(0, "Legitimate"), (1, "Fraud")]:
        axes[1, 0].hist(
            np.log1p(frame.loc[frame[TARGET] == label, "TransactionAmt"]),
            bins=50,
            alpha=0.5,
            density=True,
            label=name,
        )
    axes[1, 0].set(title="Transaction amount", xlabel="log(1 + amount)")
    axes[1, 0].legend()
    missing.head(20).sort_values().plot.barh(ax=axes[1, 1], title="Highest missing rates")
    fig.tight_layout()
    fig.savefig(output / "overview.png", dpi=140)
    plt.close(fig)
    return summary


def evaluation_plots(labels, raw_scores, scores, importance: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    precision, recall, _ = precision_recall_curve(labels, scores)
    axes[0, 0].plot(recall, precision)
    axes[0, 0].axhline(np.mean(labels), color="grey", linestyle="--")
    axes[0, 0].set(title="Holdout precision-recall", xlabel="Recall", ylabel="Precision")
    fpr, tpr, _ = roc_curve(labels, scores)
    axes[0, 1].plot(fpr, tpr)
    axes[0, 1].plot([0, 1], [0, 1], "--", color="grey")
    axes[0, 1].set(title="Holdout ROC", xlabel="False positive rate", ylabel="True positive rate")
    for values, label in [(raw_scores, "Raw"), (scores, "Calibrated")]:
        actual, predicted = calibration_curve(labels, values, n_bins=10, strategy="quantile")
        axes[1, 0].plot(predicted, actual, marker="o", label=label)
    axes[1, 0].plot([0, 1], [0, 1], "--", color="grey")
    axes[1, 0].set(title="Probability calibration", xlabel="Predicted", ylabel="Observed")
    axes[1, 0].legend()
    importance.head(20).iloc[::-1].plot.barh(
        x="feature",
        y="gain",
        ax=axes[1, 1],
        legend=False,
        title="LightGBM split gain",
    )
    fig.tight_layout()
    fig.savefig(output / "evaluation.png", dpi=140)
    plt.close(fig)


def model_card(output: Path, config, metrics: dict, splits: dict, selected: dict) -> None:
    score = metrics["lightgbm_calibrated"]
    policy = metrics["online_policy"]
    rows = [
        "# Báo cáo mô hình Fraud Detection",
        "",
        "## Thiết kế đánh giá",
        "",
        "Chia theo ngày tương đối của TransactionDT, không trộn ngày giữa các tập. "
        "Chọn cấu hình trên tuning bằng average precision; "
        "refit train + tuning với số cây đã chọn. "
        "Fit sigmoid calibration và chọn ngưỡng trên calibration. Holdout chỉ dùng đánh giá cuối.",
        "",
        "| Tập | Số dòng | Ngày đầu | Ngày cuối | Tỷ lệ fraud |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, split in splits.items():
        rows.append(
            f"| {name} | {split['rows']:,} | {split['first_day']} | {split['last_day']} "
            f"| {split['fraud_rate']:.4%} |"
        )
    rows.extend(
        [
            "",
            "## Kết quả holdout",
            "",
            f"- Average precision: {score['average_precision']:.4f}",
            f"- ROC-AUC: {score['roc_auc']:.4f}",
            f"- Brier score: {score['brier_score']:.5f}",
            f"- Ngưỡng review: {selected['threshold']:.6f}",
            f"- Số giao dịch review: {policy['reviews']:,} ({policy['review_rate']:.3%})",
            f"- Precision / recall trong review: {policy['review_precision']:.2%} / "
            f"{policy['review_recall']:.2%}",
            f"- Chi phí không review: ${policy['no_review_cost']:,.2f}",
            f"- Chi phí kỳ vọng khi review: ${policy['expected_cost']:,.2f}",
            f"- Tiết kiệm kỳ vọng: ${policy['expected_savings']:,.2f}",
            "",
            "![Evaluation](evaluation.png)",
            "",
            "## Chính sách quyết định",
            "",
            "Xử lý theo (TransactionDT, TransactionID). "
            "Sau n giao dịch trong ngày, tổng số review "
            f"không vượt floor({config.review_rate} × n). "
            "Chỉ review khi score vượt ngưỡng đã khóa. "
            "Quota reset khi sang ngày; trạng thái cần được lưu giữa các batch. "
            "Chính sách này bảo đảm giới hạn ngay cả khi chưa biết tổng lưu lượng ngày.",
            "",
            "Benchmark daily_top_k sử dụng toàn bộ điểm số trong ngày, chỉ là so sánh hồi cứu; "
            "không phải kết quả có thể triển khai trực tuyến. "
            "Quota online có thể để trống công suất hoặc bỏ lỡ fraud đến sớm. "
            "Ngưỡng được tối ưu trên calibration nên kết quả tại đó có thể lạc quan.",
            "",
            "## Giả định và giới hạn",
            "",
            f"Mỗi fraud bỏ sót có chi phí cố định ${config.fraud_loss:g}; mỗi review tốn "
            f"${config.review_cost:g}; review phát hiện {config.review_sensitivity:.0%} fraud. "
            "Đây là giả định mô phỏng, chưa phải số tiền tiết kiệm thực tế. "
            "Chưa automatic blocking nên không áp chi phí chặn nhầm $5. "
            "TransactionAmt không được dùng làm tổn thất mặc định.",
            "",
            "Các nhóm C/D/V và identity có ý nghĩa ẩn danh: "
            "cần xác minh chúng có sẵn tại thời điểm ra quyết định trước khi triển khai. "
            "Pipeline không tạo aggregate từ tương lai; frequency "
            "encoding chỉ fit trên dữ liệu huấn luyện. Không dùng SMOTE trước khi chia tập.",
            "",
            "TransactionDT là thời gian tương đối; hour/weekday cũng tương đối, không suy ra "
            "ngày lịch hoặc múi giờ thực. "
            "Chưa mô phỏng độ trễ nhãn và giới hạn thời gian xử lý review. "
            "Không có nhãn test nên submission không có đánh giá chất lượng trên test.",
            "",
            "Feature importance là split gain, không phải bằng chứng nhân quả. PSI >= 0.2 chỉ là "
            "ngưỡng gợi ý điều tra drift. "
            "Cần đánh giá lại calibration, capacity và fraud recall khi có nhãn mới.",
            "",
            "## Artifacts",
            "",
            "`model.joblib`, `metrics.json`, `splits.json`, `selection.json`, "
            "`threshold_search.csv`, "
            "`candidate_scores.csv`, `holdout_predictions.parquet`, `daily_policy.csv`, "
            "`cost_sensitivity.csv`, `feature_importance.csv`, "
            "`drift_holdout.csv`, `reference.json`.",
        ]
    )
    if config.row_limit:
        rows.extend(["", f"Run này chỉ dùng {config.row_limit:,} dòng đầu mỗi tập để smoke test."])
    (output / "model_card.md").write_text("\n".join(rows) + "\n")
