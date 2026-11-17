# Fraud Detection

Pipeline data science cho IEEE-CIS, bám theo [bài toán kinh doanh](reports/01_problem_definition.md): chấm điểm rủi ro và review tối đa 1% giao dịch mỗi ngày.

[Kết quả chạy toàn bộ dữ liệu](reports/02_results.md): holdout AP 0,5384, ROC-AUC 0,8966; review precision 83,43% tại tỷ lệ review 0,784%.

## Analytical Notebooks

The notebooks present the analysis as a linear research workflow, with English descriptions, tables and figures from the complete dataset.

| Notebook | Analysis |
| --- | --- |
| [01 · Data Quality and EDA](notebooks/01_data_quality_eda.ipynb) | Business assumptions, source provenance, missingness, class imbalance, temporal variation and group profiles |
| [02 · Features and Temporal Partitions](notebooks/02_features_temporal_split.ipynb) | Chronological evaluation, transaction-level features, fitted encoding and leakage boundaries |
| [03 · Training and Calibration](notebooks/03_training_calibration.ipynb) | Direct candidate fitting, selected-model refit, sigmoid calibration, threshold selection and holdout evaluation |
| [04 · Evaluation and Inference](notebooks/04_evaluation_inference.ipynb) | Baselines, PR/ROC, online review allocation, batch state, expected costs, drift and unlabeled inference |

Notebooks 01–02 analyze the prepared data and CLI artifacts. Notebook 03 trains directly on the complete dataset and stores its artifact in `artifacts/notebook/full`. Notebook 04 evaluates that artifact and generates its test predictions and submission. Model training follows the cell sequence without an execution-mode flag.

`make notebooks` executes the complete sequence and saves cell outputs, including full model training and inference.

## Chạy project

Python 3.11 trở lên, CPU, khoảng 4–6 GB RAM trống cho cấu hình đầy đủ.

```bash
make install
make test
make smoke
make run
```

`make smoke` chạy 50.000 dòng đầu của mỗi tập, lưu riêng tại `artifacts/smoke`. `make run` chạy toàn bộ dữ liệu, lưu tại `artifacts/full`. Hai chế độ không dùng chung dữ liệu đã xử lý hoặc mô hình. Thời gian chạy phụ thuộc CPU và dung lượng RAM.

Đặt các file gốc trong `data/raw/`:

```text
train_transaction.csv
train_identity.csv
test_transaction.csv
test_identity.csv
sample_submission.csv
```

Chạy từng bước:

```bash
.venv/bin/python -m fraud_detection prepare --config configs/default.yaml
.venv/bin/python -m fraud_detection eda --config configs/default.yaml
.venv/bin/python -m fraud_detection train --config configs/default.yaml
.venv/bin/python -m fraud_detection predict --config configs/default.yaml
```

## Quy trình

1. Kiểm tra schema, khóa trùng, nhãn, thời gian và số tiền; chuẩn hóa `id-XX` thành `id_XX`; left join identity theo TransactionID. Đọc CSV theo chunk, lưu Parquet và manifest SHA-256; không sửa dữ liệu raw.
2. EDA: tỷ lệ fraud, lưu lượng theo ngày, phân phối số tiền, missing values, cardinality, cột hằng và biểu đồ.
3. Chia các ngày theo thời gian: train khoảng 60%, tuning 15%, calibration 10%, holdout 15%. Tỷ lệ thực tế có thể khác nhẹ vì không cắt ngang ngày. Mỗi tập phải có cả hai lớp.
4. Tạo feature số tiền, giờ/ngày tương đối, missing count, email match. Category vocabulary, frequency encoding và loại cột hằng chỉ học trên tập fit. Category mới được LightGBM xử lý như missing; frequency mới bằng 0. Không dùng TransactionID hoặc thời gian tuyệt đối làm feature.
5. So sánh cấu hình LightGBM bằng average precision trên tuning, dùng early stopping theo cùng metric. Refit train + tuning với số cây đã chọn. Có baseline xác suất cố định và logistic regression dùng nhóm feature nhỏ để đối chiếu.
6. Fit sigmoid calibration và chọn ngưỡng tối đa hóa tiết kiệm kỳ vọng trên calibration. Khóa mô hình, calibrator và ngưỡng trước khi xem holdout.
7. Đánh giá average precision, ROC-AUC, log loss, Brier score, calibration, precision/recall trong review, chi phí và độ nhạy giả định. Báo cáo feature importance và drift.
8. Dự đoán test theo chunk, xuất xác suất, hành động và submission đúng thứ tự sample_submission. Không gửi submission lên dịch vụ bên ngoài.

## Review không dùng tương lai

Sắp xếp giao dịch theo `(TransactionDT, TransactionID)`. Sau `n` giao dịch của ngày, số review không vượt `floor(0.01 × n)`. Giao dịch chỉ được chọn nếu còn quota và xác suất đạt ngưỡng đã chọn trên calibration. Quota reset vào ngày tương đối tiếp theo.

Quy tắc bảo đảm giới hạn 1% ở mọi thời điểm, nhưng có thể bỏ lỡ fraud đến sớm hoặc để trống quota. Một ngày dưới 100 giao dịch có quota bằng 0. `daily_top_k` trong báo cáo chỉ là benchmark hồi cứu, cần biết toàn bộ giao dịch trong ngày nên không dùng làm chính sách vận hành.

Với nhiều batch, dùng cùng file trạng thái và đưa batch theo thứ tự thời gian:

```bash
.venv/bin/python -m fraud_detection predict \
  --config configs/default.yaml \
  --input data/processed/new_batch.parquet \
  --state artifacts/full/review_state.json
```

Input phải là bảng transaction đã join identity, có schema như dữ liệu test đã chuẩn bị và không chứa `isFraud`. Batch sau phải có khóa `(TransactionDT, TransactionID)` lớn hơn batch trước. Một luồng xử lý sở hữu một state; chưa hỗ trợ nhiều worker đồng thời. Chạy lại cùng batch với state đã cập nhật sẽ bị từ chối. Mỗi lần predict ghi đè các file dự đoán trong output_dir; lưu riêng các output cần giữ. Không truyền `--state` thì quota bắt đầu lại từ đầu, phù hợp một file test độc lập.

## Kết quả sinh ra

| Đường dẫn trong output_dir | Nội dung |
| --- | --- |
| `model_card.md` | Báo cáo tiếng Việt, kết quả, giả định và giới hạn |
| `eda/` | Tổng quan dữ liệu, profile cột, thống kê ngày, biểu đồ |
| `model.joblib` | Feature builder, LightGBM, calibrator và chính sách |
| `run.json` | Config, phiên bản thư viện, hash dữ liệu và model |
| `splits.json`, `candidate_scores.csv`, `selection.json` | Chia tập và lựa chọn cấu hình |
| `metrics.json`, `evaluation.png` | Baseline, chỉ số holdout và biểu đồ |
| `holdout_predictions.parquet`, `daily_policy.csv` | Điểm số và quota thực tế trên holdout |
| `threshold_search.csv`, `cost_sensitivity.csv` | Tìm ngưỡng và phân tích giả định chi phí |
| `feature_importance.csv` | Importance theo split gain |
| `reference.json`, `drift_holdout.csv`, `drift_test.csv` | Phân phối tham chiếu và PSI |
| `predictions.csv` | TransactionID, thời gian, risk_score, manual_review, action |
| `submission.csv` | TransactionID và xác suất isFraud |

`model.joblib` là artifact Python nội bộ; chỉ load artifact từ nguồn tin cậy. `requirements-lock.txt` ghi lại môi trường Python 3.14.4 đã kiểm thử; `requirements.txt` khai báo khoảng phiên bản hỗ trợ cho các bản Python khác.

## Cấu trúc code

```text
configs/                  Cấu hình full và smoke
notebooks/                Bốn notebook phân tích, training và đánh giá có output
scripts/run_notebooks.py  Thực thi notebook bằng môi trường Python đang dùng
src/fraud_detection/
  config.py               Cấu hình và validation
  data.py                 Kiểm tra dữ liệu, join, Parquet, manifest
  features.py             Feature engineering và encoding
  split.py                Chia tập theo ngày
  modeling.py             Baseline, LightGBM, model artifact
  policy.py               Quota online và mô phỏng chi phí
  metrics.py              Chỉ số phân loại
  monitoring.py           Tham chiếu phân phối và PSI
  reporting.py            EDA, biểu đồ, model card
  pipeline.py             Điều phối huấn luyện và đánh giá
  inference.py            Batch prediction và submission
  cli.py                  Command line
tests/                    Kiểm tra leakage, quota, schema và toàn pipeline
```

## Giả định cần giữ rõ

- Tổn thất mỗi fraud là $100, review $1, sensitivity 80%; đây là giả định trong tài liệu bài toán. Tiết kiệm báo cáo là kỳ vọng mô phỏng, chưa phải hiệu quả vận hành thực tế.
- Chỉ Approve và Manual Review, chưa automatic blocking nên không áp chi phí chặn nhầm $5.
- TransactionDT không có mốc lịch; các ngày/giờ là tương đối. Không suy diễn múi giờ hay thứ trong tuần thực.
- C/D/V và identity là các feature ẩn danh. Cần xác nhận tính sẵn có tại thời điểm giao dịch, độ trễ nhãn và thời gian xử lý review trước khi triển khai.
- Mô hình cuối fit trên train + tuning; calibration và holdout được giữ riêng. Không tự động refit toàn bộ nhãn sau đánh giá vì sẽ thay đổi calibration và ngưỡng đã kiểm chứng.
- PSI lấy calibration làm tham chiếu; mức 0.2 gợi ý điều tra, không tự động retrain. Khi có nhãn mới, cần đo lại recall, chi phí và calibration.

Kiểm tra code:

```bash
make lint
make test
```
