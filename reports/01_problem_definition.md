### Business Problem Statement

**Business Context**

Hệ thống thanh toán xử lý một lượng lớn giao dịch trực tuyến, trong đó một tỷ lệ nhỏ có hành vi gian lận. Các giao dịch gian lận có thể gây tổn thất tài chính, trong khi việc kiểm tra thủ công mọi giao dịch là không khả thi.

**Business Problem**

Đội Risk chỉ có khả năng kiểm tra tối đa 1% tổng số giao dịch mỗi ngày. Cần xây dựng một hệ thống đánh giá rủi ro giúp ưu tiên những giao dịch có khả năng gian lận cao nhất, nhằm giảm thiểu thiệt hại tài chính trong giới hạn nguồn lực.

**Machine Learning Objective**

Xây dựng mô hình phân loại nhị phân, sử dụng thông tin có sẵn tại thời điểm giao dịch để tạo điểm rủi ro gian lận cho từng giao dịch.

**Decision Objective**

Sử dụng điểm rủi ro để xác định các giao dịch cần kiểm tra thủ công, tuân thủ giới hạn tối đa 1% giao dịch mỗi ngày.

**Business Success Criteria**

- Giảm chi phí gian lận kỳ vọng so với việc không kiểm tra giao dịch.
- Ưu tiên được nhiều giao dịch gian lận trong nguồn lực review cho phép.
- Hạn chế chi phí vận hành và tác động tiêu cực đến khách hàng.
- Đảm bảo quy trình đánh giá và lựa chọn giao dịch có thể vận hành mà không sử dụng thông tin tương lai.

**Initial Business Assumptions**

- Chi phí trung bình của một giao dịch gian lận bị bỏ sót: 100 USD.
- Chi phí kiểm tra thủ công: 1 USD/giao dịch.
- Chi phí chặn nhầm giao dịch hợp lệ: 5 USD/giao dịch.
- Tỷ lệ phát hiện gian lận của nhân viên review: 80% (giả định chính).
- Phiên bản đầu tiên áp dụng hai hành động Approve và Manual Review, chưa triển khai automatic blocking.