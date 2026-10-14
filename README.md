# SULFATE-LIFE AI

Ứng dụng Streamlit phục vụ đồ án nghiên cứu: học máy hỗ trợ phân tích suy giảm tính chất bê tông trong môi trường sulfat.

## Chạy trên máy Windows

1. Cài Python 3.10 trở lên.
2. Giải nén thư mục.
3. Mở Command Prompt/Terminal tại thư mục dự án.
4. Cài thư viện:

   `pip install -r requirements.txt`

5. Chạy ứng dụng:

   `streamlit run app.py`

6. Trình duyệt sẽ mở ứng dụng. Nếu không tự mở, truy cập địa chỉ hiển thị trong Terminal (thường là http://localhost:8501).

## Cách dùng

- Vào **Tạo mẫu dữ liệu** để tải CSV mẫu.
- Thay dữ liệu minh họa bằng dữ liệu thí nghiệm đã kiểm chứng.
- Vào **Huấn luyện & đánh giá**, tải CSV hoặc Excel, chọn biến đầu vào, biến mục tiêu và thuật toán.
- Xem MAE, RMSE, R² và biểu đồ thực tế so với dự đoán.
- Vào **Dự đoán từ mô hình** để thử điều kiện mới trong phạm vi dữ liệu huấn luyện.

## Tên cột hỗ trợ

Biến đầu vào:
- `silica_fume_pct`
- `water_binder_ratio`
- `initial_strength_mpa`
- `sulfate_concentration_pct`
- `temperature_c`
- `exposure_days`
- `wet_dry_cycles`
- `curing_days`
- `relative_humidity_pct`

Biến đầu ra có thể dùng:
- `remaining_strength_mpa`
- `strength_loss_pct`
- `expansion_pct`
- `mass_change_pct`

## Cảnh báo khoa học

- CSV mẫu có số liệu minh họa, không phải kết quả thí nghiệm thực. Không dùng dữ liệu minh họa để đưa ra kết luận khoa học.
- Mô hình chỉ dự đoán biến mục tiêu đã huấn luyện; nó không tự suy ra tuổi thọ công trình nếu không có mô hình thời gian và tiêu chí giới hạn đã được xác lập.
- Kết quả chỉ phục vụ học tập/nghiên cứu, không thay thế tính toán thiết kế, đánh giá hiện trường hoặc thẩm tra kỹ sư.
- Các dữ liệu phải cùng định nghĩa, đơn vị và điều kiện thử nghiệm phù hợp.
