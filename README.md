# README tổng hợp

# Dự án hệ thống tưới cây thông minh ESP32 kết hợp DL

Hệ thống tưới cây thông minh kết hợp ESP32 với máy chủ Flask tích hợp mô hình học máy để tối ưu hóa lượng nước tưới dựa trên điều kiện môi trường thực tế.

## Tổng quan hệ thống

- **Phần cứng**: ESP32 kết nối với các cảm biến (độ ẩm đất, nhiệt độ/độ ẩm DHT22, mực nước) và máy bơm
- **Backend**: Máy chủ Flask chứa mô hình ML dự đoán lượng nước tưới tối ưu
- **Điều khiển**: Giao diện ứng dụng Blynk cho phép giám sát và điều khiển từ xa

## Cơ chế hoạt động

1. ESP32 thu thập dữ liệu từ các cảm biến
2. Dữ liệu được gửi đến máy chủ Flask qua API
3. Máy chủ kết hợp dữ liệu cảm biến với dữ liệu thời tiết từ OpenWeatherMap
4. Mô hình ML dự đoán lượng nước tưới chính xác (ml)
5. ESP32 điều khiển máy bơm theo kết quả dự đoán
6. Người dùng có thể theo dõi và can thiệp qua ứng dụng Blynk

## Các tính năng nổi bật

- Tự động điều chỉnh lượng nước tưới dựa trên nhiều yếu tố (độ ẩm đất, nhiệt độ, dự báo mưa...)
- Chế độ tự động/thủ công linh hoạt
- Giao diện giám sát trực quan trên ứng dụng Blynk
- Mô hình DL cải thiện độ chính xác của quyết định tưới
- Cảnh báo khi mực nước thấp hoặc gặp lỗi hệ thống

## Yêu cầu kỹ thuật

- **ESP32** với các cảm biến: độ ẩm đất, DHT22, mực nước, máy bơm
- **Máy chủ Flask** với Python 3.11 và các thư viện liên quan
- **Tài khoản Blynk** và OpenWeatherMap API

## Tác giả

- [thomasNguyen-196](https://github.com/thomasNguyen-196)
- [funxyz2](https://github.com/funxyz2)