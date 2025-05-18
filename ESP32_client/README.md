# 🌱 SMART IRRIGATION SYSTEM with ESP32

Hệ thống tưới tiêu thông minh sử dụng ESP32, cảm biến độ ẩm đất, cảm biến nhiệt độ/độ ẩm DHT22, cảm biến mực nước và tích hợp điều khiển từ xa qua ứng dụng Blynk. Hệ thống còn tích hợp mô hình học máy (ML) từ server Flask để đưa ra lượng nước tưới chính xác cho cây trồng.

---

## 🚀 Tính năng

* Đo độ ẩm đất, nhiệt độ, độ ẩm không khí và mực nước tự động
* Gửi dữ liệu lên server Flask định kỳ để nhận dự đoán lượng nước tưới
* Tưới cây tự động dựa trên kết quả mô hình ML
* Chế độ thủ công: Người dùng có thể bật/tắt bơm từ app Blynk
* Giao diện giám sát và điều khiển trên ứng dụng **Blynk**

---

## 🔧 Phần cứng cần thiết

| Thiết bị                    | Mô tả                         |
| --------------------------- | ----------------------------- |
| ESP32                       | Vi điều khiển chính           |
| Cảm biến độ ẩm đất (Analog) | Cắm vào chân GPIO33           |
| Cảm biến mực nước (Analog)  | Cắm vào chân GPIO34           |
| Cảm biến DHT22              | Nhiệt độ và độ ẩm (GPIO32)    |
| Rơ-le + Máy bơm mini        | Điều khiển tưới tiêu (GPIO23) |

---

## 🌐 Kết nối Blynk

* **Template ID**: `...`
* **Template Name**: `SMART IRRIGATION ESP32`
* **Auth Token**: `...`
* **Các Virtual Pins sử dụng**:

| VPin | Chức năng                     |
| ---- | ----------------------------- |
| V0   | Nhiệt độ (°C)                 |
| V1   | Độ ẩm không khí (%)           |
| V2   | Độ ẩm đất (%)                 |
| V3   | Mực nước (%)                  |
| V4   | Bật/tắt bơm                   |
| V5   | Chuyển đổi chế độ Auto/Manual |
| V7   | Kết quả dự đoán ML (ml nước)  |

---

## 🧠 Server học máy

* **API URL**: `https://dl_api.tung196.id.vn/predict`
* Gửi dữ liệu qua HTTP POST dạng JSON:

```json
{
  "temperature": 30.5,
  "soil_moisture": 45,
  "water_level": 1,
  "humidity_air": 60,
  "last_watered_hour": 5
}
```

* Nhận lại phản hồi là lượng nước cần tưới (ml), ví dụ: `"250.0"`

---

## ⚙️ Cài đặt & Upload mã

1. Cài đặt Arduino IDE và thêm ESP32 board: [ESP32 Board Manager](https://github.com/espressif/arduino-esp32)
2. Cài các thư viện:

   * `DHT sensor library by Adafruit`
   * `Blynk`
   * `HTTPClient` (có sẵn trong ESP32)
3. Nạp mã vào ESP32
4. Kiểm tra kết nối WiFi và đăng nhập ứng dụng Blynk

---

## 📝 Lưu ý

* Server Flask cần luôn hoạt động để mô hình ML có thể dự đoán.
* Mực nước quá thấp (`< 20%`) thì hệ thống không tưới (đảm bảo an toàn cho máy bơm).
* Trong chế độ Auto, người dùng không thể điều khiển bơm thủ công.
* Bơm được điều khiển dựa trên thời gian tương ứng với lượng nước tính toán (`ml / 10ml/s`).

---

## 📸 Giao diện Blynk đề xuất

| Widget | Loại | Virtual Pin     | Ghi chú |
| ------ | ---- | --------------- | ------- |
| Gauge  | V0   | Temperature     |         |
| Gauge  | V1   | Humidity        |         |
| Gauge  | V2   | Soil Moisture   |         |
| Gauge  | V3   | Water Level     |         |
| Switch | V4   | Bơm tay         |         |
| Switch | V5   | Auto/Manual     |         |
| Label  | V7   | Kết quả ML (ml) |         |

---

## 👤 Tác giả

* 👤 [thomasNguyen-196](https://github.com/thomasNguyen-196)
* 👤 [funxyz2](https://github.com/funxyz2)

---

## 🔗 Tham khảo

* 🛒 Mua linh kiện tại [Hshop](https://hshop.vn/)
* 📘 Tài liệu và hướng dẫn về IoT nói chung: [Arduino.vn](http://arduino.vn/)
* 🎓 Học kiến thức về Machine Learning/Deep Learning: [Coursera](https://www.coursera.org/) (các khóa học miễn phí chất lượng cao)