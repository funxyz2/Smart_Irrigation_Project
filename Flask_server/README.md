# Flask-based-server-for-Smart-Irrigation-System

## Mô tả dự án

🌿 Đây là một phần trong hệ thống tưới cây thông minh sử dụng vi điều khiển ESP32. Dự án bao gồm một API server viết bằng Flask, đóng vai trò như một mô-đun trung gian để ESP32 gửi dữ liệu cảm biến và nhận về quyết định tưới nước.

🚿 Máy chủ này nhận các thông số từ ESP32 như nhiệt độ, độ ẩm đất, độ ẩm không khí, mực nước trong bồn, thời điểm tưới gần nhất... Sau đó, nó kết hợp những dữ liệu này với thông tin thời tiết lấy từ OpenWeatherMap (độ che phủ mây, khả năng có mưa, thời điểm trong ngày, v.v.). Dữ liệu tổng hợp sẽ được chuẩn hóa và đưa vào mô hình AI (deep learning) đã được huấn luyện sẵn. Kết quả dự đoán là lượng nước (ml) cần tưới cho cây tại thời điểm đó.

💡 Mục tiêu của hệ thống là tối ưu hóa lượng nước tưới dựa theo điều kiện môi trường thực tế và dự báo thời tiết, giúp tiết kiệm tài nguyên và tự động hóa quá trình chăm sóc cây trồng.

## Các thông tin đầu vào cho mô hình

| Trường              | Ý nghĩa                                             |
| ------------------- | --------------------------------------------------- |
| `temperature`       | Nhiệt độ không khí (°C)                             |
| `soil_moisture`     | Độ ẩm đất (%)                                       |
| `water_level`       | Mức nước bồn chứa (%)                               |
| `humidity_air`      | Độ ẩm không khí (%)                                 |
| `last_watered_hour` | Giờ tưới lần cuối (24h, múi giờ Việt Nam)           |
| `cloudiness`        | Độ che phủ mây (từ OpenWeather API)                 |
| `rain_expected`     | Có mưa hay không (bool, từ trường "rain" trong API) |
| `lux`               | Cường độ ánh sáng (suy ra từ độ che phủ mây)        |
| `hour_now`          | Giờ hiện tại (24h, múi giờ Việt Nam)                |

## Luồng hoạt động

1. ESP32 gửi dữ liệu cảm biến qua POST request tới endpoint `/predict`
2. Server nhận và log dữ liệu
3. Gọi API OpenWeather để lấy dữ liệu thời tiết:

   * Lấy độ che phủ mây → tính lux: `lux = int(100000 * (1 - cloudiness / 100))`
   * Kiểm tra có trường "rain" trong JSON không để xác định có mưa không
4. Kết hợp dữ liệu sensor + thời tiết, chuẩn hóa, đưa vào mô hình AI
5. Mô hình dự đoán số ml nước cần tưới
6. Server ghi log và trả về cho ESP32

## Mô tả chi tiết

* Framework: **Flask**
* Mô hình: **Keras deep learning model** (`.keras`)
* Dữ liệu chuẩn hóa: **StandardScaler (pickle)**
* API thời tiết: **OpenWeatherMap**
* Gửi cảnh báo khi lỗi: **Blynk Notify**

## Yêu cầu môi trường

* Python **3.11**

* Tạo môi trường ảo:

  ```bash
  python -m venv venv
  source venv/bin/activate  # trên macOS/Linux
  venv\Scripts\activate     # trên Windows
  ```

* Cài thư viện:

  ```bash
  pip install -r requirements.txt
  ```

* Tạo file `.env` chứa các biến môi trường:

  Nội dung file `.env`:

  ```env
  OPENWEATHER_API_KEY=your_openweather_api_key
  BLYNK_AUTH_TOKEN=your_blynk_token
  ```

  **Cách tạo file `.env`:**

  * **Trên macOS/Linux**:

    ```bash
    touch .env
    nano .env  # hoặc dùng bất kỳ trình soạn thảo nào bạn thích
    ```

  * **Trên Windows**:
    Mở Command Prompt hoặc PowerShell:

    ```powershell
    echo OPENWEATHER_API_KEY=your_openweather_api_key > .env
    echo BLYNK_AUTH_TOKEN=your_blynk_token >> .env
    ```

    Hoặc dùng Notepad:

    * Mở Notepad
    * Nhập nội dung như trên
    * Chọn "Save As", đặt tên là `.env`, chọn "All Files" trong phần "Save as type", và lưu vào thư mục dự án.

---

## Cách chạy

* **Trên macOS/Linux**:

  ```bash
  source venv/bin/activate
  python app.py
  ```

* **Trên Windows**:

  ```bash
  venv\Scripts\activate
  python app.py
  ```

* **Bằng Docker** *(Khuyến nghị – cách triển khai chính của nhóm)*:

  1. **Tạo file `Dockerfile`** với nội dung sau:

     ```Dockerfile
     # Use a lightweight Python image with TensorFlow support
     FROM python:3.11-slim

     ENV PYTHONUNBUFFERED=1

     WORKDIR /app

     RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         && rm -rf /var/lib/apt/lists/*

     COPY requirements.txt .
     RUN pip install --no-cache-dir -r requirements.txt

     COPY . .

     EXPOSE 5050

     CMD ["python", "app.py"]
     ```

  2. **Tạo file `docker-compose.yml`** với nội dung sau:

     ```yaml
     version: '3.8'

     services:
       app:
         build: .
         container_name: my_dl_server
         ports:
           - "5050:5050"
         env_file:
           - .env
         restart: unless-stopped
         volumes:
           - .:/app
     ```

  3. **Cấu trúc thư mục triển khai Docker:** *(Tất cả các file nằm chung một thư mục, riêng mô hình nằm trong thư mục `models/`)*

     ```plaintext
     /project-folder
     ├── app.py
     ├── server.log
     ├── requirements.txt
     ├── .env
     ├── Dockerfile
     ├── docker-compose.yml
     ├── models
     │   ├── deep_model.keras
     │   ├── scaler.pkl
     │   └── y_scaler.pkl
     └── (các file mã nguồn và tài nguyên khác)
     ```

  4. **Chạy ứng dụng bằng Docker Compose:**

     ```bash
     docker-compose up --build
     ```

     Lần sau nếu không thay đổi code, bạn chỉ cần chạy:

     ```bash
     docker-compose up
     ```
   
## Endpoint

**POST** `/predict`

### Payload JSON mẫu:

```json
{
  "temperature": 30,
  "soil_moisture": 40,
  "water_level": 75,
  "humidity_air": 60,
  "last_watered_hour": 13
}
```

### Kết quả trả về:

```json
180  // nghĩa là cần tưới 180ml nước
```

> Bạn cũng có thể thay đổi để trả về kiểu JSON như sau nếu muốn mở rộng:
>
> ```json
> { "water_amount": 180 }
> ```

## Ghi log

* Log được ghi vào file `server.log`
* Mỗi log có timestamp theo múi giờ GMT+7

## Ghi chú

* Trong trường hợp không lấy được dữ liệu thời tiết, API trả về `-1`
* Server sẽ tự động retry OpenWeather API tối đa 3 lần nếu gặp lỗi

---

🌿 *Bạn có thể dùng Postman hoặc ESP32 để gửi test payload tới server.*

