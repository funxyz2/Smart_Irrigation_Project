# Lưu ý: tính light intensity từ độ che phủ mây
# Độ che phủ mây (%)	Mô tả thời tiết	    Ước lượng ánh sáng (lux)
# 0–10%	                Trời nắng gắt	    ~100,000 lux
# 10–30%	            Nắng đẹp	        ~60,000–80,000 lux
# 30–60%	            Nắng dịu, có mây	~20,000–50,000 lux
# 60–90%	            Mây nhiều, ít nắng	~10,000–20,000 lux
# 90–100%	            U ám, mưa	        ~1,000–10,000 lux
# => nội suy đơn giản
# lux = int(100000 * (1 - cloudiness / 100))

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import requests
import os
import logging
import time
from datetime import datetime
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from keras.models import load_model
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import quote_plus


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load biến môi trường từ file .env
load_dotenv()

app = Flask(__name__)

# Cấu hình ghi log ra file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
     datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ],
)
logging.Formatter.converter = lambda *args: time.gmtime(time.time() + 7*3600)

# Kết nối AI model trong server
model = load_model("models/deep_model.keras")
scaler = joblib.load("models/scaler.pkl")
y_scaler = joblib.load("models/y_scaler.pkl")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY") # or "YOUR_API_KEY_HERE"
DEFAULT_LOCATION = {"lat": 10.762622, "lon": 106.660172}  # TP.HCM, VietNam
BLYNK_AUTH_TOKEN = os.getenv("BLYNK_AUTH_TOKEN")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        # Nhận dữ liệu từ ESP32
        temperature = data.get("temperature")
        soil_moisture = data.get("soil_moisture")
        water_level = data.get("water_level")
        humidity_air = data.get("humidity_air")
        # dùng int vì dataset được train với last_watered_hour là int
        last_watered_hour = int(data.get("last_watered_hour"))  # epoch hoặc ISO string

        if None in (temperature, soil_moisture, water_level, humidity_air, last_watered_hour):
            logging.warning(f"Dữ liệu thiếu: {data}")
            blynk_warning(f"Dữ liệu thiếu: {data}")
            return jsonify({"error": "Thiếu temperature, soil_moisture, water_level, humidity_air hoặc last_watered_hour"}), 400

        logging.info(f"📥 Nhận từ ESP32: temp={temperature}, soil={soil_moisture}, water={water_level}, humidity={humidity_air}, last_watered_hour={last_watered_hour}")

        weather_data = get_weather_data()
        if isinstance(weather_data, str) and weather_data == "-1":
            logging.warning("⚠️ Dự báo thời tiết không khả dụng. Trả về -1ml.")
            blynk_warning("⚠️ Dự báo thời tiết không khả dụng. Trả về -1ml.")
            return str(-1)

        logging.info(f"🌤 Dữ liệu thời tiết: {weather_data}")

        full_data = {
            "temperature": temperature,
            "soil_moisture": soil_moisture,
            "water_level": water_level,
            "humidity_air": humidity_air,
            "light_intensity": weather_data.get("light_intensity"),
            "time_of_day": weather_data.get("time_of_day"),
            "rain_prediction": weather_data.get("rain_prediction"),
            "last_watered_hour": last_watered_hour
        }

        feature_order = [
            "temperature",
            "soil_moisture",
            "water_level",
            "humidity_air",
            "light_intensity",
            "time_of_day",
            "rain_prediction",
            "last_watered_hour"
        ]

        """
        Xử lý dữ liệu
        Gửi đến model để dự đoán
        Trả về kết quả cho ESP32
        """
        # Đảm bảo đúng thứ tự cột
        X_input = pd.DataFrame([full_data])[feature_order]
        X_input.fillna(0, inplace=True)

        # Chuẩn hóa đầu vào
        input_scaled = scaler.transform(X_input)

        # Dự đoán
        prediction = model.predict(input_scaled)

        # Giải chuẩn hóa đầu ra
        predicted_ml = y_scaler.inverse_transform(prediction.reshape(-1, 1))
        result = float(predicted_ml[0][0])
        rounded_result = abs(int(round(result)))  # Làm tròn và lấy trị tuyệt đối

        
        logging.info(f"✅ Dự đoán: {rounded_result} ml nước")
        print(f"✅ Dự đoán: {rounded_result} ml nước")

        return jsonify(rounded_result)

    except Exception as e:
        print("❌ Lỗi server:", e)
        logging.error(f"❌ Lỗi server: {str(e)}")
        return jsonify({"error": str(e)}), 500

def get_weather_data():
    try:
        # Retry API 3 lần nếu gặp lỗi
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5)
        session.mount('http://', HTTPAdapter(max_retries=retries))

        url = f"http://api.openweathermap.org/data/2.5/weather?lat={DEFAULT_LOCATION['lat']}&lon={DEFAULT_LOCATION['lon']}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        
        if response.status_code != 200:
            raise Exception(f"Mã lỗi {response.status_code}: {response.text}")

        data = response.json()
        cloudiness = data.get("clouds", {}).get("all", 50)
        estimated_lux = int(100000 * (1 - cloudiness / 100))

        return {
            "light_intensity": estimated_lux,
            "time_of_day": get_time_of_day(),
            "rain_prediction": 1 if "rain" in data else 0
        }

    except Exception as e:
        logging.error(f"❌ Không lấy được dữ liệu thời tiết, trả về -1ml, lỗi: {e}")
        blynk_warning("⚠️ Không lấy được thời tiết! Vui lòng kiểm tra.")
        return str(-1)

# Gửi message cảnh báo (chuỗi string) đến pin ảo V9 của Blynk
def blynk_warning(message):
    encoded_msg = quote_plus(message)
    url = f"https://blynk.cloud/external/api/update?token={BLYNK_AUTH_TOKEN}&V9={encoded_msg}"
    response = requests.get(url, timeout=5)
    if response.status_code != 200:
        raise Exception(f"Blynk V9 update failed: {response.status_code}")

def get_time_of_day():
    now_utc = datetime.utcnow()
    now_vn = now_utc + timedelta(hours=7)
    return now_vn.hour

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)