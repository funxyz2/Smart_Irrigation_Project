# Lưu ý: tính light intensity từ độ che phủ mây
# Độ che phủ mây (%)	Mô tả thời tiết	Ước lượng ánh sáng (lux)
# 0–10%	    Trời nắng gắt	    ~100,000 lux
# 10–30%	Nắng đẹp	        ~60,000–80,000 lux
# 30–60%	Nắng dịu, có mây	~20,000–50,000 lux
# 60–90%	Mây nhiều, ít nắng	~10,000–20,000 lux
# 90–100%	U ám, mưa	        ~1,000–10,000 lux
# => nội suy đơn giản
# lux = int(100000 * (1 - cloudiness / 100))

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import requests
import os
import logging
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from keras.models import load_model

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load biến môi trường từ file .env
load_dotenv()

app = Flask(__name__)

# Cấu hình ghi log ra file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()  # Ghi ra console (giúp xem dễ trên Render)
    ]
)

model = load_model("deep_model.keras")
scaler = joblib.load("scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY") # or "YOUR_API_KEY_HERE"
DEFAULT_LOCATION = {"lat": 10.762622, "lon": 106.660172}  # TP.HCM

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
            return jsonify({"error": "Thiếu temperature, soil_moisture, water_level, humidity_air hoặc last_watered_hour"}), 400

        logging.info(f"📥 Nhận từ ESP32: temp={temperature}, soil={soil_moisture}, water={water_level}, humidity={humidity_air}, last_watered_hour={last_watered_hour}")

        weather_data = get_weather_data()
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
        X_input = pd.DataFrame([full_data])
        X_input.fillna(0, inplace=True)

        # Chuẩn hóa đầu vào
        input_scaled = scaler.transform(X_input)

        # Dự đoán
        prediction = model.predict(input_scaled)

        # Giải chuẩn hóa đầu ra
        predicted_ml = y_scaler.inverse_transform(prediction.reshape(-1, 1))
        result = float(predicted_ml[0][0])

        logging.info(f"✅ Dự đoán: {result:.2f} ml nước")
        print(f"✅ Dự đoán: {result:.2f} ml nước")

        return jsonify(result)

    except Exception as e:
        print("❌ Lỗi server:", e)
        return jsonify({"error": str(e)}), 500


def get_weather_data():
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={DEFAULT_LOCATION['lat']}&lon={DEFAULT_LOCATION['lon']}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception("Không gọi được OpenWeather")

        data = response.json()
        cloudiness = data.get("clouds", {}).get("all", 50)
        estimated_lux = int(100000 * (1 - cloudiness / 100))

        return {
            "light_intensity": estimated_lux,
            "time_of_day": get_time_of_day(),
            "rain_prediction": 1 if "rain" in data else 0
        }

    except Exception as e:
        logging.warning(f"⚠️ Không lấy được dữ liệu thời tiết: {e}")
        print("⚠️ Không lấy được dữ liệu thời tiết:", e)
        print("Vì không thể connect được API OpenWeather, ta giả định trời có mây, không mưa và lấy thời gian được lưu trên mạch làm chuẩn.")
        return {
            "light_intensity": 20000, # giả định có mây
            "time_of_day": get_time_of_day(),
            "rain_prediction": 0
        }

def get_time_of_day():
    now = datetime.utcnow().hour + 7  # giờ VN
    return now % 24

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)