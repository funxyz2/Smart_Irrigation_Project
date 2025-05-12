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
import requests
import os
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler

# Lấy API key từ biến môi trường
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY") or "YOUR_API_KEY_HERE"
DEFAULT_LOCATION = {"lat": 10.762622, "lon": 106.660172}  # TP.HCM

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        # Nhận dữ liệu từ ESP32
        temperature = data.get("temperature")
        soil_moisture = data.get("soil_moisture")
        water_level = data.get("water_level")
        last_watered_timestamp = data.get("last_watered_timestamp")  # epoch hoặc ISO string

        if None in (temperature, soil_moisture, water_level, last_watered_timestamp):
            return jsonify({"error": "Thiếu temperature, soil_moisture, water_level hoặc last_watered_timestamp"}), 400

        last_watered_hour = calculate_last_watered_hour(last_watered_timestamp)
        weather_data = get_weather_data()

        full_data = {
            "temperature": temperature,
            "soil_moisture": soil_moisture,
            "water_level": water_level,
            "humidity_air": weather_data.get("humidity_air"),
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

        input_list = [full_data.get(f) if full_data.get(f) is not None else 0 for f in feature_order]
        input_array = np.array([input_list]) # 2D array cho mô hình

        # Chuẩn hóa nếu cần
        input_scaled = scaler.transform(input_array)

        # Dự đoán
        prediction = model.predict(input_scaled)

        # Đưa về định dạng số thực
        result = float(prediction[0][0])
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
            "humidity_air": data["main"]["humidity"],
            "light_intensity": estimated_lux,
            "time_of_day": get_time_of_day(),
            "rain_prediction": 1 if "rain" in data else 0
        }

    except Exception as e:
        print("⚠️ Không lấy được dữ liệu thời tiết:", e)
        return {
            "humidity_air": None,
            "light_intensity": None,
            "time_of_day": None,
            "rain_prediction": None
        }

def get_time_of_day():
    now = datetime.utcnow().hour + 7  # giờ VN
    return now % 24

# Hàm tính thời gian từ lần tưới cuối
def calculate_last_watered_hour(last_timestamp):
    try:
        if isinstance(last_timestamp, str):  # ISO 8601 string
            last_time = datetime.fromisoformat(last_timestamp)
        else:  # epoch timestamp
            last_time = datetime.fromtimestamp(float(last_timestamp), tz=timezone.utc)
        
        now = datetime.now(timezone.utc)
        diff_hours = (now - last_time).total_seconds() / 3600.0
        return round(diff_hours, 2)
    except Exception as e:
        print("⚠️ Lỗi parse thời gian tưới:", e)
        return None

