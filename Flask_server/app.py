# Note: calculate light intensity from cloud cover
# Cloud cover (%)	   Weather description	     Estimated light (lux)
# 0–10%	               Sunny, bright	         ~100,000 lux
# 10–30%	           Sunny, fair	             ~60,000–80,000 lux
# 30–60%	           Partly cloudy	             ~20,000–50,000 lux
# 60–90%	           Mostly cloudy	             ~10,000–20,000 lux
# 90–100%	           Overcast, rainy	         ~1,000–10,000 lux
# => simple linear interpolation
# lux = int(100000 * (1 - cloudiness / 100))

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import requests
import os
import logging
import time
import torch
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from keras.models import load_model
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import quote_plus

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure logging to file and console
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

# Computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load PyTorch model and scalers
class WaterNet(torch.nn.Module):
    def __init__(self, input_dim):
        super(WaterNet, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(32, 16),
            torch.nn.BatchNorm1d(16),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(16, 8),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.model(x)

# Initialize and load model
model = WaterNet(10).to(device)  # 10 features as used during training
model.load_state_dict(torch.load("models/deep_model.pth", map_location=device))
model.eval()

# Load scalers
scaler = joblib.load("models/scaler.pkl")
y_scaler = joblib.load("models/y_scaler.pkl")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")  # or "YOUR_API_KEY_HERE"
DEFAULT_LOCATION = {"lat": 10.762622, "lon": 106.660172}  # Ho Chi Minh City, Vietnam
BLYNK_AUTH_TOKEN = os.getenv("BLYNK_AUTH_TOKEN")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        # Receive data from ESP32
        temperature = data.get("temperature")
        soil_moisture = data.get("soil_moisture")
        water_level = data.get("water_level")
        humidity_air = data.get("humidity_air")
        # use int because the dataset was trained with last_watered_hour as int
        last_watered_hour = int(data.get("last_watered_hour"))

        if None in (temperature, soil_moisture, water_level, humidity_air, last_watered_hour):
            logging.warning(f"Missing data: {data}")
            blynk_warning(f"Missing data: {data}")
            return jsonify({"error": "Missing temperature, soil_moisture, water_level, humidity_air or last_watered_hour"}), 400

        logging.info(f"Received from ESP32: temp={temperature}, soil={soil_moisture}, water={water_level}, humidity={humidity_air}, last_watered_hour={last_watered_hour}")

        weather_data = get_weather_data()
        if isinstance(weather_data, str) and weather_data == "-1":
            logging.warning("Weather forecast unavailable. Returning -1ml.")
            blynk_warning("Weather forecast unavailable. Returning -1ml.")
            return str(-1)

        logging.info(f"Weather data: {weather_data}")

        # Feature engineering identical to training
        time_of_day = weather_data.get("time_of_day")
        full_data = {
            "temperature": temperature,
            "soil_moisture": soil_moisture,
            "water_level": water_level,
            "humidity_air": humidity_air,
            "light_intensity": weather_data.get("light_intensity"),
            "time_of_day": time_of_day,
            "rain_prediction": weather_data.get("rain_prediction"),
            "last_watered_hour": last_watered_hour,
            # Additional features identical to training
            "time_sin": np.sin(2 * np.pi * time_of_day / 24),
            "time_cos": np.cos(2 * np.pi * time_of_day / 24),
            "hours_since_watered": (last_watered_hour) % 24,
            "drought_index": temperature / (humidity_air + 1) * 10
        }

        # Feature order must exactly match training
        feature_order = [
            "temperature", "soil_moisture", "water_level", "humidity_air",
            "light_intensity", "rain_prediction", "time_sin", "time_cos",
            "hours_since_watered", "drought_index"
        ]

        """
        Process data
        Feed to model for prediction
        Return result to ESP32
        """
        # Create DataFrame and normalize
        X_input = pd.DataFrame([full_data])[feature_order]
        X_input.fillna(0, inplace=True)
        input_scaled = scaler.transform(X_input)

        # Convert to tensor and predict
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
        with torch.no_grad():
            prediction_scaled = model(input_tensor).cpu().numpy()

        # Inverse transform the output
        predicted_ml = y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
        result = float(predicted_ml[0][0])
        rounded_result = abs(int(round(result)))

        logging.info(f"Prediction: {rounded_result} ml water")
        print(f"Prediction: {rounded_result} ml water")

        return jsonify(rounded_result)

    except Exception as e:
        print("Server error:", e)
        logging.error(f"Server error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def get_weather_data():
    try:
        # Retry API 3 times on failure
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5)
        session.mount('http://', HTTPAdapter(max_retries=retries))

        url = f"http://api.openweathermap.org/data/2.5/weather?lat={DEFAULT_LOCATION['lat']}&lon={DEFAULT_LOCATION['lon']}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        
        if response.status_code != 200:
            raise Exception(f"Error code {response.status_code}: {response.text}")

        data = response.json()
        cloudiness = data.get("clouds", {}).get("all", 50)
        estimated_lux = int(100000 * (1 - cloudiness / 100))

        return {
            "light_intensity": estimated_lux,
            "time_of_day": get_time_of_day(),
            "rain_prediction": 1 if "rain" in data else 0
        }

    except Exception as e:
        logging.error(f"Could not fetch weather data, returning -1ml, error: {e}")
        blynk_warning("Could not fetch weather data! Please check.")
        return str(-1)

# Send warning message (string) to Blynk virtual pin V9
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