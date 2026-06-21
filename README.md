# ESP32 Smart Irrigation System with Deep Learning

An intelligent plant irrigation system combining ESP32 hardware with a Flask server integrated with a deep learning model to optimize water usage based on real-time environmental conditions.

## System Overview

- **Hardware**: ESP32 connected to sensors (soil moisture, DHT22 temperature/humidity, water level) and a water pump
- **Backend**: Flask server running an ML model that predicts the optimal watering amount
- **Control**: Blynk mobile app for remote monitoring and control

## How It Works

1. ESP32 collects data from sensors
2. Data is sent to the Flask server via API
3. The server enriches sensor data with weather data from OpenWeatherMap
4. The ML model predicts the precise amount of water needed (ml)
5. ESP32 controls the pump based on the prediction
6. Users can monitor and intervene through the Blynk app

## Key Features

- Automatically adjusts watering amount based on multiple factors (soil moisture, temperature, rain forecast, etc.)
- Flexible auto/manual mode switching
- Visual monitoring dashboard via Blynk app
- Deep learning model improves watering decision accuracy
- Low water level and system error alerts

## Requirements

- **ESP32** with sensors: soil moisture, DHT22, water level, water pump
- **Flask server** with Python 3.11 and required libraries
- **Blynk account** and **OpenWeatherMap API key**

## Authors

- [thomasNguyen-196](https://github.com/thomasNguyen-196)
- [funxyz2](https://github.com/funxyz2)