# SMART IRRIGATION SYSTEM with ESP32

Smart irrigation system using ESP32, soil moisture sensor, DHT22 temperature/humidity sensor, water level sensor, and remote control via Blynk app. The system also integrates an ML model from the Flask server to provide precise watering amounts for plants.

---

## Features

* Automatically measures soil moisture, temperature, air humidity, and water level
* Periodically sends data to the Flask server to receive watering predictions
* Auto-watering based on ML model results
* Manual mode: Users can turn the pump on/off from the Blynk app
* Monitoring and control interface on **Blynk** app

---

## Required Hardware

| Device                      | Description                  |
| --------------------------- | ---------------------------- |
| ESP32                       | Main microcontroller         |
| Soil Moisture Sensor (Analog) | Connected to GPIO33        |
| Water Level Sensor (Analog)   | Connected to GPIO34        |
| DHT22 Sensor                | Temperature & humidity (GPIO32) |
| Relay + Mini Water Pump     | Irrigation control (GPIO23)  |

---

## Blynk Connection

* **Template ID**: `...`
* **Template Name**: `SMART IRRIGATION ESP32`
* **Auth Token**: `...`
* **Virtual Pins Used**:

| VPin | Function                    |
| ---- | --------------------------- |
| V0   | Temperature (°C)            |
| V1   | Air Humidity (%)            |
| V2   | Soil Moisture (%)           |
| V3   | Water Level (%)             |
| V4   | Pump On/Off                 |
| V5   | Auto/Manual Mode Switch     |
| V7   | ML Prediction Result (ml)   |

---

## ML Server

* **API URL Format**: `https://my_api_url.com/predict`
* Sends data via HTTP POST as JSON:

```json
{
  "temperature": 30.5,
  "soil_moisture": 45,
  "water_level": 1,
  "humidity_air": 60,
  "last_watered_hour": 5
}
```

* Receives the watering amount (ml) as response, e.g.: `"250.0"`

---

## Setup & Upload

1. Install Arduino IDE and add ESP32 board: [ESP32 Board Manager](https://github.com/espressif/arduino-esp32)
2. Install libraries:

   * `DHT sensor library by Adafruit`
   * `Blynk`
   * `HTTPClient` (built-in for ESP32)
3. Upload the code to ESP32
4. Check WiFi connection and log into the Blynk app

---

## Notes

* The Flask server must be running for the ML model to make predictions.
* If the water level is too low (`< 20%`), the system will not water (to protect the pump).
* In Auto mode, the user cannot manually control the pump.
* The pump runs for a duration proportional to the calculated water amount (`ml / 10ml/s`).

---

## Recommended Blynk Widget Layout

| Widget | Type   | Virtual Pin     | Notes          |
| ------ | ------ | --------------- | -------------- |
| Gauge  | V0     | Temperature     |                |
| Gauge  | V1     | Humidity        |                |
| Gauge  | V2     | Soil Moisture   |                |
| Gauge  | V3     | Water Level     |                |
| Switch | V4     | Manual Pump     |                |
| Switch | V5     | Auto/Manual     |                |
| Label  | V7     | ML Result (ml)  |                |

---

## References

* Buy components at [Hshop](https://hshop.vn/)
* IoT documentation and tutorials: [Arduino.vn](http://arduino.vn/)
* Machine Learning/Deep Learning courses: [Coursera](https://www.coursera.org/) (high-quality free courses)