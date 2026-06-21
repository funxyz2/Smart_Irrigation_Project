# Flask-based-server-for-Smart-Irrigation-System

## Description

This is part of a smart irrigation system using the ESP32 microcontroller. The project includes a Flask API server that acts as an intermediary module: ESP32 sends sensor data and receives watering decisions.

The server receives parameters from ESP32 such as temperature, soil moisture, air humidity, water tank level, and the last watering time. It then combines this data with weather information from OpenWeatherMap (cloud cover, rain probability, time of day, etc.). The aggregated data is normalized and fed into a pre-trained deep learning model. The predicted output is the amount of water (ml) needed for the plants at that moment.

The goal is to optimize water usage based on actual environmental conditions and weather forecasts, saving resources and automating plant care.

## Model Input Features

| Field               | Description                                        |
| ------------------- | -------------------------------------------------- |
| `temperature`       | Air temperature (°C)                               |
| `soil_moisture`     | Soil moisture (%)                                  |
| `water_level`       | Water tank level (%)                               |
| `humidity_air`      | Air humidity (%)                                   |
| `last_watered_hour` | Last watering hour (24h, Vietnam timezone)          |
| `cloudiness`        | Cloud cover percentage (from OpenWeather API)      |
| `rain_expected`     | Rain expected (bool, from "rain" field in API)     |
| `lux`               | Light intensity (derived from cloud cover)         |
| `hour_now`          | Current hour (24h, Vietnam timezone)               |

## Workflow

1. ESP32 sends sensor data via POST request to endpoint `/predict`
2. Server receives and logs the data
3. Calls OpenWeather API to fetch weather data:

   * Gets cloud cover percentage → calculates lux: `lux = int(100000 * (1 - cloudiness / 100))`
   * Checks for "rain" field in JSON to determine rain probability
4. Combines sensor + weather data, normalizes, runs through the AI model
5. Model predicts the ml of water needed
6. Server logs the result and returns it to ESP32

## Technical Details

* Framework: **Flask**
* Model: **PyTorch deep learning model** (`.pth`)
* Data normalization: **StandardScaler (pickle)**
* Weather API: **OpenWeatherMap**
* Error alerts: **Blynk Notify**

## Environment Setup

* Python **3.11**

* Create a virtual environment:

  ```bash
  python -m venv venv
  source venv/bin/activate  # on macOS/Linux
  venv\Scripts\activate     # on Windows
  ```

* Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

* Create a `.env` file with environment variables:

  `.env` file contents:

  ```env
  OPENWEATHER_API_KEY=your_openweather_api_key
  BLYNK_AUTH_TOKEN=your_blynk_token
  ```

  **How to create the `.env` file:**

  * **On macOS/Linux**:

    ```bash
    touch .env
    nano .env  # or any text editor you prefer
    ```

  * **On Windows**:
    Open Command Prompt or PowerShell:

    ```powershell
    echo OPENWEATHER_API_KEY=your_openweather_api_key > .env
    echo BLYNK_AUTH_TOKEN=your_blynk_token >> .env
    ```

    Or using Notepad:

    * Open Notepad
    * Enter the content as shown above
    * Select "Save As", name it `.env`, choose "All Files" in "Save as type", and save it in the project directory.

---

## Running the Server

* **On macOS/Linux**:

  ```bash
  source venv/bin/activate
  python app.py
  ```

* **On Windows**:

  ```bash
  venv\Scripts\activate
  python app.py
  ```

* **Using Docker** *(Recommended – the team's primary deployment method)*:

  1. **Create a `Dockerfile`** with the following content:

     ```dockerfile
     # Use a lightweight Python image
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

  2. **Create a `docker-compose.yml`** file with the following content:

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

  3. **Docker deployment directory structure:** *(All files in the same directory; model files are in `models/`)*

     ```plaintext
     /project-folder
     ├── app.py
     ├── server.log
     ├── requirements.txt
     ├── .env
     ├── Dockerfile
     ├── docker-compose.yml
     ├── models
     │   ├── deep_model.pth
     │   ├── scaler.pkl
     │   └── y_scaler.pkl
     └── (other source and resource files)
     ```

  4. **Run with Docker Compose:**

     ```bash
     docker-compose up --build
     ```

     On subsequent runs (no code changes), simply:

     ```bash
     docker-compose up
     ```

---

## Endpoint

**POST** `/predict`

### Sample JSON Payload:

```json
{
  "temperature": 30,
  "soil_moisture": 40,
  "water_level": 75,
  "humidity_air": 60,
  "last_watered_hour": 13
}
```

### Response:

```json
180  // means 180ml of water needed
```

> You can also change this to return JSON for extensibility:
>
> ```json
> { "water_amount": 180 }
> ```

## Logging

* Logs are written to `server.log`
* Each log entry has a GMT+7 timestamp

## Notes

* If weather data cannot be retrieved, the API returns `-1`
* The server automatically retries the OpenWeather API up to 3 times on failure

---

*You can use Postman or ESP32 to send test payloads to the server.*