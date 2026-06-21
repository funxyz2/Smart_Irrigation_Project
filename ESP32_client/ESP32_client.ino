#include <WiFi.h>
#include <HTTPClient.h>
#include "DHT.h"
#include "secrets.h"

#define DHTPIN 32
#define DHTTYPE DHT22
#define WATER_LEVEL_THRESHOLD 20 // Above 20% sets water level sensor to 1 (boolean)
DHT dht(DHTPIN, DHTTYPE);

// Blynk
#define BLYNK_TEMPLATE_ID "TMPL6QX3iHsMo"
#define BLYNK_TEMPLATE_NAME "SMART IRRIGATION ESP32"
#define BLYNK_PRINT Serial

#include <BlynkSimpleEsp32.h>

// Pin 
#define ANALOG_WATER_LEVEL_PIN 34
#define ANALOG_SOIL_MOISTURE_PIN 33
#define DIGITAL_SOIL_MOISTURE_PIN 2
#define PUMP_PIN 23

// Setup
char ssid[] = WIFI_SSID; 
char pass[] = WIFI_PASSWORD; 
char auth[] = BLYNK_AUTH_TOKEN;

const char* serverName = FLASK_SERVER_URL; 

bool isManualMode = false; // Manual mode from Blynk App
bool pumpState = false;
unsigned long pumpStartTime = 0;
unsigned long pumpDuration = 0;
bool isPumpRunning = false;
float mlWaterGlobal = 0; // store watered amount
String lastPredictionStr = ""; // Store last prediction result

BlynkTimer timer;
float last_watered_hour = 0;

struct SensorData {
  int soil_moisture;
  bool water_level;
  float temperature;
  float humidity;
  int digital_soil_value;
};

SensorData data = {};

void readSensors() {
  int maxRetries = 5;
  bool success = false;

  for (int i = 0; i < maxRetries; i++) {
    data.soil_moisture = map(analogRead(ANALOG_SOIL_MOISTURE_PIN), 2500, 4095, 100, 0);
    data.soil_moisture = constrain(data.soil_moisture, 0, 100);
      
    /* FIX
    Ideally water_level would be int (0% -> 100%)
    However the training dataset requires this column to be bool (1/0)
    So we remove that logic and use 1/0 instead
      
    //data.water_level = map(analogRead(ANALOG_WATER_LEVEL_PIN), 0, 4095, 0, 100);
    //data.water_level = constrain(data.water_level, 0, 1);
    */  
    int water_level_percent = map(analogRead(ANALOG_WATER_LEVEL_PIN), 0, 4095, 0, 100);
    data.water_level = (water_level_percent >= WATER_LEVEL_THRESHOLD) ? 1 : 0;
    data.temperature = dht.readTemperature();
    data.humidity = dht.readHumidity();

    if (!isnan(data.temperature) && !isnan(data.humidity)) {
      success = true;

      /*
      Process sensor data, print to Serial, and assign values to corresponding Blynk App variables
      */
      Serial.print("Soil Moisture: "); Serial.println(data.soil_moisture);
      Serial.print("Water level: "); Serial.println(water_level_percent);
      Serial.print("Temperature: "); Serial.println(data.temperature);
      Serial.print("Humidity: "); Serial.println(data.humidity);
      Serial.println("--------------------------------------------");

      Blynk.virtualWrite(V2, data.soil_moisture);
      Blynk.virtualWrite(V3, water_level_percent);
      Blynk.virtualWrite(V0, data.temperature);
      Blynk.virtualWrite(V1, data.humidity);

      break;
    }

    Serial.println("Sensor read error! Retrying...");
    delay(1000); 
  }

  if (!success) {
    Serial.println("Could not read sensors after multiple attempts. Halting program.");
    // Assign special values so sendToServer() stops
    data.temperature = NAN;
    data.humidity = NAN;
  }
}

void setup() {
  Serial.begin(9600);
  Serial.println("Smart Irrigation System started. Please water your plants sufficiently for correct operation.");
  
  // Record the last watering time
  last_watered_hour = millis() / 3600000.0;

  // pinMode
  pinMode (DIGITAL_SOIL_MOISTURE_PIN, INPUT);
  pinMode (ANALOG_WATER_LEVEL_PIN, INPUT);
  pinMode (ANALOG_SOIL_MOISTURE_PIN, INPUT);
  pinMode(PUMP_PIN, OUTPUT);
  digitalWrite(PUMP_PIN, LOW);

  Blynk.begin(auth, ssid, pass);
  dht.begin(); 
  delay(2000); // Wait for sensor stabilization

  timer.setInterval(1000L, readSensors);             // Update Blynk App UI every 1 second
  timer.setInterval(7L * 60L * 60L * 1000L, sendToServer);  // Send to Flask server every 7 hours
}

void loop() {
  Blynk.run();
  timer.run();

  // Turn off pump after sufficient time
  if (isPumpRunning && millis() - pumpStartTime >= pumpDuration) {
    digitalWrite(PUMP_PIN, LOW);
    isPumpRunning = false;
    Serial.println("Pump finished automatically based on ML result.");

    if (!isManualMode) {
    pumpState = false;
    Blynk.virtualWrite(V4, 0);
    }
  }
}

// Handle pump duration delay
void runPumpForML(float ml) {
  pumpDuration = ml / 10.0 * 1000; // ml / flow rate (10ml/s) * 1000ms
  pumpStartTime = millis();
  digitalWrite(PUMP_PIN, HIGH);
  isPumpRunning = true;
}

void sendToServer() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    readSensors();

    // Calculate hours elapsed since last watering
    float current_hour = millis() / 3600000.0;
    int hours_since_last_watering = current_hour - last_watered_hour;
    Serial.print("Hours since last watering: "); Serial.println(hours_since_last_watering);

    String jsonData = "{\"temperature\":" + String(data.temperature) +
                      ",\"soil_moisture\":" + String(data.soil_moisture) +
                      ",\"water_level\":" + String(data.water_level) +
                      ",\"humidity_air\":" + String(data.humidity) +
                      ",\"last_watered_hour\":" + String(hours_since_last_watering) + "}";

    http.setTimeout(3000); // 3 second timeout
    http.begin(serverName);
    http.addHeader("Content-Type", "application/json");

    /*
    Receive Response and process
    */
    int httpResponseCode = http.POST(jsonData);
    delay(200);

    if (httpResponseCode == 200) {
      String response = http.getString();
      Serial.println("Response: " + response);
      lastPredictionStr = response;  // Save response string for V7
      float mlWater = response.toFloat();
      Serial.print("I will pump this much water: "); Serial.print(mlWater); Serial.println("ml");

      // If model requires watering
      if (mlWater > 0) {
        runPumpForML(mlWater);
        // Update last watering time
        last_watered_hour = millis() / 3600000.0;
        mlWaterGlobal = mlWater;
      }
    } else {
      Serial.print("POST failed. Error Code: "); Serial.println(httpResponseCode);
    }

    http.end();
  } else {
    Serial.println("WiFi not connected");
  }
}

// Callback when V4 switch changes state
BLYNK_WRITE(V4) {
  // Block user from switching to manual mode while pump is running
  if (isPumpRunning) {
    Serial.println("Pump is running from ML. Manual control is disabled.");
    Blynk.virtualWrite(V4, pumpState ? 1 : 0); // Reset switch state on app
    return;
  }

  if (isManualMode)
  {
    int value = param.asInt();
    digitalWrite(PUMP_PIN, value == 1 ? HIGH : LOW);
    pumpState = (value == 1);
    Blynk.virtualWrite(V4, pumpState ? 1 : 0);
    Serial.println(pumpState ? "Pump turned ON by user from BlynkApp" : "Pump turned OFF by user from BlynkApp");
  } else {
    Serial.println("PUMP (DE)ACTIVATION IS NOT PERMITTED...");
    // Reset the V4 switch state (synchronize state)
    Blynk.virtualWrite(V4, pumpState ? 1 : 0);
  }
}

// Callback when V5 switch changes state
BLYNK_WRITE(V5) {
  int mode = param.asInt();
  isManualMode = (mode == 1);
  Serial.println(isManualMode ? "INITIATING MANUAL MODE..." : "TURNING OFF MANUAL MODE...");
  
  // Ensure pump is always off when not in use
  if (!isManualMode && pumpState) {
    // If pump is on in manual mode and switching to auto, turn pump off
    digitalWrite(PUMP_PIN, LOW);
    pumpState = false;
    Blynk.virtualWrite(V4, 0);
    Serial.println("Pump OFF: Switched to Auto mode");
  }
}

// Manual API call
BLYNK_WRITE(V6) {
  int value = param.asInt();
  if (value == 1) {
    Serial.println("Manual Predict Triggered from Blynk App!");
    sendToServer();  // Call function to send data to Flask server
    Blynk.virtualWrite(V6, 0); // Reset button to OFF after press

    // Send prediction result to Blynk
    if (lastPredictionStr == "") {
      Serial.println("No prediction result available.");
      return;
    }
    String result = lastPredictionStr + " ml";
    Serial.print("Sending last prediction string to Blynk: ");
    Serial.println(result);

    Blynk.virtualWrite(V7, result);
  }
}

// Blynk button to reset ESP32 software
BLYNK_WRITE(V8) {
  int value = param.asInt();
  if (value == 1) {
    Serial.println("User requested device restart...");
    ESP.restart();
  }
}