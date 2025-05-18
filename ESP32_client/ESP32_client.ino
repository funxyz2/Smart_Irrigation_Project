#include <WiFi.h>
#include <HTTPClient.h>
#include "DHT.h"
#include "secrets.h"

#define DHTPIN 32
#define DHTTYPE DHT22
#define WATER_LEVEL_THRESHOLD 20 // Trên 20% thì bật cảm biến mực nước (1/0)
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

bool isManualMode = false; // Chế độ thủ công từ Blynk App
bool pumpState = false;
unsigned long pumpStartTime = 0;
unsigned long pumpDuration = 0;
bool isPumpRunning = false;
float mlWaterGlobal = 0; // lưu trạng thái đã tưới
String lastPredictionStr = ""; // Lưu kết quả dự đoán

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
    Ở đây đáng lẽ dùng int cho water_level (0% -> 100%)
    Tuy nhiên vì dataset train model buộc cột này là bool (1/0)
    Vì vậy, ta xóa đi logic ấy và thay bằng 1/0
      
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
      Xử lý dữ liệu từ các cảm biến và in ra Serial, đồng thời gán các giá trị tương ứng các biến trong Blynk App
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

    Serial.println("⚠️ Lỗi đọc cảm biến! Đang thử lại...");
    delay(1000); 
  }

  if (!success) {
    Serial.println("❌ Không đọc được cảm biến sau nhiều lần thử. Tạm dừng chương trình.");
    // Gán giá trị đặc biệt để hàm sendToServer() dừng lại
    data.temperature = NAN;
    data.humidity = NAN;
  }
}

void setup() {
  Serial.begin(9600);
  Serial.println("Bạn vừa bật Hệ thống Tưới tiêu thông minh, để hệ thống hoạt động đúng, bạn cần phải tưới cho cây của bạn đủ nước");
  
  // Lưu thời điểm đã tưới
  last_watered_hour = millis() / 3600000.0;

  // pinMode
  pinMode (DIGITAL_SOIL_MOISTURE_PIN, INPUT);
  pinMode (ANALOG_WATER_LEVEL_PIN, INPUT);
  pinMode (ANALOG_SOIL_MOISTURE_PIN, INPUT);
  pinMode(PUMP_PIN, OUTPUT);
  digitalWrite(PUMP_PIN, LOW);

  Blynk.begin(auth, ssid, pass);
  dht.begin(); 
  delay(2000); // Đợi cảm biến ổn định

  timer.setInterval(1000L, readSensors);             // Cập nhật UI Blynk App mỗi 1 giây
  timer.setInterval(7L * 60L * 60L * 1000L, sendToServer);  // Gửi Flask server mỗi 7 tiếng
}

void loop() {
  Blynk.run();
  timer.run();

  // Tắt máy bơm sau khi đủ thời gian
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

// Xử lý delay duration cho máy bơm
void runPumpForML(float ml) {
  pumpDuration = ml / 10.0 * 1000; // ml / tốc độ (10ml/s) * 1000ms
  pumpStartTime = millis();
  digitalWrite(PUMP_PIN, HIGH);
  isPumpRunning = true;
}

void sendToServer() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    readSensors();

    // Tính số giờ đã trôi qua kể từ lần tưới cuối
    float current_hour = millis() / 3600000.0;
    int hours_since_last_watering = current_hour - last_watered_hour;
    Serial.print("Hours since last watering: "); Serial.println(hours_since_last_watering);

    String jsonData = "{\"temperature\":" + String(data.temperature) +
                      ",\"soil_moisture\":" + String(data.soil_moisture) +
                      ",\"water_level\":" + String(data.water_level) +
                      ",\"humidity_air\":" + String(data.humidity) +
                      ",\"last_watered_hour\":" + String(hours_since_last_watering) + "}";

    http.setTimeout(3000); // TImeout 3 giây
    http.begin(serverName);
    http.addHeader("Content-Type", "application/json");

    /*
    Nhận Response và xử lý
    */
    int httpResponseCode = http.POST(jsonData);
    delay(200);

    if (httpResponseCode == 200) {
      String response = http.getString();
      Serial.println("Response: " + response);
      lastPredictionStr = response;  // Ghi chuỗi response lại cho V7
      float mlWater = response.toFloat();
      Serial.print("I will pump this much water: "); Serial.print(mlWater); Serial.println("ml");

      // Nếu Model yêu cầu tưới
      if (mlWater > 0) {
        runPumpForML(mlWater);
        // Cập nhật thời điểm tưới gần nhất
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

// Callback khi switch V4 thay đổi trạng thái
BLYNK_WRITE(V4) {
  // Chặn user chuyển sang manual mode khi đang hệ thống đang tưới
  if (isPumpRunning) {
    Serial.println("⚠️ Pump is running from ML. Manual control is disabled.");
    Blynk.virtualWrite(V4, pumpState ? 1 : 0); // Reset lại trạng thái trên app
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
    // Thực hiện ghi lại trạng thái cho switch V4 (đồng bộ hoá trạng thái)
    Blynk.virtualWrite(V4, pumpState ? 1 : 0);
  }
}

// Callback khi switch V4 thay đổi trạng thái
BLYNK_WRITE(V5) {
  int mode = param.asInt();
  isManualMode = (mode == 1);
  Serial.println(isManualMode ? "INITIATING MANUAL MODE..." : "TURNING OFF MANUAL MODE...");
  
  // Đảm bảo máy bơm luôn tắt khi không dùng
  if (!isManualMode && pumpState) {
    // Nếu đang tưới tay, và vừa chuyển sang auto mode thì tắt bơm
    digitalWrite(PUMP_PIN, LOW);
    pumpState = false;
    Blynk.virtualWrite(V4, 0);
    Serial.println("🛑 Pump OFF: Switched to Auto mode");
  }
}

// Gọi API thủ công
BLYNK_WRITE(V6) {
  int value = param.asInt();
  if (value == 1) {
    Serial.println("🔁 Manual Predict Triggered from Blynk App!");
    sendToServer();  // Gọi hàm gửi dữ liệu đến server Flask
    Blynk.virtualWrite(V6, 0); // Reset lại nút về OFF sau khi nhấn

    // Gửi kết quả dự đoán về Blynk
    if (lastPredictionStr == "") {
      Serial.println("⚠️ No prediction result available.");
      return;
    }
    String result = lastPredictionStr + " ml";
    Serial.print("Sending last prediction string to Blynk: ");
    Serial.println(result);

    Blynk.virtualWrite(V7, result);
  }
}

// Button Blynk reset phần mềm trong ESP32
BLYNK_WRITE(V8) {
  int value = param.asInt();
  if (value == 1) {
    Serial.println("🔁 Người dùng yêu cầu khởi động lại thiết bị...");
    ESP.restart();
  }
}