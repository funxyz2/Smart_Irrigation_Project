# Flask-based-server-for-Smart-Irrigation-System-
This repository is a partial project for building a smart irrigation system using an ESP32. It includes a Flask-based server that serves as a module for the ESP32 to call APIs. The server makes decisions based on the results of a trained AI model hosted on it.

// secrets.example.h - file mẫu, KHÔNG dùng trực tiếp

#ifndef SECRETS_H
#define SECRETS_H

#define WIFI_SSID "your_wifi_name"
#define WIFI_PASSWORD "your_wifi_password"

#define BLYNK_AUTH_TOKEN "your_blynk_token"

#define FLASK_SERVER_URL "http://your-flask-api.com/predict"

#endif
