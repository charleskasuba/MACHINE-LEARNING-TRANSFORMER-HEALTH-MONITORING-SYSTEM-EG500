void setup() {
  // put your setup code here, to run once:
#include <WiFi.h>
#include <HTTPClient.h>
#include <PZEM004Tv30.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <DHT.h>

// --- WiFi & Server Configurations ---
const char* WIFI_SSID = "THE METHOD ZONE";
const char* WIFI_PASS = "Chabu321+";
const char* SERVER_URL = "https://smartmeter-isps.onrender.com/api/data";

// --- Hardware Pin Definitions ---
const int DHT_PIN = 23;
const int FLAME_AO_PIN = 34; // Analog pin (ADC1)
const int FLAME_DO_PIN = 35; // Digital pin

// --- Sensor Instantiations ---
LiquidCrystal_I2C lcd(0x27, 20, 4);  
DHT dht(DHT_PIN, DHT22); 

// Dual PZEM instances on separate Hardware Serial buses
PZEM004Tv30 pzemPrimary(Serial2, 25, 26);
PZEM004Tv30 pzemSecondary(Serial1, 27, 14);

// --- Timing Variables ---
unsigned long lastSend = 0;
unsigned long lastWiFiCheck = 0;

void setup() {
  Serial.begin(115200);

  // Initialize Input Pins
  pinMode(FLAME_DO_PIN, INPUT);

  // Initialize DHT22
  dht.begin();

  // Initialize I2C Bus and LCD
  Wire.begin(21, 22);  
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Booting System...");

  // Setup Wi-Fi
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting to WiFi");
  
  lcd.setCursor(0, 1);
  lcd.print("Connecting WiFi...");
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  lcd.clear();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected");
    lcd.setCursor(0, 0);
    lcd.print("WiFi: Connected");
    lcd.setCursor(0, 1);
    lcd.print(WiFi.localIP().toString());
  } else {
    Serial.println("\nWiFi failed — retrying in loop");
    lcd.setCursor(0, 0);
    lcd.print("WiFi: Failed");
  }
  delay(2000); 
  lcd.clear();
}

void loop() {
  unsigned long now = millis();

  // --- WIFI RECOVERY LOGIC (Checks connection every 10 seconds) ---
  if (now - lastWiFiCheck >= 10000) {
    lastWiFiCheck = now;
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi lost, reconnecting...");
      WiFi.reconnect();
    }
  }

  // --- DATA TRANSMISSION & DISPLAY LOGIC (Runs every 3 seconds) ---
  if (now - lastSend >= 3000) {
    lastSend = now;

    // Read Primary PZEM values
    float p_voltage = pzemPrimary.voltage();
    float p_current = pzemPrimary.current();
    float p_power = pzemPrimary.power();
    float p_energy = pzemPrimary.energy();
    float p_pf = pzemPrimary.pf();

    if (isnan(p_voltage)) { p_voltage = 0; p_current = 0; p_power = 0; p_energy = 0; p_pf = 0; }

    // Read Secondary PZEM values
    float s_voltage = pzemSecondary.voltage();
    float s_current = pzemSecondary.current();
    float s_power = pzemSecondary.power();
    float s_energy = pzemSecondary.energy();
    float s_pf = pzemSecondary.pf();

    if (isnan(s_voltage)) { s_voltage = 0; s_current = 0; s_power = 0; s_energy = 0; s_pf = 0; }

    // Calculate Transformer Efficiency
    float efficiency = 0.0;
    if (p_power > 0) {
      efficiency = (s_power / p_power) * 100.0;
      if (efficiency > 100.0) efficiency = 100.0; // Clamp calculation anomalies
    }

    // Read Environment Sensors (DHT22 & Flame)
    float temperature = dht.readTemperature();
    float humidity = dht.readHumidity();
    if (isnan(temperature)) temperature = 0.0;
    if (isnan(humidity)) humidity = 0.0;

    int flameAnalog = analogRead(FLAME_AO_PIN);
    bool flameDigital = (digitalRead(FLAME_DO_PIN) == LOW || flameAnalog < 2000); 
    String flameStatus = flameDigital ? "FIRE!" : "SAFE";

    // --- Print to 20x4 LCD Screen ---
    // Line 0: Primary Stats & PF
    lcd.setCursor(0, 0);
    lcd.print("P:"); lcd.print(p_voltage, 0); lcd.print("V ");
    lcd.print(p_power, 0); lcd.print("W ");
    lcd.print("PF:"); lcd.print(p_pf, 2);

    // Line 1: Secondary Stats & PF
    lcd.setCursor(0, 1);
    lcd.print("S:"); lcd.print(s_voltage, 0); lcd.print("V ");
    lcd.print(s_power, 0); lcd.print("W ");
    lcd.print("PF:"); lcd.print(s_pf, 2);

    // Line 2: Transformer Efficiency
    lcd.setCursor(0, 2);
    lcd.print("System Eff: "); lcd.print(efficiency, 1); lcd.print("%   ");

    // Line 3: Environment Data
    lcd.setCursor(0, 3);
    lcd.print("T:"); lcd.print(temperature, 1); lcd.print("C ");
    lcd.print("H:"); lcd.print(humidity, 0); lcd.print("% ");
    lcd.setCursor(14, 3);
    lcd.print(flameStatus); lcd.print("   ");

    // --- Network API Submission via HTTP POST ---
    if (WiFi.status() == WL_CONNECTED) {
      String json = "{";
      json += "\"primary_voltage\":" + String(p_voltage, 1) + ",";
      json += "\"primary_current\":" + String(p_current, 2) + ",";
      json += "\"primary_power\":" + String(p_power, 1) + ",";
      json += "\"primary_energy\":" + String(p_energy, 2) + ",";
      json += "\"primary_pf\":" + String(p_pf, 2) + ",";
      json += "\"secondary_voltage\":" + String(s_voltage, 1) + ",";
      json += "\"secondary_current\":" + String(s_current, 2) + ",";
      json += "\"secondary_power\":" + String(s_power, 1) + ",";
      json += "\"secondary_energy\":" + String(s_energy, 2) + ",";
      json += "\"secondary_pf\":" + String(s_pf, 2) + ",";
      json += "\"efficiency\":" + String(efficiency, 1) + ",";
      json += "\"temperature\":" + String(temperature, 1) + ",";
      json += "\"humidity\":" + String(humidity, 1) + ",";
      json += "\"flame_raw\":" + String(flameAnalog) + ",";
      json += "\"fire_detected\":\"" + flameStatus + "\"";
      json += "}";

      HTTPClient http;
      http.begin(SERVER_URL);
      http.addHeader("Content-Type", "application/json");
      int code = http.POST(json);
      
      if (code > 0) {
        Serial.print("POST OK — Response Code: ");
        Serial.println(code);
      } else {
        Serial.print("POST failed: ");
        Serial.println(http.errorToString(code).c_str());
      }
      http.end();
    } else {
      Serial.println("WiFi down, skipping API send");
    }
  }
}o run repeatedly:

}
