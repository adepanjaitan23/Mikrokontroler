#include <WiFi.h>
#include <PubSubClient.h>

// Ganti dengan SSID dan password WiFi kamu
const char* ssid = "Infinix";
const char* password = "12345678";

// Gunakan broker publik HiveMQ (bisa ganti ke broker sendiri)
const char* mqtt_server = "broker.hivemq.com";

WiFiClient espClient;
PubSubClient client(espClient);

// Pin motor
int motor1Pin1 = 27; int motor1Pin2 = 26; int enable1Pin = 12;

// PWM setup
const int freq = 30000;
const int pwmChannel = 0;
const int resolution = 8;
int dutyCycle = 200;

// Fungsi koneksi ke WiFi
void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

// Callback ketika pesan MQTT diterima
void callback(char* topic, byte* message, unsigned int length) {
  Serial.print("Pesan diterima di topic: ");

  Serial.println(topic);

  String msg = "";
  for (int i = 0; i < length; i++) {
    msg += (char)message[i];
  }

  msg.trim();
  Serial.print("Isi pesan: ");
  Serial.println(msg);

  if (msg == "1") {
    Serial.println("Motor ON");
    digitalWrite(motor1Pin1, LOW);
    digitalWrite(motor1Pin2, HIGH);
    ledcWrite(pwmChannel, dutyCycle);
  } 
  else if (msg == "0") {
    Serial.println("Motor OFF");
    digitalWrite(motor1Pin1, LOW);
    digitalWrite(motor1Pin2, LOW);
    ledcWrite(pwmChannel, 0);
  }
  else if (msg.startsWith("SPEED")) {
    int speed = msg.substring(6).toInt();
    dutyCycle = constrain(speed, 0, 255);
    ledcWrite(pwmChannel, dutyCycle);
    Serial.print("Speed set to: ");
    Serial.println(dutyCycle);
  }
}

void reconnect() {
  // Loop sampai terhubung ke MQTT
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    if (client.connect("ESP32MotorClient")) {
      Serial.println("connected");
      // Subscribe ke topic
      client.subscribe("esp32/punto");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" mencoba lagi 5 detik...");
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(motor1Pin1, OUTPUT);
  pinMode(motor1Pin2, OUTPUT);
  pinMode(enable1Pin, OUTPUT);

  ledcSetup(pwmChannel, freq, resolution);
  ledcAttachPin(enable1Pin, pwmChannel);

  setup_wifi();
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
}