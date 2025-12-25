/* =========================================================
   iMCLab Motor LSTM Firmware - FINAL (AUTO RPM)
   PWM 0–255, Forward only
   Encoder: 2 holes / revolution
   Board  : ESP32
   ========================================================= */

#include <Arduino.h>

/* =======================
   CONSTANTS
   ======================= */
const String vers = "1.2-LSTM-AUTO-RPM";
const int baud = 115200;

/* =======================
   PIN CONFIG
   ======================= */
const int motor1Pin1 = 27;
const int motor1Pin2 = 26;
const int enable1Pin = 12;
const int pinLED     = 2;
const int pin_rpm    = 13;

/* =======================
   PWM CONFIG
   ======================= */
const int pwmFreq = 30000;
const int pwmChannel = 1;
const int pwmResolution = 8;

/* =======================
   RPM VARIABLES
   ======================= */
volatile unsigned long rev = 0;
unsigned long last_rev_count = 0;
unsigned long last_rpm_time = 0;

float rpm = 0.0;
float rpm_filtered = 0.0;

/* =======================
   SERIAL
   ======================= */
char Buffer[64];
String cmd = "";
int pwm_cmd = 0;

/* =======================
   INTERRUPT
   ======================= */
void IRAM_ATTR countPulse() {
  rev++;
}

/* =======================
   SERIAL PARSER
   ======================= */
void parseSerial() {
  if (!Serial.available()) return;

  int len = Serial.readBytesUntil('\n', Buffer, sizeof(Buffer) - 1);
  if (len <= 0) return;

  Buffer[len] = 0;
  String input = String(Buffer);
  input.trim();
  input.toUpperCase();

  if (input.startsWith("OP")) {
    pwm_cmd = input.substring(2).toInt();
    pwm_cmd = constrain(pwm_cmd, 0, 255);
    cmd = "OP";
  }
  else if (input == "X") {
    cmd = "X";
  }
}

/* =======================
   RPM CALC
   ======================= */
void updateRPM() {
  unsigned long now = millis();
  unsigned long dt = now - last_rpm_time;

  if (dt >= 1000) {
    unsigned long current_rev;

    noInterrupts();
    current_rev = rev;
    interrupts();

    unsigned long pulses = current_rev - last_rev_count;
    const int holes = 2;

    float rotations = (float)pulses / holes;
    rpm = (rotations * 60000.0) / dt;
    rpm_filtered = 0.7 * rpm_filtered + 0.3 * rpm;

    last_rev_count = current_rev;
    last_rpm_time = now;

    Serial.print("RPM:");
    Serial.println(rpm_filtered);
  }
}

/* =======================
   SETUP
   ======================= */
void setup() {
  Serial.begin(baud);
  delay(500);

  pinMode(motor1Pin1, OUTPUT);
  pinMode(motor1Pin2, OUTPUT);
  pinMode(enable1Pin, OUTPUT);
  pinMode(pinLED, OUTPUT);
  pinMode(pin_rpm, INPUT);

  ledcSetup(pwmChannel, pwmFreq, pwmResolution);
  ledcAttachPin(enable1Pin, pwmChannel);
  ledcWrite(pwmChannel, 0);

  attachInterrupt(digitalPinToInterrupt(pin_rpm), countPulse, CHANGE);

  last_rpm_time = millis();
  Serial.println("Firmware ready");
}

/* =======================
   LOOP
   ======================= */
void loop() {
  parseSerial();

  if (cmd == "OP") {
    digitalWrite(motor1Pin1, LOW);
    digitalWrite(motor1Pin2, HIGH);
    ledcWrite(pwmChannel, pwm_cmd);

    Serial.print("PWM:");
    Serial.println(pwm_cmd);
    cmd = "";
  }

  else if (cmd == "X") {
    ledcWrite(pwmChannel, 0);
    Serial.println("STOP");
    cmd = "";
  }

  updateRPM();
  delay(2);
}
