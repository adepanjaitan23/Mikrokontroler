/* =========================================================
   iMCLab Motor LSTM Feedforward + PID
   Board  : ESP32
   PWM    : 0..255
   Encoder: 2 holes / revolution
   Control: PWM_total = PWM_ff(LSTM) + PID(error)
   ========================================================= 
*/

#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>

// TensorFlow Lite Micro
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Header model dari pwm_ff_lstm.tflite yang dikonversi ke C array
#include "pwm_ff_lstm_model.h"

/* =======================
   KONSTANTA UMUM
   ======================= */
const String vers = "1.2-LSTM-FFPID-AUTO-RPM";
const int baud = 115200;

/* =======================
   KONFIGURASI PIN
   ======================= */
const int motor1Pin1 = 27;
const int motor1Pin2 = 26;
const int enable1Pin = 12;
const int pinLED     = 2;
const int pin_rpm    = 13;

/* =======================
   KONFIGURASI PWM
   ======================= */
const int pwmFreq       = 30000;
const int pwmChannel    = 1;
const int pwmResolution = 8;   // 0..255

/* =======================
   ENCODER / RPM
   ======================= */
volatile unsigned long rev = 0;
unsigned long last_rev_count = 0;
unsigned long last_rpm_time  = 0;

float rpm          = 0.0f;
float rpm_filtered = 0.0f;

const uint32_t RPM_PERIOD_MS = 200;
const int holes = 2;  // 2 lubang per putaran

/* =======================
   SERIAL COMMAND
   ======================= */
char Buffer[64];
String cmd          = "";
int   pwm_cmd_manual = 0;
float setpoint_rpm   = 0.0f;

/* =======================
   MODE FLAG
   ======================= */
bool motor_enable = false;
bool ai_enable    = true;

/* =======================
   KONFIGURASI LSTM
   ======================= */
static const int   SEQ_LEN = 25;
static const float MAX_RPM = 1000.0f;
static const float MAX_PWM = 255.0f;

const uint32_t CONTROL_PERIOD_MS = 50;  // 20 Hz

float seq_setpoint[SEQ_LEN];
float seq_rpm[SEQ_LEN];
int   seq_idx  = 0;
bool  seq_full = false;

// smoothing feedforward
float pwm_ff_filtered = 0.0f;
const float alpha_ff  = 0.2f;

/* =======================
   KONFIGURASI PID
   ======================= */
float Kp = 0.25f;
float Ki = 0.05f;
float Kd = 0.00f;

float i_term      = 0.0f;
float prev_err    = 0.0f;
uint32_t last_pid_ms = 0;

const float PID_OUT_MAX = 200.0f;  // batas koreksi PID
const float I_MAX       = 200.0f;  // batas integral

/* =======================
   OBJEK TFLITE
   ======================= */
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input  = nullptr;
static TfLiteTensor* output = nullptr;

// arena memori untuk TFLite
static constexpr int kTensorArenaSize = 60 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

/* =======================
   INTERRUPT ENCODER
   ======================= */
void IRAM_ATTR countPulse() {
  rev++;
}

/* =======================
   UTILITAS
   ======================= */
static inline float clampf(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

static inline int clampi(int x, int lo, int hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

/* =======================
   FUNGSI MOTOR
   ======================= */
void motorForwardOnly() {
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, HIGH);
}

void motorStop() {
  ledcWrite(pwmChannel, 0);
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, LOW);
}

/* =======================
   PARSER PERINTAH SERIAL
   OP<pwm>  manual PWM
   SP<rpm>  setpoint rpm AI
   AI0      matikan AI
   AI1      nyalakan AI
   X        stop
   ======================= */
void parseSerial() {
  if (!Serial.available()) return;

  int len = Serial.readBytesUntil('\n', Buffer, sizeof(Buffer) - 1);
  if (len <= 0) return;

  Buffer[len] = 0;
  String inputStr = String(Buffer);
  inputStr.trim();
  inputStr.toUpperCase();

  if (inputStr.startsWith("OP")) {
    pwm_cmd_manual = inputStr.substring(2).toInt();
    pwm_cmd_manual = constrain(pwm_cmd_manual, 0, 255);
    cmd = "OP";
  }
  else if (inputStr.startsWith("SP")) {
    setpoint_rpm = inputStr.substring(2).toFloat();
    if (setpoint_rpm < 0) setpoint_rpm = 0;
    if (setpoint_rpm > MAX_RPM) setpoint_rpm = MAX_RPM;
    cmd = "SP";
  }
  else if (inputStr == "AI0") {
    ai_enable = false;
    cmd = "AI";
  }
  else if (inputStr == "AI1") {
    ai_enable = true;
    cmd = "AI";
  }
  else if (inputStr == "X") {
    cmd = "X";
  }
}

/* =======================
   PERHITUNG
   AN RPM
   ======================= */
void updateRPM() {
  unsigned long now = millis();
  unsigned long dt  = now - last_rpm_time;
  if (dt < RPM_PERIOD_MS) return;

  unsigned long current_rev;
  noInterrupts();
  current_rev = rev;
  interrupts();

  unsigned long pulses = current_rev - last_rev_count;
  float rotations = (float)pulses / (float)holes;
  rpm = (rotations * 60000.0f) / (float)dt;

  rpm_filtered = 0.7f * rpm_filtered + 0.3f * rpm;

  last_rev_count = current_rev;
  last_rpm_time  = now;

  Serial.print("RPM:");
  Serial.println(rpm_filtered, 2);
}

/* =======================
   SETUP TFLITE
   ======================= */
bool setupTFLite() {
  Serial.println("Init TFLite...");

  const tflite::Model* model = tflite::GetModel(g_pwm_ff_lstm_model);
  if (model == nullptr) {
    Serial.println("Model pointer NULL");
    return false;
  }

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize
  );
  interpreter = &static_interpreter;

  TfLiteStatus status = interpreter->AllocateTensors();
  if (status != kTfLiteOk) {
    Serial.println("AllocateTensors gagal");
    return false;
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  Serial.print("Input dims: ");
  if (input && input->dims) {
    for (int i = 0; i < input->dims->size; i++) {
      Serial.print(input->dims->data[i]);
      Serial.print(" ");
    }
  } else {
    Serial.print("null");
  }
  Serial.println();

  Serial.println("TFLite OK");
  return true;
}

/* =======================
   BUFFER SEQUENCE
   ======================= */
void pushSequence(float sp_rpm, float rpm_meas) {
  seq_setpoint[seq_idx] = clampf(sp_rpm  / MAX_RPM, 0.0f, 1.0f);
  seq_rpm[seq_idx]      = clampf(rpm_meas / MAX_RPM, 0.0f, 1.0f);

  seq_idx++;
  if (seq_idx >= SEQ_LEN) {
    seq_idx  = 0;
    seq_full = true;
    Serial.println("SEQ FULL");
  }
}

/* =======================
   INFERENSI LSTM
   ======================= */
float inferPWMff() {
  if (!seq_full) return 0.0f;
  if (!input || !output) return 0.0f;
  if (input->type != kTfLiteFloat32) {
    Serial.println("Model bukan float32");
    return 0.0f;
  }

  int start = seq_idx;
  float* in = input->data.f;

  // diasumsikan bentuk input [1, SEQ_LEN, 2]
  for (int t = 0; t < SEQ_LEN; t++) {
    int j = (start + t) % SEQ_LEN;
    in[t * 2 + 0] = seq_setpoint[j];
    in[t * 2 + 1] = seq_rpm[j];
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke gagal");
    return 0.0f;
  }

  float y_norm = clampf(output->data.f[0], 0.0f, 1.0f);
  return y_norm * MAX_PWM;
}

/* =======================
   UJI LSTM SEKALI SAAT STARTUP
   ======================= */
void testLSTMOnce() {
  Serial.println("Test LSTM dengan dummy sequence...");

  for (int i = 0; i < SEQ_LEN; i++) {
    seq_setpoint[i] = 0.4f;  // seolah 40 persen MAX_RPM
    seq_rpm[i]      = 0.0f;
  }
  seq_idx  = 0;
  seq_full = true;

  float test_ff = inferPWMff();
  Serial.print("Hasil test FF (dummy): ");
  Serial.println(test_ff, 2);
}

/* =======================
   PID
   ======================= */
float pidCompute(float sp, float meas) {
  uint32_t now = millis();
  float dt = (last_pid_ms == 0)
           ? (CONTROL_PERIOD_MS / 1000.0f)
           : ((now - last_pid_ms) / 1000.0f);

  last_pid_ms = now;
  if (dt <= 0.0f) dt = CONTROL_PERIOD_MS / 1000.0f;

  float err = sp - meas;
  float d   = (err - prev_err) / dt;
  prev_err  = err;

  i_term += err * dt;
  i_term = clampf(i_term, -I_MAX, I_MAX);

  float u = Kp * err + Ki * i_term + Kd * d;
  return clampf(u, -PID_OUT_MAX, PID_OUT_MAX);
}

/* =======================
   SETUP
   ======================= */
void setup() {
  Serial.begin(baud);
  delay(1000);

  Serial.println("Booting...");
  Serial.print("Version:");
  Serial.println(vers);

  pinMode(motor1Pin1, OUTPUT);
  pinMode(motor1Pin2, OUTPUT);
  pinMode(enable1Pin, OUTPUT);
  pinMode(pinLED, OUTPUT);
  pinMode(pin_rpm, INPUT);

  ledcSetup(pwmChannel, pwmFreq, pwmResolution);
  ledcAttachPin(enable1Pin, pwmChannel);
  ledcWrite(pwmChannel, 0);

  attachInterrupt(digitalPinToInterrupt(pin_rpm), countPulse, CHANGE);

  for (int i = 0; i < SEQ_LEN; i++) {
    seq_setpoint[i] = 0.0f;
    seq_rpm[i]      = 0.0f;
  }

  last_rpm_time = millis();

  bool ok = setupTFLite();
  if (!ok) {
    Serial.println("TFLite init gagal, AI dimatikan");
    ai_enable = false;
  }

  Serial.println("Firmware ready");
  if (ok) {
    testLSTMOnce();  // bisa dihapus setelah yakin model bekerja
  }
  Serial.println("Commands: OP<pwm>, SP<rpm>, AI0, AI1, X");
}

/* =======================
   LOOP
   ======================= */
void loop() {
  parseSerial();
  updateRPM();

  static uint32_t last_control = 0;
  uint32_t now = millis();
  if (now - last_control < CONTROL_PERIOD_MS) {
    delay(2);
    return;
  }
  last_control = now;

  digitalWrite(pinLED, motor_enable ? HIGH : LOW);

  // penanganan perintah
  if (cmd == "OP") {
    ai_enable    = false;
    motor_enable = (pwm_cmd_manual > 0);

    if (!motor_enable) {
      motorStop();
      Serial.println("STOP");
    } else {
      motorForwardOnly();
      ledcWrite(pwmChannel, pwm_cmd_manual);
      Serial.print("PWM:");
      Serial.println(pwm_cmd_manual);
    }
    cmd = "";
  }
  else if (cmd == "SP") {
    motor_enable = (setpoint_rpm > 1.0f);
    if (!motor_enable) {
      motorStop();
      i_term          = 0.0f;
      prev_err        = 0.0f;
      pwm_ff_filtered = 0.0f;
      Serial.println("STOP");
    } else {
      ai_enable = true;
      Serial.print("SP:");
      Serial.println(setpoint_rpm, 2);
    }
    cmd = "";
  }
  else if (cmd == "AI") {
    Serial.print("AI:");
    Serial.println(ai_enable ? "ON" : "OFF");
    cmd = "";
  }
  else if (cmd == "X") {
    motor_enable    = false;
    motorStop();
    i_term          = 0.0f;
    prev_err        = 0.0f;
    pwm_ff_filtered = 0.0f;
    Serial.println("STOP");
    cmd = "";
  }

  // jika motor tidak diizinkan, jangan kirim PWM
  if (!motor_enable) {
    ledcWrite(pwmChannel, 0);
    delay(2);
    return;
  }

  motorForwardOnly();

  float sp   = setpoint_rpm;
  float meas = rpm_filtered;

  // update sequence untuk LSTM
  pushSequence(sp, meas);

  int pwm_out = 0;

  if (ai_enable) {
    float pwm_ff = inferPWMff();
    pwm_ff_filtered = (1.0f - alpha_ff) * pwm_ff_filtered + alpha_ff * pwm_ff;

    float pid_u = pidCompute(sp, meas);

    float pwm_total = pwm_ff_filtered + pid_u;
    pwm_total = clampf(pwm_total, 0.0f, MAX_PWM);

    pwm_out = (int)(pwm_total + 0.5f);
    pwm_out = clampi(pwm_out, 0, 255);

    ledcWrite(pwmChannel, pwm_out);

    Serial.print("PWM:");
    Serial.print(pwm_out);
    Serial.print(",FF:");
    Serial.print(pwm_ff_filtered, 2);
    Serial.print(",PID:");
    Serial.println(pid_u, 2);
  }
  else {
    pwm_out = clampi(pwm_cmd_manual, 0, 255);
    ledcWrite(pwmChannel, pwm_out);
    Serial.print("PWM:");
    Serial.println(pwm_out);
  }

  delay(2);
}