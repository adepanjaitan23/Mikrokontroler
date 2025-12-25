import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# =======================
# Konfigurasi
# =======================
CSV_PATH   = "telemetry.csv"
SEQ_LEN    = 15          # dipendekkan supaya model lebih kecil
TEST_RATIO = 0.2
EPOCHS     = 30
BATCH      = 32

MAX_RPM = 1000.0
MAX_PWM = 255.0

# =======================
# Load data
# =======================
df = pd.read_csv(CSV_PATH)

required = ["setpoint_rpm", "rpm", "pwm"]
for c in required:
    if c not in df.columns:
        raise ValueError(f"Kolom {c} tidak ada di CSV. Kolom yang ada: {list(df.columns)}")

setpoint = df["setpoint_rpm"].astype(np.float32).to_numpy()
rpm      = df["rpm"].astype(np.float32).to_numpy()
pwm      = df["pwm"].astype(np.float32).to_numpy()

# normalisasi ke 0..1
x1 = np.clip(setpoint / MAX_RPM, 0.0, 1.0)
x2 = np.clip(rpm      / MAX_RPM, 0.0, 1.0)
y  = np.clip(pwm      / MAX_PWM, 0.0, 1.0)

# =======================
# Bentuk sequence
# =======================
X = []
Y = []

for i in range(len(df) - SEQ_LEN):
    seq = np.stack(
        [x1[i:i+SEQ_LEN],
         x2[i:i+SEQ_LEN]],
        axis=1
    )  # (SEQ_LEN, 2)
    X.append(seq)
    Y.append(y[i + SEQ_LEN])

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32).reshape(-1, 1)

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# shuffle dan split
idx = np.arange(len(X))
np.random.shuffle(idx)
X = X[idx]
Y = Y[idx]

n_test = int(len(X) * TEST_RATIO)
X_test, Y_test = X[:n_test], Y[:n_test]
X_train, Y_train = X[n_test:], Y[n_test:]

print("Train:", X_train.shape, Y_train.shape)
print("Test :", X_test.shape, Y_test.shape)

# =======================
# Model LSTM kecil
# =======================
inp = keras.Input(shape=(SEQ_LEN, 2))
x   = keras.layers.LSTM(
        8,
        return_sequences=False,
        unroll=True
      )(inp)
x   = keras.layers.Dense(8, activation="relu")(x)
out = keras.layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inp, out)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="mse"
)

model.summary()

# =======================
# Training
# =======================
cb = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
]

history = model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    epochs=EPOCHS,
    batch_size=BATCH,
    shuffle=True,
    verbose=2,
    callbacks=cb
)

# =======================
# Evaluasi cepat
# =======================
pred = model.predict(X_test[:200])
print("Contoh pred_norm:", pred[:5].reshape(-1))
print("Contoh y_norm   :", Y_test[:5].reshape(-1))

# =======================
# Convert ke TFLite
# =======================
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

if hasattr(converter, "_experimental_lower_tensor_list_ops"):
    converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

with open("pwm_ff_lstm.tflite", "wb") as f:
    f.write(tflite_model)

print("Berhasil bikin pwm_ff_lstm.tflite")

# Simpan info normalisasi
norm = {
    "SEQ_LEN": SEQ_LEN,
    "MAX_RPM": float(MAX_RPM),
    "MAX_PWM": float(MAX_PWM)
}
with open("norm.json", "w") as f:
    json.dump(norm, f, indent=2)

print("Selesai, file pwm_ff_lstm.tflite dan norm.json sudah dibuat")