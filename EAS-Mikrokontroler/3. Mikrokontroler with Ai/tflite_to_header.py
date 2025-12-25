import sys

if len(sys.argv) < 3:
    print("Usage: python tflite_to_header.py pwm_ff_lstm.tflite pwm_ff_lstm_model.h")
    sys.exit(1)

in_path = sys.argv[1]
out_path = sys.argv[2]
var_name = "g_pwm_ff_lstm_model"

data = open(in_path, "rb").read()
with open(out_path, "w") as f:
    f.write("#pragma once\n")
    f.write("#include <cstdint>\n\n")
    f.write(f"alignas(16) const unsigned char {var_name}[] = {{\n")
    for i, b in enumerate(data):
        if i % 12 == 0:
            f.write("  ")
        f.write(f"0x{b:02x}, ")
        if i % 12 == 11:
            f.write("\n")
    f.write("\n};\n")
    f.write(f"const unsigned int {var_name}_len = {len(data)};\n")

print("Header dibuat:", out_path)