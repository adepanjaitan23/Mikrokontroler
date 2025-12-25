import serial
import threading
import tkinter as tk
from tkinter import ttk

# =====================
# SERIAL CONFIG
# =====================
PORT = "COM5"     # GANTI sesuai ESP32 kamu
BAUD = 115200

ser = None
running = False

# =====================
# GUI
# =====================
root = tk.Tk()
root.title("iMCLab Motor Control Dashboard")
root.geometry("420x520")
root.configure(bg="#0f172a")

style = ttk.Style()
style.theme_use("clam")

# =====================
# VARIABLES
# =====================
rpm_var = tk.StringVar(value="0.0")
pwm_var = tk.IntVar(value=0)
status_var = tk.StringVar(value="Disconnected")

# =====================
# SERIAL FUNCTIONS
# =====================
def connect_serial():
    global ser, running
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        running = True
        status_var.set("Connected")
        threading.Thread(target=read_serial, daemon=True).start()
    except:
        status_var.set("Failed to connect")

def send_cmd(cmd):
    if ser and ser.is_open:
        ser.write((cmd + "\n").encode())

def read_serial():
    while running:
        try:
            line = ser.readline().decode().strip()
            if line.startswith("RPM:"):
                rpm = line.replace("RPM:", "")
                rpm_var.set(rpm)
        except:
            pass

# =====================
# CALLBACKS
# =====================
def set_pwm(value):
    pwm = int(float(value))
    pwm_var.set(pwm)
    send_cmd(f"OP {pwm}")

def stop_motor():
    send_cmd("X")
    pwm_var.set(0)

# =====================
# UI COMPONENTS
# =====================
ttk.Label(root, text="iMCLab Motor Dashboard",
          font=("Segoe UI", 16, "bold")).pack(pady=15)

ttk.Button(root, text="Connect Serial",
           command=connect_serial).pack(pady=5)

ttk.Label(root, textvariable=status_var).pack(pady=5)

# RPM Display
rpm_frame = ttk.LabelFrame(root, text="RPM Monitor")
rpm_frame.pack(fill="x", padx=20, pady=15)

ttk.Label(rpm_frame, textvariable=rpm_var,
          font=("Segoe UI", 32, "bold")).pack(pady=10)

# PWM Control
pwm_frame = ttk.LabelFrame(root, text="PWM Control")
pwm_frame.pack(fill="x", padx=20, pady=15)

pwm_slider = ttk.Scale(
    pwm_frame, from_=0, to=255, orient="horizontal",
    command=set_pwm
)
pwm_slider.pack(fill="x", padx=10, pady=10)

ttk.Label(pwm_frame, textvariable=pwm_var,
          font=("Segoe UI", 14)).pack(pady=5)

# STOP BUTTON
ttk.Button(root, text="STOP MOTOR",
           command=stop_motor).pack(pady=20)

root.mainloop()
