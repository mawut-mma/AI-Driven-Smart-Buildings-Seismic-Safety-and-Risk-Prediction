import time
import serial
import joblib
import pandas as pd

# ==========================================
# 1. LOAD THE EXPORTED AI FILES
# ==========================================
print("Loading AI Model and Scaler...")
try:
    model = joblib.load("seismic_rf_model.pkl")
    scaler = joblib.load("seismic_scaler.pkl")
    print("AI Brain loaded successfully!\n")
except FileNotFoundError:
    print("ERROR: Make sure 'seismic_rf_model.pkl' and 'seismic_scaler.pkl' are in the same folder as this script.")
    exit()

FEATURES = ["vibration_hz", "acceleration_m2s", "fire_sensor", "structural_damage"]
status_labels = {0: "SAFE", 1: "WATCH", 2: "DANGER", 3: "EVACUATE"}

# ==========================================
# 2. CONNECT TO THE ARDUINO PROTOTYPE
# ==========================================
# NOTE: Your teammate MUST change 'COM3' to whatever port the Arduino is plugged into (e.g., 'COM5' on Windows or '/dev/ttyUSB0' on Mac/Linux)
arduino_port = 'COM3' 
baud_rate = 9600

try:
    ser = serial.Serial(arduino_port, baud_rate, timeout=1)
    time.sleep(2)  # Give the Arduino 2 seconds to reset after connection
    print(f"Successfully connected to Arduino on {arduino_port}")
except Exception as e:
    print(f"Could not connect to Arduino: {e}")
    print("Running in TEST MODE with simulated zeros...")
    ser = None

# ==========================================
# 3. THE REAL-TIME LISTENING LOOP
# ==========================================
print("--- LIVE PROTOTYPE MONITORING ACTIVE ---")
print("Waiting for sensor data...\n")

while True:
    try:
        # Read the live comma-separated data from the Arduino Serial Print
        if ser:
            line = ser.readline().decode('utf-8').strip()
            if not line:
                continue # Skip empty lines
                
            # The Arduino code must do: Serial.println(String(vib) + "," + String(accel) + "," + String(fire) + "," + String(damage));
            readings = line.split(',')
            vibration = float(readings[0])
            acceleration = float(readings[1])
            fire = float(readings[2])
            damage = int(readings[3])
        else:
            # Failsafe test data if Arduino isn't plugged in yet
            vibration, acceleration, fire, damage = 0.15, 0.05, 20.0, 0
            time.sleep(1)

        # 1. Format the raw hardware data
        current_reading = pd.DataFrame([[vibration, acceleration, fire, damage]], columns=FEATURES)
        
        # 2. Scale the data using the exact boundaries learned during training
        current_reading_scaled = scaler.transform(current_reading)
        
        # 3. Let the AI predict the structural status
        prediction = model.predict(current_reading_scaled)[0]
        status = status_labels[prediction]
        
        # 4. Output the result
        print(f"Vib: {vibration:.2f}Hz | Accel: {acceleration:.2f}m/s² | Fire: {fire:.1f} | STATUS: {status}")
        
        # 5. Trigger physical actions if needed
        if status in ["DANGER", "EVACUATE"]:
            print("   !!! EMERGENCY WARNING: DANGER DETECTED !!!")
            # If your teammate wants to turn on a physical LED or Buzzer on the Arduino:
            # if ser: ser.write(b'TRIGGER_ALARM\n') 

    except ValueError:
        print(f"Garbled data received: {line}") # Ignores messy startup strings
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
        if ser:
            ser.close()
        break
