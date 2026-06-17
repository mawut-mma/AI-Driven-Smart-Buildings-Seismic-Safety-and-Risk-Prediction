import serial
import time
import joblib
import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD THE TRAINED AI MODEL & SCALER
# ==========================================
print("Loading AI Model and Scaler...")
try:
    # Load your trained model and scaler files directly from the directory
    deployed_model = joblib.load("evacuation_rf_model.pkl")
    deployed_scaler = joblib.load("evacuation_scaler.pkl")
    print("Model and scaler loaded successfully!")
except Exception as e:
    print(f"Error loading files: {e}")
    print("Ensure 'evacuation_rf_model.pkl' and 'evacuation_scaler.pkl' are in this same folder.")
    io_error = True

if 'io_error' not in locals():
    # Define features and labels exactly as matching your trained model layout
    FEATURES = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz", "Temp1", "Temp2", "Flag", "Capteur1_cm", "Capteur2_cm"]
    status_labels = {0: "SAFE", 1: "WATCH", 2: "DANGER", 3: "EVACUATE"}
    path_threshold = 1.0

    # ==========================================
    # 2. CONNECT TO BOTH USB COM PORTS
    # ==========================================
    # Change COM3 and COM4 if Windows Device Manager assigned different ports to your boards
    PORT_ARDUINO_1 = 'COM5'  # Node A: Ax, Ay, Az, Gx, Gy, Gz, Temp1, Flag
    PORT_ARDUINO_2 = 'COM6'  # Node B: Temp2, Capteur1_cm, Capteur2_cm
    BAUD_RATE = 9600

    print(f"Opening port connection to Arduino 1 ({PORT_ARDUINO_1})...")
    print(f"Opening port connection to Arduino 2 ({PORT_ARDUINO_2})...")

    try:
        arduino_1 = serial.Serial(PORT_ARDUINO_1, BAUD_RATE, timeout=1)
        arduino_2 = serial.Serial(PORT_ARDUINO_2, BAUD_RATE, timeout=1)
        time.sleep(2)  # Give both hardware units time to finish booting up
        print("Hardware links fully operational!")
        connection_success = True
    except Exception as e:
        print(f"Connection failed! Close Arduino Serial Monitors and try again. Error: {e}")
        connection_success = False

    # ==========================================
  # ==========================================
    # 3. LIVE DATA INTEGRATION & PREDICTION LOOP
    # ==========================================
    if connection_success:
        print("\n--- REAL-TIME STRUCTURAL MONITORING ACTIVE ---")
        try:
            while True:
                # Process data frames only when both serial registers have incoming data lines
                if arduino_1.in_waiting > 0 and arduino_2.in_waiting > 0:
                    
                    # FIXED: Added errors='ignore' to prevent crashes from initial startup noise
                    raw_data_1 = arduino_1.readline().decode('utf-8', errors='ignore').strip()
                    raw_data_2 = arduino_2.readline().decode('utf-8', errors='ignore').strip()
                    
                    try:
                        # Split string lines by commas and transform entries to float values
                        list_1 = [float(x) for x in raw_data_1.split(',')]
                        list_2 = [float(x) for x in raw_data_2.split(',')]
                        
                        # Guard condition: ensure transmissions are full frames before parsing
                        if len(list_1) != 8 or len(list_2) != 3:
                            continue
                        
                        # Map input features from the two separate streams
                        Ax, Ay, Az, Gx, Gy, Gz, Temp1, Flag = list_1
                        Temp2, Capteur1_cm, Capteur2_cm = list_2
                        
                        # Package individual metrics into the precise array sequence used during model training
                        combined_data = [Ax, Ay, Az, Gx, Gy, Gz, Temp1, Temp2, Flag, Capteur1_cm, Capteur2_cm]

                        # Format array as a DataFrame row and push it through the normalization and ML pipelines
                        current_reading = pd.DataFrame([combined_data], columns=FEATURES)
                        current_scaled = deployed_scaler.transform(current_reading)
                        prediction = deployed_model.predict(current_scaled)[0]
                        status = status_labels[prediction]
                        
                        # Output clear system updates down to the notebook console screen
                        print(f"Ax: {Ax:.2f} Ay: {Ay:.2f} Az: {Az:.2f} | Temp: {Temp1:.1f}/{Temp2:.1f}°C | STATUS: {status}")
                        
                        # Active evacuation routing mechanism
                        if status == "EVACUATE":
                            print("\n🚨 EMERGENCY WARNING: ACTIVATING SMART BUILDING ALARMS 🚨")
                            if Capteur1_cm <= path_threshold:
                                print("=> Path A blocked -> Directing Evacuation to Path B\n")
                            else:
                                print("=> Path A clear -> Directing Evacuation to Path A\n")

                    except ValueError:
                        # Gracefully disregard noise spikes or corrupted characters from the serial buffer
                        pass 

                time.sleep(0.05)  # Restrict CPU overhead

        except KeyboardInterrupt:
            print("\nMonitoring system paused by user.")
        finally:
            # Safely release the serial channel holds so COM ports don't lock out
            arduino_1.close()
            arduino_2.close()
            print("Serial communication lines closed cleanly.")

import serial
import time

PORT_ARDUINO_1 = 'COM5'
PORT_ARDUINO_2 = 'COM6'
BAUD_RATE = 9600

print("Opening test connections...")
try:
    test_a1 = serial.Serial(PORT_ARDUINO_1, BAUD_RATE, timeout=1)
    test_a2 = serial.Serial(PORT_ARDUINO_2, BAUD_RATE, timeout=1)
    time.sleep(2)
    print("Testing stream for 5 seconds... Watch below:\n")
    
    start_time = time.time()
    while time.time() - start_time < 5:
        if test_a1.in_waiting > 0:
            line1 = test_a1.readline().decode('utf-8', errors='ignore').strip()
            print(f"[COM5 Raw Data]: {line1}")
            
        if test_a2.in_waiting > 0:
            line2 = test_a2.readline().decode('utf-8', errors='ignore').strip()
            print(f"[COM6 Raw Data]: {line2}")
            
        time.sleep(0.1)

except Exception as e:
    print(f"Error testing ports: {e}")
finally:
    try:
        test_a1.close()
        test_a2.close()
        print("\nTest finished. Ports closed safely.")
    except NameError:
        pass
