import serial
import time
import joblib
import pandas as pd
import os

# --- DECISION ENGINE ---
def get_path_decision(dist1, dist2, threshold=10.0):
    # Logic: Checking if sensors (now at index 10 and 11) report safe distances
    if dist1 > threshold:
        return "Path A is CLEAR", "Path A"
    elif dist2 > threshold:
        return "Path A BLOCKED, Diverting to Path B", "Path B"
    else:
        return "CRITICAL: Both paths BLOCKED", "NONE"

def main():
    # Load AI Assets
    try:
        model = joblib.load("evacuation_rf_model.pkl")
        scaler = joblib.load("evacuation_scaler.pkl")
    except Exception as e:
        print(f"Error loading AI: {e}")
        return
    
    # Updated Features list to match the 12 incoming values
    FEATURES = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz", "Temp1", "Temp2", "Flag", "Extra", "Capteur1_cm", "Capteur2_cm"]
    status_labels = {0: "SAFE", 1: "WATCH", 2: "DANGER", 3: "EVACUATE"}

    # Hardware Setup
    try:
        arduino_1 = serial.Serial('COM5', 115200, timeout=1)
        arduino_2 = serial.Serial('COM6', 115200, timeout=1)
        time.sleep(2) 
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    print("--- SYSTEM ONLINE: MONITORING STRUCTURAL INTEGRITY ---")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            if arduino_1.in_waiting > 0 and arduino_2.in_waiting > 0:
                raw_1 = arduino_1.readline().decode('utf-8', errors='ignore').strip().split(',')
                raw_2 = arduino_2.readline().decode('utf-8', errors='ignore').strip().split(',')
                
                # Check if we have the 10 from A1 and 2 from A2
                if len(raw_1) >= 10 and len(raw_2) >= 2:
                    try:
                        # Combine all values into one list of 12
                        all_data = [float(x) for x in raw_1[:10] + raw_2[:2]]
                        df = pd.DataFrame([all_data], columns=FEATURES)
                        
                        # AI Prediction (using the first 11 features that your model was trained on)
                        scaled_data = scaler.transform(df[FEATURES[:11]])
                        prediction = model.predict(scaled_data)[0]
                        status = status_labels.get(prediction, "UNKNOWN")
                        
                        # Decision Engine (Using index 10 and 11 which are Capteur1 and Capteur2)
                        decision_text, path = get_path_decision(all_data[10], all_data[11])
                        
                        # Output
                        print(f"Status: {status:10} | Sensors: {all_data[10]:.1f}cm, {all_data[11]:.1f}cm | {decision_text}")
                        
                        if status == "EVACUATE":
                            print(f"🚨 EMERGENCY ALERT: Proceed immediately to {path} 🚨")
                    
                    except ValueError:
                        pass
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nSystem Shutdown.")
    finally:
        arduino_1.close()
        arduino_2.close()

if __name__ == "__main__":
    main()
