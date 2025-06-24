import pandas as pd
import numpy as np
import time
from datetime import datetime
import random
import joblib
import os

# Load trained model
model = joblib.load("failure_model.pkl")

# Columns used in model
FEATURE_COLS = ["temperature", "pressure", "vibration"]

print("🚀 Starting real-time sensor simulation...\n")

while True:
    # Simulate one data point
    timestamp = datetime.now()
    temperature = round(random.uniform(60, 100), 2)
    pressure = round(random.uniform(0.9, 1.5), 2)
    vibration = round(random.uniform(0.3, 1.0), 2)

    # Prepare for prediction
    df_input = pd.DataFrame([[temperature, pressure, vibration]], columns=FEATURE_COLS)
    prediction = model.predict(df_input.values)[0]  # Using .values to avoid sklearn warning

    # New row to append
    new_data = pd.DataFrame([{
        "timestamp": timestamp,
        "temperature": temperature,
        "pressure": pressure,
        "vibration": vibration,
        "predicted_failure": prediction
    }])

    # Append to predicted_failures.csv
    file_path = "predicted_failures.csv"
    if os.path.exists(file_path):
        new_data.to_csv(file_path, mode='a', header=False, index=False)
    else:
        new_data.to_csv(file_path, index=False)

    print(f"📡 {timestamp} | Temp: {temperature} | Press: {pressure} | Vib: {vibration} ➜ Predicted Failure: {prediction}")

    # Wait before generating next point (e.g., every 5 sec)
    time.sleep(5)