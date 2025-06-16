import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate 30 days of hourly timestamps
timestamps = pd.date_range(datetime.now() - timedelta(days=30), periods=720, freq='H')

# Simulate machine sensor data
temperature = np.random.normal(75, 10, size=720)
pressure = np.random.normal(30, 5, size=720)
vibration = np.random.normal(1, 0.2, size=720)

# Label failures: if temp > 90 and vibration > 1.3
failure = (temperature > 90) & (vibration > 1.3)

# Combine all into a DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'temperature': temperature.round(2),
    'pressure': pressure.round(2),
    'vibration': vibration.round(2),
    'failure': failure
})

# Save to CSV
df.to_csv('sensor_data.csv', index=False)
print("✅ sensor_data.csv created.")
