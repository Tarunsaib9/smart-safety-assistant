import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Step 1: Load your existing sensor data
df = pd.read_csv("sensor_data.csv")

# Step 2: Define features (X) and label (y)
X = df[['temperature', 'pressure', 'vibration']]
y = df['failure']

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Evaluate performance
y_pred = model.predict(X_test)
print("🔍 Model Evaluation:\n")
print(classification_report(y_test, y_pred))

# Step 6: Predict failure on the full dataset
df['predicted_failure'] = model.predict(X)

# Step 7: Save predictions for Power BI dashboard
df.to_csv("predicted_failures.csv", index=False)
print("✅ Predictions saved to predicted_failures.csv")

# (Optional) Save model to reuse later
joblib.dump(model, 'failure_model.pkl')
