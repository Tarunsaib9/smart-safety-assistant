import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import joblib

# ========== Step 1: Load Data ==========
try:
    data = pd.read_csv("sensor_data.csv")
    assert {'temperature', 'pressure', 'vibration', 'failure'}.issubset(data.columns)
except Exception as e:
    raise ValueError("❌ Failed to load 'sensor_data.csv' or required columns missing.") from e

# ========== Step 2: Feature Preparation ==========
X = data[['temperature', 'pressure', 'vibration']]
y = data['failure']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# ========== Step 3: Define Models ==========
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

# ========== Step 4: Train & Evaluate ==========
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name:<20}: Accuracy = {acc:.4f}")

# ========== Step 5: Select & Save Best Model ==========
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
joblib.dump(best_model, "failure_model.pkl")
print(f"\n✅ Best Model: {best_model_name} (Accuracy = {results[best_model_name]:.4f})")
print("📦 Saved as 'failure_model.pkl'")

# ========== Step 6: Save Results for Dashboard ==========
results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
results_df = results_df.sort_values(by="Accuracy", ascending=False)
results_df.to_csv("model_performance.csv", index=False)
print("📊 Model performance saved to 'model_performance.csv'")
