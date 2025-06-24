
# 🔐 Smart Safety Assistant

An AI-powered, real-time safety monitoring and failure prediction platform using Python, Streamlit, and automation tools. This project reflects  the showcasing chatbot development, AI modeling, real-time logging and visualization.

---

## 🎯 Project Objective

This project simulates a real-world scenario where:
- Equipment safety data is submitted in real-time.
- Machine learning predicts potential failures using 8 algorithms.
- Dashboards visualize trends and anomalies.
- A chatbot assists users via voice/text.
- Prediction performance is tracked for analysis and decision-making.

---

## 🧩 Features

| Feature               | Description                                              |
|-----------------------|----------------------------------------------------------|
| ✅ Streamlit UI        | Interactive form for entering equipment data             |
| 📁 CSV Logger          | Logs inputs into `sensor_data.csv`                        |
| 🧠 ML Predictor        | Trains and compares 9 ML models, saves best model        |
| 📊 Dashboard           | Shows trends, anomalies, and prediction outputs          |
| 🤖 AI Chatbot         | Voice + text-based assistant powered by NLP modules     |
| 🧾 Model Tracker      | Records model accuracy to `model_performance.csv`         |
| 🧠 Failure Logger      | Saves predicted failure data to `predicted_failures.csv` |
| ☁️ Azure Ready        | ready to be executed on Azure data bricks  |

---

## 🛠 Tech Stack (Free Tools)

- Python, Streamlit, scikit-learn
- Joblib (model persistence)
- pyttsx3, speechrecognition (chatbot)
- Git + GitHub
- VS Code

---

## 📁 Full Project Structure

```
safety-assistant/
├── __pycache__/
├── app.py
├── chatbot_module.py
├── dashboard.py
├── failure_model.pkl
├── best_model.pkl
├── chat_log.csv
├── model_performance.csv
├── predicted_failures.csv
├── ml_predictor.py
├── nlp_module.py
├── sensor_data.csv
├── sensor_data_generator.py
├── requirements.txt
└── README.md
```

---

## 🧠 Machine Learning Pipeline

Trained on:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- Support Vector Machine (SVM)
- K-Nearest Neighbors
- Naive Bayes

✅ Best model saved as `failure_model.pkl`  
📊 Accuracy results saved in `model_performance.csv`

---

## ▶️ How to Run This Project

```bash
git clone https://github.com/Tarunsaib9/smart-safety-assistant
cd safety-assistant
pip install -r requirements.txt
python ml_predictor.py
streamlit run app.py
```

---

## 📊 Dashboard Features

- View real-time temperature, pressure, and vibration inputs
- Visualize failure rates and model performance
- Analyze high-risk entries based on thresholds

---

## 🤖 Chatbot Functionality

- Ask questions like “Is this equipment safe?”
- Get predictions based on the trained model
- Uses `chatbot_module.py` and `nlp_module.py`
- Logs chat history to `chat_log.csv`

---

## 🧪 Sample Prediction Example

| Temperature | Pressure | Vibration | Failure Prediction |
|-------------|----------|-----------|--------------------|
| 85.72       | 1.46     | 0.33      | 1 (Likely Failure) |

---

## 👨‍🔬 Author

**Tarun Sai**  
IT Data Scientist | AI & Data Enthusiast | Full Stack Developer | DevSecOps & Cloud Engineer  
📍 Kincardine, Canada  
📧 tarunsaib9@gmail.com

---

## 🏁 Final Thoughts

Smart Safety Assistant is a real-time, modular, and scalable platform built with free tools. It demonstrates how AI and data science can enhance workplace safety (mission of innovation and operational excellence).
