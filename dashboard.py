import pandas as pd
import streamlit as st
import os
from datetime import datetime

st.set_page_config(page_title="📊 Smart Safety Dashboard", layout="wide")
st.title("📊 Smart Safety Assistant Dashboard")
st.markdown("Real-time AI-powered monitoring system with ML & NLP insights.")

# === Load predicted failure data ===
if os.path.exists("predicted_failures.csv"):
    df = pd.read_csv("predicted_failures.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
else:
    st.error("❌ predicted_failures.csv not found.")
    st.stop()

# === Sidebar filter ===
st.sidebar.header("🔎 Filter by Date")
start = st.sidebar.date_input("Start", df['timestamp'].min().date())
end = st.sidebar.date_input("End", df['timestamp'].max().date())
mask = (df['timestamp'].dt.date >= start) & (df['timestamp'].dt.date <= end)
df = df.loc[mask]

# === KPIs ===
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(df))
col2.metric("Failures Predicted", int(df['predicted_failure'].sum()))
col3.metric("Failure Rate (%)", f"{100 * df['predicted_failure'].mean():.2f}%")

rate = df['predicted_failure'].mean()
if rate > 0.5:
    st.error("🚨 High failure rate. Take action immediately.")
elif rate > 0.2:
    st.warning("⚠️ Medium failure risk. Monitor closely.")
else:
    st.success("✅ All systems operating within normal range.")

st.markdown("---")

# === Data Table ===
st.subheader("📋 Failure Prediction Data")
st.dataframe(df.head(10), use_container_width=True)

# === Line Chart ===
st.subheader("📉 Failure Trends")
trend = df.groupby(df['timestamp'].dt.strftime('%H:%M'))['predicted_failure'].sum()
st.line_chart(trend)

# === Bar Chart ===
st.subheader("🌡️ Failure Distribution by Temperature")
df['TempLevel'] = pd.cut(df['temperature'], bins=[0, 70, 85, 100], labels=["Cool", "Warm", "Hot"])
temp_dist = df[df['predicted_failure'] == 1].groupby('TempLevel').size()
st.bar_chart(temp_dist)

# === Scatter Plot ===
st.subheader("🧪 Pressure vs Vibration (Failures Only)")
fail_df = df[df['predicted_failure'] == 1]
if not fail_df.empty:
    st.scatter_chart(fail_df[['pressure', 'vibration']])
else:
    st.warning("No failures in current selection.")

# === Optional: Chat Logs ===
if os.path.exists("chat_log.csv"):
    st.subheader("💬 Chatbot Interactions")
    chat_df = pd.read_csv("chat_log.csv")
    st.dataframe(chat_df.tail(10))

# === Optional: Model Performance ===
if os.path.exists("model_performance.csv"):
    st.subheader("🧠 Model Comparison")
    perf = pd.read_csv("model_performance.csv")
    st.dataframe(perf)
    st.bar_chart(perf.set_index("Model"))

st.markdown("---")
st.caption("Built with Python, Streamlit, scikit-learn, and DevOps mindset.")
