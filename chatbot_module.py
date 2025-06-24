import streamlit as st
from datetime import datetime
import pandas as pd
import os
import pyttsx3
import speech_recognition as sr
import threading
from nlp_module import analyze_text, generate_response

# ============ Voice Setup ============
engine = pyttsx3.init()

def speak(text):
    def run_speak():
        engine.say(text)
        engine.runAndWait()
    try:
        threading.Thread(target=run_speak).start()
    except RuntimeError:
        pass

# ============ Save Chat Log ============
def save_chat_log(user, bot):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = pd.DataFrame([[timestamp, user, bot]], columns=["Time", "User", "Bot"])
    try:
        log_entry.to_csv("chat_log.csv", mode="a", header=not pd.read_csv("chat_log.csv").empty, index=False)
    except FileNotFoundError:
        log_entry.to_csv("chat_log.csv", index=False)

# ============ Streamlit App ============
st.set_page_config(page_title="Smart Safety Assistant", page_icon="🤖")
st.title("💬 Smart Safety Assistant")
st.markdown("Ask me about temperature, vibration, failures, or Bruce Power!")

# Sidebar: Voice toggle
st.sidebar.header("🔊 Voice Settings")
enable_voice = st.sidebar.checkbox("Enable Voice Output", value=True)

# Clear log button
if st.button("🧹 Clear Chat Log"):
    open("chat_log.csv", "w").close()
    st.success("Chat log cleared!")

# TEXT input
user_text = st.text_input("🧠 Type your question:")

# VOICE input
if st.button("🎙️ Use Voice"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        st.write("🗣️ You said:", text)
        user_text = text
    except sr.UnknownValueError:
        st.error("Sorry, I couldn't understand your voice.")
    except sr.RequestError:
        st.error("Speech service not available.")

# Chat handling
if user_text:
    response = generate_response(user_text)
    st.success(response)
    save_chat_log(user_text, response)

    if enable_voice:
        speak(response)

    with st.expander("🔍 NLP Analysis"):
        result = analyze_text(user_text)
        st.write("**Tokens:**", result["tokens"])
        st.write("**POS Tags:**", result["pos_tags"])
        st.write("**Named Entities:**", result["entities"])

# Sidebar analytics
st.sidebar.header("📊 Chat Log Summary")
if os.path.exists("chat_log.csv"):
    df = pd.read_csv("chat_log.csv")
    st.sidebar.write(f"🧾 Total Chats: {len(df)}")
    st.sidebar.dataframe(df.tail(5))
