# frontend/dashboard.py

import streamlit as st
import requests
import pandas as pd
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Import chatbot logic from backend.chatbot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.chatbot import ask_chatbot  # Make sure backend is a proper Python module with __init__.py

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Network Intrusion Detection System", layout="centered")

# ---------- USER LOGIN SYSTEM ----------
USER_CREDENTIALS = {"admin": "admin123", "user": "user123"}
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Login Required")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

# ---------- SIDEBAR NAVIGATION ----------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Select Page", ["Dashboard", "Chatbot Assistant"])

# ---------- PAGE 1: MAIN DASHBOARD ----------
if page == "Dashboard":
    st.title("🚨 Network Intrusion Detection System")
    st.write("Enter packet details to check if it's normal or an attack.")

    # --- INPUT FIELDS ---
    duration = st.number_input("Duration", min_value=0.0, value=0.2)
    protocol = st.selectbox("Protocol Type", ["tcp", "udp", "icmp"])
    service = st.number_input("Service (as integer)", min_value=0)
    src_bytes = st.number_input("Source Bytes", min_value=0)
    dst_bytes = st.number_input("Destination Bytes", min_value=0)
    flag = st.selectbox("Flag", ["SF", "S0", "REJ", "RSTO", "RSTR", "SH", "OTH"])

    protocol_mapping = {"tcp": 1, "udp": 2, "icmp": 3}
    flag_mapping = {"SF": 0, "S0": 1, "REJ": 2, "RSTO": 3, "RSTR": 4, "SH": 5, "OTH": 6}
    protocol_type = protocol_mapping[protocol]
    flag_val = flag_mapping[flag]

    if st.button("🔍 Predict"):
        packet = {
            "duration": duration,
            "protocol_type": protocol_type,
            "service": service,
            "src_bytes": src_bytes,
            "dst_bytes": dst_bytes,
            "flag": flag_val
        }

        try:
            url = "http://127.0.0.1:8000/predict"
            response = requests.post(url, json=packet)
            if response.status_code == 200:
                prediction = response.json()["prediction"]
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                status = "Normal" if prediction == 0 else "Attack"

                if prediction == 0:
                    st.success("✅ Normal Traffic")
                else:
                    st.error("⚠️ Attack Detected!")

                # --- Log to CSV ---
                log_data = packet.copy()
                log_data.update({"timestamp": timestamp, "prediction": status})
                log_df = pd.DataFrame([log_data])
                if os.path.exists("logs.csv"):
                    log_df.to_csv("logs.csv", mode="a", header=False, index=False)
                else:
                    log_df.to_csv("logs.csv", index=False)
            else:
                st.error(f"❌ Failed to get prediction. Status code: {response.status_code}")
        except Exception as e:
            st.error(f"🚫 Error: {str(e)}")

    # --- VISUALIZATION ---
    st.markdown("---")
    st.subheader("📊 Traffic Analysis Dashboard")

    if os.path.exists("logs.csv"):
        logs = pd.read_csv("logs.csv")
        logs["timestamp"] = pd.to_datetime(logs["timestamp"])

        min_date = logs["timestamp"].min().date()
        max_date = logs["timestamp"].max().date()
        date_range = st.date_input("Select date range", [min_date, max_date])

        filtered = logs[(logs["timestamp"].dt.date >= date_range[0]) & (logs["timestamp"].dt.date <= date_range[1])]

        st.write("### Traffic Classification")
        st.bar_chart(filtered["prediction"].value_counts())

        protocol_labels = {1: "tcp", 2: "udp", 3: "icmp"}
        filtered["protocol_type"] = filtered["protocol_type"].map(protocol_labels)

        st.write("### Protocol Distribution")
        fig1, ax1 = plt.subplots()
        filtered["protocol_type"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax1)
        ax1.set_ylabel("")
        st.pyplot(fig1)

        st.download_button("⬇ Download Filtered Logs", data=filtered.to_csv(index=False), file_name="filtered_logs.csv")
    else:
        st.info("No logs found yet. Perform a prediction to generate logs.")

    # --- EMAIL ALERT LOG VIEW ---
    st.markdown("---")
    st.subheader("📧 Email Alert Logs")

    log_path = "email_alerts.log"
    if os.path.exists(log_path):
        with open(log_path, "r") as file:
            logs = file.readlines()
            if logs:
                for line in logs[-10:]:
                    st.text(line.strip())
            else:
                st.info("No email alerts triggered yet.")
    else:
        st.warning("Email alert log file not found.")

    if st.button("🧹 Clear Alert Logs"):
        open(log_path, "w").close()
        st.success("Alert logs cleared.")

# ---------- PAGE 2: CHATBOT ----------
elif page == "Chatbot Assistant":
    st.title("🤖 Chatbot Assistant - Cybersecurity Help")
    st.markdown("Ask anything about cybersecurity threats, attacks, or how to use this dashboard.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("You:", key="user_input")
    if user_question:
        with st.spinner("Thinking..."):
            bot_reply = ask_chatbot(user_question)
        st.session_state.chat_history.append(("You", user_question))
        st.session_state.chat_history.append(("Bot", bot_reply))

    for role, msg in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**🧑 {role}:** {msg}")
        else:
            st.markdown(f"**🤖 {role}:** {msg}")
