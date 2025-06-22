# frontend/chat_ui.py

import streamlit as st
import sys
import os

# Ensure backend directory is in the path to import chatbot.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from chatbot import ask_chatbot

st.set_page_config(page_title="Cybersecurity Chatbot", page_icon="🛡️", layout="centered")

st.title("🛡️ Cybersecurity & Network Support Chatbot")

st.markdown("""
Welcome! This AI chatbot can help explain various cybersecurity threats and network-related issues.
You can ask about:
- Cyber attacks (e.g., Phishing, DDoS, SQL Injection)
- Network threats and solutions
- Cybersecurity tools and practices
""")

# Session state to store chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask about a cybersecurity or network topic...")

if user_input:
    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get chatbot response
    with st.spinner("Thinking..."):
        bot_response = ask_chatbot(user_input)

    # Show chatbot response
    st.chat_message("assistant").markdown(bot_response)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
