from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from backend.utils import extract_features
import smtplib
from email.message import EmailMessage
import os
from dotenv import load_dotenv
from backend.chatbot import ask_chatbot  # ✅ NEW

# ✅ Load environment variables
load_dotenv()

EMAIL_SENDER = os.getenv("EMAIL_SENDER")  # Corrected .env key
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

app = FastAPI()

# ✅ Load trained model
with open("backend/model.pkl", "rb") as f:
    model = pickle.load(f)

# ✅ Input schema
class Packet(BaseModel):
    duration: float
    protocol_type: int
    service: int
    src_bytes: int
    dst_bytes: int
    flag: int

# ✅ Email Alert Function
def send_email_alert(packet_data: dict):
    message = EmailMessage()
    message.set_content(f"🚨 Alert: Anomaly Detected!\n\nPacket Info:\n{packet_data}")
    message["Subject"] = "🚨 Network Intrusion Detected!"
    message["From"] = EMAIL_SENDER
    message["To"] = EMAIL_RECEIVER

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(message)
        print("✅ Email sent successfully.")
    except Exception as e:
        print(f"❌ Email failed: {e}")

# ✅ Prediction endpoint
@app.post("/predict")
async def predict(packet: Packet):
    try:
        features = extract_features(packet.dict())
        columns = ["duration", "protocol_type", "service", "src_bytes", "dst_bytes", "flag"]
        input_df = pd.DataFrame([features], columns=columns)

        prediction = model.predict(input_df)[0]

        if prediction == 1:
            send_email_alert(packet.dict())  # Only for anomaly

        return {"prediction": int(prediction)}

    except Exception as e:
        return {"error": str(e)}

# ✅ NEW: Chatbot input schema and endpoint
class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_with_bot(chat: ChatRequest):
    try:
        reply = ask_chatbot(chat.question)
        return {"response": reply}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}
