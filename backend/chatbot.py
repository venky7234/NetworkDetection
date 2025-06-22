import openai
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Raise error if API key is not found
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY is not set in environment variables.")

# Set the API key
openai.api_key = OPENAI_API_KEY

def ask_chatbot(question: str) -> str:
    """
    Sends a question to the OpenAI chatbot and returns the response.
    
    Args:
        question (str): The user's question.
    
    Returns:
        str: The chatbot's response.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can change to 'gpt-4' if you have access
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful cybersecurity expert. Help users understand cybersecurity threats, "
                        "network vulnerabilities, common attack types (e.g., DDoS, Phishing, SQL Injection), "
                        "and best practices for prevention and response."
                    )
                },
                {"role": "user", "content": question}
            ],
            temperature=0.6,
            max_tokens=800
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        # Return a clear message for frontend or logs
        return f"⚠️ Chatbot Error: {str(e)}"
