from dotenv import load_dotenv
import requests
import os
from gmail_reader import get_latest_email

load_dotenv()

API_URL = "https://api.groq.com/openai/v1/chat/completions"
API_KEY = os.getenv("GROQ_API_KEY")

def call_groq(messages):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-70b-8192",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

def ai_chat_agent():
    print("🤖 Advanced AI Agent (Groq + LLaMA3 + Gmail) is ready! Type 'exit' to quit.\n")
    messages = []

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Agent: Goodbye!")
            break

        if "gmail" in user_input.lower() or "check my email" in user_input.lower():
            print("📥 Checking Gmail...")
            gmail_info = get_latest_email()
            print("Agent (Gmail):", gmail_info)
            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": gmail_info})
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            reply = call_groq(messages)
            print("Agent:", reply)
            messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            print("⚠️ Error:", e)

if __name__ == "__main__":
    ai_chat_agent()
