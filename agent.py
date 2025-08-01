from dotenv import load_dotenv
load_dotenv()
import requests
import os

API_URL = "https://api.groq.com/openai/v1/chat/completions"
API_KEY = os.getenv("GROQ_API_KEY")  # Groq requires an API key now

def ai_chat_agent():
    print("🤖 AI Agent (Groq + LLaMA3) is ready! Type 'exit' to quit.\n")
    messages = []

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Agent: Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "llama3-70b-8192",  # You can use smaller ones too: llama3-8b-8192
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 512
            }

            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()

            reply = response.json()['choices'][0]['message']['content']
            print("Agent:", reply)

            messages.append({"role": "assistant", "content": reply})

        except Exception as e:
            print("⚠️ Error:", e)

if __name__ == "__main__":
    ai_chat_agent()
