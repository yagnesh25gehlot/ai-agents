from dotenv import load_dotenv
load_dotenv()

import os
import requests
from datetime import datetime
import re

API_URL = "https://api.groq.com/openai/v1/chat/completions"
API_KEY = os.getenv("GROQ_API_KEY")

# === Tool Functions ===
def get_current_time():
    return f"The current time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

def calculator(expression):
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Could not calculate: {e}"

def web_search(query):
    return f"🔍 Web search not implemented yet. You asked to search: '{query}'"

# === AI Agent Logic ===
def ai_chat_agent():
    print("🤖 Advanced AI Agent (Groq + LLaMA3 + Tools) is ready! Type 'exit' to quit.\n")
    messages = []

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Agent: Goodbye!")
            break

        # === Tool Invocation Check ===
        if "time" in user_input.lower():
            tool_reply = get_current_time()
        elif re.search(r"[\d\.\+\-\*/\(\)]", user_input) and any(op in user_input for op in ['+', '-', '*', '/', '(', ')']):
            tool_reply = calculator(user_input)
        elif "search" in user_input.lower():
            query = user_input.replace("search", "").strip()
            tool_reply = web_search(query)
        else:
            tool_reply = None

        if tool_reply:
            print("🛠️ Tool:", tool_reply)
            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": tool_reply})
            continue

        messages.append({"role": "user", "content": user_input})

        try:
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

            reply = response.json()['choices'][0]['message']['content']
            print("Agent:", reply)
            messages.append({"role": "assistant", "content": reply})

        except Exception as e:
            print("⚠️ Error:", e)

if __name__ == "__main__":
    ai_chat_agent()
