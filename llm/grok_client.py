import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class LLMClient:
    @staticmethod
    def ask(system_prompt: str = None, user_prompt: str = None) -> str:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # FREE + excellent
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()


    @staticmethod
    def get_response(prompt: str = None):
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # FREE + excellent
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message
