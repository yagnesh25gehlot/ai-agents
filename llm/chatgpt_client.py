import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ChatGPTClient:
    @staticmethod
    def ask(system_prompt: str, user_prompt: str) -> str:
        """
        Generic ChatGPT call
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",   # fast + cheap; change if needed
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
