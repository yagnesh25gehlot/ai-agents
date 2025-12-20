from llm.chatgpt_client import ChatGPTClient
from llm.llm_client import LLMClient


class ResearchAgent:
    SYSTEM_PROMPT = (
        "You are a technical researcher. "
        "Provide accurate, factual, and concise information."
    )

    def research(self, question: str) -> dict:
        user_prompt = f"""
        Research the following question:

        "{question}"

        Provide:
        - Clear explanation
        - Practical examples if applicable
        - No fluff
        """

        content = LLMClient.ask(self.SYSTEM_PROMPT, user_prompt)

        return {
            "question": question,
            "content": content
        }
