from llm.chatgpt_client import ChatGPTClient
from llm.llm_client import LLMClient


class SynthesizerAgent:
    SYSTEM_PROMPT = (
        "You are a senior technical writer. "
        "Create a structured research summary."
    )

    def synthesize(self, verified_data: list[dict]) -> str:
        content_blocks = "\n\n".join(
            f"Q: {item['question']}\nA: {item['content']}"
            for item in verified_data
        )

        user_prompt = f"""
        Create a well-structured research report from the following content.

        Requirements:
        - Clear sections
        - Bullet points where useful
        - Professional tone

        Content:
        {content_blocks}
        """

        return LLMClient.ask(self.SYSTEM_PROMPT, user_prompt)
