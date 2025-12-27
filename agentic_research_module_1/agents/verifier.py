from llm.chatgpt_client import ChatGPTClient
from llm.grok_client import LLMClient


class VerifierAgent:
    SYSTEM_PROMPT = (
        "You are a strict research verifier. "
        "Reject vague or low-quality answers."
    )

    def verify(self, research_data: dict) -> bool:
        user_prompt = f"""
        Evaluate the following research answer.

        Question:
        {research_data['question']}

        Answer:
        {research_data['content']}

        Respond ONLY with:
        - APPROVED
        - REJECTED
        """

        verdict = LLMClient.ask(self.SYSTEM_PROMPT, user_prompt)
        return verdict.strip().upper() == "APPROVED"
