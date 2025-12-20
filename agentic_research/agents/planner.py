from llm.chatgpt_client import ChatGPTClient
from llm.llm_client import LLMClient


class PlannerAgent:
    SYSTEM_PROMPT = (
        "You are a research planner. "
        "Your job is to break a topic into clear research questions."
    )

    def create_plan(self, topic: str) -> list[str]:
        user_prompt = f"""
        Create a concise research plan for the topic: "{topic}"

        Rules:
        - Return 4 to 6 bullet points
        - Each bullet must be a question
        - No extra explanation
        """

        response = LLMClient.ask(self.SYSTEM_PROMPT, user_prompt)

        return [
            line.lstrip("- ").strip()
            for line in response.splitlines()
            if line.strip()
        ]
