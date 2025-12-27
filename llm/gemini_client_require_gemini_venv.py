
from dotenv import load_dotenv
from PIL import Image
from google import genai

load_dotenv()


class LLMClient:
    client = genai.Client(api_key="AIzaSyAFrMI-ZFhSbgWPI7Yw3IpEb2a6Z6IxGcY")
    model = "gemini-2.5-flash"  # ✅ CORRECT

    @staticmethod
    def ask(user_prompt: str, image_path: str = None) -> str:
        if image_path:
            image = Image.open(image_path)
            response = LLMClient.client.models.generate_content(
                model=LLMClient.model,
                contents=[user_prompt, image]
            )
        else:
            response = LLMClient.client.models.generate_content(
                model=LLMClient.model,
                contents=user_prompt
            )

        return response.text.strip()
