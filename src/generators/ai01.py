import os
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv
from .base_generator import BaseGenerator

load_dotenv()  

class AI01Generator(BaseGenerator):
    def __init__(self, model: str = "yi-large", api_key: Optional[str] = None) -> None:
        self.model = model                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        self.client = OpenAI(
            api_key=api_key if api_key else os.environ.get("API_KEY_LLM_01"),
            base_url=os.environ.get("ENDPOINT_LLM_01")
        )

    def generate(self, prompt: str, system_prompt: str = "You are a helpful AI assistant."):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content