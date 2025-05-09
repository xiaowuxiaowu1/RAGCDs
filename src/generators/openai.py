import os
from typing import Optional

from openai import AzureOpenAI
from dotenv import load_dotenv
from .base_generator import BaseGenerator

load_dotenv()  

class OpenAIGenerator(BaseGenerator):
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None) -> None:
        self.model = model                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        self.client = AzureOpenAI(
            azure_deployment="gpt-4o",
            azure_endpoint=os.environ.get("AZURE_ENDPOINT_LLM_GPT4O"),
            api_key=api_key if api_key else os.environ.get("API_KEY_LLM_GPT4O"),
            api_version=os.environ.get("API_VERSION_LLM_GPT4O"),
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