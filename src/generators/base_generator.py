from abc import ABC, abstractmethod
from typing import Optional

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = "You are a helpful AI assistant.") -> str:
        pass
