import os
from typing import List, Dict

from openai import OpenAI


class OpenAILLM:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI()

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
