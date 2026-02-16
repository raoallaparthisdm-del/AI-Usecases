from __future__ import annotations

from typing import Dict, List

from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from retriever import SimpleRetriever
from llm import OpenAILLM


def format_facts(facts: List[str]) -> str:
    if not facts:
        return "- No relevant facts found."
    return "\n".join(f"- {fact}" for fact in facts)


def format_memory(memory: List[Dict[str, str]], max_turns: int = 6) -> str:
    if not memory:
        return "(none)"
    trimmed = memory[-max_turns:]
    lines = []
    for item in trimmed:
        role = "User" if item["role"] == "user" else "Assistant"
        lines.append(f"{role}: {item['content']}")
    return "\n".join(lines)


class RAGEngine:
    def __init__(self, retriever: SimpleRetriever, llm: OpenAILLM) -> None:
        self.retriever = retriever
        self.llm = llm

    def answer(self, question: str, memory: List[Dict[str, str]]) -> str:
        facts = self.retriever.retrieve(question)
        fact_texts = [f.text for f in facts]

        user_prompt = USER_PROMPT_TEMPLATE.format(
            facts=format_facts(fact_texts),
            memory=format_memory(memory),
            question=question,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        return self.llm.generate(messages)
