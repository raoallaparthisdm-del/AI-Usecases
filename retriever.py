from __future__ import annotations

import re
from typing import Iterable, List

from models import Fact


STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "by",
    "with",
    "is",
    "are",
    "was",
    "were",
    "on",
    "at",
    "from",
    "as",
    "show",
    "me",
    "give",
    "tell",
    "about",
}


def normalize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


class SimpleRetriever:
    def __init__(self, facts: Iterable[Fact]) -> None:
        self.facts = list(facts)

    def retrieve(self, query: str, top_k: int = 8) -> List[Fact]:
        query_terms = set(normalize(query))
        scored = []
        for fact in self.facts:
            tag_score = len(query_terms.intersection({t.lower() for t in fact.tags}))
            text_score = len(query_terms.intersection(set(normalize(fact.text))))
            score = tag_score * 2 + text_score
            if score > 0:
                scored.append((score, fact))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in scored[:top_k]]
