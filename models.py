from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Fact:
    text: str
    tags: List[str]
