import re
from typing import List

from ner_pipeline.registry import ner_models
from ner_pipeline.types import Mention


@ner_models.register("simple")
class SimpleRegexNER:
    """Lightweight regex-based NER for quick tests (no external deps)."""

    def __init__(self, min_len: int = 3):
        self.pattern = re.compile(
            r"\b([A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+)*)\b"
        )
        self.min_len = min_len

    def extract(self, text: str) -> List[Mention]:
        mentions: List[Mention] = []
        for match in self.pattern.finditer(text):
            span = match.group(1)
            if len(span) < self.min_len:
                continue
            mentions.append(
                Mention(
                    start=match.start(1),
                    end=match.end(1),
                    text=span,
                    label="ENT",
                )
            )
        return mentions

