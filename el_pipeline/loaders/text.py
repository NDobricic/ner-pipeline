import json
from pathlib import Path
from typing import Iterator

from el_pipeline.registry import loaders
from el_pipeline.types import Document


@loaders.register("text")
class TextLoader:
    """Loads plain text files."""

    def load(self, path: str) -> Iterator[Document]:
        text = Path(path).read_text(encoding="utf-8")
        yield Document(id=Path(path).stem, text=text, meta={"source": path})


@loaders.register("jsonl")
class JSONLLoader:
    """Loads JSONL where each line has a `text` field."""

    def __init__(self, text_field: str = "text") -> None:
        self.text_field = text_field

    def load(self, path: str) -> Iterator[Document]:
        with Path(path).open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                text = data.get(self.text_field, "")
                doc_id = data.get("id") or f"{Path(path).stem}-{i}"
                yield Document(id=doc_id, text=text, meta={"source": path, **data})


@loaders.register("json")
class JSONLoader:
    """Loads JSON array with objects containing `text`."""

    def __init__(self, text_field: str = "text") -> None:
        self.text_field = text_field

    def load(self, path: str) -> Iterator[Document]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = [data]
        for i, item in enumerate(data):
            text = item.get(self.text_field, "")
            doc_id = item.get("id") or f"{Path(path).stem}-{i}"
            yield Document(id=doc_id, text=text, meta={"source": path, **item})

