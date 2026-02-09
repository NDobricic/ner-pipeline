from pathlib import Path
from typing import Iterator

from bs4 import BeautifulSoup

from el_pipeline.registry import loaders
from el_pipeline.types import Document


@loaders.register("html")
class HTMLLoader:
    """Parses HTML and extracts visible text."""

    def load(self, path: str) -> Iterator[Document]:
        html = Path(path).read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "lxml")
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text("\n")
        yield Document(id=Path(path).stem, text=text, meta={"source": path})

