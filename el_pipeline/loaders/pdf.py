from pathlib import Path
from typing import Iterator

import pdfplumber

from el_pipeline.registry import loaders
from el_pipeline.types import Document


@loaders.register("pdf")
class PDFLoader:
    """Extracts text from PDF files."""

    def load(self, path: str) -> Iterator[Document]:
        with pdfplumber.open(path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        text = "\n".join(pages)
        yield Document(id=Path(path).stem, text=text, meta={"source": path})

