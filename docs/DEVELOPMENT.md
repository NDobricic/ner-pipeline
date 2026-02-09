# Development Guide

This guide explains how to extend the EL Pipeline by creating custom loaders, knowledge bases, and spaCy components.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Creating Custom Loaders](#creating-custom-loaders)
- [Creating Custom Knowledge Bases](#creating-custom-knowledge-bases)
- [Creating Custom spaCy Components](#creating-custom-spacy-components)
- [Component Compatibility Matrix](#component-compatibility-matrix)
- [Registry System Overview](#registry-system-overview)

---

## Architecture Overview

The EL Pipeline uses two extension mechanisms:

1. **Registry-based** (loaders, knowledge bases): Simple decorator registration
2. **spaCy factories** (NER, candidates, rerankers, disambiguators): spaCy's `@Language.factory` decorator

```
┌───────────────────────────────────────────────────────────────┐
│                        ELPipeline                            │
├───────────────────────────────────────────────────────────────┤
│  Registry Components          │  spaCy Components             │
│  ├── Loaders                  │  ├── NER                      │
│  │   (text, pdf, json...)     │  │   (lela_gliner, simple...) │
│  └── Knowledge Bases          │  ├── Candidates               │
│      (custom, lela_jsonl...)  │  │   (bm25, fuzzy, dense...)  │
│                               │  ├── Rerankers                │
│                               │  │   (embedder, cross_encoder)│
│                               │  └── Disambiguators           │
│                               │      (vllm, first, popularity)│
└───────────────────────────────────────────────────────────────┘
```

---

## Creating Custom Loaders

Loaders parse input files and yield `Document` objects.

### Loader Protocol

```python
from typing import Iterator
from el_pipeline.types import Document

class LoaderProtocol:
    def load(self, path: str) -> Iterator[Document]:
        """Load documents from a file path."""
        ...
```

### Example: Custom CSV Loader

```python
import csv
from pathlib import Path
from typing import Iterator

from el_pipeline.registry import loaders
from el_pipeline.types import Document


@loaders.register("csv")
class CSVLoader:
    """
    Loads CSV files where each row becomes a document.

    Config params:
        text_column: Column name containing the text (default: "text")
        id_column: Column name for document ID (default: "id")
    """

    def __init__(self, text_column: str = "text", id_column: str = "id"):
        self.text_column = text_column
        self.id_column = id_column

    def load(self, path: str) -> Iterator[Document]:
        with Path(path).open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                text = row.get(self.text_column, "")
                doc_id = row.get(self.id_column) or f"{Path(path).stem}-{i}"
                yield Document(
                    id=doc_id,
                    text=text,
                    meta={"source": path, **row}
                )
```

### Using Your Custom Loader

**In configuration:**
```json
{
  "loader": {
    "name": "csv",
    "params": {
      "text_column": "content",
      "id_column": "doc_id"
    }
  }
}
```

**In Python:**
```python
from el_pipeline.registry import loaders

# Import to register
from your_module import CSVLoader

# Use via registry
loader = loaders.get("csv")(text_column="content")
for doc in loader.load("data.csv"):
    print(doc.text)
```

### Built-in Loader Reference

| Name | Class | Location |
|------|-------|----------|
| `text` | `TextLoader` | `el_pipeline/loaders/text.py` |
| `json` | `JSONLoader` | `el_pipeline/loaders/text.py` |
| `jsonl` | `JSONLLoader` | `el_pipeline/loaders/text.py` |
| `pdf` | `PDFLoader` | `el_pipeline/loaders/pdf.py` |
| `docx` | `DocxLoader` | `el_pipeline/loaders/docx.py` |
| `html` | `HTMLLoader` | `el_pipeline/loaders/html.py` |

---

## Creating Custom Knowledge Bases

Knowledge bases provide entity lookup and search functionality.

### Knowledge Base Protocol

```python
from typing import Dict, Iterable, List, Optional
from el_pipeline.types import Entity

class KnowledgeBaseProtocol:
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        ...

    def search(self, query: str, top_k: int = 10) -> List[Entity]:
        """Search entities by query."""
        ...

    def all_entities(self) -> Iterable[Entity]:
        """Iterate over all entities."""
        ...
```

### Example: SQLite Knowledge Base

```python
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from el_pipeline.registry import knowledge_bases
from el_pipeline.types import Entity


@knowledge_bases.register("sqlite")
class SQLiteKnowledgeBase:
    """
    Knowledge base backed by SQLite database.

    Expected schema:
        CREATE TABLE entities (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT
        );

    Config params:
        path: Path to SQLite database file
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        cursor = self.conn.execute(
            "SELECT id, title, description FROM entities WHERE id = ?",
            (entity_id,)
        )
        row = cursor.fetchone()
        if row:
            return Entity(
                id=row["id"],
                title=row["title"],
                description=row["description"]
            )
        return None

    def search(self, query: str, top_k: int = 10) -> List[Entity]:
        # Simple LIKE-based search
        cursor = self.conn.execute(
            """
            SELECT id, title, description FROM entities
            WHERE title LIKE ? OR description LIKE ?
            LIMIT ?
            """,
            (f"%{query}%", f"%{query}%", top_k)
        )
        return [
            Entity(id=row["id"], title=row["title"], description=row["description"])
            for row in cursor.fetchall()
        ]

    def all_entities(self) -> Iterable[Entity]:
        cursor = self.conn.execute("SELECT id, title, description FROM entities")
        for row in cursor:
            yield Entity(
                id=row["id"],
                title=row["title"],
                description=row["description"]
            )

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()
```

### Using Your Custom Knowledge Base

**In configuration:**
```json
{
  "knowledge_base": {
    "name": "sqlite",
    "params": {
      "path": "entities.db"
    }
  }
}
```

### Built-in Knowledge Base Reference

| Name | Class | Location |
|------|-------|----------|
| `custom` | `CustomJSONLKnowledgeBase` | `el_pipeline/knowledge_bases/custom.py` |
| `lela_jsonl` | `LELAJSONLKnowledgeBase` | `el_pipeline/knowledge_bases/lela_jsonl.py` |

**Note:** `CustomJSONLKnowledgeBase` supports persistent caching via the `cache_dir` parameter. When provided, parsed KB data is cached to disk and reused on subsequent loads, significantly reducing initialization time for large knowledge bases.

---

## Creating Custom spaCy Components

spaCy components are the core processing units. They use spaCy's factory pattern.

### Component Types

| Type | Input | Output | Extension |
|------|-------|--------|-----------|
| NER | `doc.text` | `doc.ents` | `ent._.context` |
| Candidates | `doc.ents` | (unchanged) | `ent._.candidates` |
| Reranker | `ent._.candidates` | (reordered) | `ent._.candidates` |
| Disambiguator | `ent._.candidates` | (unchanged) | `ent._.resolved_entity` |

### Example: Custom NER Component

```python
import re
from typing import List, Optional

from spacy.language import Language
from spacy.tokens import Doc, Span

from el_pipeline.context import extract_context
from el_pipeline.utils import filter_spans, ensure_context_extension


@Language.factory(
    "el_pipeline_email_ner",
    default_config={
        "context_mode": "sentence",
    },
)
def create_email_ner_component(
    nlp: Language,
    name: str,
    context_mode: str,
):
    """Factory for email NER component."""
    return EmailNERComponent(nlp=nlp, context_mode=context_mode)


class EmailNERComponent:
    """
    NER component that extracts email addresses.

    Demonstrates the pattern for custom NER components.
    """

    def __init__(self, nlp: Language, context_mode: str = "sentence"):
        self.nlp = nlp
        self.context_mode = context_mode
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )

        # Register extension using shared utility
        ensure_context_extension()

    def __call__(self, doc: Doc) -> Doc:
        """Process document and add email entities."""
        text = doc.text
        spans = []

        for match in self.email_pattern.finditer(text):
            start_char = match.start()
            end_char = match.end()

            # Convert character offsets to token offsets
            span = doc.char_span(start_char, end_char, alignment_mode="expand")
            if span is None:
                continue

            # Create span with EMAIL label
            new_span = Span(doc, span.start, span.end, label="EMAIL")

            # Extract context
            context = extract_context(text, start_char, end_char, mode=self.context_mode)
            new_span._.context = context

            spans.append(new_span)

        # Set entities using shared utility (handles overlap filtering)
        doc.ents = filter_spans(spans)
        return doc
```

### Example: Custom Candidate Generator

```python
from typing import List, Tuple, Optional

from spacy.language import Language
from spacy.tokens import Doc, Span

from el_pipeline.knowledge_bases.base import KnowledgeBase
from el_pipeline.utils import ensure_candidates_extension


@Language.factory(
    "el_pipeline_exact_candidates",
    default_config={
        "top_k": 10,
    },
)
def create_exact_candidates_component(
    nlp: Language,
    name: str,
    top_k: int,
):
    """Factory for exact match candidate generator."""
    return ExactMatchCandidatesComponent(nlp=nlp, top_k=top_k)


class ExactMatchCandidatesComponent:
    """
    Candidate generator using exact title matching.

    Demonstrates the pattern for custom candidate generators.
    """

    def __init__(self, nlp: Language, top_k: int = 10):
        self.nlp = nlp
        self.top_k = top_k
        self.kb = None
        self._entity_titles = {}

        # Register extensions using shared utility
        ensure_candidates_extension()

    def initialize(self, kb: KnowledgeBase):
        """Initialize with knowledge base."""
        self.kb = kb
        # Build title index
        self._entity_titles = {
            entity.title.lower(): entity
            for entity in kb.all_entities()
        }

    def __call__(self, doc: Doc) -> Doc:
        """Generate candidates for each entity."""
        if self.kb is None:
            return doc

        for ent in doc.ents:
            mention_lower = ent.text.lower()
            candidates = []
            scores = []

            # Exact match
            if mention_lower in self._entity_titles:
                entity = self._entity_titles[mention_lower]
                candidates.append((entity.title, entity.description or ""))
                scores.append(1.0)

            # Partial matches (simple substring)
            for title, entity in self._entity_titles.items():
                if len(candidates) >= self.top_k:
                    break
                if mention_lower in title and title != mention_lower:
                    candidates.append((entity.title, entity.description or ""))
                    scores.append(0.5)

            ent._.candidates = candidates[:self.top_k]
            ent._.candidate_scores = scores[:self.top_k]

        return doc
```

### Example: Custom Disambiguator

```python
from typing import Optional

from spacy.language import Language
from spacy.tokens import Doc, Span

from el_pipeline.knowledge_bases.base import KnowledgeBase
from el_pipeline.utils import ensure_candidates_extension, ensure_resolved_entity_extension


@Language.factory(
    "el_pipeline_random_disambiguator",
    default_config={},
)
def create_random_disambiguator_component(nlp: Language, name: str):
    """Factory for random disambiguator (for testing)."""
    return RandomDisambiguatorComponent(nlp=nlp)


class RandomDisambiguatorComponent:
    """
    Disambiguator that selects a random candidate.

    Demonstrates the pattern for custom disambiguators.
    Useful for baseline comparisons.
    """

    def __init__(self, nlp: Language):
        self.nlp = nlp
        self.kb = None

        # Register extensions using shared utilities
        ensure_candidates_extension()
        ensure_resolved_entity_extension()

    def initialize(self, kb: KnowledgeBase):
        """Initialize with knowledge base."""
        self.kb = kb

    def __call__(self, doc: Doc) -> Doc:
        """Randomly select a candidate for each entity."""
        import random

        if self.kb is None:
            return doc

        for ent in doc.ents:
            candidates = getattr(ent._, "candidates", [])
            if not candidates:
                continue

            # Select random candidate
            title, _ = random.choice(candidates)
            entity = self.kb.get_entity(title)
            if entity:
                ent._.resolved_entity = entity

        return doc
```

### Registering Custom Components

Components are registered when the module is imported:

```python
# In your_custom_components.py
from spacy.language import Language
# ... define your components with @Language.factory

# In your main code
import spacy
import your_custom_components  # This registers the factories

nlp = spacy.blank("en")
nlp.add_pipe("el_pipeline_email_ner")  # Now available
```

---

## Component Compatibility Matrix

All component types can be combined freely. Here are some recommended combinations:

### Lightweight (No GPU)

| Stage | Component | Notes |
|-------|-----------|-------|
| NER | `simple` | Regex-based, no downloads |
| Candidates | `fuzzy` | RapidFuzz string matching |
| Reranker | `none` | Skip reranking |
| Disambiguator | `first` | Select first candidate |

### Balanced (Some GPU)

| Stage | Component | Notes |
|-------|-----------|-------|
| NER | `lela_gliner` | Zero-shot, good accuracy |
| Candidates | `lela_bm25` | Fast BM25 with stemming |
| Reranker | `none` | Skip for speed |
| Disambiguator | `first` | Select first candidate |

### Full LELA (GPU Required)

| Stage | Component | Notes |
|-------|-----------|-------|
| NER | `lela_gliner` | Zero-shot NER |
| Candidates | `lela_bm25` | 64 candidates |
| Reranker | `lela_embedder` | Reduce to 10 |
| Disambiguator | `lela_vllm` | LLM disambiguation |

---

## Registry System Overview

### How Registries Work

```python
# In el_pipeline/registry.py
class Registry:
    def __init__(self, name: str):
        self.name = name
        self._items = {}

    def register(self, name: str):
        def decorator(cls):
            self._items[name] = cls
            return cls
        return decorator

    def get(self, name: str):
        return self._items[name]

loaders = Registry("loaders")
knowledge_bases = Registry("knowledge_bases")
```

### Using the Registry

```python
from el_pipeline.registry import loaders, knowledge_bases

# Register
@loaders.register("my_loader")
class MyLoader:
    ...

# Retrieve
LoaderClass = loaders.get("my_loader")
loader = LoaderClass(**params)
```

### Pipeline Config Name Mapping

The `ELPipeline` maps config names to spaCy factory names:

```python
# In el_pipeline/pipeline.py
NER_COMPONENT_MAP = {
    "lela_gliner": "el_pipeline_lela_gliner",
    "simple": "el_pipeline_simple",
    # ...
}
```

To add a new component to the pipeline config:

1. Create the spaCy factory with `@Language.factory("el_pipeline_my_component")`
2. Add to the appropriate map in `pipeline.py`
3. Import your module in `el_pipeline/spacy_components/__init__.py`

---

## Best Practices

### 1. Use Lazy Imports

```python
_heavy_model = None

def _get_model():
    global _heavy_model
    if _heavy_model is None:
        from heavy_library import Model
        _heavy_model = Model.load()
    return _heavy_model
```

### 2. Use Shared Utilities for Extensions

Use the shared utility functions in `el_pipeline.utils` to register extensions:

```python
from el_pipeline.utils import (
    filter_spans,                    # For NER components
    ensure_context_extension,        # For NER: ent._.context
    ensure_candidates_extension,     # For candidates: ent._.candidates, ent._.candidate_scores
    ensure_resolved_entity_extension # For disambiguators: ent._.resolved_entity
)

# In your component's __init__:
ensure_context_extension()
ensure_candidates_extension()

# In your NER component's __call__:
doc.ents = filter_spans(spans)
```

For custom extensions not covered by the shared utilities:

```python
if not Span.has_extension("my_extension"):
    Span.set_extension("my_extension", default=None)
```

### 3. Handle Missing KB Gracefully

```python
def __call__(self, doc: Doc) -> Doc:
    if self.kb is None:
        logger.warning("Component not initialized - call initialize(kb)")
        return doc
    # ...
```

### 4. Support Progress Callbacks

```python
class MyComponent:
    def __init__(self, ...):
        self.progress_callback = None

    def __call__(self, doc: Doc) -> Doc:
        for i, ent in enumerate(doc.ents):
            if self.progress_callback:
                progress = i / len(doc.ents)
                self.progress_callback(progress, f"Processing {ent.text}")
            # ...
```

### 5. Log Important Events

```python
import logging
logger = logging.getLogger(__name__)

class MyComponent:
    def __init__(self, model_name: str):
        logger.info(f"Loading model: {model_name}")
        # ...
```
