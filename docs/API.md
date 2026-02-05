# Python API Reference

This document provides a comprehensive reference for the NER Pipeline Python API, built on spaCy's component architecture.

## Table of Contents

- [Core Classes](#core-classes)
- [spaCy Components](#spacy-components)
- [Data Types](#data-types)
- [Configuration](#configuration)
- [Context Extraction](#context-extraction)
- [Advanced Features](#advanced-features)
- [Usage Examples](#usage-examples)

## Core Classes

### NERPipeline

The main orchestrator class that manages the entity linking pipeline using spaCy.

**Location:** `ner_pipeline/pipeline.py`

```python
from ner_pipeline.pipeline import NERPipeline
from ner_pipeline.config import PipelineConfig
```

#### Constructor

```python
NERPipeline(config: PipelineConfig, progress_callback: Optional[Callable[[float, str], None]] = None)
```

**Parameters:**
- `config`: A `PipelineConfig` object containing pipeline configuration
- `progress_callback`: Optional callback function `(progress: float, description: str) -> None` for tracking initialization progress (0.0 to 1.0)

**Initialization:**
- Instantiates knowledge base and document loader from registries
- Builds a spaCy `Language` pipeline with configured components
- Sets up caching directory

**Internal Structure:**
- `self.nlp`: The spaCy `Language` instance with pipeline components
- `self.kb`: Knowledge base instance
- `self.loader`: Document loader instance

#### Methods

##### `process_document(doc: Document) -> Dict`

Process a single document through the spaCy pipeline.

**Parameters:**
- `doc`: A `Document` object to process

**Returns:**
```python
{
    "id": str,           # Document ID
    "text": str,         # Full document text
    "entities": List[{   # List of resolved entities
        "text": str,              # Mention text
        "start": int,             # Start character position
        "end": int,               # End character position
        "label": str,             # Entity type label
        "context": str,           # Surrounding context
        "entity_id": Optional[str],    # Resolved entity ID
        "entity_title": Optional[str], # Entity title
        "entity_description": Optional[str],  # Entity description
        "candidates": List[{      # Candidate list
            "entity_id": str,
            "score": float,
            "description": str
        }]
    }],
    "meta": Dict         # Document metadata
}
```

##### `process_document_with_progress(doc: Document, progress_callback: Optional[Callable], base_progress: float = 0.0, progress_range: float = 1.0) -> Dict`

Process a single document with detailed progress callbacks for each pipeline stage.

**Parameters:**
- `doc`: A `Document` object to process
- `progress_callback`: Callback function `(progress: float, description: str) -> None`
- `base_progress`: Starting progress value (0.0-1.0)
- `progress_range`: How much progress this processing represents (0.0-1.0)

**Returns:**
- Same format as `process_document`

**Example:**
```python
def my_progress_callback(progress: float, description: str):
    print(f"{progress*100:.1f}%: {description}")

result = pipeline.process_document_with_progress(
    doc,
    progress_callback=my_progress_callback
)
# Output:
# 0.0%: Tokenizing document...
# 15.0%: NER (GLiNER)...
# 45.0%: Candidate generation (BM25)...
# 75.0%: Disambiguation (LLM)...
# 95.0%: Serializing results...
# 100.0%: Document processing complete
```

##### `run(paths: Iterable[str], output_path: Optional[str] = None) -> List[Dict]`

Process multiple files through the pipeline.

**Parameters:**
- `paths`: Iterable of file paths to process
- `output_path`: Optional path to write JSONL output

**Returns:**
- List of result dictionaries (same format as `process_document`)

**Example:**
```python
pipeline = NERPipeline(config)
results = pipeline.run(
    ["doc1.txt", "doc2.pdf", "doc3.html"],
    output_path="results.jsonl"
)
```

### PipelineConfig

Configuration dataclass for pipeline setup.

**Location:** `ner_pipeline/config.py`

```python
from ner_pipeline.config import PipelineConfig
```

#### Class Methods

##### `from_dict(config_dict: Dict) -> PipelineConfig`

Create a configuration from a dictionary.

**Parameters:**
- `config_dict`: Dictionary with configuration options

**Example:**
```python
config_dict = {
    "loader": {"name": "text"},
    "ner": {"name": "simple", "params": {"min_len": 3}},
    "candidate_generator": {"name": "fuzzy", "params": {"top_k": 10}},
    "reranker": {"name": "none"},
    "disambiguator": {"name": "first"},
    "knowledge_base": {"name": "custom", "params": {"path": "kb.jsonl"}},
    "cache_dir": ".ner_cache",
    "batch_size": 1
}
config = PipelineConfig.from_dict(config_dict)
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `loader` | Dict | Loader configuration |
| `ner` | Dict | NER component configuration |
| `candidate_generator` | Dict | Candidate generator configuration |
| `reranker` | Dict | Reranker configuration |
| `disambiguator` | Dict | Disambiguator configuration |
| `knowledge_base` | Dict | Knowledge base configuration |
| `cache_dir` | str | Directory for document caching |
| `batch_size` | int | Batch size for processing |

## spaCy Components

All pipeline components are implemented as spaCy factories and can be used directly with spaCy's `nlp.add_pipe()` method.

### Component Registration

Import the `spacy_components` module to register all factories:

```python
from ner_pipeline import spacy_components  # Registers all factories
import spacy

nlp = spacy.blank("en")
nlp.add_pipe("ner_pipeline_simple")  # Now available
```

### spaCy Extensions

The pipeline uses custom extensions on `Span` objects:

```python
from spacy.tokens import Span

# Automatically registered when components are loaded
Span.set_extension("context", default=None)
Span.set_extension("candidates", default=[])
Span.set_extension("resolved_entity", default=None)
```

### NER Components

#### `ner_pipeline_lela_gliner`

Zero-shot GLiNER NER with LELA defaults.

**Config Options:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "numind/NuNER_Zero-span" | GLiNER model |
| `labels` | List[str] | LELA defaults | Entity types |
| `threshold` | float | 0.5 | Detection threshold |
| `context_mode` | str | "sentence" | Context extraction mode |

**Example:**
```python
nlp.add_pipe("ner_pipeline_lela_gliner", config={
    "labels": ["person", "organization", "location"],
    "threshold": 0.4
})
```

#### `ner_pipeline_simple`

Lightweight regex-based NER.

**Config Options:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_len` | int | 3 | Minimum mention length |
| `context_mode` | str | "sentence" | Context extraction mode |

**Example:**
```python
nlp.add_pipe("ner_pipeline_simple", config={"min_len": 2})
```

#### `ner_pipeline_gliner`

Standard GLiNER wrapper.

**Config Options:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "urchade/gliner_base" | GLiNER model |
| `labels` | List[str] | ["person", "organization", "location"] | Entity types |
| `threshold` | float | 0.5 | Detection threshold |
| `context_mode` | str | "sentence" | Context extraction mode |

#### `ner_pipeline_ner_filter`

Post-filter for spaCy's built-in NER (adds context extension).

**Usage:**
```python
# Load pretrained spaCy model
spacy_nlp = spacy.load("en_core_web_sm")

# Copy NER and add filter
nlp.add_pipe("ner", source=spacy_nlp)
nlp.add_pipe("ner_pipeline_ner_filter")
```

### Candidate Generation Components

#### `ner_pipeline_lela_bm25_candidates`

BM25 retrieval using bm25s library with stemming.

**Config Options:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_k` | int | 64 | Maximum candidates |
| `use_context` | bool | True | Include context in query |
| `stemmer_language` | str | "english" | Stemmer language |

**Requires initialization:**
```python
component = nlp.add_pipe("ner_pipeline_lela_bm25_candidates")
component.initialize(kb)
```

#### `ner_pipeline_lela_dense_candidates`

Dense retrieval using SentenceTransformers and FAISS.

**Config Options:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | LELA default | Embedding model |
| `top_k` | int | 64 | Maximum candidates |
| `device` | str | None | Device override (e.g., "cuda", "cpu") |
| `use_context` | bool | True | Include context in query |

#### `ner_pipeline_fuzzy_candidates`

RapidFuzz string matching.

**Config Options:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_k` | int | 20 | Maximum candidates |

#### `ner_pipeline_bm25_candidates`

Standard BM25 using rank-bm25 library.

**Config Options:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_k` | int | 20 | Maximum candidates |

### Reranker Components

#### `ner_pipeline_lela_embedder_reranker`

Embedding-based cosine similarity reranking with marked mentions.

**Config Options:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | LELA default | Embedding model |
| `top_k` | int | 10 | Candidates to keep |
| `device` | str | None | Device override (e.g., "cuda", "cpu") |

#### `ner_pipeline_cross_encoder_reranker`

Cross-encoder reranking using sentence-transformers.

**Config Options:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "cross-encoder/ms-marco-MiniLM-L-6-v2" | Model |
| `top_k` | int | 10 | Candidates to keep |

#### `ner_pipeline_noop_reranker`

Pass-through (no reranking).

**Config Options:** None

### Disambiguator Components

#### `ner_pipeline_lela_vllm_disambiguator`

vLLM-based LLM disambiguation - sends all candidates at once.

**Config Options:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "Qwen/Qwen3-4B" | LLM model |
| `tensor_parallel_size` | int | 1 | GPU parallelism |
| `max_model_len` | int | None | Max context length |
| `add_none_candidate` | bool | True | Add "None" option for NIL linking |
| `add_descriptions` | bool | True | Include descriptions |
| `disable_thinking` | bool | True | Disable reasoning (adds `/no_think` for Qwen3) |
| `system_prompt` | str | LELA default | Custom prompt |
| `generation_config` | dict | {} | vLLM generation settings |
| `self_consistency_k` | int | 1 | Voting samples (>1 enables majority voting) |

**Requires initialization:**
```python
component = nlp.add_pipe("ner_pipeline_lela_vllm_disambiguator")
component.initialize(kb)
```

**See Also:** [Self-Consistency Voting](#self-consistency-voting), [NIL Linking](#nil-linking), [Qwen3 Thinking Mode](#qwen3-thinking-mode)

#### `ner_pipeline_lela_transformers_disambiguator`

Transformers-based LLM disambiguation (alternative to vLLM for P100/Pascal GPUs).

**Config Options:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "Qwen/Qwen3-4B" | LLM model |
| `add_none_candidate` | bool | True | Add "None" option for NIL linking |
| `add_descriptions` | bool | True | Include descriptions |
| `disable_thinking` | bool | True | Disable reasoning |
| `system_prompt` | str | LELA default | Custom prompt |
| `generation_config` | dict | {} | HuggingFace generation settings |

**Requires initialization:**
```python
component = nlp.add_pipe("ner_pipeline_lela_transformers_disambiguator")
component.initialize(kb)
```

**When to use:** Use this instead of `lela_vllm` when:
- Running on older GPUs (P100/Pascal) where vLLM has issues
- vLLM installation fails or has compatibility problems
- You need direct HuggingFace transformers integration

**Example:**
```json
{
  "disambiguator": {
    "name": "lela_transformers",
    "params": {
      "model_name": "Qwen/Qwen3-4B",
      "disable_thinking": true
    }
  }
}
```

#### `ner_pipeline_first_disambiguator`

Select first candidate.

**Requires initialization:** Yes (needs KB reference)

#### `ner_pipeline_popularity_disambiguator`

Select by highest score (first in sorted list).

**Requires initialization:** Yes (needs KB reference)

## Data Types

All core data types are defined in `ner_pipeline/types.py`.

### Document

Represents an input document.

```python
from ner_pipeline.types import Document

doc = Document(
    id="doc-001",
    text="Albert Einstein was born in Germany.",
    meta={"source": "wikipedia", "date": "2024-01-01"}
)
```

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | str | Unique document identifier |
| `text` | str | Document text content |
| `meta` | Dict | Optional metadata dictionary |

### Entity

Represents an entity in the knowledge base.

```python
from ner_pipeline.types import Entity

entity = Entity(
    id="Q937",
    title="Albert Einstein",
    description="German-born theoretical physicist",
    meta={"birth_year": 1879}
)
```

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | str | Unique entity identifier |
| `title` | str | Entity name/title |
| `description` | str | Entity description |
| `meta` | Dict | Optional metadata |

### Candidate

Represents a potential KB match for a mention.

```python
from ner_pipeline.types import Candidate

candidate = Candidate(
    entity_id="Q937",
    score=0.95,
    description="German-born theoretical physicist"
)
```

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `entity_id` | str | Entity identifier in KB |
| `score` | float | Relevance score |
| `description` | str | Entity description |

### LELA Candidate Format

Internally, spaCy components use LELA format for candidates:

```python
# List of (title, description) tuples
candidates = [
    ("Albert Einstein", "German-born theoretical physicist"),
    ("Einstein Bros. Bagels", "American bagel chain"),
]
```

Convert to Candidate objects for output:
```python
from ner_pipeline.types import tuples_to_candidates
candidates = tuples_to_candidates(tuples_list)
```

## Configuration

### Component Options Summary

#### Config Name → spaCy Factory Mapping

| Config Name | spaCy Factory |
|-------------|---------------|
| **NER** | |
| `lela_gliner` | `ner_pipeline_lela_gliner` |
| `simple` | `ner_pipeline_simple` |
| `gliner` | `ner_pipeline_gliner` |
| `spacy` | Built-in NER + `ner_pipeline_ner_filter` |
| **Candidate Generators** | |
| `lela_bm25` | `ner_pipeline_lela_bm25_candidates` |
| `lela_dense` | `ner_pipeline_lela_dense_candidates` |
| `fuzzy` | `ner_pipeline_fuzzy_candidates` |
| `bm25` | `ner_pipeline_bm25_candidates` |
| **Rerankers** | |
| `lela_embedder` | `ner_pipeline_lela_embedder_reranker` |
| `cross_encoder` | `ner_pipeline_cross_encoder_reranker` |
| `none` | `ner_pipeline_noop_reranker` |
| **Disambiguators** | |
| `lela_vllm` | `ner_pipeline_lela_vllm_disambiguator` |
| `lela_transformers` | `ner_pipeline_lela_transformers_disambiguator` |
| `first` | `ner_pipeline_first_disambiguator` |
| `popularity` | `ner_pipeline_popularity_disambiguator` |

#### Loaders (Registry-based)

| Name | Description |
|------|-------------|
| `text` | Plain text files |
| `pdf` | PDF documents |
| `docx` | Word documents |
| `html` | HTML pages |
| `json` | JSON files |
| `jsonl` | JSON Lines files |

**JSON/JSONL Loader Parameters:**

The `json` and `jsonl` loaders support a `text_field` parameter to customize which field contains the document text:

```json
{
  "loader": {
    "name": "jsonl",
    "params": {
      "text_field": "content"  // Default is "text"
    }
  }
}
```

**Example JSONL with custom field:**
```jsonl
{"id": "doc-1", "content": "Document text here", "meta": {}}
{"id": "doc-2", "content": "Another document", "meta": {}}
```

#### Knowledge Bases (Registry-based)

| Name | Parameters | Description |
|------|------------|-------------|
| `custom` | `path`, `cache_dir` | Custom JSONL KB (supports persistent caching) |
| `lela_jsonl` | `path`, `title_field`, `description_field` | LELA-format JSONL KB |

## Context Extraction

Utilities for extracting context around mentions.

**Location:** `ner_pipeline/context.py`

### Functions

#### `extract_sentence_context(text, start, end, max_sentences=1)`

Extract surrounding sentences containing the mention.

```python
from ner_pipeline.context import extract_sentence_context

text = "First sentence. Albert Einstein was born in Germany. Third sentence."
context = extract_sentence_context(text, start=16, end=31, max_sentences=1)
# Returns: "Albert Einstein was born in Germany."
```

#### `extract_window_context(text, start, end, window_chars=150)`

Extract a fixed character window around the mention.

```python
from ner_pipeline.context import extract_window_context

context = extract_window_context(text, start=16, end=31, window_chars=100)
```

#### `extract_context(text, start, end, mode="sentence", **kwargs)`

General dispatcher for context extraction.

```python
from ner_pipeline.context import extract_context

# Sentence mode
context = extract_context(text, 16, 31, mode="sentence", max_sentences=2)

# Window mode
context = extract_context(text, 16, 31, mode="window", window_chars=150)
```

## Advanced Features

### Progress Callbacks

The pipeline supports progress callbacks at multiple levels for tracking processing status.

#### Pipeline Initialization

```python
from ner_pipeline.pipeline import NERPipeline
from ner_pipeline.config import PipelineConfig

def init_callback(progress: float, description: str):
    print(f"Init {progress*100:.0f}%: {description}")

config = PipelineConfig.from_dict(config_dict)
pipeline = NERPipeline(config, progress_callback=init_callback)
# Output:
# Init 0%: Loading knowledge base...
# Init 15%: Initializing document loader...
# Init 20%: Building spaCy pipeline...
# Init 25%: Loading NER model (lela_gliner)...
# Init 45%: Loading candidate generator (lela_bm25)...
# Init 75%: Loading disambiguator (lela_vllm)...
# Init 100%: Pipeline initialization complete
```

#### Document Processing

```python
def process_callback(progress: float, description: str):
    print(f"Processing {progress*100:.0f}%: {description}")

result = pipeline.process_document_with_progress(doc, progress_callback=process_callback)
```

---

### Self-Consistency Voting

The `lela_vllm` disambiguator supports self-consistency voting for improved accuracy. When `self_consistency_k > 1`, the model generates multiple responses and selects the answer by majority vote.

**Configuration:**
```json
{
  "disambiguator": {
    "name": "lela_vllm",
    "params": {
      "self_consistency_k": 5  // Generate 5 responses, take majority vote
    }
  }
}
```

**How it works:**
1. The LLM generates `k` candidate answers for each entity
2. Each answer is parsed to extract the selected candidate index
3. The most frequently selected index wins (majority voting)

**Trade-offs:**
- Higher `k` = better accuracy but slower (k times more LLM calls)
- Recommended: `k=3` or `k=5` for important decisions
- Default: `k=1` (no voting, fastest)

---

### NIL Linking

NIL linking allows the model to reject all candidates when none match the mention. This is enabled via the `add_none_candidate` parameter.

**Configuration:**
```json
{
  "disambiguator": {
    "name": "lela_vllm",
    "params": {
      "add_none_candidate": true  // Enable NIL linking
    }
  }
}
```

**How it works:**

When `add_none_candidate=true`:
- Candidate index 0 is reserved for "None of the listed candidates"
- The LLM prompt includes this option explicitly
- If the model selects index 0, `ent._.resolved_entity` remains `None`

**Prompt format with NIL option:**
```
Candidates:
0. None of the listed candidates
1. Paris (city): Capital city of France
2. Paris (novel): 1897 novel by Emile Zola
3. Paris (Texas): City in Texas, USA
```

**Output behavior:**
```python
for ent in doc.ents:
    if ent._.resolved_entity is None:
        print(f"{ent.text}: Not linked (NIL)")
    else:
        print(f"{ent.text}: {ent._.resolved_entity.title}")
```

---

### Qwen3 Thinking Mode

Qwen3 models support a "thinking mode" where the model shows chain-of-thought reasoning. The pipeline can disable this for faster responses.

**Configuration:**
```json
{
  "disambiguator": {
    "name": "lela_vllm",
    "params": {
      "disable_thinking": true  // Skip chain-of-thought reasoning
    }
  }
}
```

**How it works:**

When `disable_thinking=true`:
- The prompt ends with the `/no_think` soft switch token
- Qwen3 models recognize this and output the answer directly
- Reduces token usage and latency significantly

**With thinking (default for some prompts):**
```
<think>
The mention "Paris" in the context about Olympics in France clearly
refers to the capital city, not the novel or the Texas city...
</think>
answer: 1
```

**Without thinking (`disable_thinking=true`):**
```
1
```

**Note:** The transformers disambiguator also handles `</think>` tags in the output when parsing, so it works correctly even if the model includes thinking.

---

## Usage Examples

### Basic Pipeline Usage

```python
from ner_pipeline.config import PipelineConfig
from ner_pipeline.pipeline import NERPipeline
from ner_pipeline.types import Document
import json

# Load configuration
with open("config.json") as f:
    config = PipelineConfig.from_dict(json.load(f))

# Create pipeline (builds spaCy nlp internally)
pipeline = NERPipeline(config)

# Process single document
doc = Document(
    id="test-doc",
    text="Albert Einstein was born in Germany and later moved to the United States.",
    meta={}
)
result = pipeline.process_document(doc)

# Print results
for entity in result["entities"]:
    print(f"Mention: {entity['text']}")
    print(f"  Label: {entity['label']}")
    print(f"  Resolved to: {entity.get('entity_title', 'N/A')}")
    print(f"  Candidates: {len(entity['candidates'])}")
```

### Direct spaCy Usage

```python
import spacy
from ner_pipeline import spacy_components  # Register factories
from ner_pipeline.knowledge_bases.custom import CustomJSONLKnowledgeBase

# Build custom pipeline
nlp = spacy.blank("en")
nlp.add_pipe("ner_pipeline_simple", config={"min_len": 3})
cand_component = nlp.add_pipe("ner_pipeline_fuzzy_candidates", config={"top_k": 10})
disamb_component = nlp.add_pipe("ner_pipeline_first_disambiguator")

# Initialize with knowledge base
kb = CustomJSONLKnowledgeBase(path="kb.jsonl")
cand_component.initialize(kb)
disamb_component.initialize(kb)

# Process text
doc = nlp("Albert Einstein was born in Germany.")

# Access entities and their attributes
for ent in doc.ents:
    print(f"Entity: {ent.text} ({ent.label_})")
    print(f"  Context: {ent._.context}")
    print(f"  Candidates: {len(ent._.candidates)}")
    if ent._.resolved_entity:
        print(f"  Resolved: {ent._.resolved_entity.title}")
```

### Processing Multiple Files

```python
# Process multiple files with output
results = pipeline.run(
    paths=["doc1.txt", "doc2.pdf", "doc3.html"],
    output_path="output/results.jsonl"
)

# Results are also returned
for result in results:
    print(f"Document {result['id']}: {len(result['entities'])} entities")
```

### Working with Knowledge Bases

```python
from ner_pipeline.knowledge_bases.custom import CustomJSONLKnowledgeBase

# Load knowledge base
kb = CustomJSONLKnowledgeBase(path="entities.jsonl")

# Get entity by ID
entity = kb.get_entity("Q937")

# Search entities
results = kb.search("Einstein", top_k=10)

# Iterate all entities
for entity in kb.all_entities():
    print(f"{entity.id}: {entity.title}")
```

### LELA-Specific Configuration

```python
config_dict = {
    "loader": {"name": "text"},
    "ner": {
        "name": "lela_gliner",
        "params": {
            "model_name": "numind/NuNER_Zero-span",
            "labels": ["person", "organization", "location"],
            "threshold": 0.5
        }
    },
    "candidate_generator": {
        "name": "lela_bm25",
        "params": {"top_k": 64, "use_context": True}
    },
    "reranker": {
        "name": "lela_embedder",
        "params": {
            "model_name": "Qwen/Qwen3-Embedding-4B",
            "top_k": 10
        }
    },
    "disambiguator": {
        "name": "lela_vllm",
        "params": {
            "model_name": "Qwen/Qwen3-8B",
            "tensor_parallel_size": 1,
            "add_none_candidate": True
        }
    },
    "knowledge_base": {
        "name": "lela_jsonl",
        "params": {"path": "entities.jsonl"}
    }
}
```

## Output Format

The pipeline outputs JSONL (JSON Lines) format:

```json
{
  "id": "doc-001",
  "text": "Albert Einstein was born in Germany.",
  "entities": [
    {
      "text": "Albert Einstein",
      "start": 0,
      "end": 15,
      "label": "PERSON",
      "context": "Albert Einstein was born in Germany.",
      "entity_id": "Q937",
      "entity_title": "Albert Einstein",
      "entity_description": "German-born theoretical physicist",
      "candidates": [
        {
          "entity_id": "Q937",
          "score": 0.95,
          "description": "German-born theoretical physicist"
        },
        {
          "entity_id": "Q1234",
          "score": 0.45,
          "description": "Another Einstein"
        }
      ]
    },
    {
      "text": "Germany",
      "start": 28,
      "end": 35,
      "label": "GPE",
      "context": "Albert Einstein was born in Germany.",
      "entity_id": "Q183",
      "entity_title": "Germany",
      "entity_description": "Country in Central Europe",
      "candidates": [...]
    }
  ],
  "meta": {
    "source": "wikipedia"
  }
}
```
