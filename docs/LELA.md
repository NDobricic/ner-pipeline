# LELA Integration Documentation

LELA (LLM-based Entity Linking Approach) is a methodology for entity linking that leverages the reasoning capabilities of large language models without requiring fine-tuning. This document describes the LELA integration in the NER Pipeline, implemented as spaCy components.

## Table of Contents

- [Overview](#overview)
- [LELA Methodology](#lela-methodology)
- [LELA spaCy Components](#lela-spacy-components)
- [Configuration](#configuration)
- [LELA Module Structure](#lela-module-structure)
- [Usage Examples](#usage-examples)

## Overview

### What is LELA?

LELA is a **true zero-shot** entity linking approach that:

- Works with any knowledge base and domain without fine-tuning
- Uses LLM reasoning for disambiguation
- Employs a tournament-style candidate selection strategy
- Is retriever-agnostic and LLM-agnostic

### Key Benefits

| Benefit | Description |
|---------|-------------|
| **No Fine-tuning** | Works out-of-the-box for any domain |
| **Domain Agnostic** | Adapts to legal, biomedical, company-specific KBs |
| **High Accuracy** | Competitive with fine-tuned approaches |
| **Modular** | Each component is independently swappable |

### Comparison with Traditional Approaches

| Setting | Domain | Knowledge Base | Fine-tuning Required |
|---------|--------|----------------|---------------------|
| Classical EL | Same | Same | Yes |
| Zero-shot EL | Similar | Different | Yes |
| **LELA (True zero-shot)** | Any | Any | No |

## LELA Methodology

### The Entity Linking Task

Given:
- A sentence S with a mention m
- A knowledge base E with entities and descriptions

Goal: Find the entity e ∈ E that m refers to, or return "none" if no match.

**Example:**
```
Sentence: "France hosted the 2024 Olympics in [Paris]."

Knowledge Base:
- Paris (city): Capital city of France
- Paris (novel): 1897 novel by Emile Zola
- Paris (Texas): City in Texas, USA

Output: Paris (city)
```

### Two-Stage Pipeline

LELA follows the standard two-stage entity linking approach:

#### Stage 1: Candidate Generation

Retrieve potential entity matches from the KB:

```
Input: "Paris" + context
Output: [Paris (city), Paris (novel), Paris (Texas), Paris Hilton, ...]
```

LELA is retriever-agnostic and supports:
- BM25 (keyword-based) → `ner_pipeline_lela_bm25_candidates`
- Dense retrieval (semantic embeddings) → `ner_pipeline_lela_dense_candidates`
- Fuzzy matching (string similarity) → `ner_pipeline_fuzzy_candidates`

#### Stage 2: Candidate Selection (Disambiguation)

Select the best candidate using LLM reasoning:

```
Input: Context + Mention + Candidates
LLM Reasoning: "The mention 'Paris' appears in context about Olympics in France..."
Output: Paris (city)
```

Implemented via: `ner_pipeline_lela_vllm_disambiguator`

### Tournament Strategy

To handle large candidate sets, LELA uses a **tournament approach**:

```
Round 1: [64 candidates] → Split into batches of 8
         8 batches × 8 candidates → 8 winners

Round 2: [8 winners] → 1 batch of 8
         → Final winner
```

**Benefits:**
1. Fits candidates into context window
2. Allows fine-grained reasoning per batch
3. Stronger candidates face each other in later rounds

**Batch Size Trade-off:**
- Smaller k: More rounds, finer reasoning, slower
- Larger k: Fewer rounds, less focus, faster
- Recommended: k = √C (e.g., k=8 for 64 candidates)

### LLM Prompt Structure

```
System: You are an expert designed to disambiguate entities in text,
taking into account the overall context and a list of entity candidates.
Your task is to determine the most appropriate entity from the candidates
based on the context and candidate entity descriptions. Please show your
choice in the answer field with only the choice index number.

User: Input: "France hosted the 2024 Olympics in [Paris]."

Candidates:
0. None of the candidates
1. Paris (city): Capital city of France
2. Paris (novel): 1897 novel by Emile Zola
3. Paris (Texas): City in Texas, USA

Answer:
```

## LELA spaCy Components

All LELA components are implemented as spaCy pipeline factories.

### LELA NER: `ner_pipeline_lela_gliner`

Zero-shot NER using GLiNER with LELA defaults.

**Default Model:** `numind/NuNER_Zero-span`

**Default Labels:**
- person
- organization
- location
- event
- work of art
- product

**Configuration:**
```python
nlp.add_pipe("ner_pipeline_lela_gliner", config={
    "model_name": "numind/NuNER_Zero-span",
    "labels": ["person", "organization", "location"],
    "threshold": 0.5,
    "context_mode": "sentence"
})
```

**JSON Config:**
```json
{
  "name": "lela_gliner",
  "params": {
    "model_name": "numind/NuNER_Zero-span",
    "labels": ["person", "organization", "location"],
    "threshold": 0.5
  }
}
```

### LELA Candidate Generation

#### `ner_pipeline_lela_bm25_candidates`

Fast BM25 retrieval using bm25s library with stemming.

**Features:**
- Numba-accelerated BM25
- Language-specific stemming (PyStemmer)
- Context integration

**spaCy Usage:**
```python
cand = nlp.add_pipe("ner_pipeline_lela_bm25_candidates", config={
    "top_k": 64,
    "use_context": True,
    "stemmer_language": "english"
})
cand.initialize(kb)
```

**JSON Config:**
```json
{
  "name": "lela_bm25",
  "params": {
    "top_k": 64,
    "use_context": true,
    "stemmer_language": "english"
  }
}
```

#### `ner_pipeline_lela_dense_candidates`

Dense retrieval using OpenAI-compatible embedding API.

**Default Model:** `Qwen/Qwen3-Embedding-4B`

**Query Format:**
```
Instruct: Given an entity mention and its context, retrieve the entity
from the knowledge base that the mention refers to.
Query: {mention_text}
```

**spaCy Usage:**
```python
cand = nlp.add_pipe("ner_pipeline_lela_dense_candidates", config={
    "model_name": "Qwen/Qwen3-Embedding-4B",
    "top_k": 64,
    "base_url": "http://localhost",
    "port": 8000,
    "use_context": True
})
cand.initialize(kb)
```

**JSON Config:**
```json
{
  "name": "lela_dense",
  "params": {
    "model_name": "Qwen/Qwen3-Embedding-4B",
    "top_k": 64,
    "base_url": "http://localhost",
    "port": 8000
  }
}
```

### LELA Reranking: `ner_pipeline_lela_embedder_reranker`

Cosine similarity reranking with marked mentions.

**Default Model:** `tomaarsen/Qwen3-Reranker-4B-seq-cls`

**Mention Marking:**
```
Original: "France hosted the Olympics in Paris."
Marked:   "France hosted the Olympics in [Paris]."
```

**Query Format:**
```
Instruct: Given the entity mention marked with [...] in the input text,
retrieve the entity from the knowledge base that the marked mention
refers to.
Query: France hosted the Olympics in [Paris].
```

**spaCy Usage:**
```python
nlp.add_pipe("ner_pipeline_lela_embedder_reranker", config={
    "model_name": "Qwen/Qwen3-Embedding-4B",
    "top_k": 10,
    "base_url": "http://localhost",
    "port": 8000
})
```

**JSON Config:**
```json
{
  "name": "lela_embedder",
  "params": {
    "model_name": "Qwen/Qwen3-Embedding-4B",
    "top_k": 10,
    "base_url": "http://localhost",
    "port": 8000
  }
}
```

### LELA Disambiguation: `ner_pipeline_lela_vllm_disambiguator`

LLM-based disambiguation using vLLM for efficient inference.

**Default Model:** `Qwen/Qwen3-8B`

**Features:**
- Tournament-style selection
- Self-consistency voting
- "None" candidate option
- Structured output parsing

**spaCy Usage:**
```python
disamb = nlp.add_pipe("ner_pipeline_lela_vllm_disambiguator", config={
    "model_name": "Qwen/Qwen3-8B",
    "tensor_parallel_size": 1,
    "add_none_candidate": True,
    "add_descriptions": True,
    "disable_thinking": False,
    "self_consistency_k": 1
})
disamb.initialize(kb)
```

**JSON Config:**
```json
{
  "name": "lela_vllm",
  "params": {
    "model_name": "Qwen/Qwen3-8B",
    "tensor_parallel_size": 1,
    "add_none_candidate": true,
    "add_descriptions": true,
    "disable_thinking": false,
    "self_consistency_k": 1
  }
}
```

### LELA Knowledge Base: `lela_jsonl`

LELA-format JSONL knowledge base where title serves as ID.

**Format:**
```jsonl
{"title": "Albert Einstein", "description": "German-born theoretical physicist"}
{"title": "Paris", "description": "Capital city of France"}
```

**JSON Config:**
```json
{
  "name": "lela_jsonl",
  "params": {
    "path": "entities.jsonl",
    "title_field": "title",
    "description_field": "description"
  }
}
```

## Configuration

### Full LELA Pipeline Configuration

```json
{
  "loader": {"name": "text"},
  "ner": {
    "name": "lela_gliner",
    "params": {
      "model_name": "numind/NuNER_Zero-span",
      "labels": ["person", "organization", "location", "event", "work of art", "product"],
      "threshold": 0.5
    }
  },
  "candidate_generator": {
    "name": "lela_bm25",
    "params": {
      "top_k": 64,
      "use_context": true,
      "stemmer_language": "english"
    }
  },
  "reranker": {
    "name": "lela_embedder",
    "params": {
      "model_name": "Qwen/Qwen3-Embedding-4B",
      "top_k": 10,
      "base_url": "http://localhost",
      "port": 8000
    }
  },
  "disambiguator": {
    "name": "lela_vllm",
    "params": {
      "model_name": "Qwen/Qwen3-8B",
      "tensor_parallel_size": 1,
      "add_none_candidate": true,
      "add_descriptions": true
    }
  },
  "knowledge_base": {
    "name": "lela_jsonl",
    "params": {"path": "kb.jsonl"}
  }
}
```

### Lightweight Configuration (BM25 only)

For faster processing without GPU requirements:

```json
{
  "loader": {"name": "text"},
  "ner": {
    "name": "lela_gliner",
    "params": {"threshold": 0.5}
  },
  "candidate_generator": {
    "name": "lela_bm25",
    "params": {"top_k": 10}
  },
  "reranker": {"name": "none"},
  "disambiguator": {"name": "first"},
  "knowledge_base": {
    "name": "lela_jsonl",
    "params": {"path": "kb.jsonl"}
  }
}
```

### NoOp Reranker

The `"none"` reranker config name maps to `ner_pipeline_noop_reranker`, which passes candidates through unchanged:

```json
{
  "reranker": {"name": "none"}
}
```

This is useful when:
- You want to skip reranking entirely
- Your candidate generator already orders candidates well
- You're using a lightweight pipeline configuration

### Cache Key Generation

The pipeline uses content-addressed caching for processed documents. Cache keys are generated as:

```python
key = SHA256(f"{file_path}-{modification_time}-{file_size}".encode()).hexdigest()
```

**Components:**
- `file_path`: Full path to the document
- `modification_time`: File's mtime from `os.stat()`
- `file_size`: File size in bytes

**Location:** Cache files are stored in `.ner_cache/` (configurable via `cache_dir`)

**Cache invalidation:** The cache is automatically invalidated when:
- The file path changes
- The file is modified (mtime changes)
- The file size changes

**Disabling cache:** Set `cache_dir` to `None` in configuration (not recommended for production)

## LELA Module Structure

**Location:** `ner_pipeline/lela/`

### `config.py`

Default configuration values for LELA components.

```python
# NER Settings
NER_LABELS = ["person", "organization", "location", "event", "work of art", "product"]
DEFAULT_GLINER_MODEL = "numind/NuNER_Zero-span"

# Candidate Generation
DEFAULT_EMBEDDER_MODEL = "Qwen/Qwen3-Embedding-4B"
CANDIDATES_TOP_K = 64
RETRIEVER_TASK = "Given an entity mention and its context..."

# Reranking
DEFAULT_RERANKER_MODEL = "tomaarsen/Qwen3-Reranker-4B-seq-cls"
RERANKER_TOP_K = 10
RERANKER_TASK = "Given the entity mention marked with [...]..."

# Disambiguation
DEFAULT_LLM_MODEL = "Qwen/Qwen3-4B"
NOT_AN_ENTITY = "None"
SPAN_OPEN = "["
SPAN_CLOSE = "]"
```

### `prompts.py`

Prompt templates for disambiguation.

**System Prompt:**
```python
SYSTEM_PROMPT = """You are an expert designed to disambiguate entities in text,
taking into account the overall context and a list of entity candidates.
You are provided with an input text that includes a full contextual narrative,
a marked mention enclosed in square brackets, and a list of candidates,
each preceded by an index number.

Your task is to determine the most appropriate entity from the candidates
based on the context and candidate entity descriptions. Please show your
choice in the answer field with only the choice index number."""
```

**Functions:**
- `create_disambiguation_messages()`: Build LLM conversation
- `mark_mention_in_text()`: Mark mention with brackets

### `llm_pool.py`

Singleton pools for resource management.

**EmbedderPool:**
- Manages OpenAI-compatible API client
- Singleton pattern for efficiency
- Methods: `get_client()`, `embed()`

**vLLM Instance Manager:**
- Lazy loading of vLLM
- Caches by model_name:tensor_parallel_size
- Functions: `get_vllm_instance()`, `clear_vllm_instances()`

## Usage Examples

### Python API with LELA (via NERPipeline)

```python
from ner_pipeline.config import PipelineConfig
from ner_pipeline.pipeline import NERPipeline
import json

# Load LELA configuration
config_dict = {
    "loader": {"name": "text"},
    "ner": {"name": "lela_gliner", "params": {"threshold": 0.5}},
    "candidate_generator": {"name": "lela_bm25", "params": {"top_k": 64}},
    "reranker": {"name": "none"},
    "disambiguator": {"name": "first"},
    "knowledge_base": {"name": "lela_jsonl", "params": {"path": "kb.jsonl"}}
}

config = PipelineConfig.from_dict(config_dict)
pipeline = NERPipeline(config)

# Process document
results = pipeline.run(["document.txt"], output_path="results.jsonl")
```

### Direct spaCy Usage with LELA Components

```python
import spacy
from ner_pipeline import spacy_components  # Register factories
from ner_pipeline.knowledge_bases.lela import LELAJSONLKnowledgeBase

# Build LELA pipeline manually
nlp = spacy.blank("en")

# Add LELA NER
nlp.add_pipe("ner_pipeline_lela_gliner", config={
    "threshold": 0.5,
    "labels": ["person", "organization", "location"]
})

# Add BM25 candidates
cand = nlp.add_pipe("ner_pipeline_lela_bm25_candidates", config={
    "top_k": 64,
    "use_context": True
})

# Add reranker (optional)
nlp.add_pipe("ner_pipeline_noop_reranker")

# Add disambiguator
disamb = nlp.add_pipe("ner_pipeline_first_disambiguator")

# Initialize with KB
kb = LELAJSONLKnowledgeBase(path="kb.jsonl")
cand.initialize(kb)
disamb.initialize(kb)

# Process text
doc = nlp("Albert Einstein was born in Germany.")
for ent in doc.ents:
    print(f"Entity: {ent.text}")
    print(f"  Context: {ent._.context}")
    print(f"  Candidates: {len(ent._.candidates)}")
    if ent._.resolved_entity:
        print(f"  Resolved: {ent._.resolved_entity.title}")
```

### CLI with LELA

```bash
python -m ner_pipeline.cli \
  --config config/lela_example.json \
  --input document.txt \
  --output results.jsonl
```

### Web App with LELA

1. Start the web app: `python app.py`
2. Select "lela_gliner" for NER
3. Select "lela_bm25" or "lela_dense" for candidate generation
4. Optionally enable "lela_embedder" reranker
5. Select "lela_vllm" for disambiguation
6. Upload your knowledge base and documents

## Performance Considerations

### GPU Requirements

| Component | GPU Recommended | Notes |
|-----------|-----------------|-------|
| lela_gliner | Yes | GLiNER model inference |
| lela_bm25 | No | CPU-based BM25 |
| lela_dense | Yes | Embedding computation |
| lela_embedder | Yes | Embedding computation |
| lela_vllm | Yes (Required) | LLM inference |

### CUDA Compatibility

For **P100/Pascal GPUs** (compute capability 6.0):
- Use `torch==2.6.0+cu118` (CUDA 11.8)
- Use `vllm==0.8.5` (newer versions drop P100 support)
- Install from PyTorch index: `pip install torch==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118`

For **newer GPUs** (A100, H100, etc.):
- Can use latest PyTorch/vLLM with CUDA 12.x

### Model Sizes

| Model | Size | VRAM Required |
|-------|------|---------------|
| NuNER_Zero-span | ~350MB | ~2GB |
| Qwen3-Embedding-4B | ~8GB | ~10GB |
| Qwen3-4B | ~8GB | ~10GB |
| Qwen3-8B | ~16GB | ~20GB |

### Optimization Tips

1. **Reduce top_k** for candidate generation (e.g., 32 instead of 64)
2. **Use BM25** instead of dense retrieval when speed matters
3. **Skip reranking** for simpler pipelines
4. **Use smaller LLMs** (Qwen3-4B instead of 8B)
5. **Enable tensor parallelism** for multi-GPU setups

## References

For more details on the LELA methodology, see the research paper:

> LELA - an LLM-based Entity Linking Approach with Zero-Shot Domain Adaptation
>
> Entity linking (mapping ambiguous mentions in text to entities in a knowledge base)
> is a foundational step in tasks such as knowledge graph construction, question-answering,
> and information extraction. LELA leverages the reasoning capabilities of large language
> models, works with different target domains, knowledge bases and LLMs, without any
> fine-tuning phase. It employs a tournament-style candidate reranking strategy that
> enhances both scalability and accuracy by allowing the models to reason over manageable
> candidate subsets.
