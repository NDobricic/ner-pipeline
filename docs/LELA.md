# LELA Integration Documentation

LELA (LLM-based Entity Linking Approach) is a methodology for entity linking that leverages the reasoning capabilities of large language models without requiring fine-tuning. This document describes the LELA integration in the EL Pipeline, implemented as spaCy components.

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
- Employs LLM reasoning for candidate selection
- Is retriever-agnostic and LLM-agnostic

### Key Benefits

| Benefit | Description |
|---------|-------------|
| **No Fine-tuning** | Works out-of-the-box for any domain |
| **Domain Agnostic** | Adapts to legal, biomedical, company-specific KBs |
| **High Accuracy** | Competitive with fine-tuned approaches |
| **Modular** | Each component is independently swappable as a spaCy pipeline factory |

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
- Dense retrieval (semantic embeddings) → `el_pipeline_lela_dense_candidates`
- BM25 (keyword-based) → `el_pipeline_bm25_candidates`
- Fuzzy matching (string similarity) → `el_pipeline_fuzzy_candidates`

#### Stage 2: Candidate Selection (Disambiguation)

Select the best candidate using LLM reasoning:

```
Input: Context + Mention + Candidates
LLM Reasoning: "The mention 'Paris' appears in context about Olympics in France..."
Output: Paris (city)
```

Implemented via: `el_pipeline_lela_vllm_disambiguator`

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

### LELA NER: `el_pipeline_lela_gliner`

Zero-shot NER using GLiNER with LELA defaults.

**Default Model:** `numind/NuNER_Zero-span`

**Default Labels:**
- person
- organization
- location

**Configuration:**
```python
nlp.add_pipe("el_pipeline_lela_gliner", config={
    "model_name": "numind/NuNER_Zero-span",
    "labels": ["person", "organization", "location"],
    "threshold": 0.5,
    "context_mode": "sentence"
})
```

**JSON Config (via ELPipeline):**
```json
{
  "name": "gliner",
  "params": {
    "model_name": "numind/NuNER_Zero-span",
    "labels": ["person", "organization", "location"],
    "threshold": 0.5
  }
}
```

**Note:** The `el_pipeline_lela_gliner` factory uses LELA defaults and can be used directly with `nlp.add_pipe()`. Through `ELPipeline`, use config name `"gliner"` with LELA model parameters.

### LELA Candidate Generation

#### `el_pipeline_lela_dense_candidates`

Dense retrieval using SentenceTransformers and FAISS.

**Default Model:** `Qwen/Qwen3-Embedding-4B`

**Available Models:**
- `sentence-transformers/all-MiniLM-L6-v2` (~0.3GB)
- `BAAI/bge-base-en-v1.5` (~0.5GB)
- `Qwen/Qwen3-Embedding-0.6B` (~2GB)
- `Qwen/Qwen3-Embedding-4B` (~9GB)

**Query Format:**
```
Instruct: Given an entity mention and its context, retrieve the entity
from the knowledge base that the mention refers to.
Query: {mention_text}
```

**spaCy Usage:**
```python
cand = nlp.add_pipe("el_pipeline_lela_dense_candidates", config={
    "model_name": "Qwen/Qwen3-Embedding-4B",
    "top_k": 64,
    "use_context": False
})
cand.initialize(kb)
```

**JSON Config:**
```json
{
  "name": "lela_dense",
  "params": {
    "model_name": "Qwen/Qwen3-Embedding-4B",
    "top_k": 64
  }
}
```

**Note:** The dense candidate generator uses SentenceTransformers directly for local model loading. No external API server is required.

### LELA Reranking

Two reranker variants are available: one using SentenceTransformers (local) and one using vLLM.

#### `el_pipeline_lela_embedder_transformers_reranker`

Bi-encoder reranker using SentenceTransformers with cosine similarity.

**Default Model:** `Qwen/Qwen3-Embedding-4B`

**Available Models:**
- `Qwen/Qwen3-Embedding-0.6B` (~2GB)
- `Qwen/Qwen3-Embedding-4B` (~9GB)

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
nlp.add_pipe("el_pipeline_lela_embedder_transformers_reranker", config={
    "model_name": "Qwen/Qwen3-Embedding-4B",
    "top_k": 10
})
```

**JSON Config:**
```json
{
  "name": "lela_embedder_transformers",
  "params": {
    "model_name": "Qwen/Qwen3-Embedding-4B",
    "top_k": 10
  }
}
```

#### `el_pipeline_lela_embedder_vllm_reranker`

Bi-encoder reranker using vLLM with task="embed". Uses manual L2 normalization of embeddings.

**spaCy Usage:**
```python
nlp.add_pipe("el_pipeline_lela_embedder_vllm_reranker", config={
    "model_name": "Qwen/Qwen3-Embedding-4B",
    "top_k": 10
})
```

**JSON Config:**
```json
{
  "name": "lela_embedder_vllm",
  "params": {
    "model_name": "Qwen/Qwen3-Embedding-4B",
    "top_k": 10
  }
}
```

### LELA vLLM Disambiguation: `el_pipeline_lela_vllm_disambiguator`

LLM-based disambiguation using vLLM for fast batched inference.

**Default Model:** `Qwen/Qwen3-4B`

**Features:**
- Self-consistency voting
- "None" candidate option
- Simpler, faster for small candidate sets

**spaCy Usage:**
```python
disamb = nlp.add_pipe("el_pipeline_lela_vllm_disambiguator", config={
    "model_name": "Qwen/Qwen3-4B",
    "tensor_parallel_size": 1,
    "add_none_candidate": True,
    "add_descriptions": True,
    "disable_thinking": True,
    "self_consistency_k": 1
})
disamb.initialize(kb)
```

**JSON Config:**
```json
{
  "name": "lela_vllm",
  "params": {
    "model_name": "Qwen/Qwen3-4B",
    "tensor_parallel_size": 1,
    "add_none_candidate": true,
    "add_descriptions": true,
    "disable_thinking": true,
    "self_consistency_k": 1
  }
}
```

### LELA Transformers Disambiguation: `el_pipeline_lela_transformers_disambiguator`

LLM-based disambiguation using HuggingFace Transformers - alternative for older GPUs.

**Default Model:** `Qwen/Qwen3-4B`

**When to Use:**
- Running on older GPUs (P100/Pascal) where vLLM has compatibility issues
- vLLM installation fails or has problems
- You need direct HuggingFace transformers integration

**spaCy Usage:**
```python
disamb = nlp.add_pipe("el_pipeline_lela_transformers_disambiguator", config={
    "model_name": "Qwen/Qwen3-4B",
    "add_none_candidate": True,
    "add_descriptions": True,
    "disable_thinking": True
})
disamb.initialize(kb)
```

**JSON Config:**
```json
{
  "name": "lela_transformers",
  "params": {
    "model_name": "Qwen/Qwen3-4B",
    "add_none_candidate": true,
    "add_descriptions": true,
    "disable_thinking": true
  }
}
```

**Note:** This component loads the model directly with HuggingFace transformers, which may be slower than vLLM but has better compatibility with older hardware.

### Knowledge Base

LELA components use the `custom` JSONL knowledge base.

**Format:**
```jsonl
{"id": "Q937", "title": "Albert Einstein", "description": "German-born theoretical physicist"}
{"id": "Q90", "title": "Paris", "description": "Capital city of France"}
```

**JSON Config:**
```json
{
  "name": "custom",
  "params": {
    "path": "entities.jsonl"
  }
}
```

## Configuration

### Full LELA Pipeline Configuration

```json
{
  "loader": {"name": "text"},
  "ner": {
    "name": "gliner",
    "params": {
      "model_name": "numind/NuNER_Zero-span",
      "labels": ["person", "organization", "location"],
      "threshold": 0.5
    }
  },
  "candidate_generator": {
    "name": "lela_dense",
    "params": {
      "top_k": 64
    }
  },
  "reranker": {
    "name": "lela_embedder_transformers",
    "params": {
      "model_name": "Qwen/Qwen3-Embedding-4B",
      "top_k": 10
    }
  },
  "disambiguator": {
    "name": "lela_vllm",
    "params": {
      "model_name": "Qwen/Qwen3-4B",
      "tensor_parallel_size": 1,
      "add_none_candidate": true,
      "add_descriptions": true,
      "disable_thinking": true
    }
  },
  "knowledge_base": {
    "name": "custom",
    "params": {"path": "kb.jsonl"}
  }
}
```

### Lightweight Configuration

For faster processing without GPU requirements:

```json
{
  "loader": {"name": "text"},
  "ner": {
    "name": "gliner",
    "params": {"threshold": 0.5}
  },
  "candidate_generator": {
    "name": "fuzzy",
    "params": {"top_k": 10}
  },
  "reranker": {"name": "none"},
  "disambiguator": {"name": "first"},
  "knowledge_base": {
    "name": "custom",
    "params": {"path": "kb.jsonl"}
  }
}
```

### NoOp Reranker

The `"none"` reranker config name maps to `el_pipeline_noop_reranker`, which passes candidates through unchanged:

```json
{
  "reranker": {"name": "none"}
}
```

This is useful when:
- You want to skip reranking entirely
- Your candidate generator already orders candidates well
- You're using a lightweight pipeline configuration

### Caching System

The pipeline uses multi-level persistent caching for documents, knowledge bases, and candidate generator indexes.

#### Cache Directory Structure

```
.ner_cache/
  <hash>.pkl                          # Document parsing cache
  kb/<kb_hash>.pkl                    # KB entity data cache
  index/lela_dense_<hash>/            # Dense index
    index.faiss                       # FAISS index
  index/bm25_<hash>.pkl               # rank-bm25 index
```

#### Performance Impact

| Component | Cold Load | Warm Load | Speedup |
|-----------|-----------|-----------|---------|
| KB (YAGO 5M entities) | ~70s | ~10-15s | ~5-7x |
| Dense index | ~20-30s | ~1-3s | ~10-15x |
| rank-bm25 index | ~5-10s | ~1-2s | ~5x |

#### Cache Key Generation

**Document cache:**
```python
key = SHA256(f"{file_path}-{mtime}-{file_size}".encode()).hexdigest()
```

**KB cache:**
```python
key = SHA256(f"kb:{path}:{mtime}:{size}".encode()).hexdigest()
```

**Index caches (depend on KB identity):**
```python
key = SHA256(f"lela_dense:{kb.identity_hash}:{model_name}".encode()).hexdigest()
key = SHA256(f"bm25:{kb.identity_hash}".encode()).hexdigest()
```

#### Cache Invalidation

- **Automatic**: Modifying a source file changes its mtime, invalidating the cache
- **Cascading**: Index caches depend on KB identity_hash, so they auto-invalidate when KB changes
- **Corruption handling**: All loads are wrapped in try/except, corrupted caches fall back to rebuild
- **Manual**: Delete `.ner_cache/` to force full rebuild

**Disabling cache:** Set `cache_dir` to `None` in configuration (not recommended for production)

## LELA Module Structure

**Location:** `el_pipeline/lela/`

### `config.py`

Default configuration values for LELA components.

```python
# NER Settings
NER_LABELS = ["person", "organization", "location"]
DEFAULT_GLINER_MODEL = "numind/NuNER_Zero-span"

# Candidate Generation
DEFAULT_EMBEDDER_MODEL = "Qwen/Qwen3-Embedding-4B"
CANDIDATES_TOP_K = 64
RETRIEVER_TASK = "Given an ambiguous mention, retrieve relevant entities..."

# Reranking
DEFAULT_RERANKER_MODEL = "tomaarsen/Qwen3-Reranker-4B-seq-cls"
RERANKER_TOP_K = 10
RERANKER_TASK = "Given a text with a mention enclosed between '[' and ']'..."

# Disambiguation
DEFAULT_LLM_MODEL = "Qwen/Qwen3-4B"
NOT_AN_ENTITY = ""
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

**SentenceTransformer Pool:**
- Manages local SentenceTransformer instances for embeddings
- Lazy loading with memory-aware eviction of unused models
- Functions: `get_sentence_transformer_instance()`, `release_sentence_transformer()`

**vLLM Instance Manager:**
- Lazy loading of vLLM with memory-aware eviction
- Caches by model_name:tensor_parallel_size
- Functions: `get_vllm_instance()`, `release_vllm()`, `clear_vllm_instances()`

## Usage Examples

### Python API with LELA (via ELPipeline)

```python
from el_pipeline.config import PipelineConfig
from el_pipeline.pipeline import ELPipeline
import json

# Load LELA configuration
config_dict = {
    "loader": {"name": "text"},
    "ner": {"name": "gliner", "params": {"threshold": 0.5}},
    "candidate_generator": {"name": "lela_dense", "params": {"top_k": 64}},
    "reranker": {"name": "none"},
    "disambiguator": {"name": "first"},
    "knowledge_base": {"name": "custom", "params": {"path": "kb.jsonl"}}
}

config = PipelineConfig.from_dict(config_dict)
pipeline = ELPipeline(config)

# Process document
results = pipeline.run(["document.txt"], output_path="results.jsonl")
```

### Direct spaCy Usage with LELA Components

```python
import spacy
from el_pipeline import spacy_components  # Register factories
from el_pipeline.knowledge_bases.custom import CustomJSONLKnowledgeBase

# Build LELA pipeline manually
nlp = spacy.blank("en")

# Add LELA NER
nlp.add_pipe("el_pipeline_lela_gliner", config={
    "threshold": 0.5,
    "labels": ["person", "organization", "location"]
})

# Add dense candidates
cand = nlp.add_pipe("el_pipeline_lela_dense_candidates", config={
    "top_k": 64
})

# Add reranker (optional)
nlp.add_pipe("el_pipeline_noop_reranker")

# Add disambiguator
disamb = nlp.add_pipe("el_pipeline_first_disambiguator")

# Initialize with KB
kb = CustomJSONLKnowledgeBase(path="kb.jsonl")
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
python -m el_pipeline.cli \
  --config config/lela_example.json \
  --input document.txt \
  --output results.jsonl
```

### Web App with LELA

1. Start the web app: `python app.py`
2. Select "lela_gliner" or "gliner" for NER
3. Select "lela_dense" or "bm25" for candidate generation
4. Optionally enable "lela_embedder_transformers" or "lela_embedder_vllm" reranker
5. Select "lela_vllm" or "lela_transformers" for disambiguation
6. Upload your knowledge base and documents

## Performance Considerations

### GPU Requirements

| Component | GPU Recommended | Notes |
|-----------|-----------------|-------|
| lela_gliner | Yes | GLiNER model inference |
| lela_dense | Yes | Embedding computation via SentenceTransformers |
| lela_embedder_transformers | Yes | Embedding computation via SentenceTransformers |
| lela_embedder_vllm | Yes (Required) | vLLM-based embedding |
| lela_cross_encoder_vllm | Yes (Required) | vLLM-based cross-encoder |
| lela_vllm | Yes (Required) | vLLM-based LLM inference |
| lela_transformers | Yes | HuggingFace transformers (better P100 support) |

### CUDA Compatibility

For **P100/Pascal GPUs** (compute capability 6.0):
- Use `torch==2.7.1+cu118` (CUDA 11.8)
- Use `vllm==0.10.1` (latest version compatible with torch 2.7.1+cu118)
- Install from PyTorch index: `pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118`

For **newer GPUs** (A100, H100, etc.):
- Can use latest PyTorch/vLLM with CUDA 12.x

### Model Sizes

| Model | Size | VRAM Required |
|-------|------|---------------|
| **NER** | | |
| NuNER_Zero-span | ~350MB | ~2GB |
| **Embedding** | | |
| MiniLM-L6 | ~80MB | ~0.3GB |
| BGE-Base | ~400MB | ~0.5GB |
| Qwen3-Embedding-0.6B | ~1.2GB | ~2GB |
| Qwen3-Embedding-4B | ~8GB | ~9GB |
| **LLM (Disambiguation)** | | |
| Qwen3-0.6B | ~1.2GB | ~2GB |
| Qwen3-1.7B | ~3.4GB | ~4GB |
| Qwen3-4B | ~8GB | ~9GB |
| Qwen3-8B | ~16GB | ~18GB |

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
