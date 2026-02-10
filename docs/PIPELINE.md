# Pipeline Architecture and spaCy Components

This document provides detailed documentation of the EL Pipeline architecture built on spaCy's component system.

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [spaCy Integration](#spacy-integration)
- [NER Components](#ner-components)
- [Candidate Generation Components](#candidate-generation-components)
- [Reranker Components](#reranker-components)
- [Disambiguator Components](#disambiguator-components)
- [Document Loaders](#document-loaders)
- [Knowledge Bases](#knowledge-bases)
- [Context Extraction](#context-extraction)
- [Caching System](#caching-system)

## Pipeline Overview

### Processing Flow

```
Input Files → Loader → Documents → spaCy Pipeline → Serialization → Output
                                        │
                   ┌────────────────────┼────────────────────┐
                   │                    │                    │
                   ▼                    ▼                    ▼
            NER Component    Candidate Component    Disambiguator
            (doc.ents)       (ent._.candidates)    (ent._.resolved_entity)
```

### Stage Responsibilities

| Stage | spaCy Component | Output | Purpose |
|-------|-----------------|--------|---------|
| Loader | Registry-based | Document(s) | Parse file format, extract text |
| NER | `el_pipeline_*` | `doc.ents` | Identify entity mentions |
| Candidate Gen | `el_pipeline_*_candidates` | `ent._.candidates` | Find KB matches |
| Reranker | `el_pipeline_*_reranker` | `ent._.candidates` (reordered) | Reorder by relevance |
| Disambiguator | `el_pipeline_*_disambiguator` | `ent._.resolved_entity` | Select final entity |

## spaCy Integration

### How the Pipeline Works

The EL Pipeline uses spaCy's native component architecture:

1. **Factory Registration**: All components are registered as spaCy factories on import
2. **Pipeline Building**: `ELPipeline` creates a blank English `Language` and adds components
3. **Component Initialization**: Components requiring KB are initialized after being added
4. **Document Processing**: Text is passed through `nlp(text)` which runs all components

### Custom Extensions

Components communicate via spaCy's extension system on `Span` objects:

```python
from spacy.tokens import Span

# Registered automatically by components
Span.set_extension("context", default=None)
Span.set_extension("candidates", default=[])
Span.set_extension("resolved_entity", default=None)
```

### Building a Pipeline Manually

```python
import spacy
from el_pipeline import spacy_components  # Register all factories

nlp = spacy.blank("en")

# Add NER
nlp.add_pipe("el_pipeline_lela_gliner", config={
    "threshold": 0.5,
    "labels": ["person", "organization"]
})

# Add candidate generation
cand = nlp.add_pipe("el_pipeline_lela_dense_candidates", config={
    "top_k": 64
})

# Add reranker
nlp.add_pipe("el_pipeline_noop_reranker")

# Add disambiguator
disamb = nlp.add_pipe("el_pipeline_first_disambiguator")

# Initialize components with KB
from el_pipeline.knowledge_bases.custom import CustomJSONLKnowledgeBase
kb = CustomJSONLKnowledgeBase(path="kb.jsonl")
cand.initialize(kb)
disamb.initialize(kb)

# Process
doc = nlp("Albert Einstein visited Paris.")
```

### Factory Names Reference

| Config Name | spaCy Factory Name |
|-------------|-------------------|
| `simple` | `el_pipeline_simple` |
| `gliner` | `el_pipeline_gliner` |
| `lela_dense` | `el_pipeline_lela_dense_candidates` |
| `fuzzy` | `el_pipeline_fuzzy_candidates` |
| `bm25` | `el_pipeline_bm25_candidates` |
| `lela_embedder_transformers` | `el_pipeline_lela_embedder_transformers_reranker` |
| `lela_embedder_vllm` | `el_pipeline_lela_embedder_vllm_reranker` |
| `lela_cross_encoder_vllm` | `el_pipeline_lela_cross_encoder_vllm_reranker` |
| `cross_encoder` | `el_pipeline_lela_cross_encoder_reranker` |
| `lela_vllm_api_client` | `el_pipeline_lela_vllm_api_client_reranker` |
| `lela_llama_server` | `el_pipeline_lela_llama_server_reranker` |
| `none` | `el_pipeline_noop_reranker` |
| `lela_vllm` | `el_pipeline_lela_vllm_disambiguator` |
| `lela_transformers` | `el_pipeline_lela_transformers_disambiguator` |
| `lela_openai_api` | `el_pipeline_lela_openai_api_disambiguator` |
| `first` | `el_pipeline_first_disambiguator` |

**Note:** The `el_pipeline_lela_gliner` factory is registered and can be used directly with `nlp.add_pipe()`, but is not yet available as a config name through `ELPipeline`.

---

## NER Components

NER components populate `doc.ents` with detected entity spans and set `ent._.context`.

**Location:** `el_pipeline/spacy_components/ner.py`

### LELAGLiNERComponent

Zero-shot GLiNER NER with LELA defaults.

**Factory:** `el_pipeline_lela_gliner`

**Config:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "numind/NuNER_Zero-span" | GLiNER model |
| `labels` | List[str] | LELA defaults | Entity types |
| `threshold` | float | 0.5 | Detection threshold |
| `context_mode` | str | "sentence" | Context extraction |

**LELA Default Labels:**
- person
- organization
- location

**Behavior:**
- Uses GLiNER library for zero-shot NER
- Filters overlapping spans (keeps longest)
- Extracts context around each mention

### SimpleNERComponent

Lightweight regex-based NER.

**Factory:** `el_pipeline_simple`

**Config:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_len` | int | 3 | Minimum mention length |
| `context_mode` | str | "sentence" | Context extraction |

**Regex Pattern:** `\b([A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+)*)\b`

**Behavior:**
- Finds capitalized word sequences
- Labels all mentions as "ENT"
- Fast, no model downloads required

### GLiNERComponent

Standard GLiNER wrapper.

**Factory:** `el_pipeline_gliner`

**Config:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "urchade/gliner_base" | GLiNER model |
| `labels` | List[str] | ["person", "organization", "location"] | Entity types |
| `threshold` | float | 0.5 | Detection threshold |
| `context_mode` | str | "sentence" | Context extraction |

### NER Filter Component

Post-filter for spaCy's built-in NER.

**Component:** `el_pipeline_ner_filter` (not a factory, use `@Language.component`)

**Usage:**
```python
# Use spaCy's built-in NER
spacy_nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("ner", source=spacy_nlp)
nlp.add_pipe("el_pipeline_ner_filter")
```

**Behavior:**
- Adds `_.context` to existing entities from spaCy's NER
- Use after spaCy's built-in `ner` component

---

## Long Document Handling

NER components automatically handle documents that exceed model context limits through chunking strategies.

### GLiNER Chunking

The LELA GLiNER component (`el_pipeline_lela_gliner`) chunks long documents with overlap:

**Parameters:**
- Chunk size: ~1500 characters
- Overlap: 200 characters
- Boundary detection: Sentence-aware

**Algorithm:**
1. If document ≤ 1500 chars: process directly
2. Otherwise, split into overlapping chunks:
   - Start at position 0
   - Find chunk end at ~1500 chars
   - Look for sentence boundary (`. `, `.\n`, `? `, `!\n`, `\n\n`) near end
   - If found after half the chunk, break there
   - Process chunk, adjust entity offsets back to document coordinates
   - Move start forward by (chunk_length - overlap)
   - Repeat until end of document

**Overlap rationale:**
- 200-char overlap ensures entities near chunk boundaries aren't missed
- Duplicate entities (same span, different chunks) are filtered by longest-span-wins

**Example:**
```
Document: 3000 characters
├── Chunk 1: chars 0-1500 (processes entities in this range)
├── Chunk 2: chars 1300-2800 (200-char overlap with chunk 1)
└── Chunk 3: chars 2600-3000 (200-char overlap with chunk 2)

Entity at chars 1450-1480:
- Found in Chunk 1 (at local 1450-1480)
- Also found in Chunk 2 (at local 150-180)
- Deduplication keeps one copy
```

### spaCy NER

spaCy's built-in NER does not have explicit context limits and processes documents as a whole. For very long documents, consider:

1. Using a pipeline-level document chunking strategy
2. Switching to GLiNER NER which handles chunking automatically

---

## Candidate Generation Components

Candidate components populate `ent._.candidates` with `List[Candidate]` objects (each with `entity_id`, `score`, and `description` fields).

**Location:** `el_pipeline/spacy_components/candidates.py`

### LELADenseCandidatesComponent

Dense retrieval using embeddings and FAISS.

**Factory:** `el_pipeline_lela_dense_candidates`

**Config:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | LELA default | Embedding model |
| `top_k` | int | 64 | Maximum candidates |
| `device` | str | None | Device override (e.g., "cuda", "cpu") |
| `use_context` | bool | False | Include context |

**Requires:** `initialize(kb)` call

**Query Format:**
```
Instruct: Given an entity mention and its context, retrieve the entity
from the knowledge base that the mention refers to.
Query: {mention_text}: {context}
```

### FuzzyCandidatesComponent

RapidFuzz string matching.

**Factory:** `el_pipeline_fuzzy_candidates`

**Config:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_k` | int | 20 | Maximum candidates |

**Requires:** `initialize(kb)` call

**Algorithm:**
- Uses `rapidfuzz.process.extract()`
- Matches mention text against entity titles

### BM25CandidatesComponent

Standard BM25 using rank-bm25 library.

**Factory:** `el_pipeline_bm25_candidates`

**Config:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_k` | int | 20 | Maximum candidates |

**Requires:** `initialize(kb)` call

---

## Reranker Components

Reranker components reorder `ent._.candidates` by relevance.

**Location:** `el_pipeline/spacy_components/rerankers.py`

### LELAEmbedderTransformersRerankerComponent

Bi-encoder reranker using SentenceTransformers. Uses cosine similarity between query and candidate embeddings.

**Factory:** `el_pipeline_lela_embedder_transformers_reranker`

**Config:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | LELA default | Embedding model |
| `top_k` | int | 10 | Candidates to keep |
| `device` | str | None | Device override (e.g., "cuda", "cpu") |

### LELAEmbedderVLLMRerankerComponent

Bi-encoder reranker using vLLM with task="embed". Manual L2 normalization of embeddings.

**Factory:** `el_pipeline_lela_embedder_vllm_reranker`

**Config:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | LELA default | Embedding model |
| `top_k` | int | 10 | Candidates to keep |

### LELACrossEncoderVLLMRerankerComponent

Cross-encoder reranker using vLLM `.score()` API with the Qwen3-Reranker-seq-cls model variant.

**Factory:** `el_pipeline_lela_cross_encoder_vllm_reranker`

**Config:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | LELA default | Cross-encoder model |
| `top_k` | int | 10 | Candidates to keep |

### CrossEncoderRerankerComponent

Cross-encoder reranking using sentence-transformers.

**Factory:** `el_pipeline_lela_cross_encoder_reranker`

**Config:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "cross-encoder/ms-marco-MiniLM-L-6-v2" | Model |
| `top_k` | int | 10 | Candidates to keep |

**Algorithm:**
- Creates pairs: `({mention} | {full_text}, {candidate_description})`
- Scores each pair with cross-encoder
- Sorts by score, keeps top_k

### NoOpRerankerComponent

Pass-through reranker.

**Factory:** `el_pipeline_noop_reranker`

**Config:** None

**Behavior:** Returns candidates unchanged

---

## Disambiguator Components

Disambiguator components set `ent._.resolved_entity` with the selected `Entity`.

**Location:** `el_pipeline/spacy_components/disambiguators.py`

### LELAvLLMDisambiguatorComponent

vLLM-based LLM disambiguation - sends all candidates at once.

**Factory:** `el_pipeline_lela_vllm_disambiguator`

**Config:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "Qwen/Qwen3-4B" | LLM model |
| `tensor_parallel_size` | int | 1 | GPU parallelism |
| `max_model_len` | int | None | Max context length |
| `add_none_candidate` | bool | True | Add "None" option |
| `add_descriptions` | bool | True | Include descriptions |
| `disable_thinking` | bool | True | Disable reasoning |
| `system_prompt` | str | LELA default | Custom prompt |
| `generation_config` | dict | {} | vLLM settings |
| `self_consistency_k` | int | 1 | Voting samples |

**Requires:** `initialize(kb)` call

**Prompt Format:**
```
System: You are an expert designed to disambiguate entities in text...

User: Input: "France hosted the 2024 Olympics in [Paris]."

Candidates:
0. None of the candidates
1. Paris (city): Capital city of France
2. Paris (novel): 1897 novel by Emile Zola
...
```

### FirstDisambiguatorComponent

Select first candidate.

**Factory:** `el_pipeline_first_disambiguator`

**Config:** None

**Requires:** `initialize(kb)` call

---

## Document Loaders

Loaders are registry-based (not spaCy components) and parse file formats.

**Location:** `el_pipeline/loaders/`

### TextLoader

**Registration:** `text`

**Behavior:** Reads UTF-8 text files

### JSONLoader

**Registration:** `json`

**Expected Format:**
```json
{"id": "doc-001", "text": "Document content", "meta": {}}
```

### JSONLLoader

**Registration:** `jsonl`

**Expected Format:**
```jsonl
{"id": "doc-001", "text": "First document"}
{"id": "doc-002", "text": "Second document"}
```

### PDFLoader

**Registration:** `pdf`

**Dependencies:** `pdfplumber`

### DocxLoader

**Registration:** `docx`

**Dependencies:** `python-docx`

### HTMLLoader

**Registration:** `html`

**Dependencies:** `beautifulsoup4`, `lxml`

---

## Knowledge Bases

Knowledge bases are registry-based (not spaCy components).

**Location:** `el_pipeline/knowledge_bases/`

### CustomJSONLKnowledgeBase

**Registration:** `custom`

**Config:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `path` | required | Path to JSONL file |
| `cache_dir` | None | Optional directory for persistent caching |

**Format:**
```jsonl
{"id": "Q937", "title": "Albert Einstein", "description": "Theoretical physicist"}
```

---

## Context Extraction

**Location:** `el_pipeline/context.py`

### Sentence Mode

```python
extract_sentence_context(text, start, end, max_sentences=1)
```

Extracts complete sentences containing the mention.

### Window Mode

```python
extract_window_context(text, start, end, window_chars=150)
```

Extracts fixed character window around mention.

### General Function

```python
extract_context(text, start, end, mode="sentence", **kwargs)
```

Dispatcher for both modes.

---

## Caching System

The pipeline implements multi-level persistent caching to significantly reduce initialization time on subsequent runs.

### Cache Directory Structure

```
.ner_cache/
  <hash>.pkl                          # Document parsing cache
  kb/<kb_hash>.pkl                    # KB entity data cache
  index/lela_dense_<hash>/            # Dense index cache
    index.faiss                       # FAISS index file
  index/bm25_<hash>.pkl               # rank-bm25 index cache
```

### Cache Types

| Cache | Cold Load | Warm Load | Speedup |
|-------|-----------|-----------|---------|
| KB (YAGO 5M entities) | ~70s | ~10-15s | ~5-7x |
| Dense index | ~20-30s | ~1-3s | ~10-15x |
| rank-bm25 index | ~5-10s | ~1-2s | ~5x |

### Cache Key Generation

**Document cache:**
```python
key = SHA256(f"{path}-{mtime}-{size}".encode())
```

**KB cache:**
```python
key = SHA256(f"kb:{path}:{mtime}:{size}".encode())
```

**Index caches:**
```python
# Dense
key = SHA256(f"lela_dense:{kb.identity_hash}:{model_name}".encode())

# rank-bm25
key = SHA256(f"bm25:{kb.identity_hash}".encode())
```

### Cache Invalidation

- **Automatic**: Caches are keyed by file path + mtime + size, so modifying the source file invalidates the cache
- **Index dependency**: Index caches depend on KB identity_hash, so they auto-invalidate when KB changes
- **Manual**: Delete the `.ner_cache/` directory to force a full rebuild
- **Corruption handling**: All cache loads are wrapped in try/except; corrupted caches fall back to full rebuild

### Cache Location

Default: `.ner_cache/` (configurable via `cache_dir` in pipeline config)
