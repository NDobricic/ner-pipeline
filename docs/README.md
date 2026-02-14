# LELA Documentation

A modular Named Entity Recognition (NER) and Entity Linking pipeline built on **spaCy's component architecture**. This project provides a complete solution for extracting named entities from documents and linking them to entities in a knowledge base.

## Table of Contents

- [Overview](#overview)
- [Project Purpose](#project-purpose)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Documentation Index](#documentation-index)

## Overview

LELA leverages spaCy's native pipeline system to create a configurable entity linking solution. It features:

- **spaCy Integration**: All core components (NER, candidate generation, reranking, disambiguation) are implemented as spaCy pipeline components
- **Modular Architecture**: Each pipeline stage is a pluggable spaCy component that can be swapped independently
- **Multiple Interfaces**: CLI, Python API, and Web UI (Gradio)
- **LELA Integration**: Advanced entity linking components based on the LELA research methodology
- **Flexible Input**: Supports text, PDF, DOCX, HTML, JSON, and JSONL documents
- **Caching**: Intelligent document caching for efficient reprocessing

## Project Purpose

This pipeline addresses the entity linking task: mapping ambiguous mentions of entities in natural language text to reference entities in a knowledge base. For example, given the text:

> "France hosted the 2024 Olympics in Paris."

The pipeline identifies "Paris" as a mention and links it to the correct entity (Paris, the capital city of France) from candidates like "Paris (city)", "Paris (novel)", or "Paris (Texas)".

### Key Use Cases

1. **Knowledge Graph Construction**: Extract and link entities to build structured knowledge
2. **Information Extraction**: Identify and disambiguate entities in documents
3. **Question Answering**: Provide entity context for QA systems
4. **Domain-Specific Entity Linking**: Adapt to custom knowledge bases without fine-tuning

## Architecture

The pipeline uses spaCy's component system where each stage is a registered factory:

```
┌───────────────────────────────────────────────────────────────┐
│                       Input Document                          │
│            (text, PDF, DOCX, HTML, JSON, JSONL)               │
└───────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                           LOADER                              │
│       Parse document format and extract text content          │
│      (Registry-based: text, pdf, docx, html, json, jsonl)     │
└───────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                     spaCy Pipeline (nlp)                      │
├───────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  NER Component (doc.ents populated)                     │  │
│  │  Factories: lela_lela_gliner, _simple, _gliner, │  │
│  │             or spaCy's built-in NER + _ner_filter       │  │
│  └─────────────────────────────────────────────────────────┘  │
│                              │                                │
│                              ▼                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Candidate Generator (ent._.candidates populated)       │  │
│  │  Factories: lela_lela_dense_candidates,          │  │
│  │             _fuzzy_candidates, _bm25_candidates         │  │
│  └─────────────────────────────────────────────────────────┘  │
│                              │                                │
│                              ▼                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Reranker (ent._.candidates reordered)                  │  │
│  │  Factories: lela_lela_embedder_transformers_reranker, │  │
│  │             _lela_embedder_vllm_reranker,                │  │
│  │             _lela_cross_encoder_vllm_reranker,           │  │
│  │             _cross_encoder_reranker, _noop_reranker     │  │
│  └─────────────────────────────────────────────────────────┘  │
│                              │                                │
│                              ▼                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Disambiguator (ent._.resolved_entity set)              │  │
│  │  Factories: lela_lela_vllm_disambiguator,       │  │
│  │             _lela_transformers_disambiguator,            │  │
│  │             _first_disambiguator                        │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                           OUTPUT                              │
│                 JSONL with resolved entities                  │
└───────────────────────────────────────────────────────────────┘
```

### spaCy Extensions

The pipeline uses spaCy's custom extension system on `Span` objects:

| Extension | Type | Description |
|-----------|------|-------------|
| `ent._.context` | `str` | Surrounding context for the entity mention |
| `ent._.candidates` | `List[Candidate]` | Candidate entities (each with `entity_id`, `score`, `description`) |
| `ent._.candidate_scores` | `List[float]` | Candidate scores (parallel to candidates list) |
| `ent._.resolved_entity` | `Entity` | The final resolved entity object |

## Project Structure

```
lela/
├── lela/              # Main Python package
│   ├── __init__.py            # Package exports
│   ├── types.py               # Data models (Document, Mention, Entity, etc.)
│   ├── config.py              # PipelineConfig for configuration parsing
│   ├── registry.py            # Component registries (loaders, KBs)
│   ├── pipeline.py            # Main ELPipeline orchestrator (spaCy-based)
│   ├── context.py             # Context extraction utilities
│   ├── cli.py                 # CLI entry point
│   │
│   ├── spacy_components/      # spaCy pipeline components
│   │   ├── __init__.py        # Factory registration
│   │   ├── ner.py             # NER components
│   │   ├── candidates.py      # Candidate generation components
│   │   ├── rerankers.py       # Reranking components
│   │   └── disambiguators.py  # Disambiguation components
│   │
│   ├── loaders/               # Document input handlers (registry-based)
│   ├── knowledge_bases/       # Entity knowledge bases (registry-based)
│   ├── utils/                 # Shared utilities (extensions, span filtering)
│   ├── lela/                  # LELA configuration and utilities
│   └── scripts/               # Utility scripts
│
├── app.py                     # Gradio web UI
├── tests/                     # Test suite (unit, integration, slow)
├── config/                    # Configuration examples
├── data/                      # Sample data and test files
├── docs/                      # Documentation
├── requirements.txt           # Dependencies
└── pyproject.toml             # Project metadata
```

## Quick Start

### Installation

**Requirements:** Python 3.10 (recommended), CUDA 12.x for GPU support

```bash
cd lela

# Create virtual environment with Python 3.10
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# For spaCy models (optional - only if using spaCy NER)
python -m spacy download en_core_web_sm
```

### Basic Usage

**Using the CLI:**
```bash
python -m lela.cli \
  --config config/lela_bm25_only.json \
  --input document.txt \
  --output results.jsonl
```

**Using the Python API:**
```python
from lela import PipelineConfig, ELPipeline

# Load configuration and run pipeline
config = PipelineConfig.from_json("config.json")
pipeline = ELPipeline(config)
results = pipeline.run(["document.txt"], output_path="results.jsonl")
```

**Using spaCy directly (advanced):**
```python
import spacy
from lela import spacy_components  # Register factories

# Build custom pipeline
nlp = spacy.blank("en")
nlp.add_pipe("lela_simple", config={"min_len": 3})
nlp.add_pipe("lela_fuzzy_candidates", config={"top_k": 10})
nlp.add_pipe("lela_first_disambiguator")

# Initialize components with KB
from lela.knowledge_bases.custom import CustomJSONLKnowledgeBase
kb = CustomJSONLKnowledgeBase(path="kb.jsonl")

for name, component in nlp.pipeline:
    if hasattr(component, "initialize"):
        component.initialize(kb)

# Process text
doc = nlp("Albert Einstein was born in Germany.")
for ent in doc.ents:
    print(f"{ent.text}: {ent._.resolved_entity}")
```

**Using the Web UI:**
```bash
python app.py --port 7860
# Open http://localhost:7860 in your browser
```

### Minimal Configuration

```json
{
  "loader": {"name": "text"},
  "ner": {"name": "simple", "params": {"min_len": 3}},
  "candidate_generator": {"name": "fuzzy", "params": {"top_k": 10}},
  "reranker": {"name": "none"},
  "disambiguator": {"name": "first"},
  "knowledge_base": {"name": "custom", "params": {"path": "kb.jsonl"}},
  "cache_dir": ".ner_cache"
}
```

## Documentation Index

| Document | Description |
|----------|-------------|
| [API.md](API.md) | Python API reference, spaCy components, and programmatic usage |
| [CLI.md](CLI.md) | Command-line interface documentation |
| [WEB_APP.md](WEB_APP.md) | Gradio web application guide |
| [PIPELINE.md](PIPELINE.md) | Detailed pipeline architecture and spaCy component documentation |
| [LELA.md](LELA.md) | LELA methodology and integration |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common errors and solutions |
| [REQUIREMENTS.md](REQUIREMENTS.md) | System requirements and GPU compatibility |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Creating custom components |
| [TESTING.md](TESTING.md) | Running and writing tests |

## Requirements

- Python 3.10 (recommended; 3.13 is not supported due to vLLM incompatibility)
- PyTorch 2.9.x
- vLLM 0.15.x (for LELA vLLM disambiguator/reranker)
- See `requirements.txt` for full dependency list

### GPU Support

- **CUDA 12.x**: Required for PyTorch GPU acceleration
- Recommended GPUs: V100, A100, H100, RTX 3090/4090

### Optional Dependencies

- **spaCy Models**: Required for spaCy's built-in NER (`python -m spacy download en_core_web_sm`)
