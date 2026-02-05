# NER Pipeline Documentation

A modular Named Entity Recognition (NER) and Entity Linking pipeline built on **spaCy's component architecture**. This project provides a complete solution for extracting named entities from documents and linking them to entities in a knowledge base.

## Table of Contents

- [Overview](#overview)
- [Project Purpose](#project-purpose)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Documentation Index](#documentation-index)

## Overview

The NER Pipeline leverages spaCy's native pipeline system to create a configurable entity linking solution. It features:

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
│  │  Factories: ner_pipeline_lela_gliner, _simple, _gliner, │  │
│  │             or spaCy's built-in NER                     │  │
│  └─────────────────────────────────────────────────────────┘  │
│                              │                                │
│                              ▼                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Candidate Generator (ent._.candidates populated)       │  │
│  │  Factories: ner_pipeline_lela_bm25_candidates,          │  │
│  │             _lela_dense_candidates, _fuzzy_candidates,  │  │
│  │             _bm25_candidates                            │  │
│  └─────────────────────────────────────────────────────────┘  │
│                              │                                │
│                              ▼                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Reranker (ent._.candidates reordered)                  │  │
│  │  Factories: ner_pipeline_lela_embedder_reranker,        │  │
│  │             _cross_encoder_reranker, _noop_reranker     │  │
│  └─────────────────────────────────────────────────────────┘  │
│                              │                                │
│                              ▼                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Disambiguator (ent._.resolved_entity set)              │  │
│  │  Factories: ner_pipeline_lela_vllm_disambiguator,       │  │
│  │             _lela_transformers_disambiguator,            │  │
│  │             _first_disambiguator,                       │  │
│  │             _popularity_disambiguator                   │  │
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
| `ent._.resolved_entity` | `Entity` | The final resolved entity object |

## Project Structure

```
ner-pipeline/
├── ner_pipeline/              # Main Python package
│   ├── __init__.py            # Package exports
│   ├── types.py               # Data models (Document, Mention, Entity, etc.)
│   ├── config.py              # PipelineConfig for configuration parsing
│   ├── registry.py            # Component registries (loaders, KBs)
│   ├── pipeline.py            # Main NERPipeline orchestrator (spaCy-based)
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

**Requirements:** Python 3.10 (recommended), CUDA 11.8 for GPU support

```bash
cd ner-pipeline

# Create virtual environment with Python 3.10
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install PyTorch with CUDA 11.8 (required for P100/older GPUs)
pip install torch==2.6.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt

# For spaCy models (optional - only if using spaCy NER)
python -m spacy download en_core_web_sm
```

### Basic Usage

**Using the CLI:**
```bash
python -m ner_pipeline.cli \
  --config config/lela_bm25_only.json \
  --input document.txt \
  --output results.jsonl
```

**Using the Python API:**
```python
from ner_pipeline.config import PipelineConfig
from ner_pipeline.pipeline import NERPipeline
import json

# Load configuration
with open("config.json") as f:
    config = PipelineConfig.from_dict(json.load(f))

# Create and run pipeline
pipeline = NERPipeline(config)
results = pipeline.run(["document.txt"], output_path="results.jsonl")
```

**Using spaCy directly (advanced):**
```python
import spacy
from ner_pipeline import spacy_components  # Register factories

# Build custom pipeline
nlp = spacy.blank("en")
nlp.add_pipe("ner_pipeline_simple", config={"min_len": 3})
nlp.add_pipe("ner_pipeline_fuzzy_candidates", config={"top_k": 10})
nlp.add_pipe("ner_pipeline_first_disambiguator")

# Initialize components with KB
from ner_pipeline.knowledge_bases.custom import CustomJSONLKnowledgeBase
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
  "knowledge_base": {"name": "custom", "params": {"path": "kb.jsonl"}}
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

- Python 3.10 (recommended; 3.13 has compatibility issues with vLLM)
- spaCy 3.8+
- PyTorch 2.6.0+cu118
- vLLM 0.8.5 (for LELA vLLM disambiguator)
- See `requirements.txt` for full dependency list

### GPU Support

- **CUDA 11.8**: Required for PyTorch GPU acceleration
- **P100/Pascal GPUs**: Use `torch==2.6.0+cu118` and `vllm==0.8.5` (newer versions drop support for compute capability 6.0)
- **Newer GPUs (A100, etc.)**: Can use newer PyTorch/vLLM versions with CUDA 12.x

### Optional Dependencies

- **spaCy Models**: Required for spaCy's built-in NER (`python -m spacy download en_core_web_sm`)
