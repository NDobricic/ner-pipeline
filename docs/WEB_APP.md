# Web Application (Gradio UI) Documentation

The NER Pipeline includes an interactive web interface built with Gradio for experimenting with different pipeline configurations. The web app uses spaCy's pipeline architecture under the hood.

## Table of Contents

- [Quick Start](#quick-start)
- [Command-Line Options](#command-line-options)
- [Interface Overview](#interface-overview)
- [Configuration Options](#configuration-options)
- [Using the Interface](#using-the-interface)
- [Features](#features)

## Quick Start

```bash
# Start the web application
python app.py

# With custom port
python app.py --port 8080

# With public share link
python app.py --share

# With debug logging
python app.py --log DEBUG
```

Then open your browser to `http://localhost:7860` (or the specified port).

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--port` | 7860 | Port number for the web server |
| `--share` | False | Create a public share link via Gradio |
| `--log` | INFO | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |

### Examples

```bash
# Run on port 8080
python app.py --port 8080

# Create public URL for remote access
python app.py --share

# Enable debug logging for troubleshooting
python app.py --log DEBUG

# Combine options
python app.py --port 8080 --share --log INFO
```

## Interface Overview

The web interface is divided into two main columns:

### Left Column: Input & Configuration

1. **Input Section**
   - Text input box for direct text entry
   - File upload for documents
   - Knowledge Base upload

2. **Loader Configuration**
   - Automatically detected from uploaded file type
   - Manual override available

3. **NER Configuration**
   - Model selection (maps to spaCy component factories)
   - Model-specific parameters

4. **Candidate Generation Configuration**
   - Generator selection
   - Top-K setting

5. **Reranking Configuration**
   - Reranker selection
   - Model-specific parameters

6. **Disambiguation Configuration**
   - Disambiguator selection
   - Model-specific parameters

### Right Column: Output

1. **Linked Entities**
   - Highlighted text showing detected entities
   - Color-coded by entity type

2. **Full Pipeline Output**
   - Complete JSON output with all details
   - Entity information, candidates, scores

## Configuration Options

### Loader Options

| Name | Description | Auto-detected Extensions |
|------|-------------|-------------------------|
| `text` | Plain text files | `.txt` |
| `pdf` | PDF documents | `.pdf` |
| `docx` | Word documents | `.docx` |
| `html` | HTML pages | `.html`, `.htm` |
| `json` | JSON files | `.json` |
| `jsonl` | JSON Lines files | `.jsonl` |

### NER Options

Each NER option maps to a spaCy pipeline factory:

#### Simple (Regex-based)
- **spaCy Factory:** `ner_pipeline_simple`
- **min_len**: Minimum mention length (1-10, default: 3)
- Lightweight, no model downloads required
- Uses regex pattern to find capitalized words

#### spaCy
- Uses spaCy's built-in NER + `ner_pipeline_ner_filter`
- **model**: spaCy model name
  - `en_core_web_sm` (default)
  - `en_core_web_md`
  - `en_core_web_lg`
- Standard NER labels: PERSON, ORG, GPE, LOC, etc.

#### GLiNER
- **spaCy Factory:** `ner_pipeline_gliner`
- **model_name**: GLiNER model (default: `urchade/gliner_large`)
- **labels**: Comma-separated entity labels to detect
- Zero-shot NER with custom labels

#### Transformers
- **spaCy Factory:** `ner_pipeline_transformers`
- **model_name**: HuggingFace model (default: `dslim/bert-base-NER`)
- Standard transformer-based NER

#### LELA GLiNER
- **spaCy Factory:** `ner_pipeline_lela_gliner`
- **model_name**: Default `numind/NuNER_Zero-span`
- **labels**: LELA default labels (person, organization, location, event, work of art, product)
- **threshold**: Detection threshold (default: 0.5)

### Candidate Generation Options

#### Fuzzy
- **spaCy Factory:** `ner_pipeline_fuzzy_candidates`
- **top_k**: Number of candidates (1-20, default: 10)
- Uses RapidFuzz string matching on entity titles

#### BM25
- **spaCy Factory:** `ner_pipeline_bm25_candidates`
- **top_k**: Number of candidates (1-20, default: 10)
- Keyword-based retrieval on entity descriptions

#### Dense
- **spaCy Factory:** `ner_pipeline_fuzzy_candidates` (with sentence-transformers)
- **model_name**: Embedding model (default: `all-MiniLM-L6-v2`)
- **top_k**: Number of candidates
- Uses FAISS for similarity search

#### LELA BM25
- **spaCy Factory:** `ner_pipeline_lela_bm25_candidates`
- **top_k**: Number of candidates (default: 64)
- **use_context**: Include mention context in query
- Uses bm25s with stemming for better matching

#### LELA Dense
- **spaCy Factory:** `ner_pipeline_lela_dense_candidates`
- **model_name**: Embedding model
- **top_k**: Number of candidates
- **base_url**: API endpoint URL
- **port**: API port
- Uses OpenAI-compatible embedding API

### Reranking Options

#### None
- **spaCy Factory:** `ner_pipeline_noop_reranker`
- No reranking, returns candidates as-is

#### Cross Encoder
- **spaCy Factory:** `ner_pipeline_cross_encoder_reranker`
- **model_name**: Cross-encoder model (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **top_k**: Number of candidates to keep

#### LELA Embedder
- **spaCy Factory:** `ner_pipeline_lela_embedder_reranker`
- **model_name**: Embedding model
- **top_k**: Number of candidates to keep
- **base_url**: API endpoint URL
- **port**: API port
- Reranks using cosine similarity with marked mention

### Disambiguation Options

#### None
- No disambiguation, returns candidates without selection

#### First
- **spaCy Factory:** `ner_pipeline_first_disambiguator`
- Selects the first candidate from the list

#### Popularity
- **spaCy Factory:** `ner_pipeline_popularity_disambiguator`
- Selects the candidate with the highest score

#### LLM
- **model_name**: HuggingFace model for zero-shot classification
- Uses NLI-based relevance scoring

#### LELA vLLM
- **spaCy Factory:** `ner_pipeline_lela_vllm_disambiguator`
- **model_name**: LLM model (default: `Qwen/Qwen3-8B`)
- **tensor_parallel_size**: GPU parallelism
- **add_none_candidate**: Include "None" option
- **add_descriptions**: Include entity descriptions in prompt
- Uses vLLM for fast batched inference

## Using the Interface

### Basic Workflow

1. **Enter or Upload Text**
   - Type text directly in the input box, OR
   - Upload a file (txt, pdf, docx, html)

2. **Upload Knowledge Base**
   - Upload a JSONL file containing entities
   - Format: `{"id": "...", "title": "...", "description": "..."}`

3. **Configure Pipeline**
   - Select NER model and parameters
   - Choose candidate generator
   - Optionally configure reranker and disambiguator

4. **Run Pipeline**
   - Click "Run Entity Linking"
   - View results in the output panels

### Default Configuration

The interface starts with sensible defaults:
- **NER**: Simple (min_len: 3)
- **Candidate Generator**: Fuzzy (top_k: 10)
- **Reranker**: None
- **Disambiguator**: First

### Example Input

**Sample Text:**
```
Albert Einstein was born in Germany in 1879. He later moved to the United States
and worked at Princeton University. Einstein received the Nobel Prize in Physics
in 1921 for his work on theoretical physics.
```

**Sample Knowledge Base (JSONL):**
```jsonl
{"id": "Q937", "title": "Albert Einstein", "description": "German-born theoretical physicist"}
{"id": "Q183", "title": "Germany", "description": "Country in Central Europe"}
{"id": "Q30", "title": "United States", "description": "Country in North America"}
{"id": "Q653321", "title": "Princeton University", "description": "Private research university in New Jersey"}
{"id": "Q38104", "title": "Nobel Prize in Physics", "description": "Annual physics award"}
```

## Features

### Highlighted Entity Display

The output shows detected entities with color-coded highlighting:
- Each entity type gets a distinct color
- Hover over entities to see the resolved information
- Labels appear next to each mention

### Dynamic Parameter UI

- Parameters change based on selected component
- Only relevant options are displayed
- LELA-specific components show additional configuration

### Progress Tracking

The interface shows processing progress:
- 10% - Loading document
- 30% - Running NER
- 50% - Generating candidates
- 70% - Reranking candidates
- 90% - Disambiguating entities
- 100% - Complete

### Error Handling

- Clear error messages for common issues
- Missing file errors
- Configuration validation
- Model loading failures

### Full JSON Output

The JSON viewer shows complete results including:
- All detected mentions
- Candidate lists with scores
- Resolved entity information
- Document metadata

## Troubleshooting

### Common Issues

**1. "No entities found"**
- Check that the NER model is configured correctly
- Try a different NER model or lower the threshold
- Ensure the input text contains recognizable entities

**2. "Knowledge base not loaded"**
- Upload a valid JSONL file
- Check the file format matches expected structure

**3. "Model loading failed"**
- Ensure required dependencies are installed
- Check for GPU memory issues with large models
- Try using a smaller model variant

**4. "Connection error" for LELA components**
- Verify the embedding server is running
- Check base_url and port settings

### Performance Tips

- Use Simple NER for quick experiments
- Use LELA components for production quality
- Reduce top_k for faster processing
- Consider GPU acceleration for transformer models

## Architecture Notes

The web app is built using:
- **Gradio 4.0+**: Modern UI framework
- **spaCy Pipeline**: All NER/EL operations via spaCy components
- **Accordion Layout**: Collapsible configuration sections
- **Reactive Updates**: Components update based on selections
- **File Handlers**: Support for multiple document formats

### File Structure

```
app.py                    # Main Gradio application
├── create_ui()           # UI layout definition
├── process_input()       # Main processing function
├── get_highlighted_text() # Entity highlighting
└── build_config()        # Configuration assembly
```

### How It Works

1. User configures pipeline options in the UI
2. Configuration is translated to `PipelineConfig`
3. `NERPipeline` builds a spaCy `Language` with selected components
4. Text is processed through `nlp(text)`
5. Results are serialized and displayed
