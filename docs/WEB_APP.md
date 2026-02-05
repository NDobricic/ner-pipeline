# Web Application (Gradio UI) Documentation

The NER Pipeline includes an interactive web interface built with Gradio for experimenting with different pipeline configurations. The web app uses spaCy's pipeline architecture under the hood.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command-Line Options](#command-line-options)
- [Interface Overview](#interface-overview)
- [Configuration Options](#configuration-options)
- [Using the Interface](#using-the-interface)
- [Features](#features)
- [Example Configurations](#example-configurations)

## Installation

```bash
cd ner-pipeline
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# If using spaCy NER
python -m spacy download en_core_web_sm
```

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

The web interface uses a single-page layout with configuration at the top and results below.

### Input Section

- **Text Input**: Direct text entry box with sample text pre-filled
- **File Upload**: Upload documents (txt, pdf, docx, html)
- **Knowledge Base**: Upload JSONL file containing entities (optional — defaults to YAGO 4.5, auto-downloaded on first use)

### Configuration Section

A horizontal row of component configuration columns:

1. **NER**: Model selection and model-specific parameters
2. **Candidates**: Generator selection, embedding model (for dense), top-K, context usage
3. **Reranking**: Reranker selection, embedding model (for LELA embedder), top-K
4. **Disambiguation**: Method selection, LLM model choice

**Memory Estimation Display**: Shows real-time GPU memory estimates above the configuration, including:
- GPU name and available VRAM
- Allocatable memory (considering vLLM's 90% utilization)
- Updates dynamically as you change component selections

### Output Section

1. **Linked Entities**
   - Highlighted text showing detected entities
   - Color-coded by entity type (consistent colors based on label hash)
   - **Interactive popups on hover** showing entity details

2. **Confidence Filter Slider**
   - Adjusts threshold (0-1) for graying out low-confidence links
   - Entities below threshold display in gray
   - Statistics update to show above/below threshold counts

3. **Statistics Panel**
   - Total entities, linked vs. not-in-KB counts
   - Average confidence score
   - Threshold breakdown when filter is active

4. **Full JSON Output** (collapsible accordion)
   - Complete pipeline output with all entity details
   - Includes `linking_confidence_normalized` scores

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
- **Embedding Model**: Selectable from dropdown:
  - MiniLM-L6 (~0.3GB VRAM)
  - BGE-Base (~0.5GB VRAM)
  - Qwen3-Embed-0.6B (~2GB VRAM)
  - Qwen3-Embed-4B (~9GB VRAM)
- **top_k**: Number of candidates
- **use_context**: Include mention context in query
- Uses SentenceTransformer for local embedding computation

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
- **Embedding Model**: Selectable from dropdown (same choices as LELA Dense)
- **top_k**: Number of candidates to keep
- Reranks using cosine similarity with marked mention
- Uses SentenceTransformer for local embedding computation

### Disambiguation Options

#### None
- No disambiguation, returns candidates without selection

#### First
- **spaCy Factory:** `ner_pipeline_first_disambiguator`
- Selects the first candidate from the list

#### Popularity
- **spaCy Factory:** `ner_pipeline_popularity_disambiguator`
- Selects the candidate with the highest score

#### LELA vLLM
- **spaCy Factory:** `ner_pipeline_lela_vllm_disambiguator`
- **LLM Model**: Same model choices as LELA vLLM
- Sends all candidates at once (simpler, faster for small candidate sets)
- Uses vLLM for fast batched inference

#### LELA Transformers
- **spaCy Factory:** `ner_pipeline_lela_transformers_disambiguator`
- **LLM Model**: Same model choices as above
- Alternative for older GPUs (P100/Pascal) where vLLM has issues
- Uses HuggingFace transformers directly

## Using the Interface

### Basic Workflow

1. **Enter or Upload Text**
   - Type text directly in the input box, OR
   - Upload a file (txt, pdf, docx, html)

2. **Upload Knowledge Base** (optional)
   - Upload a JSONL file containing entities, or skip to use YAGO 4.5 (auto-downloaded on first use)
   - Format: `{"id": "...", "title": "...", "description": "..."}`

3. **Configure Pipeline**
   - Select NER model and parameters
   - Choose candidate generator and embedding model (if using dense)
   - Optionally configure reranker and disambiguator
   - For LLM disambiguation, select model size based on available VRAM
   - Watch the memory estimate display to ensure configuration fits

4. **Run Pipeline**
   - Click **Run Pipeline** button
   - Button changes to **Cancel** during execution
   - Progress bar shows current stage
   - Click Cancel to interrupt long-running operations

5. **Explore Results**
   - Hover over highlighted entities to see details
   - Adjust confidence threshold to filter low-confidence links
   - Expand JSON output for full details

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

### Pipeline Cancellation

Long-running pipelines can be cancelled:
- Click **Cancel** button (replaces Run button during execution)
- Pipeline stops at the next checkpoint
- Button shows "Cancelling..." while stopping
- Useful for LLM-based disambiguation which can take time

### Highlighted Entity Display

The output shows detected entities with color-coded highlighting:
- Each entity type gets a distinct, consistent color (based on D3 Category20 palette)
- **Interactive popups on hover** showing:
  - Entity title from knowledge base
  - Entity ID
  - Entity type/label
  - Linking confidence (percentage)
  - Entity description (truncated if long)
- Legend items also have hover popups with summary info
- Hovering a legend item highlights only entities of that type (others turn gray)

### Confidence Filtering

Filter results by linking confidence in real-time:
- **Confidence slider** (0.0 - 1.0) controls the threshold
- Entities below threshold are grayed out (per-instance)
- Legend is grayed only when ALL instances of that label are below threshold
- Statistics panel shows above/below threshold breakdown
- Filtering is instant (no re-run needed)

### Memory Estimation

Real-time GPU memory estimates help you choose appropriate models:
- Shows GPU name and available VRAM
- Displays allocatable memory (90% of free, per vLLM)
- Updates dynamically as you change components
- Warns if configuration may exceed available memory

### Dynamic Parameter UI

- Parameters change based on selected component
- Only relevant options are displayed
- LELA-specific components show additional configuration
- Embedding model dropdowns for dense candidates and rerankers
- LLM model dropdown for disambiguators with VRAM estimates

### Progress Tracking

The interface shows processing progress:
- 10% - Building configuration
- 15-35% - Initializing pipeline (loading models)
- 40% - Loading document
- 45-85% - Processing document (NER, candidates, disambiguation)
- 90% - Formatting output
- 100% - Complete

### Error Handling

- Clear error messages for common issues
- Missing file errors with helpful suggestions
- Configuration validation
- Model loading failures with traceback
- GPU memory warnings

### Full JSON Output

The JSON viewer (collapsible accordion) shows complete results including:
- All detected mentions with positions
- Candidate lists with scores
- Resolved entity information
- `linking_confidence` and `linking_confidence_normalized` scores
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

**4. "CUDA out of memory"**
- Select smaller LLM model (e.g., Qwen3-0.6B instead of Qwen3-4B)
- Reduce candidate top_k and reranker top_k
- Watch the memory estimate display before running

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
app.py                              # Main Gradio application
├── run_pipeline()                  # Main processing generator function
├── filter_entities_by_confidence() # Confidence threshold filtering
├── format_highlighted_text_with_threshold() # Per-instance confidence coloring
├── highlighted_to_html()           # Entity highlighting with popups
├── compute_linking_stats()         # Statistics calculation
├── compute_memory_estimate()       # GPU memory estimation
├── start_cancellation()            # Cancel button handler
├── clear_outputs_for_new_run()     # Run button handler
└── update_*_params()               # Dynamic UI update handlers
```

### How It Works

1. User configures pipeline options in the UI
2. Configuration is translated to `PipelineConfig`
3. `NERPipeline` builds a spaCy `Language` with selected components
4. Text is processed through `nlp(text)`
5. Results are serialized and displayed

## Example Configurations

### Lightweight (No Heavy Models)

Best for quick experiments and small knowledge bases:

- **NER**: simple
- **Candidates**: fuzzy
- **Reranker**: none
- **Disambiguator**: first

### Accurate (With Models)

Better accuracy for production use with larger KBs:

- **NER**: spacy (en_core_web_sm)
- **Candidates**: bm25
- **Reranker**: cross_encoder
- **Disambiguator**: popularity

### Zero-shot

For custom entity types and domain adaptation:

- **NER**: gliner or lela_gliner
- **Candidates**: dense or lela_bm25
- **Reranker**: none or lela_embedder
- **Disambiguator**: popularity or lela_vllm

### Full LELA Pipeline

Maximum accuracy with LLM disambiguation:

- **NER**: lela_gliner (threshold: 0.5)
- **Candidates**: lela_bm25 (top_k: 64)
- **Reranker**: lela_embedder (top_k: 10, Qwen3-Embed-4B)
- **Disambiguator**: lela_vllm (Qwen3-4B)

**VRAM Requirements:**
- Qwen3-0.6B LLM: ~4GB total
- Qwen3-4B LLM: ~12GB total
- Qwen3-8B LLM: ~20GB total
