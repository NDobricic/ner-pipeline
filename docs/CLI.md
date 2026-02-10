# Command-Line Interface (CLI) Documentation

The EL Pipeline provides a command-line interface for processing documents. Under the hood, the CLI uses spaCy's pipeline architecture for all NER and entity linking operations.

## Table of Contents

- [Quick Start](#quick-start)
- [Command Syntax](#command-syntax)
- [Arguments](#arguments)
- [Configuration File](#configuration-file)
- [Examples](#examples)
- [Output Format](#output-format)

## Quick Start

```bash
# Basic usage
python -m el_pipeline.cli \
  --config config.json \
  --input document.txt \
  --output results.jsonl
```

## Command Syntax

```bash
python -m el_pipeline.cli --config <config_file> --input <input_files...> [--output <output_file>]
```

Or using the wrapper script:

```bash
python cli.py --config <config_file> --input <input_files...> [--output <output_file>]
```

## Arguments

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--config`, `-c` | Path to the JSON configuration file |
| `--input`, `-i` | One or more input file paths |

### Optional Arguments

| Argument | Description |
|----------|-------------|
| `--output`, `-o` | Path to write JSONL output file. If not specified, results are printed to stdout |

## Configuration File

The CLI uses a JSON configuration file that maps to spaCy pipeline components.

### Configuration Structure

```json
{
  "loader": {
    "name": "<loader_type>",
    "params": {}
  },
  "ner": {
    "name": "<ner_type>",
    "params": {}
  },
  "candidate_generator": {
    "name": "<generator_type>",
    "params": {}
  },
  "reranker": {
    "name": "<reranker_type>",
    "params": {}
  },
  "disambiguator": {
    "name": "<disambiguator_type>",
    "params": {}
  },
  "knowledge_base": {
    "name": "<kb_type>",
    "params": {}
  },
  "cache_dir": ".ner_cache",
  "batch_size": 1
}
```

### Config to spaCy Factory Mapping

The configuration names map to spaCy component factories:

| Config Name | spaCy Factory | Description |
|-------------|---------------|-------------|
| **NER** | | |
| `simple` | `el_pipeline_simple` | Regex-based NER |
| `spacy` | Built-in + filter | spaCy's pretrained NER |
| `gliner` | `el_pipeline_gliner` | GLiNER zero-shot |
| **Candidate Generators** | | |
| `fuzzy` | `el_pipeline_fuzzy_candidates` | RapidFuzz matching |
| `bm25` | `el_pipeline_bm25_candidates` | rank-bm25 retrieval |
| `lela_dense` | `el_pipeline_lela_dense_candidates` | Dense retrieval |
| **Rerankers** | | |
| `none` | `el_pipeline_noop_reranker` | No reranking |
| `cross_encoder` | `el_pipeline_lela_cross_encoder_reranker` | Cross-encoder |
| `lela_embedder_transformers` | `el_pipeline_lela_embedder_transformers_reranker` | Bi-encoder (SentenceTransformers) |
| `lela_embedder_vllm` | `el_pipeline_lela_embedder_vllm_reranker` | Bi-encoder (vLLM embed) |
| `lela_cross_encoder_vllm` | `el_pipeline_lela_cross_encoder_vllm_reranker` | Cross-encoder (vLLM score) |
| `lela_vllm_api_client` | `el_pipeline_lela_vllm_api_client_reranker` | vLLM API client reranker |
| `lela_llama_server` | `el_pipeline_lela_llama_server_reranker` | Llama server reranker |
| **Disambiguators** | | |
| `first` | `el_pipeline_first_disambiguator` | Select first |
| `lela_vllm` | `el_pipeline_lela_vllm_disambiguator` | vLLM disambiguation |
| `lela_transformers` | `el_pipeline_lela_transformers_disambiguator` | Transformers LLM disambiguation |
| `lela_openai_api` | `el_pipeline_lela_openai_api_disambiguator` | OpenAI API disambiguation |

### Example Configurations

#### Minimal Configuration (Simple NER + Fuzzy Matching)

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

#### spaCy NER + BM25 + Cross-Encoder

```json
{
  "loader": {"name": "text"},
  "ner": {
    "name": "spacy",
    "params": {"model": "en_core_web_sm"}
  },
  "candidate_generator": {
    "name": "bm25",
    "params": {"top_k": 20}
  },
  "reranker": {
    "name": "cross_encoder",
    "params": {"model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2", "top_k": 10}
  },
  "disambiguator": {"name": "first"},
  "knowledge_base": {"name": "custom", "params": {"path": "kb.jsonl"}}
}
```

#### Full LELA Pipeline

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
    "params": {"top_k": 64, "use_context": true}
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
      "model_name": "Qwen/Qwen3-8B",
      "tensor_parallel_size": 1,
      "add_none_candidate": true,
      "add_descriptions": true
    }
  },
  "knowledge_base": {
    "name": "custom",
    "params": {"path": "entities.jsonl"}
  }
}
```

## Examples

### Process a Single Text File

```bash
python -m el_pipeline.cli \
  --config config/minimal.json \
  --input document.txt \
  --output results.jsonl
```

### Process Multiple Files

```bash
python -m el_pipeline.cli \
  --config config/lela_example.json \
  --input doc1.txt doc2.txt doc3.txt \
  --output all_results.jsonl
```

### Process Different File Types

```bash
# PDF file
python -m el_pipeline.cli \
  --config config.json \
  --input report.pdf \
  --output report_entities.jsonl

# Word document
python -m el_pipeline.cli \
  --config config.json \
  --input document.docx \
  --output doc_entities.jsonl

# HTML file
python -m el_pipeline.cli \
  --config config.json \
  --input webpage.html \
  --output page_entities.jsonl
```

### Process Without Output File (Print to stdout)

```bash
python -m el_pipeline.cli \
  --config config.json \
  --input document.txt
```

### Using with Sample Data

```bash
# Using provided test data
python -m el_pipeline.cli \
  --config data/test/config_simple_fuzzy.json \
  --input data/test/sample_doc.txt \
  --output results.jsonl
```

### Batch Processing with Glob Pattern

```bash
# Process all .txt files in a directory (using shell expansion)
python -m el_pipeline.cli \
  --config config.json \
  --input documents/*.txt \
  --output batch_results.jsonl
```

## Output Format

The CLI outputs results in JSONL (JSON Lines) format, where each line is a valid JSON object representing a processed document.

### Output Structure

```json
{
  "id": "document-id",
  "text": "Full document text...",
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
        }
      ]
    }
  ],
  "meta": {}
}
```

### Reading Output

```bash
# View output file
cat results.jsonl

# Pretty-print with jq
cat results.jsonl | jq '.'

# Count total entities
cat results.jsonl | jq '.entities | length' | paste -sd+ | bc

# Extract only entity texts
cat results.jsonl | jq '.entities[].text'

# Filter entities by label
cat results.jsonl | jq '.entities[] | select(.label == "PERSON")'
```

## Supported File Types

| Extension | Loader | Description |
|-----------|--------|-------------|
| `.txt` | text | Plain text files |
| `.pdf` | pdf | PDF documents (requires pdfplumber) |
| `.docx` | docx | Microsoft Word documents (requires python-docx) |
| `.html`, `.htm` | html | HTML pages (requires beautifulsoup4) |
| `.json` | json | JSON files with text content |
| `.jsonl` | jsonl | JSON Lines files with multiple documents |

## Knowledge Base Format

### Custom JSONL Format

The knowledge base should be a JSONL file with one entity per line:

```jsonl
{"id": "Q937", "title": "Albert Einstein", "description": "German-born theoretical physicist"}
{"id": "Q183", "title": "Germany", "description": "Country in Central Europe"}
{"id": "Q90", "title": "Paris", "description": "Capital city of France"}
```

## Caching

The CLI uses multi-level persistent caching to speed up subsequent runs:

### Cache Types

| Cache | Purpose | Speedup |
|-------|---------|---------|
| Document | Parsed document data | Avoids re-parsing |
| Knowledge Base | Parsed KB entities | ~5-7x for large KBs |
| Dense Index | FAISS embeddings | ~10-15x |
| rank-bm25 Index | BM25Okapi index | ~5x |

### Cache Location

```
.ner_cache/
  <hash>.pkl                    # Document cache
  kb/<hash>.pkl                 # KB entity cache
  index/lela_dense_<hash>/      # Dense (FAISS) index
  index/bm25_<hash>.pkl         # rank-bm25 index
```

### Cache Invalidation

- **Automatic**: Modifying a file changes its mtime, invalidating its cache
- **Index dependency**: Index caches depend on KB hash, so they auto-invalidate when KB changes
- **Manual**: Delete `.ner_cache/` to force a full rebuild

```bash
rm -rf .ner_cache/
```

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | Configuration error |
| 2 | Input file not found |
| 3 | Processing error |

## Troubleshooting

### Common Issues

**1. Missing spaCy model:**
```bash
python -m spacy download en_core_web_sm
```

**2. Knowledge base not found:**
Ensure the path in the configuration is relative to the current working directory or use an absolute path.

**3. Memory issues with large files:**
Reduce `batch_size` in configuration or process files individually.

**4. GPU out of memory:**
For LELA components, reduce `tensor_parallel_size` or use smaller models.

### Verbose Output

To see detailed processing information, set the logging level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
