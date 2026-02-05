# Modular NER Pipeline

Standalone, swappable NER → candidate generation → rerank → disambiguation pipeline. Uses file-based storage (JSONL for KB and outputs) and optional caching in `.ner_cache/`.

## Install

**Requirements:** Python 3.10-3.12 (Python 3.13 is NOT supported due to vLLM), CUDA 11.8 for GPU support

```bash
cd ner-pipeline
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install PyTorch with CUDA 11.8 (required for P100/older GPUs)
pip install torch==2.6.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

## Quick start

### Web UI (Gradio)
Launch the interactive web interface:
```bash
python app.py
```
Open `http://localhost:7860` and configure the pipeline through the UI. See [docs/WEB_APP.md](docs/WEB_APP.md) for details.

### Troubleshooting

If you encounter issues, see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for solutions to common problems including:
- PyTorch CUDA mismatch
- vLLM installation failures
- GPU memory issues
- P100/Pascal GPU compatibility

### CLI
1) Prepare a JSONL knowledge base with fields: `id`, `title`, `description` (plus optional metadata).
2) Create a config file, e.g. `config.json`:
```json
{
  "loader": {"name": "pdf", "params": {}},
  "ner": {"name": "spacy", "params": {"model": "en_core_web_sm"}},
  "candidate_generator": {"name": "bm25", "params": {}},
  "reranker": {"name": "none", "params": {}},
  "disambiguator": {"name": "popularity", "params": {}},
  "knowledge_base": {"name": "custom", "params": {"path": "kb.jsonl"}},
  "cache_dir": ".ner_cache",
  "batch_size": 1
}
```
3) Run:
```bash
python -m ner_pipeline.cli --config config.json --input docs/file1.pdf docs/file2.pdf --output outputs.jsonl
```

### Example: lightweight fuzzy run (no heavy models)
```bash
python -m ner_pipeline.cli \
  --config data/configs/simplewiki_fuzzy_simple.json \
  --input data/docs/simple-english-wiki/corpus.txt \
  --output outputs.jsonl
```
This uses the `simple` regex NER, fuzzy candidates, first-candidate disambiguation, and the YAGO-derived KB JSONL.

## Python API
```python
from ner_pipeline.config import PipelineConfig
from ner_pipeline.pipeline import NERPipeline
import json

config = PipelineConfig.from_dict(json.load(open("config.json")))
pipeline = NERPipeline(config)
results = pipeline.run(["docs/file1.txt"])
```

## Available components
- Loaders: `text`, `json`, `jsonl`, `pdf`, `docx`, `html`
- NER: `spacy`, `gliner`, `simple` (regex)
- Candidate generators: `bm25`, `dense`, `fuzzy`
- Rerankers: `cross_encoder`, `none`
- Disambiguators: `popularity`, `first`, `llm`
- Knowledge bases: `custom`, `wikipedia`, `wikidata`

## Data & configs
- The `data/` directory is gitignored by default. Keep shareable configs in `data/configs/` (tracked).
- Sample configs provided:
  - `data/configs/simplewiki_fuzzy_simple.json`

## Conversion utilities
- YAGO labels TSV → JSONL KB:
  ```bash
  python -m ner_pipeline.scripts.convert_yago_labels data/kb/yagoLabels.tsv data/kb/yago_labels_en.jsonl
  ```

## Notes
- Outputs are JSONL (one line per document with resolved entities).
  - Each line: `id`, `text`, `entities` (with `text`, `start`, `end`, `label`, `entity_id`, `entity_title`, `entity_description`, `candidates`).
- Cache lives in `.ner_cache/` keyed by file path, mtime, and size.
- No dependency on LELA; integration would be optional if added later. 
