# Troubleshooting Guide

This guide provides solutions for common issues encountered when installing and running the EL Pipeline.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Runtime Errors](#runtime-errors)
- [GPU and Memory Issues](#gpu-and-memory-issues)
- [Debugging Guide](#debugging-guide)

---

## Installation Issues

### PyTorch CUDA Mismatch

**Symptom:** CUDA errors when loading models, or PyTorch fails to detect GPU.

**Solution:**

1. Check your CUDA version:
   ```bash
   nvidia-smi
   nvcc --version
   ```

2. Install the correct PyTorch version for your CUDA:

   For **CUDA 11.8** (P100/Pascal GPUs):
   ```bash
   pip install torch==2.7.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

   For **CUDA 12.x** (newer GPUs):
   ```bash
   pip install torch torchvision torchaudio
   ```

3. Verify installation:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.version.cuda)          # Should match your CUDA version
   ```

---

### vLLM Installation Failures

**Symptom:** vLLM fails to install or import.

**Causes and Solutions:**

1. **Python version incompatibility:**
   - vLLM requires Python 3.10-3.12 (Python 3.13 is NOT supported)
   - Check your version: `python --version`
   - Create a new environment with Python 3.10-3.12

2. **CUDA/PyTorch mismatch:**
   ```bash
   # Ensure PyTorch is installed first with correct CUDA
   pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
   pip install vllm==0.10.1
   ```

3. **Build dependencies missing:**
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install build-essential python3-dev
   ```

---

### Missing spaCy Models

**Symptom:** `OSError: [E050] Can't find model 'en_core_web_sm'`

**Solution:**
```bash
python -m spacy download en_core_web_sm

# For other models:
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
```

---

### GLiNER Import Errors

**Symptom:** `ImportError: gliner package required for GLiNER NER`

**Solution:**
```bash
pip install gliner==0.2.24
```

---

## Runtime Errors

### vLLM V1 Multiprocessing Failure

**Symptom:** Errors like `RuntimeError: Cannot re-initialize CUDA in forked subprocess` or multiprocessing errors when running the web app or using vLLM from worker threads.

**Cause:** vLLM's V1 engine uses multiprocessing that conflicts with Gradio's threading model.

**Solution:**

The web app (`app.py`) automatically sets this environment variable, but if you're using vLLM in your own code:

```python
import os
# Set BEFORE importing vLLM
os.environ["VLLM_USE_V1"] = "0"

# Now import vLLM
from vllm import LLM
```

Or set it in your shell before running:
```bash
export VLLM_USE_V1=0
python your_script.py
```

---

### GLiNER Context Limit Exceeded

**Symptom:** Errors or truncated results with long documents using GLiNER.

**Cause:** GLiNER has a token/character limit for input text.

**Solution:**

The pipeline automatically chunks long documents. The default settings are:
- Chunk size: ~1500 characters
- Overlap: 200 characters
- Sentence boundary detection to avoid splitting mid-sentence

The chunking behavior is handled in `el_pipeline/spacy_components/ner.py` (lines 118-162). If you need to adjust:

```python
# In your custom code, you can configure chunk parameters
nlp.add_pipe("el_pipeline_lela_gliner", config={
    "threshold": 0.5,
    "labels": ["person", "organization", "location"]
})
# Note: Chunk size is currently fixed in the component
```

---

### Knowledge Base Not Initialized

**Symptom:** `UserWarning: Disambiguator not initialized - call initialize(kb) first`

**Cause:** Candidate generators and disambiguators need a knowledge base reference.

**Solution:**

When building pipelines manually with spaCy:
```python
import spacy
from el_pipeline import spacy_components
from el_pipeline.knowledge_bases.custom import CustomJSONLKnowledgeBase

nlp = spacy.blank("en")
nlp.add_pipe("el_pipeline_simple")
cand = nlp.add_pipe("el_pipeline_fuzzy_candidates")
disamb = nlp.add_pipe("el_pipeline_first_disambiguator")

# Initialize with knowledge base
kb = CustomJSONLKnowledgeBase(path="kb.jsonl")
cand.initialize(kb)
disamb.initialize(kb)
```

When using `ELPipeline`, initialization is automatic.

---

### Empty Results / No Entities Found

**Possible causes:**

1. **NER threshold too high:**
   ```json
   {
     "ner": {
       "name": "gliner",
       "params": {"threshold": 0.3}  // Try lowering from 0.5
     }
   }
   ```

2. **Text doesn't contain recognizable entities:**
   - Simple NER requires capitalized words
   - Try different NER models for different text types

3. **Knowledge base doesn't contain relevant entities:**
   - Check your KB has entities matching the text
   - Use fuzzy matching for more lenient candidate generation

---

## GPU and Memory Issues

### P100/Pascal GPU Compatibility

**Symptom:** CUDA errors mentioning compute capability, or vLLM fails on P100 GPUs.

**Cause:** Newer versions of vLLM drop support for compute capability 6.0 (Pascal architecture).

**Solution:**

Use specific versions that support P100:
```bash
# CUDA 11.8 + PyTorch 2.7.1 + vLLM 0.10.1
pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install vllm==0.10.1
```

Alternatively, use the `lela_transformers` disambiguator which uses HuggingFace transformers directly:
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

---

### Out of Memory (OOM) Errors

**Symptom:** `CUDA out of memory` or process killed.

**Solutions:**

1. **Use smaller models:**
   ```json
   {
     "disambiguator": {
       "name": "lela_vllm",
       "params": {
         "model_name": "Qwen/Qwen3-4B"  // Instead of 8B
       }
     }
   }
   ```

2. **Reduce candidate counts:**
   ```json
   {
     "candidate_generator": {
       "name": "bm25",
       "params": {"top_k": 32}  // Instead of 64
     },
     "reranker": {
       "name": "lela_embedder_transformers",
       "params": {"top_k": 5}  // Instead of 10
     }
   }
   ```

3. **Enable tensor parallelism** (multi-GPU):
   ```json
   {
     "disambiguator": {
       "name": "lela_vllm",
       "params": {
         "tensor_parallel_size": 2
       }
     }
   }
   ```

4. **Process documents in smaller batches:**
   ```json
   {
     "batch_size": 1
   }
   ```

---

### CPU Fallback

**Symptom:** GPU not being used, slow processing.

**Check GPU availability:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

**Solutions:**

1. Ensure CUDA is installed and visible:
   ```bash
   nvidia-smi  # Should show your GPU
   ```

2. Reinstall PyTorch with CUDA support:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
   ```

3. For components without GPU, they will run on CPU (this is expected for BM25, fuzzy matching, etc.)

---

## Debugging Guide

### Enable Verbose Logging

**Python API:**
```python
import logging

# Enable debug logging for the pipeline
logging.basicConfig(level=logging.DEBUG)

# Or for specific modules:
logging.getLogger("el_pipeline").setLevel(logging.DEBUG)
logging.getLogger("el_pipeline.spacy_components").setLevel(logging.DEBUG)
```

**CLI:**
```bash
python -m el_pipeline.cli --config config.json --input doc.txt --output out.jsonl 2>&1 | tee debug.log
```

**Web App:**
```bash
python app.py --log DEBUG
```

---

### Check Component Loading

```python
import spacy
from el_pipeline import spacy_components

nlp = spacy.blank("en")
# List available factories
print(nlp.factory_names)  # Should include el_pipeline_* factories
```

---

### Inspect Pipeline State

```python
# After processing a document
doc = nlp("Albert Einstein was born in Germany.")

for ent in doc.ents:
    print(f"Entity: {ent.text}")
    print(f"  Label: {ent.label_}")
    print(f"  Context: {ent._.context}")
    print(f"  Candidates: {len(ent._.candidates)}")
    print(f"  Resolved: {ent._.resolved_entity}")
```

---

### Common Log Messages

| Message | Meaning | Action |
|---------|---------|--------|
| `Loading LELA GLiNER model: ...` | GLiNER model being loaded | Wait for download on first run |
| `vLLM check: installed=True, cuda=True` | vLLM is ready | Normal |
| `vLLM check: installed=True, cuda=False` | No GPU for vLLM | Check CUDA installation |
| `Disambiguator not initialized` | Missing KB | Call `component.initialize(kb)` |
| `Could not parse answer from output` | LLM response parsing failed | Check LLM output format |

---

## Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/NDobricic/el-pipeline/issues)
2. Enable debug logging and examine the output
3. Verify your environment matches the [Requirements](REQUIREMENTS.md)
4. Try a minimal configuration to isolate the issue
