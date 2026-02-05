# System Requirements

This document details the hardware, software, and environment requirements for running the NER Pipeline.

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [GPU Compatibility Matrix](#gpu-compatibility-matrix)
- [Python Dependencies](#python-dependencies)
- [Model Storage](#model-storage)
- [Offline Setup](#offline-setup)

---

## Hardware Requirements

### Minimum Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **RAM** | 8 GB | 16 GB+ |
| **GPU VRAM** | 4 GB (basic NER) | 16 GB+ (full LELA) |
| **Disk Space** | 5 GB | 50 GB+ (with models) |
| **CPU** | 4 cores | 8+ cores |

### Component-Specific Requirements

| Component | CPU-Only | GPU Required | VRAM Needed |
|-----------|----------|--------------|-------------|
| Simple NER (regex) | Yes | No | - |
| GLiNER NER | Yes (slow) | Recommended | ~2 GB |
| BM25 Candidates | Yes | No | - |
| Fuzzy Candidates | Yes | No | - |
| Dense Candidates | Yes (slow) | Recommended | ~8-10 GB |
| Cross-Encoder Reranker | Yes (slow) | Recommended | ~1 GB |
| LELA Embedder Reranker | No | Yes | ~8-10 GB |
| LELA vLLM Disambiguator | No | **Required** | 10-20 GB |
| LELA Transformers Disambiguator | No | **Required** | 10-20 GB |

### Full LELA Pipeline

Running the complete LELA pipeline with vLLM disambiguation requires:
- **GPU with 16+ GB VRAM** (e.g., V100, A100)
- **32+ GB system RAM** recommended for large knowledge bases
- **CUDA 11.8+** for GPU acceleration

---

## Software Requirements

### Operating System

- **Linux**: Ubuntu 20.04+, Debian 11+ (recommended)
- **macOS**: 12+ (CPU-only, no vLLM support)
- **Windows**: WSL2 with Ubuntu (native Windows not fully supported)

### Python Version

| Version | Status | Notes |
|---------|--------|-------|
| Python 3.10 | **Recommended** | Best compatibility |
| Python 3.11 | Supported | Full functionality |
| Python 3.12 | Supported | Full functionality |
| Python 3.13 | **NOT Supported** | vLLM incompatible |

**Important:** Python 3.13 is not supported due to vLLM compatibility issues.

### CUDA Requirements

| GPU Generation | CUDA Version | PyTorch Version | vLLM Version |
|----------------|--------------|-----------------|--------------|
| P100/Pascal (CC 6.0) | CUDA 11.8 | 2.6.0+cu118 | 0.8.5 |
| V100/Volta (CC 7.0) | CUDA 11.8+ | 2.6.0+ | 0.8.5+ |
| A100/Ampere (CC 8.0) | CUDA 12.x | Latest | Latest |
| H100/Hopper (CC 9.0) | CUDA 12.x | Latest | Latest |

---

## GPU Compatibility Matrix

### NVIDIA GPU Support

| GPU | Architecture | Compute Capability | PyTorch | vLLM | Notes |
|-----|--------------|-------------------|---------|------|-------|
| P100 | Pascal | 6.0 | 2.6.0+cu118 | 0.8.5 | Use CUDA 11.8 only |
| V100 | Volta | 7.0 | 2.6.0+ | 0.8.5+ | Recommended for LELA |
| A100 | Ampere | 8.0 | Latest | Latest | Optimal performance |
| H100 | Hopper | 9.0 | Latest | Latest | Best performance |
| RTX 3090 | Ampere | 8.6 | Latest | Latest | Consumer GPU option |
| RTX 4090 | Ada | 8.9 | Latest | Latest | Consumer GPU option |

### P100/Pascal GPU Setup

P100 GPUs (compute capability 6.0) require specific versions:

```bash
# Install PyTorch with CUDA 11.8
pip install torch==2.6.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install vLLM 0.8.5 (last version with P100 support)
pip install vllm==0.8.5
```

**Alternative for P100:** Use `lela_transformers` disambiguator instead of `lela_vllm`:
```json
{
  "disambiguator": {
    "name": "lela_transformers",
    "params": {"model_name": "Qwen/Qwen3-4B"}
  }
}
```

### Newer GPU Setup (A100, H100, RTX 40xx)

```bash
# Install latest PyTorch with CUDA 12
pip install torch torchvision torchaudio

# Install latest vLLM
pip install vllm
```

---

## Python Dependencies

### Core Dependencies

From `requirements.txt`:

```
spacy==3.8.11
gliner==0.2.24
transformers==4.57.6
torch==2.6.0+cu118
sentence-transformers==5.2.0
faiss-cpu==1.13.2
rank-bm25==0.2.2
rapidfuzz==3.14.3
pdfplumber==0.11.9
python-docx==1.2.0
beautifulsoup4==4.14.3
lxml==6.0.2
tqdm==4.67.1
gradio==6.3.0

# LELA core
bm25s==0.2.14
PyStemmer==3.0.0

# LELA vLLM (requires GPU)
vllm==0.8.5
openai==2.15.0
```

### Optional Dependencies

| Package | Required For |
|---------|--------------|
| `spacy[en_core_web_sm]` | spaCy's built-in NER |
| `faiss-gpu` | GPU-accelerated FAISS (instead of faiss-cpu) |

### Installing spaCy Models

```bash
# Small model (recommended for most cases)
python -m spacy download en_core_web_sm

# Medium model (better accuracy)
python -m spacy download en_core_web_md

# Large model (best accuracy)
python -m spacy download en_core_web_lg
```

---

## Model Storage

### Default Storage Locations

| Model Type | Location | Size Range |
|------------|----------|------------|
| HuggingFace models | `~/.cache/huggingface/hub/` | 100 MB - 20 GB each |
| spaCy models | Python site-packages | 15 MB - 800 MB each |
| Pipeline cache | `.ner_cache/` (configurable) | Varies |
| GLiNER models | `~/.cache/huggingface/hub/` | ~350 MB |

### Model Sizes

| Model | Download Size | VRAM Required |
|-------|---------------|---------------|
| NuNER_Zero-span (GLiNER) | ~350 MB | ~2 GB |
| dslim/bert-base-NER | ~440 MB | ~1 GB |
| Qwen3-Embedding-4B | ~8 GB | ~10 GB |
| Qwen3-4B | ~8 GB | ~10 GB |
| Qwen3-8B | ~16 GB | ~20 GB |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | ~90 MB | ~500 MB |

### Clearing Model Cache

```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/hub/

# Clear pipeline cache
rm -rf .ner_cache/
```

---

## Offline Setup

For air-gapped environments, pre-download all required models.

### Step 1: Download Models on Connected Machine

```python
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Download NER models
AutoModel.from_pretrained("numind/NuNER_Zero-span")
AutoTokenizer.from_pretrained("numind/NuNER_Zero-span")

AutoModel.from_pretrained("dslim/bert-base-NER")
AutoTokenizer.from_pretrained("dslim/bert-base-NER")

# Download embedder models
SentenceTransformer("all-MiniLM-L6-v2")

# Download LLM for disambiguation
AutoModel.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
```

### Step 2: Transfer Cache Directory

```bash
# Package the cache
tar -czvf huggingface_cache.tar.gz ~/.cache/huggingface/

# Transfer to offline machine
scp huggingface_cache.tar.gz user@offline-machine:~/

# Extract on offline machine
cd ~
tar -xzvf huggingface_cache.tar.gz
```

### Step 3: Download spaCy Models

```bash
# Download model package
python -m spacy download en_core_web_sm --no-cache

# Or download the wheel file manually from:
# https://github.com/explosion/spacy-models/releases

# Install offline
pip install en_core_web_sm-3.8.0-py3-none-any.whl
```

### Step 4: Configure Offline Mode

Set environment variables to prevent online lookups:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

---

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `VLLM_USE_V1` | vLLM engine version (`0` for V0) | `1` |
| `HF_HOME` | HuggingFace cache directory | `~/.cache/huggingface` |
| `TRANSFORMERS_CACHE` | Transformers cache (deprecated) | `~/.cache/huggingface` |
| `CUDA_VISIBLE_DEVICES` | GPU selection | All GPUs |

### Example: Multi-GPU Setup

```bash
# Use only GPUs 0 and 1
export CUDA_VISIBLE_DEVICES=0,1

# Configure vLLM tensor parallelism
python -c "
from ner_pipeline.config import PipelineConfig
config = PipelineConfig.from_dict({
    'disambiguator': {
        'name': 'lela_vllm',
        'params': {'tensor_parallel_size': 2}
    },
    # ... rest of config
})
"
```

---

## Verification Script

Run this script to verify your environment is correctly configured:

```python
#!/usr/bin/env python3
"""Verify NER Pipeline environment setup."""

import sys

def check_python():
    v = sys.version_info
    print(f"Python: {v.major}.{v.minor}.{v.micro}")
    if v.major == 3 and 10 <= v.minor <= 12:
        print("  [OK] Python version supported")
    else:
        print("  [WARN] Python 3.10-3.12 recommended")

def check_torch():
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        print("PyTorch: NOT INSTALLED")

def check_vllm():
    try:
        import vllm
        print(f"vLLM: {vllm.__version__}")
    except ImportError:
        print("vLLM: NOT INSTALLED (optional, needed for lela_vllm)")

def check_spacy():
    try:
        import spacy
        print(f"spaCy: {spacy.__version__}")
        try:
            spacy.load("en_core_web_sm")
            print("  [OK] en_core_web_sm model installed")
        except OSError:
            print("  [WARN] en_core_web_sm not installed")
    except ImportError:
        print("spaCy: NOT INSTALLED")

def check_gliner():
    try:
        import gliner
        print(f"GLiNER: installed")
    except ImportError:
        print("GLiNER: NOT INSTALLED")

if __name__ == "__main__":
    print("=== NER Pipeline Environment Check ===\n")
    check_python()
    check_torch()
    check_vllm()
    check_spacy()
    check_gliner()
    print("\n=== Check Complete ===")
```

Save as `check_environment.py` and run:
```bash
python check_environment.py
```
