"""
Memory estimation and system resource detection for the NER pipeline.

Provides utilities to:
- Estimate VRAM/RAM requirements for each component
- Detect available system resources
- Validate pipeline configuration against available memory
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ner_pipeline.lela.config import VLLM_GPU_MEMORY_UTILIZATION

logger = logging.getLogger(__name__)


# ============================================================================
# Model Memory Estimates (in GB)
# Based on empirical measurements and documentation
# ============================================================================

# NER Models
NER_MODEL_MEMORY: Dict[str, float] = {
    # GLiNER models
    "numind/NuNER_Zero-span": 2.0,
    "urchade/gliner_base": 1.5,
    "urchade/gliner_large": 2.0,
    "urchade/gliner_multi": 2.0,
}

# Embedding models (used by dense candidates and rerankers)
EMBEDDING_MODEL_MEMORY: Dict[str, float] = {
    "Qwen/Qwen3-Embedding-0.6B": 2.0,
    "Qwen/Qwen3-Embedding-4B": 10.0,
    "Qwen/Qwen3-Embedding-8B": 18.0,
    "sentence-transformers/all-MiniLM-L6-v2": 0.5,
    "sentence-transformers/all-mpnet-base-v2": 1.0,
    "BAAI/bge-small-en-v1.5": 0.5,
    "BAAI/bge-base-en-v1.5": 1.0,
    "BAAI/bge-large-en-v1.5": 1.5,
    "tomaarsen/Qwen3-Reranker-0.6B-seq-cls": 2.0,
    "tomaarsen/Qwen3-Reranker-4B-seq-cls": 10.0,
}

# Cross-encoder models
CROSS_ENCODER_MODEL_MEMORY: Dict[str, float] = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": 0.5,
    "cross-encoder/ms-marco-TinyBERT-L-2-v2": 0.3,
}

# LLM models for disambiguation (base model VRAM, before vLLM overhead)
LLM_MODEL_MEMORY: Dict[str, float] = {
    "Qwen/Qwen3-0.6B": 3.0,
    "Qwen/Qwen3-1.7B": 5.0,
    "Qwen/Qwen3-4B": 10.0,
    "Qwen/Qwen3-8B": 18.0,
    "Qwen/Qwen3-14B": 32.0,
    "meta-llama/Llama-3.2-1B-Instruct": 4.0,
    "meta-llama/Llama-3.2-3B-Instruct": 8.0,
    "microsoft/Phi-3-mini-4k-instruct": 8.0,
    "microsoft/Phi-3.5-mini-instruct": 8.0,
}

# Available LLM choices for disambiguation (displayed in UI)
AVAILABLE_LLM_MODELS: List[Tuple[str, str, float]] = [
    ("Qwen/Qwen3-0.6B", "Qwen3-0.6B (~3GB)", 3.0),
    ("Qwen/Qwen3-1.7B", "Qwen3-1.7B (~5GB)", 5.0),
    ("Qwen/Qwen3-4B", "Qwen3-4B (~10GB)", 10.0),
    ("Qwen/Qwen3-8B", "Qwen3-8B (~18GB)", 18.0),
]

# Component type to default memory estimate (when model not specified)
COMPONENT_DEFAULT_MEMORY: Dict[str, float] = {
    # NER
    "simple": 0.0,
    "spacy": 0.5,
    "gliner": 2.0,
    "lela_gliner": 2.0,
    # Candidates (CPU-based have 0 GPU memory)
    "fuzzy": 0.0,
    "bm25": 0.0,
    "lela_dense": 10.0,
    # Rerankers
    "none": 0.0,
    "cross_encoder": 0.5,
    "lela_embedder": 10.0,
    # Disambiguators
    "first": 0.0,
    "popularity": 0.0,
    "lela_vllm": 10.0,
    "lela_transformers": 10.0,
}


@dataclass
class MemoryEstimate:
    """Memory estimate for a pipeline configuration."""

    component_estimates: Dict[str, float]  # component name -> GB
    total_vram_gb: float
    total_ram_gb: float
    shared_models: List[str]  # Models that are shared between components
    warnings: List[str]


@dataclass
class SystemResources:
    """Available system resources."""

    gpu_available: bool
    gpu_name: Optional[str]
    gpu_vram_total_gb: float
    gpu_vram_free_gb: float
    ram_total_gb: float
    ram_available_gb: float


def get_system_resources() -> SystemResources:
    """Detect available system resources (GPU VRAM and system RAM)."""
    import psutil

    # Get RAM info
    ram = psutil.virtual_memory()
    ram_total_gb = ram.total / (1024**3)
    ram_available_gb = ram.available / (1024**3)

    # Get GPU info
    gpu_available = False
    gpu_name = None
    gpu_vram_total_gb = 0.0
    gpu_vram_free_gb = 0.0

    try:
        import torch

        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_vram_total_gb = props.total_memory / (1024**3)

            # Get free memory
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            gpu_vram_free_gb = free_mem / (1024**3)
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Error detecting GPU: {e}")

    return SystemResources(
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_vram_total_gb=gpu_vram_total_gb,
        gpu_vram_free_gb=gpu_vram_free_gb,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
    )


def estimate_model_memory(model_name: str, model_type: str = "auto") -> float:
    """
    Estimate VRAM needed for a model.

    Args:
        model_name: HuggingFace model ID
        model_type: One of 'ner', 'embedding', 'cross_encoder', 'llm', or 'auto'

    Returns:
        Estimated VRAM in GB
    """
    # Try to find in known models
    if model_type == "auto" or model_type == "ner":
        if model_name in NER_MODEL_MEMORY:
            return NER_MODEL_MEMORY[model_name]

    if model_type == "auto" or model_type == "embedding":
        if model_name in EMBEDDING_MODEL_MEMORY:
            return EMBEDDING_MODEL_MEMORY[model_name]

    if model_type == "auto" or model_type == "cross_encoder":
        if model_name in CROSS_ENCODER_MODEL_MEMORY:
            return CROSS_ENCODER_MODEL_MEMORY[model_name]

    if model_type == "auto" or model_type == "llm":
        if model_name in LLM_MODEL_MEMORY:
            return LLM_MODEL_MEMORY[model_name]

    # Estimate based on model name patterns
    model_lower = model_name.lower()

    # LLM size estimation from name
    if "0.5b" in model_lower or "0.6b" in model_lower:
        return 3.0
    elif "1b" in model_lower or "1.5b" in model_lower or "1.7b" in model_lower:
        return 5.0
    elif "3b" in model_lower:
        return 8.0
    elif "4b" in model_lower:
        return 10.0
    elif "7b" in model_lower or "8b" in model_lower:
        return 18.0
    elif "13b" in model_lower or "14b" in model_lower:
        return 32.0
    elif "70b" in model_lower:
        return 140.0

    # Default estimates by type
    if model_type == "llm":
        return 10.0
    elif model_type == "embedding":
        return 2.0
    elif model_type == "ner":
        return 2.0

    # Unknown - assume moderate size
    return 2.0


def estimate_component_memory(
    component_type: str,
    component_name: str,
    params: Optional[Dict] = None,
) -> Tuple[float, Optional[str]]:
    """
    Estimate VRAM for a single component.

    Args:
        component_type: 'ner', 'candidate_generator', 'reranker', 'disambiguator'
        component_name: Component name (e.g., 'lela_gliner', 'lela_vllm')
        params: Component parameters (may include model_name)

    Returns:
        Tuple of (VRAM in GB, model name if applicable)
    """
    params = params or {}

    # CPU-only components
    if component_name in ("simple", "fuzzy", "bm25", "none", "first", "popularity"):
        return 0.0, None

    # SpaCy NER
    if component_name == "spacy":
        model = params.get("model", "en_core_web_sm")
        if "lg" in model:
            return 0.8, model
        elif "md" in model:
            return 0.4, model
        return 0.2, model

    # GLiNER NER
    if component_name in ("gliner", "lela_gliner"):
        model = params.get("model_name", "numind/NuNER_Zero-span")
        return estimate_model_memory(model, "ner"), model

    # Dense candidates
    if component_name == "lela_dense":
        model = params.get("model_name", "Qwen/Qwen3-Embedding-4B")
        return estimate_model_memory(model, "embedding"), model

    # Cross-encoder reranker
    if component_name == "cross_encoder":
        model = params.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        return estimate_model_memory(model, "cross_encoder"), model

    # Embedder reranker
    if component_name == "lela_embedder":
        model = params.get("model_name", "Qwen/Qwen3-Embedding-4B")
        return estimate_model_memory(model, "embedding"), model

    # vLLM disambiguators
    if component_name in ("lela_vllm", "lela_transformers"):
        model = params.get("model_name", "Qwen/Qwen3-4B")
        return estimate_model_memory(model, "llm"), model

    # Unknown - use default
    return COMPONENT_DEFAULT_MEMORY.get(component_name, 2.0), None


def estimate_pipeline_memory(config_dict: Dict) -> MemoryEstimate:
    """
    Estimate total memory requirements for a pipeline configuration.

    Args:
        config_dict: Pipeline configuration dictionary

    Returns:
        MemoryEstimate with breakdown by component
    """
    component_estimates: Dict[str, float] = {}
    models_used: Dict[str, float] = {}  # model -> memory
    warnings: List[str] = []

    # NER component
    ner_config = config_dict.get("ner", {})
    ner_name = ner_config.get("name", "simple")
    ner_params = ner_config.get("params", {})
    ner_mem, ner_model = estimate_component_memory("ner", ner_name, ner_params)
    component_estimates["ner"] = ner_mem
    if ner_model:
        models_used[ner_model] = ner_mem

    # Candidate generator
    cand_config = config_dict.get("candidate_generator", {})
    cand_name = cand_config.get("name", "fuzzy")
    cand_params = cand_config.get("params", {})
    cand_mem, cand_model = estimate_component_memory(
        "candidate_generator", cand_name, cand_params
    )
    component_estimates["candidates"] = cand_mem
    if cand_model:
        # Check if model is shared with another component
        if cand_model in models_used:
            warnings.append(f"Model '{cand_model}' is shared - memory counted once")
        else:
            models_used[cand_model] = cand_mem

    # Reranker
    reranker_config = config_dict.get("reranker", {})
    reranker_name = reranker_config.get("name", "none")
    reranker_params = reranker_config.get("params", {})
    reranker_mem, reranker_model = estimate_component_memory(
        "reranker", reranker_name, reranker_params
    )
    component_estimates["reranker"] = reranker_mem
    if reranker_model:
        if reranker_model in models_used:
            warnings.append(f"Model '{reranker_model}' is shared - memory counted once")
        else:
            models_used[reranker_model] = reranker_mem

    # Disambiguator
    disambig_config = config_dict.get("disambiguator", {})
    if disambig_config:
        disambig_name = disambig_config.get("name", "first")
        disambig_params = disambig_config.get("params", {})
        disambig_mem, disambig_model = estimate_component_memory(
            "disambiguator", disambig_name, disambig_params
        )
        component_estimates["disambiguator"] = disambig_mem
        if disambig_model:
            if disambig_model in models_used:
                warnings.append(
                    f"Model '{disambig_model}' is shared - memory counted once"
                )
            else:
                models_used[disambig_model] = disambig_mem
    else:
        component_estimates["disambiguator"] = 0.0

    # Total is sum of unique models (not double-counting shared models)
    total_vram_gb = sum(models_used.values())

    # Estimate some RAM overhead (KB loading, spaCy, etc.)
    total_ram_gb = 2.0  # Base overhead
    kb_config = config_dict.get("knowledge_base", {})
    if kb_config:
        # Large KBs need more RAM
        total_ram_gb += 1.0

    return MemoryEstimate(
        component_estimates=component_estimates,
        total_vram_gb=total_vram_gb,
        total_ram_gb=total_ram_gb,
        shared_models=list(models_used.keys()),
        warnings=warnings,
    )


def check_memory_requirements(
    config_dict: Dict,
    resources: Optional[SystemResources] = None,
) -> Tuple[bool, str, MemoryEstimate]:
    """
    Check if system has enough memory to run the pipeline.

    Takes into account that vLLM uses gpu_memory_utilization=0.9, meaning it will
    try to claim 90% of GPU memory. Other models must fit in the remaining 10%
    or be loaded before vLLM.

    Args:
        config_dict: Pipeline configuration dictionary
        resources: System resources (auto-detected if not provided)

    Returns:
        Tuple of (can_run, message, estimate)
    """
    if resources is None:
        resources = get_system_resources()

    estimate = estimate_pipeline_memory(config_dict)

    messages = []
    can_run = True

    # Check GPU memory with vLLM consideration
    if estimate.total_vram_gb > 0:
        if not resources.gpu_available:
            can_run = False
            messages.append(
                f"⚠️ GPU required but not available. "
                f"Pipeline needs ~{estimate.total_vram_gb:.1f}GB VRAM."
            )
        else:
            # Calculate effective available memory considering vLLM's 90% utilization
            # vLLM will try to use 90% of total VRAM, leaving 10% for other models
            # But we load other models first, so they reduce what vLLM can claim

            # Check if LLM disambiguator is used
            disambig_config = config_dict.get("disambiguator", {})
            disambig_name = disambig_config.get("name", "") if disambig_config else ""
            uses_vllm = disambig_name in ("lela_vllm", "lela_transformers")

            if uses_vllm:
                # vLLM claims 90% of total VRAM for itself
                vllm_allocation = (
                    resources.gpu_vram_total_gb * VLLM_GPU_MEMORY_UTILIZATION
                )
                non_llm_memory = (
                    estimate.total_vram_gb
                    - estimate.component_estimates.get("disambiguator", 0)
                )
                llm_memory = estimate.component_estimates.get("disambiguator", 0)

                # Non-LLM models must fit alongside vLLM's allocation
                available_for_others = resources.gpu_vram_total_gb - vllm_allocation

                if non_llm_memory > available_for_others + 1.0:  # 1GB tolerance
                    can_run = False
                    messages.append(
                        f"⚠️ Non-LLM models (~{non_llm_memory:.1f}GB) may not fit. "
                        f"vLLM reserves {VLLM_GPU_MEMORY_UTILIZATION*100:.0f}% ({vllm_allocation:.1f}GB), "
                        f"leaving ~{available_for_others:.1f}GB for other models."
                    )
                elif llm_memory > vllm_allocation:
                    can_run = False
                    messages.append(
                        f"⚠️ LLM model (~{llm_memory:.1f}GB) exceeds vLLM allocation "
                        f"({vllm_allocation:.1f}GB at {VLLM_GPU_MEMORY_UTILIZATION*100:.0f}% utilization)."
                    )
                elif estimate.total_vram_gb > resources.gpu_vram_free_gb * 0.95:
                    messages.append(
                        f"⚡ Memory is tight. "
                        f"Need ~{estimate.total_vram_gb:.1f}GB, "
                        f"available ~{resources.gpu_vram_free_gb:.1f}GB."
                    )
                else:
                    messages.append(
                        f"✅ GPU memory OK. "
                        f"Need ~{estimate.total_vram_gb:.1f}GB, "
                        f"available ~{resources.gpu_vram_free_gb:.1f}GB."
                    )
            else:
                # No vLLM - simple check
                if estimate.total_vram_gb > resources.gpu_vram_free_gb:
                    can_run = False
                    messages.append(
                        f"⚠️ Insufficient GPU memory. "
                        f"Need ~{estimate.total_vram_gb:.1f}GB, "
                        f"available ~{resources.gpu_vram_free_gb:.1f}GB."
                    )
                elif estimate.total_vram_gb > resources.gpu_vram_free_gb * 0.9:
                    messages.append(
                        f"⚡ Memory is tight. "
                        f"Need ~{estimate.total_vram_gb:.1f}GB, "
                        f"available ~{resources.gpu_vram_free_gb:.1f}GB."
                    )
                else:
                    messages.append(
                        f"✅ GPU memory OK. "
                        f"Need ~{estimate.total_vram_gb:.1f}GB, "
                        f"available ~{resources.gpu_vram_free_gb:.1f}GB."
                    )

    # Check RAM
    if estimate.total_ram_gb > resources.ram_available_gb:
        messages.append(
            f"⚠️ Low system RAM. "
            f"Need ~{estimate.total_ram_gb:.1f}GB, "
            f"available ~{resources.ram_available_gb:.1f}GB."
        )

    # Add GPU info
    if resources.gpu_available:
        messages.insert(0, f"🖥️ GPU: {resources.gpu_name}")

    return can_run, "\n".join(messages), estimate


def format_memory_estimate_for_ui(
    estimate: MemoryEstimate,
    resources: Optional[SystemResources] = None,
) -> str:
    """Format memory estimate as a human-readable string for UI display."""
    lines = ["**Memory Estimate**", ""]

    # Component breakdown
    for comp, mem in estimate.component_estimates.items():
        if mem > 0:
            lines.append(f"- {comp.title()}: ~{mem:.1f}GB VRAM")
        else:
            lines.append(f"- {comp.title()}: CPU only")

    lines.append("")
    lines.append(f"**Total VRAM:** ~{estimate.total_vram_gb:.1f}GB")

    # Add comparison to available
    if resources:
        if resources.gpu_available:
            used_pct = (estimate.total_vram_gb / resources.gpu_vram_total_gb) * 100
            lines.append(
                f"**Available:** ~{resources.gpu_vram_free_gb:.1f}GB / {resources.gpu_vram_total_gb:.1f}GB"
            )
            if used_pct > 90:
                lines.append("⚠️ **May exceed available memory!**")
            elif used_pct > 70:
                lines.append("⚡ Memory usage will be high")
        else:
            if estimate.total_vram_gb > 0:
                lines.append("⚠️ **GPU not available!**")

    if estimate.warnings:
        lines.append("")
        lines.append("**Notes:**")
        for w in estimate.warnings:
            lines.append(f"- {w}")

    return "\n".join(lines)
