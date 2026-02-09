"""
Singleton pool for managing expensive LLM and embedder instances.

This module provides lazy initialization and reuse of:
- SentenceTransformer instances for embeddings
- vLLM instances for text generation

Memory Management:
- Models are cached for reuse between pipeline stages
- Models can be marked as "not in use" after a stage completes
- When loading a new model and memory is tight, unused models are evicted
- This allows large LLMs to be loaded after NER/embedding stages complete
"""

import logging
import os
from typing import Dict, Set, Optional, Any

from el_pipeline.lela.config import VLLM_GPU_MEMORY_UTILIZATION

# Disable vLLM V1 engine and configure multiprocessing to work from worker threads
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

logger = logging.getLogger(__name__)


# ============================================================================
# Memory Management
# ============================================================================

def _get_free_vram_gb() -> float:
    """Get free GPU VRAM in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            free_mem, _ = torch.cuda.mem_get_info(0)
            return free_mem / (1024**3)
    except Exception:
        pass
    return 0.0


def _clear_cuda_cache():
    """Clear CUDA cache and run garbage collection."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        logger.warning(f"Error clearing CUDA cache: {e}")


# ============================================================================
# SentenceTransformer Pool with Lazy Eviction
# ============================================================================

_sentence_transformers_module = None
_sentence_transformer_instances: Dict[str, Any] = {}
_sentence_transformer_in_use: Set[str] = set()  # Keys currently in use


def _get_sentence_transformers():
    """Lazy import of sentence_transformers module."""
    global _sentence_transformers_module
    if _sentence_transformers_module is None:
        try:
            import sentence_transformers
            _sentence_transformers_module = sentence_transformers
        except ImportError:
            raise ImportError(
                "sentence-transformers package required for embedding. "
                "Install with: pip install sentence-transformers"
            )
    return _sentence_transformers_module


def _evict_unused_sentence_transformers(required_vram_gb: float = 0):
    """
    Evict unused SentenceTransformer instances to free memory.
    
    Args:
        required_vram_gb: Minimum VRAM needed (will keep evicting until enough is free)
    """
    global _sentence_transformer_instances, _sentence_transformer_in_use
    
    # Find unused instances
    unused_keys = [k for k in _sentence_transformer_instances.keys() 
                   if k not in _sentence_transformer_in_use]
    
    if not unused_keys:
        return
    
    for key in unused_keys:
        # Check if we have enough memory now
        if required_vram_gb > 0 and _get_free_vram_gb() >= required_vram_gb:
            break
            
        logger.info(f"Evicting unused SentenceTransformer: {key}")
        try:
            del _sentence_transformer_instances[key]
        except Exception as e:
            logger.warning(f"Error evicting SentenceTransformer {key}: {e}")
    
    _clear_cuda_cache()


def get_sentence_transformer_instance(
    model_name: str,
    device: Optional[str] = None,
    estimated_vram_gb: float = 2.0,
):
    """
    Get or create a SentenceTransformer instance.

    Uses lazy eviction: if memory is tight and there are unused cached models,
    they will be evicted before loading the new model.

    Args:
        model_name: HuggingFace model ID (e.g., 'Qwen/Qwen3-Embedding-4B')
        device: Device to load model on ('cuda', 'cpu', or None for auto)
        estimated_vram_gb: Estimated VRAM needed for this model (for eviction decisions)

    Returns:
        Tuple of (SentenceTransformer instance, bool indicating if it was cached)
    """
    global _sentence_transformer_in_use
    key = f"{model_name}:{device or 'auto'}"

    was_cached = key in _sentence_transformer_instances

    if not was_cached:
        # Check if we need to free memory before loading
        free_vram = _get_free_vram_gb()
        if free_vram < estimated_vram_gb:
            logger.info(f"Free VRAM ({free_vram:.1f}GB) < needed ({estimated_vram_gb:.1f}GB), evicting unused models")
            _evict_unused_sentence_transformers(estimated_vram_gb)
        
        sentence_transformers = _get_sentence_transformers()

        logger.info(f"Loading SentenceTransformer model: {model_name}")

        import torch
        model_kwargs = {"torch_dtype": torch.float16}

        model = sentence_transformers.SentenceTransformer(
            model_name,
            device=device,
            model_kwargs=model_kwargs,
            trust_remote_code=True,
        )

        _sentence_transformer_instances[key] = model
        logger.info(f"SentenceTransformer model loaded: {model_name}")
    else:
        logger.info(f"Reusing cached SentenceTransformer model: {model_name}")

    # Mark as in use
    _sentence_transformer_in_use.add(key)
    return _sentence_transformer_instances[key], was_cached


def release_sentence_transformer(model_name: str, device: Optional[str] = None):
    """
    Mark a SentenceTransformer instance as no longer in use.
    
    The instance stays cached but can be evicted if memory is needed later.
    
    Args:
        model_name: HuggingFace model ID
        device: Device the model was loaded on
    """
    global _sentence_transformer_in_use
    key = f"{model_name}:{device or 'auto'}"
    _sentence_transformer_in_use.discard(key)
    logger.debug(f"Released SentenceTransformer: {key}")


def release_all_sentence_transformers():
    """Mark all SentenceTransformer instances as no longer in use."""
    global _sentence_transformer_in_use
    _sentence_transformer_in_use.clear()
    logger.debug("Released all SentenceTransformer instances")


def clear_sentence_transformer_instances(force: bool = False):
    """
    Clear all cached SentenceTransformer instances.

    Args:
        force: If True, actually delete instances and free GPU memory.
               If False (default), do nothing - instances should be reused.

    Note: SentenceTransformer instances are expensive to create and should be
    reused across pipeline runs. Only use force=True when shutting down.
    """
    global _sentence_transformer_instances, _sentence_transformer_in_use

    if not force:
        return

    for key in list(_sentence_transformer_instances.keys()):
        try:
            logger.info(f"Shutting down SentenceTransformer instance: {key}")
            del _sentence_transformer_instances[key]
        except Exception as e:
            logger.warning(f"Error cleaning up SentenceTransformer instance {key}: {e}")

    _sentence_transformer_instances.clear()
    _sentence_transformer_in_use.clear()
    _clear_cuda_cache()


# ============================================================================
# vLLM Pool with Lazy Eviction
# ============================================================================

_vllm_module = None
_vllm_instances: Dict[str, Any] = {}
_vllm_in_use: Set[str] = set()  # Keys currently in use


def _get_vllm():
    """Lazy import of vllm module."""
    global _vllm_module
    if _vllm_module is None:
        try:
            import vllm
            _vllm_module = vllm
        except ImportError:
            raise ImportError(
                "vllm package required for LELA vLLM disambiguator. "
                "Install with: pip install vllm"
            )
    return _vllm_module


def _evict_unused_vllm_instances():
    """Evict all unused vLLM instances to free memory."""
    global _vllm_instances, _vllm_in_use
    
    unused_keys = [k for k in _vllm_instances.keys() if k not in _vllm_in_use]
    
    for key in unused_keys:
        logger.info(f"Evicting unused vLLM instance: {key}")
        try:
            del _vllm_instances[key]
        except Exception as e:
            logger.warning(f"Error evicting vLLM instance {key}: {e}")
    
    if unused_keys:
        _clear_cuda_cache()


def _evict_all_unused_models(required_vram_gb: float = 0):
    """
    Evict ALL unused models (both SentenceTransformer and vLLM) to free memory.
    
    Called before loading large models like vLLM.
    """
    # First evict unused SentenceTransformers
    _evict_unused_sentence_transformers(required_vram_gb)
    
    # Then evict unused vLLM instances
    _evict_unused_vllm_instances()
    
    # Final cache clear
    _clear_cuda_cache()


def get_vllm_instance(
    model_name: str,
    tensor_parallel_size: int = 1,
    max_model_len: Optional[int] = None,
    estimated_vram_gb: float = 10.0,
    **kwargs,
):
    """
    Get or create a vLLM LLM instance.

    Uses lazy eviction: if memory is tight and there are unused cached models
    (including SentenceTransformers), they will be evicted before loading.

    Args:
        model_name: HuggingFace model ID
        tensor_parallel_size: Number of GPUs for tensor parallelism
        max_model_len: Maximum sequence length
        estimated_vram_gb: Estimated VRAM needed (for eviction decisions)
        **kwargs: Additional vLLM arguments

    Returns:
        Tuple of (vLLM LLM instance, bool indicating if it was cached)
    """
    global _vllm_in_use
    key = f"{model_name}:tp{tensor_parallel_size}"

    was_cached = key in _vllm_instances

    if not was_cached:
        # vLLM needs a lot of memory - evict ALL unused models first
        free_vram = _get_free_vram_gb()
        if free_vram < estimated_vram_gb:
            logger.info(f"Free VRAM ({free_vram:.1f}GB) < needed ({estimated_vram_gb:.1f}GB), evicting unused models")
            _evict_all_unused_models(estimated_vram_gb)
        
        vllm = _get_vllm()

        llm_kwargs = {
            "model": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "enforce_eager": True,  # Disable CUDA graphs to avoid multiprocessing issues
            "dtype": "half",  # float16 for P100 (compute capability 6.0)
            "max_model_len": max_model_len or 4096,  # Qwen3-4B supports up to 32K
            "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
            "trust_remote_code": True,  # Required for Qwen models to load tokenizer/chat template
            **kwargs,
        }

        logger.info(f"Loading vLLM model: {model_name}")
        _vllm_instances[key] = vllm.LLM(**llm_kwargs)
        logger.info(f"vLLM model loaded: {model_name}")
    else:
        logger.info(f"Reusing cached vLLM model: {model_name}")

    # Mark as in use
    _vllm_in_use.add(key)
    return _vllm_instances[key], was_cached


def release_vllm(model_name: str, tensor_parallel_size: int = 1):
    """
    Mark a vLLM instance as no longer in use.
    
    The instance stays cached but can be evicted if memory is needed later.
    """
    global _vllm_in_use
    key = f"{model_name}:tp{tensor_parallel_size}"
    _vllm_in_use.discard(key)
    logger.debug(f"Released vLLM instance: {key}")


def release_all_vllm():
    """Mark all vLLM instances as no longer in use."""
    global _vllm_in_use
    _vllm_in_use.clear()
    logger.debug("Released all vLLM instances")


def clear_vllm_instances(force: bool = False):
    """
    Clear all cached vLLM instances.

    Args:
        force: If True, actually delete instances and free GPU memory.
               If False (default), do nothing - instances should be reused.

    Note: vLLM instances are expensive to create and should be reused across
    pipeline runs. Only use force=True when shutting down the application.
    """
    global _vllm_instances, _vllm_in_use

    if not force:
        return

    for key in list(_vllm_instances.keys()):
        try:
            logger.info(f"Shutting down vLLM instance: {key}")
            del _vllm_instances[key]
        except Exception as e:
            logger.warning(f"Error cleaning up vLLM instance {key}: {e}")

    _vllm_instances.clear()
    _vllm_in_use.clear()
    _clear_cuda_cache()


# ============================================================================
# Convenience Functions
# ============================================================================

def release_all_models():
    """Mark all cached models as no longer in use (available for eviction)."""
    release_all_sentence_transformers()
    release_all_vllm()
    logger.info("All models marked as available for eviction")


def is_sentence_transformer_cached(model_name: str, device: Optional[str] = None) -> bool:
    """Check if a SentenceTransformer model is currently cached."""
    key = f"{model_name}:{device or 'auto'}"
    return key in _sentence_transformer_instances


def is_vllm_cached(model_name: str, tensor_parallel_size: int = 1) -> bool:
    """Check if a vLLM model is currently cached."""
    key = f"{model_name}:tp{tensor_parallel_size}"
    return key in _vllm_instances


def get_cached_models_info() -> dict:
    """
    Get information about currently cached models.
    
    Returns:
        Dict with 'sentence_transformers' and 'vllm' keys, each containing
        a list of dicts with 'key' and 'in_use' fields.
    """
    return {
        "sentence_transformers": [
            {"key": key, "in_use": key in _sentence_transformer_in_use}
            for key in _sentence_transformer_instances.keys()
        ],
        "vllm": [
            {"key": key, "in_use": key in _vllm_in_use}
            for key in _vllm_instances.keys()
        ],
    }
