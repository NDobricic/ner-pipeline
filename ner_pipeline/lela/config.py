"""Default configuration values for LELA components."""

# NER labels for zero-shot entity recognition
NER_LABELS = ["person", "organization", "location", "event", "work of art", "product"]

# Model IDs
DEFAULT_GLINER_MODEL = "numind/NuNER_Zero-span"
# Qwen3-4B for entity disambiguation (use lela_transformers on P100 GPUs)
DEFAULT_LLM_MODEL = "Qwen/Qwen3-4B"
DEFAULT_EMBEDDER_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_RERANKER_MODEL = "tomaarsen/Qwen3-Reranker-4B-seq-cls"

# Available LLM models for disambiguation (model_id, display_name, vram_gb)
AVAILABLE_LLM_MODELS = [
    ("Qwen/Qwen3-0.6B", "Qwen3-0.6B (~2GB VRAM)", 2.0),
    ("Qwen/Qwen3-1.7B", "Qwen3-1.7B (~4GB VRAM)", 4.0),
    ("Qwen/Qwen3-4B", "Qwen3-4B (~9GB VRAM)", 9.0),
    ("Qwen/Qwen3-8B", "Qwen3-8B (~18GB VRAM)", 18.0),
]

# Available embedding models (model_id, display_name, vram_gb)
AVAILABLE_EMBEDDING_MODELS = [
    # ("sentence-transformers/all-MiniLM-L6-v2", "MiniLM-L6 (~0.3GB)", 0.3),
    # ("BAAI/bge-base-en-v1.5", "BGE-Base (~0.5GB)", 0.5),
    ("Qwen/Qwen3-Embedding-0.6B", "Qwen3-Embed-0.6B (~2GB)", 2.0),
    ("Qwen/Qwen3-Embedding-4B", "Qwen3-Embed-4B (~9GB)", 9.0),
]

# Available cross-encoder models for reranking
AVAILABLE_CROSS_ENCODER_MODELS = [
    # ("cross-encoder/ms-marco-TinyBERT-L-2-v2", "TinyBERT (~0.1GB)", 0.1),
    ("tomaarsen/Qwen3-Reranker-4B-seq-cls", "Qwen3-Reranker-4B (~9GB)", 9.0),
    ("tomaarsen/Qwen3-Reranker-0.6B-seq-cls", "Qwen3-Reranker-0.6B (~2GB)", 2.0),
]

# Retrieval settings
CANDIDATES_TOP_K = 64
RERANKER_TOP_K = 10

# Span markers for disambiguation prompts
SPAN_OPEN = "["
SPAN_CLOSE = "]"

# Special entity label
NOT_AN_ENTITY = "None"

# vLLM settings
DEFAULT_TENSOR_PARALLEL_SIZE = 1
DEFAULT_MAX_MODEL_LEN = None
VLLM_GPU_MEMORY_UTILIZATION = (
    0.8  # Fraction of GPU memory vLLM will use (0.8 leaves headroom for loading)
)

# Embedding task descriptions
RETRIEVER_TASK = (
    "Given an ambiguous mention, retrieve relevant entities that the mention refers to."
)
RERANKER_TASK = (
    "Given a text with a mention enclosed between the '[' and ']' characters, "
    "retrieve relevant entities that the mention refers to."
)

# Default generation config for LLM disambiguation
# Qwen3 needs more tokens due to thinking mode
DEFAULT_GENERATION_CONFIG = {
    "max_tokens": 2048,  # More tokens for thinking mode + answer
    "temperature": 0.1,  # Low temperature for deterministic outputs
    "top_p": 0.9,
    "repetition_penalty": 1.1,  # Prevent repetitive garbage output
}
