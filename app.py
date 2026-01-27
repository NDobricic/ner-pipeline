import os
# Disable vLLM V1 engine before any imports - V1 uses multiprocessing that fails from worker threads
os.environ["VLLM_USE_V1"] = "0"

import argparse
import gc
import importlib.util
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import torch

from ner_pipeline.config import PipelineConfig
from ner_pipeline.pipeline import NERPipeline

DESCRIPTION = """
# NER Pipeline 🔗

Modular NER → candidate generation → rerank → disambiguation pipeline built on spaCy. 
Swap components, configure parameters, and test with your own knowledge bases.
"""


def _is_vllm_usable() -> bool:
    """Check if vllm is installed and CUDA is available for it to run."""
    vllm_installed = importlib.util.find_spec("vllm") is not None
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass
    
    # Log for debugging
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"vLLM check: installed={vllm_installed}, cuda={cuda_available}")
    
    return vllm_installed and cuda_available


def get_available_components() -> Dict[str, List[str]]:
    """Get list of available spaCy pipeline components."""
    # These map to spaCy factories registered in ner_pipeline.spacy_components
    # lela_tournament is the full LELA paper implementation with tournament batching
    # lela_vllm sends all candidates at once (simpler but less accurate for many candidates)
    available_disambiguators = ["none", "first", "popularity", "lela_tournament", "lela_vllm", "lela_transformers"]
    
    return {
        "loaders": ["text", "pdf", "docx", "html", "json", "jsonl"],
        "ner": ["simple", "spacy", "gliner", "transformers", "lela_gliner"],
        "candidates": ["fuzzy", "bm25", "lela_bm25", "lela_dense"],
        "rerankers": ["none", "cross_encoder", "lela_embedder"],
        "disambiguators": available_disambiguators,
        "knowledge_bases": ["custom"],
    }


def filter_entities_by_confidence(result: Dict, threshold: float) -> Dict:
    """Filter entities by normalized linking confidence threshold.

    Args:
        result: Full pipeline result dict
        threshold: Minimum normalized confidence (0-1) to include entity

    Returns:
        New result dict with filtered entities
    """
    if not result:
        return result

    filtered_entities = []
    for entity in result.get("entities", []):
        conf = entity.get("linking_confidence_normalized")
        # Include entity if: no confidence score (keep unlinked), or confidence >= threshold
        if conf is None or conf >= threshold:
            filtered_entities.append(entity)

    return {
        **result,
        "entities": filtered_entities,
    }


def format_highlighted_text(result: Dict) -> List[Tuple[str, Optional[str]]]:
    """Convert pipeline result to HighlightedText format.

    Entities are shown with different labels based on linking status:
    - Linked: "LABEL: Entity Title" (e.g., "PERSON: Albert Einstein")
    - Unlinked: "LABEL [NOT IN KB]" (e.g., "PERSON [NOT IN KB]")
    """
    highlighted, _ = format_highlighted_text_with_threshold(result, threshold=0.0)
    return highlighted


GRAY_COLOR = "#D1D5DB"  # Tailwind gray-300 (light gray)

# Color palette for consistent entity colors
# Based on D3 Category20 / Tableau 20 - industry standard for categorical data visualization
# First 10: saturated colors for primary distinction
# Next 10: lighter variants for additional categories
# Source: https://d3js.org/d3-scale-chromatic/categorical
ENTITY_COLORS = [
    "#1F77B4",  # Blue
    "#FF7F0E",  # Orange
    "#2CA02C",  # Green
    "#D62728",  # Red
    "#9467BD",  # Purple
    "#8C564B",  # Brown
    "#E377C2",  # Pink
    "#7F7F7F",  # Gray
    "#BCBD22",  # Olive
    "#17BECF",  # Cyan
    "#AEC7E8",  # Light Blue
    "#FFBB78",  # Light Orange
    "#98DF8A",  # Light Green
    "#FF9896",  # Light Red
    "#C5B0D5",  # Light Purple
    "#C49C94",  # Light Brown
    "#F7B6D2",  # Light Pink
    "#C7C7C7",  # Light Gray
    "#DBDB8D",  # Light Olive
    "#9EDAE5",  # Light Cyan
]


def get_label_color(label: str) -> str:
    """Get consistent color for a label based on its hash."""
    idx = hash(label) % len(ENTITY_COLORS)
    return ENTITY_COLORS[idx]


def format_highlighted_text_with_threshold(
    result: Dict,
    threshold: float = 0.0,
) -> Tuple[List[Tuple[str, Optional[str]]], Dict[str, str]]:
    """Convert pipeline result to HighlightedText format with confidence-based coloring.

    Entities below the threshold are shown in gray.
    Returns (highlighted_data, color_map) for use with gr.HighlightedText.
    """
    text = result["text"]
    entities = result["entities"]

    if not entities:
        return [(text, None)], {}

    # Process entities: build labels and track max confidence per label
    entity_data = []  # (entity, label, conf)
    label_max_conf = {}  # Track max confidence for each unique label

    for entity in entities:
        conf = entity.get("linking_confidence_normalized")

        label_type = entity.get("label", "ENT")
        if entity.get("entity_title"):
            label = f"{label_type}: {entity['entity_title']}"
        else:
            label = f"{label_type} [NOT IN KB]"

        entity_data.append((entity, label, conf))

        # Track max confidence for this label (None means unlinked, treat as above threshold)
        if label not in label_max_conf:
            label_max_conf[label] = conf
        elif conf is not None:
            if label_max_conf[label] is None or conf > label_max_conf[label]:
                label_max_conf[label] = conf

    # Determine which labels are above/below threshold based on their max confidence
    above_threshold_labels = []
    below_threshold_labels = []

    for label, max_conf in label_max_conf.items():
        # Label is "low confidence" only if it has a confidence value AND it's below threshold
        is_low = max_conf is not None and max_conf < threshold
        if is_low:
            below_threshold_labels.append(label)
        else:
            above_threshold_labels.append(label)

    # Build color_map: above-threshold labels FIRST (for legend ordering), then below
    color_map = {}

    for label in above_threshold_labels:
        color_map[label] = get_label_color(label)

    for label in below_threshold_labels:
        color_map[label] = GRAY_COLOR

    # Sort by position for text reconstruction
    entity_data.sort(key=lambda x: x[0]["start"])

    # Build highlighted text
    highlighted = []
    last_end = 0

    for entity, label, _ in entity_data:
        if entity["start"] > last_end:
            highlighted.append((text[last_end:entity["start"]], None))

        highlighted.append((entity["text"], label))
        last_end = entity["end"]

    if last_end < len(text):
        highlighted.append((text[last_end:], None))

    return highlighted, color_map


def compute_linking_stats(result: Dict, threshold: float = 0.0) -> str:
    """Compute statistics about entity linking results with threshold breakdown."""
    entities = result.get("entities", [])
    if not entities:
        return "No entities found."

    total = len(entities)
    linked = sum(1 for e in entities if e.get("entity_title"))
    unlinked = total - linked

    # Compute average confidence for linked entities
    confidences = [e.get("linking_confidence") for e in entities if e.get("linking_confidence") is not None]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    # Count entities above/below threshold
    above_threshold = 0
    below_threshold = 0
    for e in entities:
        conf = e.get("linking_confidence_normalized")
        if conf is not None and conf < threshold:
            below_threshold += 1
        else:
            above_threshold += 1

    stats = f"**Entity Linking Statistics**\n\n"
    stats += f"- Total entities: {total}\n"
    stats += f"- Linked to KB: {linked} ({100*linked/total:.1f}%)\n"
    stats += f"- Not in KB: {unlinked} ({100*unlinked/total:.1f}%)\n"
    if confidences:
        stats += f"- Avg. confidence (linked): {avg_confidence:.3f}\n"

    if threshold > 0:
        stats += f"\n**Confidence Filter** (threshold: {threshold:.2f})\n\n"
        stats += f"- Above threshold: {above_threshold}\n"
        stats += f"- Below threshold (gray): {below_threshold}\n"

    return stats


def run_pipeline(
    text_input: str,
    file_input: Optional[gr.File],
    kb_file: Optional[gr.File],
    loader_type: str,
    ner_type: str,
    spacy_model: str,
    gliner_model: str,
    gliner_labels: str,
    gliner_threshold: float,
    simple_min_len: int,
    cand_type: str,
    cand_top_k: int,
    cand_use_context: bool,
    reranker_type: str,
    reranker_top_k: int,
    disambig_type: str,
    tournament_batch_size: int,
    tournament_shuffle: bool,
    tournament_thinking: bool,
    kb_type: str,
    progress=gr.Progress(),
) -> Tuple[List[Tuple[str, Optional[str]]], str, Dict]:
    """Run the NER pipeline with selected configuration."""
    # Note: We intentionally don't clear vLLM instances here - they should be
    # reused across runs to avoid expensive reinitialization and resource leaks.
    # Only general garbage collection is performed.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not kb_file:
        raise gr.Error("Please upload a knowledge base JSONL file.")

    if not text_input and not file_input:
        raise gr.Error("Please provide either text input or upload a file.")

    progress(0.1, desc="Building pipeline configuration...")
    
    # Build NER params based on type
    ner_params = {}
    if ner_type == "spacy":
        ner_params["model"] = spacy_model
    elif ner_type in ("gliner", "lela_gliner"):
        ner_params["model_name"] = gliner_model
        ner_params["threshold"] = gliner_threshold
        if gliner_labels:
            ner_params["labels"] = [l.strip() for l in gliner_labels.split(",")]
    elif ner_type == "simple":
        ner_params["min_len"] = simple_min_len
    
    # Build candidate params
    cand_params = {"top_k": cand_top_k}
    if cand_type in ("lela_bm25", "lela_dense"):
        cand_params["use_context"] = cand_use_context
    
    # Build reranker params
    reranker_params = {}
    if reranker_type != "none":
        reranker_params["top_k"] = reranker_top_k
    
    # Build disambiguator params
    disambig_params = {}
    if disambig_type == "lela_tournament":
        # batch_size=0 means auto (sqrt of candidates)
        disambig_params["batch_size"] = tournament_batch_size if tournament_batch_size > 0 else None
        disambig_params["shuffle_candidates"] = tournament_shuffle
        disambig_params["disable_thinking"] = not tournament_thinking
    
    config_dict = {
        "loader": {"name": loader_type, "params": {}},
        "ner": {"name": ner_type, "params": ner_params},
        "candidate_generator": {"name": cand_type, "params": cand_params},
        "reranker": {"name": reranker_type, "params": reranker_params} if reranker_type != "none" else {"name": "none", "params": {}},
        "disambiguator": {"name": disambig_type, "params": disambig_params} if disambig_type != "none" else None,
        "knowledge_base": {"name": kb_type, "params": {"path": kb_file.name}},
        "cache_dir": ".ner_cache",
        "batch_size": 1,
    }
    
    progress(0.15, desc="Initializing pipeline...")

    try:
        config = PipelineConfig.from_dict(config_dict)

        # Progress callback for initialization (maps 0-1 to 0.15-0.35)
        def init_progress_callback(local_progress: float, description: str):
            actual_progress = 0.15 + local_progress * 0.2
            progress(actual_progress, desc=description)

        pipeline = NERPipeline(config, progress_callback=init_progress_callback)
    except Exception as e:
        raise gr.Error(f"Failed to initialize pipeline: {str(e)}")
    
    progress(0.4, desc="Loading document...")

    try:
        if file_input:
            input_path = file_input.name
        else:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
                f.write(text_input)
                input_path = f.name

        # Load document using the pipeline's loader
        docs = list(pipeline.loader.load(input_path))

        if not file_input:
            os.unlink(input_path)

        if not docs:
            raise gr.Error("No documents loaded from input.")

        doc = docs[0]

        # Process with fine-grained progress callback
        def progress_callback(local_progress: float, description: str):
            # Map local progress (0.0-1.0) to our range (0.45-0.85)
            actual_progress = 0.45 + local_progress * 0.4
            progress(actual_progress, desc=description)

        result = pipeline.process_document_with_progress(
            doc,
            progress_callback=progress_callback,
            base_progress=0.0,
            progress_range=1.0,
        )

    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Pipeline execution failed: {str(e)}")
    
    progress(0.9, desc="Formatting output...")

    highlighted = format_highlighted_text(result)
    stats = compute_linking_stats(result)

    progress(1.0, desc="Done!")

    return highlighted, stats, result


def update_ner_params(ner_choice: str):
    """Show/hide NER-specific parameters based on selection."""
    return {
        spacy_params: gr.update(visible=(ner_choice == "spacy")),
        gliner_params: gr.update(visible=(ner_choice in ("gliner", "lela_gliner"))),
        simple_params: gr.update(visible=(ner_choice == "simple")),
    }


def update_cand_params(cand_choice: str):
    """Show/hide candidate-specific parameters based on selection."""
    show_context = cand_choice in ("lela_bm25", "lela_dense")
    return gr.update(visible=show_context)


def update_disambig_params(disambig_choice: str):
    """Show/hide disambiguator-specific parameters based on selection."""
    show_tournament = disambig_choice == "lela_tournament"
    return gr.update(visible=show_tournament)


def update_loader_from_file(file: Optional[gr.File]):
    """Auto-detect loader type from file extension."""
    if not file:
        return gr.update()

    ext = Path(file.name).suffix.lower()
    loader_map = {
        ".txt": "text",
        ".pdf": "pdf",
        ".docx": "docx",
        ".html": "html",
        ".htm": "html",
    }

    if ext in loader_map:
        return gr.update(value=loader_map[ext])
    return gr.update()


def apply_confidence_filter(
    full_result: Optional[Dict],
    threshold: float,
):
    """Apply confidence threshold filter and regenerate outputs.

    Args:
        full_result: The complete unfiltered pipeline result (from State)
        threshold: Confidence threshold from slider

    Returns:
        Tuple of (gr.HighlightedText, stats, full_json)
    """
    if not full_result:
        return gr.HighlightedText(value=[]), "*Run the pipeline to see results.*", {}

    highlighted, color_map = format_highlighted_text_with_threshold(full_result, threshold)
    stats = compute_linking_stats(full_result, threshold)

    return gr.HighlightedText(value=highlighted, color_map=color_map), stats, full_result


def apply_confidence_filter_display(
    full_result: Optional[Dict],
    threshold: float,
):
    """Apply confidence filter to display components only (skip JSON for performance)."""
    if not full_result:
        return gr.HighlightedText(value=[]), "*Run the pipeline to see results.*"

    highlighted, color_map = format_highlighted_text_with_threshold(full_result, threshold)
    stats = compute_linking_stats(full_result, threshold)

    return gr.HighlightedText(value=highlighted, color_map=color_map), stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER Pipeline Gradio UI")
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    args = parser.parse_args()
    
    logging.basicConfig(level=args.log)
    logger = logging.getLogger(__name__)
    
    # Log vLLM availability status
    vllm_ok = _is_vllm_usable()
    logger.info(f"vLLM disambiguator available: {vllm_ok}")
    
    components = get_available_components()
    
    with gr.Blocks(title="NER Pipeline", fill_height=True) as demo:
        gr.Markdown(DESCRIPTION)

        # State for storing full pipeline result (before confidence filtering)
        full_result_state = gr.State(value=None)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                
                text_input = gr.Textbox(
                    label="Text Input",
                    placeholder="Enter text to process...",
                    lines=8,
                    value="Albert Einstein was born in Germany. Marie Curie was a pioneering scientist.",
                )
                
                file_input = gr.File(
                    label="Or Upload Document",
                    file_types=[".txt", ".pdf", ".docx", ".html"],
                )
                
                kb_file = gr.File(
                    label="Knowledge Base (JSONL)",
                    file_types=[".jsonl"],
                    value="data/test/sample_kb.jsonl" if os.path.exists("data/test/sample_kb.jsonl") else None,
                )
                
                gr.Markdown("### Pipeline Configuration")
                
                with gr.Accordion("Loader", open=False):
                    loader_type = gr.Dropdown(
                        choices=components["loaders"],
                        value="text",
                        label="Loader Type",
                        info="Auto-detected from file extension",
                    )
                
                with gr.Accordion("Named Entity Recognition (NER)", open=True):
                    ner_type = gr.Dropdown(
                        choices=components["ner"],
                        value="simple",
                        label="NER Model",
                        info="Maps to spaCy component factory",
                    )
                    
                    with gr.Group(visible=False) as spacy_params:
                        spacy_model = gr.Textbox(
                            label="SpaCy Model",
                            value="en_core_web_sm",
                            info="Requires: python -m spacy download en_core_web_sm",
                        )
                    
                    with gr.Group(visible=False) as gliner_params:
                        gliner_model = gr.Textbox(
                            label="GLiNER Model",
                            value="urchade/gliner_large",
                        )
                        gliner_labels = gr.Textbox(
                            label="Entity Labels (comma-separated)",
                            value="person, organization, location",
                        )
                        gliner_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.5,
                            step=0.05,
                            label="Detection Threshold",
                        )
                    
                    with gr.Group(visible=True) as simple_params:
                        simple_min_len = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Minimum Length",
                        )
                
                with gr.Accordion("Candidate Generation", open=True):
                    cand_type = gr.Dropdown(
                        choices=components["candidates"],
                        value="fuzzy",
                        label="Candidate Generator",
                        info="Maps to spaCy component factory",
                    )
                    cand_top_k = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=64,
                        step=1,
                        label="Top K Candidates",
                        info="Retrieve this many candidates before reranking",
                    )
                    cand_use_context = gr.Checkbox(
                        label="Use Context",
                        value=True,
                        visible=False,
                        info="Include surrounding context in retrieval query",
                    )
                
                with gr.Accordion("Reranking", open=False):
                    reranker_type = gr.Dropdown(
                        choices=components["rerankers"],
                        value="none",
                        label="Reranker",
                        info="Maps to spaCy component factory",
                    )
                    reranker_top_k = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=10,
                        step=1,
                        label="Reranker Top K",
                    )
                
                with gr.Accordion("Disambiguation", open=True):
                    disambig_type = gr.Dropdown(
                        choices=components["disambiguators"],
                        value="first",
                        label="Disambiguator",
                        info="lela_tournament uses batched tournament selection (LELA paper)",
                    )
                    
                    with gr.Group(visible=False) as tournament_params:
                        tournament_batch_size = gr.Slider(
                            minimum=2,
                            maximum=32,
                            value=8,
                            step=1,
                            label="Tournament Batch Size (k)",
                            info="Candidates per batch. 0 = auto (√candidates). Paper recommends √C.",
                        )
                        tournament_shuffle = gr.Checkbox(
                            label="Shuffle Candidates",
                            value=True,
                            info="Randomize candidate order before tournament (as per LELA paper)",
                        )
                        tournament_thinking = gr.Checkbox(
                            label="Enable Reasoning",
                            value=True,
                            info="Enable LLM chain-of-thought reasoning (slower but more accurate)",
                        )
                
                with gr.Accordion("Knowledge Base", open=False):
                    kb_type = gr.Dropdown(
                        choices=components["knowledge_bases"],
                        value="custom",
                        label="KB Format",
                        info="JSONL format with fields: id (optional), title, description",
                    )
                
                run_btn = gr.Button("Run Pipeline", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### Output")

                confidence_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.01,
                    label="Confidence Threshold",
                    info="Filter display by linking confidence (JSON always shows full results)",
                )

                highlighted_output = gr.HighlightedText(
                    label="Linked Entities",
                    color_map={},
                    show_legend=True,
                )

                stats_output = gr.Markdown(
                    label="Linking Statistics",
                    value="*Run the pipeline to see entity linking statistics.*",
                )

                json_output = gr.JSON(
                    label="Full Pipeline Output",
                )
        
        file_input.change(
            fn=update_loader_from_file,
            inputs=[file_input],
            outputs=[loader_type],
        )
        
        ner_type.change(
            fn=update_ner_params,
            inputs=[ner_type],
            outputs=[spacy_params, gliner_params, simple_params],
        )
        
        cand_type.change(
            fn=update_cand_params,
            inputs=[cand_type],
            outputs=[cand_use_context],
        )
        
        disambig_type.change(
            fn=update_disambig_params,
            inputs=[disambig_type],
            outputs=[tournament_params],
        )
        
        run_btn.click(
            fn=lambda: ([], "*Processing...*", None, None),
            inputs=None,
            outputs=[highlighted_output, stats_output, json_output, full_result_state],
        ).then(
            fn=run_pipeline,
            inputs=[
                text_input,
                file_input,
                kb_file,
                loader_type,
                ner_type,
                spacy_model,
                gliner_model,
                gliner_labels,
                gliner_threshold,
                simple_min_len,
                cand_type,
                cand_top_k,
                cand_use_context,
                reranker_type,
                reranker_top_k,
                disambig_type,
                tournament_batch_size,
                tournament_shuffle,
                tournament_thinking,
                kb_type,
            ],
            outputs=[highlighted_output, stats_output, full_result_state],
            show_progress_on=highlighted_output,
        ).then(
            fn=apply_confidence_filter,
            inputs=[full_result_state, confidence_threshold],
            outputs=[highlighted_output, stats_output, json_output],
        )

        confidence_threshold.change(
            fn=apply_confidence_filter_display,
            inputs=[full_result_state, confidence_threshold],
            outputs=[highlighted_output, stats_output],
        )
        
        gr.Markdown("""
## Quick Start

1. **Upload Knowledge Base**: Provide a JSONL file with entities (fields: `title`, `description`, and optional `id`)
2. **Enter Text or Upload File**: Input text directly or upload a document
3. **Configure Pipeline**: Select spaCy components and adjust parameters
4. **Run**: Click "Run Pipeline" to process

### Example Files

Test files are available in `data/test/`:
- `sample_kb.jsonl` - Sample knowledge base with 10 entities
- `sample_doc.txt` - Sample document for testing

### spaCy Component Mapping

| Config Name | spaCy Factory |
|-------------|---------------|
| simple | ner_pipeline_simple |
| lela_gliner | ner_pipeline_lela_gliner |
| lela_bm25 | ner_pipeline_lela_bm25_candidates |
| lela_embedder | ner_pipeline_lela_embedder_reranker |
| lela_tournament | ner_pipeline_lela_tournament_disambiguator |
| lela_vllm | ner_pipeline_lela_vllm_disambiguator |
        """)
        
        with gr.Accordion("Entity Type Legend (SpaCy)", open=False):
            gr.Markdown("""
| Label | Meaning | Example |
|-------|---------|---------|
| **PERSON** | People, including fictional | *Albert Einstein*, *Marie Curie* |
| **ORG** | Organizations, companies, agencies | *Google*, *United Nations*, *NASA* |
| **GPE** | Geopolitical entities (countries, cities, states) | *France*, *New York*, *California* |
| **LOC** | Non-GPE locations (mountains, water bodies) | *Mount Everest*, *Pacific Ocean* |
| **FAC** | Facilities (buildings, airports, highways) | *Empire State Building*, *JFK Airport* |
| **PRODUCT** | Objects, vehicles, foods (not services) | *iPhone*, *Boeing 747* |
| **EVENT** | Named events (hurricanes, battles, wars) | *World War II*, *Hurricane Katrina* |
| **WORK_OF_ART** | Titles of books, songs, etc. | *The Great Gatsby*, *Mona Lisa* |
| **LAW** | Named documents made into laws | *Roe v. Wade*, *GDPR* |
| **LANGUAGE** | Any named language | *English*, *Mandarin* |
| **DATE** | Absolute or relative dates/periods | *January 2020*, *next week* |
| **TIME** | Times smaller than a day | *3:00 PM*, *morning* |
| **PERCENT** | Percentages | *50%*, *ten percent* |
| **MONEY** | Monetary values | *$100*, *€50 million* |
| **QUANTITY** | Measurements | *10 kg*, *five miles* |
| **ORDINAL** | Ordinal numbers | *first*, *3rd* |
| **CARDINAL** | Numerals not covered by other types | *one*, *100*, *millions* |
| **NORP** | Nationalities, religious/political groups | *American*, *Buddhist*, *Republican* |
| **ENT** | Generic entity (used by simple regex NER) | Any capitalized phrase |
            """)
    
    logger.info(f"Launching Gradio UI on port {args.port}...")
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
