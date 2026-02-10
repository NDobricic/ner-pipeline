import os

# Disable vLLM V1 engine before any imports - V1 uses multiprocessing that fails from worker threads
os.environ["VLLM_USE_V1"] = "0"

import argparse
import gc
import importlib.util
import logging
import tempfile
import threading
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import gradio as gr
import torch

# Global cancellation event for cooperative cancellation
_cancel_event = threading.Event()

from el_pipeline.config import PipelineConfig
from el_pipeline.pipeline import ELPipeline
from el_pipeline.memory import get_system_resources
from el_pipeline.lela.config import (
    AVAILABLE_LLM_MODELS as LLM_MODEL_CHOICES,
    AVAILABLE_EMBEDDING_MODELS as EMBEDDING_MODEL_CHOICES,
    AVAILABLE_CROSS_ENCODER_MODELS as CROSS_ENCODER_MODEL_CHOICES,
    AVAILABLE_VLLM_RERANKER_MODELS as VLLM_RERANKER_MODEL_CHOICES,
    DEFAULT_VLLM_RERANKER_MODEL,
    DEFAULT_GLINER_MODEL,
)

DESCRIPTION = """
*Modular entity recognition and linking pipeline. Upload a knowledge base, enter text, configure the pipeline, and run.*"

## Links
- Repository: https://github.com/samyhaff/LELA
- LELA Paper: https://arxiv.org/abs/2601.05192
"""

LOGO = """
<div style="display: flex; justify-content: center; align-items: center; gap: 40px; margin-top: 40px;">
    <img src="https://www.telecom-paris.fr/wp-content-EvDsK19/uploads/2024/01/logo_telecom_ipparis_rvb_fond_h-768x359.png" alt="Telecom Paris Logo" style="height: 80px;">
    <img src="https://www.ip-paris.fr/sites/default/files/image002.png" alt="IP Paris Logo" style="height: 80px;">
    <img src="https://yago-knowledge.org/assets/images/logo.png" alt="YAGO Logo" style="height: 80px;">
</div>
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
    # These map to spaCy factories registered in el_pipeline.spacy_components
    available_disambiguators = [
        "none",
        "first",
        "lela_vllm",
        "lela_transformers",
        "lela_openai_api",
    ]

    return {
        "loaders": ["text", "pdf", "docx", "html", "json", "jsonl"],
        "ner": ["simple", "spacy", "gliner"],
        "candidates": ["none", "fuzzy", "bm25", "lela_dense", "lela_openai_api_dense"],
        "rerankers": [
            "none",
            "lela_cross_encoder",
            "lela_vllm_api_client",
            "lela_llama_server",
            "lela_embedder_transformers",
            "lela_embedder_vllm",
            "lela_cross_encoder_vllm",
        ],
        "disambiguators": available_disambiguators,
        "knowledge_bases": ["custom"],
    }


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


ERROR_COLOR = "#DC2626"  # Tailwind red-600


def get_label_color(label: str) -> str:
    """Get consistent color for a label based on its hash."""
    if label == "ERROR":
        return ERROR_COLOR
    idx = hash(label) % len(ENTITY_COLORS)
    return ENTITY_COLORS[idx]


def highlighted_to_html(
    highlighted: List[Tuple[str, Optional[str], Optional[Dict]]],
    color_map: Dict[str, str],
    show_legend: bool = True,
) -> str:
    """Convert highlighted text data to HTML with inline styles, interactive hover, and popups.

    This bypasses Gradio's buggy HighlightedText component.
    When hovering a legend item, only entities of that type are highlighted; others turn gray.
    When hovering an entity, a popup shows detailed information.
    """
    import html
    import hashlib
    import uuid
    import json

    def label_to_class(label: str) -> str:
        """Convert label to a valid CSS class name."""
        return "ent-" + hashlib.md5(label.encode()).hexdigest()[:8]

    def escape_js_string(s: str) -> str:
        """Escape a string for use in JavaScript within HTML attributes."""
        if s is None:
            return ""
        s = str(s)
        # Escape backslashes first, then other special chars
        s = s.replace("\\", "\\\\")
        s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        s = html.escape(s)
        s = s.replace("'", "\\'").replace('"', "&quot;")
        return s

    # Generate unique container ID for this render
    container_id = f"ner-output-{uuid.uuid4().hex[:8]}"
    popup_id = f"popup-{container_id}"

    # Collect entity info by label for legend popups (first instance + count)
    label_entity_info = {}  # label -> entity_info (first instance)
    label_counts = {}  # label -> count of occurrences

    # Build text with entity marks
    parts = []
    for item in highlighted:
        text, label = item[0], item[1]
        entity_info = item[2] if len(item) > 2 else None

        escaped_text = html.escape(text)
        if label is None:
            parts.append(escaped_text)
        else:
            # Use per-instance color if available, otherwise fall back to color_map
            color = (
                entity_info.get("display_color") if entity_info else None
            ) or color_map.get(label, "#808080")
            css_class = label_to_class(label)

            # Track count and store first entity info for each label (for legend popup)
            label_counts[label] = label_counts.get(label, 0) + 1
            if label not in label_entity_info and entity_info:
                label_entity_info[label] = entity_info

            # Build popup content for this entity
            popup_lines = []
            if entity_info:
                if entity_info.get("kb_title"):
                    popup_lines.append(
                        f"<strong style=&quot;color:#333;&quot;>{escape_js_string(entity_info['kb_title'])}</strong>"
                    )
                if entity_info.get("kb_id"):
                    popup_lines.append(
                        f"<em style=&quot;color:#555;&quot;>ID: {escape_js_string(entity_info['kb_id'])}</em>"
                    )
                if entity_info.get("type"):
                    popup_lines.append(
                        f"<span style=&quot;color:#333;&quot;>Type: {escape_js_string(entity_info['type'])}</span>"
                    )
                if entity_info.get("mention") and entity_info.get(
                    "mention"
                ) != entity_info.get("kb_title"):
                    popup_lines.append(
                        f"<span style=&quot;color:#333;&quot;>Mention: &quot;{escape_js_string(entity_info['mention'])}&quot;</span>"
                    )
                if entity_info.get("kb_description"):
                    desc = entity_info["kb_description"]
                    if len(desc) > 150:
                        desc = desc[:150] + "..."
                    popup_lines.append(
                        f"<div style=&quot;margin-top:0.3em;font-size:0.9em;color:#666;&quot;>{escape_js_string(desc)}</div>"
                    )

            popup_content = (
                "<br>".join(popup_lines) if popup_lines else escape_js_string(label)
            )

            # JavaScript for showing/hiding popup (absolute positioning relative to container)
            # Account for container scroll position and use viewport for bounds checking
            show_popup_js = (
                f"var p=document.getElementById('{popup_id}');"
                f"p.innerHTML='{popup_content}';"
                f"var cont=document.getElementById('{container_id}');"
                f"var r=this.getBoundingClientRect();"
                f"var c=cont.getBoundingClientRect();"
                f"var left=r.left-c.left+cont.scrollLeft;"
                f"var top=r.bottom-c.top+cont.scrollTop+5;"
                f"p.style.left=left+'px';p.style.top=top+'px';"
                f"p.style.display='block';"
            )
            hide_popup_js = (
                f"document.getElementById('{popup_id}').style.display='none';"
            )

            parts.append(
                f'<mark class="entity-mark {css_class}" '
                f'data-color="{color}" '
                f'style="background-color: {color}; padding: 0.1em 0.2em; '
                f"border-radius: 0.2em; margin: 0 0.1em; cursor: pointer; "
                f'transition: background-color 0.2s ease, opacity 0.2s ease;" '
                f'onmouseenter="{show_popup_js}" '
                f'onmouseleave="{hide_popup_js}">'
                f"{escaped_text}</mark>"
            )

    # JavaScript functions for legend hover (highlight/dim entities)
    hover_in_js = (
        "var c=document.getElementById('{cid}');"
        "c.querySelectorAll('.entity-mark').forEach(function(m){{"
        "if(m.classList.contains('{cls}')){{m.style.opacity='1';}}"
        "else{{m.style.backgroundColor='#E5E7EB';m.style.opacity='0.6';}}"
        "}});"
    )
    hover_out_js = (
        "var c=document.getElementById('{cid}');"
        "c.querySelectorAll('.entity-mark').forEach(function(m){{"
        "m.style.backgroundColor=m.getAttribute('data-color');"
        "m.style.opacity='1';"
        "}});"
    )

    # Build legend with inline hover handlers and popups
    legend_parts = []
    seen_labels = set()
    for item in highlighted:
        label = item[1]
        if label and label not in seen_labels:
            seen_labels.add(label)
            color = color_map.get(label, "#808080")
            css_class = label_to_class(label)
            enter_js = hover_in_js.format(cid=container_id, cls=css_class)
            leave_js = hover_out_js.format(cid=container_id)

            # Build popup content for legend item (summary info, no instance-specific values)
            entity_info = label_entity_info.get(label)
            count = label_counts.get(label, 1)
            popup_lines = []
            if entity_info:
                if entity_info.get("kb_title"):
                    popup_lines.append(
                        f"<strong style=&quot;color:#333;&quot;>{escape_js_string(entity_info['kb_title'])}</strong>"
                    )
                if entity_info.get("kb_id"):
                    popup_lines.append(
                        f"<em style=&quot;color:#555;&quot;>ID: {escape_js_string(entity_info['kb_id'])}</em>"
                    )
                if entity_info.get("type"):
                    popup_lines.append(
                        f"<span style=&quot;color:#333;&quot;>Type: {escape_js_string(entity_info['type'])}</span>"
                    )
                # Show occurrence count for legend hover
                popup_lines.append(
                    f"<span style=&quot;color:#333;&quot;>Mentions: {count}</span>"
                )
                if entity_info.get("kb_description"):
                    desc = entity_info["kb_description"]
                    if len(desc) > 150:
                        desc = desc[:150] + "..."
                    popup_lines.append(
                        f"<div style=&quot;margin-top:0.3em;font-size:0.9em;color:#666;&quot;>{escape_js_string(desc)}</div>"
                    )

            popup_content = (
                "<br>".join(popup_lines) if popup_lines else escape_js_string(label)
            )

            show_popup_js = (
                f"var p=document.getElementById('{popup_id}');"
                f"p.innerHTML='{popup_content}';"
                f"var cont=document.getElementById('{container_id}');"
                f"var r=this.getBoundingClientRect();"
                f"var c=cont.getBoundingClientRect();"
                f"var left=r.left-c.left+cont.scrollLeft;"
                f"var top=r.bottom-c.top+cont.scrollTop+5;"
                f"p.style.left=left+'px';p.style.top=top+'px';"
                f"p.style.display='block';"
            )
            hide_popup_js = (
                f"document.getElementById('{popup_id}').style.display='none';"
            )

            # Combine highlight JS with popup JS
            combined_enter = enter_js + show_popup_js
            combined_leave = leave_js + hide_popup_js

            legend_parts.append(
                f'<span class="legend-item" '
                f'style="display: inline-block; margin-right: 1em; cursor: pointer;" '
                f'onmouseenter="{combined_enter}" '
                f'onmouseleave="{combined_leave}">'
                f'<span style="background-color: {color}; padding: 0.1em 0.3em; '
                f'border-radius: 0.2em; font-size: 0.85em;">{html.escape(label)}</span></span>'
            )

    legend_html = ""
    if show_legend:
        legend_html = (
            f'<div class="entity-legend" style="margin-bottom: 0.5em; line-height: 1.8;">{"".join(legend_parts)}</div>'
            if legend_parts
            else ""
        )
    text_html = f'<div class="entity-text" style="line-height: 1.6; white-space: pre-wrap;">{"".join(parts)}</div>'

    # Popup div (hidden by default, absolute positioning relative to container)
    popup_html = (
        f'<div id="{popup_id}" style="'
        f"display: none; "
        f"position: absolute; "
        f"background: white; "
        f"color: #333; "
        f"border: 1px solid #ccc; "
        f"border-radius: 6px; "
        f"padding: 0.5em 0.75em; "
        f"box-shadow: 0 2px 8px rgba(0,0,0,0.15); "
        f"max-width: 350px; "
        f"z-index: 1000; "
        f"font-size: 0.9em; "
        f"line-height: 1.4; "
        f"pointer-events: none;"
        f'"></div>'
    )

    return f'<div id="{container_id}" class="highlighted-container" style="position: relative;">{legend_html}{text_html}{popup_html}</div>'


def format_highlighted_text(
    result: Dict,
) -> Tuple[List[Tuple[str, Optional[str], Optional[Dict]]], Dict[str, str]]:
    """Convert pipeline result to highlighted format.

    Returns (highlighted_data, color_map) for use with highlighted_to_html().
    Each highlighted item is (text, label, entity_info) where entity_info contains details for popup.
    """
    text = result["text"]
    entities = result["entities"]

    if not entities:
        return [(text, None, None)], {}

    # Process entities: build labels
    entity_data = []
    labels = set()

    for entity in entities:
        label_type = entity.get("label", "ENT")
        if entity.get("entity_title"):
            label = f"{label_type}: {entity['entity_title']}"
        else:
            label = f"{label_type} [NOT IN KB]"
        entity_data.append((entity, label))
        labels.add(label)

    # Build color_map
    color_map = {label: get_label_color(label) for label in labels}

    # Sort by position for text reconstruction
    entity_data.sort(key=lambda x: x[0]["start"])

    # Build highlighted text with entity info for popups
    highlighted = []
    last_end = 0

    for entity, label in entity_data:
        if entity["start"] > last_end:
            highlighted.append((text[last_end : entity["start"]], None, None))

        instance_color = color_map.get(label)

        # Build entity info dict for popup
        entity_info = {
            "mention": entity["text"],
            "type": entity.get("label", "ENT"),
            "kb_id": entity.get("entity_id"),
            "kb_title": entity.get("entity_title"),
            "kb_description": entity.get("entity_description"),
            "display_color": instance_color,
        }

        highlighted.append((entity["text"], label, entity_info))
        last_end = entity["end"]

    if last_end < len(text):
        highlighted.append((text[last_end:], None, None))

    return highlighted, color_map


def compute_linking_stats(result: Dict) -> str:
    """Compute statistics about entity linking results."""
    entities = result.get("entities", [])
    if not entities:
        return "No entities found."

    total = len(entities)
    linked = sum(1 for e in entities if e.get("entity_title"))
    unlinked = total - linked

    stats = f"**Entity Linking Statistics**\n\n"
    stats += f"- Total entities: {total}\n"
    stats += f"- Linked to KB: {linked} ({100*linked/total:.1f}%)\n"
    stats += f"- Not in KB: {unlinked} ({100*unlinked/total:.1f}%)\n"

    return stats


def format_error_output(error_title: str, error_message: str) -> Tuple[str, str, Dict]:
    """Format an error for display in the output components.

    Returns (html_output, stats, result) for consistency with run_pipeline.
    """
    import traceback

    # Get full traceback if available
    tb = traceback.format_exc()
    if tb and tb.strip() != "NoneType: None":
        full_error = f"{error_message}\n\n**Traceback:**\n```\n{tb}\n```"
    else:
        full_error = error_message

    # Create HTML error display
    html_output = (
        f'<div style="color: {ERROR_COLOR}; padding: 1em; '
        f"border: 1px solid {ERROR_COLOR}; border-radius: 6px; "
        f'background-color: #FEF2F2;">'
        f"<strong>Error: {error_title}</strong></div>"
    )
    stats = f"**Error**\n\n{full_error}"
    result = {"error": error_title, "details": error_message}

    return html_output, stats, result


def _run_with_heartbeat(fn, progress_fn, initial_progress, initial_desc):
    """Run fn(report) in a background thread, sending progress heartbeats.

    Blocks until fn completes.  Calls progress_fn every 0.5s to keep
    Gradio's SSE connection alive (without yielding, which would re-render
    output components and reset the UI timer).

    fn receives a report(progress_value, description) callback for updates.
    Returns fn's return value.  Re-raises any exception from fn.
    """
    import queue as _q

    q = _q.Queue()
    result_holder = []
    error_holder = []

    def report(prog, desc):
        q.put((prog, desc))

    def _worker():
        try:
            r = fn(report)
            result_holder.append(r)
        except Exception as e:
            error_holder.append(e)
        finally:
            q.put(None)

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    last_progress = initial_progress
    last_desc = initial_desc
    while True:
        try:
            msg = q.get(timeout=0.5)
        except _q.Empty:
            try:
                progress_fn(last_progress, desc=last_desc)
            except Exception:
                pass
            continue
        if msg is None:
            break
        last_progress = msg[0]
        last_desc = msg[1]
        try:
            progress_fn(msg[0], desc=msg[1])
        except Exception:
            pass

    worker.join()

    if error_holder:
        raise error_holder[0]
    return result_holder[0] if result_holder else None


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
    cand_embedding_model: str,
    cand_top_k: int,
    cand_use_context: bool,
    cand_api_base_url: str,
    cand_api_key: str,
    reranker_type: str,
    reranker_embedding_model: str,
    reranker_cross_encoder_model: str,
    reranker_api_url: str,
    reranker_api_port: int,
    reranker_top_k: int,
    disambig_type: str,
    llm_model: str,
    lela_thinking: bool,
    lela_none_candidate: bool,
    disambig_api_base_url: str,
    disambig_api_key: str,
    kb_type: str,
    progress=gr.Progress(),
):
    """Run the EL pipeline with selected configuration.

    This is a generator function that yields (html_output, stats, result) tuples.
    Yielding allows Gradio to check for cancellation between steps.
    """
    import sys

    logger = logging.getLogger(__name__)
    logger.info(f"=== run_pipeline ENTERED (run #{_run_counter}) ===")
    sys.stderr.flush()

    # Clear cancellation flag at the start of a new run
    _cancel_event.clear()

    # Note: We intentionally don't clear vLLM instances here - they should be
    # reused across runs to avoid expensive reinitialization and resource leaks.
    # Only general garbage collection is performed.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    no_vis_change = gr.update()

    if not kb_file:
        from el_pipeline.knowledge_bases.yago_downloader import ensure_yago_kb

        kb_path = _run_with_heartbeat(
            lambda report: ensure_yago_kb(),
            progress,
            0.05,
            "Resolving knowledge base...",
        )
    else:
        kb_path = kb_file.name

    if not text_input and not file_input:
        yield (
            *format_error_output(
                "Missing Input", "Please provide either text input or upload a file."
            ),
            no_vis_change,
        )
        return

    # Check for cancellation
    if _cancel_event.is_set():
        logger.info("Pipeline cancelled before configuration")
        yield "", "*Pipeline cancelled.*", {}, no_vis_change
        return

    progress(0.1, desc="Building pipeline configuration...")
    yield "", "*Building configuration...*", {}, no_vis_change

    # Build NER params based on type
    ner_params = {}
    if ner_type == "spacy":
        ner_params["model"] = spacy_model
    elif ner_type == "gliner":
        ner_params["model_name"] = gliner_model
        ner_params["threshold"] = gliner_threshold
        if gliner_labels:
            ner_params["labels"] = [l.strip() for l in gliner_labels.split(",")]
    elif ner_type == "simple":
        ner_params["min_len"] = simple_min_len

    # Build candidate params
    cand_params = {"top_k": cand_top_k}
    if cand_type == "lela_dense":
        cand_params["use_context"] = cand_use_context
        cand_params["model_name"] = cand_embedding_model
    elif cand_type == "lela_openai_api_dense":
        cand_params["use_context"] = cand_use_context
        cand_params["model_name"] = cand_embedding_model
        cand_params["base_url"] = cand_api_base_url
        cand_params["api_key"] = cand_api_key

    # Build reranker params
    reranker_params = {"top_k": reranker_top_k}
    if reranker_type == "lela_embedder":
        reranker_params["model_name"] = reranker_embedding_model
    if reranker_type in ("lela_embedder_transformers", "lela_embedder_vllm"):
        reranker_params["model_name"] = reranker_embedding_model
    if reranker_type == "lela_cross_encoder_vllm":
        reranker_params["model_name"] = reranker_cross_encoder_model
    if reranker_type == "lela_cross_encoder":
        reranker_params["model_name"] = reranker_cross_encoder_model
    if reranker_type == "lela_vllm_api_client":
        reranker_params["base_url"] = reranker_api_url
        reranker_params["port"] = reranker_api_port

    # Build disambiguator params
    disambig_params = {}
    if disambig_type in ("lela_vllm", "lela_transformers"):
        disambig_params["model_name"] = llm_model
        disambig_params["disable_thinking"] = not lela_thinking
        disambig_params["add_none_candidate"] = lela_none_candidate
    if disambig_type == "lela_openai_api":
        disambig_params["base_url"] = disambig_api_base_url
        disambig_params["api_key"] = disambig_api_key or None

    # Override components if candidate_generator is "none" for NER-only pipeline
    if cand_type == "none":
        candidate_generator_config = {"name": "none", "params": {}}
        reranker_config = {"name": "none", "params": {}}
        disambiguator_config = None
    else:
        candidate_generator_config = {"name": cand_type, "params": cand_params}
        reranker_config = (
            {"name": reranker_type, "params": reranker_params}
            if reranker_type != "none"
            else {"name": "none", "params": {}}
        )
        disambiguator_config = (
            {"name": disambig_type, "params": disambig_params}
            if disambig_type != "none"
            else None
        )

    config_dict = {
        "loader": {"name": loader_type, "params": {}},
        "ner": {"name": ner_type, "params": ner_params},
        "candidate_generator": candidate_generator_config,
        "reranker": reranker_config,
        "disambiguator": disambiguator_config,
        "knowledge_base": {"name": kb_type, "params": {"path": kb_path}},
        "cache_dir": ".ner_cache",
        "batch_size": 1,
    }

    # Check for cancellation
    if _cancel_event.is_set():
        logger.info("Pipeline cancelled before initialization")
        yield "", "*Pipeline cancelled.*", {}, no_vis_change
        return

    progress(0.15, desc="Initializing pipeline...")
    yield "", "*Initializing pipeline...*", {}, no_vis_change

    try:
        config = PipelineConfig.from_dict(config_dict)

        def _init_pipeline(report):
            def init_progress_callback(local_progress: float, description: str):
                report(0.15 + local_progress * 0.2, description)
                if _cancel_event.is_set():
                    raise InterruptedError("Pipeline cancelled by user")

            return ELPipeline(
                config,
                progress_callback=init_progress_callback,
                cancel_event=_cancel_event,
            )

        pipeline = _run_with_heartbeat(
            _init_pipeline,
            progress,
            0.15,
            "Initializing pipeline...",
        )
    except InterruptedError:
        logger.info("Pipeline cancelled during initialization")
        yield "", "*Pipeline cancelled.*", {}, no_vis_change
        return
    except Exception as e:
        logger.exception("Pipeline initialization failed")
        yield (
            *format_error_output("Pipeline Initialization Failed", str(e)),
            no_vis_change,
        )
        return

    # Check for cancellation
    if _cancel_event.is_set():
        logger.info("Pipeline cancelled after initialization")
        yield "", "*Pipeline cancelled.*", {}, no_vis_change
        return

    progress(0.4, desc="Loading document...")
    yield "", "*Loading document...*", {}, no_vis_change

    try:
        if file_input:
            input_path = file_input.name
        else:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as f:
                f.write(text_input)
                input_path = f.name

        # Load document (in background thread to keep SSE alive for large files)
        docs = _run_with_heartbeat(
            lambda report: list(pipeline.loader.load(input_path)),
            progress,
            0.4,
            "Loading document...",
        )

        if not file_input:
            os.unlink(input_path)

        if not docs:
            yield (
                *format_error_output(
                    "No Documents Loaded",
                    "The input file was empty or could not be parsed.",
                ),
                no_vis_change,
            )
            return

        doc = docs[0]

        # Check for cancellation
        if _cancel_event.is_set():
            logger.info("Pipeline cancelled before processing")
            yield "", "*Pipeline cancelled.*", {}, no_vis_change
            return

        progress(0.45, desc="Processing document...")
        yield "", "*Processing document...*", {}, no_vis_change

        def _run_processing(report):
            def progress_callback(local_progress: float, description: str):
                report(0.45 + local_progress * 0.4, description)
                if _cancel_event.is_set():
                    raise InterruptedError("Pipeline cancelled by user")

            return pipeline.process_document_with_progress(
                doc,
                progress_callback=progress_callback,
                base_progress=0.0,
                progress_range=1.0,
            )

        result = _run_with_heartbeat(
            _run_processing,
            progress,
            0.45,
            "Processing document...",
        )

    except InterruptedError:
        logger.info("Pipeline cancelled during processing")
        yield "", "*Pipeline cancelled.*", {}, no_vis_change
        return
    except Exception as e:
        yield (*format_error_output("Pipeline Execution Failed", str(e)), no_vis_change)
        return

    logger.info("Pipeline processing complete, formatting output...")
    sys.stderr.flush()
    progress(0.9, desc="Formatting output...")

    logger.info("Calling format_highlighted_text...")
    sys.stderr.flush()
    highlighted, color_map = format_highlighted_text(result)
    logger.info(f"format_highlighted_text done, got {len(highlighted)} segments")
    sys.stderr.flush()

    # Convert to HTML for the gr.HTML component (no legend for inline preview)
    html_output = highlighted_to_html(highlighted, color_map, show_legend=False)

    logger.info("Calling compute_linking_stats...")
    sys.stderr.flush()
    stats = compute_linking_stats(result)
    logger.info("compute_linking_stats done")
    sys.stderr.flush()

    logger.info("Calling progress(1.0, Done!)...")
    sys.stderr.flush()
    try:
        progress(1.0, desc="Done!")
        logger.info("progress(1.0) returned successfully")
    except Exception as e:
        logger.error(f"progress(1.0) raised exception: {e}")
    sys.stderr.flush()

    # Ensure all GPU operations are complete before returning
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        logger.info("CUDA synchronized")
        sys.stderr.flush()

    logger.info(
        f"=== run_pipeline RETURNING (run #{_run_counter}, {len(result.get('entities', []))} entities) ==="
    )
    sys.stderr.flush()
    # Final yield with complete results — stay in result mode (text_input hidden)
    yield html_output, stats, result, gr.update(visible=False)


def update_ner_params(ner_choice: str):
    """Show/hide NER-specific parameters based on selection."""
    return {
        spacy_params: gr.update(visible=(ner_choice == "spacy")),
        gliner_params: gr.update(visible=(ner_choice == "gliner")),
        simple_params: gr.update(visible=(ner_choice == "simple")),
    }


def update_cand_params(cand_choice: str):
    """Show/hide candidate-specific parameters based on selection."""
    if cand_choice == "none":
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    show_context = cand_choice in ("lela_dense", "lela_openai_api_dense")
    show_embedding_model = cand_choice in ("lela_dense", "lela_openai_api_dense")
    show_openai_api_dense_params = cand_choice == "lela_openai_api_dense"

    return (
        gr.update(visible=show_embedding_model),
        gr.update(visible=show_context),
        gr.update(visible=show_openai_api_dense_params),
    )


def update_reranker_params(reranker_choice: str):
    """Show/hide reranker-specific parameters based on selection."""
    show_cross_encoder_model = reranker_choice in (
        "lela_cross_encoder",
        "lela_cross_encoder_vllm",
    )
    show_embedding_model = reranker_choice in (
        "lela_embedder_transformers",
        "lela_embedder_vllm",
    )
    show_lela_vllm_api_client = reranker_choice in ("lela_vllm_api_client", "lela_llama_server")
    # Use different model lists for vLLM vs transformers cross-encoder
    if reranker_choice == "lela_cross_encoder_vllm":
        ce_choices = [(m[1], m[0]) for m in VLLM_RERANKER_MODEL_CHOICES]
        ce_default = DEFAULT_VLLM_RERANKER_MODEL
    else:
        ce_choices = [(m[1], m[0]) for m in CROSS_ENCODER_MODEL_CHOICES]
        ce_default = "tomaarsen/Qwen3-Reranker-4B-seq-cls"
    return (
        gr.update(
            visible=show_cross_encoder_model, choices=ce_choices, value=ce_default
        ),
        gr.update(visible=show_embedding_model),
        gr.update(visible=show_lela_vllm_api_client),
    )


def update_disambig_params(disambig_choice: str):
    """Show/hide disambiguator-specific parameters based on selection."""
    show_llm = disambig_choice in ("lela_vllm", "lela_transformers")
    show_openai_api = disambig_choice == "lela_openai_api"
    return (
        gr.update(visible=show_llm),
        gr.update(visible=show_llm),
        gr.update(visible=show_openai_api),
    )


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


def compute_memory_estimate(
    ner_type: str,
    gliner_model: str,
    cand_type: str,
    reranker_type: str,
    disambig_type: str,
    llm_model: str,
) -> str:
    """Compute and format memory estimate for current configuration."""
    from el_pipeline.lela.config import VLLM_GPU_MEMORY_UTILIZATION
    from el_pipeline.lela.llm_pool import get_cached_models_info
    from el_pipeline.knowledge_bases.custom import get_kb_cache_info

    try:
        resources = get_system_resources()

        lines = []
        if resources.gpu_available:
            lines.append(f"**GPU:** {resources.gpu_name}")
            lines.append(
                f"**VRAM:** {resources.gpu_vram_free_gb:.1f}GB free / {resources.gpu_vram_total_gb:.1f}GB total"
            )

            # Show allocatable memory (considering vLLM's memory fraction)
            allocatable = resources.gpu_vram_free_gb * VLLM_GPU_MEMORY_UTILIZATION
            lines.append(
                f"**Allocatable:** ~{allocatable:.1f}GB ({VLLM_GPU_MEMORY_UTILIZATION*100:.0f}% of free)"
            )
        else:
            lines.append("**GPU:** Not available")

        # Show cached models
        cached_info = get_cached_models_info()
        cached_models = []
        for st_info in cached_info.get("sentence_transformers", []):
            model_name = st_info["key"].split(":")[0].split("/")[-1]
            status = "in use" if st_info["in_use"] else "cached"
            cached_models.append(f"{model_name} ({status})")
        for vllm_info in cached_info.get("vllm", []):
            model_name = vllm_info["key"].split(":")[0].split("/")[-1]
            status = "in use" if vllm_info["in_use"] else "cached"
            cached_models.append(f"{model_name} ({status})")

        if cached_models:
            lines.append(f"**Cached models:** {', '.join(cached_models)}")

        # Show cached KBs
        kb_cache_info = get_kb_cache_info()
        if kb_cache_info:
            kb_names = [
                f"{Path(kb['path']).name} ({kb['entity_count']:,} entities)"
                for kb in kb_cache_info
            ]
            lines.append(f"**Cached KBs:** {', '.join(kb_names)}")

        return "\n".join(lines)

    except Exception as e:
        return f"*Could not estimate memory: {e}*"


_run_counter = 0


def clear_outputs_for_new_run():
    """Clear outputs and log when a new run starts."""
    global _run_counter
    _run_counter += 1
    # Clear cancellation flag for new run
    _cancel_event.clear()
    logger = logging.getLogger(__name__)
    logger.info(f"=== BUTTON CLICKED - Starting run #{_run_counter} ===")
    import sys

    sys.stderr.flush()
    # Return: preview_html (cleared + shown), stats, json, run_btn, cancel_btn,
    #         text_input (hidden), edit_btn (label→"Edit"), view_mode (→"preview")
    return (
        gr.update(value="", visible=True),  # preview_html: clear + show (result mode)
        "*Processing...*",
        None,
        gr.update(visible=False),  # run_btn hidden
        gr.update(visible=True),  # cancel_btn shown
        gr.update(visible=False),  # text_input hidden (result mode)
        gr.update(value="✏️ Edit"),  # edit_btn label
        "preview",  # view_mode
    )


def restore_buttons_after_run():
    """Restore button visibility after pipeline completes or is cancelled."""
    # Show Run button, hide Cancel button (reset text and make interactive again)
    return gr.update(visible=True), gr.update(
        visible=False, value="Cancel", interactive=True
    )


def start_cancellation():
    """Called when cancel button is clicked. Set flag and show cancelling state."""
    logger = logging.getLogger(__name__)
    logger.info("=== CANCEL BUTTON CLICKED ===")
    # Set the cancellation flag - pipeline will check this and stop
    _cancel_event.set()
    # Update button to show "Cancelling..." with loading state (non-interactive)
    return gr.update(value="Cancelling...", interactive=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EL Pipeline Gradio UI")
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to run the server on"
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a public share link"
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log)
    logger = logging.getLogger(__name__)

    # Silence noisy loggers
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("numba.core").setLevel(logging.WARNING)

    # Log vLLM availability status
    vllm_ok = _is_vllm_usable()
    logger.info(f"vLLM disambiguator available: {vllm_ok}")

    components = get_available_components()

    # Custom CSS for cleaner design
    custom_css = """
    .main-header {
        margin-bottom: 0.25rem;
    }
    .subtitle {
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    .config-row {
        align-items: flex-start !important;
    }
    .output-section {
        min-height: 300px;
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid var(--border-color-primary);
        border-radius: var(--radius-lg);
        padding: 1em;
    }
    .run-button {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    #edit-preview-btn {
        width: 100px !important;
        max-width: 100px !important;
        flex: none !important;
    }
    #upload-file-btn {
        width: 120px !important;
        max-width: 120px !important;
        flex: none !important;
    }
    """

    # JavaScript to cancel pipeline when tab is closed + drag-and-drop support
    custom_head = """
    <script>
    (function() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', init);
        } else {
            init();
        }

        function init() {
            setupPageCloseHandler();
            setupTextAreaDrop();
        }

        function setupPageCloseHandler() {
            console.log("EL Pipeline: Page close handler initialized");

            function findCancelButton() {
                const buttons = document.querySelectorAll('button');
                for (const btn of buttons) {
                    const text = btn.textContent.trim();
                    if (text === 'Cancel' && btn.offsetParent !== null) {
                        const style = window.getComputedStyle(btn);
                        if (style.display !== 'none' && style.visibility !== 'hidden') {
                            return btn;
                        }
                    }
                }
                return null;
            }

            window.addEventListener('pagehide', function() {
                const cancelBtn = findCancelButton();
                if (cancelBtn) {
                    cancelBtn.click();
                    console.log("EL Pipeline: Triggered cancellation on pagehide");
                }
            }, { capture: true });

            window.addEventListener('beforeunload', function() {
                const cancelBtn = findCancelButton();
                if (cancelBtn) {
                    cancelBtn.click();
                    console.log("EL Pipeline: Triggered cancellation on beforeunload");
                }
            }, { capture: true });
        }

        function setupTextAreaDrop() {
            // Use event delegation on document so listeners survive Gradio DOM rebuilds
            document.addEventListener('dragover', function(e) {
                var el = e.target.closest('#main-text-input');
                if (!el) return;
                e.preventDefault();
                e.stopPropagation();
                el.style.outline = '2px dashed #2563EB';
                el.style.outlineOffset = '-2px';
            });
            document.addEventListener('dragleave', function(e) {
                var el = e.target.closest('#main-text-input');
                if (!el) return;
                e.preventDefault();
                el.style.outline = '';
                el.style.outlineOffset = '';
            });
            document.addEventListener('drop', function(e) {
                var el = e.target.closest('#main-text-input');
                if (!el) return;
                e.preventDefault();
                e.stopPropagation();
                el.style.outline = '';
                el.style.outlineOffset = '';
                var file = e.dataTransfer.files[0];
                if (!file) return;

                if (file.name.endsWith('.txt')) {
                    var reader = new FileReader();
                    reader.onload = function() {
                        var ta = el.querySelector('textarea');
                        if (ta) {
                            var nativeSet = Object.getOwnPropertyDescriptor(
                                window.HTMLTextAreaElement.prototype, "value").set;
                            nativeSet.call(ta, reader.result);
                            ta.dispatchEvent(new Event('input', { bubbles: true }));
                        }
                    };
                    reader.readAsText(file);
                } else {
                    var uploadInput = document.querySelector('#upload-file-btn input[type="file"]');
                    if (uploadInput) {
                        var dt = new DataTransfer();
                        dt.items.add(file);
                        uploadInput.files = dt.files;
                        uploadInput.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                }
            });
            console.log("EL Pipeline: Drag-and-drop initialized on text area");
        }
    })();
    </script>
    """

    with gr.Blocks(title="EL Pipeline 🔗", fill_height=True, head=custom_head) as demo:
        gr.Markdown("# EL Pipeline 🔗", elem_classes=["main-header"])
        gr.Markdown(
            DESCRIPTION,
            elem_classes=["subtitle"],
        )

        with gr.Tabs():
            # ===== MAIN PIPELINE TAB =====
            with gr.Tab("Pipeline"):
                # --- INPUT SECTION ---
                with gr.Row():
                    with gr.Column(scale=2):
                        # Header row with label, Edit button, and Upload button
                        with gr.Row():
                            gr.HTML(
                                "<span style='font-weight:600;font-size:1.1em;line-height:2.4;'>Input text</span>"
                            )
                            edit_btn = gr.Button(
                                "👁 Preview",
                                size="sm",
                                variant="secondary",
                                scale=0,
                                min_width=40,
                                elem_id="edit-preview-btn",
                            )
                            upload_btn = gr.UploadButton(
                                "📄 Upload",
                                size="sm",
                                variant="secondary",
                                scale=0,
                                min_width=100,
                                file_types=[".txt", ".pdf", ".docx", ".html"],
                                elem_id="upload-file-btn",
                            )
                        # Edit mode: Textbox (visible by default)
                        text_input = gr.Textbox(
                            placeholder="Enter or paste text, or drop a file...",
                            lines=12,
                            show_label=False,
                            elem_id="main-text-input",
                            value="Albert Einstein was born in Germany. Marie Curie was a pioneering scientist.",
                        )
                        # Result mode: HTML output (hidden by default)
                        preview_html = gr.HTML(
                            visible=False,
                            elem_id="main-result-output",
                            elem_classes=["output-section"],
                        )

                        # Stats + JSON as collapsible section below
                        with gr.Accordion("Stats / JSON", open=False):
                            with gr.Tabs():
                                with gr.Tab("Stats"):
                                    stats_output = gr.Markdown(
                                        "*Run the pipeline to see statistics.*"
                                    )
                                with gr.Tab("JSON"):
                                    json_output = gr.JSON(label="Pipeline Output")

                    with gr.Column(scale=1, min_width=200):
                        kb_file = gr.File(
                            label="Knowledge Base (JSONL) — optional, defaults to YAGO 4.5",
                            file_types=[".jsonl"],
                            value=None,
                        )
                        run_btn = gr.Button(
                            "Run Pipeline",
                            variant="primary",
                            size="lg",
                            elem_classes=["run-button"],
                        )
                        cancel_btn = gr.Button(
                            "Cancel",
                            variant="stop",
                            size="lg",
                            visible=False,
                            elem_classes=["run-button"],
                        )

                # Hidden file component for internal file tracking (used by pipeline)
                file_input = gr.File(visible=False)
                # State: "edit" or "preview" — tracks which mode the text area is in
                view_mode = gr.State("edit")

                # --- CONFIGURATION SECTION (Horizontal Layout) ---
                gr.Markdown("### Configuration")

                # GPU memory info (left-aligned, above components)
                memory_estimate_display = gr.Markdown(
                    value="*Detecting GPU...*",
                    elem_id="memory-estimate",
                )

                with gr.Row(equal_height=False, elem_classes=["config-row"]):
                    # NER Column
                    with gr.Column(scale=1, min_width=200):
                        gr.Markdown("**NER**")
                        ner_type = gr.Dropdown(
                            choices=components["ner"],
                            value="simple",
                            label="Model",
                            container=False,
                        )
                        with gr.Group(visible=False) as spacy_params:
                            spacy_model = gr.Textbox(
                                label="SpaCy Model",
                                value="en_core_web_sm",
                                scale=1,
                            )
                        with gr.Group(visible=False) as gliner_params:
                            gliner_model = gr.Textbox(
                                label="GLiNER Model",
                                value=DEFAULT_GLINER_MODEL,
                            )
                            gliner_labels = gr.Textbox(
                                label="Labels (comma-sep)",
                                value="person, organization, location",
                            )
                            gliner_threshold = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.5,
                                step=0.05,
                                label="Threshold",
                            )
                        with gr.Group(visible=True) as simple_params:
                            simple_min_len = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=3,
                                step=1,
                                label="Min Length",
                            )

                    # Candidates Column
                    with gr.Column(scale=1, min_width=200):
                        gr.Markdown("**Candidates**")
                        cand_type = gr.Dropdown(
                            choices=components["candidates"],
                            value="fuzzy",
                            label="Generator",
                            container=False,
                        )
                        # Embedding model selection for dense candidates
                        embedding_model_choices = [
                            (m[1], m[0]) for m in EMBEDDING_MODEL_CHOICES
                        ]
                        cand_embedding_model = gr.Dropdown(
                            choices=embedding_model_choices,
                            value="Qwen/Qwen3-Embedding-4B",
                            label="Embedding Model",
                            visible=False,
                        )
                        with gr.Group(
                            visible=False
                        ) as lela_openai_api_dense_cand_params:
                            cand_api_base_url = gr.Textbox(
                                label="Cand. OpenAI API Base URL",
                                value="http://localhost:8000/v1",
                            )
                            cand_api_key = gr.Textbox(
                                label="Cand. OpenAI API Key",
                                value="",
                                type="password",
                            )
                        cand_top_k = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=64,
                            step=1,
                            label="Top K",
                        )
                        cand_use_context = gr.Checkbox(
                            label="Use Context",
                            value=False,
                            visible=False,
                        )

                    # Reranking Column
                    with gr.Column(scale=1, min_width=200):
                        gr.Markdown("**Reranking**")
                        reranker_type = gr.Dropdown(
                            choices=components["rerankers"],
                            value="none",
                            label="Reranker",
                            container=False,
                        )
                        # Cross-encoder model selection
                        cross_encoder_model_choices = [
                            (m[1], m[0]) for m in CROSS_ENCODER_MODEL_CHOICES
                        ]
                        reranker_cross_encoder_model = gr.Dropdown(
                            choices=cross_encoder_model_choices,
                            value="tomaarsen/Qwen3-Reranker-4B-seq-cls",
                            label="Cross-Encoder Model",
                            visible=False,
                        )
                        # Embedding model selection for embedder reranker
                        reranker_embedding_model = gr.Dropdown(
                            choices=embedding_model_choices,
                            value="Qwen/Qwen3-Embedding-4B",
                            label="Embedding Model",
                            visible=False,
                        )
                        with gr.Group(visible=False) as lela_vllm_api_client_params:
                            reranker_api_url = gr.Textbox(
                                label="Reranker API URL",
                                value="http://localhost",
                            )
                            reranker_api_port = gr.Number(
                                label="Reranker API Port",
                                value=8000,
                            )
                        reranker_top_k = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=10,
                            step=1,
                            label="Top K",
                        )

                    # Disambiguation Column
                    with gr.Column(scale=1, min_width=200):
                        gr.Markdown("**Disambiguation**")
                        disambig_type = gr.Dropdown(
                            choices=components["disambiguators"],
                            value="first",
                            label="Method",
                            container=False,
                        )
                        # LLM model selection for vLLM-based disambiguators
                        llm_model_choices = [(m[1], m[0]) for m in LLM_MODEL_CHOICES]
                        llm_model = gr.Dropdown(
                            choices=llm_model_choices,
                            value="Qwen/Qwen3-4B",
                            label="LLM Model",
                            visible=False,
                        )
                        with gr.Group(visible=False) as lela_common_params:
                            lela_thinking = gr.Checkbox(
                                label="Reasoning",
                                value=True,
                            )
                            lela_none_candidate = gr.Checkbox(
                                label="'None' Candidate",
                                value=True,
                            )
                        with gr.Group(visible=False) as lela_openai_api_params:
                            disambig_api_base_url = gr.Textbox(
                                label="OpenAI API Base URL",
                                value="http://localhost:8000/v1",
                            )
                            disambig_api_key = gr.Textbox(
                                label="OpenAI API Key",
                                value="",
                                type="password",
                            )
                # Hidden loader and KB type (auto-detected)
                loader_type = gr.Dropdown(
                    choices=components["loaders"],
                    value="text",
                    visible=False,
                )
                kb_type = gr.Dropdown(
                    choices=components["knowledge_bases"],
                    value="custom",
                    visible=False,
                )

                gr.Markdown(LOGO)

            # ===== DOCUMENTATION TAB =====
            with gr.Tab("Help"):
                gr.Markdown("""
## Quick Start

1. **Upload Knowledge Base**: JSONL file with entities (`title`, `description`, optional `id`)
2. **Enter Text or Upload File**: Input text directly or upload a document
3. **Configure Pipeline**: Select components and adjust parameters
4. **Run**: Click "Run Pipeline" to process

---

## Example Files

Test files are available in `data/test/`:
- `sample_kb.jsonl` - Sample knowledge base with 10 entities
- `sample_doc.txt` - Sample document for testing

---

## Pipeline Components

### NER Models
| Name | Description |
|------|-------------|
| **simple** | Regex-based extraction of capitalized phrases |
| **spacy** | SpaCy NER models (requires download) |
| **gliner** | GLiNER zero-shot NER with custom labels |

### Candidate Generators
| Name | Description |
|------|-------------|
| **fuzzy** | Fuzzy string matching |
| **bm25** | BM25 text retrieval |
| **lela_dense** | Dense embedding retrieval |

### Rerankers
| Name | Description |
|------|-------------|
| **none** | No reranking |
| **lela_cross_encoder** | Cross-encoder reranking |
| **lela_embedder** | Embedding-based reranking |

### Disambiguators
| Name | Description |
|------|-------------|
| **none** | No disambiguation |
| **first** | Select first candidate |
| **popularity** | Select by popularity |
| **lela_vllm** | vLLM-based disambiguation |
| **lela_openai_api** | OpenAI-compatible API disambiguation |

---

## spaCy Component Mapping

| Config Name | spaCy Factory |
|-------------|---------------|
| simple | el_pipeline_simple |
| lela_embedder | el_pipeline_lela_embedder_reranker |
| lela_vllm | el_pipeline_lela_vllm_disambiguator |
| lela_openai_api | el_pipeline_lela_openai_api_disambiguator |

---

## Entity Type Legend (SpaCy)

| Label | Meaning | Example |
|-------|---------|---------|
| **PERSON** | People, including fictional | *Albert Einstein*, *Marie Curie* |
| **ORG** | Organizations, companies, agencies | *Google*, *United Nations*, *NASA* |
| **GPE** | Geopolitical entities | *France*, *New York*, *California* |
| **LOC** | Non-GPE locations | *Mount Everest*, *Pacific Ocean* |
| **FAC** | Facilities | *Empire State Building*, *JFK Airport* |
| **PRODUCT** | Objects, vehicles, foods | *iPhone*, *Boeing 747* |
| **EVENT** | Named events | *World War II*, *Hurricane Katrina* |
| **WORK_OF_ART** | Creative works | *The Great Gatsby*, *Mona Lisa* |
| **LAW** | Legal documents | *Roe v. Wade*, *GDPR* |
| **LANGUAGE** | Languages | *English*, *Mandarin* |
| **DATE** | Dates/periods | *January 2020*, *next week* |
| **TIME** | Times | *3:00 PM*, *morning* |
| **PERCENT** | Percentages | *50%*, *ten percent* |
| **MONEY** | Monetary values | *$100*, *€50 million* |
| **QUANTITY** | Measurements | *10 kg*, *five miles* |
| **ORDINAL** | Ordinal numbers | *first*, *3rd* |
| **CARDINAL** | Cardinal numbers | *one*, *100*, *millions* |
| **NORP** | Nationalities/groups | *American*, *Buddhist* |
| **ENT** | Generic entity | Any capitalized phrase |
                """)

        # --- EVENT HANDLERS ---

        # Edit/Preview toggle button
        def toggle_edit_preview(current_mode, current_preview):
            if current_mode == "edit":
                # Switch to preview mode
                if current_preview:
                    html = current_preview
                else:
                    html = "<div style='color:#6B7280;padding:1em;'>Run the pipeline to see results here.</div>"
                return (
                    gr.update(visible=False),  # text_input
                    gr.update(value=html, visible=True),  # preview_html
                    gr.update(value="✏️ Edit"),  # edit_btn label
                    "preview",  # view_mode
                )
            else:
                # Switch to edit mode
                return (
                    gr.update(visible=True),  # text_input
                    gr.update(visible=False),  # preview_html
                    gr.update(value="👁 Preview"),  # edit_btn label
                    "edit",  # view_mode
                )

        edit_btn.click(
            fn=toggle_edit_preview,
            inputs=[view_mode, preview_html],
            outputs=[text_input, preview_html, edit_btn, view_mode],
        )

        # Upload button → load file, populate textbox or store file ref
        def handle_file_upload(file):
            if not file:
                return gr.update(), gr.update(), gr.update()
            ext = Path(file.name).suffix.lower()
            loader_map = {
                ".txt": "text",
                ".pdf": "pdf",
                ".docx": "docx",
                ".html": "html",
                ".htm": "html",
            }
            new_loader = loader_map.get(ext, "text")

            # For text files, read content into the textbox
            if ext == ".txt":
                try:
                    content = Path(file.name).read_text(encoding="utf-8")
                    return content, gr.update(value=new_loader), file
                except Exception:
                    pass
            # For non-text files, just store the file reference
            return gr.update(), gr.update(value=new_loader), file

        upload_btn.upload(
            fn=handle_file_upload,
            inputs=[upload_btn],
            outputs=[text_input, loader_type, file_input],
        )

        ner_type.change(
            fn=update_ner_params,
            inputs=[ner_type],
            outputs=[spacy_params, gliner_params, simple_params],
        )

        cand_type.change(
            fn=update_cand_params,
            inputs=[cand_type],
            outputs=[
                cand_embedding_model,
                cand_use_context,
                lela_openai_api_dense_cand_params,
            ],
        )

        reranker_type.change(
            fn=update_reranker_params,
            inputs=[reranker_type],
            outputs=[
                reranker_cross_encoder_model,
                reranker_embedding_model,
                lela_vllm_api_client_params,
            ],
        )

        disambig_type.change(
            fn=update_disambig_params,
            inputs=[disambig_type],
            outputs=[llm_model, lela_common_params, lela_openai_api_params],
        )

        # Memory estimate updates
        memory_inputs = [
            ner_type,
            gliner_model,
            cand_type,
            reranker_type,
            disambig_type,
            llm_model,
        ]

        for component in [ner_type, cand_type, reranker_type, disambig_type, llm_model]:
            component.change(
                fn=compute_memory_estimate,
                inputs=memory_inputs,
                outputs=[memory_estimate_display],
            )

        # Initial memory estimate on load
        demo.load(
            fn=compute_memory_estimate,
            inputs=memory_inputs,
            outputs=[memory_estimate_display],
        )

        # Chain: clear outputs → run pipeline → restore buttons
        run_event = (
            run_btn.click(
                fn=clear_outputs_for_new_run,
                inputs=None,
                outputs=[
                    preview_html,
                    stats_output,
                    json_output,
                    run_btn,
                    cancel_btn,
                    text_input,
                    edit_btn,
                    view_mode,
                ],
            )
            .then(
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
                    cand_embedding_model,
                    cand_top_k,
                    cand_use_context,
                    cand_api_base_url,
                    cand_api_key,
                    reranker_type,
                    reranker_embedding_model,
                    reranker_cross_encoder_model,
                    reranker_api_url,
                    reranker_api_port,
                    reranker_top_k,
                    disambig_type,
                    llm_model,
                    lela_thinking,
                    lela_none_candidate,
                    disambig_api_base_url,
                    disambig_api_key,
                    kb_type,
                ],
                outputs=[preview_html, stats_output, json_output, text_input],
            )
            .then(
                fn=restore_buttons_after_run,
                inputs=None,
                outputs=[run_btn, cancel_btn],
            )
        )

        # Cancel button: show "Cancelling..." and set flag
        # The pipeline generator checks the flag and exits, then restore_buttons_after_run runs
        cancel_btn.click(
            fn=start_cancellation,
            inputs=None,
            outputs=[cancel_btn],
        )

    logger.info(f"Launching Gradio UI on port {args.port}...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        css=custom_css,
        theme=gr.themes.Default(primary_hue="green"),
    )
