import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass
class ComponentConfig:
    """Generic component configuration."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    loader: ComponentConfig
    ner: ComponentConfig
    candidate_generator: ComponentConfig
    reranker: Optional[ComponentConfig] = None
    disambiguator: Optional[ComponentConfig] = None
    knowledge_base: Optional[ComponentConfig] = None
    cache_dir: str = ".ner_cache"
    batch_size: int = 1

    @staticmethod
    def from_json(path: Union[str, Path]) -> "PipelineConfig":
        """Create a PipelineConfig from a JSON file path."""
        with open(path) as f:
            return PipelineConfig.from_dict(json.load(f))

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PipelineConfig":
        def build(section: str) -> Optional[ComponentConfig]:
            if section not in data or data[section] is None:
                return None
            entry = data[section]
            return ComponentConfig(name=entry["name"], params=entry.get("params", {}))

        loader = build("loader") or ComponentConfig(name="text")

        return PipelineConfig(
            loader=loader,
            ner=build("ner"),  # type: ignore[arg-type]
            candidate_generator=build("candidate_generator"),  # type: ignore[arg-type]
            reranker=build("reranker"),
            disambiguator=build("disambiguator"),
            knowledge_base=build("knowledge_base"),
            cache_dir=data.get("cache_dir", ".ner_cache"),
            batch_size=data.get("batch_size", 1),
        )

