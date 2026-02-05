"""Shared fixtures for NER pipeline tests."""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import pytest

from ner_pipeline.types import Candidate, Document, Entity, Mention, ResolvedMention


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_text() -> str:
    """Sample text containing named entities."""
    return (
        "Barack Obama was the 44th President of the United States. "
        "He was born in Honolulu, Hawaii. "
        "Obama graduated from Columbia University and Harvard Law School."
    )


@pytest.fixture
def sample_document(sample_text: str) -> Document:
    """Sample Document instance."""
    return Document(id="doc-001", text=sample_text, meta={"source": "test"})


@pytest.fixture
def sample_mentions() -> List[Mention]:
    """Sample mentions as would be extracted by NER."""
    return [
        Mention(start=0, end=12, text="Barack Obama", label="PERSON", context="Barack Obama was the 44th President"),
        Mention(start=45, end=58, text="United States", label="GPE", context="President of the United States"),
        Mention(start=80, end=88, text="Honolulu", label="GPE", context="born in Honolulu, Hawaii"),
        Mention(start=90, end=96, text="Hawaii", label="GPE", context="born in Honolulu, Hawaii"),
    ]


@pytest.fixture
def sample_entities() -> List[Entity]:
    """Sample Entity records."""
    return [
        Entity(id="Q76", title="Barack Obama", description="44th President of the United States"),
        Entity(id="Q30", title="United States", description="Country in North America"),
        Entity(id="Q18094", title="Honolulu", description="Capital city of Hawaii"),
        Entity(id="Q782", title="Hawaii", description="U.S. state in the Pacific Ocean"),
        Entity(id="Q49088", title="Columbia University", description="Private university in New York City"),
        Entity(id="Q13371", title="Harvard Law School", description="Law school of Harvard University"),
    ]


@pytest.fixture
def sample_candidates(sample_entities: List[Entity]) -> List[Candidate]:
    """Sample candidates for a mention."""
    return [
        Candidate(entity_id="Q76", score=0.95, description="44th President of the United States"),
        Candidate(entity_id="Q123456", score=0.65, description="Some other Barack Obama"),
        Candidate(entity_id="Q789", score=0.45, description="Yet another candidate"),
    ]


# ---------------------------------------------------------------------------
# Mock classes
# ---------------------------------------------------------------------------


class MockKnowledgeBase:
    """In-memory mock knowledge base for testing."""

    def __init__(self, entities: Optional[List[Entity]] = None):
        self._entities: Dict[str, Entity] = {}
        self._by_title: Dict[str, Entity] = {}
        if entities:
            for e in entities:
                self._entities[e.id] = e
                self._by_title[e.title] = e

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        # Try by ID first, then by title (for LELA compatibility)
        result = self._entities.get(entity_id)
        if result is None:
            result = self._by_title.get(entity_id)
        return result

    def search(self, query: str, top_k: int = 10) -> List[Entity]:
        # Simple substring match for testing
        results = []
        query_lower = query.lower()
        for e in self._entities.values():
            if query_lower in e.title.lower():
                results.append(e)
            elif e.description and query_lower in e.description.lower():
                results.append(e)
        return results[:top_k]

    def all_entities(self) -> Iterable[Entity]:
        return self._entities.values()

    def add_entity(self, entity: Entity) -> None:
        self._entities[entity.id] = entity


class MockNERModel:
    """Mock NER model that returns predefined mentions."""

    def __init__(self, mentions: Optional[List[Mention]] = None):
        self._mentions = mentions or []

    def extract(self, text: str) -> List[Mention]:
        return self._mentions

    def set_mentions(self, mentions: List[Mention]) -> None:
        self._mentions = mentions


class MockCandidateGenerator:
    """Mock candidate generator that returns predefined candidates."""

    def __init__(self, candidates: Optional[List[Candidate]] = None):
        self._candidates = candidates or []

    def generate(self, mention: Mention, doc: Document) -> List[Candidate]:
        return self._candidates

    def set_candidates(self, candidates: List[Candidate]) -> None:
        self._candidates = candidates


class MockDocumentLoader:
    """Mock document loader that returns predefined documents."""

    def __init__(self, documents: Optional[List[Document]] = None):
        self._documents = documents or []

    def load(self, path: str) -> Iterator[Document]:
        for doc in self._documents:
            yield doc

    def set_documents(self, documents: List[Document]) -> None:
        self._documents = documents


# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_kb(sample_entities: List[Entity]) -> MockKnowledgeBase:
    """Mock knowledge base populated with sample entities."""
    return MockKnowledgeBase(sample_entities)


@pytest.fixture
def mock_ner(sample_mentions: List[Mention]) -> MockNERModel:
    """Mock NER model populated with sample mentions."""
    return MockNERModel(sample_mentions)


@pytest.fixture
def mock_candidate_generator(sample_candidates: List[Candidate]) -> MockCandidateGenerator:
    """Mock candidate generator populated with sample candidates."""
    return MockCandidateGenerator(sample_candidates)


@pytest.fixture
def mock_loader(sample_document: Document) -> MockDocumentLoader:
    """Mock document loader populated with sample document."""
    return MockDocumentLoader([sample_document])


# ---------------------------------------------------------------------------
# Temporary file fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_text_file(sample_text: str) -> Iterator[str]:
    """Create a temporary text file with sample content."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(sample_text)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def temp_jsonl_kb(sample_entities: List[Entity]) -> Iterator[str]:
    """Create a temporary JSONL knowledge base file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for entity in sample_entities:
            line = json.dumps({
                "id": entity.id,
                "title": entity.title,
                "description": entity.description,
                **(entity.metadata or {}),
            })
            f.write(line + "\n")
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def temp_cache_dir() -> Iterator[str]:
    """Create a temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_config_dict(temp_jsonl_kb: str, temp_cache_dir: str) -> Dict:
    """Minimal config dict for lightweight pipeline testing."""
    return {
        "loader": {
            "name": "text",
            "params": {},
        },
        "ner": {
            "name": "simple",
            "params": {"min_len": 3},
        },
        "candidate_generator": {
            "name": "fuzzy",
            "params": {"top_k": 5},
        },
        "disambiguator": {
            "name": "first",
            "params": {},
        },
        "knowledge_base": {
            "name": "custom",
            "params": {"path": temp_jsonl_kb},
        },
        "cache_dir": temp_cache_dir,
        "batch_size": 1,
    }


# ---------------------------------------------------------------------------
# Test data path fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_data_dir() -> Path:
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_kb_path(test_data_dir: Path) -> Path:
    """Path to sample KB JSONL file."""
    return test_data_dir / "sample_kb.jsonl"


# ---------------------------------------------------------------------------
# App/CLI test fixtures
# ---------------------------------------------------------------------------


class MockGradioFile:
    """Mock Gradio file object."""

    def __init__(self, path: str):
        self.name = path


class MockGradioProgress:
    """Mock Gradio progress callback."""

    def __init__(self):
        self.calls: List[Tuple] = []

    def __call__(self, progress: float, desc: str = ""):
        self.calls.append((progress, desc))


@pytest.fixture
def mock_gradio_file(temp_text_file: str) -> MockGradioFile:
    """Mock Gradio file object pointing to temp text file."""
    return MockGradioFile(temp_text_file)


@pytest.fixture
def mock_kb_file(temp_jsonl_kb: str) -> MockGradioFile:
    """Mock Gradio file object pointing to temp KB file."""
    return MockGradioFile(temp_jsonl_kb)


@pytest.fixture
def mock_progress() -> MockGradioProgress:
    """Mock Gradio progress callback."""
    return MockGradioProgress()


@pytest.fixture
def sample_pipeline_result() -> Dict:
    """Sample result for format_highlighted_text testing."""
    return {
        "text": "Barack Obama was president.",
        "entities": [
            {
                "text": "Barack Obama",
                "start": 0,
                "end": 12,
                "label": "PERSON",
                "entity_title": "Barack Obama",
                "entity_id": "Q76",
                "candidates": [],
            }
        ],
    }


@pytest.fixture
def temp_config_file(minimal_config_dict: Dict) -> Iterator[str]:
    """Temporary config JSON file for CLI testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(minimal_config_dict, f)
        path = f.name
    yield path
    os.unlink(path)
