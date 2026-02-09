# Testing Guide

This guide documents the test infrastructure, how to run tests, and how to write new tests for the EL Pipeline.

## Table of Contents

- [Quick Start](#quick-start)
- [Test Categories](#test-categories)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Writing Tests](#writing-tests)
- [Fixtures Reference](#fixtures-reference)
- [Coverage Reports](#coverage-reports)

---

## Quick Start

```bash
# Run all unit tests (fast)
pytest tests/unit

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=el_pipeline --cov-report=html

# Run specific test file
pytest tests/unit/test_types.py

# Run specific test
pytest tests/unit/test_types.py::test_document_creation

# Run with verbose output
pytest tests/ -v
```

---

## Test Categories

Tests are organized by markers defined in `pyproject.toml`:

| Marker | Description | Speed | Requirements |
|--------|-------------|-------|--------------|
| (none) | Unit tests | Fast | None |
| `slow` | Tests requiring model downloads | Slow | Internet, disk space |
| `integration` | Full pipeline integration tests | Medium | Some components |
| `requires_spacy` | Tests requiring spaCy models | Medium | spaCy model downloaded |
| `requires_transformers` | Tests requiring HuggingFace | Slow | GPU recommended |
| `requires_sentence_transformers` | Tests requiring sentence-transformers | Slow | GPU recommended |

### Running by Category

```bash
# Skip slow tests
pytest tests/ -m "not slow"

# Run only integration tests
pytest tests/ -m integration

# Skip tests requiring external models
pytest tests/ -m "not (slow or requires_spacy or requires_transformers)"

# Run only fast unit tests
pytest tests/unit -m "not slow"
```

---

## Test Structure

```
tests/
├── conftest.py           # Shared fixtures
├── unit/                 # Unit tests (fast, isolated)
│   ├── test_types.py     # Data type tests
│   ├── test_config.py    # Configuration tests
│   ├── test_loaders.py   # Loader tests
│   └── test_context.py   # Context extraction tests
├── integration/          # Integration tests
│   └── test_pipeline.py  # Full pipeline tests
└── data/                 # Test data files
    └── sample_kb.jsonl   # Sample knowledge base
```

---

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest tests/

# Run with pytest verbose output
pytest tests/ -v

# Run with short traceback
pytest tests/ --tb=short

# Run with full traceback
pytest tests/ --tb=long

# Stop on first failure
pytest tests/ -x

# Run last failed tests
pytest tests/ --lf
```

### Filtering Tests

```bash
# By file pattern
pytest tests/unit/test_load*.py

# By test name pattern
pytest tests/ -k "loader"

# By marker
pytest tests/ -m slow

# Exclude marker
pytest tests/ -m "not slow"

# Combine filters
pytest tests/ -k "test_document" -m "not integration"
```

### Parallel Execution

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
pytest tests/ -n auto
pytest tests/ -n 4  # Use 4 workers
```

---

## Writing Tests

### Basic Test Structure

```python
"""Tests for the document loader module."""

import pytest
from el_pipeline.types import Document
from el_pipeline.loaders.text import TextLoader


class TestTextLoader:
    """Tests for TextLoader class."""

    def test_load_simple_file(self, temp_text_file):
        """Test loading a simple text file."""
        loader = TextLoader()
        docs = list(loader.load(temp_text_file))

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].text == "Sample text content"

    def test_load_empty_file(self, tmp_path):
        """Test loading an empty file."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        loader = TextLoader()
        docs = list(loader.load(str(empty_file)))

        assert len(docs) == 1
        assert docs[0].text == ""


def test_document_creation():
    """Test Document dataclass creation."""
    doc = Document(id="test-1", text="Hello world", meta={"source": "test"})

    assert doc.id == "test-1"
    assert doc.text == "Hello world"
    assert doc.meta["source"] == "test"
```

### Using Markers

```python
import pytest


@pytest.mark.slow
def test_gliner_model_loading():
    """Test that requires downloading GLiNER model."""
    from gliner import GLiNER
    model = GLiNER.from_pretrained("numind/NuNER_Zero-span")
    assert model is not None


@pytest.mark.integration
def test_full_pipeline():
    """Integration test for the full pipeline."""
    # ...


@pytest.mark.requires_spacy
def test_spacy_ner():
    """Test that requires spaCy model."""
    import spacy
    nlp = spacy.load("en_core_web_sm")
    # ...


@pytest.mark.parametrize("min_len,expected", [
    (1, 5),
    (3, 3),
    (10, 0),
])
def test_simple_ner_min_length(min_len, expected):
    """Test simple NER with different min_len settings."""
    # ...
```

### Using Fixtures

```python
def test_with_sample_document(sample_document):
    """Test using the sample_document fixture."""
    assert sample_document.id == "doc-001"
    assert "Barack Obama" in sample_document.text


def test_with_mock_kb(mock_kb, sample_entities):
    """Test using mock knowledge base."""
    entity = mock_kb.get_entity("Q76")
    assert entity.title == "Barack Obama"

    results = mock_kb.search("Einstein")
    assert len(results) > 0


def test_with_temp_files(temp_text_file, temp_jsonl_kb):
    """Test using temporary file fixtures."""
    # temp_text_file and temp_jsonl_kb are paths to temporary files
    # that are automatically cleaned up after the test
    pass
```

### Testing spaCy Components

```python
import pytest
import spacy
from el_pipeline import spacy_components


@pytest.fixture
def nlp_with_simple_ner():
    """Create a spaCy pipeline with simple NER."""
    nlp = spacy.blank("en")
    nlp.add_pipe("el_pipeline_simple", config={"min_len": 3})
    return nlp


def test_simple_ner_extracts_entities(nlp_with_simple_ner):
    """Test that simple NER extracts capitalized entities."""
    doc = nlp_with_simple_ner("Albert Einstein was born in Germany.")

    entities = [ent.text for ent in doc.ents]
    assert "Albert Einstein" in entities
    assert "Germany" in entities


def test_simple_ner_context_extension(nlp_with_simple_ner):
    """Test that context extension is populated."""
    doc = nlp_with_simple_ner("Albert Einstein was born in Germany.")

    for ent in doc.ents:
        assert ent._.context is not None
        assert ent.text in ent._.context
```

### Testing with Mocks

```python
from unittest.mock import Mock, patch
from el_pipeline.pipeline import ELPipeline


def test_pipeline_with_mock_kb(minimal_config_dict):
    """Test pipeline with mocked knowledge base."""
    with patch("el_pipeline.pipeline.knowledge_bases") as mock_registry:
        mock_kb = Mock()
        mock_kb.get_entity.return_value = None
        mock_registry.get.return_value = lambda **kwargs: mock_kb

        # Create pipeline with mocked KB
        # ...


def test_vllm_disambiguator_output_parsing():
    """Test LLM output parsing without running actual model."""
    from el_pipeline.spacy_components.disambiguators import LELAvLLMDisambiguatorComponent

    # Test the static parsing method
    assert LELAvLLMDisambiguatorComponent._parse_output('"answer": 3') == 3
    assert LELAvLLMDisambiguatorComponent._parse_output('answer: 2') == 2
    assert LELAvLLMDisambiguatorComponent._parse_output('5') == 5
    assert LELAvLLMDisambiguatorComponent._parse_output('invalid') == 0
```

---

## Fixtures Reference

Fixtures defined in `tests/conftest.py`:

### Sample Data Fixtures

| Fixture | Type | Description |
|---------|------|-------------|
| `sample_text` | `str` | Sample text with named entities |
| `sample_document` | `Document` | Sample Document instance |
| `sample_mentions` | `List[Mention]` | Sample extracted mentions |
| `sample_entities` | `List[Entity]` | Sample KB entities |
| `sample_candidates` | `List[Candidate]` | Sample candidates for a mention |

### Mock Fixtures

| Fixture | Type | Description |
|---------|------|-------------|
| `mock_kb` | `MockKnowledgeBase` | In-memory mock KB |
| `mock_ner` | `MockNERModel` | Mock NER model |
| `mock_candidate_generator` | `MockCandidateGenerator` | Mock candidate generator |
| `mock_loader` | `MockDocumentLoader` | Mock document loader |

### File Fixtures

| Fixture | Type | Description |
|---------|------|-------------|
| `temp_text_file` | `str` | Path to temporary text file |
| `temp_jsonl_kb` | `str` | Path to temporary JSONL KB |
| `temp_cache_dir` | `str` | Path to temporary cache directory |
| `test_data_dir` | `Path` | Path to test data directory |

### Configuration Fixtures

| Fixture | Type | Description |
|---------|------|-------------|
| `minimal_config_dict` | `Dict` | Minimal pipeline config dictionary |

### Example Usage

```python
def test_example(
    sample_text,
    sample_document,
    sample_entities,
    mock_kb,
    temp_text_file,
    minimal_config_dict,
):
    """Example showing all available fixtures."""
    # sample_text is a string
    assert "Barack Obama" in sample_text

    # sample_document is a Document instance
    assert sample_document.id == "doc-001"

    # sample_entities is a list of Entity objects
    assert any(e.title == "Barack Obama" for e in sample_entities)

    # mock_kb is a MockKnowledgeBase with sample_entities loaded
    entity = mock_kb.get_entity("Q76")
    assert entity.title == "Barack Obama"

    # temp_text_file is a path string (cleaned up automatically)
    with open(temp_text_file) as f:
        content = f.read()
    assert "Barack Obama" in content

    # minimal_config_dict is ready for PipelineConfig.from_dict()
    assert "ner" in minimal_config_dict
```

---

## Coverage Reports

### Generate Coverage Report

```bash
# Run tests with coverage
pytest tests/ --cov=el_pipeline

# Generate HTML report
pytest tests/ --cov=el_pipeline --cov-report=html
open htmlcov/index.html  # View in browser

# Generate XML report (for CI)
pytest tests/ --cov=el_pipeline --cov-report=xml

# Show coverage in terminal
pytest tests/ --cov=el_pipeline --cov-report=term-missing
```

### Coverage Configuration

From `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["el_pipeline"]
omit = [
    "el_pipeline/scripts/*",
    "*/__pycache__/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
]
```

### Adding Coverage Exclusions

```python
def experimental_feature():  # pragma: no cover
    """This code is excluded from coverage."""
    pass


if TYPE_CHECKING:  # Excluded by default
    from expensive_import import Type
```

---

## CI Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run fast tests
      run: |
        pytest tests/unit -m "not slow" --cov=el_pipeline --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v4
      with:
        file: ./coverage.xml
```

---

## Best Practices

### 1. Keep Unit Tests Fast

```python
# Good: Test logic without loading models
def test_output_parsing():
    result = parse_output("answer: 3")
    assert result == 3

# Avoid: Loading models in unit tests
@pytest.mark.slow  # Mark it if you must
def test_with_real_model():
    model = load_heavy_model()
    # ...
```

### 2. Use Appropriate Markers

```python
@pytest.mark.slow  # Takes > 1 second
@pytest.mark.integration  # Tests multiple components together
@pytest.mark.requires_spacy  # Needs spaCy model
def test_something():
    pass
```

### 3. Clean Up Resources

```python
@pytest.fixture
def temp_file():
    """Create and clean up a temporary file."""
    path = Path("temp_test_file.txt")
    path.write_text("test")
    yield str(path)
    path.unlink()  # Clean up
```

### 4. Test Edge Cases

```python
@pytest.mark.parametrize("input_text,expected_count", [
    ("", 0),  # Empty
    ("no entities here", 0),  # No entities
    ("Albert Einstein", 1),  # Single entity
    ("Albert Einstein and Marie Curie", 2),  # Multiple
    ("A" * 10000, 0),  # Very long text
])
def test_ner_edge_cases(input_text, expected_count):
    # ...
```

### 5. Document Test Purpose

```python
def test_gliner_handles_long_documents():
    """
    Regression test for issue #123.

    GLiNER should chunk documents longer than 1500 chars
    to avoid context limit errors.
    """
    # ...
```
