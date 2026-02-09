"""Unit tests for data types."""

import pytest

from el_pipeline.types import Candidate, Document, Entity, Mention, ResolvedMention


class TestDocument:
    """Tests for Document dataclass."""

    def test_create_with_required_fields(self):
        doc = Document(id="doc-1", text="Hello world")
        assert doc.id == "doc-1"
        assert doc.text == "Hello world"
        assert doc.meta == {}

    def test_create_with_meta(self):
        doc = Document(id="doc-2", text="Test", meta={"source": "test.txt"})
        assert doc.meta == {"source": "test.txt"}

    def test_create_with_none_id(self):
        doc = Document(id=None, text="No ID document")
        assert doc.id is None
        assert doc.text == "No ID document"


class TestMention:
    """Tests for Mention dataclass."""

    def test_create_with_required_fields(self):
        mention = Mention(start=0, end=5, text="Hello")
        assert mention.start == 0
        assert mention.end == 5
        assert mention.text == "Hello"
        assert mention.label is None
        assert mention.context is None

    def test_create_with_all_fields(self):
        mention = Mention(
            start=10,
            end=20,
            text="Example",
            label="ENTITY",
            context="This is Example context",
        )
        assert mention.label == "ENTITY"
        assert mention.context == "This is Example context"

    def test_mention_span_length(self):
        mention = Mention(start=5, end=15, text="0123456789")
        assert mention.end - mention.start == len(mention.text)


class TestCandidate:
    """Tests for Candidate dataclass."""

    def test_create_with_required_fields(self):
        candidate = Candidate(entity_id="Q123")
        assert candidate.entity_id == "Q123"
        assert candidate.score is None
        assert candidate.description is None

    def test_create_with_all_fields(self):
        candidate = Candidate(
            entity_id="Q456",
            score=0.95,
            description="Test entity description",
        )
        assert candidate.score == 0.95
        assert candidate.description == "Test entity description"


class TestEntity:
    """Tests for Entity dataclass."""

    def test_create_with_required_fields(self):
        entity = Entity(id="Q1", title="Test Entity")
        assert entity.id == "Q1"
        assert entity.title == "Test Entity"
        assert entity.description is None
        assert entity.metadata == {}

    def test_create_with_all_fields(self):
        entity = Entity(
            id="Q2",
            title="Full Entity",
            description="A complete entity",
            metadata={"type": "person", "aliases": ["Entity"]},
        )
        assert entity.description == "A complete entity"
        assert entity.metadata["type"] == "person"


class TestResolvedMention:
    """Tests for ResolvedMention dataclass."""

    def test_create_with_mention_only(self):
        mention = Mention(start=0, end=5, text="Test")
        resolved = ResolvedMention(mention=mention, entity=None)
        assert resolved.mention == mention
        assert resolved.entity is None
        assert resolved.candidates == []

    def test_create_with_entity(self):
        mention = Mention(start=0, end=5, text="Test")
        entity = Entity(id="Q1", title="Test")
        resolved = ResolvedMention(mention=mention, entity=entity)
        assert resolved.entity == entity

    def test_create_with_candidates(self):
        mention = Mention(start=0, end=5, text="Test")
        candidates = [
            Candidate(entity_id="Q1", score=0.9),
            Candidate(entity_id="Q2", score=0.8),
        ]
        resolved = ResolvedMention(
            mention=mention,
            entity=None,
            candidates=candidates,
        )
        assert len(resolved.candidates) == 2
        assert resolved.candidates[0].entity_id == "Q1"
