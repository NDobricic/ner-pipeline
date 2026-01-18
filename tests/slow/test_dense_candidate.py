"""Slow tests for DenseCandidateGenerator (requires model downloads)."""

import json
import os
import tempfile

import pytest

from ner_pipeline.types import Candidate, Document, Entity, Mention


@pytest.mark.slow
@pytest.mark.requires_sentence_transformers
class TestDenseCandidateGenerator:
    """Tests for DenseCandidateGenerator (requires sentence-transformers + FAISS)."""

    @pytest.fixture
    def kb_entities(self) -> list[Entity]:
        return [
            Entity(
                id="Q76",
                title="Barack Obama",
                description="44th President of the United States, serving from 2009 to 2017",
            ),
            Entity(
                id="Q6279",
                title="Joe Biden",
                description="46th President of the United States, serving since 2021",
            ),
            Entity(
                id="Q30",
                title="United States",
                description="Country in North America, federal republic",
            ),
            Entity(
                id="Q84",
                title="London",
                description="Capital city of England and the United Kingdom",
            ),
            Entity(
                id="Q60",
                title="New York City",
                description="Most populous city in the United States",
            ),
            Entity(
                id="Q49088",
                title="Columbia University",
                description="Private Ivy League research university in New York City",
            ),
        ]

    @pytest.fixture
    def temp_kb_file(self, kb_entities: list[Entity]) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for entity in kb_entities:
                line = json.dumps({
                    "id": entity.id,
                    "title": entity.title,
                    "description": entity.description,
                })
                f.write(line + "\n")
            path = f.name
        yield path
        os.unlink(path)

    @pytest.fixture
    def kb(self, temp_kb_file: str):
        from ner_pipeline.knowledge_bases.custom import CustomJSONLKnowledgeBase
        return CustomJSONLKnowledgeBase(path=temp_kb_file)

    @pytest.fixture
    def generator(self, kb):
        from ner_pipeline.candidates.dense import DenseCandidateGenerator
        return DenseCandidateGenerator(
            kb=kb,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            top_k=3,
        )

    @pytest.fixture
    def sample_doc(self) -> Document:
        return Document(id="test-doc", text="Test document content.")

    def test_requires_knowledge_base(self):
        from ner_pipeline.candidates.dense import DenseCandidateGenerator
        with pytest.raises(ValueError, match="requires a knowledge base"):
            DenseCandidateGenerator(kb=None)

    def test_generate_returns_candidates(self, generator, sample_doc: Document):
        mention = Mention(
            start=0,
            end=12,
            text="Barack Obama",
            context="Barack Obama was the president",
        )
        candidates = generator.generate(mention, sample_doc)
        assert len(candidates) > 0
        assert all(isinstance(c, Candidate) for c in candidates)

    def test_semantic_matching(self, generator, sample_doc: Document):
        """Test that semantic matching finds relevant entities."""
        mention = Mention(
            start=0,
            end=9,
            text="President",
            context="The President of the United States gave a speech",
        )
        candidates = generator.generate(mention, sample_doc)
        # Should rank presidents higher due to semantic similarity
        entity_ids = [c.entity_id for c in candidates]
        # Either Obama or Biden should be in top results
        assert "Q76" in entity_ids or "Q6279" in entity_ids

    def test_respects_top_k(self, kb, sample_doc: Document):
        from ner_pipeline.candidates.dense import DenseCandidateGenerator
        generator = DenseCandidateGenerator(kb=kb, top_k=2)
        mention = Mention(start=0, end=4, text="city", context="A major city")
        candidates = generator.generate(mention, sample_doc)
        assert len(candidates) <= 2

    def test_candidates_have_scores(self, generator, sample_doc: Document):
        mention = Mention(start=0, end=6, text="London", context="London is great")
        candidates = generator.generate(mention, sample_doc)
        for c in candidates:
            assert c.score is not None
            assert isinstance(c.score, float)

    def test_candidates_have_descriptions(self, generator, sample_doc: Document):
        mention = Mention(start=0, end=6, text="London", context="Visit London")
        candidates = generator.generate(mention, sample_doc)
        london_candidates = [c for c in candidates if c.entity_id == "Q84"]
        if london_candidates:
            assert london_candidates[0].description is not None

    def test_context_improves_matching(self, kb, sample_doc: Document):
        """Test that context helps with disambiguation."""
        from ner_pipeline.candidates.dense import DenseCandidateGenerator

        # With context about presidents
        generator_with_context = DenseCandidateGenerator(kb=kb, top_k=3, use_context=True)
        mention_with_context = Mention(
            start=0,
            end=5,
            text="Obama",
            context="Obama served as the 44th President of the United States",
        )
        candidates_with = generator_with_context.generate(mention_with_context, sample_doc)

        # Without context
        generator_no_context = DenseCandidateGenerator(kb=kb, top_k=3, use_context=False)
        mention_no_context = Mention(
            start=0,
            end=5,
            text="Obama",
            context="Obama served as the 44th President of the United States",
        )
        candidates_without = generator_no_context.generate(mention_no_context, sample_doc)

        # Both should find Obama, but results may differ
        assert len(candidates_with) > 0
        assert len(candidates_without) > 0

    def test_use_context_false(self, kb, sample_doc: Document):
        """Test generator with use_context=False."""
        from ner_pipeline.candidates.dense import DenseCandidateGenerator
        generator = DenseCandidateGenerator(kb=kb, top_k=3, use_context=False)
        mention = Mention(
            start=0,
            end=12,
            text="Barack Obama",
            context="Some irrelevant context that should be ignored",
        )
        candidates = generator.generate(mention, sample_doc)
        # Should still find Barack Obama based on text alone
        entity_ids = [c.entity_id for c in candidates]
        assert "Q76" in entity_ids


@pytest.mark.slow
@pytest.mark.requires_sentence_transformers
class TestDenseCandidateGeneratorModels:
    """Test DenseCandidateGenerator with different models."""

    @pytest.fixture
    def simple_kb(self):
        from ner_pipeline.knowledge_bases.custom import CustomJSONLKnowledgeBase
        data = [{"id": "Q1", "title": "Test", "description": "A test entity"}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            path = f.name
        kb = CustomJSONLKnowledgeBase(path=path)
        yield kb
        os.unlink(path)

    def test_default_model(self, simple_kb):
        """Test with default model."""
        from ner_pipeline.candidates.dense import DenseCandidateGenerator
        generator = DenseCandidateGenerator(kb=simple_kb)
        doc = Document(id="test", text="test")
        mention = Mention(start=0, end=4, text="Test")
        candidates = generator.generate(mention, doc)
        assert len(candidates) > 0
