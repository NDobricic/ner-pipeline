"""
spaCy candidate generation components for the NER pipeline.

Provides factories and components for candidate generation:
- LELADenseCandidatesComponent: Dense retrieval using embeddings + FAISS
- FuzzyCandidatesComponent: RapidFuzz string matching
- BM25CandidatesComponent: Standard rank-bm25 based
"""

import hashlib
import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from spacy.language import Language
from spacy.tokens import Doc, Span

from ner_pipeline.knowledge_bases.base import KnowledgeBase
from ner_pipeline.lela.config import (
    CANDIDATES_TOP_K,
    DEFAULT_EMBEDDER_MODEL,
    RETRIEVER_TASK,
)
from ner_pipeline.lela.llm_pool import (
    get_sentence_transformer_instance,
    release_sentence_transformer,
)
from ner_pipeline.utils import ensure_candidates_extension
from ner_pipeline.types import Candidate, ProgressCallback

logger = logging.getLogger(__name__)

# Lazy imports
_Stemmer = None
_faiss = None


def _get_stemmer():
    """Lazy import of Stemmer."""
    global _Stemmer
    if _Stemmer is None:
        try:
            import Stemmer

            _Stemmer = Stemmer
        except ImportError:
            raise ImportError(
                "PyStemmer package required for BM25 candidates. "
                "Install with: pip install PyStemmer"
            )
    return _Stemmer


def _get_faiss():
    """Lazy import of faiss."""
    global _faiss
    if _faiss is None:
        try:
            import faiss

            _faiss = faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu package required for dense retrieval. "
                "Install with: pip install faiss-cpu"
            )
    return _faiss


# ============================================================================
# LELA Dense Candidates Component
# ============================================================================


@Language.factory(
    "ner_pipeline_lela_dense_candidates",
    default_config={
        "model_name": DEFAULT_EMBEDDER_MODEL,
        "top_k": CANDIDATES_TOP_K,
        "device": None,
        "use_context": False,
    },
)
def create_lela_dense_candidates_component(
    nlp: Language,
    name: str,
    model_name: str,
    top_k: int,
    device: Optional[str],
    use_context: bool,
):
    """Factory for LELA dense candidates component."""
    return LELADenseCandidatesComponent(
        nlp=nlp,
        model_name=model_name,
        top_k=top_k,
        device=device,
        use_context=use_context,
    )


class LELADenseCandidatesComponent:
    """
    Dense retrieval candidate generation component for spaCy.

    Uses SentenceTransformers and FAISS for nearest neighbor search.
    Candidates are stored in span._.candidates as List[Candidate].

    Memory management: Model is loaded on-demand and released after use,
    allowing it to be evicted if memory is needed for later stages.
    """

    def __init__(
        self,
        nlp: Language,
        model_name: str = DEFAULT_EMBEDDER_MODEL,
        top_k: int = CANDIDATES_TOP_K,
        device: Optional[str] = None,
        use_context: bool = False,
    ):
        self.nlp = nlp
        self.model_name = model_name
        self.top_k = top_k
        self.device = device
        self.use_context = use_context

        ensure_candidates_extension()

        # Model loaded on-demand in __call__, not here
        # This allows lazy eviction when memory is needed

        # Initialize lazily
        self.kb = None
        self.entities = None
        self.index = None

        # Optional progress callback for fine-grained progress reporting
        self.progress_callback: Optional[ProgressCallback] = None

    def initialize(self, kb: KnowledgeBase, cache_dir: Optional[Path] = None):
        """Initialize the component with a knowledge base."""
        if kb is None:
            raise ValueError("LELA dense retrieval requires a knowledge base.")

        self.kb = kb

        faiss = _get_faiss()

        self.entities = list(kb.all_entities())
        if not self.entities:
            raise ValueError("Knowledge base is empty.")

        # Try loading from cache
        cache_hash = None
        index_file = None
        if cache_dir and hasattr(kb, "identity_hash"):
            raw = f"lela_dense:{kb.identity_hash}:{self.model_name}".encode()
            cache_hash = hashlib.sha256(raw).hexdigest()
            index_dir = Path(cache_dir) / "index" / f"lela_dense_{cache_hash}"
            index_file = index_dir / "index.faiss"
            try:
                if index_file.exists():
                    self.index = faiss.read_index(str(index_file))
                    logger.info(
                        f"Loaded LELA dense index from cache ({cache_hash[:12]}): "
                        f"{self.index.ntotal} vectors"
                    )
                    return
            except Exception:
                logger.warning(
                    "LELA dense cache load failed, will rebuild", exc_info=True
                )

        # Build entity texts
        entity_texts = [f"{e.title} {e.description or ''}" for e in self.entities]

        logger.info(f"Building dense index over {len(self.entities)} entities")

        # Load model for embedding, then release after index is built
        model, _ = get_sentence_transformer_instance(self.model_name, self.device)
        embeddings = model.encode(
            entity_texts, normalize_embeddings=True, convert_to_numpy=True
        )
        release_sentence_transformer(self.model_name, self.device)

        # Build FAISS index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        logger.info(f"Dense index built: {self.index.ntotal} vectors, dim={dim}")

        # Save to cache
        if index_file is not None:
            try:
                index_file.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self.index, str(index_file))
                logger.info(f"Saved LELA dense index cache ({cache_hash[:12]})")
            except Exception:
                logger.warning("Failed to save LELA dense cache", exc_info=True)

    def _embed_texts(self, texts: List[str], model) -> np.ndarray:
        """Embed texts using the SentenceTransformer model."""
        return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def _format_query(self, mention_text: str, context: Optional[str] = None) -> str:
        """Format query with task instruction."""
        query = f"{mention_text}: {context}" if context else mention_text
        return f"Instruct: {RETRIEVER_TASK}\nQuery: {query}"

    def __call__(self, doc: Doc) -> Doc:
        """Generate candidates for all entities in the document."""
        if self.index is None:
            logger.warning(
                "Dense component not initialized - call initialize(kb) first"
            )
            return doc

        entities = list(doc.ents)
        num_entities = len(entities)

        if num_entities == 0:
            return doc

        # Report model loading
        if self.progress_callback:
            self.progress_callback(
                0.0, f"Loading embedding model ({self.model_name.split('/')[-1]})..."
            )

        # Load model for this stage (will reuse cached if available)
        model, was_cached = get_sentence_transformer_instance(
            self.model_name, self.device
        )

        if self.progress_callback:
            status = "Using cached model" if was_cached else "Model loaded"
            self.progress_callback(0.1, f"{status}, generating candidates...")

        # Progress: 0.0-0.1 = model loading, 0.1-1.0 = processing entities
        processing_start = 0.1
        processing_range = 0.9

        try:
            for i, ent in enumerate(entities):
                # Report progress if callback is set
                if self.progress_callback and num_entities > 0:
                    progress = processing_start + (i / num_entities) * processing_range
                    ent_text = ent.text[:25] + "..." if len(ent.text) > 25 else ent.text
                    self.progress_callback(
                        progress,
                        f"Generating candidates {i+1}/{num_entities}: {ent_text}",
                    )

                # Build query
                context = None
                if self.use_context and hasattr(ent._, "context"):
                    context = ent._.context
                query_text = self._format_query(ent.text, context)

                # Embed query (already normalized by encode())
                query_embedding = self._embed_texts([query_text], model)

                # Search
                k = min(self.top_k, len(self.entities))
                scores, indices = self.index.search(query_embedding, k)

                # Build candidates as List[Candidate] with entity ID
                candidates = []
                candidate_scores = []
                for score, idx in zip(scores[0], indices[0]):
                    entity = self.entities[int(idx)]
                    score_val = float(score)
                    candidates.append(
                        Candidate(
                            entity_id=entity.id,
                            score=score_val,
                            description=entity.description,
                        )
                    )
                    candidate_scores.append(score_val)

                ent._.candidates = candidates
                ent._.candidate_scores = candidate_scores
                logger.debug(
                    f"Dense-retrieved {len(candidates)} candidates for '{ent.text}'"
                )
        finally:
            # Release model - stays cached but can be evicted if memory needed
            release_sentence_transformer(self.model_name, self.device)

        # Clear progress callback after processing
        self.progress_callback = None

        return doc


# ============================================================================
# Fuzzy Candidates Component
# ============================================================================


@Language.factory(
    "ner_pipeline_fuzzy_candidates",
    default_config={
        "top_k": 20,
    },
)
def create_fuzzy_candidates_component(
    nlp: Language,
    name: str,
    top_k: int,
):
    """Factory for fuzzy candidates component."""
    return FuzzyCandidatesComponent(nlp=nlp, top_k=top_k)


class FuzzyCandidatesComponent:
    """
    Fuzzy string matching candidate generation component for spaCy.

    Uses RapidFuzz for efficient fuzzy string matching against KB titles.
    """

    def __init__(
        self,
        nlp: Language,
        top_k: int = 20,
    ):
        self.nlp = nlp
        self.top_k = top_k

        ensure_candidates_extension()

        # Initialize lazily
        self.kb = None
        self.entities = None
        self.titles = None

        # Optional progress callback for fine-grained progress reporting
        self.progress_callback: Optional[ProgressCallback] = None

    def initialize(self, kb: KnowledgeBase, cache_dir: Optional[Path] = None):
        """Initialize the component with a knowledge base."""
        self.kb = kb
        self.entities = list(kb.all_entities())
        self.titles = [e.title for e in self.entities]

        logger.info(f"Fuzzy index built over {len(self.entities)} entities")

    def __call__(self, doc: Doc) -> Doc:
        """Generate candidates for all entities in the document."""
        if self.entities is None:
            logger.warning(
                "Fuzzy component not initialized - call initialize(kb) first"
            )
            return doc

        from rapidfuzz import process, fuzz, utils

        entities = list(doc.ents)
        num_entities = len(entities)
        num_titles = len(self.titles)
        # Chunk size for sub-entity progress reporting on large KBs
        CHUNK_SIZE = 500_000
        use_chunks = num_titles > CHUNK_SIZE

        if self.progress_callback:
            self.progress_callback(
                0.0, f"Generating candidates for {num_entities} entities..."
            )

        for i, ent in enumerate(entities):
            ent_text = ent.text[:25] + "..." if len(ent.text) > 25 else ent.text

            # Report progress if callback is set
            if self.progress_callback and num_entities > 0:
                progress = i / num_entities
                self.progress_callback(
                    progress, f"Generating candidates {i+1}/{num_entities}: {ent_text}"
                )

            if use_chunks:
                # Process in chunks to allow progress updates during long searches
                results = []
                for chunk_start in range(0, num_titles, CHUNK_SIZE):
                    chunk_end = min(chunk_start + CHUNK_SIZE, num_titles)
                    chunk_titles = self.titles[chunk_start:chunk_end]

                    chunk_results = process.extract(
                        ent.text,
                        chunk_titles,
                        limit=self.top_k,
                        scorer=fuzz.WRatio,
                        processor=utils.default_process,
                        score_cutoff=30,
                    )
                    # Remap indices back to global
                    results.extend(
                        (title, score, idx + chunk_start)
                        for title, score, idx in chunk_results
                    )

                    if self.progress_callback and num_entities > 0:
                        chunk_frac = chunk_end / num_titles
                        progress = (i + chunk_frac) / num_entities
                        self.progress_callback(
                            progress,
                            f"Generating candidates {i+1}/{num_entities}: {ent_text} ({int(chunk_frac*100)}%)",
                        )

                # Keep only top_k across all chunks
                results.sort(key=lambda x: x[1], reverse=True)
                results = results[: self.top_k]
            else:
                results = process.extract(
                    ent.text,
                    self.titles,
                    limit=self.top_k,
                    scorer=fuzz.WRatio,
                    processor=utils.default_process,
                    score_cutoff=30,
                )

            # Build candidates as List[Candidate] with entity ID
            candidates = []
            candidate_scores = []
            for title, score, idx in results:
                entity = self.entities[idx]
                # Normalize fuzzy score from 0-100 to 0-1
                score_val = float(score) / 100.0
                candidates.append(
                    Candidate(
                        entity_id=entity.id,
                        score=score_val,
                        description=entity.description,
                    )
                )
                candidate_scores.append(score_val)

            ent._.candidates = candidates
            ent._.candidate_scores = candidate_scores
            logger.debug(f"Fuzzy-matched {len(candidates)} candidates for '{ent.text}'")

        # Clear progress callback after processing
        self.progress_callback = None

        return doc


# ============================================================================
# Standard BM25 Candidates Component (using rank-bm25)
# ============================================================================


@Language.factory(
    "ner_pipeline_bm25_candidates",
    default_config={
        "top_k": 20,
    },
)
def create_bm25_candidates_component(
    nlp: Language,
    name: str,
    top_k: int,
):
    """Factory for standard BM25 candidates component."""
    return BM25CandidatesComponent(nlp=nlp, top_k=top_k)


class BM25CandidatesComponent:
    """
    BM25 candidate generation using rank-bm25 library.

    Alternative to LELA BM25 using the simpler rank-bm25 package.
    """

    def __init__(
        self,
        nlp: Language,
        top_k: int = 20,
    ):
        self.nlp = nlp
        self.top_k = top_k

        ensure_candidates_extension()

        # Initialize lazily
        self.kb = None
        self.entities = None
        self.bm25 = None
        self.corpus = None

        # Optional progress callback for fine-grained progress reporting
        self.progress_callback: Optional[ProgressCallback] = None

    def initialize(self, kb: KnowledgeBase, cache_dir: Optional[Path] = None):
        """Initialize the component with a knowledge base."""
        if kb is None:
            raise ValueError("BM25 requires a knowledge base.")

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 package required. Install with: pip install rank-bm25"
            )

        self.kb = kb
        self.entities = list(kb.all_entities())

        # Try loading from cache
        cache_hash = None
        cache_file = None
        if cache_dir and hasattr(kb, "identity_hash"):
            raw = f"bm25:{kb.identity_hash}".encode()
            cache_hash = hashlib.sha256(raw).hexdigest()
            idx_dir = Path(cache_dir) / "index"
            idx_dir.mkdir(parents=True, exist_ok=True)
            cache_file = idx_dir / f"bm25_{cache_hash}.pkl"
            try:
                if cache_file.exists():
                    with cache_file.open("rb") as f:
                        self.bm25, self.corpus = pickle.load(f)
                    logger.info(
                        f"Loaded rank-bm25 index from cache ({cache_hash[:12]})"
                    )
                    return
            except Exception:
                logger.warning(
                    "rank-bm25 cache load failed, will rebuild", exc_info=True
                )

        # Build corpus
        self.corpus = []
        for entity in self.entities:
            text = f"{entity.title} {entity.description or ''}"
            tokens = text.lower().split()
            self.corpus.append(tokens)

        self.bm25 = BM25Okapi(self.corpus)
        logger.info(f"rank-bm25 index built over {len(self.entities)} entities")

        # Save to cache
        if cache_file is not None:
            try:
                with cache_file.open("wb") as f:
                    pickle.dump(
                        (self.bm25, self.corpus), f, protocol=pickle.HIGHEST_PROTOCOL
                    )
                logger.info(f"Saved rank-bm25 index cache ({cache_hash[:12]})")
            except Exception:
                logger.warning("Failed to save rank-bm25 cache", exc_info=True)

    def __call__(self, doc: Doc) -> Doc:
        """Generate candidates for all entities in the document."""
        if self.bm25 is None:
            logger.warning("BM25 component not initialized - call initialize(kb) first")
            return doc

        entities = list(doc.ents)
        num_entities = len(entities)

        for i, ent in enumerate(entities):
            # Report progress if callback is set
            if self.progress_callback and num_entities > 0:
                progress = i / num_entities
                ent_text = ent.text[:25] + "..." if len(ent.text) > 25 else ent.text
                self.progress_callback(
                    progress, f"Generating candidates {i+1}/{num_entities}: {ent_text}"
                )

            query_tokens = ent.text.lower().split()
            scores = self.bm25.get_scores(query_tokens)

            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][: self.top_k]

            # Build candidates as List[Candidate] with entity ID
            candidates = []
            candidate_scores = []
            for idx in top_indices:
                entity = self.entities[idx]
                score_val = float(scores[idx])
                candidates.append(
                    Candidate(
                        entity_id=entity.id,
                        score=score_val,
                        description=entity.description,
                    )
                )
                candidate_scores.append(score_val)

            ent._.candidates = candidates
            ent._.candidate_scores = candidate_scores
            logger.debug(
                f"BM25 retrieved {len(candidates)} candidates for '{ent.text}'"
            )

        # Clear progress callback after processing
        self.progress_callback = None

        return doc
