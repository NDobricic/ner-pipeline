"""
spaCy candidate generation components for the EL pipeline.

Provides factories and components for candidate generation:
- LELADenseCandidatesComponent: Dense retrieval using embeddings + FAISS
- FuzzyCandidatesComponent: RapidFuzz string matching
- BM25CandidatesComponent: Standard rank-bm25 based
"""

import logging
import hashlib
from pathlib import Path
from typing import List, Optional

import numpy as np
from spacy.language import Language
from spacy.tokens import Doc, Span

from el_pipeline.knowledge_bases.base import KnowledgeBase
from el_pipeline.lela.config import (
    CANDIDATES_TOP_K,
    DEFAULT_EMBEDDER_MODEL,
    RETRIEVER_TASK,
)
from el_pipeline.lela.llm_pool import (
    get_sentence_transformer_instance,
    release_sentence_transformer,
)
from el_pipeline.utils import ensure_candidates_extension
from el_pipeline.types import Candidate, ProgressCallback

logger = logging.getLogger(__name__)

# Lazy imports
_bm25s = None
_Stemmer = None
_faiss = None


# def _get_bm25s():
#     """Lazy import of bm25s."""
#     global _bm25s
#     if _bm25s is None:
#         try:
#             import bm25s
#
#             _bm25s = bm25s
#         except ImportError:
#             raise ImportError(
#                 "bm25s package required for BM25 candidates. "
#                 "Install with: pip install 'bm25s[full]'"
#             )
#     return _bm25s


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
# LELA BM25 Candidates Component
# ============================================================================


# @Language.factory(
#     "el_pipeline_lela_bm25_candidates",
#     default_config={
#         "top_k": CANDIDATES_TOP_K,
#         "use_context": True,
#         "stemmer_language": "english",
#     },
# )
# def create_lela_bm25_candidates_component(
#     nlp: Language,
#     name: str,
#     top_k: int,
#     use_context: bool,
#     stemmer_language: str,
# ):
#     """Factory for LELA BM25 candidates component."""
#     return LELABM25CandidatesComponent(
#         nlp=nlp,
#         top_k=top_k,
#         use_context=use_context,
#         stemmer_language=stemmer_language,
#     )
#
#
# class LELABM25CandidatesComponent:
#     """
#     BM25 candidate generation component for spaCy.
#
#     Uses bm25s library with Stemmer for efficient BM25 retrieval.
#     Candidates are stored in span._.candidates as List[Candidate].
#     """
#
#     def __init__(
#         self,
#         nlp: Language,
#         top_k: int = CANDIDATES_TOP_K,
#         use_context: bool = True,
#         stemmer_language: str = "english",
#     ):
#         self.nlp = nlp
#         self.top_k = top_k
#         self.use_context = use_context
#         self.stemmer_language = stemmer_language
#
#         ensure_candidates_extension()
#
#         # Initialize lazily - will be built when KB is set
#         self.kb = None
#         self.entities = None
#         self.corpus_records = None
#         self.retriever = None
#         self.stemmer = None
#         self.tokenizer = None
#
#         # Optional progress callback for fine-grained progress reporting
#         self.progress_callback: Optional[ProgressCallback] = None
#
#     def initialize(self, kb: KnowledgeBase):
#         """Initialize the component with a knowledge base."""
#         if kb is None:
#             raise ValueError("LELA BM25 requires a knowledge base.")
#
#         self.kb = kb
#
#         bm25s = _get_bm25s()
#         Stemmer = _get_stemmer()
#
#         self.entities = list(kb.all_entities())
#         if not self.entities:
#             raise ValueError("Knowledge base is empty.")
#
#         # Build corpus - include entity ID for proper resolution
#         self.corpus_records = []
#         corpus_texts = []
#         for entity in self.entities:
#             record = {
#                 "id": entity.id,
#                 "title": entity.title,
#                 "description": entity.description or "",
#             }
#             self.corpus_records.append(record)
#             corpus_texts.append(f"{entity.title} {entity.description or ''}")
#
#         logger.info(f"Building BM25 index over {len(corpus_texts)} entities")
#
#         # Create stemmer and tokenize
#         self.stemmer = Stemmer.Stemmer(self.stemmer_language)
#         self.tokenizer = bm25s.tokenization.Tokenizer(stemmer=self.stemmer)
#         corpus_tokens = self.tokenizer.tokenize(corpus_texts, return_as="tuple")
#
#         # Build index
#         self.retriever = bm25s.BM25(corpus=self.corpus_records, backend="numba")
#         self.retriever.index(corpus_tokens)
#
#         logger.info("BM25 index built successfully")
#
#     def __call__(self, doc: Doc) -> Doc:
#         """Generate candidates for all entities in the document."""
#         if self.retriever is None:
#             logger.warning("BM25 component not initialized - call initialize(kb) first")
#             return doc
#
#         bm25s = _get_bm25s()
#         entities = list(doc.ents)
#         num_entities = len(entities)
#
#         for i, ent in enumerate(entities):
#             # Report progress if callback is set
#             if self.progress_callback and num_entities > 0:
#                 progress = i / num_entities
#                 ent_text = ent.text[:25] + "..." if len(ent.text) > 25 else ent.text
#                 self.progress_callback(
#                     progress, f"Generating candidates {i+1}/{num_entities}: {ent_text}"
#                 )
#
#             # Build query
#             if self.use_context and hasattr(ent._, "context") and ent._.context:
#                 query_text = f"{ent.text} {ent._.context}"
#             else:
#                 query_text = ent.text
#
#             # Tokenize query
#             query_tokens = bm25s.tokenize(
#                 [query_text],
#                 stemmer=self.stemmer,
#                 return_ids=False,
#             )
#
#             if not query_tokens[0]:
#                 ent._.candidates = []
#                 continue
#
#             # Retrieve
#             k = min(self.top_k, len(self.entities))
#             results = self.retriever.retrieve(query_tokens, k=k)
#             candidates_docs = results.documents[0]
#             scores = (
#                 results.scores[0]
#                 if hasattr(results, "scores")
#                 else [0.0] * len(candidates_docs)
#             )
#
#             # Store as List[Candidate] with entity ID for proper resolution
#             candidates = []
#             candidate_scores = []
#             for j, record in enumerate(candidates_docs):
#                 score = float(scores[j]) if j < len(scores) else 0.0
#                 candidates.append(
#                     Candidate(
#                         entity_id=record["id"],
#                         score=score,
#                         description=record["description"],
#                     )
#                 )
#                 candidate_scores.append(score)
#
#             ent._.candidates = candidates
#             ent._.candidate_scores = candidate_scores
#             logger.debug(f"Retrieved {len(candidates)} candidates for '{ent.text}'")
#
#         # Clear progress callback after processing
#         self.progress_callback = None
#
#         return doc


# ============================================================================
# LELA Dense Candidates Component
# ============================================================================


@Language.factory(
    "el_pipeline_lela_dense_candidates",
    default_config={
        "model_name": DEFAULT_EMBEDDER_MODEL,
        "top_k": CANDIDATES_TOP_K,
        "device": None,
        "use_context": False,
        "cache_dir": None,
    },
)
def create_lela_dense_candidates_component(
    nlp: Language,
    name: str,
    model_name: str,
    top_k: int,
    device: Optional[str],
    use_context: bool,
    cache_dir: Optional[str],
):
    """Factory for LELA dense candidates component."""
    return LELADenseCandidatesComponent(
        nlp=nlp,
        model_name=model_name,
        top_k=top_k,
        device=device,
        use_context=use_context,
        cache_dir=cache_dir,
    )


import hashlib
from pathlib import Path

# ... (rest of imports)


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
        cache_dir: Optional[str] = None,
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
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./.ner_cache")

        # Optional progress callback for fine-grained progress reporting
        self.progress_callback: Optional[ProgressCallback] = None

    def _get_cache_path(self, kb_path: str) -> Path:
        """Generate a unique cache path for the FAISS index."""
        kb_hash = hashlib.sha256(kb_path.encode()).hexdigest()[:16]
        model_hash = hashlib.sha256(self.model_name.encode()).hexdigest()[:16]
        device_hash = hashlib.sha256(str(self.device).encode()).hexdigest()[:8]

        index_name = f"faiss_index_{kb_hash}_{model_hash}_{device_hash}.faiss"
        return self.cache_dir / index_name

    def _get_cache_path(self, kb_path: str) -> Path:
        """Generate a unique cache path for the FAISS index."""
        kb_hash = hashlib.sha256(kb_path.encode()).hexdigest()[:16]
        model_hash = hashlib.sha256(self.model_name.encode()).hexdigest()[:16]
        device_hash = hashlib.sha256(str(self.device).encode()).hexdigest()[:8]

        index_name = f"faiss_index_{kb_hash}_{model_hash}_{device_hash}.faiss"
        return self.cache_dir / index_name

    def initialize(self, kb: KnowledgeBase):
        """Initialize the component with a knowledge base."""
        self.kb = kb

        faiss = _get_faiss()

        self.entities = list(kb.all_entities())
        if not self.entities:
            raise ValueError("Knowledge base is empty.")

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._get_cache_path(kb.path)  # Assuming KB has a 'path' attribute

        if cache_path.exists():
            logger.info(f"Loading FAISS index from cache: {cache_path}")
            self.index = faiss.read_index(str(cache_path))
            logger.info(f"FAISS index loaded from cache: {self.index.ntotal} vectors")
        else:
            # Build entity texts
            entity_texts = [f"{e.title} {e.description or ''}" for e in self.entities]

            logger.info(f"Building dense index over {len(self.entities)} entities")

            # Load model for embedding, then release after index is built
            model = get_sentence_transformer_instance(self.model_name, self.device)
            embeddings = model.encode(
                entity_texts, normalize_embeddings=True, convert_to_numpy=True
            )
            release_sentence_transformer(self.model_name, self.device)

            # Build FAISS index
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)

            logger.info(
                f"Dense index built: {self.index.ntotal} vectors, dim={dim}. Saving to {cache_path}"
            )
            faiss.write_index(self.index, str(cache_path))
            logger.info("FAISS index saved to cache.")

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
        model = get_sentence_transformer_instance(self.model_name, self.device)

        if self.progress_callback:
            self.progress_callback(0.1, "Model loaded, generating candidates...")

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
    "el_pipeline_fuzzy_candidates",
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

    def initialize(self, kb: KnowledgeBase):
        """Initialize the component with a knowledge base."""
        if kb is None:
            raise ValueError("Fuzzy matching requires a knowledge base.")

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

        from rapidfuzz import process

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

            results = process.extract(ent.text, self.titles, limit=self.top_k)

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
    "el_pipeline_bm25_candidates",
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

    def initialize(self, kb: KnowledgeBase):
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

        # Build corpus
        self.corpus = []
        for entity in self.entities:
            text = f"{entity.title} {entity.description or ''}"
            tokens = text.lower().split()
            self.corpus.append(tokens)

        self.bm25 = BM25Okapi(self.corpus)
        logger.info(f"rank-bm25 index built over {len(self.entities)} entities")

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
