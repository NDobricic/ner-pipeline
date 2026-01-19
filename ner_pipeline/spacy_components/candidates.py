"""
spaCy candidate generation components for the NER pipeline.

Provides factories and components for candidate generation:
- LELABM25CandidatesComponent: BM25 using bm25s library
- LELADenseCandidatesComponent: Dense retrieval using embeddings + FAISS
- FuzzyCandidatesComponent: RapidFuzz string matching
- BM25CandidatesComponent: Standard rank-bm25 based
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from spacy.language import Language
from spacy.tokens import Doc, Span

from ner_pipeline.knowledge_bases.base import KnowledgeBase
from ner_pipeline.lela.config import (
    CANDIDATES_TOP_K,
    DEFAULT_EMBEDDER_MODEL,
    RETRIEVER_TASK,
)
from ner_pipeline.lela.llm_pool import embedder_pool

logger = logging.getLogger(__name__)

# Lazy imports
_bm25s = None
_Stemmer = None
_faiss = None


def _get_bm25s():
    """Lazy import of bm25s."""
    global _bm25s
    if _bm25s is None:
        try:
            import bm25s
            _bm25s = bm25s
        except ImportError:
            raise ImportError(
                "bm25s package required for BM25 candidates. "
                "Install with: pip install 'bm25s[full]'"
            )
    return _bm25s


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


def _ensure_candidates_extension():
    """Ensure the candidates extension is registered on Span."""
    if not Span.has_extension("candidates"):
        Span.set_extension("candidates", default=[])


# ============================================================================
# LELA BM25 Candidates Component
# ============================================================================

@Language.factory(
    "ner_pipeline_lela_bm25_candidates",
    default_config={
        "top_k": CANDIDATES_TOP_K,
        "use_context": True,
        "stemmer_language": "english",
    },
)
def create_lela_bm25_candidates_component(
    nlp: Language,
    name: str,
    top_k: int,
    use_context: bool,
    stemmer_language: str,
):
    """Factory for LELA BM25 candidates component."""
    return LELABM25CandidatesComponent(
        nlp=nlp,
        top_k=top_k,
        use_context=use_context,
        stemmer_language=stemmer_language,
    )


class LELABM25CandidatesComponent:
    """
    BM25 candidate generation component for spaCy.

    Uses bm25s library with Stemmer for efficient BM25 retrieval.
    Candidates are stored in span._.candidates as List[Tuple[str, str]] (title, description).
    """

    def __init__(
        self,
        nlp: Language,
        top_k: int = CANDIDATES_TOP_K,
        use_context: bool = True,
        stemmer_language: str = "english",
    ):
        self.nlp = nlp
        self.top_k = top_k
        self.use_context = use_context
        self.stemmer_language = stemmer_language

        _ensure_candidates_extension()

        # Initialize lazily - will be built when KB is set
        self.kb = None
        self.entities = None
        self.corpus_records = None
        self.retriever = None
        self.stemmer = None
        self.tokenizer = None

    def initialize(self, kb: KnowledgeBase):
        """Initialize the component with a knowledge base."""
        if kb is None:
            raise ValueError("LELA BM25 requires a knowledge base.")

        self.kb = kb

        bm25s = _get_bm25s()
        Stemmer = _get_stemmer()

        self.entities = list(kb.all_entities())
        if not self.entities:
            raise ValueError("Knowledge base is empty.")

        # Build corpus
        self.corpus_records = []
        corpus_texts = []
        for entity in self.entities:
            record = {
                "title": entity.title,
                "description": entity.description or "",
            }
            self.corpus_records.append(record)
            corpus_texts.append(f"{entity.title} {entity.description or ''}")

        logger.info(f"Building BM25 index over {len(corpus_texts)} entities")

        # Create stemmer and tokenize
        self.stemmer = Stemmer.Stemmer(self.stemmer_language)
        self.tokenizer = bm25s.tokenization.Tokenizer(stemmer=self.stemmer)
        corpus_tokens = self.tokenizer.tokenize(corpus_texts, return_as="tuple")

        # Build index
        self.retriever = bm25s.BM25(corpus=self.corpus_records, backend="numba")
        self.retriever.index(corpus_tokens)

        logger.info("BM25 index built successfully")

    def __call__(self, doc: Doc) -> Doc:
        """Generate candidates for all entities in the document."""
        if self.retriever is None:
            logger.warning("BM25 component not initialized - call initialize(kb) first")
            return doc

        bm25s = _get_bm25s()

        for ent in doc.ents:
            # Build query
            if self.use_context and hasattr(ent._, "context") and ent._.context:
                query_text = f"{ent.text} {ent._.context}"
            else:
                query_text = ent.text

            # Tokenize query
            query_tokens = bm25s.tokenize(
                [query_text],
                stemmer=self.stemmer,
                return_ids=False,
            )

            if not query_tokens[0]:
                ent._.candidates = []
                continue

            # Retrieve
            k = min(self.top_k, len(self.entities))
            results = self.retriever.retrieve(query_tokens, k=k)
            candidates_docs = results.documents[0]
            scores = results.scores[0] if hasattr(results, 'scores') else [None] * len(candidates_docs)

            # Store as LELA format: List[Tuple[str, str]] (title, description)
            candidates = []
            for i, record in enumerate(candidates_docs):
                candidates.append((record["title"], record["description"]))

            ent._.candidates = candidates
            logger.debug(f"Retrieved {len(candidates)} candidates for '{ent.text}'")

        return doc


# ============================================================================
# LELA Dense Candidates Component
# ============================================================================

@Language.factory(
    "ner_pipeline_lela_dense_candidates",
    default_config={
        "model_name": DEFAULT_EMBEDDER_MODEL,
        "top_k": CANDIDATES_TOP_K,
        "base_url": "http://localhost",
        "port": 8000,
        "use_context": True,
    },
)
def create_lela_dense_candidates_component(
    nlp: Language,
    name: str,
    model_name: str,
    top_k: int,
    base_url: str,
    port: int,
    use_context: bool,
):
    """Factory for LELA dense candidates component."""
    return LELADenseCandidatesComponent(
        nlp=nlp,
        model_name=model_name,
        top_k=top_k,
        base_url=base_url,
        port=port,
        use_context=use_context,
    )


class LELADenseCandidatesComponent:
    """
    Dense retrieval candidate generation component for spaCy.

    Uses OpenAI-compatible embeddings and FAISS for nearest neighbor search.
    Candidates are stored in span._.candidates as List[Tuple[str, str]].
    """

    def __init__(
        self,
        nlp: Language,
        model_name: str = DEFAULT_EMBEDDER_MODEL,
        top_k: int = CANDIDATES_TOP_K,
        base_url: str = "http://localhost",
        port: int = 8000,
        use_context: bool = True,
    ):
        self.nlp = nlp
        self.model_name = model_name
        self.top_k = top_k
        self.base_url = base_url
        self.port = port
        self.use_context = use_context

        _ensure_candidates_extension()

        # Initialize lazily
        self.kb = None
        self.entities = None
        self.index = None

    def initialize(self, kb: KnowledgeBase):
        """Initialize the component with a knowledge base."""
        if kb is None:
            raise ValueError("LELA dense retrieval requires a knowledge base.")

        self.kb = kb

        faiss = _get_faiss()

        self.entities = list(kb.all_entities())
        if not self.entities:
            raise ValueError("Knowledge base is empty.")

        # Build entity texts
        entity_texts = [
            f"{e.title} {e.description or ''}" for e in self.entities
        ]

        logger.info(f"Building dense index over {len(self.entities)} entities")

        # Embed entities
        embeddings = self._embed_texts(entity_texts)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-9)

        # Build FAISS index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        logger.info(f"Dense index built: {self.index.ntotal} vectors, dim={dim}")

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using the embedding service."""
        return embedder_pool.embed(
            texts,
            model_name=self.model_name,
            base_url=self.base_url,
            port=self.port,
        )

    def _format_query(self, mention_text: str, context: Optional[str] = None) -> str:
        """Format query with task instruction."""
        query = mention_text
        if context:
            query = f"{mention_text}: {context}"
        return f"Instruct: {RETRIEVER_TASK}\nQuery: {query}"

    def __call__(self, doc: Doc) -> Doc:
        """Generate candidates for all entities in the document."""
        if self.index is None:
            logger.warning("Dense component not initialized - call initialize(kb) first")
            return doc

        for ent in doc.ents:
            # Build query
            context = None
            if self.use_context and hasattr(ent._, "context"):
                context = ent._.context
            query_text = self._format_query(ent.text, context)

            # Embed query
            query_embedding = self._embed_texts([query_text])
            query_embedding = np.array(query_embedding, dtype=np.float32)

            # Normalize
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm

            # Search
            k = min(self.top_k, len(self.entities))
            scores, indices = self.index.search(query_embedding, k)

            # Build candidates as LELA format
            candidates = []
            for score, idx in zip(scores[0], indices[0]):
                entity = self.entities[int(idx)]
                candidates.append((entity.title, entity.description or ""))

            ent._.candidates = candidates
            logger.debug(f"Dense-retrieved {len(candidates)} candidates for '{ent.text}'")

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

        _ensure_candidates_extension()

        # Initialize lazily
        self.kb = None
        self.entities = None
        self.titles = None

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
            logger.warning("Fuzzy component not initialized - call initialize(kb) first")
            return doc

        from rapidfuzz import process

        for ent in doc.ents:
            results = process.extract(ent.text, self.titles, limit=self.top_k)

            # Build candidates as LELA format
            candidates = []
            for title, score, idx in results:
                entity = self.entities[idx]
                candidates.append((entity.title, entity.description or ""))

            ent._.candidates = candidates
            logger.debug(f"Fuzzy-matched {len(candidates)} candidates for '{ent.text}'")

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

        _ensure_candidates_extension()

        # Initialize lazily
        self.kb = None
        self.entities = None
        self.bm25 = None
        self.corpus = None

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

        for ent in doc.ents:
            query_tokens = ent.text.lower().split()
            scores = self.bm25.get_scores(query_tokens)

            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:self.top_k]

            # Build candidates
            candidates = []
            for idx in top_indices:
                entity = self.entities[idx]
                candidates.append((entity.title, entity.description or ""))

            ent._.candidates = candidates
            logger.debug(f"BM25 retrieved {len(candidates)} candidates for '{ent.text}'")

        return doc
