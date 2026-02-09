import hashlib
import json
import logging
import os
import pickle
import threading
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Any

from rapidfuzz import process

from el_pipeline.registry import knowledge_bases
from el_pipeline.types import Entity
from el_pipeline.knowledge_bases.base import KnowledgeBase

logger = logging.getLogger(__name__)


# ============================================================================
# In-Memory KB Cache
# ============================================================================

_kb_cache: Dict[str, "CustomJSONLKnowledgeBase"] = {}  # identity_hash -> KB instance


def _compute_kb_identity_hash(path: str) -> str:
    """Compute identity hash for a KB file (path + mtime + size)."""
    stat = os.stat(path)
    raw = f"kb:{path}:{stat.st_mtime}:{stat.st_size}".encode()
    return hashlib.sha256(raw).hexdigest()


def get_cached_kb(path: str) -> Optional["CustomJSONLKnowledgeBase"]:
    """Get a KB from the in-memory cache if it exists and is still valid."""
    try:
        identity_hash = _compute_kb_identity_hash(path)
        if identity_hash in _kb_cache:
            logger.info(f"Reusing cached KB from memory: {path}")
            return _kb_cache[identity_hash]
    except Exception:
        pass
    return None


def cache_kb(kb: "CustomJSONLKnowledgeBase") -> None:
    """Add a KB to the in-memory cache."""
    try:
        _kb_cache[kb.identity_hash] = kb
        logger.debug(f"KB cached in memory: {kb.source_path}")
    except Exception as e:
        logger.warning(f"Failed to cache KB in memory: {e}")


def get_kb_cache_info() -> List[Dict[str, Any]]:
    """Get information about currently cached KBs."""
    return [
        {
            "path": kb.source_path,
            "identity_hash": kb.identity_hash[:12],
            "entity_count": len(kb.entities),
        }
        for kb in _kb_cache.values()
    ]


def clear_kb_cache() -> None:
    """Clear the in-memory KB cache."""
    _kb_cache.clear()
    logger.info("KB cache cleared")


@knowledge_bases.register("custom")
class CustomJSONLKnowledgeBase:
    """
    Loads entities from a JSONL file.

    Supports two formats:
    - Full format: {"id": "...", "title": "...", "description": "..."}
    - Simple format: {"title": "...", "description": "..."} (id defaults to title)

    The 'id' field is optional - if not provided, 'title' is used as the ID.
    
    Memory management:
    - KBs are cached in memory and reused between pipeline runs
    - Use get_cached_kb() to check if a KB is already loaded
    - The cache is invalidated if the source file changes (mtime/size)
    """

    def __new__(
        cls,
        path: str,
        cache_dir: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        # Check in-memory cache first
        cached = get_cached_kb(path)
        if cached is not None:
            return cached
        
        # Create new instance
        instance = super().__new__(cls)
        return instance

    def __init__(
        self,
        path: str,
        cache_dir: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        # Skip initialization if we got a cached instance
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.source_path = path
        self.entities: Dict[str, Entity] = {}  # Indexed by ID
        self.titles: List[str] = []
        self._cancel_event = cancel_event
        self._progress_callback = progress_callback

        # Try loading from disk cache
        if cache_dir and self._load_from_cache(cache_dir):
            self._initialized = True
            cache_kb(self)  # Add to in-memory cache
            return

        self._parse_jsonl(path)

        # Save to disk cache
        if cache_dir:
            self._save_to_cache(cache_dir)
        
        self._initialized = True
        cache_kb(self)  # Add to in-memory cache

    def _check_cancelled(self):
        """Raise InterruptedError if cancellation was requested."""
        if self._cancel_event and self._cancel_event.is_set():
            raise InterruptedError("KB loading cancelled")

    @property
    def identity_hash(self) -> str:
        """Content-based hash for cache invalidation (uses path + mtime + size)."""
        stat = os.stat(self.source_path)
        raw = f"kb:{self.source_path}:{stat.st_mtime}:{stat.st_size}".encode()
        return hashlib.sha256(raw).hexdigest()

    def _cache_path(self, cache_dir: str) -> Path:
        """Return the cache file path for this KB."""
        kb_dir = Path(cache_dir) / "kb"
        kb_dir.mkdir(parents=True, exist_ok=True)
        return kb_dir / f"{self.identity_hash}.pkl"

    def _load_from_cache(self, cache_dir: str) -> bool:
        """Try to load parsed KB data from cache. Returns True on success."""
        try:
            self._check_cancelled()
            cache_file = self._cache_path(cache_dir)
            if not cache_file.exists():
                return False
            if self._progress_callback:
                self._progress_callback(0.0, "Loading KB from cache...")
            with cache_file.open("rb") as f:
                entities, titles = pickle.load(f)
            self._check_cancelled()
            self.entities = entities
            self.titles = titles
            logger.info(
                f"Loaded {len(self.entities)} entities from cache ({cache_file.name})"
            )
            return True
        except InterruptedError:
            raise  # Re-raise cancellation
        except Exception:
            logger.warning("KB cache load failed, will rebuild from JSONL", exc_info=True)
            return False

    def _save_to_cache(self, cache_dir: str) -> None:
        """Save parsed KB data to cache."""
        try:
            cache_file = self._cache_path(cache_dir)
            with cache_file.open("wb") as f:
                pickle.dump((self.entities, self.titles), f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved KB cache ({cache_file.name})")
        except Exception:
            logger.warning("Failed to save KB cache", exc_info=True)

    def _parse_jsonl(self, path: str) -> None:
        """Parse the JSONL file and populate entities and titles."""
        # Get file size for progress reporting
        file_size = os.path.getsize(path)
        bytes_read = 0
        last_progress_report = 0
        check_interval = 50000  # Check cancellation every N lines

        with Path(path).open(encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                # Check for cancellation periodically
                if line_num % check_interval == 0:
                    self._check_cancelled()
                    # Report progress
                    if self._progress_callback and file_size > 0:
                        progress = bytes_read / file_size
                        if progress - last_progress_report >= 0.05:  # Report every 5%
                            self._progress_callback(progress, f"Loading KB: {len(self.entities):,} entities...")
                            last_progress_report = progress

                bytes_read += len(line.encode("utf-8"))
                line = line.strip()
                if not line:
                    continue
                # Handle YAGO N-Triples trailing "\t."
                if line.endswith("\t."):
                    line = line[:-2].rstrip()
                item = json.loads(line)
                # ID is optional - fall back to title if not provided
                entity_id = item.get("id") or item.get("title")
                entity_title = item.get("title") or item.get("id")
                if not entity_id or not entity_title:
                    logger.warning(f"Skipping entity without id or title: {item}")
                    continue
                entity = Entity(
                    id=entity_id,
                    title=entity_title,
                    description=item.get("description"),
                    metadata={k: v for k, v in item.items() if k not in {"id", "title", "description"}},
                )
                self.entities[entity.id] = entity
                self.titles.append(entity.title)
        logger.info(f"Loaded {len(self.entities)} entities from {path}")

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)

    def search(self, query: str, top_k: int = 10) -> List[Entity]:
        """Fuzzy search entities by title."""
        if not self.titles:
            return []
        results = process.extract(query, self.titles, limit=top_k)
        hits: List[Entity] = []
        for _, _, idx in results:
            title = self.titles[idx]
            # Find entity with this title
            for entity in self.entities.values():
                if entity.title == title:
                    hits.append(entity)
                    break
        return hits

    def all_entities(self) -> Iterable[Entity]:
        return self.entities.values()

