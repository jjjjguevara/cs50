from typing import Dict, Optional, Any, Union, Set, Type, TypeVar, List
from datetime import datetime, timedelta
from threading import Lock
import logging
import re
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import json

# Custom types
from ..models.types import (
    ElementType,
    ProcessingPhase,
    ProcessingState,
    ContentScope,
    ValidationResult,
    ValidationMessage,
    ValidationSeverity,
    MetadataState,
    ProcessingMetadata
)

class CacheEntryType(Enum):
    """Types of cache entries for specialized handling."""
    CONTENT = "content"          # Processed content
    METADATA = "metadata"        # Metadata entries
    TRANSFORM = "transform"      # Transformation results
    VALIDATION = "validation"    # Validation results
    REFERENCE = "reference"      # Reference lookups
    STATE = "state"             # State information
    FEATURE = "feature"         # Feature flags

@dataclass
class CacheEntry:
    """Individual cache entry with metadata tracking."""
    data: Any
    entry_type: CacheEntryType
    element_type: ElementType
    phase: ProcessingPhase
    scope: ContentScope
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    last_accessed: datetime = field(default_factory=datetime.now)
    validation_level: str = "strict"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def access(self) -> None:
        """Update last access time."""
        self.last_accessed = datetime.now()

class ContentCache:
    """Enhanced cache system with validation and scope awareness."""

    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = 3600, validation_level: str = "strict"):
            """Initialize the cache system."""
            self.logger = logging.getLogger(__name__)
            self.max_size = max_size
            self.default_ttl = default_ttl
            self.validation_level = validation_level

            # Unified initialization for caches and scope entries
            self._scope_entries = {scope: set() for scope in ContentScope}

            self._invalidation_in_progress: Set[str] = set()
            self._invalidation_depth: int = 0
            self._max_invalidation_depth: int = 10
            self._batch_size: int = 100

            # Lock for synchronization
            self._cache_lock = Lock()

            # Validation patterns and stats
            self._validation_patterns = {}
            self._stats = {"hits": 0, "misses": 0, "evictions": 0, "invalidations": 0}

            # Cache storage by type
            self._caches: Dict[CacheEntryType, Dict[str, CacheEntry]] = {
                entry_type: {} for entry_type in CacheEntryType
            }

    def get(
            self,
            key: str,
            entry_type: CacheEntryType,
            default: Any = None,
            touch: bool = True
        ) -> Optional[Any]:
            """
            Retrieve cached data with type checking.

            Args:
                key: Cache key
                entry_type: Type of cache entry
                default: Default value if key not found
                touch: Whether to update last access time

            Returns:
                Cached data if present, otherwise default.
            """
            try:
                cache = self._caches[entry_type]
                entry = cache.get(key)

                if entry is None:
                    return default

                if entry.is_expired():
                    self.invalidate(key, entry_type)
                    return default

                if touch:
                    entry.access()

                return entry.data

            except Exception as e:
                self.logger.error(f"Error retrieving from cache: {str(e)}")
                return default

    def _calculate_expiration(self, ttl: Optional[int]) -> Optional[datetime]:
        """Calculate expiration datetime based on TTL."""
        if ttl is None:
            ttl = self.default_ttl

        # Ensure ttl is not None before passing to timedelta
        if ttl is not None:
            return datetime.now() + timedelta(seconds=float(ttl))
        return None

    def _evict_entries(self, entry_type: CacheEntryType, count: int = 100) -> None:
        """Evict least recently used entries of a specific type."""
        try:
            cache = self._caches[entry_type]
            if len(cache) <= self.max_size - count:
                return

            # Sort by last accessed time
            sorted_entries = sorted(cache.items(), key=lambda x: x[1].last_accessed)

            # Remove oldest entries
            for key, _ in sorted_entries[:count]:
                self.invalidate(key, entry_type)
                self._stats["evictions"] += 1

        except Exception as e:
            self.logger.error(f"Error evicting cache entries: {str(e)}")

    def set(self, key, data, entry_type, element_type, phase, scope=ContentScope.LOCAL, ttl=None, metadata=None):
        """Cache data with type and scope awareness."""
        with self._cache_lock:
            try:
                # Evict if size exceeds max
                if len(self._caches[entry_type]) >= self.max_size:
                    self._evict_entries(entry_type)

                # Create and store cache entry
                entry = CacheEntry(
                    data=data,
                    entry_type=entry_type,
                    element_type=element_type,
                    phase=phase,
                    scope=scope,
                    expires_at=self._calculate_expiration(ttl),
                    validation_level=self.validation_level,
                    metadata=metadata or {}
                )
                self._caches[entry_type][key] = entry
                self._scope_entries[scope].add(key)

                # Log successful cache set
                self.logger.debug(f"Cache entry set for key: {key}, entry_type: {entry_type}")
            except Exception as e:
                self.logger.error(f"Error setting cache entry for key {key}: {str(e)}")

    def invalidate(self, key: str, entry_type: CacheEntryType) -> None:
            """Invalidate cache entry with recursion protection."""
            try:
                # Check recursion depth
                if self._invalidation_depth >= self._max_invalidation_depth:
                    self.logger.warning(f"Maximum invalidation depth reached for key: {key}")
                    return

                # Check if already invalidating
                invalidation_key = f"{key}_{entry_type.value}"
                if invalidation_key in self._invalidation_in_progress:
                    return

                self._invalidation_in_progress.add(invalidation_key)
                self._invalidation_depth += 1

                try:
                    cache = self._caches[entry_type]
                    if entry := cache.pop(key, None):
                        self._scope_entries[entry.scope].discard(key)
                        self._stats["invalidations"] += 1
                finally:
                    self._invalidation_in_progress.remove(invalidation_key)
                    self._invalidation_depth -= 1

            except Exception as e:
                self.logger.error(f"Error invalidating cache entry: {str(e)}")

    def sync_with_metadata(self, content_id: str, updates: Dict[str, Any]):
        """Synchronize cache with metadata storage."""
        try:
            self.set(
                key=f"metadata_{content_id}",
                data=updates,
                entry_type=CacheEntryType.METADATA,
                element_type=ElementType.UNKNOWN,
                phase=ProcessingPhase.DISCOVERY
            )
        except Exception as e:
            self.logger.error(f"Error synchronizing cache with metadata: {str(e)}")

    def invalidate_by_pattern(self, pattern: str, entry_type: Optional[CacheEntryType] = None) -> None:
        """Invalidate cache entries matching a pattern using batch processing."""
        try:
            regex = re.compile(pattern.replace('*', '.*'))
            cache_types = [entry_type] if entry_type else list(CacheEntryType)

            for cache_type in cache_types:
                # Collect all keys first
                cache = self._caches[cache_type]
                keys_to_invalidate = {
                    key for key in cache
                    if regex.match(key) and
                    f"{key}_{cache_type.value}" not in self._invalidation_in_progress
                }

                # Process in batches
                for i in range(0, len(keys_to_invalidate), self._batch_size):
                    batch = list(keys_to_invalidate)[i:i + self._batch_size]
                    self._process_invalidation_batch(batch, cache_type)

                self.logger.debug(f"Invalidated {len(keys_to_invalidate)} keys for pattern {pattern}")

        except re.error as regex_error:
            self.logger.error(f"Invalid regex pattern {pattern}: {regex_error}")
        except Exception as e:
            self.logger.error(f"Error invalidating pattern {pattern}: {str(e)}")


    def _process_invalidation_batch(self, keys: List[str], entry_type: CacheEntryType) -> None:
            """Process a batch of invalidations atomically."""
            try:
                with self._cache_lock:
                    cache = self._caches[entry_type]
                    for key in keys:
                        if entry := cache.pop(key, None):
                            self._scope_entries[entry.scope].discard(key)
                            self._stats["invalidations"] += 1
            except Exception as e:
                self.logger.error(f"Error processing invalidation batch: {str(e)}")

    def clear(self) -> None:
        """Clear all cache entries safely."""
        try:
            with self._cache_lock:
                self._caches = {entry_type: {} for entry_type in CacheEntryType}
                self._scope_entries = {scope: set() for scope in ContentScope}
                self._invalidation_in_progress.clear()
                self._invalidation_depth = 0
                self._stats["invalidations"] += 1
                self.logger.debug("Cache cleared")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
