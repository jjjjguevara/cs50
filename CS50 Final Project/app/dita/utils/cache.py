"""Enhanced cache system with pattern support and batch operations."""

from typing import Dict, Optional, Any, Union, Set, Type, TypeVar, List, Callable, Tuple
from datetime import datetime, timedelta
from threading import Lock
import logging
import re
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import json

from ..models.types import (
    ElementType,
    ProcessingPhase,
    ProcessingState,
    ContentScope,
    ValidationResult,
    ValidationMessage,
    ValidationSeverity,
    MetadataState,
    ProcessingMetadata,
    CacheEntryType
)

from .cache_patterns import (
    CACHE_PATTERNS,
    CacheScope,
    CacheStrategy
)


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
    pattern_key: Optional[str] = None

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def access(self) -> None:
        """Update last access time."""
        self.last_accessed = datetime.now()

class BatchOperation(Enum):
    """Types of batch operations."""
    SET = "set"
    DELETE = "delete"
    UPDATE = "update"

@dataclass
class BatchItem:
    """Item for batch processing."""
    operation: BatchOperation
    key: str
    entry_type: CacheEntryType
    data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None

class ContentCache:
    """Enhanced cache system with pattern support and batch operations."""

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[int] = 3600,
        validation_level: str = "strict",
        warmup: bool = True
    ):
        """Initialize the cache system."""
        self.logger = logging.getLogger(__name__)
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.validation_level = validation_level
        self._warmup = warmup

        # Initialize caches by type
        self._caches: Dict[CacheEntryType, Dict[str, CacheEntry]] = {
            entry_type: {} for entry_type in CacheEntryType
        }

        # Initialize pattern registry
        self._patterns = CACHE_PATTERNS

        # Batch operation settings
        self._batch_size: int = 100
        self._pending_operations: List[BatchItem] = []

        # Cache statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "invalidations": 0
        }

        # Locks for thread safety
        self._cache_lock = Lock()
        self._stats_lock = Lock()
        self._batch_lock = Lock()

        # Initialize warmup if enabled
        if self._warmup:
            self._warmup_cache()

    def _warmup_cache(self) -> None:
        """Warm up cache with frequently accessed data."""
        try:
            for entry_type, pattern in self._patterns.items():
                if pattern.config.warmup:
                    self._warmup_pattern(entry_type, pattern)
        except Exception as e:
            self.logger.error(f"Error during cache warmup: {str(e)}")

    def _warmup_pattern(self, entry_type: CacheEntryType, pattern: Any) -> None:
        """Warm up specific pattern data."""
        # Implementation depends on pattern-specific warmup logic
        pass

    def get(
        self,
        key: str,
        entry_type: CacheEntryType,
        default: Any = None,
        touch: bool = True
    ) -> Optional[Any]:
        """
        Retrieve cached data with pattern awareness.

        Args:
            key: Cache key
            entry_type: Type of cache entry
            default: Default value if key not found
            touch: Whether to update last access time
        """
        try:
            cache = self._caches[entry_type]
            entry = cache.get(key)

            if entry is None:
                with self._stats_lock:
                    self._stats["misses"] += 1
                return default

            if entry.is_expired():
                self.invalidate(key, entry_type)
                with self._stats_lock:
                    self._stats["misses"] += 1
                return default

            if touch:
                entry.access()

            with self._stats_lock:
                self._stats["hits"] += 1

            return entry.data

        except Exception as e:
            self.logger.error(f"Error retrieving from cache: {str(e)}")
            return default

    def set(
        self,
        key: str,
        data: Any,
        entry_type: CacheEntryType,
        element_type: ElementType,
        phase: ProcessingPhase,
        scope: ContentScope = ContentScope.LOCAL,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Set cache entry with pattern-based key generation."""
        with self._cache_lock:
            try:
                # Get pattern for entry type
                pattern = self._patterns.get(entry_type)
                if pattern and not pattern.should_cache(data):
                    return

                # Handle cache size limits
                if len(self._caches[entry_type]) >= self.max_size:
                    self._evict_entries(entry_type)

                # Calculate expiration
                expires_at = None
                if ttl is not None:
                    expires_at = datetime.now() + timedelta(seconds=ttl)
                elif pattern and pattern.get_ttl():
                    expires_at = datetime.now() + timedelta(seconds=pattern.get_ttl())

                # Create and store entry
                entry = CacheEntry(
                    data=data,
                    entry_type=entry_type,
                    element_type=element_type,
                    phase=phase,
                    scope=scope,
                    expires_at=expires_at,
                    metadata=metadata or {},
                    pattern_key=pattern.generate_key(**metadata) if pattern and metadata else None
                )

                self._caches[entry_type][key] = entry

            except Exception as e:
                self.logger.error(f"Error setting cache entry for key {key}: {str(e)}")

    def invalidate(self, key: str, entry_type: CacheEntryType) -> None:
        """Invalidate cache entry with pattern consideration."""
        try:
            cache = self._caches[entry_type]
            if entry := cache.pop(key, None):
                # Handle pattern-specific invalidation
                if pattern := self._patterns.get(entry_type):
                    self._handle_pattern_invalidation(pattern, entry)

                with self._stats_lock:
                    self._stats["invalidations"] += 1

        except Exception as e:
            self.logger.error(f"Error invalidating cache entry: {str(e)}")

    def _handle_pattern_invalidation(self, pattern: Any, entry: CacheEntry) -> None:
        """Handle pattern-specific invalidation logic."""
        if pattern.config.dependencies and entry.pattern_key:
            # Invalidate dependent entries
            for dep_type, dep_pattern in pattern.config.dependencies.items():
                self.invalidate_by_pattern(dep_pattern)

    def batch_operation(self, operations: List[BatchItem]) -> None:
        """Process batch operations atomically."""
        with self._batch_lock:
            try:
                self._pending_operations.extend(operations)

                if len(self._pending_operations) >= self._batch_size:
                    self._process_batch()

            except Exception as e:
                self.logger.error(f"Error in batch operation: {str(e)}")
                self._pending_operations.clear()

    def _evict_entries(
        self,
        entry_type: CacheEntryType,
        count: int = 100
    ) -> None:
        """
        Evict entries based on pattern strategy.

        Args:
            entry_type: Type of entries to evict
            count: Number of entries to evict
        """
        try:
            cache = self._caches[entry_type]
            if len(cache) <= self.max_size - count:
                return

            pattern = self._patterns.get(entry_type)
            if not pattern:
                return

            with self._cache_lock:
                entries = list(cache.items())

                # Apply eviction strategy
                if pattern.config.strategy == CacheStrategy.LRU:
                    # Sort by last accessed time
                    entries.sort(key=lambda x: x[1].last_accessed)
                elif pattern.config.strategy == CacheStrategy.FIFO:
                    # Sort by creation time
                    entries.sort(key=lambda x: x[1].created_at)
                elif pattern.config.strategy == CacheStrategy.LIFO:
                    # Sort by creation time in reverse
                    entries.sort(key=lambda x: x[1].created_at, reverse=True)
                elif pattern.config.strategy == CacheStrategy.TTL:
                    # Sort by expiration time
                    entries.sort(key=lambda x: x[1].expires_at or datetime.max)

                # Remove oldest entries
                for key, entry in entries[:count]:
                    cache.pop(key)
                    with self._stats_lock:
                        self._stats["evictions"] += 1

                    # Handle pattern-specific eviction
                    self._handle_pattern_invalidation(pattern, entry)

        except Exception as e:
            self.logger.error(f"Error evicting cache entries: {str(e)}")

    def invalidate_by_pattern(
        self,
        pattern: str,
        entry_type: Optional[CacheEntryType] = None
    ) -> None:
        """
        Invalidate cache entries matching a pattern.

        Args:
            pattern: Pattern to match against cache keys
            entry_type: Optional type to restrict invalidation to
        """
        try:
            regex = re.compile(pattern.replace('*', '.*'))

            with self._cache_lock:
                # Track keys to invalidate
                invalidation_batch: List[Tuple[CacheEntryType, str]] = []

                # Collect all matching keys
                for cache_type, cache in self._caches.items():
                    if entry_type and cache_type != entry_type:
                        continue

                    # Find matching entries
                    matching_entries = [
                        (cache_type, key)
                        for key, entry in cache.items()
                        if regex.match(key) or (
                            entry.pattern_key and
                            regex.match(entry.pattern_key)
                        )
                    ]

                    invalidation_batch.extend(matching_entries)

                # Process invalidations in batch
                if invalidation_batch:
                    for batch in self._chunk_batch(invalidation_batch, self._batch_size):
                        self._process_invalidation_batch(batch)

                    self.logger.debug(
                        f"Invalidated {len(invalidation_batch)} entries matching pattern: {pattern}"
                    )

        except re.error as regex_error:
            self.logger.error(f"Invalid regex pattern {pattern}: {regex_error}")
        except Exception as e:
            self.logger.error(f"Error invalidating pattern {pattern}: {str(e)}")

    def _chunk_batch(
        self,
        items: List[Tuple[CacheEntryType, str]],
        size: int
    ) -> List[List[Tuple[CacheEntryType, str]]]:
        """Split batch into chunks."""
        return [items[i:i + size] for i in range(0, len(items), size)]

    def _process_invalidation_batch(
        self,
        batch: List[Tuple[CacheEntryType, str]]
    ) -> None:
        """Process a batch of invalidations atomically."""
        try:
            with self._cache_lock:
                for entry_type, key in batch:
                    cache = self._caches[entry_type]
                    if entry := cache.pop(key, None):
                        # Handle pattern-specific invalidation
                        if pattern := self._patterns.get(entry_type):
                            self._handle_pattern_invalidation(pattern, entry)

                        with self._stats_lock:
                            self._stats["invalidations"] += 1

        except Exception as e:
            self.logger.error(f"Error processing invalidation batch: {str(e)}")

    def _process_batch(self) -> None:
        """Process pending batch operations."""
        with self._cache_lock:
            try:
                for op in self._pending_operations:
                    if op.operation == BatchOperation.SET:
                        # Ensure metadata exists
                        metadata = op.metadata or {}
                        element_type = metadata.get('element_type')
                        phase = metadata.get('phase')

                        if not (element_type and phase):
                            self.logger.error(
                                f"Missing required parameters for batch SET operation: {op.key}"
                            )
                            continue

                        self.set(
                            op.key,
                            op.data,
                            op.entry_type,
                            element_type=element_type,
                            phase=phase,
                            metadata=metadata
                        )
                    elif op.operation == BatchOperation.DELETE:
                        self.invalidate(op.key, op.entry_type)
                    elif op.operation == BatchOperation.UPDATE:
                        cache = self._caches.get(op.entry_type, {})
                        if entry := cache.get(op.key):
                            entry.data = op.data
                            if op.metadata:
                                entry.metadata.update(op.metadata)

            finally:
                self._pending_operations.clear()



    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._stats_lock:
            return self._stats.copy()

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._cache_lock:
            self._caches = {
                entry_type: {} for entry_type in CacheEntryType
            }
            with self._stats_lock:
                self._stats = {k: 0 for k in self._stats}
