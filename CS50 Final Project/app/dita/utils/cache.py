from typing import Dict, Optional, Any, Union, Set, Type, TypeVar
from datetime import datetime, timedelta
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
    """
    Enhanced cache system with validation and scope awareness.
    Handles both transient and persistent caching with TTL.
    """
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[int] = 3600,  # 1 hour default
        validation_level: str = "strict"
    ):
        # Core initialization
        self.logger = logging.getLogger(__name__)
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.validation_level = validation_level

        # Cache storage by type
        self._caches: Dict[CacheEntryType, Dict[str, CacheEntry]] = {
            entry_type: {} for entry_type in CacheEntryType
        }

        # Validation patterns
        self._validation_patterns: Dict[str, re.Pattern] = {}

        # Scope tracking
        self._scope_entries: Dict[ContentScope, Set[str]] = {
            ContentScope.LOCAL: set(),
            ContentScope.PEER: set(),
            ContentScope.EXTERNAL: set(),
            ContentScope.GLOBAL: set()
        }

        # Statistics tracking
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "invalidations": 0
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
        """
        try:
            cache = self._caches[entry_type]
            entry = cache.get(key)

            if entry is None:
                self._stats["misses"] += 1
                return default

            if entry.is_expired():
                self.invalidate(key, entry_type)
                self._stats["invalidations"] += 1
                return default

            if touch:
                entry.access()

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
        """
        Cache data with type and scope awareness.

        Args:
            key: Cache key
            data: Data to cache
            entry_type: Type of cache entry
            element_type: Type of element being cached
            phase: Processing phase of cached data
            scope: Content scope
            ttl: Time to live in seconds
            metadata: Optional metadata
        """
        try:
            # Check cache size limits
            cache = self._caches[entry_type]
            if len(cache) >= self.max_size:
                self._evict_entries(entry_type)

            # Calculate expiration
            ttl_seconds = ttl if ttl is not None else self.default_ttl
            expires_at = (
                datetime.now() + timedelta(seconds=float(ttl_seconds))
                if ttl_seconds is not None
                else None
            )

            # Create entry
            entry = CacheEntry(
                data=data,
                entry_type=entry_type,
                element_type=element_type,
                phase=phase,
                scope=scope,
                expires_at=expires_at,
                validation_level=self.validation_level,
                metadata=metadata or {}
            )

            # Store entry
            cache[key] = entry
            self._scope_entries[scope].add(key)

        except Exception as e:
            self.logger.error(f"Error setting cache entry: {str(e)}")

    def invalidate(self, key: str, entry_type: CacheEntryType) -> None:
        """
        Invalidate a specific cache entry.

        Args:
            key: Cache key
            entry_type: Type of cache entry
        """
        try:
            cache = self._caches[entry_type]
            if entry := cache.pop(key, None):
                # Remove from scope tracking
                self._scope_entries[entry.scope].discard(key)
                self._stats["invalidations"] += 1

        except Exception as e:
            self.logger.error(f"Error invalidating cache key {key}: {str(e)}")

    def invalidate_by_type(
        self,
        element_type: ElementType,
        phase: Optional[ProcessingPhase] = None,
        entry_type: Optional[CacheEntryType] = None
    ) -> None:
        """
        Invalidate entries by element type and optional phase.

        Args:
            element_type: Type of elements to invalidate
            phase: Optional phase to limit invalidation
            entry_type: Optional type of entries to invalidate
        """
        try:
            cache_types = (
                [entry_type] if entry_type
                else list(CacheEntryType)
            )

            for cache_type in cache_types:
                cache = self._caches[cache_type]
                keys_to_remove = [
                    key for key, entry in cache.items()
                    if entry.element_type == element_type
                    and (phase is None or entry.phase == phase)
                ]

                for key in keys_to_remove:
                    self.invalidate(key, cache_type)

        except Exception as e:
            self.logger.error(
                f"Error invalidating cache for type {element_type}: {str(e)}"
            )

    def invalidate_by_scope(
        self,
        scope: ContentScope,
        entry_type: Optional[CacheEntryType] = None
    ) -> None:
        """
        Invalidate entries by scope.

        Args:
            scope: Scope to invalidate
            entry_type: Optional type of entries to invalidate
        """
        try:
            scope_keys = self._scope_entries[scope].copy()
            cache_types = (
                [entry_type] if entry_type
                else list(CacheEntryType)
            )

            for key in scope_keys:
                for cache_type in cache_types:
                    self.invalidate(key, cache_type)

        except Exception as e:
            self.logger.error(f"Error invalidating scope {scope}: {str(e)}")

    def invalidate_pattern(
        self,
        pattern: str,
        entry_type: Optional[CacheEntryType] = None
    ) -> None:
        """
        Invalidate entries matching a pattern.

        Args:
            pattern: Pattern to match against cache keys
            entry_type: Optional type of entries to invalidate
        """
        try:
            pattern_regex = re.compile(pattern.replace('*', '.*'))
            cache_types = (
                [entry_type] if entry_type
                else list(CacheEntryType)
            )

            for cache_type in cache_types:
                cache = self._caches[cache_type]
                keys_to_remove = [
                    key for key in cache.keys()
                    if pattern_regex.match(key)
                ]

                for key in keys_to_remove:
                    self.invalidate(key, cache_type)

        except Exception as e:
            self.logger.error(
                f"Error invalidating cache pattern {pattern}: {str(e)}"
            )

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return self._stats.copy()

    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            for cache_type in CacheEntryType:
                self._caches[cache_type].clear()
            for scope in ContentScope:
                self._scope_entries[scope].clear()
            self._stats["invalidations"] += 1
            self.logger.debug("Cache cleared")

        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")

    def _evict_entries(
        self,
        entry_type: CacheEntryType,
        count: int = 100
    ) -> None:
        """
        Evict least recently used entries of a specific type.

        Args:
            entry_type: Type of entries to evict
            count: Number of entries to evict
        """
        try:
            cache = self._caches[entry_type]
            if len(cache) <= self.max_size - count:
                return

            # Sort by last accessed time
            sorted_entries = sorted(
                cache.items(),
                key=lambda x: x[1].last_accessed
            )

            # Remove oldest entries
            for key, entry in sorted_entries[:count]:
                self.invalidate(key, entry_type)
                self._stats["evictions"] += 1

        except Exception as e:
            self.logger.error(f"Error evicting cache entries: {str(e)}")

    def _validate_pattern(self, key: str, pattern: str) -> bool:
        """
        Validate a key against a pattern.

        Args:
            key: Key to validate
            pattern: Pattern to validate against

        Returns:
            bool: True if key matches pattern
        """
        try:
            if pattern not in self._validation_patterns:
                self._validation_patterns[pattern] = re.compile(pattern)

            return bool(self._validation_patterns[pattern].match(key))

        except Exception as e:
            self.logger.error(f"Error validating pattern: {str(e)}")
            return False
