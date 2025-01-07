# app/dita/utils/cache.py

from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import logging
import re
from pathlib import Path
import json

from ..models.types import ContentElement, ProcessingPhase, ElementType

class CacheEntry:
    """Individual cache entry with metadata."""
    def __init__(
        self,
        data: Any,
        element_type: ElementType,
        phase: ProcessingPhase,
        expires_at: Optional[datetime] = None
    ):
        self.data = data
        self.element_type = element_type
        self.phase = phase
        self.created_at = datetime.now()
        self.expires_at = expires_at
        self.last_accessed = datetime.now()

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return (
            self.expires_at is not None
            and datetime.now() > self.expires_at
        )

    def access(self) -> None:
        """Update last access time."""
        self.last_accessed = datetime.now()

class ContentCache:
    """
    Cache system for processed content.
    Handles both transient and semi-persistent caching with TTL.
    """
    def __init__(self, max_size: int = 1000):
        self.logger = logging.getLogger(__name__)
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}

    def get(
        self,
        key: str,
        default: Any = None,
        touch: bool = True
    ) -> Optional[Any]:
        """
        Retrieve cached data with optional access tracking.

        Args:
            key: Cache key
            default: Default value if key not found
            touch: Whether to update last access time

        Returns:
            Cached data or default value
        """
        try:
            entry = self._cache.get(key)
            if entry is None:
                return default

            if entry.is_expired():
                self.invalidate(key)
                return default

            if touch:
                entry.access()

            return entry.data

        except Exception as e:
            self.logger.error(f"Error retrieving from cache: {str(e)}")
            return default

    def set(
        self,
        key: str,
        data: Any,
        element_type: ElementType,
        phase: ProcessingPhase,
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache data with optional TTL in seconds.

        Args:
            key: Cache key
            data: Data to cache
            element_type: Type of element being cached
            phase: Processing phase of cached data
            ttl: Time to live in seconds
        """
        try:
            # Enforce cache size limit
            if len(self._cache) >= self.max_size:
                self._evict_entries()

            expires_at = (
                datetime.now() + timedelta(seconds=ttl)
                if ttl is not None else None
            )

            self._cache[key] = CacheEntry(
                data=data,
                element_type=element_type,
                phase=phase,
                expires_at=expires_at
            )

        except Exception as e:
            self.logger.error(f"Error setting cache entry: {str(e)}")

    def invalidate(self, key: str) -> None:
        """Invalidate a specific cache entry."""
        try:
            self._cache.pop(key, None)
        except Exception as e:
            self.logger.error(f"Error invalidating cache key {key}: {str(e)}")

    def invalidate_by_type(
        self,
        element_type: ElementType,
        phase: Optional[ProcessingPhase] = None
    ) -> None:
        """
        Invalidate all entries of a specific type and optional phase.

        Args:
            element_type: Type of elements to invalidate
            phase: Optional phase to limit invalidation
        """
        try:
            keys_to_remove = [
                key for key, entry in self._cache.items()
                if entry.element_type == element_type
                and (phase is None or entry.phase == phase)
            ]

            for key in keys_to_remove:
                self.invalidate(key)

        except Exception as e:
            self.logger.error(
                f"Error invalidating cache for type {element_type}: {str(e)}"
            )

    def invalidate_pattern(self, pattern: str) -> None:
        """
        Invalidate all cache entries matching a pattern.

        Args:
            pattern: Pattern to match against cache keys
        """
        try:
            # Use regex pattern matching
            pattern_regex = re.compile(pattern.replace('*', '.*'))

            # Find matching keys
            keys_to_remove = [
                key for key in self._cache.keys()
                if pattern_regex.match(key)
            ]

            # Remove matching entries
            for key in keys_to_remove:
                self._cache.pop(key, None)

                # Log invalidation
                self.logger.debug(f"Invalidated cache key: {key}")

        except Exception as e:
            self.logger.error(f"Error invalidating cache pattern {pattern}: {str(e)}")
            raise

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def _evict_entries(self, count: int = 100) -> None:
        """
        Evict least recently used entries.

        Args:
            count: Number of entries to evict
        """
        try:
            if len(self._cache) <= self.max_size - count:
                return

            # Sort by last accessed time
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].last_accessed
            )

            # Remove oldest entries
            for key, _ in sorted_entries[:count]:
                self.invalidate(key)

        except Exception as e:
            self.logger.error(f"Error evicting cache entries: {str(e)}")
