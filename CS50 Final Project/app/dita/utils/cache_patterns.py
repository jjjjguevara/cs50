"""Cache pattern definitions for DITA processing."""
from typing import Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from datetime import timedelta

from app.dita.models.types import (
    ElementType,
    ProcessingPhase,
    ContentScope,
    CacheEntryType
)

class CacheScope(Enum):
    """Cache scope definitions."""
    REQUEST = "request"      # Valid for single request
    SESSION = "session"      # Valid for user session
    APPLICATION = "app"      # Valid across application
    PERSISTENT = "persist"   # Persisted to disk/storage

class CacheStrategy(Enum):
    """Cache invalidation strategies."""
    LRU = "lru"           # Least Recently Used
    FIFO = "fifo"         # First In First Out
    LIFO = "lifo"         # Last In First Out
    TTL = "ttl"           # Time To Live

@dataclass
class CachePatternConfig:
    """Configuration for cache pattern."""
    scope: CacheScope
    strategy: CacheStrategy
    ttl: Optional[timedelta] = None
    max_size: Optional[int] = None
    warmup: bool = False
    dependencies: Dict[str, str] = field(default_factory=dict)


class CachePattern:
    """Base class for cache patterns."""

    def __init__(
        self,
        config: CachePatternConfig
    ):
        self.config = config

    def generate_key(self, **kwargs: Any) -> str:
        """
        Generate cache key from parameters.

        Args:
            **kwargs: Key parameters

        Returns:
            str: Generated cache key
        """
        raise NotImplementedError

    def should_cache(self, value: Any) -> bool:
        """Determine if value should be cached."""
        return True

    def get_ttl(self) -> Optional[int]:
        """Get TTL in seconds."""
        if self.config.ttl:
            return int(self.config.ttl.total_seconds())
        return None

class ContentCachePattern(CachePattern):
    """Pattern for content caching."""

    def generate_key(self, **kwargs: Any) -> str:
        element_id = kwargs.get('element_id')
        element_type = kwargs.get('element_type')
        phase = kwargs.get('phase')

        if not all([element_id, element_type, phase]):
            raise ValueError("Missing required parameters for content cache key")

        element_type_value = element_type.value if element_type else 'unknown'
        phase_value = phase.value if phase else 'unknown'

        return f"content:{element_type_value}:{element_id}:{phase_value}"

class MetadataCachePattern(CachePattern):
    """Pattern for metadata caching."""

    def generate_key(self, **kwargs: Any) -> str:
        content_id = kwargs.get('content_id')
        scope = kwargs.get('scope')

        if not all([content_id, scope]):
            raise ValueError("Missing required parameters for metadata cache key")

        scope_value = scope.value if scope else 'unknown'

        return f"metadata:{content_id}:{scope_value}"

    def should_cache(self, value: Any) -> bool:
        """Only cache non-empty metadata."""
        return bool(value)

class ValidationCachePattern(CachePattern):
    """Pattern for validation results."""

    def generate_key(self, **kwargs: Any) -> str:
        element_id = kwargs.get('element_id')
        phase = kwargs.get('phase')
        context_id = kwargs.get('context_id')

        if not all([element_id, phase]):
            raise ValueError("Missing required parameters for validation cache key")

        phase_value = phase.value if phase else 'unknown'
        base = f"validation:{element_id}:{phase_value}"

        if context_id:
            return f"{base}:{context_id}"
        return base

class TransformationCachePattern(CachePattern):
    """Pattern for transformation results."""

    def generate_key(self, **kwargs: Any) -> str:
        element_id = kwargs.get('element_id')
        phase = kwargs.get('phase')
        element_type = kwargs.get('element_type')

        if not all([element_id, phase, element_type]):
            raise ValueError("Missing required parameters for transformation cache key")

        element_type_value = element_type.value if element_type else 'unknown'
        phase_value = phase.value if phase else 'unknown'

        return f"transform:{element_type_value}:{element_id}:{phase_value}"

class SchemaCachePattern(CachePattern):
    """Pattern for schema caching."""

    def generate_key(self, **kwargs: Any) -> str:
        schema_name = kwargs.get('schema_name')
        version = kwargs.get('version')

        if not schema_name:
            raise ValueError("Missing required parameter schema_name for schema cache key")

        if version:
            return f"schema:{schema_name}:{version}"
        return f"schema:{schema_name}"

class RuleCachePattern(CachePattern):
    """Pattern for rule resolution caching."""

    def generate_key(self, **kwargs: Any) -> str:
        element_type = kwargs.get('element_type')
        rule_type = kwargs.get('rule_type')
        context_id = kwargs.get('context_id')

        if not all([element_type, rule_type]):
            raise ValueError("Missing required parameters for rule cache key")

        element_type_value = element_type.value if element_type else 'unknown'
        base = f"rule:{element_type_value}:{rule_type}"

        if context_id:
            return f"{base}:{context_id}"
        return base

class KeyRefCachePattern(CachePattern):
    """Pattern for key reference caching."""

    def generate_key(self, **kwargs: Any) -> str:
        key = kwargs.get('key')
        scope = kwargs.get('scope')
        map_id = kwargs.get('map_id')

        if not all([key, scope]):
            raise ValueError("Missing required parameters for keyref cache key")

        scope_value = scope.value if scope else 'unknown'
        base = f"keyref:{key}:{scope_value}"

        if map_id:
            return f"{base}:{map_id}"
        return base

# Pattern registry
CACHE_PATTERNS = {
    CacheEntryType.CONTENT: ContentCachePattern(
        CachePatternConfig(
            scope=CacheScope.APPLICATION,
            strategy=CacheStrategy.LRU,
            ttl=timedelta(hours=1),
            max_size=1000,
            warmup=True
        )
    ),
    CacheEntryType.METADATA: MetadataCachePattern(
        CachePatternConfig(
            scope=CacheScope.SESSION,
            strategy=CacheStrategy.TTL,
            ttl=timedelta(minutes=30)
        )
    ),
    CacheEntryType.VALIDATION: ValidationCachePattern(
        CachePatternConfig(
            scope=CacheScope.REQUEST,
            strategy=CacheStrategy.FIFO,
            ttl=timedelta(minutes=5)
        )
    ),
    CacheEntryType.TRANSFORM: TransformationCachePattern(
        CachePatternConfig(
            scope=CacheScope.APPLICATION,
            strategy=CacheStrategy.LRU,
            ttl=timedelta(hours=2),
            max_size=500
        )
    ),
    CacheEntryType.REFERENCE: KeyRefCachePattern(
        CachePatternConfig(
            scope=CacheScope.SESSION,
            strategy=CacheStrategy.TTL,
            ttl=timedelta(minutes=15)
        )
    )
}
