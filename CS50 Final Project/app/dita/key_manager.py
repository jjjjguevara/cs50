"""Key reference management for DITA content processing."""

from typing import Dict, Optional, Any, List, Set
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging

from .models.types import (
    KeyDefinition,
    ElementType,
    ProcessingPhase,
    ValidationResult,
    ProcessingContext,
    ContentScope
)
from .event_manager import EventManager, EventType
from .utils.cache import ContentCache
from .utils.logger import DITALogger
from .config_manager import ConfigManager
from .context_manager import ContextManager

class KeyManager:
    """
    Manages DITA key definitions and resolution.
    Handles key inheritance, scope, and processing rules.
    """

    def __init__(
        self,
        event_manager: EventManager,
        cache: ContentCache,
        config_manager: ConfigManager,
        context_manager: ContextManager,
        logger: Optional[DITALogger] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.event_manager = event_manager
        self.cache = cache
        self.config_manager = config_manager
        self.context_manager = context_manager

        # Storage
        self._key_definitions: Dict[str, KeyDefinition] = {}
        self._key_hierarchy: Dict[str, List[str]] = {}

        # Resolution cache
        self._resolution_cache: Dict[str, Dict[str, Any]] = {}

        # Initialize from config
        self._initialize_from_config()

        # Register for events
        self._register_events()

    def _initialize_from_config(self) -> None:
        """Initialize key processing rules from config."""
        try:
            # Get key resolution config
            self._resolution_config = self.config_manager._keyref_config[
                "keyref_resolution"
            ]

            # Get processing hierarchy
            self._processing_hierarchy = self._resolution_config[
                "processing_hierarchy"
            ]

            # Get inheritance rules
            self._inheritance_rules = self._resolution_config[
                "inheritance_rules"
            ]

        except Exception as e:
            self.logger.error(f"Error initializing from config: {str(e)}")
            raise

    def _register_events(self) -> None:
        """Register for key-related events."""
        self.event_manager.subscribe(
            EventType.CACHE_INVALIDATE,
            self._handle_cache_invalidation
        )
        self.event_manager.subscribe(
            EventType.STATE_CHANGE,
            self._handle_state_change
        )

    def resolve_key(
        self,
        key: str,
        context_map: str,
        scope: Optional[str] = None
    ) -> Optional[KeyDefinition]:
        """
        Resolve key using configuration rules.
        """
        try:
            # Check resolution cache
            cache_key = f"keyref_{key}_{context_map}"
            if cached := self._resolution_cache.get(cache_key):
                # Convert cached dict back to KeyDefinition
                return KeyDefinition(**cached)

            # Follow resolution path
            for resolution_scope in self._resolution_config["scopes"]:
                if scope and resolution_scope != scope:
                    continue

                if definition := self._resolve_in_scope(
                    key=key,
                    context_map=context_map,
                    scope=resolution_scope
                ):
                    # Store dict representation in cache
                    self._resolution_cache[cache_key] = {
                        'key': definition.key,
                        'href': definition.href,
                        'scope': definition.scope,
                        'processing_role': definition.processing_role,
                        'metadata': definition.metadata,
                        'source_map': definition.source_map
                    }
                    return definition

            return None

        except Exception as e:
            self.logger.error(f"Error resolving key {key}: {str(e)}")
            return None



    def _handle_cache_invalidation(self, **event_data: Any) -> None:
        """Handle cache invalidation events."""
        try:
            if element_id := event_data.get("element_id"):
                if element_id.startswith("key_"):
                    key = element_id[4:]  # Remove 'key_' prefix
                    self.invalidate_key(key)

        except Exception as e:
            self.logger.error(f"Error handling cache invalidation: {str(e)}")

    def _handle_state_change(self, **event_data: Any) -> None:
        """Handle state change events."""
        try:
            element_id = event_data.get("element_id")
            if not element_id or not element_id.startswith("key_"):
                return

            metadata = event_data.get("metadata", {})
            if map_id := metadata.get("map_id"):
                # Invalidate resolution cache for this map
                pattern = f"keyref_*_{map_id}"
                self.cache.invalidate_pattern(pattern)

        except Exception as e:
            self.logger.error(f"Error handling state change: {str(e)}")

    def _resolve_in_scope(
        self,
        key: str,
        context_map: str,
        scope: str
    ) -> Optional[KeyDefinition]:
        """
        Resolve key in specific scope following inheritance rules.
        """
        try:
            # Get context
            context = self.context_manager.get_context(context_map)
            if not context:
                return None

            # Follow inheritance chain
            current_def = self._key_definitions.get(key)
            while current_def:
                if self._validate_key_scope(current_def, scope, context):
                    return current_def  # Now returns KeyDefinition directly

                # Check for inherited definition
                if parent_key := current_def.metadata.get("extends"):
                    current_def = self._key_definitions.get(parent_key)
                else:
                    break

            return None

        except Exception as e:
            self.logger.error(f"Error resolving key in scope: {str(e)}")
            return None

    def store_key_definition(
        self,
        key_def: KeyDefinition,
        map_id: str
    ) -> None:
        """Store key definition with context tracking."""
        try:
            # Register context
            self.context_manager.register_context(
                content_id=f"key_{key_def.key}",
                element_type=ElementType.UNKNOWN,
                metadata=key_def.metadata
            )

            # Store definition
            self._key_definitions[key_def.key] = key_def

            # Update hierarchy if key extends others
            if parent_key := key_def.metadata.get("extends"):
                self._key_hierarchy.setdefault(key_def.key, []).append(parent_key)

            # Cache definition (convert KeyDefinition to dict for caching)
            self.cache.set(
                f"key_{key_def.key}",
                vars(key_def),  # Convert to dict for storage
                ElementType.UNKNOWN,
                ProcessingPhase.DISCOVERY
            )

            # Emit event
            self.event_manager.emit(
                EventType.STATE_CHANGE,
                element_id=f"key_{key_def.key}",
                metadata={'map_id': map_id}
            )

        except Exception as e:
            self.logger.error(f"Error storing key definition: {str(e)}")
            raise

    def _validate_key_scope(
        self,
        key_def: KeyDefinition,
        scope: str,
        context: ProcessingContext
    ) -> bool:
        """Validate key definition against scope rules."""
        try:
            # Check key scope matches
            if key_def.scope != scope:
                return False

            # Check context compatibility
            if scope == "local":
                return context.scope == ContentScope.LOCAL
            elif scope == "peer":
                return context.scope in {ContentScope.LOCAL, ContentScope.PEER}
            elif scope == "external":
                return True  # External keys are always available

            return False

        except Exception as e:
            self.logger.error(f"Error validating key scope: {str(e)}")
            return False

    def get_key_hierarchy(self, key: str) -> List[str]:
        """Get inheritance hierarchy for a key."""
        try:
            hierarchy = []
            current = key

            while current and current not in hierarchy:
                hierarchy.append(current)
                if parent := self._key_hierarchy.get(current, []):
                    current = parent[0]  # Take first parent
                else:
                    break

            return hierarchy

        except Exception as e:
            self.logger.error(f"Error getting key hierarchy: {str(e)}")
            return [key]

    def invalidate_key(self, key: str) -> None:
        """Invalidate key cache."""
        try:
            # Invalidate direct key cache
            self.cache.invalidate(f"key_{key}")

            # Invalidate resolution cache for this key
            pattern = f"keyref_{key}_*"
            self.cache.invalidate_pattern(pattern)

            # Invalidate dependent keys
            for dep_key, parents in self._key_hierarchy.items():
                if key in parents:
                    self.invalidate_key(dep_key)

        except Exception as e:
            self.logger.error(f"Error invalidating key: {str(e)}")

    def cleanup(self) -> None:
        """Clean up manager resources."""
        try:
            self._key_definitions.clear()
            self._key_hierarchy.clear()
            self._resolution_cache.clear()
            self.logger.debug("Key manager cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise
