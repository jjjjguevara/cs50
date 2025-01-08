"""Key reference management for DITA content processing."""

from typing import Dict, Optional, Any, List, Set, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging
if TYPE_CHECKING:
    from .metadata.metadata_manager import MetadataManager

from .models.types import (
    KeyDefinition,
    ElementType,
    ProcessingPhase,
    ValidationResult,
    ProcessingContext,
    ContentScope
)
from .event_manager import EventManager, EventType
from .utils.cache import ContentCache, CacheEntryType
from .utils.logger import DITALogger
from .config.config_manager import ConfigManager
from .context_manager import ContextManager
from .metadata.storage import MetadataStorage

class KeyManager:
    """
    Manages DITA key definitions and resolution.
    Handles key inheritance, scope, and processing rules.
    """

    def __init__(
        self,
        event_manager: EventManager,
        cache: ContentCache,
        config_manager: 'ConfigManager',
        context_manager: 'ContextManager',
        metadata_storage: MetadataStorage,  # Add metadata_storage parameter
        logger: Optional[DITALogger] = None
    ):

        self.metadata_storage = metadata_storage  # Store metadata_storage
        self.logger = logger or logging.getLogger(__name__)
        self.event_manager = event_manager
        self.cache = cache
        self.config_manager = config_manager
        self.context_manager = context_manager

        # Storage
        self._key_definitions: Dict[str, KeyDefinition] = {}
        self._key_hierarchy: Dict[str, List[str]] = {}

        self.storage = MetadataStorage


        # Resolution cache
        self._resolution_cache: Dict[str, Dict[str, Any]] = {}

        # Initialize from config
        self._initialize_from_config()

        # Register for events
        self._register_events()

        # Debugging log for key hierarchy
        self.logger.debug(f"Key hierarchy initialized: {self._key_hierarchy}")

        self._invalidation_in_progress = set()  # Track ongoing invalidations
        self._max_invalidation_depth = 10


    def _initialize_from_config(self) -> None:
        """Initialize key processing rules from config."""
        try:
            # Get configurations using public methods
            keyref_config = self.config_manager.keyref_config
            key_resolution = self.config_manager.load_key_resolution_config()

            # Get defaults
            default_hierarchy = ["map", "topic", "element"]
            default_inheritance_rules = {
                "props": "merge",
                "outputclass": "append",
                "other": "override"
            }

            # Combine configurations using safe access
            self._resolution_config = {
                "keyref_resolution": keyref_config.get("keyref_resolution", {}),
                "processing_hierarchy": keyref_config.get("processing_hierarchy", {
                    "order": default_hierarchy
                }),
                "inheritance_rules": (
                    keyref_config.get("inheritance_rules") or
                    key_resolution.get("inheritance", {}) or
                    default_inheritance_rules
                ),
                "global_defaults": keyref_config.get("global_defaults", {}),
                "element_defaults": keyref_config.get("element_defaults", {})
            }

            # Get processing hierarchy with safe fallback
            self._processing_hierarchy = (
                self._resolution_config.get("processing_hierarchy", {})
                .get("order", default_hierarchy)
            )

            # Get inheritance rules with safe fallback
            self._inheritance_rules = (
                self._resolution_config.get("inheritance_rules") or
                default_inheritance_rules
            )

        except Exception as e:
            self.logger.error(f"Error initializing from config: {str(e)}")
            # Set defaults if initialization fails
            default_hierarchy = ["map", "topic", "element"]
            default_inheritance_rules = {
                "props": "merge",
                "outputclass": "append",
                "other": "override"
            }

            # Set all defaults
            self._processing_hierarchy = default_hierarchy
            self._inheritance_rules = default_inheritance_rules
            self._resolution_config = {
                "keyref_resolution": {},
                "processing_hierarchy": {"order": default_hierarchy},
                "inheritance_rules": default_inheritance_rules,
                "global_defaults": {},
                "element_defaults": {}
            }


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
        """Handle cache invalidation events with recursion protection."""
        try:
            if element_id := event_data.get("element_id"):
                if element_id.startswith("key_"):
                    key = element_id[4:]  # Remove 'key_' prefix

                    # Check if we're already processing this key
                    if key in self._invalidation_in_progress:
                        self.logger.debug(f"Skipping recursive invalidation for key: {key}")
                        return

                    self._invalidation_in_progress.add(key)
                    try:
                        self._process_invalidation(key)
                    finally:
                        self._invalidation_in_progress.remove(key)

        except Exception as e:
            self.logger.error(f"Error handling cache invalidation: {str(e)}")

    def _process_invalidation(self, initial_key: str) -> None:
        """Process key invalidation using iterative approach."""
        try:
            # Use breadth-first traversal to handle dependencies
            to_process = {initial_key}
            processed = set()
            depth = 0

            while to_process and depth < self._max_invalidation_depth:
                current_level = to_process
                to_process = set()

                for key in current_level:
                    if key not in processed:
                        # Invalidate current key
                        self._invalidate_single_key(key)
                        processed.add(key)

                        # Add dependencies for next level
                        dependencies = self._get_key_dependencies(key)
                        to_process.update(dependencies - processed)

                depth += 1

            if depth >= self._max_invalidation_depth:
                self.logger.warning(f"Max invalidation depth reached for key: {initial_key}")

        except Exception as e:
            self.logger.error(f"Error processing invalidation for key {initial_key}: {str(e)}")

    def _invalidate_single_key(self, key: str) -> None:
            """Invalidate a single key's cache entries."""
            try:
                # Invalidate in metadata storage
                self.metadata_storage.invalidate_keys_batch(keys_to_invalidate={key})

                # Invalidate in cache
                cache_key = f"key_{key}"
                self.cache.invalidate(cache_key, entry_type=CacheEntryType.METADATA)

            except Exception as e:
                self.logger.error(f"Error invalidating single key {key}: {str(e)}")

    def _get_key_dependencies(self, key: str) -> Set[str]:
        """Get all keys that depend on the given key."""
        try:
            dependencies = set()
            # Get direct dependencies from hierarchy
            if deps := self._key_hierarchy.get(key, []):
                dependencies.update(deps)
            return dependencies
        except Exception as e:
            self.logger.error(f"Error getting key dependencies for {key}: {str(e)}")
            return set()


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
                self.cache.invalidate_by_pattern(pattern)

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
                    key=f"key_{key_def.key}",
                    data=vars(key_def),  # Convert to dict for storage
                    entry_type=CacheEntryType.CONTENT,  # Added entry_type
                    element_type=ElementType.UNKNOWN,
                    phase=ProcessingPhase.DISCOVERY,  # Added phase
                    scope=ContentScope.LOCAL  # Added scope for completeness
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



    def invalidate_key_iterative(self, key: str) -> None:
        """Invalidate key cache using an iterative approach."""
        try:
            stack = [key]
            visited = set()

            while stack:
                current_key = stack.pop()

                if current_key in visited:
                    self.logger.error(f"Cyclic dependency detected at key: {current_key}")
                    continue

                visited.add(current_key)

                # Invalidate the current key
                cache_key = f"key_{current_key}"
                self.cache.invalidate(cache_key, entry_type=CacheEntryType.CONTENT)

                # Add dependent keys to the stack
                for dependent_key, parents in self._key_hierarchy.items():
                    if current_key in parents and dependent_key not in visited:
                        stack.append(dependent_key)

        except Exception as e:
            self.logger.error(f"Error invalidating key '{key}': {str(e)}")

    def detect_circular_dependencies(self, key: str) -> bool:
        """Detect if a key is part of a circular dependency."""
        visited = set()
        stack = [key]
        while stack:
            current = stack.pop()
            if current in visited:
                return True
            visited.add(current)
            stack.extend(self._key_hierarchy.get(current, []))
        return False

    def cleanup(self) -> None:
        """Clean up manager resources."""
        try:
            # Clear in-memory caches
            self._key_definitions.clear()
            self._key_hierarchy.clear()
            self._resolution_cache.clear()

            # Log cleanup
            self.logger.debug("Key manager cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise
