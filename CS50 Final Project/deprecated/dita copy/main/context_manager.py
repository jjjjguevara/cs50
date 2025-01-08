# app/dita/context_manager.py
from typing import Dict, List, Optional, Any, Set, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from uuid import uuid4
import re
if TYPE_CHECKING:
    from .metadata.metadata_manager import MetadataManager
    from .config.config_manager import ConfigManager

# Event and Cache
from .event_manager import EventManager, EventType
from .utils.cache import ContentCache, CacheEntryType
from .utils.logger import DITALogger
from .validation_manager import ValidationManager


# Custom Types
from .models.types import (
    ContentScope,
    ContentRelationType,
    ContentRelationship,
    NavigationContext,
    ProcessingContext,
    ElementType,
    ContentElement,
    MetadataState,
    ProcessingPhase,
    ProcessingState,
    ProcessingStatus,
    ValidationResult,
    ValidationMessage,
    ValidationSeverity
)


class ContextManager:
    """
    Mediates context relationships and content hierarchy.
    Uses schema-based rules for validation and relationships.
    """
    def __init__(
        self,
        event_manager: EventManager,
        content_cache: ContentCache,
        metadata_manager: 'MetadataManager',  # Forward reference in quotes
        config_manager: 'ConfigManager',      # Forward reference in quotes
        validation_manager: ValidationManager,
        logger: Optional[DITALogger] = None
    ):
        """
        Initialize context manager with required dependencies.

        Args:
            event_manager: For event-based communication
            content_cache: For transient context storage
            metadata_handler: For persistent metadata storage
            config_manager: For schema and rule access
            logger: For structured logging
        """
        # Core dependencies
        self.event_manager = event_manager
        self.content_cache = content_cache
        self.metadata_manager = metadata_manager
        self.config_manager = config_manager
        self.logger = logger or DITALogger()

        self.validation_manager: ValidationManager
        self.config_manager: ConfigManager
        self.event_manager: EventManager
        self.cache: ContentCache

        # Context tracking
        self._active_contexts: Dict[str, ProcessingContext] = {}
        self._content_relationships: Dict[str, List[ContentRelationship]] = {}
        self._navigation_contexts: Dict[str, NavigationContext] = {}

        # Scope & hierarchy tracking
        self._hierarchy_paths: Dict[str, List[str]] = {}

        # Metadata tracking
        self._metadata_states: Dict[str, MetadataState] = {}
        self._key_relationships: Dict[str, Set[str]] = {}

        # Register for events
        self._register_event_handlers()




    #########################
    # Core mediator methods #
    #########################

    def _register_event_handlers(self) -> None:
        """Register for state change events to track context states."""
        try:
            # Single event subscription for state changes
            self.event_manager.subscribe(
                EventType.STATE_CHANGE,
                self._update_context_state
            )

        except Exception as e:
            self.logger.error(f"Error registering event handlers: {str(e)}")
            raise

    def _register_metadata_handlers(self) -> None:
            """Register for metadata-related events."""
            self.event_manager.subscribe(
                EventType.STATE_CHANGE,
                self._handle_metadata_state_change
            )

    def _update_context_state(self, **event_data: Any) -> None:
        """
        Update context state based on event.

        Args:
            event_data: Event information including element_id and state
        """
        try:
            element_id = event_data.get("element_id")
            state_info = event_data.get("state_info")

            if element_id and state_info:
                context = self._active_contexts.get(element_id)
                if context:
                    context.state_info = state_info

        except Exception as e:
            self.logger.error(f"Error updating context state: {str(e)}")


    def update_context(
        self,
        content_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """
        Update existing context with metadata changes.

        Args:
            content_id: Content identifier
            updates: Context updates to apply
        """
        try:
            # Get active context
            context = self._active_contexts.get(content_id)
            if not context:
                self.logger.warning(f"No active context found for {content_id}")
                return

            # Update metadata
            if metadata_updates := updates.get('metadata'):
                # Store persistent metadata
                self.metadata_manager.store_metadata(content_id, metadata_updates)

            # Update navigation context if provided
            if nav_updates := updates.get('navigation'):
                context.navigation.update(nav_updates)

            # Update state if provided
            if state_updates := updates.get('state'):
                context.state_info.update(state_updates)

            # Update scope if provided
            if new_scope := updates.get('scope'):
                context.scope = new_scope

            # Emit state change event
            self.event_manager.emit(
                EventType.STATE_CHANGE,
                element_id=content_id,
                state_info=context.state_info
            )

        except Exception as e:
            self.logger.error(f"Error updating context: {str(e)}")
            raise

    # In context_manager.py

    def register_context(
        self,
        content_id: str,
        element_type: ElementType,
        metadata: Dict[str, Any]
    ) -> ProcessingContext:
        """Register context with metadata awareness."""
        try:
            # Create metadata state with proper transient/persistent separation
            metadata_state = MetadataState(
                content_id=content_id,
                phase=ProcessingPhase.DISCOVERY,
                state=ProcessingState.PENDING,
                cached=False,  # Will be set to true when cached
                metadata_refs={},  # For key references
                key_references=[]  # For tracking key dependencies
            )

            # Create navigation context
            navigation = NavigationContext(
                path=[],
                level=0,
                sequence=0,
                parent_id=None,
                root_map=content_id,
                siblings=[]
            )

            # Create processing state info
            state_info = ProcessingStatus(
                element_id=content_id,
                phase=ProcessingPhase.DISCOVERY,
                state=ProcessingState.PENDING
            )

            # Create processing context
            context = ProcessingContext(
                context_id=content_id,
                element_id=content_id,
                element_type=element_type,
                state_info=state_info,
                navigation=navigation,
                scope=ContentScope.LOCAL,
                metadata_state=metadata_state,
                metadata_cache={},  # For transient metadata
                metadata_refs={}    # For metadata references
            )

            # Store context
            self._active_contexts[content_id] = context

            # Handle metadata separation
            # Transient metadata goes to cache
            transient_metadata = {
                k: v for k, v in metadata.items()
                if k not in {'persistent', 'key_refs', 'metadata_refs'}
            }
            self.content_cache.set(
                key=f"metadata_{content_id}",
                data=transient_metadata,
                entry_type=CacheEntryType.METADATA,
                element_type=element_type,
                phase=ProcessingPhase.DISCOVERY
            )

            # Persistent metadata goes to storage via metadata manager
            persistent_metadata = metadata.get('persistent', {})
            if persistent_metadata:
                self.metadata_manager.store_metadata(content_id, persistent_metadata)

            # Process key references and metadata references
            if key_refs := metadata.get('key_refs', []):
                metadata_state.key_references = key_refs

            if metadata_refs := metadata.get('metadata_refs', {}):
                context.metadata_refs.update(metadata_refs)

            # Emit event
            self.event_manager.emit(
                EventType.STATE_CHANGE,
                element_id=content_id,
                state_info=context.state_info
            )

            return context

        except Exception as e:
            self.logger.error(f"Error registering context for {content_id}: {str(e)}")
            # Create and return a fallback context instead of None
            fallback_context = ProcessingContext(
                context_id=content_id,
                element_id=content_id,
                element_type=element_type,
                state_info=ProcessingStatus(
                    element_id=content_id,
                    phase=ProcessingPhase.ERROR,
                    state=ProcessingState.ERROR,
                    error_message=str(e)
                ),
                navigation=NavigationContext(
                    path=[],
                    level=0,
                    sequence=0,
                    parent_id=None,
                    root_map=content_id,
                    siblings=[]
                ),
                scope=ContentScope.LOCAL,
                metadata_state=MetadataState(
                    content_id=content_id,
                    phase=ProcessingPhase.ERROR,
                    state=ProcessingState.ERROR
                )
            )
            self._active_contexts[content_id] = fallback_context
            return fallback_context


    def _process_initial_metadata(
            self,
            content_id: str,
            metadata: Dict[str, Any]
        ) -> None:
            """Process initial metadata for context."""
            try:
                # Extract key references
                if key_refs := metadata.get('key_refs', []):
                    self._key_relationships[content_id] = set(key_refs)
                    self._metadata_states[content_id].key_references = key_refs

                # Store metadata references
                if metadata_refs := metadata.get('metadata_refs', {}):
                    self._metadata_states[content_id].metadata_refs = metadata_refs

                # Cache metadata
                self.content_cache.set(
                    key=f"metadata_{content_id}",
                    data=metadata,
                    entry_type=CacheEntryType.METADATA,
                    element_type=ElementType.UNKNOWN,
                    phase=ProcessingPhase.DISCOVERY
                )

            except Exception as e:
                self.logger.error(f"Error processing initial metadata: {str(e)}")

    def _handle_metadata_state_change(self, **event_data: Any) -> None:
            """Handle metadata state changes."""
            try:
                element_id = event_data.get("element_id")
                metadata_state = event_data.get("metadata_state")

                if element_id and metadata_state:
                    if current_state := self._metadata_states.get(element_id):
                        # Update state
                        current_state.phase = metadata_state.phase
                        current_state.state = metadata_state.state
                        current_state.timestamp = datetime.now()

                        # Update context
                        if context := self._active_contexts.get(element_id):
                            context.metadata_state = current_state

            except Exception as e:
                self.logger.error(f"Error handling metadata state change: {str(e)}")

    def register_metadata_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: ContentRelationType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register metadata-based relationship."""
        try:
            relationship = ContentRelationship(
                source_id=source_id,
                target_id=target_id,
                relation_type=relationship_type,
                scope=self._determine_scope(source_id, target_id),
                metadata=metadata or {},
                key_refs=set()  # Will be populated if keys are referenced
            )

            # Store relationship
            if source_id not in self._content_relationships:
                self._content_relationships[source_id] = []
            self._content_relationships[source_id].append(relationship)

            # Process metadata references
            if metadata and 'key_refs' in metadata:
                relationship.key_refs.update(metadata['key_refs'])
                # Update key relationships
                self._key_relationships.setdefault(source_id, set()).update(
                    metadata['key_refs']
                )

        except Exception as e:
            self.logger.error(f"Error registering metadata relationship: {str(e)}")
            raise

    def get_metadata_state(
        self,
        content_id: str
    ) -> Optional[MetadataState]:
        """Get current metadata state."""
        return self._metadata_states.get(content_id)

    def get_key_relationships(
        self,
        content_id: str
    ) -> Set[str]:
        """Get key relationships for content."""
        return self._key_relationships.get(content_id, set())

    def validate_metadata_relationship(self, source_id: str, target_id: str, metadata: Dict[str, Any]) -> bool:
        """Validate metadata relationship."""
        try:
            # Get source and target contexts
            source_context = self._active_contexts.get(source_id)
            target_context = self._active_contexts.get(target_id)

            if not source_context or not target_context:
                return False

            # Check scope compatibility
            if not self._validate_scope_compatibility(source_context.scope, target_context.scope):
                return False

            # Validate metadata manager is initialized
            if self.metadata_manager is None or self.metadata_manager.key_manager is None:
                self.logger.warning("Metadata manager or key manager not initialized")
                return False

            # Check key reference validity
            if key_refs := metadata.get('key_refs', []):
                if not all(
                    self.metadata_manager.key_manager.resolve_key(
                        key=key,
                        context_map=source_context.navigation.root_map
                    ) for key in key_refs
                ):
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating metadata relationship: {str(e)}")
            return False


    def validate_scope(
        self,
        key: str,
        required_scope: ContentScope,
        context: Optional[ProcessingContext] = None
    ) -> bool:
        """
        Validate if a key can be accessed in the required scope.

        Args:
            key: Key to validate
            required_scope: Required scope for access
            context: Optional context for additional validation

        Returns:
            bool: True if key is accessible in scope
        """
        try:
            # Basic scope check
            if not self._is_in_scope(key, required_scope):
                return False

            # Additional context validation if provided
            if context:
                # Check context scope compatibility
                if not self._validate_scope_compatibility(
                    context.scope,
                    required_scope
                ):
                    return False

                # Check context-specific rules
                if not self._validate_context_scope(key, required_scope, context):
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating scope for {key}: {str(e)}")
            return False

    def get_accessible_keys(
        self,
        context: ProcessingContext,
        scope: Optional[ContentScope] = None
    ) -> Set[str]:
        """
        Get keys accessible in the given scope.

        Args:
            context: Current processing context
            scope: Optional scope filter (uses context scope if not provided)

        Returns:
            Set[str]: Set of accessible keys
        """
        try:
            # Use context scope if none provided
            target_scope = scope or context.scope

            # Get all keys from context
            all_keys = set(context.metadata_refs.keys())

            # Filter by scope
            return {
                key for key in all_keys
                if self._is_in_scope(key, target_scope)
            }

        except Exception as e:
            self.logger.error(f"Error getting accessible keys: {str(e)}")
            return set()

    def _validate_context_scope(
        self,
        key: str,
        required_scope: ContentScope,
        context: ProcessingContext
    ) -> bool:
        """
        Validate scope against context-specific rules.

        Args:
            key: Key to validate
            required_scope: Required scope
            context: Processing context

        Returns:
            bool: True if scope is valid for context
        """
        try:
            # Check navigation context
            if required_scope == ContentScope.LOCAL:
                # Local scope requires same root map
                if context.navigation.root_map != context.element_id:
                    return False

            # Check feature flags
            features = context.features
            if not features:
                return True

            # Get scope restrictions from features
            restrictions = features.get('scope_restriction')
            if not restrictions:
                return True

            # Type guard against string list
            if not isinstance(restrictions, (list, tuple)):
                self.logger.warning(
                    f"Invalid scope_restriction format for {key}: "
                    f"expected list, got {type(restrictions)}"
                )
                return True

            # Check scope restriction
            if required_scope.value not in map(str, restrictions):
                return False

            # Check conditions from context
            if not isinstance(context.metadata_refs, dict):
                return True

            conditions = context.metadata_refs.get('scope_conditions')
            if not isinstance(conditions, dict):
                return True

            return self._evaluate_scope_conditions(
                key,
                required_scope,
                conditions
            )

        except Exception as e:
            self.logger.error(f"Error validating context scope: {str(e)}")
            return False

    def _evaluate_scope_conditions(
        self,
        key: str,
        scope: ContentScope,
        conditions: Dict[str, Any]
    ) -> bool:
        """
        Evaluate scope conditions from context.

        Args:
            key: Key being validated
            scope: Required scope
            conditions: Scope conditions to evaluate

        Returns:
            bool: True if conditions are satisfied
        """
        try:
            # Iterate over conditions dictionary properly
            for condition_name, condition_data in conditions.items():
                if isinstance(condition_data, dict):
                    if target_scope := condition_data.get('scope'):
                        if scope.value == target_scope:
                            # Check patterns
                            if patterns := condition_data.get('patterns', []):
                                if not any(
                                    re.match(pattern, key)
                                    for pattern in patterns
                                ):
                                    return False

                            # Check prefixes
                            if prefixes := condition_data.get('prefixes', []):
                                if not any(
                                    key.startswith(prefix)
                                    for prefix in prefixes
                                ):
                                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error evaluating scope conditions: {str(e)}")
            return False

    def _validate_scope_compatibility(
            self,
            source_scope: ContentScope,
            target_scope: ContentScope
        ) -> bool:
            """Validate scope compatibility between source and target."""
            # Local scope can only reference local or peer
            if source_scope == ContentScope.LOCAL:
                return target_scope in {ContentScope.LOCAL, ContentScope.PEER}

            # Peer scope can reference any scope
            if source_scope == ContentScope.PEER:
                return True

            # External scope can only reference external
            if source_scope == ContentScope.EXTERNAL:
                return target_scope == ContentScope.EXTERNAL

            return False

    def _is_in_scope(self, key: str, scope: ContentScope) -> bool:
        """
        Check if a key is in the given scope.
        Used for metadata, relationships, and references.

        Args:
            key: Key to check
            scope: ContentScope to validate against

        Returns:
            bool: True if key is in scope
        """
        try:
            # Global scope includes all keys
            if scope == ContentScope.GLOBAL:
                return True

            # Local scope includes only local keys
            if scope == ContentScope.LOCAL:
                return not key.startswith(('peer_', 'external_'))

            # Peer scope includes local and peer keys
            if scope == ContentScope.PEER:
                return not key.startswith('external_')

            # External scope only includes external keys
            if scope == ContentScope.EXTERNAL:
                return key.startswith('external_')

            # Log warning for unknown scope
            self.logger.warning(f"Unknown scope: {scope}")
            return False

        except Exception as e:
            self.logger.error(f"Error checking scope for {key}: {str(e)}")
            return False

    def register_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: ContentRelationType
    ) -> None:
        """
        Register relationship between content elements.

        Args:
            source_id: Source content identifier
            target_id: Target content identifier
            relation_type: Type of relationship
        """
        try:
            # Create relationship
            relationship = ContentRelationship(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                scope=self._determine_scope(source_id, target_id)
            )

            # Store relationship
            if source_id not in self._content_relationships:
                self._content_relationships[source_id] = []
            self._content_relationships[source_id].append(relationship)

        except Exception as e:
            self.logger.error(f"Error registering relationship: {str(e)}")
            raise

    def _determine_scope(self, source_id: str, target_id: str) -> ContentScope:
        """
        Determine scope based on content relationship.

        Args:
            source_id: Source content identifier
            target_id: Target content identifier

        Returns:
            ContentScope: Determined scope
        """
        try:
            source_context = self._active_contexts.get(source_id)
            target_context = self._active_contexts.get(target_id)

            if not source_context or not target_context:
                return ContentScope.LOCAL

            # Same map = local scope
            if source_context.navigation.root_map == target_context.navigation.root_map:
                return ContentScope.LOCAL

            # External link = external scope
            if target_context.navigation.root_map.startswith(('http://', 'https://')):
                return ContentScope.EXTERNAL

            # Default to peer scope
            return ContentScope.PEER

        except Exception as e:
            self.logger.error(f"Error determining scope: {str(e)}")
            return ContentScope.LOCAL

    def get_relationships(
        self,
        content_id: str,
        relation_type: Optional[ContentRelationType] = None
    ) -> List[ContentRelationship]:
        """
        Get relationships for content.

        Args:
            content_id: Content to get relationships for
            relation_type: Optional type to filter by

        Returns:
            List of relationships
        """
        try:
            relationships = self._content_relationships.get(content_id, [])

            if relation_type:
                relationships = [
                    rel for rel in relationships
                    if rel.relation_type == relation_type
                ]

            return relationships

        except Exception as e:
            self.logger.error(f"Error getting relationships: {str(e)}")
            return []

    def get_context(self, content_id: str) -> Optional[ProcessingContext]:
        """
        Get active context for content.

        Args:
            content_id: Content identifier

        Returns:
            Optional[ProcessingContext]: Active context if found
        """
        return self._active_contexts.get(content_id)



    def invalidate_context(self, content_id: str) -> None:
        """Invalidate cached context."""
        try:
            cache_key = f"context_{content_id}"
            self.content_cache.invalidate(
                key=cache_key,
                entry_type=CacheEntryType.CONTENT
            )

            # Emit event
            self.event_manager.emit(
                EventType.CACHE_INVALIDATE,
                element_id=content_id
            )

            self.logger.debug(f"Invalidated context for {content_id}")

        except Exception as e:
            self.logger.error(f"Failed to invalidate context: {str(e)}")
            raise

    def notify_context_change(
        self,
        content_id: str,
        change_type: str
    ) -> None:
        """
        Notify observers about context changes.

        Args:
            content_id: Content identifier
            change_type: Type of change
        """
        try:
            context = self.get_context(content_id)

            # Emit change event
            self.event_manager.emit(
                EventType.STATE_CHANGE,
                element_id=content_id,
                change_type=change_type,
                context=context
            )

            # Invalidate affected contexts
            self._invalidate_dependent_contexts(content_id)

            self.logger.debug(f"Notified context change for {content_id}")

        except Exception as e:
            self.logger.error(f"Failed to notify context change: {str(e)}")
            raise

    def _merge_contexts(
        self,
        parent_context: ProcessingContext,
        child_context: ProcessingContext
    ) -> ProcessingContext:
        """
        Merge parent and child contexts maintaining proper inheritance.

        Args:
            parent_context: Parent ProcessingContext
            child_context: Child ProcessingContext

        Returns:
            ProcessingContext: Merged context
        """
        try:
            # Create new context with base parent values
            merged_context = ProcessingContext(
                context_id=child_context.context_id,
                element_id=child_context.element_id,
                element_type=child_context.element_type,
                state_info=child_context.state_info,
                navigation=NavigationContext(
                    path=[*parent_context.navigation.path, child_context.element_id],
                    level=parent_context.navigation.level + 1,
                    sequence=len(parent_context.navigation.siblings),
                    parent_id=parent_context.element_id,
                    root_map=parent_context.navigation.root_map
                ),
                scope=child_context.scope
            )

            # Update metadata references based on inheritance
            merged_context.metadata_refs.update(parent_context.metadata_refs)
            merged_context.metadata_refs.update(child_context.metadata_refs)

            # Maintain child relationships
            merged_context.relationships = child_context.relationships

            # Keep child's feature state
            merged_context.features = child_context.features

            return merged_context

        except Exception as e:
            self.logger.error(f"Error merging contexts: {str(e)}")
            return child_context

    def _invalidate_dependent_contexts(self, content_id: str) -> None:
        """
        Invalidate contexts that depend on the given content.

        Args:
            content_id: Content identifier whose dependents need invalidation
        """
        try:
            # Find all dependent contexts
            dependent_ids = set()

            # Check direct relationships
            if relationships := self._content_relationships.get(content_id):
                dependent_ids.update(rel.target_id for rel in relationships)

            # Check navigation hierarchy
            for ctx in self._active_contexts.values():
                if content_id in ctx.navigation.path:
                    dependent_ids.add(ctx.context_id)

            # Remove from active contexts
            for dep_id in dependent_ids:
                if dep_id in self._active_contexts:
                    del self._active_contexts[dep_id]

                    # Emit invalidation event
                    self.event_manager.emit(
                        EventType.CACHE_INVALIDATE,
                        element_id=dep_id
                    )

        except Exception as e:
            self.logger.error(f"Error invalidating dependent contexts: {str(e)}")

    ###########################
    # Relationship management #
    ##########################



    def validate_relationship(self, source_id: str, target_id: str) -> bool:
        """
        Validate a potential relationship between content elements.

        Args:
            source_id: Source content identifier
            target_id: Target content identifier

        Returns:
            bool: True if relationship is valid
        """
        try:
            # Check for self-referential relationships
            if source_id == target_id:
                self.logger.warning(f"Self-referential relationship detected: {source_id}")
                return False

            # Get source and target contexts
            source_context = self._active_contexts.get(source_id)
            target_context = self._active_contexts.get(target_id)

            # Both contexts must exist
            if not source_context or not target_context:
                return False

            # Check for circular references
            if self._has_circular_reference(source_id, target_id):
                self.logger.warning(f"Circular reference detected: {source_id} -> {target_id}")
                return False

            # Check scope compatibility
            source_scope = source_context.scope
            target_scope = target_context.scope

            if source_scope == ContentScope.LOCAL and target_scope == ContentScope.EXTERNAL:
                return False

            if source_scope == ContentScope.PEER and target_scope == ContentScope.EXTERNAL:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating relationship: {str(e)}")
            return False

    def _has_circular_reference(self, source_id: str, target_id: str, visited: Optional[Set[str]] = None) -> bool:
        """
        Check for circular references in relationships.

        Args:
            source_id: Starting point for checking
            target_id: Target to check against
            visited: Set of already visited IDs

        Returns:
            bool: True if circular reference detected
        """
        if visited is None:
            visited = set()

        if source_id in visited:
            return True

        visited.add(source_id)

        # Check current relationships
        relationships = self._content_relationships.get(source_id, [])
        for rel in relationships:
            if rel.target_id == target_id:
                return True
            if self._has_circular_reference(rel.target_id, target_id, visited):
                return True

        return False

    def get_related_content(
        self,
        content_id: str,
        relationship_type: Optional[ContentRelationType] = None
    ) -> List[Dict[str, Any]]:
        """
        Get related content with optional relationship type filter.

        Args:
            content_id: Content identifier to get relationships for
            relationship_type: Optional relationship type to filter by

        Returns:
            List of related content items with metadata
        """
        try:
            # Check cache first
            cache_key = f"related_{content_id}"
            if cached := self.content_cache.get(
                key=cache_key,
                entry_type=CacheEntryType.CONTENT
            ):
                return cached

            # Get relationships from metadata handler
            relationships = self.metadata_manager.get_content_relationships(content_id)

            # Filter by relationship type if specified
            if relationship_type:
                relationships = [
                    rel for rel in relationships
                    if ContentRelationType(rel['type']) == relationship_type
                ]

            # Process relationships and build result
            result = []
            for rel in relationships:
                # Add base relationship info
                related_item = {
                    'target_id': rel['target_id'],
                    'type': rel['type'],
                    'scope': rel['scope'],
                    'metadata': rel['metadata'],
                    'created_at': rel['created_at']
                }

                # Get target context if available
                if target_context := self._active_contexts.get(rel['target_id']):
                    related_item['context'] = {
                        'element_type': target_context.element_type.value,
                        'scope': target_context.scope.value,
                        'features': target_context.features,
                        'navigation': {
                            'level': target_context.navigation.level,
                            'sequence': target_context.navigation.sequence
                        }
                    }

                result.append(related_item)

            # Cache results
            self.content_cache.set(
                key=cache_key,
                data=result,
                entry_type=CacheEntryType.CONTENT,
                element_type=ElementType.UNKNOWN,
                phase=ProcessingPhase.DISCOVERY,
                ttl=3600  # Cache for 1 hour
            )

            return result

        except Exception as e:
            self.logger.error(f"Error getting related content: {str(e)}")
            return []


    #########################
    # Hierarchy management #
    ########################

    def register_hierarchy_node(
        self,
        content_id: str,
        parent_id: Optional[str] = None
    ) -> None:
        """Register a node in the content hierarchy."""
        try:
            # Get current context
            context = self._active_contexts.get(content_id)
            if not context:
                raise ValueError(f"No active context found for {content_id}")

            # Get parent context if exists
            parent_context = self._active_contexts.get(parent_id) if parent_id else None

            # Calculate hierarchy level
            level = (parent_context.navigation.level + 1) if parent_context else 0

            # Create navigation context
            navigation = NavigationContext(
                path=[parent_id] if parent_id else [],
                level=level,
                sequence=len(self._content_relationships.get(parent_id, [])) if parent_id else 0,
                parent_id=parent_id,
                root_map=parent_context.navigation.root_map if parent_context else content_id
            )

            # Update context with new navigation
            context.navigation = navigation

            # Store hierarchy in metadata
            hierarchy_metadata = {
                'parent_id': parent_id,
                'level': level,
                'path': navigation.path,
                'root_map': navigation.root_map
            }

            # Store using metadata manager
            self.metadata_manager.store_metadata(
                content_id,
                {'hierarchy': hierarchy_metadata}
            )

            # Update hierarchy paths
            if content_id not in self._hierarchy_paths:
                self._hierarchy_paths[content_id] = []
            if parent_id:
                # Copy parent's path and append current
                parent_path = self._hierarchy_paths.get(parent_id, [])
                self._hierarchy_paths[content_id] = parent_path + [parent_id]

            # Invalidate related caches
            self.content_cache.invalidate_by_pattern(f"context_{content_id}")
            if parent_id:
                self.content_cache.invalidate_by_pattern(f"context_{parent_id}")

            self.logger.debug(
                f"Registered hierarchy node: {content_id} "
                f"(parent: {parent_id}, level: {level})"
            )

        except Exception as e:
            self.logger.error(f"Error registering hierarchy node: {str(e)}")
            raise

    def get_content_path(self, content_id: str) -> List[str]:
        """
        Get the full path to a content element.

        Args:
            content_id: Content identifier

        Returns:
            List of content IDs representing the path from root to content
        """
        try:
            # Check cache first
            cache_key = f"path_{content_id}"
            if cached := self.content_cache.get(
                key=cache_key,
                entry_type=CacheEntryType.CONTENT
            ):
                return cached
            # Get current context
            context = self._active_contexts.get(content_id)
            if not context:
                return []

            # Build path from navigation context
            path = context.navigation.path.copy()
            path.append(content_id)  # Add current element

            # Cache result
            self.content_cache.set(
                key=cache_key,
                data=path,
                entry_type=CacheEntryType.CONTENT,
                element_type=ElementType.UNKNOWN,
                phase=ProcessingPhase.DISCOVERY,
                ttl=3600  # Cache for 1 hour
            )

            return path

        except Exception as e:
            self.logger.error(f"Error getting content path: {str(e)}")
            return []

    def get_content_level(self, content_id: str) -> int:
        """
        Get the hierarchy level of a content element.

        Args:
            content_id: Content identifier

        Returns:
            int: Hierarchy level (0 for root elements)
        """
        try:
            # Get current context
            context = self._active_contexts.get(content_id)
            if not context:
                return 0

            # Check cache first
            cache_key = f"level_{content_id}"
            if cached := self.content_cache.get(
                key=cache_key,
                entry_type=CacheEntryType.CONTENT
            ):
                return cached

            # Get level from navigation context
            level = context.navigation.level

            # Cache result
            self.content_cache.set(
                key=cache_key,
                data=level,
                entry_type=CacheEntryType.CONTENT,
                element_type=ElementType.UNKNOWN,
                phase=ProcessingPhase.DISCOVERY,
                ttl=3600  # Cache for 1 hour
            )

            return level

        except Exception as e:
            self.logger.error(f"Error getting content level: {str(e)}")
            return 0


    ##########################
    # Validation management #
    #########################

    def _validate_scope_rules(
        self,
        context: ProcessingContext,
        rules: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate scope rules against configuration.

        Args:
            context: Processing context to validate
            rules: Optional rules override (gets from config if not provided)

        Returns:
            ValidationResult: Validation results
        """
        try:
            # Get rules from config if not provided
            if not rules and self.config_manager:
                rules = self.config_manager.get_processing_rules(
                    element_type=context.element_type,
                    context=context
                ).get('scope_rules', {})

            if not rules:
                return ValidationResult(is_valid=True, messages=[])

            messages = []

            # Validate allowed scopes
            if allowed_scopes := rules.get('allowed_scopes'):
                if context.scope.value not in allowed_scopes:
                    messages.append(ValidationMessage(
                        path=f"context.{context.context_id}.scope",
                        message=f"Invalid scope: {context.scope.value}. Allowed: {allowed_scopes}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_scope"
                    ))

            # Validate scope inheritance
            if parent_id := context.navigation.parent_id:
                if parent_context := self.get_context(parent_id):
                    if not self._validate_scope_compatibility(
                        parent_context.scope,
                        context.scope
                    ):
                        messages.append(ValidationMessage(
                            path=f"context.{context.context_id}.scope",
                            message=f"Incompatible scope with parent: {parent_context.scope.value}",
                            severity=ValidationSeverity.ERROR,
                            code="scope_inheritance_error"
                        ))

            # Validate scope conditions
            if conditions := rules.get('conditions'):
                if not self._evaluate_scope_conditions(
                    context.element_id,
                    context.scope,
                    conditions
                ):
                    messages.append(ValidationMessage(
                        path=f"context.{context.context_id}.conditions",
                        message="Failed scope conditions validation",
                        severity=ValidationSeverity.ERROR,
                        code="scope_conditions_error"
                    ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating scope rules: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path=f"context.{context.context_id}",
                    message=f"Scope rules validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="scope_validation_error"
                )]
            )

    def _validate_navigation(
        self,
        context: ProcessingContext
    ) -> ValidationResult:
        """
        Validate navigation hierarchy and relationships.

        Args:
            context: Processing context to validate

        Returns:
            ValidationResult: Validation results
        """
        try:
            messages = []

            # Validate navigation path
            if context.navigation.path:
                # Check path exists
                for path_id in context.navigation.path:
                    if not self.get_context(path_id):
                        messages.append(ValidationMessage(
                            path=f"context.{context.context_id}.navigation.path",
                            message=f"Invalid path element: {path_id}",
                            severity=ValidationSeverity.ERROR,
                            code="invalid_path"
                        ))

            # Validate parent relationship
            if parent_id := context.navigation.parent_id:
                if not self.get_context(parent_id):
                    messages.append(ValidationMessage(
                        path=f"context.{context.context_id}.navigation.parent",
                        message=f"Invalid parent: {parent_id}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_parent"
                    ))

            # Validate root map
            if root_map := context.navigation.root_map:
                if not self.get_context(root_map):
                    messages.append(ValidationMessage(
                        path=f"context.{context.context_id}.navigation.root_map",
                        message=f"Invalid root map: {root_map}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_root_map"
                    ))

            # Validate hierarchy level
            if parent_id and context.navigation.level <= 0:
                messages.append(ValidationMessage(
                    path=f"context.{context.context_id}.navigation.level",
                    message="Invalid hierarchy level",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_level"
                ))

            # Validate siblings
            if siblings := context.navigation.siblings:
                for sibling_id in siblings:
                    if not self.get_context(sibling_id):
                        messages.append(ValidationMessage(
                            path=f"context.{context.context_id}.navigation.siblings",
                            message=f"Invalid sibling: {sibling_id}",
                            severity=ValidationSeverity.ERROR,
                            code="invalid_sibling"
                        ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating navigation: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path=f"context.{context.context_id}",
                    message=f"Navigation validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="navigation_validation_error"
                )]
            )

    def _validate_relationships(
        self,
        context: ProcessingContext
    ) -> ValidationResult:
        """
        Validate content relationships and dependencies.

        Args:
            context: Processing context to validate

        Returns:
            ValidationResult: Validation results
        """
        try:
            messages = []

            # Validate each relationship
            for relationship in context.relationships:
                # Validate source exists
                if not self.get_context(relationship.source_id):
                    messages.append(ValidationMessage(
                        path=f"context.{context.context_id}.relationships",
                        message=f"Invalid relationship source: {relationship.source_id}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_relationship_source"
                    ))

                # Validate target exists
                if not self.get_context(relationship.target_id):
                    messages.append(ValidationMessage(
                        path=f"context.{context.context_id}.relationships",
                        message=f"Invalid relationship target: {relationship.target_id}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_relationship_target"
                    ))

                # Validate scope compatibility
                if not self._validate_scope_compatibility(
                    relationship.scope,
                    context.scope
                ):
                    messages.append(ValidationMessage(
                        path=f"context.{context.context_id}.relationships",
                        message=f"Incompatible relationship scope: {relationship.scope.value}",
                        severity=ValidationSeverity.ERROR,
                        code="incompatible_relationship_scope"
                    ))

                # Validate relationship type rules
                if rules := self.config_manager.get_processing_rules(
                    element_type=context.element_type,
                    context=context
                ).get('relationship_rules', {}):
                    if not self._validate_relationship_rules(relationship, rules):
                        messages.append(ValidationMessage(
                            path=f"context.{context.context_id}.relationships",
                            message=f"Invalid relationship type: {relationship.relation_type.value}",
                            severity=ValidationSeverity.ERROR,
                            code="invalid_relationship_type"
                        ))

                # Check for circular references
                if self._has_circular_reference(
                    relationship.source_id,
                    relationship.target_id
                ):
                    messages.append(ValidationMessage(
                        path=f"context.{context.context_id}.relationships",
                        message="Circular relationship detected",
                        severity=ValidationSeverity.ERROR,
                        code="circular_relationship"
                    ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating relationships: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path=f"context.{context.context_id}",
                    message=f"Relationship validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="relationship_validation_error"
                )]
            )

    def _validate_relationship_rules(
        self,
        relationship: ContentRelationship,
        rules: Dict[str, Any]
    ) -> bool:
        """
        Validate relationship against configured rules.

        Args:
            relationship: Relationship to validate
            rules: Configuration rules

        Returns:
            bool: True if relationship is valid
        """
        try:
            # Check allowed relationship types
            if allowed_types := rules.get('allowed_types', []):
                if relationship.relation_type.value not in allowed_types:
                    return False

            # Check scope rules
            if scope_rules := rules.get('scope_rules', {}):
                if allowed_scopes := scope_rules.get('allowed_scopes', []):
                    if relationship.scope.value not in allowed_scopes:
                        return False

            # Check relationship-specific rules
            if type_rules := rules.get(relationship.relation_type.value, {}):
                # Check cardinality
                if max_targets := type_rules.get('max_targets'):
                    target_count = len([
                        rel for rel in self._content_relationships.get(
                            relationship.source_id, []
                        )
                        if rel.relation_type == relationship.relation_type
                    ])
                    if target_count >= max_targets:
                        return False

                # Check allowed target types
                if allowed_targets := type_rules.get('allowed_targets', []):
                    target_context = self.get_context(relationship.target_id)
                    if target_context and target_context.element_type.value not in allowed_targets:
                        return False

            return True

        except Exception as e:
            self.logger.error(
                f"Error validating relationship rules: {str(e)}"
            )
            return False


    ##################
    # Helper methods #
    ##################

    def _get_children(self, content_id: str) -> List[str]:
        """
        Get immediate children of a content element.

        Args:
            content_id: Content identifier

        Returns:
            List of child content IDs
        """
        try:
            # Check relationships for children
            return [
                rel.target_id for rel in self._content_relationships.get(content_id, [])
                if rel.relation_type == ContentRelationType.CHILD
            ]
        except Exception as e:
            self.logger.error(f"Error getting children: {str(e)}")
            return []

    def _validate_hierarchy_operation(
        self,
        content_id: str,
        parent_id: Optional[str]
    ) -> bool:
        """
        Validate a hierarchy operation.

        Args:
            content_id: Content being added/moved
            parent_id: Potential parent ID

        Returns:
            bool: True if operation is valid
        """
        try:
            # Self-reference check
            if content_id == parent_id:
                return False

            # If no parent, valid root operation
            if not parent_id:
                return True

            # Check for circular reference
            if self._has_circular_reference(parent_id, content_id):
                return False

            # Get contexts
            content_context = self._active_contexts.get(content_id)
            parent_context = self._active_contexts.get(parent_id)

            if not content_context or not parent_context:
                return False

            # Validate element types can be nested
            valid_nesting = {
                ElementType.MAP: {ElementType.TOPIC, ElementType.MAP},
                ElementType.TOPIC: {ElementType.TOPIC},
                # Add other nesting rules as needed
            }

            parent_type = parent_context.element_type
            content_type = content_context.element_type

            if parent_type in valid_nesting:
                return content_type in valid_nesting[parent_type]

            return False

        except Exception as e:
            self.logger.error(f"Error validating hierarchy: {str(e)}")
            return False

    def _update_child_levels(
        self,
        parent_id: str,
        parent_level: int
    ) -> None:
        """
        Recursively update levels of child elements.

        Args:
            parent_id: Parent content identifier
            parent_level: Level of parent element
        """
        try:
            children = self._get_children(parent_id)
            for child_id in children:
                child_context = self._active_contexts.get(child_id)
                if child_context:
                    # Update navigation level
                    child_context.navigation.level = parent_level + 1

                    # Recursively update children
                    self._update_child_levels(child_id, parent_level + 1)

                    # Invalidate cache
                    self.content_cache.invalidate_by_pattern(f"context_{child_id}")

        except Exception as e:
            self.logger.error(f"Error updating child levels: {str(e)}")



    def cleanup(self) -> None:
        """Clean up manager resources."""
        try:
            self._active_contexts.clear()
            self._content_relationships.clear()
            self._navigation_contexts.clear()
            self._hierarchy_paths.clear()
            self.logger.debug("Context manager cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise
