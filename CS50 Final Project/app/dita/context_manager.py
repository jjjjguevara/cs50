# app/dita/context_manager.py
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from uuid import uuid4

# Event and Cache
from .event_manager import EventManager, EventType
from .utils.cache import ContentCache

# Handlers & Managers
from .utils.metadata import MetadataHandler
from .config_manager import ConfigManager

# Custom Types
from .models.types import (
    ContentScope,
    ContentRelationType,
    ContentRelationship,
    NavigationContext,
    ProcessingContext,
    ElementType,
    TrackedElement,
    ProcessingPhase,
    ProcessingState,
    ProcessingStateInfo
)

# Logger
from .utils.logger import DITALogger

class ContextManager:
    """
    Mediates context relationships and content hierarchy.
    Uses schema-based rules for validation and relationships.
    """
    def __init__(
        self,
        event_manager: EventManager,
        content_cache: ContentCache,
        metadata_handler: MetadataHandler,
        config_manager: ConfigManager,
        logger: DITALogger
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
        self.metadata_handler = metadata_handler
        self.config_manager = config_manager
        self.logger = logger

        # Context tracking
        self._active_contexts: Dict[str, ProcessingContext] = {}
        self._content_relationships: Dict[str, List[ContentRelationship]] = {}
        self._navigation_contexts: Dict[str, NavigationContext] = {}

        # Scope & hierarchy tracking
        self._hierarchy_paths: Dict[str, List[str]] = {}

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
                self.metadata_handler.store_metadata(content_id, metadata_updates)

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

    def register_context(
        self,
        content_id: str,
        element_type: ElementType,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Register new context for content tracking.

        Args:
            content_id: Content identifier
            element_type: Type of element
            metadata: Initial metadata
        """
        try:
            # Create processing context
            context = ProcessingContext(
                context_id=content_id,
                element_id=content_id,
                element_type=element_type,
                state_info=ProcessingStateInfo(
                    phase=ProcessingPhase.DISCOVERY,
                    state=ProcessingState.PENDING,
                    element_id=content_id
                ),
                navigation=NavigationContext(
                    path=[],
                    level=0,
                    sequence=0,
                    parent_id=None,
                    root_map=content_id
                ),
                scope=ContentScope.LOCAL
            )

            # Store context
            self._active_contexts[content_id] = context

            # Store metadata
            self.metadata_handler.store_metadata(content_id, metadata)

            # Emit event
            self.event_manager.emit(
                EventType.STATE_CHANGE,
                element_id=content_id,
                state_info=context.state_info
            )

        except Exception as e:
            self.logger.error(f"Error registering context: {str(e)}")
            raise

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
        """
        Invalidate cached context.

        Args:
            content_id: Content identifier
        """
        try:
            cache_key = f"context_{content_id}"
            self.content_cache.invalidate(cache_key)

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
            cache_key = (
                f"related_{content_id}"
                f"{'_' + relationship_type.value if relationship_type else ''}"
            )

            if cached := self.content_cache.get(cache_key):
                return cached

            # Get relationships from metadata handler
            relationships = self.metadata_handler.get_content_relationships(content_id)

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
                cache_key,
                result,
                ElementType.UNKNOWN,
                ProcessingPhase.DISCOVERY,
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
        """
        Register a node in the content hierarchy.

        Args:
            content_id: Content identifier to register
            parent_id: Optional parent content identifier
        """
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

            with self.metadata_handler.transaction(content_id) as txn:
                txn.updates = {
                    'hierarchy': hierarchy_metadata
                }

            # Update hierarchy paths
            if content_id not in self._hierarchy_paths:
                self._hierarchy_paths[content_id] = []
            if parent_id:
                # Copy parent's path and append current
                parent_path = self._hierarchy_paths.get(parent_id, [])
                self._hierarchy_paths[content_id] = parent_path + [parent_id]

            # Invalidate related caches
            self.content_cache.invalidate_pattern(f"context_{content_id}")
            if parent_id:
                self.content_cache.invalidate_pattern(f"context_{parent_id}")

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
            if cached := self.content_cache.get(cache_key):
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
                cache_key,
                path,
                ElementType.UNKNOWN,
                ProcessingPhase.DISCOVERY,
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
            if cached := self.content_cache.get(cache_key):
                return cached

            # Get level from navigation context
            level = context.navigation.level

            # Cache result
            self.content_cache.set(
                cache_key,
                level,
                ElementType.UNKNOWN,
                ProcessingPhase.DISCOVERY,
                ttl=3600  # Cache for 1 hour
            )

            return level

        except Exception as e:
            self.logger.error(f"Error getting content level: {str(e)}")
            return 0


    ##########################
    # Validation management #
    #########################





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
                    self.content_cache.invalidate_pattern(f"context_{child_id}")

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
