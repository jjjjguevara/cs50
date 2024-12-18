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
    Provides framework for contextual processing without enforcing processing rules.
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
            metadata_handler: For persistent context storage
            config_manager: For accessing feature flags
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
        self._scope_registry: Dict[str, ContentScope] = {}
        self._hierarchy_paths: Dict[str, List[str]] = {}

        # Initialize context rules
        self._init_context_rules()

        # Context rules registry
        self.CONTEXT_RULES = {
            # Scientific Publication Context
            "journal": {
                "metadata_hierarchy": [
                    "journal_metadata",  # Journal-level metadata
                    "issue_metadata",    # Issue-specific
                    "article_metadata",  # Article-level
                    "section_metadata",  # Section-specific
                    "content_metadata"   # Content-specific
                ],

                "content_relationships": {
                    "article": {
                        "required": ["abstract", "authors", "doi"],
                        "optional": ["supplementary", "acknowledgments"],
                        "allowed_children": ["section", "appendix", "references"],
                        "allowed_parents": ["issue", "volume"]
                    },
                    "section": {
                        "required": ["title"],
                        "allowed_children": ["subsection", "paragraph", "figure"],
                        "allowed_parents": ["article", "section"]
                    }
                },

                "reuse_contexts": {
                    "citation": {
                        "scope": "global",
                        "validation": ["doi", "authors", "year"],
                        "required_metadata": ["citation_style", "reference_type"]
                    },
                    "equation": {
                        "scope": "local",
                        "validation": ["equation_id", "latex_content"],
                        "required_metadata": ["equation_number", "reference_type"]
                    }
                },

                "attribute_inheritance": {
                    "audience": {
                        "inherit": True,
                        "override": "child",
                        "valid_values": ["researcher", "student", "practitioner"]
                    },
                    "access_level": {
                        "inherit": True,
                        "override": "parent",
                        "valid_values": ["public", "subscriber", "institution"]
                    }
                }
            }
        }

        self.CONTENT_SCOPES = {
            "local": {
                "description": "Content valid within current topic",
                "allows_external_refs": False
            },
            "map": {
                "description": "Content valid within current map",
                "allows_external_refs": True,
                "requires_validation": True
            },
            "global": {
                "description": "Content valid across all maps",
                "allows_external_refs": True,
                "requires_validation": True
            }
        }


    #########################
    # Core mediator methods #
    #########################

    def _init_context_rules(self) -> None:
            """Initialize context management rules."""
            try:
                # Register scopes
                for scope_name, scope_info in self.CONTENT_SCOPES.items():
                    self._scope_registry[scope_name] = ContentScope(scope_name)

                # Log initialization
                self.logger.debug("Context rules initialized successfully")

            except Exception as e:
                self.logger.error(f"Failed to initialize context rules: {str(e)}")
                raise

    def register_context(
        self,
        content_id: str,
        context_type: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Register a new context with proper caching."""
        try:
            cache_key = f"context_{content_id}"

            # Create structured context data
            context_data = {
                "type": context_type,
                "metadata": metadata,
                "registered_at": datetime.now().isoformat()
            }

            # Store in cache with required parameters
            self.content_cache.set(
                key=cache_key,
                data=context_data,
                element_type=ElementType.UNKNOWN,  # We'll use UNKNOWN since this is context data
                phase=ProcessingPhase.DISCOVERY,   # Default to discovery phase
                ttl=3600  # Cache for 1 hour
            )

            # Store in metadata handler
            self.metadata_handler.store_metadata(content_id, metadata)

        except Exception as e:
            self.logger.error(f"Error registering context: {str(e)}")
            raise

    def update_context(
        self,
        content_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """Update existing context with proper caching."""
        try:
            cache_key = f"context_{content_id}"

            # Get current context
            current = self.get_context(content_id) or {}

            # Merge updates
            current.update(updates)

            # Update cache with required parameters
            self.content_cache.set(
                key=cache_key,
                data=current,
                element_type=ElementType.UNKNOWN,
                phase=ProcessingPhase.DISCOVERY,
                ttl=3600
            )

            # Update persistent storage
            self.metadata_handler.store_metadata(content_id, current)

        except Exception as e:
            self.logger.error(f"Error updating context: {str(e)}")
            raise


    def get_context(self, content_id: str) -> Dict[str, Any]:
       """Get context with proper caching and metadata handling."""
       try:
           cache_key = f"context_{content_id}"

           # Check cache first
           if cached := self.content_cache.get(cache_key):
               return cached

           # Get from metadata storage using store_metadata method
           # Note: We'll use store_metadata since get_metadata isn't defined
           context = {}
           try:
               with self.metadata_handler.transaction(content_id) as txn:
                   context = txn.updates  # Get current metadata state
           except Exception as e:
               self.logger.warning(f"Could not retrieve metadata for {content_id}: {str(e)}")

           # Get parent context if exists
           if parent_id := self._hierarchy_paths.get(content_id, [None])[0]:
               parent_context = self.get_context(parent_id)
               context = self._merge_contexts(parent_context, context)

           # Cache result with required parameters
           self.content_cache.set(
               key=cache_key,
               data=context,
               element_type=ElementType.UNKNOWN,
               phase=ProcessingPhase.DISCOVERY,
               ttl=3600
           )

           return context

       except Exception as e:
           self.logger.error(f"Error getting context: {str(e)}")
           return {}

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
        parent: Dict[str, Any],
        child: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Helper method to merge contexts following inheritance rules."""
        merged = parent.copy()

        for attr, value in child.items():
            inheritance_rule = self.CONTEXT_RULES["attribute_inheritance"].get(attr, {})
            if inheritance_rule.get("inherit", True):
                if inheritance_rule.get("override") == "child":
                    merged[attr] = value
            else:
                merged[attr] = value

        return merged

    def _invalidate_dependent_contexts(self, content_id: str) -> None:
        """Helper method to invalidate dependent contexts."""
        # Get dependent IDs (children in hierarchy)
        dependent_ids = [
            id for id, path in self._hierarchy_paths.items()
            if content_id in path
        ]

        # Invalidate each dependent context
        for dep_id in dependent_ids:
            self.invalidate_context(dep_id)

    ###########################
    # Relationship management #
    ##########################

    def register_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: ContentRelationType,
        scope: Optional[ContentScope] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a relationship between content elements.

        Args:
            source_id: Source content identifier
            target_id: Target content identifier
            relationship_type: Type of relationship
            scope: Optional relationship scope (defaults to LOCAL)
            metadata: Optional relationship metadata
        """
        try:
            # Validate the relationship first
            if not self.validate_relationship(source_id, target_id):
                raise ValueError(f"Invalid relationship between {source_id} and {target_id}")

            # Create relationship object
            relationship = ContentRelationship(
                source_id=source_id,
                target_id=target_id,
                relation_type=relationship_type,
                scope=scope or ContentScope.LOCAL,
                metadata=metadata or {}
            )

            # Add to relationships registry
            if source_id not in self._content_relationships:
                self._content_relationships[source_id] = []
            self._content_relationships[source_id].append(relationship)

            # Store relationship in metadata handler
            self.metadata_handler.store_content_relationships(
                source_id,
                [{
                    'target_id': target_id,
                    'type': relationship_type.value,
                    'scope': relationship.scope.value,
                    'metadata': relationship.metadata
                }]
            )

            # Invalidate related caches
            self.content_cache.invalidate_pattern(f"context_{source_id}")
            self.content_cache.invalidate_pattern(f"context_{target_id}")

            self.logger.debug(
                f"Registered relationship: {source_id} -> {target_id} "
                f"({relationship_type.value})"
            )

        except Exception as e:
            self.logger.error(f"Error registering relationship: {str(e)}")
            raise

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

    def validate_content_relationship(
        self,
        source: str,
        target: str,
        relationship_type: ContentRelationType
    ) -> bool:
        """
        Validate relationships between content elements considering context and type.

        Args:
            source: Source content identifier
            target: Target content identifier
            relationship_type: Type of relationship to validate

        Returns:
            bool: True if relationship is valid for the content context
        """
        try:
            # Get contexts
            source_context = self._active_contexts.get(source)
            target_context = self._active_contexts.get(target)

            if not source_context or not target_context:
                self.logger.warning(f"Missing context for relationship validation: {source} -> {target}")
                return False

            # Define valid relationship rules for scientific content
            valid_relationships = {
                ContentRelationType.PREREQ: {
                    # Source -> Target allowed combinations
                    (ElementType.TOPIC, ElementType.TOPIC): True,  # Topic can require another topic
                    (ElementType.MAP, ElementType.MAP): True,      # Map can require another map
                    (ElementType.TOPIC, ElementType.MAP): False,   # Topic cannot require an entire map
                    (ElementType.MAP, ElementType.TOPIC): True     # Map can require specific topics
                },
                ContentRelationType.RELATED: {
                    # Allow related content between similar types
                    (ElementType.TOPIC, ElementType.TOPIC): True,
                    (ElementType.MAP, ElementType.MAP): True,
                    (ElementType.TOPIC, ElementType.MAP): True,
                    (ElementType.MAP, ElementType.TOPIC): True
                },
                ContentRelationType.REFERENCE: {
                    # References have more lenient rules
                    (ElementType.TOPIC, ElementType.TOPIC): True,
                    (ElementType.MAP, ElementType.MAP): True,
                    (ElementType.TOPIC, ElementType.MAP): True,
                    (ElementType.MAP, ElementType.TOPIC): True
                }
            }

            # Check if relationship type is valid for the element types
            type_pair = (source_context.element_type, target_context.element_type)
            if not valid_relationships.get(relationship_type, {}).get(type_pair, False):
                self.logger.warning(
                    f"Invalid relationship type {relationship_type} "
                    f"for {type_pair[0]} -> {type_pair[1]}"
                )
                return False

            # Validate scope compatibility
            scope_rules = {
                ContentScope.LOCAL: {ContentScope.LOCAL, ContentScope.PEER},
                ContentScope.PEER: {ContentScope.LOCAL, ContentScope.PEER},
                ContentScope.EXTERNAL: {ContentScope.LOCAL, ContentScope.PEER, ContentScope.EXTERNAL},
                ContentScope.GLOBAL: {ContentScope.LOCAL, ContentScope.PEER, ContentScope.EXTERNAL, ContentScope.GLOBAL}
            }

            if target_context.scope not in scope_rules.get(source_context.scope, set()):
                self.logger.warning(
                    f"Invalid scope relationship: {source_context.scope} -> {target_context.scope}"
                )
                return False

            # Check for circular references
            if self._has_circular_reference(source, target):
                self.logger.warning(f"Circular reference detected: {source} -> {target}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating content relationship: {str(e)}")
            return False

    def validate_attribute_inheritance(
        self,
        content_id: str,
        attribute: str
    ) -> bool:
        """
        Validate attribute inheritance based on context hierarchy.

        Args:
            content_id: Content identifier
            attribute: Attribute to validate

        Returns:
            bool: True if attribute inheritance is valid
        """
        try:
            # Get current context
            context = self._active_contexts.get(content_id)
            if not context:
                return False

            # Define attribute inheritance rules for scientific content
            inheritance_rules = {
                "audience": {
                    "inheritable": True,
                    "override_allowed": True,
                    "valid_values": {"researchers", "students", "practitioners"},
                    "scope": ContentScope.LOCAL
                },
                "distribution": {
                    "inheritable": True,
                    "override_allowed": False,
                    "valid_values": {"public", "private", "institutional"},
                    "scope": ContentScope.GLOBAL
                },
                "review_status": {
                    "inheritable": False,
                    "override_allowed": True,
                    "valid_values": {"draft", "peer_review", "published"},
                    "scope": ContentScope.LOCAL
                },
                "publication_state": {
                    "inheritable": True,
                    "override_allowed": False,
                    "valid_values": {"preprint", "published", "retracted"},
                    "scope": ContentScope.GLOBAL
                }
            }

            # Check if attribute is defined in rules
            if attribute not in inheritance_rules:
                self.logger.warning(f"Unknown attribute for inheritance: {attribute}")
                return False

            # Get attribute rules
            rules = inheritance_rules[attribute]

            # Check if attribute is inheritable
            if not rules["inheritable"]:
                return False

            # Check scope compatibility
            if context.scope != rules["scope"]:
                return False

            # If attribute allows override, we don't need to check parent
            if rules["override_allowed"]:
                return True

            # If no override allowed, check if parent has the attribute
            parent_id = context.navigation.parent_id
            if parent_id:
                parent_context = self._active_contexts.get(parent_id)
                if parent_context:
                    parent_value = parent_context.metadata_refs.get(attribute)
                    if parent_value and parent_value not in rules["valid_values"]:
                        return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating attribute inheritance: {str(e)}")
            return False

    def validate_reuse_context(
        self,
        content_id: str,
        reuse_type: str
    ) -> bool:
        """
        Validate content reuse based on type and context.

        Args:
            content_id: Content identifier
            reuse_type: Type of content reuse (e.g., 'conref', 'keyref')

        Returns:
            bool: True if reuse is valid in current context
        """
        try:
            context = self._active_contexts.get(content_id)
            if not context:
                return False

            # Define reuse rules for scientific content
            reuse_rules = {
                "conref": {
                    "allowed_scopes": {ContentScope.LOCAL, ContentScope.PEER},
                    "allowed_elements": {
                        ElementType.TOPIC: {
                            "sections": ["abstract", "methodology", "results", "discussion"],
                            "max_depth": 2
                        },
                        ElementType.MAP: {
                            "sections": ["front_matter", "back_matter"],
                            "max_depth": 1
                        }
                    },
                    "requires_citation": True
                },
                "keyref": {
                    "allowed_scopes": {ContentScope.LOCAL, ContentScope.PEER, ContentScope.GLOBAL},
                    "allowed_elements": {
                        ElementType.TOPIC: {
                            "sections": ["all"],
                            "max_depth": None
                        },
                        ElementType.MAP: {
                            "sections": ["all"],
                            "max_depth": None
                        }
                    },
                    "requires_citation": False
                }
            }

            # Check if reuse type is supported
            if reuse_type not in reuse_rules:
                self.logger.warning(f"Unsupported reuse type: {reuse_type}")
                return False

            rules = reuse_rules[reuse_type]

            # Validate scope
            if context.scope not in rules["allowed_scopes"]:
                self.logger.warning(
                    f"Invalid scope for {reuse_type}: {context.scope}"
                )
                return False

            # Get element rules
            element_rules = rules["allowed_elements"].get(context.element_type)
            if not element_rules:
                return False

            # Check section constraints
            current_section = context.metadata_refs.get("section")
            if current_section and "all" not in element_rules["sections"]:
                if current_section not in element_rules["sections"]:
                    return False

            # Check depth constraints
            if element_rules["max_depth"] is not None:
                current_depth = len(self.get_content_path(content_id))
                if current_depth > element_rules["max_depth"]:
                    return False

            # Check citation requirement
            if rules["requires_citation"]:
                has_citation = bool(context.metadata_refs.get("citation"))
                if not has_citation:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating reuse context: {str(e)}")
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
                    self.content_cache.invalidate_pattern(f"context_{child_id}")

        except Exception as e:
            self.logger.error(f"Error updating child levels: {str(e)}")
