# app/dita/context_manager.py
from typing import Optional, TYPE_CHECKING
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import sqlite3
import logging
from datetime import datetime
import json
from .config_manager import DITAConfig
from .models.types import (
    TrackedElement,
    ProcessingPhase,
    ProcessingState,
    ProcessingContext,
    ProcessingError,
    ProcessingMetadata,
    ElementType,
    Topic,
    Map
)

if TYPE_CHECKING:
    from .config_manager import ConfigManager


class ContextManager:
    """
    Manages processing contexts, conditional attributes, and element tracking.
    Provides centralized context management for the DITA processing pipeline.
    """
    def __init__(self, db_path: str):
        """Initialize the ContextManager."""
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None  # Explicitly define type
        self._active_contexts: List[ProcessingContext] = []
        self.context_stack: List[ProcessingContext] = []  # Initialize context stack

    def _init_db_connection(self) -> sqlite3.Connection:
        """Initialize SQLite connection with proper settings."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable dict-like row access
            self.logger.debug("Database connection initialized")
            return conn
        except Exception as e:
            self.logger.error(f"Error initializing database connection: {str(e)}")
            raise

    def _ensure_db_connection(self) -> sqlite3.Connection:
        """Ensure database connection is initialized and return it."""
        if self._conn is None:
            self._conn = self._init_db_connection()
        return self._conn



    def create_context(self, element: TrackedElement, map_id: str) -> ProcessingContext:
        """
        Create a new processing context for a given element.

        Args:
            element: A `TrackedElement` instance representing the content.
            map_id: The ID of the map this context belongs to.

        Returns:
            ProcessingContext: The processing context for the pipeline.
        """
        try:
            # Initialize the context with map-level information
            context = ProcessingContext(
                map_id=map_id,
                map_metadata=element.metadata.get("map_metadata", {}),
                topic_metadata=element.metadata.get("topic_metadata", {}),
            )

            # Register the element in the context
            if element.type == ElementType.DITAMAP:
                context.map_id = element.id
            elif element.type in [ElementType.DITA, ElementType.MARKDOWN]:
                context.set_topic(
                    topic_id=element.id,
                    metadata=element.metadata
                )

            return context

        except Exception as e:
            self.logger.error(f"Error creating processing context: {str(e)}")
            raise


    def get_current_context(self) -> Optional[ProcessingContext]:
            """
            Get the current processing context.

            Returns:
                Optional[ProcessingContext]: The current context, or None if the stack is empty.
            """
            try:
                return self.context_stack[-1] if self.context_stack else None
            except Exception as e:
                self.logger.error(f"Error accessing current context: {str(e)}")
                return None

    def pop_context(self) -> Optional[ProcessingContext]:
        """
        Remove and return the current processing context.

        Returns:
            Optional[ProcessingContext]: The popped context, or None if the stack is empty.
        """
        try:
            if self.context_stack:
                context = self.context_stack.pop()
                self.logger.debug(f"Popped context with map ID {context.map_id}")
                return context
            self.logger.warning("No context to pop")
            return None
        except Exception as e:
            self.logger.error(f"Error popping context: {str(e)}")
            return None

    def push_context(self, context: ProcessingContext) -> None:
        """
        Push a new context onto the stack.

        Args:
            context: The `ProcessingContext` to push onto the stack.
        """
        try:
            self.context_stack.append(context)
            self.logger.debug(f"Pushed context with map ID {context.map_id}")
        except Exception as e:
            self.logger.error(f"Error pushing context: {str(e)}")

    def update_current_context(self, key: str, value: Any) -> None:
        """Update metadata in the current processing context."""
        current_context = self.get_current_context()
        if not current_context:
            self.logger.error("No current context to update")
            return
        current_context.map_metadata[key] = value
        self.logger.debug(f"Updated current context metadata: {key} -> {value}")


    def create_topic_context(
        self, element: TrackedElement, map_id: str, phase: ProcessingPhase
    ) -> ProcessingContext:
        """
        Create a processing context for a topic.

        Args:
            element: The `TrackedElement` representing the topic.
            map_id: ID of the parent map.
            phase: The current processing phase.

        Returns:
            A `ProcessingContext` configured for the topic.
        """
        try:
            # Update metadata directly in the element
            element.metadata.update({
                "parent_map_id": map_id,
                "processing_phase": phase.value,
                "sequence_num": self._get_next_sequence(map_id),
            })

            return ProcessingContext(
                map_id=map_id,
                current_topic_id=element.id,
                map_metadata={
                    "context_path": f"{map_id}/{element.id}",
                    "topic_type": element.type.value,
                },
                topic_metadata={
                    element.id: element.metadata,
                },
            )
        except Exception as e:
            self.logger.error(f"Error creating topic context for {element.id}: {str(e)}")
            raise


    def create_map_context(
        self, element: TrackedElement, phase: ProcessingPhase
    ) -> ProcessingContext:
        """
        Create a processing context for a map.

        Args:
            element: The `TrackedElement` representing the map.
            phase: The current processing phase.

        Returns:
            A `ProcessingContext` configured for the map.
        """
        try:
            # Update metadata directly in the element
            element.metadata.update({
                "processing_phase": phase.value,
                "index_numbers_enabled": element.metadata.get("index_numbers_enabled", True),
                "toc_enabled": element.metadata.get("toc_enabled", True),
            })

            return ProcessingContext(
                map_id=element.id,
                features={
                    "process_latex": element.metadata.get("latex_enabled", False),
                    "number_headings": element.metadata.get("index_numbers_enabled", True),
                    "show_toc": element.metadata.get("toc_enabled", True),
                },
                map_metadata={
                    "title": element.title,
                    "context_root": element.metadata.get("context_root", ""),
                },
                topic_order=element.metadata.get("topic_order", []),
            )
        except Exception as e:
            self.logger.error(f"Error creating map context for {element.id}: {str(e)}")
            raise




    def get_element_context(self, element: TrackedElement) -> Optional[ProcessingContext]:
        """
        Get context for a specific element.

        Args:
            element: The `TrackedElement` to retrieve context for.

        Returns:
            A `ProcessingContext` for the element, or None if not found.
        """
        try:
            conn = self._ensure_db_connection()
            cur = conn.cursor()
            cur.execute("""
                SELECT e.*, ec.*
                FROM topic_elements e
                JOIN element_context ec ON e.element_id = ec.element_id
                WHERE e.element_id = ?
            """, (element.id,))

            row = cur.fetchone()
            if not row:
                self.logger.debug(f"No context found for element ID: {element.id}")
                return None

            # Construct and return the ProcessingContext
            return ProcessingContext(
                map_id=row["map_id"],
                current_topic_id=row["topic_id"],
                topic_metadata={
                    row["topic_id"]: json.loads(row["metadata"] or "{}"),
                },
                features=json.loads(row["features"] or "{}"),
            )
        except Exception as e:
            self.logger.error(f"Error getting element context for {element.id}: {str(e)}")
            return None



    def get_topic_conditions(
        self,
        topic_id: str,
        map_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get all conditional attributes for a topic, including inherited conditions from map.
        Handles content toggles, version toggles, and feature flags.

        Args:
            topic_id: ID of the topic
            map_id: Optional map ID for context-aware conditions

        Returns:
            Dict containing organized conditions and their values.
        """
        try:
            # Ensure database connection is initialized
            if self._conn is None:
                self._conn = self._init_db_connection()

            cur = self._conn.cursor()

            # Initialize conditions dictionary
            conditions = {
                'content_toggles': {},
                'version_toggles': {},
                'features': {},
                'metadata': {},
                'processing': {
                    'enabled': True,
                    'phase': None,
                    'context': {}
                }
            }

            # Get topic-specific conditions
            cur.execute("""
                SELECT
                    ca.attribute_id,
                    ca.name,
                    ca.attribute_type,
                    ca.scope,
                    ca.is_toggle,
                    cv.value,
                    cc.content_type,
                    ca.context_dependent
                FROM content_conditions cc
                JOIN conditional_attributes ca ON cc.attribute_id = ca.attribute_id
                JOIN conditional_values cv ON cc.value_id = cv.value_id
                WHERE (cc.content_id = ? AND cc.content_type = 'topic')
                   OR (cc.content_id = ? AND cc.content_type = 'map' AND ca.scope = 'global')
                   OR (ca.scope = 'global' AND ca.context_dependent = FALSE)
                ORDER BY ca.scope DESC
            """, (topic_id, map_id if map_id else ''))

            # Process conditions
            for row in cur.fetchall():
                attr_name = row['name']
                attr_type = row['attribute_type']
                attr_value = row['value']
                attr_scope = row['scope']
                is_toggle = row['is_toggle']

                if attr_type == 'content_toggle':
                    if is_toggle:
                        conditions['content_toggles'][attr_name] = attr_value.lower() == 'true'
                    else:
                        conditions['content_toggles'][attr_name] = attr_value

                elif attr_type == 'version_toggle':
                    conditions['version_toggles'][attr_name] = {
                        'value': attr_value,
                        'scope': attr_scope
                    }

                if attr_name in ['process-latex', 'show-toc', 'index-numbers', 'process-artifacts']:
                    conditions['features'][attr_name] = attr_value.lower() == 'true'

            # Get processing context if exists
            cur.execute("""
                SELECT phase, state, features, conditions
                FROM processing_contexts
                WHERE content_id = ? AND content_type = 'topic'
                ORDER BY created_at DESC
                LIMIT 1
            """, (topic_id,))

            if context_row := cur.fetchone():
                conditions['processing'].update({
                    'phase': context_row['phase'],
                    'state': context_row['state'],
                    'features': json.loads(context_row['features'] or '{}'),
                    'context': json.loads(context_row['conditions'] or '{}')
                })

            # Get metadata attributes
            cur.execute("""
                SELECT t.topic_type, t.specialization_type, t.status, t.language
                FROM topics t
                WHERE t.topic_id = ?
            """, (topic_id,))

            if metadata_row := cur.fetchone():
                conditions['metadata'] = dict(metadata_row)

            self.logger.debug(f"Retrieved conditions for topic {topic_id}: {conditions}")
            return conditions

        except Exception as e:
            self.logger.error(f"Error getting topic conditions: {str(e)}")
            # Return safe defaults
            return {
                'content_toggles': {},
                'version_toggles': {},
                'features': {
                    'process-latex': False,
                    'show-toc': True,
                    'index-numbers': True,
                    'process-artifacts': False
                },
                'metadata': {},
                'processing': {
                    'enabled': True,
                    'phase': None,
                    'context': {}
                }
            }


    def evaluate_conditions(
        self,
        conditions: Dict[str, Any],
        context: Optional[ProcessingContext] = None
    ) -> bool:
        """
        Evaluate if content should be processed based on conditions.

        Args:
            conditions: Conditions dict from `get_topic_conditions`.
            context: Optional `ProcessingContext` for context-aware evaluation.

        Returns:
            bool: True if content should be processed.
        """
        try:
            # Check if processing is enabled
            if not conditions.get('processing', {}).get('enabled', True):
                return False

            # Evaluate content toggles
            for name, value in conditions.get('content_toggles', {}).items():
                if isinstance(value, bool) and not value:
                    self.logger.debug(f"Content toggle '{name}' is disabled.")
                    return False

            # Evaluate version toggles if context provided
            if context and 'version_toggles' in conditions:
                current_version = context.map_metadata.get('version') or context.topic_metadata.get('version')
                if current_version:
                    for name, info in conditions['version_toggles'].items():
                        if info['value'] > current_version:
                            self.logger.debug(f"Version toggle '{name}' excludes current version '{current_version}'.")
                            return False

            # Check required features
            required_features = {
                feature for feature, enabled in conditions.get('features', {}).items() if enabled
            }

            if context and required_features:
                # Get available features from the context
                available_features = set(context.features.keys())
                missing_features = required_features - available_features
                if missing_features:
                    self.logger.debug(f"Missing required features: {missing_features}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error evaluating conditions: {str(e)}")
            # Default to allowing content in case of errors
            return True


    def update_processing_state(
        self,
        content_id: str,
        phase: ProcessingPhase,
        state: ProcessingState,
        error: Optional[str] = None
    ) -> None:
        """Update processing state in the database."""
        try:
            conn = self._ensure_db_connection()
            cur = conn.cursor()
            cur.execute("""
                UPDATE processing_contexts
                SET state = ?, error = ?, updated_at = ?
                WHERE content_id = ? AND phase = ?
            """, (state.value, error, datetime.now().isoformat(), content_id, phase.value))
            conn.commit()
        except Exception as e:
            self.logger.error(f"Error updating processing state: {str(e)}")
            raise

    def _get_topic_type(self, type_id: int) -> Dict[str, Any]:
        """Get topic type information from the database."""
        conn = self._ensure_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM topic_types WHERE type_id = ?
        """, (type_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Topic type {type_id} not found")

        return {
            "id": row["type_id"],
            "name": row["name"],
            "base_type": row["base_type"],
            "description": row["description"],
            "schema_file": row["schema_file"],
            "is_custom": row["is_custom"],
        }


    def _get_context_path(self, topic_id: str, map_id: str) -> str:
        """Get context path for a topic in a map."""
        conn = self._ensure_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT context_path
            FROM context_hierarchy
            WHERE topic_id = ? AND map_id = ?
        """, (topic_id, map_id))
        row = cur.fetchone()
        return row["context_path"] if row else f"/{topic_id}"



    def _get_topic_order(self, map_id: str) -> List[str]:
        """Get ordered list of topic IDs for a map."""
        conn = self._ensure_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT topic_id
            FROM map_topics
            WHERE map_id = ?
            ORDER BY sequence_num
        """, (map_id,))
        return [row["topic_id"] for row in cur.fetchall()]


    def _get_map_conditions(self, map_id: str) -> Dict[str, Any]:
        """Get conditional attributes for a map."""
        try:
            return self.get_topic_conditions(
                topic_id=map_id,
                map_id=None  # Use the method's existing signature
            )
        except Exception as e:
            self.logger.error(f"Error getting map conditions: {str(e)}")
            return {}


    def _get_next_sequence(self, map_id: str) -> int:
        """Get the next sequence number for map topics."""
        conn = self._ensure_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT COALESCE(MAX(sequence_num), 0) + 1
            FROM map_topics
            WHERE map_id = ?
        """, (map_id,))
        return cur.fetchone()[0]



    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self._active_contexts.clear()
            if self._conn:
                self._conn.close()
                self._conn = None
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
