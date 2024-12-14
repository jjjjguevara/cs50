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
    ProcessingPhase,
    ProcessingState,
    ProcessingContext,
    ProcessingError,
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
    def __init__(self, db_path: Path):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self._conn = self._init_db_connection()
        self._active_contexts: Dict[str, ProcessingContext] = {}

    def _init_db_connection(self) -> sqlite3.Connection:
        """Initialize SQLite connection with proper settings."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def create_topic_context(
        self,
        topic: Topic,
        map_id: str,
        phase: ProcessingPhase
    ) -> TopicContext:
        """Create a new topic processing context."""
        try:
            # Create base context
            base = BaseContext(
                id=topic.id,
                content_type='topic',
                processing_phase=phase,
                processing_state=ProcessingState.PENDING,
                features={},
                metadata={}
            )

            # Get topic type from database
            topic_type = self._get_topic_type(topic.type_id)

            # Get context path
            context_path = self._get_context_path(topic.id, map_id)

            # Create topic context
            return TopicContext(
                base=base,
                map_id=map_id,
                parent_id=topic.parent_topic_id,
                level=len(context_path.split('/')),
                sequence_num=self._get_next_sequence(map_id),
                heading_number=None,  # Will be set during processing
                context_path=context_path,
                topic_type=topic_type,
                processing_features=ProcessingFeatures()
            )

        except Exception as e:
            self.logger.error(f"Error creating topic context: {str(e)}")
            raise

    def create_map_context(
        self,
        map_obj: Map,
        phase: ProcessingPhase
    ) -> MapContext:
        """Create a new map processing context."""
        try:
            # Create base context
            base = BaseContext(
                id=map_obj.id,
                content_type='map',
                processing_phase=phase,
                processing_state=ProcessingState.PENDING,
                features={},
                metadata={}
            )

            # Get topic order from database
            topic_order = self._get_topic_order(map_obj.id)

            return MapContext(
                base=base,
                topic_order=topic_order,
                root_context=str(map_obj.context_root or ''),
                conditions=self._get_map_conditions(map_obj.id),
                features=ProcessingFeatures(  # Changed from processing_features
                    needs_heading_numbers=map_obj.index_numbers_enabled,
                    needs_toc=map_obj.toc_enabled
                )
            )

        except Exception as e:
            self.logger.error(f"Error creating map context: {str(e)}")
            raise

    def get_element_context(
        self,
        element_id: str,
        topic_id: str
    ) -> Optional[ElementContext]:
        """Get context for a specific element."""
        try:
            cur = self._conn.cursor()
            cur.execute("""
                SELECT e.*, ec.*
                FROM topic_elements e
                JOIN element_context ec ON e.element_id = ec.element_id
                WHERE e.element_id = ? AND e.topic_id = ?
            """, (element_id, topic_id))

            row = cur.fetchone()
            if not row:
                return None

            return ElementContext(
                element_id=row['element_id'],
                topic_id=row['topic_id'],
                element_type=row['element_type'],
                context_type=row['context_type'],
                parent_context=row['parent_context'],
                level=row['level'],
                xpath=row['xpath'],
                conditions=json.loads(row['conditions'] or '{}'),
                processing_features=ProcessingFeatures(**json.loads(row['processing_features'] or '{}'))
            )

        except Exception as e:
            self.logger.error(f"Error getting element context: {str(e)}")
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
            Dict containing organized conditions and their values
        """
        try:
            cur = self._conn.cursor()

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

            # Process each condition
            for row in dict(cur.fetchall()):
                attr_name = row['name']
                attr_type = row['attribute_type']
                attr_value = row['value']
                attr_scope = row['scope']
                is_toggle = row['is_toggle']

                # Handle different attribute types
                if attr_type == 'content_toggle':
                    if is_toggle:
                        # Convert string 'true'/'false' to boolean for toggles
                        conditions['content_toggles'][attr_name] = attr_value.lower() == 'true'
                    else:
                        # Store as regular value for non-toggles
                        conditions['content_toggles'][attr_name] = attr_value

                elif attr_type == 'version_toggle':
                    conditions['version_toggles'][attr_name] = {
                        'value': attr_value,
                        'scope': attr_scope
                    }

                # Handle feature flags and processing options
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
                    'conditions': json.loads(context_row['conditions'] or '{}')
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
        scope: ContentScope,
        context: Optional[ProcessingContext] = None
    ) -> bool:
        """
        Evaluate if content should be processed based on conditions.

        Args:
            conditions: Conditions dict from get_topic_conditions
            scope: The scope of evaluation (MAP, TOPIC, ELEMENT)
            context: Optional processing context for context-aware evaluation

        Returns:
            bool: True if content should be processed
        """
        try:
            # Check if processing is enabled
            if not conditions['processing']['enabled']:
                return False

            # Evaluate content toggles
            for name, value in conditions['content_toggles'].items():
                if isinstance(value, bool) and not value:
                    return False

            # Evaluate version toggles if context provided
            if context and conditions['version_toggles']:
                current_version = context.map_context.base.metadata.get('version')  # Updated path
                if current_version:
                    for name, info in conditions['version_toggles'].items():
                        if info['value'] > current_version:
                            return False

            # Check features
            required_features = {
                feature for feature, enabled in conditions['features'].items()
                if enabled
            }

            if context and required_features:
                # Access features through ProcessingFeatures object
                available_features = set(context.map_context.features.__dict__.keys())
                if not required_features.issubset(available_features):
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error evaluating conditions: {str(e)}")
            return True  # Default to showing content on error

    def update_processing_state(
        self,
        content_id: str,
        phase: ProcessingPhase,
        state: ProcessingState,
        error: Optional[str] = None
    ) -> None:
        """Update processing state in database."""
        try:
            cur = self._conn.cursor()
            cur.execute("""
                UPDATE processing_contexts
                SET state = ?, error = ?, updated_at = ?
                WHERE content_id = ? AND phase = ?
            """, (state.value, error, datetime.now().isoformat(),
                 content_id, phase.value))
            self._conn.commit()

        except Exception as e:
            self.logger.error(f"Error updating processing state: {str(e)}")
            raise

    def _get_topic_type(self, type_id: int) -> TopicType:
        """Get topic type information from database."""
        cur = self._conn.cursor()
        cur.execute("""
            SELECT * FROM topic_types WHERE type_id = ?
        """, (type_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Topic type {type_id} not found")

        return TopicType(
            id=row['type_id'],
            name=row['name'],
            base_type=row['base_type'],
            description=row['description'],
            schema_file=row['schema_file'],
            is_custom=row['is_custom']
        )

    def _get_context_path(self, topic_id: str, map_id: str) -> str:
        """Get context path for topic in map."""
        cur = self._conn.cursor()
        cur.execute("""
            SELECT context_path
            FROM context_hierarchy
            WHERE topic_id = ? AND map_id = ?
        """, (topic_id, map_id))
        row = cur.fetchone()
        return row['context_path'] if row else f"/{topic_id}"

    def _get_topic_order(self, map_id: str) -> List[str]:
        """Get ordered list of topic IDs for map."""
        cur = self._conn.cursor()
        cur.execute("""
            SELECT topic_id
            FROM map_topics
            WHERE map_id = ?
            ORDER BY sequence_num
        """, (map_id,))
        return [row['topic_id'] for row in cur.fetchall()]

    def _get_map_conditions(self, map_id: str) -> Dict[str, Any]:
        """Get conditional attributes for map."""
        try:
            return self.get_topic_conditions(
                topic_id=map_id,
                map_id=None  # Use the method's existing signature
            )
        except Exception as e:
            self.logger.error(f"Error getting map conditions: {str(e)}")
            return {}

    def _get_next_sequence(self, map_id: str) -> int:
        """Get next sequence number for map topics."""
        cur = self._conn.cursor()
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
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
