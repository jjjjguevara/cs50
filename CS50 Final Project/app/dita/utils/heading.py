from typing import Dict, Set, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import re
from pathlib import Path

# Core types
from ..models.types import (
    HeadingState,
    ProcessingError,
    ProcessingPhase,
    ProcessingState,
    ContentElement,
    ElementType,
    ValidationResult,
    ContentScope,
    ProcessingContext
)

# Event system for state tracking
from ..event_manager import EventManager, EventType

# Utils
from ..utils.id_handler import DITAIDHandler, IDType
from ..utils.logger import DITALogger

@dataclass
class HeadingMetadata:
    """Metadata for heading tracking."""
    id: str
    level: int
    text: str
    number: Optional[str] = None
    parent_id: Optional[str] = None
    is_topic_title: bool = False
    created_at: datetime = field(default_factory=datetime.now)

class HeadingHandler:
    """Manages heading hierarchy and state tracking."""

    def __init__(
        self,
        event_manager: EventManager,
        id_handler: Optional[DITAIDHandler] = None,
        logger: Optional[DITALogger] = None
    ):
        """Initialize heading handler."""
        # Core dependencies
        self.event_manager = event_manager
        self.id_handler = id_handler or DITAIDHandler()
        self.logger = logger or logging.getLogger(__name__)

        # State tracking
        self._state = HeadingState()

        # Initialize state management
        self._saved_states: List[HeadingState] = []

        # Hierarchy tracking
        self._heading_metadata: Dict[str, HeadingMetadata] = {}
        self._heading_hierarchy: Dict[str, List[str]] = {}

        # Topic tracking
        self._topic_headings: Dict[str, List[str]] = {}
        self._current_topic_id: Optional[str] = None

        # Register for events
        self._register_events()

    def _register_events(self) -> None:
        """Register for relevant events."""
        self.event_manager.subscribe(
            EventType.STATE_CHANGE,
            self._handle_state_change
        )


    # State Management Methods
    def start_new_topic(self, topic_id: str) -> None:
        """
        Initialize state for a new topic.

        Args:
            topic_id: ID of the new topic
        """
        try:
            # Reset state for new topic
            self._state = HeadingState()
            self._current_topic_id = topic_id

            # Initialize topic tracking
            if topic_id not in self._topic_headings:
                self._topic_headings[topic_id] = []

            self.logger.debug(f"Started new topic: {topic_id}")

        except Exception as e:
            self.logger.error(f"Error starting new topic: {str(e)}")
            raise

    def save_state(self) -> None:
        """Save current heading state."""
        try:
            # Create copy of current state
            saved_state = HeadingState(
                current_h1=self._state.current_h1,
                counters=self._state.counters.copy()
            )

            # Store in saved states stack
            self._saved_states.append(saved_state)

            self.logger.debug(f"Saved heading state with H1={saved_state.current_h1}")

        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")
            raise

    def restore_state(self) -> None:
        """Restore previous heading state."""
        try:
            if not self._saved_states:
                self.logger.warning("No saved state to restore")
                return

            # Pop the last saved state
            restored_state = self._saved_states.pop()

            # Restore state
            self._state = restored_state

            self.logger.debug(f"Restored heading state with H1={restored_state.current_h1}")

        except Exception as e:
            self.logger.error(f"Error restoring state: {str(e)}")
            raise

    # Heading Registration Methods
    def register_heading(
        self,
        text: str,
        level: int,
        element: ContentElement,
        is_topic_title: bool = False
    ) -> HeadingMetadata:
        """
        Register a new heading.

        Args:
            text: Heading text
            level: Heading level (1-6)
            element: ContentElement containing the heading
            is_topic_title: Whether this is a topic title

        Returns:
            HeadingMetadata for the registered heading
        """
        try:
            # Validate level
            if not 1 <= level <= 6:
                raise ValueError(f"Invalid heading level: {level}")

            # Generate heading ID
            heading_id = self.id_handler.generate_id(
                base=text,
                id_type=IDType.HEADING,
                level=level  # level is now handled by kwargs in generate_id
            )

            # Update heading number
            self._state.increment(level)
            heading_number = self._state.current_heading_number()

            # Find parent heading
            parent_id = self._find_parent_heading(level)

            # Create metadata
            metadata = HeadingMetadata(
                id=heading_id,
                level=level,
                text=text,
                number=heading_number,
                parent_id=parent_id,
                is_topic_title=is_topic_title
            )

            # Store metadata
            self._heading_metadata[heading_id] = metadata

            # Update hierarchy
            if parent_id:
                if parent_id not in self._heading_hierarchy:
                    self._heading_hierarchy[parent_id] = []
                self._heading_hierarchy[parent_id].append(heading_id)

            # Update topic tracking
            if self._current_topic_id:
                self._topic_headings[self._current_topic_id].append(heading_id)

            self.logger.debug(f"Registered heading: {heading_id}")
            return metadata

        except Exception as e:
            self.logger.error(f"Error registering heading: {str(e)}")
            raise

    def _find_parent_heading(self, level: int) -> Optional[str]:
        """Find parent heading ID for a given level."""
        try:
            if level == 1:
                return None

            if not self._current_topic_id:
                return None

            # Get headings for current topic
            topic_headings = self._topic_headings.get(self._current_topic_id, [])

            # Look for most recent heading of higher level
            for heading_id in reversed(topic_headings):
                metadata = self._heading_metadata.get(heading_id)
                if metadata and metadata.level < level:
                    return heading_id

            return None

        except Exception as e:
            self.logger.error(f"Error finding parent heading: {str(e)}")
            return None

    # Hierarchy Management Methods
    def get_heading_chain(self, heading_id: str) -> List[HeadingMetadata]:
        """Get chain of parent headings."""
        try:
            chain = []
            current_id = heading_id

            while current_id:
                if metadata := self._heading_metadata.get(current_id):
                    chain.append(metadata)
                    current_id = metadata.parent_id
                else:
                    break

            return chain

        except Exception as e:
            self.logger.error(f"Error getting heading chain: {str(e)}")
            return []

    def get_heading_children(self, heading_id: str) -> List[HeadingMetadata]:
        """Get immediate child headings."""
        try:
            child_ids = self._heading_hierarchy.get(heading_id, [])
            return [
                self._heading_metadata[child_id]
                for child_id in child_ids
                if child_id in self._heading_metadata
            ]
        except Exception as e:
            self.logger.error(f"Error getting heading children: {str(e)}")
            return []

    # Metadata Access Methods
    def get_heading_metadata(self, heading_id: str) -> Optional[HeadingMetadata]:
        """Get metadata for a heading."""
        return self._heading_metadata.get(heading_id)

    def get_topic_headings(
        self,
        topic_id: str
    ) -> List[HeadingMetadata]:
        """Get all headings for a topic."""
        try:
            heading_ids = self._topic_headings.get(topic_id, [])
            return [
                self._heading_metadata[h_id]
                for h_id in heading_ids
                if h_id in self._heading_metadata
            ]
        except Exception as e:
            self.logger.error(f"Error getting topic headings: {str(e)}")
            return []

    def get_heading_hierarchy(
        self,
        topic_id: str
    ) -> Dict[str, List[HeadingMetadata]]:
        """Get complete heading hierarchy for a topic."""
        try:
            hierarchy = {}
            heading_ids = self._topic_headings.get(topic_id, [])

            for heading_id in heading_ids:
                if children := self.get_heading_children(heading_id):
                    if metadata := self._heading_metadata.get(heading_id):
                        hierarchy[metadata.id] = children

            return hierarchy

        except Exception as e:
            self.logger.error(f"Error getting heading hierarchy: {str(e)}")
            return {}

    # Event Handling Methods
    def _handle_state_change(self, **event_data: Any) -> None:
        """Handle state change events."""
        try:
            if element_id := event_data.get("element_id"):
                if element_id.startswith("topic_"):
                    # New topic detected
                    if event_data.get("state") == ProcessingState.PROCESSING:
                        self.start_new_topic(element_id)

        except Exception as e:
            self.logger.error(f"Error handling state change: {str(e)}")

    # Cleanup Methods
    def cleanup(self) -> None:
        """Clean up handler resources."""
        try:
            self._state = HeadingState()
            self._heading_metadata.clear()
            self._heading_hierarchy.clear()
            self._topic_headings.clear()
            self._current_topic_id = None
            self.logger.debug("Heading handler cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise
