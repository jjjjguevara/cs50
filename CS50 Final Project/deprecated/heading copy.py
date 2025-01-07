from typing import Dict, Set, List, Optional, Tuple, Any
import logging
import re
from dataclasses import dataclass, field
from app.dita.models.types import (
    HeadingState,
    ProcessingError,
    ProcessingPhase,
    ProcessingState,
    ContentElement,
    ElementType
)
from app.dita.event_manager import EventManager, EventType
from app.dita.utils.id_handler import DITAIDHandler


class HeadingHandler:
    """Manages heading state and hierarchy across content types."""

    def __init__(
            self,
            event_manager: EventManager,
        ):
            """
            Initialize HeadingHandler with event management.

            Args:
                event_manager: Event management system
                index_numbers_enabled: Whether to enable heading numbering
            """
            self.logger = logging.getLogger(__name__)
            self.event_manager = event_manager
            self._state = HeadingState()
            self.id_handler = DITAIDHandler()

            # Track heading hierarchy
            self._heading_elements: Dict[str, ContentElement] = {}
            self._heading_hierarchy: Dict[str, str] = {}  # child_id -> parent_id
            self._current_level = 0

            # Initialize missing attributes
            self._saved_states: List[HeadingState] = []  # To store saved heading states
            self.used_ids: Set[str] = set()  # To track unique heading IDs
            self.current_topic_number = 0
            self.first_heading_in_topic = True

            # Register for configuration events
            self.event_manager.subscribe(
                EventType.CONFIG_UPDATE,  # New event type needed
                self._handle_config_update
            )

    def _handle_config_update(self, config: Dict[str, Any]) -> None:
            """Handle configuration updates from event system."""
            try:
                # Reset state with new configuration
                self._state = HeadingState(
                    current_h1=self._state.current_h1,  # Preserve current count
                    counters=self._state.counters.copy()  # Preserve current counters
                )

                # Apply configuration based on hierarchy
                if heading_config := config.get('heading', {}):
                    self._apply_heading_config(heading_config)

            except Exception as e:
                self.logger.error(f"Error handling config update: {str(e)}")

    def _apply_heading_config(self, config: Dict[str, Any]) -> None:
        """Apply heading configuration respecting inheritance rules."""
        # Numbers and format configuration
        if 'numbering' in config:
            numbering = config['numbering']
            self._state.numbering_enabled = numbering.get('enabled', True)
            self._state.number_format = numbering.get('format', 'numeric')

        # Heading level configuration
        if 'levels' in config:
            for level, level_config in config['levels'].items():
                self._state.level_config[int(level)] = level_config

        # Audience-specific configuration
        if audience_config := config.get('audience_overrides', {}):
            if current_audience := config.get('current_audience'):
                if audience_rules := audience_config.get(current_audience):
                    self._apply_audience_rules(audience_rules)

        # Distribution channel configuration
        if channel_config := config.get('channel_overrides', {}):
            if current_channel := config.get('current_channel'):
                if channel_rules := channel_config.get(current_channel):
                    self._apply_channel_rules(channel_rules)

    def process_heading(
        self, text: str, level: int, is_topic_title: bool = False
    ) -> Tuple[str, str, Optional[str]]:
        """
        Process a heading and return its ID, formatted text, and hierarchy-based numbering.

        Args:
            text: The text of the heading.
            level: The level of the heading (1-6).
            is_topic_title: Whether this is the title of the topic.

        Returns:
            Tuple containing:
            - heading_id: The unique ID for the heading.
            - heading_text: The raw text of the heading.
            - heading_number: The sequential hierarchy number (e.g., 1., 1.1.) or None if not required.
        """
        try:
            # Generate a unique ID for the heading
            heading_id = self.generate_heading_id(text, level)

            # Update the heading hierarchy
            self.update_hierarchy(level)

            # Retrieve the current numbering for this heading
            heading_number = self.get_current_number(level)

            return heading_id, text, heading_number

        except Exception as e:
            self.logger.error(f"Error processing heading '{text}': {str(e)}")
            raise


     # State Management
    def init_state(self) -> None:
        """Initialize a fresh heading state."""
        self._state = HeadingState()
        self.logger.debug("Initialized fresh heading state.")

    def reset_section(self) -> None:
        """Reset counters for a new section while maintaining the current H1."""
        current_h1 = self._state.current_h1
        self._state = HeadingState(current_h1=current_h1)
        self.logger.debug(f"Reset heading counters while maintaining H1={current_h1}.")

    def validate_hierarchy(self, level: int) -> bool:
        """
        Validate and adjust the heading hierarchy.

        Args:
            level: The level of the heading (1-6)

        Returns:
            bool: True if the hierarchy is valid or adjusted
        """
        try:
            if not 1 <= level <= 6:
                self.logger.error(f"Invalid heading level: {level}")
                return False

            # Adjust hierarchy
            self.update_hierarchy(level)
            return True

        except Exception as e:
            self.logger.error(f"Error validating hierarchy: {str(e)}")
            return False

    # Metadata Extraction
    def extract_heading_metadata(self) -> Dict[str, Any]:
        """
        Extract metadata for headings from tracked elements.

        Returns:
            Dict containing heading metadata.
        """
        try:
            heading_metadata = {}

            for heading_id, element in self._heading_elements.items():
                heading_metadata[heading_id] = {
                    "text": element.content,
                    "level": element.metadata.get("heading_level"),
                    "number": element.metadata.get("heading_number"),
                    "is_title": element.metadata.get("is_topic_title", False),
                    "parent_id": self._heading_hierarchy.get(heading_id)
                }

            return heading_metadata

        except Exception as e:
            self.logger.error(f"Error extracting heading metadata: {str(e)}")
            return {}

    # ID Generation
    def generate_heading_id(self, text: str, level: int) -> str:
        """
        Generate a unique ID for a heading.

        Args:
            text: The heading text
            level: The heading level (1-6)

        Returns:
            str: A unique heading ID
        """
        try:
            # Generate base ID
            base_id = re.sub(r"[^\w\-]+", "-", text.lower()).strip("-")
            heading_id = f"{base_id}-h{level}"

            # Ensure uniqueness
            if heading_id in self.used_ids:
                counter = 1
                while f"{heading_id}-{counter}" in self.used_ids:
                    counter += 1
                heading_id = f"{heading_id}-{counter}"

            self.used_ids.add(heading_id)
            return heading_id

        except Exception as e:
            self.logger.error(f"Error generating heading ID: {str(e)}")
            return self.id_handler.generate_id(f"heading-{level}")

    def track_heading(
        self,
        element: ContentElement,
        level: int,
        is_topic_title: bool = False
    ) -> None:
        """
        Track a heading element in the hierarchy.

        Args:
            element: The heading ContentElement
            level: Heading level (1-6)
            is_topic_title: Whether this is a topic title
        """
        try:
            # Update state
            if is_topic_title:
                self.start_new_topic()
            else:
                self.update_hierarchy(level)

            # Store heading
            self._heading_elements[element.id] = element

            # Generate heading ID and number
            heading_id = self.generate_heading_id(element.content, level)
            heading_number = self.get_current_number(level)

            # Update heading metadata
            element.metadata.update({
                "heading_id": heading_id,
                "heading_level": level,
                "heading_number": heading_number,
                "is_topic_title": is_topic_title
            })

            # Track hierarchy
            if level > 1 and self._heading_elements:
                # Find parent heading
                for h_id, h_elem in reversed(list(self._heading_elements.items())):
                    h_level = h_elem.metadata.get("heading_level", 1)
                    if h_level < level:
                        self._heading_hierarchy[element.id] = h_id
                        break

            # Emit event for state change
            self.event_manager.emit(
                EventType.STATE_CHANGE,
                element_id=element.id,
                old_state=element.state,
                new_state=ProcessingState.PROCESSING
            )

        except Exception as e:
            self.logger.error(f"Error tracking heading: {str(e)}")
            raise

    def get_parent_heading(self, heading_id: str) -> Optional[ContentElement]:
        """Get parent heading for a given heading ID."""
        parent_id = self._heading_hierarchy.get(heading_id)
        return self._heading_elements.get(parent_id) if parent_id else None

    def get_heading_chain(self, heading_id: str) -> List[ContentElement]:
        """Get the full chain of parent headings."""
        chain = []
        current_id = heading_id
        while current_id and current_id in self._heading_hierarchy:
            parent_id = self._heading_hierarchy[current_id]
            if parent := self._heading_elements.get(parent_id):
                chain.append(parent)
                current_id = parent_id
            else:
                break
        return chain

    def update_hierarchy(self, level: int) -> None:
        """Update heading counters for a new heading level."""
        try:
            # Reset deeper levels
            for i in range(level + 1, 7):
                self._state.counters[f"h{i}"] = 0

            # Increment current level
            self._state.counters[f"h{level}"] += 1

            # Update H1 tracking
            if level == 1:
                self._state.current_h1 = self._state.counters["h1"]

            self._current_level = level

        except Exception as e:
            self.logger.error(f"Error updating hierarchy: {str(e)}")
            raise

    def get_current_number(self, level: int) -> Optional[str]:
        """Get the current heading number for a level."""
        try:
            if not 1 <= level <= 6:
                return None

            numbers = []
            for i in range(1, level + 1):
                if count := self._state.counters.get(f"h{i}", 0):
                    numbers.append(str(count))

            return ".".join(numbers) + "." if numbers else None

        except Exception as e:
            self.logger.error(f"Error getting heading number: {str(e)}")
            return None

    def start_new_topic(self) -> None:
        """Reset state for a new topic."""
        self._state = HeadingState()
        self._state.counters["h1"] = 1
        self._state.current_h1 = 1
        self._current_level = 0
        self.logger.debug("Started new topic heading state")

    def save_state(self) -> None:
        """Save current state for later restoration."""
        self._saved_states.append(HeadingState(
            current_h1=self._state.current_h1,
            counters=self._state.counters.copy()
        ))

    def restore_state(self) -> None:
        """Restore the last saved state."""
        if self._saved_states:
            self._state = self._saved_states.pop()
            self.logger.debug("Restored previous heading state")

    def cleanup(self) -> None:
        """Clean up handler resources and state."""
        try:
            self._state = HeadingState()
            self._heading_elements.clear()
            self._heading_hierarchy.clear()
            self._saved_states.clear()
            self._current_level = 0
            self.logger.debug("Heading handler cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise
