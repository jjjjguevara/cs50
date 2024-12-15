from typing import Dict, Set, List, Optional, Tuple, Any
import logging
import re
from dataclasses import dataclass, field
from app.dita.models.types import HeadingState, ProcessingError, ProcessingMetadata


class HeadingHandler:
    """Manages heading state and hierarchy across content types."""

    def __init__(self, processing_metadata: ProcessingMetadata):
            """
            Initialize the HeadingHandler with ProcessingMetadata.

            Args:
                processing_metadata: The shared metadata object for rendering and transformation.
            """
            self.logger = logging.getLogger(__name__)
            self._state = HeadingState()
            self.processing_metadata = processing_metadata
            self.index_numbers_enabled = processing_metadata.features.get("number_headings", True)

            # Initialize missing attributes
            self._saved_states: List[HeadingState] = []  # To store saved heading states
            self.used_ids: Set[str] = set()  # To track unique heading IDs
            self.current_topic_number = 0
            self.first_heading_in_topic = True

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
            self.update_hierarchy(level, is_topic_title)

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

    def save_state(self) -> None:
        """Save the current heading state for restoration later."""
        self._saved_states.append(HeadingState(
            current_h1=self._state.current_h1,
            counters=self._state.counters.copy()
        ))
        self.logger.debug("Saved heading state.")

    def restore_state(self) -> None:
        """Restore the last saved heading state."""
        if not self._saved_states:
            self.logger.warning("No saved states available for restoration.")
            return
        self._state = self._saved_states.pop()
        self.logger.debug("Restored a saved heading state.")


    # Numbering and Hierarchy

    def update_hierarchy(self, level: int, is_topic_title: bool = False) -> None:
        """
        Update the heading hierarchy for numbering.

        Args:
            level: The level of the heading (1-6).
            is_topic_title: Whether this heading is the topic title.
        """
        try:
            # If this is a topic title, reset the hierarchy
            if is_topic_title:
                self._state.counters = {f"h{i}": 0 for i in range(1, 7)}
                self._state.counters["h1"] = 1  # Topic titles start with 1
                self._state.current_h1 = 1
                return

            # Adjust counters for the heading level
            for i in range(level, 7):
                self._state.counters[f"h{i}"] = 0  # Reset deeper levels
            self._state.counters[f"h{level}"] += 1  # Increment current level

            # Update current H1 if this is a top-level heading
            if level == 1:
                self._state.current_h1 = self._state.counters["h1"]

        except Exception as e:
            self.logger.error(f"Error updating heading hierarchy for level {level}: {str(e)}")
            raise

    def get_current_number(self, level: int) -> Optional[str]:
        """
        Get the hierarchy-based numbering for the specified heading level.

        Args:
            level: The level of the heading (1-6).

        Returns:
            A string representing the heading number (e.g., "1.1.2.") or None if invalid.
        """
        try:
            if not 1 <= level <= 6:
                self.logger.warning(f"Invalid heading level: {level}")
                return None

            # Build the numbering string up to the specified level
            number_parts = [
                str(self._state.counters[f"h{i}"])
                for i in range(1, level + 1)
                if self._state.counters[f"h{i}"] > 0
            ]

            return ".".join(number_parts) + "." if number_parts else None

        except Exception as e:
            self.logger.error(f"Error retrieving heading number for level {level}: {str(e)}")
            raise

    def reset_section(self) -> None:
        """Reset counters for a new section while maintaining the current H1."""
        current_h1 = self._state.current_h1
        self._state = HeadingState(current_h1=current_h1)
        self.logger.debug(f"Reset heading counters while maintaining H1={current_h1}.")

    def start_new_topic(self) -> None:
        """Start a new topic, resetting the hierarchy and counters."""
        self.init_state()  # Reset the state
        self._state.counters["h1"] = 1  # Start numbering at 1 for the new topic
        self._state.current_h1 = 1
        self.logger.debug(f"Started a new topic with H1={self._state.current_h1}.")

    def validate_hierarchy(self, level: int, is_topic_title: bool = False) -> bool:
        """
        Validate and adjust the heading hierarchy.

        Args:
            level: The level of the heading (1-6).
            is_topic_title: Whether this heading is the topic title.

        Returns:
            True if the hierarchy is valid or adjusted; otherwise, False.
        """
        try:
            if not 1 <= level <= 6:
                self.logger.error(f"Invalid heading level: {level}")
                return False

            if is_topic_title:
                self.start_new_topic()
                return True

            # Adjust hierarchy using `update_hierarchy`
            self.update_hierarchy(level, is_topic_title=False)
            return True

        except Exception as e:
            self.logger.error(f"Error validating hierarchy for level {level}: {str(e)}")
            return False



    # Metadata Extraction
    def extract_heading_metadata(self) -> Dict[str, Any]:
        """
        Extract metadata for headings from ProcessingMetadata.

        Returns:
            Dict containing heading metadata.
        """
        try:
            return self.processing_metadata.references.get("headings", {})
        except Exception as e:
            self.logger.error(f"Error extracting heading metadata: {str(e)}")
            return {}

    # ID Generation
    def generate_heading_id(self, text: str, level: int) -> str:
        """
        Generate a unique ID for a heading.

        Args:
            text: The heading text.
            level: The heading level (1-6).

        Returns:
            str: A unique heading ID.
        """
        base_id = re.sub(r"[^\w\-]+", "-", text.lower()).strip("-")
        heading_id = f"{base_id}-h{level}"
        self.processing_metadata.add_heading(heading_id, text, level)
        self.logger.debug(f"Generated heading ID: {heading_id}")
        return heading_id


    def set_numbering_enabled(self, enabled: bool):
            """
            Enable or disable numbering for headings.

            Args:
                enabled (bool): Whether numbering should be enabled.
            """
            self.numbering_enabled = enabled
            self.used_ids.clear()
            logging.getLogger(__name__).debug(f"Heading numbering set to {'enabled' if enabled else 'disabled'}.")



    def cleanup(self) -> None:
        """Clean up the handler by resetting the state."""
        self.init_state()
        self._saved_states.clear()
        self.logger.debug("Cleaned up HeadingHandler state.")
