from typing import Dict, Set, List, Optional, Tuple
import logging
import re
from dataclasses import dataclass, field
from app.dita.models.types import HeadingState, HeadingContext, HeadingReference, ProcessingError


class HeadingHandler:
    """Manages heading state and hierarchy across content types."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._state = HeadingState()
        self.index_numbers_enabled = True
        self._saved_states: List[HeadingState] = []
        self.used_ids: Set[str] = set()
        self.heading_references: List[HeadingReference] = []
        self.current_topic_number = 0
        self.first_heading_in_topic = True

    def init_state(self) -> None:
        """Initialize fresh heading state."""
        self._state = HeadingState()
        self.logger.debug("Initialized fresh heading state")

    def save_state(self) -> None:
        """Save the current heading state for restoration later."""
        self._saved_states.append(HeadingState(
            current_h1=self._state.current_h1,
            counters=self._state.counters.copy(),
            used_ids=self._state.used_ids.copy()
        ))
        self.logger.debug("Saved heading state.")

    def restore_state(self) -> None:
        """Restore the last saved heading state."""
        if not self._saved_states:
            self.logger.warning("No saved states available for restoration.")
            return
        self._state = self._saved_states.pop()
        self.logger.debug("Restored a saved heading state.")

    def reset_section(self) -> None:
        """Reset counters for a new section while maintaining the current H1."""
        current_h1 = self._state.current_h1
        self._state = HeadingState(current_h1=current_h1)
        self.logger.debug(f"Reset heading counters while maintaining H1={current_h1}")

    def start_new_topic(self) -> None:
            """Start a new topic, incrementing the main counter."""
            self.current_topic_number += 1
            self.first_heading_in_topic = True
            self._state.counters = {
                'h1': 0, 'h2': 0, 'h3': 0, 'h4': 0, 'h5': 0, 'h6': 0
            }
            self.logger.debug(f"Starting new topic {self.current_topic_number}")

    def validate_hierarchy(self, level: int, context: HeadingContext) -> bool:
        """
        Validate and adjust the heading hierarchy.

        Args:
            level: The level of the heading (1-6).
            context: The current heading context.

        Returns:
            True if the hierarchy is valid or adjusted; otherwise, False.
        """
        if not 1 <= level <= 6:
            self.logger.error(f"Invalid heading level: {level}")
            return False

        if context.is_topic_title:
            self.start_new_topic()
            return True

        # If we get an Hn without previous levels, adjust the hierarchy
        if level > 1:
            # If this is the first heading in a topic and it's not h1
            if self._state.counters['h1'] == 0:
                self.start_new_topic()

            # Ensure all previous levels are set
            for prev_level in range(1, level):
                if self._state.counters[f'h{prev_level}'] == 0:
                    self._state.counters[f'h{prev_level}'] = 1
                    self.logger.debug(f"Automatically set h{prev_level} counter to 1")

        return True

    def set_numbering_enabled(self, enabled: bool):
            """
            Enable or disable numbering for headings.

            Args:
                enabled (bool): Whether numbering should be enabled.
            """
            self.numbering_enabled = enabled
            self.used_ids.clear()
            logging.getLogger(__name__).debug(f"Heading numbering set to {'enabled' if enabled else 'disabled'}.")

    def add_heading_reference(self, heading_ref: HeadingReference) -> None:
            """
            Add a heading reference to the internal list.

            Args:
                heading_ref: A HeadingReference object to track.
            """
            self.heading_references.append(heading_ref)

    def get_heading_references(self) -> List[HeadingReference]:
            """
            Get all tracked heading references.

            Returns:
                A list of HeadingReference objects.
            """
            return self.heading_references

    def __iter__(self):
        """
        Make HeadingHandler iterable over its heading references.

        Yields:
            HeadingReference: Each heading reference in the internal list.
        """
        yield from self.heading_references

    def process_heading(self, text: str, level: int, is_topic_title: bool = False) -> Tuple[str, str]:
        """
        Process a heading with proper numbering.

        Args:
            text: The heading text.
            level: The heading level (1-6).
            is_topic_title: Whether this is the main topic title.

        Returns:
            Tuple[str, str]: (heading_id, numbered_heading)
        """
        if not self.index_numbers_enabled:
            heading_id = self.generate_id(text)
            return heading_id, text

        # For topic titles, use the topic number directly
        if is_topic_title:
            self._state.counters['h1'] = self.current_topic_number
            self.first_heading_in_topic = False
            base_number = str(self.current_topic_number)
        else:
            # For subsequent headings, maintain hierarchy within the topic
            self._state.counters[f'h{level}'] += 1
            # Reset lower level counters
            for l in range(level + 1, 7):
                self._state.counters[f'h{l}'] = 0

            # Build the number based on hierarchy
            number_parts = []
            for l in range(1, level + 1):
                if self._state.counters[f'h{l}'] > 0:
                    number_parts.append(str(self._state.counters[f'h{l}']))
            base_number = '.'.join(number_parts)

        heading_id = self.generate_id(f"{base_number}-{text}")
        numbered_heading = f"{base_number}. {text}"

        self.add_heading_reference(HeadingReference(
            id=heading_id,
            text=numbered_heading,
            level=level
        ))

        return heading_id, numbered_heading

    def generate_id(self, text: str) -> str:
        """
        Generate a unique ID for a heading.

        Args:
            text: The heading text.

        Returns:
            str: The generated ID.
        """
        base_id = re.sub(r"[^\w\- ]", "", text.lower())
        base_id = re.sub(r"[-\s]+", "-", base_id).strip("-")
        return base_id

    def cleanup(self) -> None:
        """Clean up the handler by resetting the state."""
        self.init_state()
        self._saved_states.clear()
        self.logger.debug("Cleaned up HeadingHandler state.")
