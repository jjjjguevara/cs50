from typing import Dict, Set, List, Optional, Tuple
import logging
import re
from dataclasses import dataclass, field
from .types import HeadingState, HeadingContext, HeadingReference, ProcessingError


class HeadingHandler:
    """Manages heading state and hierarchy across content types."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._state = HeadingState()
        self.index_numbers_enabled = True
        self._saved_states: List[HeadingState] = []
        self.used_ids: Set[str] = set()

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

    def validate_hierarchy(self, level: int, context: HeadingContext) -> bool:
        """
        Validate the hierarchy of a heading.

        Args:
            level: The level of the heading (1-6).
            context: The current heading context.

        Returns:
            True if the hierarchy is valid; otherwise, False.
        """
        if not 1 <= level <= 6:
            self.logger.error(f"Invalid heading level: {level}")
            return False
        if context.is_topic_title:
            return True
        if level > 1 and self._state.counters[f'h{level - 1}'] == 0:
            self.logger.error(f"H{level} is invalid without a preceding H{level - 1}.")
            return False
        return True

    def process_heading(self, text: str, level: int, is_topic_title: bool = False) -> HeadingReference:
        """
        Process a heading, assigning it a unique ID and optionally numbering it.

        Args:
            text: The heading text.
            level: The heading level (1-6).
            is_topic_title: Whether this heading is the topic title.

        Returns:
            A HeadingReference object with ID, text, and level.
        """
        heading_id = self._generate_id(text)

        if not is_topic_title and self.index_numbers_enabled:
            self._state.increment(level)
            number = self._state.current_heading_number()
            text = f"{number} {text}"

        self.logger.debug(f"Processed heading: {text} (ID: {heading_id}) at level {level}")
        return HeadingReference(id=heading_id, text=text, level=level)

    def _generate_id(self, text: str) -> str:
        """Generate a unique ID for a heading."""
        base_id = re.sub(r'[^\w\- ]', '', text.lower())
        base_id = re.sub(r'[-\s]+', '-', base_id).strip('-')
        unique_id = base_id
        counter = 1
        while unique_id in self._state.used_ids:
            unique_id = f"{base_id}-{counter}"
            counter += 1
        self._state.used_ids.add(unique_id)
        self.logger.debug(f"Generated unique ID: {unique_id}")
        return unique_id

    def cleanup(self) -> None:
        """Clean up the handler by resetting the state."""
        self.init_state()
        self._saved_states.clear()
        self.logger.debug("Cleaned up HeadingHandler state.")
