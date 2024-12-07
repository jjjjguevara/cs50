# app/dita/utils/heading.py
from typing import Dict, Set, Tuple, Optional, List
import logging
import re
from dataclasses import dataclass, field


# Global config
from config import DITAConfig

from .types import HeadingState, HeadingContext

class HeadingHandler:
    """Manages heading state and hierarchy across content types."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._state = HeadingState()
        self._saved_states: List[HeadingState] = []
        self.used_ids: Set[str] = set()

    def init_state(self) -> None:
        """Initialize fresh heading state."""
        try:
            self._state = HeadingState()
            self.logger.debug("Initialized fresh heading state")
        except Exception as e:
            self.logger.error(f"Error initializing heading state: {str(e)}")
            raise

    def configure(self, config: DITAConfig) -> None:
        """Configure heading handler."""
        try:
            self.logger.debug("Configuring heading handler")
            # Add any configuration-specific settings here
            self.logger.debug("Heading handler configuration completed")
        except Exception as e:
            self.logger.error(f"Heading handler configuration failed: {str(e)}")
            raise

    def save_state(self) -> None:
        """Save current heading state for later restoration."""
        try:
            self._saved_states.append(HeadingState(
                current_h1=self._state.current_h1,
                counters=self._state.counters.copy(),
                used_ids=self._state.used_ids.copy()
            ))
            self.logger.debug(f"Saved heading state - H1: {self._state.current_h1}")
        except Exception as e:
            self.logger.error(f"Error saving heading state: {str(e)}")
            raise

    def restore_state(self) -> None:
        """Restore previously saved heading state."""
        try:
            if not self._saved_states:
                raise ValueError("No saved state available to restore")

            self._state = self._saved_states.pop()
            self.logger.debug(f"Restored heading state - H1: {self._state.current_h1}")
        except Exception as e:
            self.logger.error(f"Error restoring heading state: {str(e)}")
            raise

    def reset_section(self) -> None:
        """Reset counters for a new section while maintaining H1."""
        try:
            current_h1 = self._state.current_h1
            self._state.counters = {
                'h1': current_h1,
                'h2': 0,
                'h3': 0,
                'h4': 0,
                'h5': 0,
                'h6': 0
            }
            self.logger.debug(f"Reset section counters, maintaining H1: {current_h1}")
        except Exception as e:
            self.logger.error(f"Error resetting section: {str(e)}")
            raise

    def validate_hierarchy(self, level: int, context: HeadingContext) -> bool:
        """
        Validate heading hierarchy at given level.

        Args:
            level: Heading level to validate
            context: Current heading context

        Returns:
            True if hierarchy is valid
        """
        try:
            # Validate level range
            if not 1 <= level <= 6:
                self.logger.error(f"Invalid heading level: {level}")
                return False

            # Topic titles are always valid
            if context.is_topic_title:
                return True

            # Check parent exists if not H1
            if level > 1:
                parent_level = level - 1
                parent_count = self._state.counters[f'h{parent_level}']
                if parent_count == 0:
                    self.logger.error(
                        f"Invalid hierarchy: H{level} without H{parent_level}"
                    )
                    return False

            # Validate no skipped levels
            for l in range(1, level):
                if self._state.counters[f'h{l}'] == 0:
                    self.logger.error(f"Invalid hierarchy: Skipped H{l}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating hierarchy: {str(e)}")
            return False

    def process_heading(self,
                       text: str,
                       level: int,
                       context: Optional[HeadingContext] = None,
                       is_map_title: bool = False) -> Tuple[str, str]:
        """
        Process heading with improved state management.

        Args:
            text: Heading text
            level: Heading level (1-6)
            context: Optional heading context
            is_map_title: Whether this is a map title

        Returns:
            Tuple of (heading_id, formatted_text)
        """
        try:
            # Use provided context or create default
            ctx = context or HeadingContext(level=level)

            # Validate hierarchy unless map title
            if not is_map_title and not self.validate_hierarchy(level, ctx):
                self.logger.warning(f"Invalid heading hierarchy at level {level}")

            # Generate ID
            heading_id = self._generate_id(text)

            # Update counters
            if not is_map_title:
                self._update_counters(level)

            # Format text with number
            formatted_text = self._format_heading(text, level, is_map_title)

            return heading_id, formatted_text

        except Exception as e:
            self.logger.error(f"Error processing heading: {str(e)}")
            raise

    def _update_counters(self, level: int) -> None:
        """Update heading counters for given level."""
        try:
            # Update H1 counter specially
            if level == 1:
                self._state.current_h1 += 1
                self._state.counters['h1'] = self._state.current_h1
            else:
                self._state.counters[f'h{level}'] += 1

            # Reset lower levels
            for l in range(level + 1, 7):
                self._state.counters[f'h{l}'] = 0

        except Exception as e:
            self.logger.error(f"Error updating counters: {str(e)}")
            raise

    def _generate_id(self, text: str) -> str:
        """Generate unique ID for heading."""
        try:
            # Clean text for ID
            base_id = re.sub(r'[^\w\- ]', '', text.lower())
            base_id = re.sub(r'[-\s]+', '-', base_id).strip('-')

            # Ensure uniqueness
            heading_id = base_id
            counter = 1
            while heading_id in self._state.used_ids:
                heading_id = f"{base_id}-{counter}"
                counter += 1

            self._state.used_ids.add(heading_id)
            return heading_id

        except Exception as e:
            self.logger.error(f"Error generating heading ID: {str(e)}")
            raise

    def _format_heading(self, text: str, level: int, is_map_title: bool) -> str:
        """Format heading text with number."""
        try:
            if is_map_title:
                return text

            # Get number components
            numbers = []
            for l in range(1, level + 1):
                count = self._state.counters[f'h{l}']
                if count > 0:
                    numbers.append(str(count))

            # Format with numbers
            if numbers:
                return f"{'.'.join(numbers)}. {text}"
            return text

        except Exception as e:
            self.logger.error(f"Error formatting heading: {str(e)}")
            raise

    def reset(self) -> None:
        """Reset all heading state."""
        self.init_state()
        self._saved_states.clear()

    def cleanup(self) -> None:
            """Clean up heading handler resources and state."""
            try:
                self.logger.debug("Starting heading handler cleanup")

                # Reset counters and state
                self._state.counters = {
                    'h1': 0,
                    'h2': 0,
                    'h3': 0,
                    'h4': 0,
                    'h5': 0,
                    'h6': 0
                }
                self._state.current_h1 = 0
                self._state.used_ids.clear()

                self.logger.debug("Heading handler cleanup completed")

            except Exception as e:
                self.logger.error(f"Heading handler cleanup failed: {str(e)}")
                raise
