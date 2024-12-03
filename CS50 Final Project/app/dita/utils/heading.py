# app/dita/utils/heading.py
from typing import Dict, Set, Optional, Tuple
import re
import logging

class HeadingHandler:
    def __init__(self):
            self.logger = logging.getLogger(__name__)
            self.used_heading_ids: Dict[str, int] = {}
            self.heading_map: Dict[str, str] = {}
            self.registered_ids: Set[str] = set()
            self._current_h1 = 0
            # Initialize all counters to 0
            self.counters: Dict[str, int] = {
                'h1': 0, 'h2': 0, 'h3': 0,
                'h4': 0, 'h5': 0, 'h6': 0
            }

    def set_current_h1(self, number: int) -> None:
            """Set the current H1 number explicitly"""
            self.counters['h1'] = number

    def process_map_title(self, title_text: str) -> Tuple[str, str]:
        """
        Process map title without affecting heading numbers.

        Args:
            title_text: The text of the map title

        Returns:
            Tuple[str, str]: (heading_id, title_text)
        """
        try:
            # Generate ID without affecting counters
            heading_id = self._sanitize_id(title_text)

            # Register ID to prevent duplicates
            if heading_id in self.used_heading_ids:
                self.used_heading_ids[heading_id] += 1
                heading_id = f"{heading_id}-{self.used_heading_ids[heading_id]}"
            else:
                self.used_heading_ids[heading_id] = 1

            self.logger.debug(
                f"Processed map title: "
                f"ID: {heading_id}, "
                f"Text: {title_text}"
            )

            return heading_id, title_text

        except Exception as e:
            self.logger.error(f"Error processing map title: {str(e)}")
            return self._sanitize_id(title_text), title_text

    def process_heading(self, heading_text: str, level: int) -> Tuple[str, str]:
        """Process a heading and return both its ID and formatted text"""
        try:
            # Generate section number
            section_number = self._get_section_number(level)

            # Generate ID
            heading_id = self._generate_heading_id(heading_text)

            # Format text with section number
            formatted_text = f"{section_number} {heading_text}"

            self.logger.debug(
                f"Processed heading: Level {level}, "
                f"Number: {section_number}, "
                f"ID: {heading_id}, "
                f"Text: {formatted_text}"
            )

            return heading_id, formatted_text

        except Exception as e:
            self.logger.error(f"Error processing heading: {str(e)}")
            return self._sanitize_id(heading_text), heading_text

    def _get_section_number(self, level: int) -> str:
        """Generate hierarchical section number (1.1.2.3 format)"""
        try:
            if level < 1 or level > 6:
                raise ValueError(f"Invalid heading level: {level}")

            # For H1s (main sections)
            if level == 1:
                return str(self._current_h1)

            # For subheadings (H2-H6)
            if self._current_h1 == 0:
                self.start_new_section()

            # Reset counters for levels deeper than current level
            for i in range(level + 1, 7):
                self.counters[f'h{i}'] = 0

            # Build hierarchical number
            numbers = [str(self._current_h1)]  # Start with current H1

            # Only increment the current level
            if level == 2:
                self.counters['h2'] += 1
                numbers.append(str(self.counters['h2']))
            else:
                # For H3+, include parent H2's counter
                if self.counters['h2'] == 0:
                    self.counters['h2'] = 1
                numbers.append(str(self.counters['h2']))

                # Then increment current level and add its counter
                self.counters[f'h{level}'] += 1
                for i in range(3, level + 1):
                    numbers.append(str(self.counters[f'h{i}']))

            return '.'.join(numbers)

        except Exception as e:
            self.logger.error(f"Error in _get_section_number: {str(e)}")
            return str(level)

    def _generate_heading_id(self, text: str) -> str:
        """Generate unique ID for heading"""
        base_id = self._sanitize_id(text)

        if base_id in self.used_heading_ids:
            self.used_heading_ids[base_id] += 1
            return f"{base_id}-{self.used_heading_ids[base_id]}"

        self.used_heading_ids[base_id] = 1
        return base_id

    def _sanitize_id(self, text: str) -> str:
        """Create URL-friendly ID from text"""
        # Convert to lowercase and replace spaces/special chars with hyphens
        sanitized = re.sub(r'[^\w\s-]', '', text.lower())
        sanitized = re.sub(r'[-\s]+', '-', sanitized).strip('-')

        # Take first 4 words maximum
        words = sanitized.split('-')[:4]
        return '-'.join(words)

    def reset(self) -> None:
        """Reset all counters and mappings"""
        self.used_heading_ids: Dict[str, int] = {}
        self.heading_map: Dict[str, str] = {}
        self.registered_ids: Set[str] = set()
        self._current_h1 = 0
        self.counters = {
            'h1': 0, 'h2': 0, 'h3': 0,
            'h4': 0, 'h5': 0, 'h6': 0
        }
        self.logger.debug("Reset heading handler state")

    def reset_sub_headings(self) -> None:
        """Reset only sub-heading counters (h2-h6) while preserving h1"""
        current_h1 = self._current_h1  # Preserve current section
        for level in range(2, 7):
            self.counters[f'h{level}'] = 0
        self.logger.debug(f"Reset sub-headings while preserving H1: {current_h1}")

    def set_h1_number(self, number: int) -> None:
        """Set H1 number explicitly and reset sub-headings"""
        self.counters['h1'] = number
        self.reset_sub_headings()
        self.logger.debug(f"Set H1 to {number} and reset sub-headings")

    def start_new_section(self) -> None:
        """Called when starting a new topic section"""
        self._current_h1 += 1
        # Reset ALL subheading counters when starting new section
        for i in range(2, 7):
            self.counters[f'h{i}'] = 0
        self.logger.debug(f"Started new section {self._current_h1}, reset all subheading counters")

    def register_existing_id(self, heading_text: str, id_value: str) -> None:
        """Register an existing ID to prevent duplicates"""
        self.heading_map[heading_text] = id_value
        self.registered_ids.add(id_value)
        self.logger.debug(f"Registered existing ID: {id_value} for heading: {heading_text}")
