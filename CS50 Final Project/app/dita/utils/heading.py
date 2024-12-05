# app/dita/utils/heading.py
from typing import Dict, Set, Optional, Tuple
from dataclasses import dataclass
import re
import logging
from .types import (
    ElementType,
    ProcessingState,
    ParsedElement,
    TrackedElement,
    ProcessedContent,
)

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

    def process_heading(self, heading_text: str, level: int, is_map_title: bool = False) -> Tuple[str, str]:
        """Process a heading and return both its ID and formatted text"""
        try:
            self.logger.debug(
                f"Processing heading: text='{heading_text}', "
                f"level={level}, is_map_title={is_map_title}, "
                f"current_state={self.counters}"
            )

            # Generate ID
            heading_id = self._generate_heading_id(heading_text)

            # Skip section numbering only for map titles
            if is_map_title:
                formatted_text = heading_text
                self.logger.debug("Processed as map title")
            else:
                # Get section number
                section_number = self._get_section_number(level)
                formatted_text = f"{section_number} {heading_text}"
                self.logger.debug(f"Generated section number: {section_number}")

            self.logger.debug(
                f"Final result: id='{heading_id}', "
                f"text='{formatted_text}', "
                f"new_state={self.counters}"
            )

            return heading_id, formatted_text

        except Exception as e:
            self.logger.error(f"Error processing heading: {str(e)}")
            return self._sanitize_id(heading_text), heading_text

    ### Critical for Heading numbering logic!!!
    def _get_section_number(self, level: int) -> str:
            """Generate hierarchical section number (1. 1.1. 1.1.1. format)"""
            try:
                if level < 1 or level > 6:
                    raise ValueError(f"Invalid heading level: {level}")

                # For H1s
                if level == 1:
                    self._current_h1 += 1  # Increment the current H1 number
                    self.counters['h1'] = self._current_h1  # Sync with counters
                    return f"{self._current_h1}."

                # For lower levels (H2-H6)
                if self._current_h1 == 0:
                    self._current_h1 = 1
                    self.counters['h1'] = 1

                # For this specific level, increment its counter
                self.counters[f'h{level}'] += 1

                # Build number using current H1 and this level's counter
                numbers = [str(self._current_h1)]
                for i in range(2, level + 1):
                    numbers.append(str(self.counters[f'h{i}']))

                self.logger.debug(
                    f"Numbering for level {level}:"
                    f" current_h1={self._current_h1},"
                    f" counters={self.counters},"
                    f" result={'.'.join(numbers)}"
                )

                return f"{'.'.join(numbers)}."

            except Exception as e:
                self.logger.error(f"Error in _get_section_number: {str(e)}")
                return f"{level}."

    def _generate_heading_id(self, text: str) -> str:
        """Generate unique ID for heading"""
        base_id = self._sanitize_id(text)

        if base_id in self.used_heading_ids:
            self.used_heading_ids[base_id] += 1
            return f"{base_id}-{self.used_heading_ids[base_id]}"

        self.used_heading_ids[base_id] = 1
        return base_id

    def process_content_headings(self, content: ProcessedContent) -> None:
            """
            Process headings in content.

            Args:
                content: Processed content with heading information
            """
            try:
                if content.heading_level is not None:
                    level = f'h{content.heading_level}'
                    if level in self.counters:
                        self.counters[level] += 1
                        self.logger.debug(
                            f"Processed heading level {content.heading_level}. "
                            f"New counter: {self.counters[level]}"
                        )
            except Exception as e:
                self.logger.error(f"Error processing content headings: {str(e)}")

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

    def start_new_section(self):
            """Start a new section, resetting appropriate counters."""
            self.counters['h1'] += 1
            for i in range(2, 7):
                self.counters[f'h{i}'] = 0
            self.logger.debug(f"Started new section. Counters: {self.counters}")

    def register_existing_id(self, heading_text: str, id_value: str) -> None:
        """Register an existing ID to prevent duplicates"""
        self.heading_map[heading_text] = id_value
        self.registered_ids.add(id_value)
        self.logger.debug(f"Registered existing ID: {id_value} for heading: {heading_text}")
