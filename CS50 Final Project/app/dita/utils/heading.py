# app/dita/utils/heading.py
from typing import Dict, Set, Optional, Tuple
import re
import logging

class HeadingHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.used_heading_ids: Dict[str, int] = {}
        self.heading_map: Dict[str, str] = {}  # Map original text to generated IDs
        self.registered_ids: Set[str] = set()  # Track explicitly registered IDs

    def process_heading(self, heading_text: str, level: int,
                       counters: Dict[str, int]) -> Tuple[str, str]:
        """
        Process a heading and return both its ID and formatted text.
        Returns: (heading_id, formatted_text)
        """
        try:
            section_number = self.get_section_number(level, counters)
            heading_id = self.generate_heading_id(heading_text, section_number)
            formatted_text = self.format_heading_text(level, heading_text, counters)

            self.logger.debug(
                f"Processed heading: Level {level}, "
                f"ID: {heading_id}, "
                f"Text: {formatted_text}"
            )

            return heading_id, formatted_text

        except Exception as e:
            self.logger.error(f"Error processing heading: {str(e)}")
            # Return safe defaults
            return self.sanitize_id(heading_text), heading_text

    def get_section_number(self, level: int, counters: Dict[str, int]) -> str:
        """
        Generate section number (1.1.2.3.4.5.6 format).
        """
        try:
            # Ensure all counters exist
            for i in range(1, 7):
                key = f'h{i}'
                if key not in counters:
                    counters[key] = 0

            # Reset all lower-level counters
            for i in range(level + 1, 7):
                counters[f'h{i}'] = 0

            # Increment current level counter
            counters[f'h{level}'] += 1

            # Build section number string
            section_parts = []
            for i in range(1, level + 1):
                section_parts.append(str(counters[f'h{i}']))

            section_number = '.'.join(section_parts)
            self.logger.debug(f"Generated section number: {section_number} for level: {level}")
            return section_number

        except Exception as e:
            self.logger.error(f"Error generating section number: {str(e)}")
            return str(level)

    def format_heading_text(self, level: int, text: str,
                          counters: Dict[str, int]) -> str:
        """Format heading text with section number"""
        try:
            section_number = self.get_section_number(level, counters)
            return f"{section_number}. {text}"
        except Exception as e:
            self.logger.error(f"Error formatting heading text: {str(e)}")
            return text

    def generate_heading_id(self, heading_text: str,
                          section_number: Optional[str] = None) -> str:
        """Generate heading ID from text"""
        try:
            # Check if heading already has a registered ID
            if heading_text in self.heading_map:
                return self.heading_map[heading_text]

            # Clean and normalize the text
            clean_text = re.sub(r'[^\w\s-]', '', heading_text.lower())
            words = clean_text.split()[:4]  # Take up to 4 words
            base_id = '-'.join(words)

            # Add section number if provided
            if section_number:
                base_id = f"{section_number}-{base_id}"

            # Handle duplicates
            if base_id in self.used_heading_ids:
                self.used_heading_ids[base_id] += 1
                final_id = f"{base_id}-{self.used_heading_ids[base_id]}"
            else:
                self.used_heading_ids[base_id] = 1
                final_id = base_id

            # Store mapping
            self.heading_map[heading_text] = final_id
            return final_id

        except Exception as e:
            self.logger.error(f"Error generating heading ID: {str(e)}")
            return self.sanitize_id(heading_text)

    def sanitize_id(self, text: str) -> str:
        """Create safe ID from text"""
        return re.sub(r'[^\w\-]', '-', text.lower()).strip('-')

    def reset(self) -> None:
        """Reset all counters and mappings"""
        self.used_heading_ids.clear()
        self.heading_map.clear()
        self.registered_ids.clear()
        self.logger.debug("Reset heading handler state")

    def register_existing_id(self, heading_text: str, id_value: str) -> None:
        """Register an existing ID to prevent duplicates"""
        self.heading_map[heading_text] = id_value
        self.registered_ids.add(id_value)
        self.logger.debug(f"Registered existing ID: {id_value} for heading: {heading_text}")
