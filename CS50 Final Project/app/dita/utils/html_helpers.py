# app/dita/utils/html_helpers.py

import html
import re
from typing import Dict, List, Optional, Any
from bs4 import BeautifulSoup, Tag
import logging

class HTMLHelper:
    """Utilities for HTML manipulation and validation."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def escape_html(self, content: str) -> str:
        """
        Escape HTML special characters in content.

        Args:
            content: String content to escape

        Returns:
            Escaped HTML content
        """
        try:
            return html.escape(content, quote=True)
        except Exception as e:
            self.logger.error(f"Error escaping HTML: {str(e)}")
            return ""

    def process_final_content(self, content: str) -> str:
        """
        Process final HTML content before output.

        Args:
            content: HTML content to process

        Returns:
            Processed HTML content
        """
        try:
            # Parse HTML
            soup = BeautifulSoup(content, 'html.parser')

            # Clean whitespace
            output = self._clean_whitespace(str(soup))

            # Ensure proper nesting
            output = self._ensure_proper_nesting(output)

            return output

        except Exception as e:
            self.logger.error(f"Error processing final content: {str(e)}")
            return content

    def find_target_element(self, soup: BeautifulSoup, target_id: str) -> Optional[Tag]:
        """
        Find target element by ID in BeautifulSoup object.

        Args:
            soup: BeautifulSoup object to search
            target_id: ID to find

        Returns:
            Target element if found and is a Tag, None otherwise
        """
        try:
            element = soup.find(id=target_id)
            # Only return if element is a Tag
            if isinstance(element, Tag):
                return element
            return None
        except Exception as e:
            self.logger.error(f"Error finding target element: {str(e)}")
            return None

    def _clean_whitespace(self, content: str) -> str:
        """
        Clean excess whitespace from HTML content.

        Args:
            content: Content to clean

        Returns:
            Cleaned content
        """
        try:
            # Remove multiple spaces
            content = re.sub(r'\s+', ' ', content)

            # Clean space around tags
            content = re.sub(r'>\s+<', '><', content)

            # Clean empty lines
            content = re.sub(r'\n\s*\n', '\n', content)

            return content.strip()

        except Exception as e:
            self.logger.error(f"Error cleaning whitespace: {str(e)}")
            return content

    def _ensure_proper_nesting(self, content: str) -> str:
        """
        Ensure proper HTML tag nesting.

        Args:
            content: HTML content to check

        Returns:
            Properly nested HTML
        """
        try:
            soup = BeautifulSoup(content, 'html.parser')
            return str(soup)
        except Exception as e:
            self.logger.error(f"Error ensuring proper nesting: {str(e)}")
            return content

    def validate_html(self, content: str) -> bool:
        """
        Validate HTML content structure.

        Args:
            content: HTML content to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            soup = BeautifulSoup(content, 'html.parser')
            return True
        except Exception as e:
            self.logger.error(f"HTML validation failed: {str(e)}")
            return False

    def add_classes(self, tag: Tag, classes: List[str]) -> None:
        """
        Add classes to HTML tag.

        Args:
            tag: BeautifulSoup tag
            classes: List of classes to add
        """
        try:
            existing_classes = tag.get('class', [])
            if isinstance(existing_classes, str):
                existing_classes = existing_classes.split()

            # Add new classes
            tag['class'] = list(set(existing_classes + classes))

        except Exception as e:
            self.logger.error(f"Error adding classes: {str(e)}")

    def remove_classes(self, tag: Tag, classes: List[str]) -> None:
        """
        Remove classes from HTML tag.

        Args:
            tag: BeautifulSoup tag
            classes: List of classes to remove
        """
        try:
            existing_classes = tag.get('class', [])
            if isinstance(existing_classes, str):
                existing_classes = existing_classes.split()

            # Remove specified classes
            remaining_classes = [c for c in existing_classes if c not in classes]
            if remaining_classes:
                tag['class'] = remaining_classes
            else:
                del tag['class']

        except Exception as e:
            self.logger.error(f"Error removing classes: {str(e)}")

    def wrap_content(self, content: str, wrapper_tag: str, classes: Optional[List[str]] = None) -> str:
        """
        Wrap content in HTML tag.

        Args:
            content: Content to wrap
            wrapper_tag: HTML tag to wrap with
            classes: Optional classes to add

        Returns:
            Wrapped content
        """
        try:
            class_attr = f' class="{" ".join(classes)}"' if classes else ''
            return f'<{wrapper_tag}{class_attr}>{content}</{wrapper_tag}>'
        except Exception as e:
            self.logger.error(f"Error wrapping content: {str(e)}")
            return content
