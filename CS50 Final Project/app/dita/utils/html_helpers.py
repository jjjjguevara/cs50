# app/dita/utils/html_helpers.py
from typing import List, Optional, Dict, Any, Union
from bs4 import BeautifulSoup, Tag, NavigableString
import logging
import html
import re
from pathlib import Path
import json

class HTMLHelper:
    """Comprehensive HTML manipulation helper"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    # Content Wrapper Methods
    def ensure_wrapper(self, soup: BeautifulSoup,
                      wrapper_class: str = 'content-wrapper') -> Tag:
        """
        Ensure content has wrapper div.
        Returns the wrapper div as a Tag.
        """
        content_div = soup.find('div', class_=wrapper_class)

        if not content_div or not isinstance(content_div, Tag):
            content_div = soup.new_tag('div')
            content_div['class'] = wrapper_class

            # Move all content inside wrapper
            for tag in soup.contents[:]:
                content_div.append(tag.extract())
            soup.append(content_div)

        return content_div

    # Class Manipulation Methods
    def add_class(self, element: Optional[Tag], class_name: str) -> None:
        """Add class to element safely"""
        if element and isinstance(element, Tag):
            classes = self.get_classes(element)
            if class_name not in classes:
                classes.append(class_name)
                element['class'] = ' '.join(classes)

    def remove_class(self, element: Optional[Tag], class_name: str) -> None:
        """Remove class from element safely"""
        if element and isinstance(element, Tag):
            classes = self.get_classes(element)
            if class_name in classes:
                classes.remove(class_name)
                element['class'] = ' '.join(classes)

    def has_class(self, element: Optional[Tag], class_name: str) -> bool:
        """Check if element has specific class"""
        if element and isinstance(element, Tag):
            return class_name in self.get_classes(element)
        return False

    def get_classes(self, element: Tag) -> List[str]:
        """Get element's classes as list"""
        classes = element.get('class', [])
        if isinstance(classes, str):
            return classes.split()
        elif isinstance(classes, list):
            return classes
        return []

    def set_classes(self, element: Tag, classes: List[str]) -> None:
        """Set element's classes from list"""
        if element and isinstance(element, Tag):
            element['class'] = ' '.join(classes)

    # Attribute Manipulation Methods
    def set_data_attributes(self, element: Tag,
                              attributes: Dict[str, Any]) -> None:
            """
            Set data attributes on element safely.

            Args:
                element: BeautifulSoup Tag to modify
                attributes: Dictionary of attribute names and values
            """
            if not self.is_valid_element(element):
                self.logger.warning("Invalid element for setting data attributes")
                return

            try:
                for key, value in attributes.items():
                    # Sanitize key
                    safe_key = self.sanitize_attribute_name(key)
                    # Convert value to string safely
                    safe_value = str(value)
                    # Set attribute
                    element[f'data-{safe_key}'] = safe_value

            except Exception as e:
                self.logger.error(f"Error setting data attributes: {str(e)}")

    def sanitize_attribute_name(self, name: str) -> str:
        """
        Sanitize attribute name for safe use in HTML.
        Converts spaces and special characters to hyphens.
        """
        return re.sub(r'[^\w\-]', '-', name.lower()).strip('-')

    def is_valid_element(self, element: Any) -> bool:
        """
        Check if element is valid BS4 Tag.
        """
        return element is not None and isinstance(element, Tag)

    def find_target_element(self, soup: BeautifulSoup, target_id: str) -> Optional[Tag]:
        """
        Find target element by ID, with fallback strategies.

        Args:
            soup: BeautifulSoup object to search
            target_id: Target ID to find

        Returns:
            Optional[Tag]: Found element or None
        """
        try:
            # Try exact match first
            target_elem = soup.find(id=target_id)
            if target_elem and isinstance(target_elem, Tag):
                self.logger.debug(f"Found target element with exact ID: {target_id}")
                return target_elem

            # Try finding by generated ID patterns
            for elem in soup.find_all(True):  # Find all tags
                if isinstance(elem, Tag):
                    elem_id = elem.get('id')
                    if isinstance(elem_id, str):  # Ensure ID is a string
                        # Check if current ID is a variation of target ID
                        if (elem_id.startswith(target_id) or
                            target_id in elem_id or
                            elem_id.endswith(target_id)):
                            self.logger.debug(f"Found target element with related ID: {elem_id}")
                            return elem

            self.logger.warning(f"Target element not found: {target_id}")
            return None

        except Exception as e:
            self.logger.error(f"Error finding target element: {str(e)}")
            return None

    # Artifact helper methods
    def create_artifact_error_message(self, artifact: Dict[str, Any], error: str) -> str:
            """Create error message for failed artifact injection"""
            return f"""
            <div class="artifact-error bg-red-50 border-l-4 border-red-500 p-4 mb-4">
                <div class="flex">
                    <div class="flex-shrink-0">
                        <svg class="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                            <path fill-rule="evenodd"
                                  d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                                  clip-rule="evenodd"/>
                        </svg>
                    </div>
                    <div class="ml-3">
                        <h3 class="text-sm font-medium text-red-800">
                            Failed to load artifact: {html.escape(artifact.get('name', 'Unknown'))}
                        </h3>
                        <div class="mt-2 text-sm text-red-700">
                            {html.escape(error)}
                        </div>
                    </div>
                </div>
            </div>
            """

    def add_injection_status(self, soup: BeautifulSoup,
                            successful: List[str],
                            failed: List[Dict[str, str]]) -> None:
        """Add injection status information to document"""
        try:
            status_div = soup.new_tag('div', attrs={
                'class': 'hidden',
                'data-artifact-status': 'true',
                'data-successful': json.dumps(successful),
                'data-failed': json.dumps(failed)
            })

            # Find or create content wrapper
            wrapper = self.ensure_wrapper(soup)
            wrapper.append(status_div)

        except Exception as e:
            self.logger.error(f"Error adding injection status: {str(e)}")

    def add_critical_error_message(self, content: str, error: str) -> str:
        """Add critical error message to content"""
        error_html = f"""
        <div class="critical-error bg-red-50 border-l-4 border-red-500 p-4 mb-8">
            <div class="flex">
                <div class="ml-3">
                    <h3 class="text-sm font-medium text-red-800">
                        Critical Error Loading Content
                    </h3>
                    <div class="mt-2 text-sm text-red-700">
                        {html.escape(error)}
                    </div>
                    <div class="mt-1 text-xs text-red-600">
                        The article content is still available below.
                    </div>
                </div>
            </div>
        </div>
        {content}
        """
        return error_html
