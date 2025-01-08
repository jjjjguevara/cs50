from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from bs4 import BeautifulSoup, Tag
import re
from datetime import datetime

# Core types
from ..models.types import (
    ProcessedContent,
    ProcessingMetadata,
    ProcessingContext,
    ElementType,
    ValidationResult,
    ValidationMessage,
    ValidationSeverity,
    ContentScope
)

# Utils
from .id_handler import DITAIDHandler, IDType
from .logger import DITALogger
from ..utils.heading import HeadingMetadata

@dataclass
class HTMLRenderOptions:
    """Configuration options for HTML rendering."""
    pretty_print: bool = True
    minify: bool = False
    escape_html: bool = True
    include_metadata: bool = True
    add_aria: bool = True

class HTMLHelper:
    """
    Helper class for HTML manipulation and validation.
    Handles HTML generation and element manipulation.
    """

    def __init__(
        self,
        dita_root: Optional[Path] = None,
        id_handler: Optional[DITAIDHandler] = None,
        logger: Optional[DITALogger] = None,
        render_options: Optional[HTMLRenderOptions] = None
    ):
        """
        Initialize HTML helper.

        Args:
            dita_root: Optional path to DITA root directory
            id_handler: Optional ID handler instance
            logger: Optional logger instance
            render_options: Optional rendering configuration
        """
        # Core dependencies
        self.logger = logger or logging.getLogger(__name__)
        self.dita_root = dita_root
        self.id_handler = id_handler or DITAIDHandler()
        self.render_options = render_options or HTMLRenderOptions()

        # Element caching
        self._element_cache: Dict[str, Tag] = {}
        self._tag_cache: Dict[str, Dict[str, Any]] = {}

        # Template storage
        self._templates: Dict[str, str] = {}

        # HTML validation patterns
        self._validation_patterns: Dict[str, re.Pattern] = {}

        # Statistics tracking
        self._stats = {
            "elements_created": 0,
            "elements_modified": 0,
            "validations_performed": 0,
            "validation_failures": 0
        }


    def create_element(
        self,
        tag: str,
        attrs: Dict[str, Any],
        content: str = "",
        element_id: Optional[str] = None
    ) -> Tag:
        """
        Create HTML element with attributes and content.

        Args:
            tag: HTML tag name
            attrs: Element attributes
            content: Element content
            element_id: Optional unique identifier

        Returns:
            BeautifulSoup Tag element
        """
        try:
            # Generate or validate ID
            element_id = element_id or self.id_handler.generate_id(
                base=f"element_{self._stats['elements_created']}",
                id_type=IDType.HTML_ELEMENT
            )

            # Check cache first
            if element_id in self._element_cache:
                return self._element_cache[element_id]

            # Create new tag
            soup = BeautifulSoup("", "html.parser")
            element = soup.new_tag(tag)

            # Add attributes
            for key, value in attrs.items():
                if key == "class":
                    if isinstance(value, (list, tuple)):
                        element["class"] = " ".join(value)
                    else:
                        element["class"] = value
                else:
                    element[key] = value

            # Add content
            if content:
                if self.render_options.escape_html:
                    element.string = content
                else:
                    element.append(BeautifulSoup(content, "html.parser"))

            # Add to cache
            self._element_cache[element_id] = element
            self._stats["elements_created"] += 1

            return element

        except Exception as e:
            self.logger.error(f"Error creating element: {str(e)}")
            raise

    def generate_toc(self, headings: List['HeadingMetadata']) -> str:
        """Generate table of contents HTML."""
        try:
            toc_html = ['<nav class="toc"><ul>']
            for heading in headings:
                toc_html.append(
                    f'<li class="toc-level-{heading.level}">'
                    f'<a href="#{heading.id}">{heading.text}</a></li>'
                )
            toc_html.append('</ul></nav>')
            return '\n'.join(toc_html)
        except Exception as e:
            self.logger.error(f"Error generating TOC: {str(e)}")
            return ""

    def create_container(
        self,
        tag: str,
        children: List[Union[Tag, str]],
        attrs: Dict[str, Any],
        container_id: Optional[str] = None
    ) -> Tag:
        """
        Create container element with child elements.

        Args:
            tag: Container tag name
            children: List of child elements
            attrs: Container attributes
            container_id: Optional container identifier

        Returns:
            BeautifulSoup Tag container
        """
        try:
            # Create base container
            container = self.create_element(tag, attrs, "", container_id)

            # Add children
            for child in children:
                if isinstance(child, Tag):
                    container.append(child)
                else:
                    if self.render_options.escape_html:
                        container.append(child)
                    else:
                        container.append(BeautifulSoup(child, "html.parser"))

            return container

        except Exception as e:
            self.logger.error(f"Error creating container: {str(e)}")
            raise

    def validate_html(self, html_content: str) -> ValidationResult:
        """
        Validate HTML syntax and structure.

        Args:
            html_content: HTML content to validate

        Returns:
            ValidationResult with validation status and messages
        """
        try:
            self._stats["validations_performed"] += 1
            messages: List[ValidationMessage] = []

            if not html_content.strip():
                messages.append(
                    ValidationMessage(
                        path="content",
                        message="Empty HTML content",
                        severity=ValidationSeverity.ERROR,
                        code="empty_content"
                    )
                )
                return ValidationResult(is_valid=False, messages=messages)

            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')

            # Check for basic HTML structure
            if soup.find():
                # Validate specific elements
                messages.extend(self._validate_elements(soup))

                # Validate attributes
                messages.extend(self._validate_attributes(soup))

                # Validate accessibility
                if self.render_options.add_aria:
                    messages.extend(self._validate_accessibility(soup))

            else:
                messages.append(
                    ValidationMessage(
                        path="structure",
                        message="Invalid HTML structure",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_structure"
                    )
                )

            is_valid = not any(
                msg.severity == ValidationSeverity.ERROR
                for msg in messages
            )

            if not is_valid:
                self._stats["validation_failures"] += 1

            return ValidationResult(is_valid=is_valid, messages=messages)

        except Exception as e:
            self.logger.error(f"Error validating HTML: {str(e)}")
            self._stats["validation_failures"] += 1
            return ValidationResult(
                is_valid=False,
                messages=[
                    ValidationMessage(
                        path="validation",
                        message=str(e),
                        severity=ValidationSeverity.ERROR,
                        code="validation_error"
                    )
                ]
            )

    def _validate_elements(self, soup: BeautifulSoup) -> List[ValidationMessage]:
        """Validate HTML elements structure."""
        messages = []

        # Validate heading hierarchy
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        current_level = 0
        for heading in headings:
            level = int(heading.name[1])
            if level > current_level + 1:
                messages.append(
                    ValidationMessage(
                        path=f"heading_{heading.get('id', '')}",
                        message=f"Invalid heading hierarchy: h{current_level} to h{level}",
                        severity=ValidationSeverity.WARNING,
                        code="heading_hierarchy"
                    )
                )
            current_level = level

        return messages

    def _validate_attributes(self, soup: BeautifulSoup) -> List[ValidationMessage]:
        """Validate HTML element attributes."""
        messages = []

        for tag in soup.find_all(True):  # Find all tags
            # Check for required attributes
            if tag.name == 'img':
                if not tag.get('alt'):
                    messages.append(
                        ValidationMessage(
                            path=f"img_{tag.get('id', '')}",
                            message="Image missing alt attribute",
                            severity=ValidationSeverity.ERROR,
                            code="missing_alt"
                        )
                    )
                if not tag.get('src'):
                    messages.append(
                        ValidationMessage(
                            path=f"img_{tag.get('id', '')}",
                            message="Image missing src attribute",
                            severity=ValidationSeverity.ERROR,
                            code="missing_src"
                        )
                    )

            # Validate links
            if tag.name == 'a':
                if not tag.get('href'):
                    messages.append(
                        ValidationMessage(
                            path=f"link_{tag.get('id', '')}",
                            message="Link missing href attribute",
                            severity=ValidationSeverity.ERROR,
                            code="missing_href"
                        )
                    )

        return messages

    def _validate_accessibility(self, soup: BeautifulSoup) -> List[ValidationMessage]:
        """Validate accessibility requirements."""
        messages = []

        # Check for ARIA landmarks
        if not soup.find(role="main"):
            messages.append(
                ValidationMessage(
                    path="landmarks",
                    message="Missing main landmark",
                    severity=ValidationSeverity.WARNING,
                    code="missing_landmark"
                )
            )

        # Check for proper ARIA attributes
        for tag in soup.find_all(True):
            if tag.get('role') and not tag.get('aria-label'):
                messages.append(
                    ValidationMessage(
                        path=f"{tag.name}_{tag.get('id', '')}",
                        message=f"Element with role missing aria-label",
                        severity=ValidationSeverity.WARNING,
                        code="missing_aria_label"
                    )
                )

        return messages

    def resolve_image_path(self, src: str, topic_path: Path) -> str:
        """
        Resolve image path relative to topic file.

        Args:
            src: Source image path
            topic_path: Path to topic file

        Returns:
            Resolved image path
        """
        try:
            if not self.dita_root:
                self.logger.error("DITA root not set")
                return src

            # Clean the source path
            src = src.strip('/')

            # Get topic directory and media path
            topic_dir = topic_path.parent
            media_dir = topic_dir / 'media'

            # Construct full path
            img_path = (media_dir / src).resolve()

            # Check if image exists
            if img_path.exists():
                # Make path relative to DITA root
                relative_path = img_path.relative_to(self.dita_root)
                return f'/static/topics/{relative_path}'
            else:
                self.logger.warning(f"Image not found: {img_path}")
                return src

        except Exception as e:
            self.logger.error(f"Error resolving image path: {str(e)}")
            return src

    def get_stats(self) -> Dict[str, int]:
        """Get HTML helper statistics."""
        return self._stats.copy()

    def cleanup(self) -> None:
        """Clean up helper resources."""
        try:
            self._element_cache.clear()
            self._tag_cache.clear()
            self._templates.clear()
            self._validation_patterns.clear()
            self._stats = {
                "elements_created": 0,
                "elements_modified": 0,
                "validations_performed": 0,
                "validation_failures": 0
            }
            self.logger.debug("HTML helper cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise
