# app/dita/utils/html_helpers.py

import html
import re
from typing import Dict, List, Optional, Any, Union, Sequence
from pathlib import Path
from bs4 import BeautifulSoup, Tag
import logging
from app.dita.models.types import (
    ProcessedContent,
    ProcessingMetadata,
    ProcessingContext
)

from app.dita.utils.id_handler import DITAIDHandler

# Global config
from app_config import DITAConfig

def consolidate_transformed_html(transformed_contents: Sequence[Union[str, ProcessedContent]]) -> str:
    """Consolidate unique transformed HTML contents."""
    seen = set()
    unique_html = []
    for content in transformed_contents:
        if isinstance(content, ProcessedContent):
            html_content = content.html
        elif isinstance(content, str):
            html_content = content
        else:
            raise ValueError(f"Unexpected content type: {type(content)}")

        if html_content not in seen:
            unique_html.append(html_content)
            seen.add(html_content)
    return "\n".join(unique_html)

class HTMLHelper:
    """Utilities for HTML manipulation and validation."""


    def __init__(self, dita_root: Optional[Path] = None):
        """
        Initialize HTML helper.

        Args:
            dita_root: Optional path to DITA root directory
        """
        self.logger = logging.getLogger(__name__)
        self.dita_root = dita_root
        self._cache: Dict[str, Any] = {}

    def configure_helper(self, config: DITAConfig) -> None:
        """
        Configure HTML helper with provided settings.

        Args:
            config (DITAConfig): Configuration object containing relevant settings.
        """
        try:
            self.logger.debug("Configuring HTML helper")

            # Validate that the dita_root attribute exists in the config
            if not hasattr(config, 'topics_dir') or not config.topics_dir:
                raise ValueError("DITAConfig must have a valid 'topics_dir' attribute.")

            # Update root path
            self.dita_root = config.topics_dir.parent

            self.logger.debug(f"HTML helper configured with DITA root: {self.dita_root}")

        except Exception as e:
            self.logger.error(f"HTML helper configuration failed: {str(e)}")
            raise



    def create_element(self, tag: str, attrs: Dict[str, Any], content: str) -> str:
        """Create HTML element with attributes and content."""
        try:
            # Create new tag
            soup = BeautifulSoup("", "html.parser")
            element = soup.new_tag(tag)

            # Add attributes
            for key, value in attrs.items():
                if key == "class":
                    element["class"] = value if isinstance(value, str) else " ".join(value)
                else:
                    element[key] = value

            # Add content
            if content:
                element.string = content

            return str(element)

        except Exception as e:
            self.logger.error(f"Error creating element: {str(e)}")
            return ""

    def create_container(self, tag: str, children: List[str], attrs: Dict[str, Any]) -> str:
        """Create container element with child elements."""
        try:
            # Create base container tag
            soup = BeautifulSoup("", "html.parser")
            container = soup.new_tag(tag)

            # Add attributes
            for key, value in attrs.items():
                if key == "class":
                    container["class"] = value if isinstance(value, str) else " ".join(value)
                else:
                    container[key] = value

            # Add children
            for child in children:
                child_soup = BeautifulSoup(child, "html.parser")
                if child_soup.body:
                    container.append(child_soup.body.decode_contents())
                else:
                    container.append(child_soup.decode_contents())

            return str(container)

        except Exception as e:
            self.logger.error(f"Error creating container: {str(e)}")
            return ""

    def assemble_topic(self, elements: List[str], metadata: Dict[str, Any]) -> str:
        """Assemble topic HTML with metadata."""
        try:
            soup = BeautifulSoup("", "html.parser")

            # Create article container
            article = soup.new_tag("article", attrs={
                "class": "topic-content",
                "data-topic-id": metadata.get("id", ""),
                "data-topic-type": metadata.get("type", "topic")
            })
            soup.append(article)

            # Add metadata section
            meta_div = soup.new_tag("div", attrs={"class": "topic-metadata"})
            for key, value in metadata.items():
                meta_div["data-" + key] = str(value)
            article.append(meta_div)

            # Add content elements
            for element in elements:
                elem_soup = BeautifulSoup(element, "html.parser")
                if elem_soup.body:
                    article.append(elem_soup.body.decode_contents())
                else:
                    article.append(elem_soup.decode_contents())

            return str(soup)

        except Exception as e:
            self.logger.error(f"Error assembling topic: {str(e)}")
            return ""


    def convert_html(self, content: str) -> str:
        """Basic HTML conversion with proper parsing."""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            return str(soup)
        except Exception as e:
            self.logger.error(f"HTML conversion failed: {str(e)}")
            return content

    def validate_html(self, html_content: str) -> bool:
        """Validate HTML syntax and structure."""
        try:
            if not html_content.strip():
                return False
            soup = BeautifulSoup(html_content, 'html.parser')
            return bool(soup.find() and "<html" not in html_content.lower())
        except Exception:
            return False


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

    def process_final_content(self, html_content: str) -> str:
            """Process final HTML content before rendering."""
            try:
                # Unescape any previously escaped HTML
                html_content = html.unescape(html_content)

                # Parse with BeautifulSoup to properly format HTML
                soup = BeautifulSoup(html_content, 'html.parser')

                # Return properly formatted HTML
                return str(soup)

            except Exception as e:
                self.logger.error(f"Error processing HTML content: {str(e)}")
                return html_content



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



    def resolve_image_path(self, src: str, topic_path: Path) -> str:
        """Resolve image path relative to topic file."""
        try:
            if not self.dita_root:
                self.logger.error("DITA root not set")
                return src

            # Get topic directory and media path
            topic_dir = topic_path.parent
            media_dir = topic_dir / 'media'

            # Clean the source path and construct full path
            src = src.strip('/')
            img_path = (media_dir / src).resolve()

            self.logger.debug(f"Trying to resolve image: {img_path}")

            # Make path relative to DITA root for serving
            if img_path.exists():
                relative_path = img_path.relative_to(self.dita_root)
                self.logger.debug(f"Image found, serving from: {relative_path}")
                return f'/static/topics/{relative_path}'
            else:
                self.logger.warning(f"Image not found at: {img_path}")
                return src

        except Exception as e:
            self.logger.error(f"Error resolving image path: {str(e)}")
            return src

    # Cleanup

    def cleanup(self) -> None:
        """Clean up helper resources and state."""
        try:
            self.logger.debug("Starting HTML helper cleanup")

            # Reset state
            self.dita_root = None
            self._cache.clear()

            self.logger.debug("HTML helper cleanup completed")

        except Exception as e:
            self.logger.error(f"HTML helper cleanup failed: {str(e)}")
            raise


    #########
    # Content Objects
    #########

    def generate_toc(self, metadata: ProcessingMetadata) -> str:
        """
        Generate a Table of Contents (TOC) from ProcessingMetadata.

        Args:
            metadata: The ProcessingMetadata object containing heading references.

        Returns:
            str: A structured HTML string for the TOC.
        """
        try:
            toc_html = "<nav class='toc'><ul>"
            for heading_id, heading_data in metadata.references["headings"].items():
                level = heading_data["level"]
                text = heading_data["text"]
                toc_html += f"<li class='toc-level-{level}'><a href='#{heading_id}'>{text}</a></li>"
            toc_html += "</ul></nav>"
            return toc_html
        except Exception as e:
            self.logger.error(f"Error generating TOC: {str(e)}")
            return "<!-- TOC generation failed -->"

    def generate_xref(self, source_id: str, target_ref: str, id_handler: DITAIDHandler) -> str:
        """
        Generate an HTML anchor for a cross-reference.

        Args:
            source_id: The source topic ID.
            target_ref: The target reference (e.g., topic#heading).
            id_handler: An instance of the DITAIDHandler.

        Returns:
            str: An HTML anchor tag.
        """
        try:
            href = id_handler.resolve_xref(source_id, target_ref)
            return f"<a href='{href}'>{target_ref}</a>"
        except Exception as e:
            self.logger.error(f"Error generating xref for {target_ref}: {str(e)}")
            return f"<a href='#'>{target_ref}</a>"
