# app/dita/utils/html_helpers.py

import html
import re
from typing import Dict, List, Optional, Any, Union, Sequence
from pathlib import Path
from bs4 import BeautifulSoup, Tag
import logging
from app.dita.models.types import ProcessedContent


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


    def render(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Transform raw XML content into styled HTML, enriched with metadata.

        Args:
            content: The raw XML or intermediate HTML content.
            metadata: Optional metadata to append to the HTML output.

        Returns:
            Styled and enriched HTML as a string.
        """
        try:
            # Parse content with BeautifulSoup
            soup = BeautifulSoup(content, "lxml-xml")  # Use XML parser for DITA content

            # Transform <title> tags to <h1> for HTML display
            for title in soup.find_all("title"):
                h1 = soup.new_tag("h1")
                h1.string = title.text
                title.replace_with(h1)

            # Replace <topicgroup> with styled sections
            for group in soup.find_all("topicgroup"):
                div = soup.new_tag("div", **{"class": "topic-group"})
                navtitle = group.find("navtitle")
                if navtitle:
                    h2 = soup.new_tag("h2")
                    h2.string = navtitle.text
                    div.append(h2)
                ul = soup.new_tag("ul")
                for ref in group.find_all("topicref"):
                    li = soup.new_tag("li")
                    a = soup.new_tag("a", href=ref.get("href", "#"))
                    a.string = ref.get("href", "Unnamed Topic")
                    li.append(a)
                    ul.append(li)
                div.append(ul)
                group.replace_with(div)

            # Handle metadata (if provided and <body> exists)
            if metadata and soup.body:
                metadata_div = soup.new_tag("div", **{"class": "metadata-section"})
                for key, value in metadata.items():
                    meta_item = soup.new_tag("div", **{"class": "metadata-item"})
                    meta_item.string = f"{key.capitalize()}: {value}"
                    metadata_div.append(meta_item)
                soup.body.insert(0, metadata_div)  # Add metadata at the top of <body>

            # Safely handle optional <metadata>, ensuring the tag exists
            metadata_tag = soup.find("metadata")
            if metadata_tag and metadata_tag.name == "metadata":
                metadata_tag.decompose()

            # Return final sanitized and enriched HTML
            return str(soup)

        except Exception as e:
            self.logger.error(f"Error rendering content: {str(e)}")
            raise



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

    def validate_html(self, html_content: str) -> bool:
            """Validate HTML content."""
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                return bool(soup.find())
            except Exception:
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


    def render_headings(self, content: str, toc_enabled: bool) -> str:
        """
        Enhance content with heading links or numbering.
        """
        soup = BeautifulSoup(content, 'html.parser')

        # Handle cases where soup.body is None
        if soup.body is None:
            self.logger.warning("Content does not have a <body> tag")
            return content

        if toc_enabled:
            toc = soup.new_tag('div', id='table-of-contents')
            toc.append(soup.new_tag('h2', string='Table of Contents'))

            for heading in soup.find_all(['h1', 'h2', 'h3']):
                toc.append(soup.new_tag('a', href=f"#{heading['id']}", string=heading.text))
            soup.body.insert(0, toc)

        return str(soup)


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
