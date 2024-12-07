# app/dita/utils/markdown/md_transform.py

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from bs4 import BeautifulSoup, Tag
import markdown
from markdown.extensions import Extension

# Global config
from config import DITAConfig

from ..types import (
    MDElementType,
    MDElementInfo,
    MDElementContext,
    ElementAttributes,
    ProcessingError,
    ProcessingState
)
from .md_elements import MarkdownContentProcessor
from ..html_helpers import HTMLHelper
from ..id_handler import DITAIDHandler

class MarkdownTransformer:
    """Transforms Markdown content to HTML using element definitions."""

    def __init__(self, dita_root: Path):
        self.logger = logging.getLogger(__name__)
        self.id_handler = DITAIDHandler()
        self.dita_root = dita_root
        self.html = HTMLHelper(dita_root)
        self.element_processor = MarkdownContentProcessor()

    def configure(self, config: DITAConfig) -> None:
        """Configure element processor with provided settings."""
        try:
            self.logger.debug("Configuring Markdown element processor")

            # Configure paths if needed
            if hasattr(config.paths, 'dita_root'):
                self.dita_root = config.paths.dita_root

            # Configure ID handler
            if hasattr(self.id_handler, 'configure'):
                self.id_handler.configure(config)

            self.logger.debug("Markdown element processor configuration completed")

        except Exception as e:
            self.logger.error(f"Markdown element processor configuration failed: {str(e)}")
            raise

    def transform_topic(self, topic_path: Path) -> str:
        """Transform Markdown topic to HTML."""
        try:
            self.logger.debug(f"Transforming Markdown topic: {topic_path}")

            # Read markdown content
            with open(topic_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Convert to HTML using python-markdown
            html_content = markdown.markdown(
                content,
                extensions=['extra', 'tables', 'sane_lists']  # Add necessary extensions
            )

            # Parse HTML to process elements
            soup = BeautifulSoup(html_content, 'html.parser')

            # Process each element
            transformed_content = []
            for elem in soup.find_all(True):  # Find all tags
                if isinstance(elem, Tag):
                    element_info = self.element_processor.process_element(elem)
                    if element_info:
                        html = self._transform_element(element_info)
                        transformed_content.append(html)

            # Combine and return
            return '\n'.join(transformed_content)

        except Exception as e:
            self.logger.error(f"Error transforming topic {topic_path}: {str(e)}")
            raise ProcessingError(
                error_type="markdown_transformation",
                message=f"Failed to transform Markdown: {str(e)}",
                context=str(topic_path)
            )

    def _transform_element(self, element_info: MDElementInfo) -> str:
        """Transform element info to HTML."""
        try:
            # Get transformation method based on element type
            transform_method = self._get_transform_method(element_info.type)

            if transform_method:
                return transform_method(element_info)

            # Default transformation
            return self._transform_default(element_info)

        except Exception as e:
            self.logger.error(f"Error transforming element: {str(e)}")
            return ""

    def _transform_table(self, element_info: MDElementInfo) -> str:
        """Transform table element."""
        try:
            classes = [*element_info.attributes.classes, "md-table"]
            return (
                f'<div class="table-wrapper overflow-x-auto">'
                f'<table class="{" ".join(classes)}">'
                f'{element_info.content}'
                f'</table>'
                f'</div>'
            )
        except Exception as e:
            self.logger.error(f"Error transforming table: {str(e)}")
            return ""

    def _get_transform_method(self, element_type: MDElementType):
        """Get appropriate transformation method for element type."""
        transform_methods = {
            MDElementType.HEADING: self._transform_heading,
            MDElementType.PARAGRAPH: self._transform_paragraph,
            MDElementType.LIST: self._transform_list,
            MDElementType.LIST_ITEM: self._transform_list_item,
            MDElementType.CODE: self._transform_code,
            MDElementType.BLOCKQUOTE: self._transform_blockquote,
            MDElementType.LINK: self._transform_link,
            MDElementType.IMAGE: self._transform_image,
            MDElementType.TABLE: self._transform_table,
            MDElementType.EMPHASIS: self._transform_emphasis,
            MDElementType.STRONG: self._transform_strong,
            MDElementType.INLINE_CODE: self._transform_inline_code
        }
        return transform_methods.get(element_type)

    def _transform_heading(self, element_info: MDElementInfo) -> str:
        """Transform heading element."""
        level = element_info.metadata.get('level', 1)
        return (
            f'<h{level} id="{element_info.attributes.id}" '
            f'class="{" ".join(element_info.attributes.classes)}">'
            f'{element_info.content}'
            f'<a href="#{element_info.attributes.id}" class="heading-anchor">¶</a>'
            f'</h{level}>'
        )

    def _transform_paragraph(self, element_info: MDElementInfo) -> str:
        """Transform paragraph element."""
        return (
            f'<p class="{" ".join(element_info.attributes.classes)}">'
            f'{element_info.content}'
            f'</p>'
        )

    def _transform_list(self, element_info: MDElementInfo) -> str:
        """Transform list element."""
        tag = 'ul' if element_info.type == MDElementType.LIST else 'ol'
        return (
            f'<{tag} class="{" ".join(element_info.attributes.classes)}">'
            f'{element_info.content}'
            f'</{tag}>'
        )

    def _transform_list_item(self, element_info: MDElementInfo) -> str:
        """Transform list item element."""
        return (
            f'<li class="{" ".join(element_info.attributes.classes)}">'
            f'{element_info.content}'
            f'</li>'
        )

    def _transform_code(self, element_info: MDElementInfo) -> str:
        """Transform code block element."""
        language = element_info.metadata.get('language', '')
        lang_class = f'language-{language}' if language else ''
        classes = [*element_info.attributes.classes, lang_class]

        return (
            f'<pre><code class="{" ".join(classes)}">'
            f'{self.html.escape_html(element_info.content)}'
            f'</code></pre>'
        )

    def _transform_blockquote(self, element_info: MDElementInfo) -> str:
        """Transform blockquote element."""
        return (
            f'<blockquote class="{" ".join(element_info.attributes.classes)}">'
            f'{element_info.content}'
            f'</blockquote>'
        )

    def _transform_link(self, element_info: MDElementInfo) -> str:
        """Transform link element."""
        href = element_info.metadata.get('href', '#')
        title = element_info.metadata.get('title', '')
        title_attr = f' title="{title}"' if title else ''

        return (
            f'<a href="{href}" class="{" ".join(element_info.attributes.classes)}"'
            f'{title_attr}>{element_info.content}</a>'
        )

    def _transform_image(self, element_info: MDElementInfo) -> str:
        """Transform image element."""
        src = element_info.metadata.get('src', '')
        alt = element_info.metadata.get('alt', '')

        # Resolve image path if topic path is available
        if src and element_info.context.topic_path is not None:
            src = self.html.resolve_image_path(src, element_info.context.topic_path)

        return (
            f'<img src="{src}" alt="{alt}" '
            f'class="{" ".join(element_info.attributes.classes)}">'
        )

    def _transform_emphasis(self, element_info: MDElementInfo) -> str:
        """Transform emphasis element."""
        return f'<em>{element_info.content}</em>'

    def _transform_strong(self, element_info: MDElementInfo) -> str:
        """Transform strong element."""
        return f'<strong>{element_info.content}</strong>'

    def _transform_inline_code(self, element_info: MDElementInfo) -> str:
        """Transform inline code element."""
        return (
            f'<code class="{" ".join(element_info.attributes.classes)}">'
            f'{self.html.escape_html(element_info.content)}'
            f'</code>'
        )

    def _transform_default(self, element_info: MDElementInfo) -> str:
        """Default transformation for unknown elements."""
        return element_info.content


    # Cleanup

    def cleanup(self) -> None:
        """Clean up transformer resources and state."""
        try:
            self.logger.debug("Starting Markdown transformer cleanup")

            # Reset internal state
            self.reset()

            # Clean up element processor
            if hasattr(self.element_processor, 'cleanup'):
                self.element_processor.cleanup()

            # Clean up HTML helper
            if hasattr(self.html, 'cleanup'):
                self.html.cleanup()

            # Reset ID handler
            self.id_handler = DITAIDHandler()

            self.logger.debug("Markdown transformer cleanup completed")

        except Exception as e:
            self.logger.error(f"Markdown transformer cleanup failed: {str(e)}")
            raise

    def reset(self) -> None:
        """Reset transformer to initial state."""
        try:
            self.logger.debug("Resetting Markdown transformer")

            # Reset element processor
            self.element_processor = MarkdownContentProcessor()

            # Reset HTML helper
            self.html = HTMLHelper(self.dita_root)

            self.logger.debug("Markdown transformer reset completed")

        except Exception as e:
            self.logger.error(f"Markdown transformer reset failed: {str(e)}")
            raise
