# app/dita/utils/markdown/md_transform.py

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from bs4 import BeautifulSoup, Tag
import markdown
import yaml
import re
from markdown.extensions import Extension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.footnotes import FootnoteExtension
from markdown.extensions.attr_list import AttrListExtension
from markdown.extensions.tables import TableExtension
from markdown.extensions.extra import ExtraExtension




# Global config
from config import DITAConfig

from ..types import (
    MDElementType,
    MDElementInfo,
    MDElementContext,
    ElementAttributes,
    ProcessedContent,
    ProcessingError,
    ProcessingState
)
from .md_elements import MarkdownContentProcessor
from ..html_helpers import HTMLHelper
from ..id_handler import DITAIDHandler
from ..heading import HeadingHandler
from ..metadata import MetadataHandler

class MarkdownTransformer:
    """Transforms Markdown content to HTML using element definitions."""

    def __init__(self, dita_root: Path):
            self.logger = logging.getLogger(__name__)
            self.id_handler = DITAIDHandler()
            self.dita_root = dita_root
            self.html = HTMLHelper(dita_root)
            self.heading_handler = HeadingHandler()
            self.metadata_handler = MetadataHandler()
            self.extensions = [
                FencedCodeExtension(),
                FootnoteExtension(),
                AttrListExtension(),
                TableExtension(),
                ExtraExtension(),
            ]


    def configure(self, config: DITAConfig) -> None:
        """Configure element processor with provided settings."""
        try:
            self.logger.debug("Configuring Markdown element processor")
            if hasattr(config.paths, 'dita_root'):
                self.dita_root = config.paths.dita_root
            self.id_handler.configure(config)
            self.logger.debug("Markdown element processor configuration completed")
        except Exception as e:
            self.logger.error(f"Markdown element processor configuration failed: {str(e)}")
            raise

    def extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from Markdown content."""
        try:
            match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
            if match:
                metadata = yaml.safe_load(match.group(1))
                self.logger.debug(f"Extracted metadata: {metadata}")
                return metadata
            return {}
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {str(e)}")
            return {}

    def transform_topic(self, topic_path: Path) -> ProcessedContent:
            try:
                with open(topic_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                # Transform Markdown to HTML
                html_content = markdown.markdown(content, extensions=self.extensions)

                # Extract metadata using MetadataHandler
                metadata = self.metadata_handler.extract_metadata(topic_path, topic_path.stem)

                # Process headings
                heading_ids = []
                soup = BeautifulSoup(html_content, 'html.parser')
                for heading in soup.find_all(['h1', 'h2', 'h3']):
                    heading_id = self.id_handler.generate_id(heading.text)
                    heading['id'] = heading_id
                    heading_ids.append(heading_id)

                return ProcessedContent(
                    html=str(soup),
                    element_id=self.id_handler.generate_id(topic_path.name),
                    metadata=metadata
                )
            except Exception as e:
                self.logger.error(f"Error transforming Markdown topic {topic_path}: {str(e)}")
                raise


    def _transform_element(self, element_info: MDElementInfo) -> str:
        """Transform element info to HTML."""
        try:
            content = element_info.content

            # Handle marked content types
            if content.startswith('LINK:'):
                _, href, text = content.split(':', 2)
                return f'<a href="{href}" class="markdown-link">{text}</a>'

            if content.startswith('QUOTE:'):
                _, text = content.split(':', 1)
                return f'<blockquote class="markdown-blockquote">{text}</blockquote>'

            if content.startswith('LIST:'):
                _, list_type, items = content.split(':', 2)
                items_html = []
                for item in items.split(','):
                    if item.strip():
                        items_html.append(f'<li class="markdown-list-item">{item.strip()}</li>')
                return f'<{list_type} class="markdown-list">{" ".join(items_html)}</{list_type}>'

            if content.startswith('CODE:'):
                _, lang, code = content.split(':', 2)
                classes = [f'language-{lang}'] if lang else []
                return (
                    f'<pre><code class="{" ".join(classes)}">'
                    f'{self.html.escape_html(code)}'
                    f'</code></pre>'
                )

            if content.startswith('MERMAID:'):
                _, diagram = content.split(':', 1)
                return f'<div class="mermaid">{diagram}</div>'

            # Default transformation for unmarked content
            return content

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
        """
        Transform heading element from Markdown content.

        Args:
            element_info (MDElementInfo): Information about the Markdown element.

        Returns:
            str: HTML string for the transformed heading.
        """
        try:
            # Extract heading level and generate a unique ID
            level = int(element_info.metadata.get('level', 1))
            heading_id, formatted_text = self.heading_handler.process_heading(
                text=element_info.content,
                level=level,
                is_map_title=False  # Markdown headings are not map titles
            )

            # Generate the HTML for the heading
            return (
                f'<h{level} id="{heading_id}" '
                f'class="{" ".join(element_info.attributes.classes)}">'
                f'{formatted_text}'
                f'<a href="#{heading_id}" class="heading-anchor">¶</a>'
                f'</h{level}>'
            )

        except Exception as e:
            self.logger.error(f"Error transforming heading: {str(e)}", exc_info=True)
            return element_info.content




    def _transform_paragraph(self, element_info: MDElementInfo) -> str:
        """Transform paragraph element."""
        return (
            f'<p class="{" ".join(element_info.attributes.classes)}">'
            f'{element_info.content}'
            f'</p>'
        )

    def _transform_list(self, element_info: MDElementInfo) -> str:
        """Transform list element."""
        items = []
        for line in element_info.content.split('\n'):
            if line.startswith('LINK:'):
                _, href, text = line.split(':', 2)
                items.append(
                    f'<li class="markdown-list-item">'
                    f'<a href="{href}" class="markdown-link">{text}</a>'
                    f'</li>'
                )
            elif line.startswith('TODO:'):
                _, content = line.split(':', 1)
                items.append(
                    f'<li class="markdown-list-item todo-item">'
                    f'<input type="checkbox" {"checked" if "[x]" in content else ""} disabled>'
                    f'{content.replace("[x]", "").replace("[ ]", "").strip()}'
                    f'</li>'
                )
            elif line:
                items.append(f'<li class="markdown-list-item">{line}</li>')

        return f'<ul class="markdown-list">\n{"".join(items)}\n</ul>'

    def _transform_list_item(self, element_info: MDElementInfo) -> str:
        """Transform list item."""
        return (
            f'<li class="markdown-list-item">'
            f'{element_info.content}'
            f'</li>'
        )

    def _transform_code(self, element_info: MDElementInfo) -> str:
        """Transform code block element."""
        code = element_info.content
        lang = element_info.metadata.get('language', '')

        # Add language class only if one is specified
        lang_class = f'language-{lang}' if lang else ''
        classes = [c for c in [lang_class, 'markdown-code'] if c]

        return (
            f'<pre><code class="{" ".join(classes)}">'
            f'{self.html.escape_html(code)}'
            f'</code></pre>'
        )

    def _transform_blockquote(self, element_info: MDElementInfo) -> str:
        """Transform blockquote element."""
        if not element_info.content.startswith('QUOTE:'):
            return ''

        _, content = element_info.content.split(':', 1)
        return (
            f'<blockquote class="markdown-blockquote">'
            f'{content}'
            f'</blockquote>'
        )

    def _transform_link(self, element_info: MDElementInfo) -> str:
        """Transform link element."""
        if not element_info.content.startswith('LINK:'):
            return ''

        _, href, text = element_info.content.split(':', 2)
        return f'<a href="{href}" class="markdown-link">{text}</a>'


    def _transform_image(self, element_info: MDElementInfo) -> str:
        """Transform image element."""
        if not element_info.content.startswith('IMAGE:'):
            return ''

        _, src, alt = element_info.content.split(':', 2)

        # Get the topic directory
        if element_info.context.topic_path:
            topic_dir = element_info.context.topic_path.parent
            # First try media subdirectory
            media_path = (topic_dir / 'media' / src).resolve()
            if media_path.exists():
                src = f'/static/topics/{topic_dir.name}/media/{src}'
            else:
                # Try direct path
                direct_path = (topic_dir / src).resolve()
                if direct_path.exists():
                    src = f'/static/topics/{topic_dir.name}/{src}'

        return (
            f'<figure class="markdown-figure">'
            f'<img src="{src}" alt="{alt}" class="markdown-image" />'
            f'</figure>'
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
            f'<code class="markdown-inline-code">'
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
