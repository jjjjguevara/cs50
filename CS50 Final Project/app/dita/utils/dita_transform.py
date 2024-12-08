# app/dita/utils/dita_transform.py

import logging
from pathlib import Path
from typing import Any, Dict
from bs4 import BeautifulSoup
from lxml import etree

# Global config
from config import DITAConfig

from .types import (
    DITAElementType,
    DITAElementInfo,
    DITAElementContext,
    ElementAttributes,
    ProcessedContent,
    ProcessingError,
    ProcessingState
)
from .metadata import MetadataHandler
from .id_handler import DITAIDHandler
from .heading import HeadingHandler
from .html_helpers import HTMLHelper

class DITATransformer:
    """Transforms DITA content to HTML using element definitions."""

    def __init__(self, dita_root: Path):
        self.logger = logging.getLogger(__name__)
        self.id_handler = DITAIDHandler()
        self.metadata_handler = MetadataHandler()
        self.heading_handler = HeadingHandler()
        self.html = HTMLHelper(dita_root)
        self.dita_root = dita_root

    def configure(self, config: DITAConfig) -> None:
        """Configure element processor with provided settings."""
        try:
            self.logger.debug("Configuring DITA element processor")

            # Configure paths if needed
            if hasattr(config.paths, 'dita_root'):
                self.dita_root = config.paths.dita_root

            # Configure ID handler
            if hasattr(self.id_handler, 'configure'):
                self.id_handler.configure(config)

            self.logger.debug("DITA element processor configuration completed")

        except Exception as e:
            self.logger.error(f"DITA element processor configuration failed: {str(e)}")
            raise

    def _get_transform_method(self, element_type: DITAElementType):
        """Get appropriate transformation method for element type."""
        transform_methods = {
            DITAElementType.CONCEPT: self._transform_concept,
            DITAElementType.TASK: self._transform_task,
            DITAElementType.REFERENCE: self._transform_reference,
            DITAElementType.TOPIC: self._transform_topic_element,
            DITAElementType.SECTION: self._transform_section,
            DITAElementType.PARAGRAPH: self._transform_paragraph,
            DITAElementType.NOTE: self._transform_note,
            DITAElementType.TABLE: self._transform_table,
            DITAElementType.LIST: self._transform_list,
            DITAElementType.ORDERED_LIST: self._transform_ordered_list,
            DITAElementType.LIST_ITEM: self._transform_list_item,
            DITAElementType.CODE_BLOCK: self._transform_code_block,
            DITAElementType.CODE_PHRASE: self._transform_code_phrase,
            DITAElementType.FIGURE: self._transform_figure,
            DITAElementType.IMAGE: self._transform_image,
            DITAElementType.XREF: self._transform_xref,
            DITAElementType.LINK: self._transform_link,
            DITAElementType.TITLE: self._transform_title,
            DITAElementType.SHORTDESC: self._transform_shortdesc,
            DITAElementType.ABSTRACT: self._transform_abstract,
            DITAElementType.PREREQ: self._transform_prereq,
            DITAElementType.STEPS: self._transform_steps,
            DITAElementType.BOLD: self._transform_bold,
            DITAElementType.ITALIC: self._transform_italic,
            DITAElementType.UNDERLINE: self._transform_underline,
            DITAElementType.PHRASE: self._transform_phrase,
            DITAElementType.QUOTE: self._transform_quote
        }
        return transform_methods.get(element_type)


    def transform_topic(self, topic_path: Path) -> ProcessedContent:
        """
        Transforms a DITA topic into HTML.

        Args:
            topic_path (Path): Path to the DITA topic file.

        Returns:
            ProcessedContent: The transformed HTML content with metadata.
        """
        try:
            # Read the DITA file
            with open(topic_path, 'r', encoding='utf-8') as file:
                dita_content = file.read()

            # Extract metadata
            metadata = self.metadata_handler.extract_metadata(
                file_path=topic_path,
                content_id=topic_path.stem
            )

            # Parse the DITA content to XML
            soup = BeautifulSoup(dita_content, 'xml')

            # Process headings
            for heading in soup.find_all(['title', 'topic']):
                is_map_title = heading.name == "title"
                heading_id, processed_heading = self.heading_handler.process_heading(
                    text=heading.string or '',
                    level=1 if is_map_title else 2,  # Map title = H1; topics = H2 by default
                    is_map_title=is_map_title
                )
                heading['id'] = heading_id

            # Finalize HTML transformation
            html_content = self.html.process_final_content(str(soup))

            return ProcessedContent(
                html=html_content,
                element_id=self.id_handler.generate_id(topic_path.name),
                metadata=metadata
            )

        except Exception as e:
            self.logger.error(f"Error transforming DITA topic {topic_path}: {str(e)}", exc_info=True)
            raise


    def _transform_element(self, element_info: DITAElementInfo) -> str:
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


    # Topic-level transformations
    def _transform_concept(self, element_info: DITAElementInfo) -> str:
        """Transform concept topic."""
        return self.html.wrap_content(
            element_info.content,
            'div',
            ['dita-concept']
        )

    def _transform_task(self, element_info: DITAElementInfo) -> str:
        """Transform task topic."""
        return self.html.wrap_content(
            element_info.content,
            'div',
            ['dita-task']
        )

    def _transform_reference(self, element_info: DITAElementInfo) -> str:
        """Transform reference topic."""
        return self.html.wrap_content(
            element_info.content,
            'div',
            ['dita-reference']
        )

    def _transform_topic_element(self, element_info: DITAElementInfo) -> str:
        """Transform generic topic."""
        return self.html.wrap_content(
            element_info.content,
            'div',
            ['dita-topic']
        )

    # Section-level transformations
    def _transform_section(self, element_info: DITAElementInfo) -> str:
        """Transform section."""
        return self.html.wrap_content(
            element_info.content,
            'section',
            ['dita-section']
        )

    def _transform_title(self, element_info: DITAElementInfo) -> str:
        """Transform title."""
        level = element_info.metadata.get('level', 1)
        return (
            f'<h{level} id="{element_info.attributes.id}" '
            f'class="dita-title {" ".join(element_info.attributes.classes)}">'
            f'{element_info.content}'
            f'<a href="#{element_info.attributes.id}" class="heading-anchor">¶</a>'
            f'</h{level}>'
        )

    # Block-level transformations
    def _transform_paragraph(self, element_info: DITAElementInfo) -> str:
        """Transform paragraph."""
        return (
            f'<p class="dita-p {" ".join(element_info.attributes.classes)}">'
            f'{element_info.content}'
            f'</p>'
        )

    def _transform_note(self, element_info: DITAElementInfo) -> str:
        """Transform note."""
        note_type = element_info.metadata.get('type', 'note')
        return (
            f'<div class="dita-note note-{note_type} '
            f'{" ".join(element_info.attributes.classes)}">'
            f'{element_info.content}'
            f'</div>'
        )

    # List transformations
    def _transform_list(self, element_info: DITAElementInfo) -> str:
        """Transform unordered list."""
        return (
            f'<ul class="dita-ul {" ".join(element_info.attributes.classes)}">'
            f'{element_info.content}'
            f'</ul>'
        )

    def _transform_ordered_list(self, element_info: DITAElementInfo) -> str:
        """Transform ordered list."""
        return (
            f'<ol class="dita-ol {" ".join(element_info.attributes.classes)}">'
            f'{element_info.content}'
            f'</ol>'
        )

    def _transform_list_item(self, element_info: DITAElementInfo) -> str:
        """Transform list item."""
        return (
            f'<li class="dita-li {" ".join(element_info.attributes.classes)}">'
            f'{element_info.content}'
            f'</li>'
        )

    # Code transformations
    def _transform_code_block(self, element_info: DITAElementInfo) -> str:
        """Transform code block."""
        return (
            f'<pre><code class="dita-codeblock '
            f'{" ".join(element_info.attributes.classes)}">'
            f'{self.html.escape_html(element_info.content)}'
            f'</code></pre>'
        )

    def _transform_code_phrase(self, element_info: DITAElementInfo) -> str:
        """Transform inline code."""
        return (
            f'<code class="dita-codeph {" ".join(element_info.attributes.classes)}">'
            f'{self.html.escape_html(element_info.content)}'
            f'</code>'
        )

    # Table transformations

    def _transform_table(self, element_info: DITAElementInfo) -> str:
        """Transform table element."""
        try:
            classes = ['dita-table', *element_info.attributes.classes]
            frame = element_info.metadata.get('frame', 'all')
            if frame != 'none':
                classes.append(f'frame-{frame}')

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

    def _transform_shortdesc(self, element_info: DITAElementInfo) -> str:
        """Transform short description."""
        return (
            f'<p class="dita-shortdesc {" ".join(element_info.attributes.classes)}">'
            f'{element_info.content}'
            f'</p>'
        )

    def _transform_abstract(self, element_info: DITAElementInfo) -> str:
        """Transform abstract."""
        return (
            f'<div class="dita-abstract {" ".join(element_info.attributes.classes)}">'
            f'<div class="abstract-title">Abstract</div>'
            f'{element_info.content}'
            f'</div>'
        )

    def _transform_prereq(self, element_info: DITAElementInfo) -> str:
        """Transform prerequisites."""
        return (
            f'<div class="dita-prereq {" ".join(element_info.attributes.classes)}">'
            f'<div class="prereq-title">Prerequisites</div>'
            f'{element_info.content}'
            f'</div>'
        )

    def _transform_steps(self, element_info: DITAElementInfo) -> str:
        """Transform steps element."""
        return (
            f'<div class="dita-steps {" ".join(element_info.attributes.classes)}">'
            f'<div class="steps-title">Steps</div>'
            f'<ol class="steps-list">'
            f'{element_info.content}'
            f'</ol>'
            f'</div>'
        )

    # Media transformations
    def _transform_figure(self, element_info: DITAElementInfo) -> str:
        """Transform figure."""
        return (
            f'<figure class="dita-fig {" ".join(element_info.attributes.classes)}">'
            f'{element_info.content}'
            f'</figure>'
        )

    def _transform_image(self, element_info: DITAElementInfo) -> str:
        """Transform image element."""
        href = element_info.metadata.get('href', '')
        alt = element_info.metadata.get('alt', '')

        # Resolve image path if topic path is available
        if href and element_info.context.topic_path is not None:
            href = self.html.resolve_image_path(href, element_info.context.topic_path)

        return (
            f'<img src="{href}" alt="{alt}" '
            f'class="dita-image {" ".join(element_info.attributes.classes)}">'
        )

    # Link transformations
    def _transform_xref(self, element_info: DITAElementInfo) -> str:
        """Transform cross-reference."""
        href = element_info.metadata.get('href', '#')
        return (
            f'<a href="{href}" class="dita-xref '
            f'{" ".join(element_info.attributes.classes)}">'
            f'{element_info.content}</a>'
        )

    def _transform_link(self, element_info: DITAElementInfo) -> str:
        """Transform link."""
        href = element_info.metadata.get('href', '#')
        scope = element_info.metadata.get('scope', 'local')
        classes = ['dita-link']
        if scope == 'external':
            classes.append('external-link')
        return (
            f'<a href="{href}" class="{" ".join(classes)} '
            f'{" ".join(element_info.attributes.classes)}">'
            f'{element_info.content}</a>'
        )

    # Inline transformations
    def _transform_bold(self, element_info: DITAElementInfo) -> str:
        """Transform bold text."""
        return f'<strong>{element_info.content}</strong>'

    def _transform_italic(self, element_info: DITAElementInfo) -> str:
        """Transform italic text."""
        return f'<em>{element_info.content}</em>'

    def _transform_underline(self, element_info: DITAElementInfo) -> str:
        """Transform underlined text."""
        return (
            f'<span class="underline {" ".join(element_info.attributes.classes)}">'
            f'{element_info.content}</span>'
        )

    def _transform_phrase(self, element_info: DITAElementInfo) -> str:
        """Transform phrase."""
        return (
            f'<span class="dita-ph {" ".join(element_info.attributes.classes)}">'
            f'{element_info.content}</span>'
        )

    def _transform_quote(self, element_info: DITAElementInfo) -> str:
        """Transform quote."""
        return (
            f'<q class="dita-quote {" ".join(element_info.attributes.classes)}">'
            f'{element_info.content}</q>'
        )

    def _transform_default(self, element_info: DITAElementInfo) -> str:
        """Default transformation for unknown elements."""
        return element_info.content


    # Cleanup

    def cleanup(self) -> None:
        """Clean up transformer resources and state."""
        try:
            self.logger.debug("Starting DITA transformer cleanup")

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

            self.logger.debug("DITA transformer cleanup completed")

        except Exception as e:
            self.logger.error(f"DITA transformer cleanup failed: {str(e)}")
            raise

    def reset(self) -> None:
        """Reset transformer to initial state."""
        try:
            self.logger.debug("Resetting DITA transformer")

            # Reset element processor
            self.element_processor = self.__class__(self.dita_root)

            # Reset HTML helper
            self.html = HTMLHelper(self.dita_root)

            # Reset ID handler
            self.id_handler = DITAIDHandler()

            # Reset Metadata handler
            self.metadata_handler = MetadataHandler()

            # Reset Heading handler
            self.heading_handler = HeadingHandler()

            self.logger.debug("DITA transformer reset completed successfully")

        except Exception as e:
            self.logger.error(f"DITA transformer reset failed: {str(e)}")
            raise
