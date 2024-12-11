from datetime import datetime
import logging
from pathlib import Path
from html import escape
import html
from bs4 import BeautifulSoup, Tag
from typing import Optional, Dict, Any, Callable, List
from lxml import etree
from ..models.types import (
    DITAElementType,
    DITAElementInfo,
    ParsedElement,
    ProcessedContent,
    ProcessingContext,
    ElementAttributes,
    DITAElementContext,
    HeadingContext
)
from app_config import DITAConfig
from .base_transformer import BaseTransformer
from ..processors.dita_parser import DITAParser
from app.dita.processors.content_processors import ContentProcessor
from ..processors.dita_elements import DITAElementProcessor
from ..utils.html_helpers import HTMLHelper
from ..utils.id_handler import DITAIDHandler
from ..utils.heading import HeadingHandler
from xml.etree import ElementTree
from app.dita.utils import heading


class DITATransformer(BaseTransformer):
    """Transforms DITA content to HTML using the BaseTransformer approach."""

    def __init__(self, dita_root: Path):
        super().__init__(dita_root)
        self.logger = logging.getLogger(__name__)
        self.dita_root = dita_root
        self.dita_parser = DITAParser()
        self.html_helper = HTMLHelper(dita_root)
        self.id_handler = DITAIDHandler()
        self.heading_handler = HeadingHandler()

        # Initialize content processor first
        content_processor = ContentProcessor(
            dita_root=dita_root,
            markdown_root=dita_root
        )

        self.content_processor = DITAElementProcessor(content_processor)

        # Map DITA elements to their transformers
        self._element_transformers: Dict[DITAElementType, Callable[[DITAElementInfo], str]] = {
            DITAElementType.CODE_BLOCK: self._transform_code_block,
            DITAElementType.STEP: self._transform_step,
            DITAElementType.PREREQ: self._transform_prereq,
            DITAElementType.CMD: self._transform_cmd,
            DITAElementType.INFO: self._transform_info,
            DITAElementType.SUBSTEP: self._transform_substep,
            DITAElementType.METADATA: self._transform_metadata,
            DITAElementType.QUOTE: self._transform_quote,
            DITAElementType.CONCEPT: self._transform_concept,
            DITAElementType.TASK: self._transform_task,
            DITAElementType.REFERENCE: self._transform_reference,
            DITAElementType.SECTION: self._transform_section,
            DITAElementType.PARAGRAPH: self._transform_paragraph,
            DITAElementType.NOTE: self._transform_note,
            DITAElementType.TABLE: self._transform_table,
            DITAElementType.LIST: self._transform_list,
            DITAElementType.ORDERED_LIST: self._transform_ordered_list,
            DITAElementType.CODE_PHRASE: self._transform_code_phrase,
            DITAElementType.FIGURE: self._transform_figure,
            DITAElementType.IMAGE: self._transform_image,
            DITAElementType.XREF: self._transform_xref,
            DITAElementType.LINK: self._transform_link,
            DITAElementType.SHORTDESC: self._transform_shortdesc,
            DITAElementType.ABSTRACT: self._transform_abstract,
            DITAElementType.STEPS: self._transform_steps,
            DITAElementType.SUBSTEPS: self._transform_substeps,
            DITAElementType.DEFINITION: self._transform_definition,
            DITAElementType.TERM: self._transform_term,
            DITAElementType.TASKBODY: self._transform_taskbody
        }

    def configure(self, config: DITAConfig):
            """
            Apply additional configuration settings.
            """
            self.logger.debug("Configuring DITATransformer")
            # Example: Add custom configuration logic
            # e.g., setting up specific extensions or paths
            self.some_configured_attribute = getattr(config, 'some_attribute', None)

    def transform_topic(
            self,
            parsed_element: ParsedElement,
            context: ProcessingContext,
            html_converter: Optional[Callable[[str, ProcessingContext], str]] = None
        ) -> ProcessedContent:
            try:
                # Let parser handle XML parsing
                xml_tree = self.dita_parser.parse_xml_content(parsed_element.content)

                # Let content processor handle element processing
                processed_elements = self.content_processor.process_elements(xml_tree)

                # Transform to HTML (our only concern)
                html_content = self._convert_to_html(processed_elements, context)

                return ProcessedContent(
                    html=html_content,
                    element_id=parsed_element.id,
                    metadata=parsed_element.metadata
                )
            except Exception as e:
                self.logger.error(f"Error transforming DITA topic: {str(e)}")
                raise


    def _transform_step(self, element_info: DITAElementInfo) -> str:
        """Transform a step element."""
        content = html.escape(element_info.content or "")
        step_number = element_info.attributes.get('step-number', '1')
        return f'<div class="step"><h3 class="step-title">{step_number}. {content}</h3>'

    def _transform_substep(self, element_info: DITAElementInfo) -> str:
        """Transform a substep element."""
        content = html.escape(element_info.content or "")
        substep_num = element_info.attributes.get('substep-number')
        try:
            num = int(substep_num) if substep_num is not None else 0
            substep_letter = chr(ord('a') + num)
        except (ValueError, TypeError):
            substep_letter = 'a'
        return f'<div class="substep"><p class="substep-title">{substep_letter}) {content}</p>'

    def _transform_info(self, element_info: DITAElementInfo) -> str:
        """Transform an info element."""
        content = html.escape(element_info.content or "")
        return f'<div class="info-block alert alert-info">{content}</div>'

    def _transform_cmd(self, element_info: DITAElementInfo) -> str:
        """Transform a command element."""
        content = html.escape(element_info.content or "")
        return f'<h4 class="cmd-title">{content}</h4>'

    def _transform_prereq(self, element_info: DITAElementInfo) -> str:
        """Transform a prerequisite element."""
        content = html.escape(element_info.content or "")
        return f'<div class="prereq-block alert alert-warning">{content}</div>'

    def _transform_steps_container(self, _: DITAElementInfo) -> str:
        """Transform a steps container element."""
        return '<div class="steps-container">'

    def _transform_substeps_container(self, _: DITAElementInfo) -> str:
        """Transform a substeps container element."""
        return '<div class="substeps-container ms-4">'

    def _transform_unordered_list(self, element_info: DITAElementInfo) -> str:
        """Transform an unordered list element."""
        content = html.escape(element_info.content or "")
        return f'<ul class="task-list">{content}</ul>'

    def _transform_list_item(self, element_info: DITAElementInfo) -> str:
        """Transform a list item element."""
        content = html.escape(element_info.content or "")
        return f'<li class="task-list-item">• {content}</li>'


    def _transform_code_block(self, element_info: DITAElementInfo) -> str:
        """Transform DITA code blocks with consistent styling."""
        try:
            if element_info.type != DITAElementType.CODE_BLOCK:
                return ""

            language = element_info.attributes.get('outputclass', '')
            content = html.escape(element_info.content or "")

            language_label = f'<div class="code-label">{language}</div>' if language else ''

            return (
                f'<pre class="code-block" data-language="{language}">'
                f'{language_label}'
                f'<code class="language-{language}">{content}</code>'
                '</pre>'
            )
        except Exception as e:
            self.logger.error(f"Error transforming code block: {str(e)}")
            return ""

    def _transform_metadata(self, metadata_elem: Any) -> str:
        """Transform metadata into an HTML table."""
        try:
            rows = []
            rows.append("<table class='metadata-table'>")
            rows.append("<thead><tr><th>Property</th><th>Value</th></tr></thead>")
            rows.append("<tbody>")

            for child in metadata_elem:
                if isinstance(child.tag, str):
                    name = child.tag
                    value = child.text or child.get('content', '')
                    rows.append(f"<tr><td>{name}</td><td>{value}</td></tr>")

            rows.append("</tbody></table>")
            return "\n".join(rows)
        except Exception as e:
            self.logger.error(f"Error transforming metadata: {str(e)}")
            return ""



    def _create_html_element(
            self,
            soup: BeautifulSoup,
            element_info: DITAElementInfo
        ) -> Optional[Tag]:
            """
            Create HTML element from DITA element info.

            Args:
                soup: BeautifulSoup instance for creating new tags
                element_info: Processed DITA element information

            Returns:
                Optional[Tag]: Created HTML element or None if creation fails
            """
            try:
                # Get HTML tag name from mapping
                tag_name = self._element_mappings.get(element_info.type, 'div')
                new_elem = soup.new_tag(tag_name)

                # Apply attributes
                if element_info.attributes.id:
                    new_elem['id'] = element_info.attributes.id
                if element_info.attributes.classes:
                    new_elem['class'] = ' '.join(element_info.attributes.classes)
                for key, value in element_info.attributes.custom_attrs.items():
                    new_elem[key] = value

                # Add content
                if element_info.content:
                    new_elem.string = element_info.content

                # Process children if any
                if element_info.children:
                    for child in element_info.children:
                        child_elem = self._create_html_element(soup, child)
                        if child_elem:
                            new_elem.append(child_elem)

                return new_elem

            except Exception as e:
                self.logger.error(f"Error creating HTML element: {str(e)}")
                return None
