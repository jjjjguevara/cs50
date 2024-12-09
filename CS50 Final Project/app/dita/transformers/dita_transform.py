import logging
from pathlib import Path
from html import escape
import html
from typing import Optional, Any

from lxml import etree
from ..models.types import (
    DITAElementType,
    DITAElementInfo,
    ParsedElement,
    ProcessedContent,
    ElementAttributes,
    DITAElementContext,
    HeadingContext
)
from app_config import DITAConfig
from ..processors.dita_elements import DITAContentProcessor
from ..utils.html_helpers import HTMLHelper
from ..utils.id_handler import DITAIDHandler
from ..utils.heading import HeadingHandler
from xml.etree import ElementTree
from app.dita.utils import heading


class DITATransformer:
    """Transforms DITA content to HTML using the BaseTransformer approach."""

    def __init__(self, dita_root: Path):
        self.logger = logging.getLogger(__name__)
        self.dita_root = dita_root
        self.html_helper = HTMLHelper(dita_root)
        self.id_handler = DITAIDHandler()
        self.content_processor = DITAContentProcessor()
        self.heading_handler = HeadingHandler()
        self._element_transformers = {
            DITAElementType.STEP: self._transform_step,
            DITAElementType.SUBSTEP: self._transform_substep,
            DITAElementType.INFO: self._transform_info,
            DITAElementType.CMD: self._transform_cmd,
            DITAElementType.PREREQ: self._transform_prereq,
            DITAElementType.STEPS: self._transform_steps_container,
            DITAElementType.SUBSTEPS: self._transform_substeps_container,
            DITAElementType.UL: self._transform_unordered_list,
            DITAElementType.LI: self._transform_list_item,
            DITAElementType.CODE_BLOCK: self._transform_code_block,
            DITAElementType.IMAGE: self._process_images,
        }

    def configure(self, config: DITAConfig):
            """
            Apply additional configuration settings.
            """
            self.logger.debug("Configuring DITATransformer")
            # Example: Add custom configuration logic
            # e.g., setting up specific extensions or paths
            self.some_configured_attribute = getattr(config, 'some_attribute', None)

    def transform_topic(self, parsed_element: ParsedElement) -> ProcessedContent:
        try:
            self.logger.debug(f"Transforming DITA topic: {parsed_element.topic_path}")

            # Start new topic
            self.heading_handler.start_new_topic()

            if not parsed_element.content:
                raise ValueError(f"Empty content in topic {parsed_element.topic_path}.")

            # Parse XML content
            content_bytes = parsed_element.content.encode('utf-8') if isinstance(parsed_element.content, str) else parsed_element.content
            parser = etree.XMLParser(remove_blank_text=True)
            xml_tree = etree.fromstring(content_bytes, parser=parser)

            # Process elements
            html_parts = []

            # Process metadata if present
            metadata_elem = xml_tree.find(".//metadata")
            if metadata_elem is not None:
                metadata_html = self._transform_metadata(metadata_elem)
                if metadata_html:
                    html_parts.append(metadata_html)

            # Process main content
            for elem in xml_tree.iter():
                if isinstance(elem.tag, str):
                    element_info = self.content_processor.process_element(elem)

                    if element_info.type == DITAElementType.TITLE:
                        parent = elem.getparent()
                        is_topic_title = parent is not None and parent.tag in ['concept', 'task', 'reference']
                        level = 1 if is_topic_title else 2

                        heading_id, numbered_heading = self.heading_handler.process_heading(
                            text=element_info.content.strip(),
                            level=level,
                            is_topic_title=is_topic_title
                        )

                        html_parts.append(
                            f'<h{level} id="{heading_id}">{numbered_heading}'
                            f'<a href="#{heading_id}" class="pilcrow">¶</a></h{level}>'
                        )
                    else:
                        html_parts.append(self._transform_element(element_info))

            # Combine parts and wrap in container
            final_html = f'<div class="dita-content">{"".join(html_parts)}</div>'

            return ProcessedContent(
                html=final_html,
                element_id=self.id_handler.generate_content_id(parsed_element.topic_path),
                metadata=parsed_element.metadata
            )
        except Exception as e:
            self.logger.error(f"Error transforming DITA topic: {str(e)}")
            return ProcessedContent(
                html=f"<div class='error'>Error processing topic {parsed_element.topic_path}</div>",
                element_id="",
                metadata=parsed_element.metadata
            )


    def _process_elements(self, xml_tree: Any) -> str:
        try:
            html_parts = []

            # Start new topic and handle metadata first
            self.heading_handler.start_new_topic()

            metadata_elem = xml_tree.find(".//metadata")
            if metadata_elem is not None:
                metadata_html = self._transform_metadata(metadata_elem)
                if metadata_html:
                    html_parts.append(metadata_html)

            # Process main title first
            title_elem = xml_tree.find(".//title")
            if title_elem is not None:
                heading_id, numbered_heading = self.heading_handler.process_heading(
                    text=title_elem.text.strip(),
                    level=1,
                    is_topic_title=True
                )
                html_parts.append(
                    f'<h1 id="{heading_id}">{numbered_heading}<a href="#{heading_id}" class="pilcrow">¶</a></h1>'
                )

            # Process remaining elements
            for elem in xml_tree.iter():
                if isinstance(elem.tag, str) and elem != title_elem:
                    element_info = self.content_processor.process_element(elem)

                    # Handle section titles
                    if element_info.type == DITAElementType.TITLE and elem.getparent().tag == 'section':
                        heading_id, numbered_heading = self.heading_handler.process_heading(
                            text=element_info.content.strip(),
                            level=2,
                            is_topic_title=False
                        )
                        html_parts.append(
                            f'<h2 id="{heading_id}">{numbered_heading}<a href="#{heading_id}" class="pilcrow">¶</a></h2>'
                        )
                        continue

                    # Handle other elements
                    html_parts.append(self._transform_element(element_info))

            return "".join(filter(None, html_parts))
        except Exception as e:
            self.logger.error(f"Error processing elements: {str(e)}")
            return ""

    def _transform_element(self, element_info: DITAElementInfo) -> str:
            """Transform DITA elements into HTML with proper formatting."""
            try:
                # Get the appropriate transformer function
                transformer = self._element_transformers.get(element_info.type)
                if transformer:
                    return transformer(element_info)

                # Default transformation for unknown elements
                content = html.escape(element_info.content or "")
                return f'<div>{content}</div>'

            except Exception as e:
                self.logger.error(f"Error transforming element: {str(e)}")
                return ""

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

    def _process_images(self, element_info: DITAElementInfo) -> str:
        """Process image elements."""
        try:
            if element_info.type != DITAElementType.IMAGE:
                return ""

            src = element_info.attributes.get('href')
            if not src:  # Handle missing src
                self.logger.warning("Image element missing 'href' attribute")
                return ""

            alt = element_info.attributes.get('alt', '')

            # Handle relative paths
            if not any(src.startswith(prefix) for prefix in ('http://', 'https://')):
                src = f"/static/dita/topics/{src}"

            return (
                f'<figure class="image-container">'
                f'<img src="{src}" alt="{alt}" class="img-fluid">'
                f'<figcaption>{alt}</figcaption>'
                f'</figure>'
            )
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return ""

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
