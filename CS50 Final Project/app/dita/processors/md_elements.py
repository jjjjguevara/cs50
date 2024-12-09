# app/dita/utils/markdown/md_elements.py

from enum import Enum
from typing import Dict, Optional, Union, List, Any
from pathlib import Path
import logging, html
from bs4 import BeautifulSoup, NavigableString, Tag
import re


from app.dita.utils.id_handler import DITAIDHandler
from app.dita.models.types import MDElementInfo, MDElementContext, ElementAttributes, MDElementType


class MarkdownContentProcessor:
    """Defines Markdown element structures and attributes."""

    def __init__(self):
            self.logger = logging.getLogger(__name__)
            self.id_handler = DITAIDHandler()
            self._processed_elements: Dict[str, Any] = {}

    def process_element(self, elem: Tag, source_path: Optional[Path] = None) -> MDElementInfo:
        """
        Process a Markdown element into structured information.

        Args:
            elem: BeautifulSoup Tag representing the element.
            source_path: Path to the source Markdown file.

        Returns:
            MDElementInfo object containing the element's details.
        """
        try:
            if not isinstance(elem, Tag):
                return self._default_element()

            element_type = self._get_element_type(elem)
            context = MDElementContext(
                parent_id=None,
                element_type=element_type.value,
                classes=self._get_element_classes(elem),
                attributes=self._get_element_attributes(elem),
            )
            attributes = ElementAttributes(
                id=self.id_handler.generate_content_id(Path(str(elem.get('id', '')))),
                classes=self._get_element_classes(elem),
                custom_attrs=self._get_element_attributes(elem),
            )

            return MDElementInfo(
                type=element_type,
                content=self._get_element_content(elem),
                attributes=attributes,
                context=context,
                metadata=self._get_element_metadata(elem),
            )
        except Exception as e:
            self.logger.error(f"Error processing Markdown element: {str(e)}")
            return self._default_element()

    def _get_element_type(self, elem: Tag) -> MDElementType:
        """Determine the type of Markdown element based on its tag."""
        tag_mapping = {
            'h1': MDElementType.HEADING,
            'h2': MDElementType.HEADING,
            'h3': MDElementType.HEADING,
            'h4': MDElementType.HEADING,
            'h5': MDElementType.HEADING,
            'h6': MDElementType.HEADING,
            'p': MDElementType.PARAGRAPH,
            'ul': MDElementType.UNORDERED_LIST,
            'ol': MDElementType.ORDERED_LIST,
            'li': MDElementType.LIST_ITEM,
            'code': MDElementType.INLINE_CODE,
            'pre': MDElementType.CODE_BLOCK,
            'blockquote': MDElementType.BLOCKQUOTE,
            'a': MDElementType.LINK,
            'img': MDElementType.IMAGE,
            'table': MDElementType.TABLE,
            'th': MDElementType.TABLE_HEADER,
            'tr': MDElementType.TABLE_ROW,
            'td': MDElementType.TABLE_CELL,
            'strong': MDElementType.BOLD,
            'em': MDElementType.ITALIC,
        }
        return tag_mapping.get(elem.name, MDElementType.UNKNOWN)

    def _get_element_content(self, elem: Union[Tag, NavigableString]) -> str:
        """Extract the raw content of the element."""
        if isinstance(elem, NavigableString):
            return elem.strip()
        if isinstance(elem, Tag) and elem.string:
            return elem.string.strip()
        return elem.get_text(strip=True)

    def _get_inline_code_content(self, element: dict) -> str:
        """
        Extract and process inline code content from a Markdown element.

        Args:
            element (dict): Parsed Markdown element.

        Returns:
            str: HTML representation of the inline code.
        """
        try:
            code_content = element.get('text', '')
            return f'<code>{html.escape(code_content)}</code>'
        except Exception as e:
            self.logger.error(f"Error processing inline code content: {str(e)}")
            return ''



    def _get_link_content(self, elem: Tag) -> str:
        """Process link content."""
        # Skip processing if this link is part of a li or p
        if elem.find_parent(['li', 'p', 'blockquote']):
            return ''
        href = elem.get('href', '')
        text = elem.get_text(strip=True)
        return text  # Just return text, href is handled in metadata

    def _get_paragraph_content(self, elem: Tag) -> str:
        """Process paragraph content with potential links."""
        parts = []
        for child in elem.children:
            if isinstance(child, Tag) and child.name == 'a':
                href = child.get('href', '')
                text = child.get_text(strip=True)
                parts.append(f"LINK:{href}:{text}")
            else:
                parts.append(str(child).strip())
        return ' '.join(filter(None, parts))

    def _get_blockquote_content(self, elem: Tag) -> str:
        """Process blockquote content."""
        parts = []
        for child in elem.children:
            if isinstance(child, Tag) and child.name == 'a':
                href = child.get('href', '')
                text = child.get_text(strip=True)
                parts.append(f"LINK:{href}:{text}")
            else:
                parts.append(str(child).strip())
        return f"QUOTE:{' '.join(filter(None, parts))}"

    def _get_list_content(self, elem: Tag) -> str:
        """Process list content (ul/ol)."""
        items = []
        for li in elem.find_all('li', recursive=False):
            items.append(self._get_list_item_content(li))
        return f"LIST:{elem.name}:{','.join(items)}"

    def _get_list_item_content(self, elem: Tag) -> str:
        """Process list item content."""
        content = []
        for child in elem.children:
            if isinstance(child, Tag) and child.name == 'a':
                href = child.get('href', '')
                text = child.get_text(strip=True)
                content.append(f"LINK:{href}:{text}")
            else:
                content.append(str(child).strip())
        return ' '.join(filter(None, content))

    def _get_code_content(self, elem: Tag) -> str:
        """Process code block content."""
        code = elem.find('code')
        if code and isinstance(code, Tag):
            classes = code.get('class', [])
            if 'language-mermaid' in classes:
                return f"MERMAID:{code.get_text(strip=True)}"
            lang = next((c.replace('language-', '') for c in classes if c.startswith('language-')), '')
            return f"CODE:{lang}:{code.get_text()}"
        return f"CODE::{elem.get_text()}"

    def _get_image_content(self, elem: Tag) -> str:
        """Process image content."""
        src = elem.get('src', '')
        alt = elem.get('alt', '')
        return f"IMAGE:{src}:{alt}"

    def _get_element_attributes(self, elem: Tag) -> Dict[str, str]:
            """Extract attributes of the element."""
            return {k: v for k, v in elem.attrs.items() if isinstance(v, str)}

    def _get_element_classes(self, elem: Tag) -> List[str]:
        """Extract class names from the element as a list of strings."""
        classes = elem.get('class', [])
        if isinstance(classes, str):
            # If a single class is provided as a string, wrap it in a list
            return [classes]
        return classes if isinstance(classes, list) else []

    def _get_element_metadata(self, elem: Tag) -> Dict[str, Any]:
        """Extract metadata for the element, if available."""
        metadata = {}
        raw_metadata = elem.attrs.get('data-metadata')
        if raw_metadata:
            try:
                # Parse the metadata string into key-value pairs
                for item in raw_metadata.split(';'):
                    if '=' in item:
                        key, value = item.split('=', 1)
                        metadata[key.strip()] = value.strip()
            except Exception as e:
                self.logger.error(f"Error parsing metadata for element: {e}")
        return metadata

    def _default_element(self) -> MDElementInfo:
        """Return a default Markdown element for unprocessable cases."""
        return MDElementInfo(
            type=MDElementType.PARAGRAPH,
            content="",
            attributes=ElementAttributes(id="", classes=[], custom_attrs={}),
            context=MDElementContext(
                parent_id=None, element_type="unknown", classes=[], attributes={}
            ),
            metadata={},
        )

    def _get_code_language(self, elem: Tag) -> Optional[str]:
        """Get code block language."""
        try:
            classes = self._get_element_classes(elem)
            for cls in classes:
                if cls.startswith('language-'):
                    return cls.replace('language-', '')
            return None
        except Exception as e:
            self.logger.error(f"Error getting code language: {str(e)}")
            return None

    def _validate_element(self, elem: Tag) -> bool:
        """Validate element structure."""
        try:
            if not isinstance(elem, Tag):
                return False

            # Validate based on element type
            element_type = self._get_element_type(elem)

            if element_type == MDElementType.HEADING:
                return self._validate_heading(elem)
            elif element_type == MDElementType.LINK:
                return self._validate_link(elem)
            elif element_type == MDElementType.IMAGE:
                return self._validate_image(elem)

            return True

        except Exception as e:
            self.logger.error(f"Error validating element: {str(e)}")
            return False

    def _validate_heading(self, elem: Tag) -> bool:
        """Validate heading element."""
        try:
            level = int(elem.name[1])
            return 1 <= level <= 6 and bool(elem.get_text().strip())
        except Exception:
            return False

    def _validate_link(self, elem: Tag) -> bool:
        """Validate link element."""
        return bool(elem.get('href'))

    def _validate_image(self, elem: Tag) -> bool:
        """Validate image element."""
        return bool(elem.get('src'))


    # Cleanup

    def cleanup(self) -> None:
            """Clean up processor resources and state."""
            try:
                self.logger.debug("Starting Markdown element processor cleanup")

                # Reset ID handler
                self.id_handler = DITAIDHandler()

                # Clear processed elements cache
                self._processed_elements.clear()

                self.logger.debug("Markdown element processor cleanup completed")

            except Exception as e:
                self.logger.error(f"Markdown element processor cleanup failed: {str(e)}")
                raise
