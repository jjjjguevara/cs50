# app/dita/utils/markdown/md_elements.py

from enum import Enum
from typing import Dict, Optional, Union, List, Any
from pathlib import Path
import logging, html
from bs4 import BeautifulSoup, NavigableString, Tag
import re


from ..id_handler import DITAIDHandler
from ..types import MDElementInfo, MDElementContext, ElementAttributes, MDElementType


class MarkdownContentProcessor:
    """Defines Markdown element structures and attributes."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.id_handler = DITAIDHandler()
        self._processed_elements: Dict[str, Any] = {}

    def process_element(self, elem: Tag, source_path: Optional[Path] = None) -> MDElementInfo:
        """Process Markdown element into structured info."""
        try:
            if not isinstance(elem, Tag):
                # Return empty/default MDElementInfo instead of None
                return MDElementInfo(
                    type=MDElementType.PARAGRAPH,  # Default type
                    content="",
                    attributes=ElementAttributes(
                        id="",
                        classes=[],
                        custom_attrs={}
                    ),
                    context=MDElementContext(
                        parent_id=None,
                        element_type="paragraph",
                        classes=[],
                        attributes={}
                    ),
                    metadata={}
                )

            # Get element type
            element_type = self._get_element_type(elem)

            # Get context
            context = MDElementContext(
                parent_id=None,  # Set by transformer
                element_type=element_type.value,
                classes=self._get_element_classes(elem),
                attributes=self._get_element_attributes(elem)
            )

            # Get base attributes
            attributes = ElementAttributes(
                id=self.id_handler.generate_content_id(Path(str(elem.get('id', '')))),
                classes=self._get_element_classes(elem),
                custom_attrs=self._get_element_attributes(elem)
            )

            # Create element info
            return MDElementInfo(
                type=element_type,
                content=self._get_element_content(elem),
                attributes=attributes,
                context=context,
                metadata=self._get_element_metadata(elem)
            )

        except Exception as e:
            self.logger.error(f"Error processing element: {str(e)}")
            # Return empty/default MDElementInfo instead of None
            return MDElementInfo(
                type=MDElementType.PARAGRAPH,
                content="",
                attributes=ElementAttributes(
                    id="",
                    classes=[],
                    custom_attrs={}
                ),
                context=MDElementContext(
                    parent_id=None,
                    element_type="paragraph",
                    classes=[],
                    attributes={}
                ),
                metadata={'error': str(e)}
            )

    def _get_element_type(self, elem: Tag) -> MDElementType:
        """Determine element type."""
        tag_mapping = {
            'h1': MDElementType.HEADING,
            'h2': MDElementType.HEADING,
            'h3': MDElementType.HEADING,
            'h4': MDElementType.HEADING,
            'h5': MDElementType.HEADING,
            'h6': MDElementType.HEADING,
            'p': MDElementType.PARAGRAPH,
            'ul': MDElementType.LIST,
            'ol': MDElementType.LIST,
            'li': MDElementType.LIST_ITEM,
            'pre': MDElementType.CODE,
            'code': MDElementType.INLINE_CODE,
            'blockquote': MDElementType.BLOCKQUOTE,
            'a': MDElementType.LINK,
            'img': MDElementType.IMAGE,
            'table': MDElementType.TABLE,
            'em': MDElementType.EMPHASIS,
            'strong': MDElementType.STRONG
        }
        return tag_mapping.get(elem.name, MDElementType.PARAGRAPH)

    def _get_element_content(self, elem: Union[Tag, NavigableString]) -> str:
        """Get element's raw content."""
        try:
            # Handle NavigableString
            if isinstance(elem, NavigableString):
                return str(elem).strip()

            # Handle non-Tag
            if not isinstance(elem, Tag):
                return str(elem) if elem is not None else ''

            # Route to specific handlers based on element type
            handlers = {
                'a': self._get_link_content,
                'p': self._get_paragraph_content,
                'blockquote': self._get_blockquote_content,
                'ul': self._get_list_content,
                'ol': self._get_list_content,
                'li': self._get_list_item_content,
                'pre': self._get_code_content,
                'code': self._get_inline_code_content
            }

            handler = handlers.get(elem.name)
            if handler:
                content = handler(elem)
                # Only process if content was returned
                return content if content else ''

            # Default text content
            return elem.get_text(strip=True)

        except Exception as e:
            self.logger.error(f"Error getting element content: {str(e)}")
            return ''

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

    def _get_element_classes(self, elem: Tag) -> List[str]:
        """Get element classes."""
        try:
            classes = elem.get('class', [])
            if isinstance(classes, str):
                classes = classes.split()
            return [c for c in classes if c]
        except Exception as e:
            self.logger.error(f"Error getting element classes: {str(e)}")
            return []

    def _get_element_attributes(self, elem: Tag) -> Dict[str, str]:
        """Get element attributes."""
        try:
            # Get all attributes except class and id
            return {
                k: v for k, v in elem.attrs.items()
                if k not in ['class', 'id']
            }
        except Exception as e:
            self.logger.error(f"Error getting element attributes: {str(e)}")
            return {}

    def _get_element_metadata(self, elem: Tag) -> Dict[str, Any]:
        """Get element-specific metadata."""
        try:
            metadata = {}

            if elem.name == 'a':
                metadata.update({
                    'href': elem.get('href', ''),
                    'title': elem.get('title', '')
                })

            elif elem.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                metadata['level'] = int(elem.name[1])

            elif elem.name == 'code':
                if elem.parent and elem.parent.name == 'pre':
                    classes = elem.get('class', [])
                    if isinstance(classes, str):
                        classes = classes.split()
                    # Extract language from class
                    for cls in classes:
                        if cls.startswith('language-'):
                            metadata['language'] = cls.replace('language-', '')
                            break

            return metadata

        except Exception as e:
                self.logger.error(f"Error getting element metadata: {str(e)}")
                return {}

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
