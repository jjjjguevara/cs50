# app/dita/utils/markdown/md_elements.py

from enum import Enum
from typing import Dict, Optional, Union, List, Any
from pathlib import Path
import logging
from bs4 import BeautifulSoup, Tag
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

    def _get_element_content(self, elem: Tag) -> str:
        """Get element's raw content."""
        try:
            if elem.name in ['pre', 'code']:
                return elem.string or ''
            return elem.get_text()
        except Exception as e:
            self.logger.error(f"Error getting element content: {str(e)}")
            return ''

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

            # Heading specific
            if elem.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                metadata['level'] = int(elem.name[1])

            # Link specific
            if elem.name == 'a':
                metadata['href'] = elem.get('href', '')
                metadata['title'] = elem.get('title', '')

            # Image specific
            if elem.name == 'img':
                metadata['src'] = elem.get('src', '')
                metadata['alt'] = elem.get('alt', '')

            # Code specific
            if elem.name in ['pre', 'code']:
                metadata['language'] = self._get_code_language(elem)

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
