# app/dita/utils/dita_elements.py

from typing import Dict, Optional, Union, List, Any
from pathlib import Path
import logging
from lxml import etree

from .id_handler import DITAIDHandler
from .types import (
    DITAElementType,
    DITAElementInfo,
    DITAElementContext,
    ElementAttributes
)

class DITAContentProcessor:
    """Defines DITA element structures and attributes."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.id_handler = DITAIDHandler()
        self._processed_elements: Dict[str, Any] = {}

    def process_element(self,
                       elem: etree._Element,
                       source_path: Optional[Path] = None) -> DITAElementInfo:
        """Process DITA element into structured info."""
        try:
            if not isinstance(elem, etree._Element):
                return DITAElementInfo(
                    type=DITAElementType.PARAGRAPH,
                    content="",
                    attributes=ElementAttributes(
                        id="",
                        classes=[],
                        custom_attrs={}
                    ),
                    context=DITAElementContext(
                        parent_id=None,
                        element_type="p",
                        classes=[],
                        attributes={},
                        topic_type=None,
                        is_body=False
                    ),
                    metadata={}
                )

            # Get element type
            element_type = self._get_element_type(elem)

            # Get context
            context = DITAElementContext(
                parent_id=None,  # Set by transformer
                element_type=element_type.value,
                classes=self._get_element_classes(elem),
                attributes=self._get_element_attributes(elem),
                topic_type=self._get_topic_type(elem),
                is_body=self._is_body_element(elem)
            )

            # Get base attributes
            attributes = ElementAttributes(
                id=self.id_handler.generate_content_id(
                    Path(str(elem.get('id', '')))
                ),
                classes=self._get_element_classes(elem),
                custom_attrs=self._get_element_attributes(elem)
            )

            return DITAElementInfo(
                type=element_type,
                content=self._get_element_content(elem),
                attributes=attributes,
                context=context,
                metadata=self._get_element_metadata(elem)
            )

        except Exception as e:
            self.logger.error(f"Error processing DITA element: {str(e)}")
            return DITAElementInfo(
                type=DITAElementType.PARAGRAPH,
                content="",
                attributes=ElementAttributes(
                    id="",
                    classes=[],
                    custom_attrs={}
                ),
                context=DITAElementContext(
                    parent_id=None,
                    element_type="p",
                    classes=[],
                    attributes={},
                    topic_type=None,
                    is_body=False
                ),
                metadata={'error': str(e)}
            )

    def _get_element_type(self, elem: etree._Element) -> DITAElementType:
        """Determine element type."""
        tag = etree.QName(elem).localname
        try:
            return DITAElementType(tag)
        except ValueError:
            return DITAElementType.PARAGRAPH

    def _get_element_content(self, elem: etree._Element) -> str:
        """Get element's raw content."""
        try:
            if elem.text:
                return elem.text.strip()
            return ""
        except Exception as e:
            self.logger.error(f"Error getting element content: {str(e)}")
            return ""

    def _get_element_classes(self, elem: etree._Element) -> List[str]:
        """Get element classes."""
        try:
            classes = elem.get('class', '').split()
            outputclass = elem.get('outputclass', '').split()
            return classes + outputclass
        except Exception as e:
            self.logger.error(f"Error getting element classes: {str(e)}")
            return []

    def _get_element_attributes(self, elem: etree._Element) -> Dict[str, str]:
        """Get element attributes."""
        try:
            return {
                k: v for k, v in elem.attrib.items()
                if k not in ['class', 'id', 'outputclass']
            }
        except Exception as e:
            self.logger.error(f"Error getting element attributes: {str(e)}")
            return {}

    def _get_topic_type(self, elem: etree._Element) -> Optional[str]:
        """Get topic type if element is inside a topic."""
        try:
            for parent in elem.iterancestors():
                if etree.QName(parent).localname in ['concept', 'task', 'reference', 'topic']:
                    return etree.QName(parent).localname
            return None
        except Exception as e:
            self.logger.error(f"Error getting topic type: {str(e)}")
            return None

    def _is_body_element(self, elem: etree._Element) -> bool:
        """Check if element is inside a body element."""
        try:
            for parent in elem.iterancestors():
                if etree.QName(parent).localname in ['conbody', 'taskbody', 'refbody', 'body']:
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Error checking body element: {str(e)}")
            return False

    def _get_element_metadata(self, elem: etree._Element) -> Dict[str, Any]:
        """Get element-specific metadata."""
        try:
            metadata = {}

            # Get common metadata
            if 'type' in elem.attrib:
                metadata['type'] = elem.get('type')
            if 'scale' in elem.attrib:
                metadata['scale'] = elem.get('scale')

            # Element-specific metadata
            tag = etree.QName(elem).localname

            if tag == 'xref':
                metadata.update({
                    'href': elem.get('href', ''),
                    'format': elem.get('format', ''),
                    'scope': elem.get('scope', 'local')
                })
            elif tag == 'image':
                metadata.update({
                    'href': elem.get('href', ''),
                    'alt': elem.get('alt', ''),
                    'width': elem.get('width', ''),
                    'height': elem.get('height', '')
                })
            elif tag in ['note', 'hazardstatement']:
                metadata['type'] = elem.get('type', 'note')
            elif tag == 'table':
                metadata.update({
                    'frame': elem.get('frame', 'all'),
                    'colsep': elem.get('colsep', '1'),
                    'rowsep': elem.get('rowsep', '1')
                })

            return metadata

        except Exception as e:
            self.logger.error(f"Error getting element metadata: {str(e)}")
            return {}

    def _validate_element(self, elem: etree._Element) -> bool:
        """Validate element structure."""
        try:
            if not isinstance(elem, etree._Element):
                return False

            tag = etree.QName(elem).localname

            # Validate based on element type
            if tag in ['xref', 'link']:
                return bool(elem.get('href'))
            elif tag == 'image':
                return bool(elem.get('href'))
            elif tag in ['title', 'shortdesc']:
                return bool(elem.text and elem.text.strip())

            return True

        except Exception as e:
            self.logger.error(f"Error validating element: {str(e)}")
            return False

    # Cleanup

    def cleanup(self) -> None:
            """Clean up processor resources and state."""
            try:
                self.logger.debug("Starting DITA element processor cleanup")

                # Reset ID handler
                self.id_handler = DITAIDHandler()

                # Clear processed elements cache
                self._processed_elements.clear()

                self.logger.debug("DITA element processor cleanup completed")

            except Exception as e:
                self.logger.error(f"DITA element processor cleanup failed: {str(e)}")
                raise
