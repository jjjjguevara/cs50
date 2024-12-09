# app/dita/utils/dita_elements.py

from typing import Dict, Optional, Union, List, Any
from pathlib import Path
import logging
from lxml import etree

from app.dita.utils.id_handler import DITAIDHandler
from app.dita.models.types import (
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
            self._processed_elements = {}

    def process_element(
            self, elem: etree._Element, context: Optional[DITAElementContext] = None
        ) -> DITAElementInfo:
            """
            Process a DITA XML element into a structured element info object.

            Args:
                elem: The DITA XML element to process.
                context: Optional processing context for additional metadata or hierarchy.

            Returns:
                DITAElementInfo: Structured data representation of the DITA element.
            """
            try:
                # Determine the type of the element
                element_type = self._get_element_type(elem)

                # Process attributes
                attributes = self._get_element_attributes(elem)

                # Build the element context
                context = context or DITAElementContext(
                    parent_id=None,
                    element_type=element_type.value,
                    classes=attributes.classes,
                    attributes=attributes.custom_attrs,
                    topic_type=None,
                    is_body=False,
                )

                # Extract content
                content = str(elem.text).strip() if elem.text is not None else ""

                # Process children
                children = self._process_children(elem)

                # Construct DITAElementInfo
                return DITAElementInfo(
                    type=element_type,
                    content=content,
                    attributes=attributes,
                    context=context,
                    metadata=self._get_element_metadata(elem),
                    children=children,
                )

            except Exception as e:
                self.logger.error(f"Error processing DITA element: {str(e)}")
                return DITAElementInfo(
                    type=DITAElementType.UNKNOWN,
                    content="",
                    attributes=ElementAttributes(id="", classes=[], custom_attrs={}),
                    context=DITAElementContext(
                        parent_id=None, element_type="unknown", classes=[], attributes={}
                    ),
                    metadata={"error": str(e)},
                    children=[],
                )

    def _get_element_type(self, elem: etree._Element) -> DITAElementType:
        """
        Determine the type of the DITA element.

        Args:
            elem: The DITA XML element.

        Returns:
            DITAElementType: The determined element type.
        """
        tag = etree.QName(elem).localname
        try:
            return DITAElementType[tag.upper()]
        except KeyError:
            return DITAElementType.UNKNOWN



    def _get_element_content(self, elem: etree._Element) -> str:
        """
        Get the text content of an element, ensuring robust handling.

        Args:
            elem: The DITA XML element.

        Returns:
            str: The text content of the element, stripped of whitespace.
        """
        try:
            return str(elem.text).strip() if elem.text is not None else ""
        except Exception as e:
            self.logger.error(f"Error extracting content for element {etree.QName(elem).localname}: {str(e)}")
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

    def _get_element_attributes(self, elem: etree._Element) -> ElementAttributes:
        """
        Extract attributes from the element for processing.

        Args:
            elem: The DITA XML element.

        Returns:
            ElementAttributes: Extracted attributes for the element.
        """
        try:
            # Extract common attributes
            attributes = {
                "id": elem.get("id") or self.id_handler.generate_content_id(Path(str(elem.tag))),
                "class": elem.get("class", "").split(),
            }
            # Extract any additional attributes
            custom_attrs = {k: v for k, v in elem.attrib.items() if k not in attributes}
            return ElementAttributes(
                id=attributes["id"],
                classes=attributes["class"],
                custom_attrs=custom_attrs,
            )
        except Exception as e:
            self.logger.error(f"Error extracting attributes: {str(e)}")
            return ElementAttributes(id="", classes=[], custom_attrs={})


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
        """
        Extract metadata for a DITA element with extensibility in mind.

        Args:
            elem: The DITA XML element.

        Returns:
            dict: Metadata extracted from the element.
        """
        try:
            tag = etree.QName(elem).localname
            metadata = {}

            # Extract common attributes
            common_attributes = ["type", "scale", "id", "class"]
            for attr in common_attributes:
                if attr in elem.attrib:
                    metadata[attr] = elem.get(attr)

            # Dispatch to element-specific metadata handlers
            element_metadata_extractors = {
                "xref": self._extract_xref_metadata,
                "image": self._extract_image_metadata,
                "note": self._extract_note_metadata,
                "table": self._extract_table_metadata,
            }

            if tag in element_metadata_extractors:
                metadata.update(element_metadata_extractors[tag](elem))

            # Add remaining attributes as general metadata
            for attr, value in elem.attrib.items():
                if attr not in metadata:
                    metadata[attr] = value

            self.logger.debug(f"Extracted metadata for {tag}: {metadata}")
            return metadata

        except Exception as e:
            self.logger.error(f"Error extracting metadata for {etree.QName(elem).localname}: {str(e)}")
            return {"error": str(e)}

    def _extract_xref_metadata(self, elem: etree._Element) -> Dict[str, Any]:
        """Extract metadata specific to <xref> elements."""
        return {
            "href": elem.get("href", ""),
            "format": elem.get("format", ""),
            "scope": elem.get("scope", "local"),
        }

    def _extract_image_metadata(self, elem: etree._Element) -> Dict[str, Any]:
        """Extract metadata specific to <image> elements."""
        return {
            "href": elem.get("href", ""),
            "alt": elem.get("alt", ""),
            "width": elem.get("width", ""),
            "height": elem.get("height", ""),
        }

    def _extract_note_metadata(self, elem: etree._Element) -> Dict[str, Any]:
        """Extract metadata specific to <note> elements."""
        return {"type": elem.get("type", "note")}

    def _extract_table_metadata(self, elem: etree._Element) -> Dict[str, Any]:
        """Extract metadata specific to <table> elements."""
        return {
            "frame": elem.get("frame", "all"),
            "colsep": elem.get("colsep", "1"),
            "rowsep": elem.get("rowsep", "1"),
        }

    def _process_children(self, elem: etree._Element) -> List[DITAElementInfo]:
            """
            Process child elements of a given element.

            Args:
                elem: The parent DITA XML element.

            Returns:
                List[DITAElementInfo]: List of processed child elements.
            """
            try:
                children = []
                for child in elem:
                    if isinstance(child, etree._Element):
                        children.append(self.process_element(child))
                return children
            except Exception as e:
                self.logger.error(f"Error processing children of {etree.QName(elem).localname}: {str(e)}")
                return []

    def _validate_element(self, elem: etree._Element) -> bool:
        """
        Validate an element to ensure it has meaningful content.

        Args:
            elem: The DITA XML element.

        Returns:
            bool: True if the element is valid, False otherwise.
        """
        try:
            # Check if the element has text content
            return bool(elem.text and str(elem.text).strip())
        except Exception as e:
            self.logger.error(f"Error validating element {etree.QName(elem).localname}: {str(e)}")
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
