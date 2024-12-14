# app/dita/utils/markdown/md_elements.py
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Union, List, Any
from pathlib import Path
import logging, html
from bs4 import BeautifulSoup, NavigableString, Tag
import re
import yaml
from app.dita.processors.content_processors import ContentProcessor
from app.dita.utils.id_handler import DITAIDHandler
from app.dita.models.models import (
    Author,
    Citation,
    ContentStatus,
    ContentType,
    TopicMetadata,
    MapMetadata
)
from app.dita.models.types import (
    DITAElementType,
    ElementType,
    MDElementInfo,
    MDElementType,
    ProcessingPhase,
    TrackedElement,
    ProcessingState
)


class MarkdownElementProcessor:
    def __init__(self, content_processor: ContentProcessor, document_metadata: Dict[str, Any], map_metadata: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.id_handler = DITAIDHandler()
        self.content_processor = content_processor

        # Initialize caches
        self._processed_elements: Dict[str, MDElementInfo] = {}
        self._type_mapping_cache: Dict[str, MDElementType] = {}
        self._validation_cache: Dict[str, bool] = {}

        # Initialize element tracking
        self._tracked_elements: Dict[str, Any] = {}
        self._element_hierarchy: Dict[str, List[str]] = {}
        self._element_refs: Dict[str, int] = {}

        # Initialize metadata tracking
        self.map_metadata = map_metadata
        self.document_metadata = document_metadata
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._custom_metadata: Dict[str, Dict[str, Any]] = {}
        self._metadata_validation: Dict[str, bool] = {}
        self._strategy_feature_condition = self._strategy_feature_condition

        # Initialize metadata extraction strategies
        self._metadata_extractors = {
            MDElementType.IMAGE: self._extract_image_metadata,
            MDElementType.LINK: self._extract_link_metadata,
            MDElementType.CODE_BLOCK: self._extract_code_metadata,
            MDElementType.TODO: self._extract_todo_metadata,
            # Add more element types as needed
        }


        # Initialize processing state
        self._processing_depth: int = 0
        self._current_element_id: Optional[str] = None

        # Maps Markdown elements to their HTML output rules
        self._processing_rules = {
            MDElementType.HEADING: {
                'html_tag': 'h{level}',
                'default_classes': ['md-heading'],
                'attributes': {'role': 'heading', 'aria-level': '{level}'},
                'content_wrapper': None
            },
            MDElementType.PARAGRAPH: {
                'html_tag': 'p',
                'default_classes': ['md-paragraph'],
                'attributes': {}
            },
            MDElementType.UNORDERED_LIST: {
                'html_tag': 'ul',
                'default_classes': ['md-list'],
                'attributes': {'role': 'list'}
            },
            MDElementType.ORDERED_LIST: {
                'html_tag': 'ol',
                'default_classes': ['md-list', 'md-list-ordered'],
                'attributes': {'role': 'list'}
            },
            MDElementType.LIST_ITEM: {
                'html_tag': 'li',
                'default_classes': ['md-list-item'],
                'attributes': {'role': 'listitem'}
            },
            MDElementType.CODE_BLOCK: {
                'html_tag': 'pre',
                'default_classes': ['code-block', 'highlight'],
                'attributes': {
                    'spellcheck': 'false',
                    'translate': 'no'
                },
                'inner_tag': 'code',
                'inner_classes': ['language-{outputclass}']
            },
            MDElementType.BLOCKQUOTE: {
                'html_tag': 'blockquote',
                'default_classes': ['md-blockquote'],
                'content_wrapper': 'p'
            },
            MDElementType.LINK: {
                'html_tag': 'a',
                'default_classes': ['md-link'],
                'required_attributes': ['href'],
                'attribute_mapping': {
                    'url': 'href',
                    'title': 'title'
                },
                'attributes': {
                    'target': '_blank',
                    'rel': 'noopener noreferrer'
                }
            },
            MDElementType.IMAGE: {
                'html_tag': 'img',
                'default_classes': ['md-image', 'img-fluid'],
                'required_attributes': ['src', 'alt'],
                'attribute_mapping': {
                    'url': 'src',
                    'description': 'alt',
                    'title': 'title'
                },
                'attributes': {
                    'loading': 'lazy',
                    'decoding': 'async'
                }
            },
            MDElementType.TABLE: {
                'html_tag': 'table',
                'default_classes': ['table', 'table-bordered', 'md-table'],
                'attributes': {
                    'role': 'grid'
                }
            },
            MDElementType.TABLE_HEADER: {
                'html_tag': 'th',
                'default_classes': ['md-th'],
                'attributes': {'scope': 'col'}
            },
            MDElementType.TABLE_ROW: {
                'html_tag': 'tr',
                'default_classes': ['md-tr']
            },
            MDElementType.TABLE_CELL: {
                'html_tag': 'td',
                'default_classes': ['md-td']
            },
            MDElementType.BOLD: {
                'html_tag': 'strong',
                'default_classes': ['md-bold']
            },
            MDElementType.ITALIC: {
                'html_tag': 'em',
                'default_classes': ['md-italic']
            },
            MDElementType.TODO: {
                'html_tag': 'div',
                'default_classes': ['todo-item'],
                'inner_tag': 'input',
                'inner_attributes': {
                    'type': 'checkbox'
                },
                'content_wrapper': 'label'
            },
            MDElementType.FOOTNOTE: {
                'html_tag': 'div',
                'default_classes': ['footnote'],
                'attributes': {'role': 'doc-footnote'},
                'content_wrapper': 'p'
            },
            MDElementType.YAML_METADATA: {
                'html_tag': 'div',
                'default_classes': ['metadata-section'],
                'attributes': {'aria-hidden': 'true'}
            },
            MDElementType.UNKNOWN: {
                'html_tag': 'div',
                'default_classes': ['md-unknown']
            }
        }


    def set_processing_phase(self, phase: ProcessingPhase) -> None:
        """Update current processing phase."""
        self.current_phase = phase
        self.logger.debug(f"Updated processing phase to {phase.value}")

    def process_element(
        self,
        elem: Tag,
        source_path: Path,
        document_metadata: Dict[str, Any],
        map_metadata: Dict[str, Any]
    ) -> TrackedElement:
        """Process a markdown element into a TrackedElement with proper metadata."""
        try:
            # Get element type - MDElementType is already an ElementType
            element_type = self._get_element_type(elem)

            # Initialize tracked element
            element = TrackedElement.from_discovery(
                path=source_path,
                element_type=ElementType(element_type.value),  # Convert to ElementType
                id_handler=self.id_handler
            )

            # Set content
            element.content = self._get_element_content(elem)

            # Process attributes
            rules = self._processing_rules.get(element_type) or {}
            element.html_metadata["attributes"] = {}
            element.html_metadata["classes"] = []
            self._process_attributes(elem, rules, element.html_metadata)

            # Set metadata
            element.metadata = self._extract_element_metadata(elem, document_metadata, map_metadata)

            # Handle heading level in metadata
            if element_type == MDElementType.HEADING:
                element.metadata["heading_level"] = int(elem.name[1]) if elem.name[1:2].isdigit() else 1

            element.state = ProcessingState.PROCESSING

            return element

        except Exception as e:
            self.logger.error(f"Error processing element: {str(e)}")
            return TrackedElement.from_discovery(
                path=source_path,
                element_type=ElementType.UNKNOWN,
                id_handler=self.id_handler
            )

    def _process_attributes(
        self,
        elem: Tag,
        rules: Dict[str, Any],
        html_metadata: Dict[str, Any]
    ) -> None:
        """
        Process element attributes directly into html_metadata.

        Args:
            elem: BeautifulSoup tag
            rules: Processing rules for the element type
            html_metadata: HTML metadata dict to update
        """
        try:
            # Generate ID
            element_id = elem.get('id') or self.id_handler.generate_id(elem.name)
            html_metadata["attributes"]["id"] = str(element_id)

            # Get classes from rules and element
            classes = rules.get('default_classes', []).copy()
            if elem_classes := elem.get('class', []):
                if isinstance(elem_classes, str):
                    classes.extend(elem_classes.split())
                else:
                    classes.extend(elem_classes)
            html_metadata["classes"] = classes

            # Add rule-defined attributes
            html_metadata["attributes"].update(rules.get('attributes', {}))

            # Add element attributes, handling lists
            for key, value in elem.attrs.items():
                if key not in {'id', 'class'}:
                    html_metadata["attributes"][key] = value[0] if isinstance(value, list) else value

            # Apply attribute mappings from rules
            for md_attr, html_attr in rules.get('attribute_mapping', {}).items():
                if md_attr in elem.attrs:
                    attr_value = elem.attrs[md_attr]
                    html_metadata["attributes"][html_attr] = attr_value[0] if isinstance(attr_value, list) else attr_value

        except Exception as e:
            self.logger.error(f"Error processing attributes: {str(e)}")


    def _get_element_type(self, elem: Tag) -> MDElementType:
       """
       Determine the type of Markdown element based on its tag and attributes.

       Args:
           elem: BeautifulSoup tag to analyze

       Returns:
           MDElementType: The determined element type
       """
       try:
           # Get tag name
           tag = elem.name.lower() if elem.name else ''

           # Direct mapping for most elements
           tag_mappings = {
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
               'pre': MDElementType.CODE_BLOCK,
               'blockquote': MDElementType.BLOCKQUOTE,
               'table': MDElementType.TABLE,
               'thead': MDElementType.TABLE_HEADER,
               'tr': MDElementType.TABLE_ROW,
               'td': MDElementType.TABLE_CELL,
               'strong': MDElementType.BOLD,
               'em': MDElementType.ITALIC,
               'u': MDElementType.UNDERLINE
           }

           if tag in tag_mappings:
               return tag_mappings[tag]

           # Special cases requiring attribute checks
           if tag == 'a':
               # Check for backlinks (Obsidian-specific)
               if 'data-footnote-backref' in elem.attrs:
                   return MDElementType.BACKLINK
               return MDElementType.LINK

           if tag == 'div':
               # Check div classes for special types
               classes = elem.get('class', [])
               if 'footnotes' in classes:
                   return MDElementType.FOOTNOTE
               if 'todo-item' in classes:
                   return MDElementType.TODO
               if 'metadata-section' in classes:
                   return MDElementType.YAML_METADATA

           if tag == 'img':
               return MDElementType.IMAGE

           if tag == 'code':
               parent = elem.find_parent('pre')
               return MDElementType.CODE_BLOCK if parent else MDElementType.CODE_PHRASE

           # Log warning for unknown elements
           self.logger.warning(f"Unknown Markdown element type: {tag}")
           return MDElementType.UNKNOWN

       except Exception as e:
           self.logger.error(f"Error determining element type: {str(e)}")
           return MDElementType.UNKNOWN

    def _get_element_content(self, elem: Tag) -> str:
       """
       Extract content from Markdown element based on its type and rules.

       Args:
           elem: The BeautifulSoup tag

       Returns:
           str: Processed content string for the element
       """
       try:
           # Get element type and rules
           element_type = self._get_element_type(elem)
           rules = self._processing_rules.get(element_type,
                                            self._processing_rules[MDElementType.UNKNOWN])

           # Handle empty elements
           if not elem.string and len(elem.contents) == 0:
               return ""

           # Special content handling based on element type
           if element_type == MDElementType.CODE_BLOCK:
               return self._get_code_content(elem, rules)

           elif element_type == MDElementType.TODO:
               return self._get_todo_content(elem, rules)

           elif element_type == MDElementType.FOOTNOTE:
               return self._get_footnote_content(elem, rules)

           elif element_type == MDElementType.IMAGE:
               return ""  # Images don't have text content

           elif element_type == MDElementType.LINK:
               return self._get_link_content(elem, rules)

           # Standard content processing
           content_parts = []

           # Add element's direct text if present
           if elem.string:
               content_parts.append(elem.string.strip())

           # Process child elements' content
           for child in elem.children:
               if isinstance(child, Tag):
                   # Recursively get child content if it's a text-containing element
                   if self._is_text_element(child):
                       child_content = self._get_element_content(child)
                       if child_content:
                           content_parts.append(child_content)
               elif isinstance(child, NavigableString):  # Type check NavigableString
                  content_parts.append(str(child).strip())

           # Join all content parts
           return ' '.join(filter(None, content_parts))

       except Exception as e:
           self.logger.error(f"Error extracting content from {elem.name}: {str(e)}")
           return ""


    def _get_footnote_content(self, elem: Tag, rules: dict) -> str:
       """
       Process footnote content.

       Args:
           elem: BeautifulSoup tag representing footnote element
           rules: Processing rules for the element

       Returns:
           str: Processed footnote content
       """
       try:
           # Get footnote ID
           footnote_id = elem.get('id', '')[0] if isinstance(elem.get('id'), list) else elem.get('id', '')

           # Get content
           content = elem.get_text(strip=True)

           return f"FOOTNOTE:{footnote_id}:{content}"

       except Exception as e:
           self.logger.error(f"Error processing footnote content: {str(e)}")
           return ""

    def _is_text_element(self, elem: Tag) -> bool:
       """
       Determine if an element can contain meaningful text content.

       Args:
           elem: BeautifulSoup tag to check

       Returns:
           bool: True if element can contain text content
       """
       try:
           element_type = self._get_element_type(elem)
           return element_type not in {
               MDElementType.IMAGE,
               MDElementType.YAML_METADATA,
               MDElementType.UNKNOWN
           }

       except Exception as e:
           self.logger.error(f"Error checking text element: {str(e)}")
           return False

    def _get_inline_code_content(self, elem: Tag, rules: dict) -> str:
       """
       Extract and process inline code content.

       Args:
           elem: BeautifulSoup tag representing code element
           rules: Processing rules for the element

       Returns:
           str: Content of the inline code element
       """
       try:
           # Extract raw code content
           content = elem.string or ""

           # Get language if specified in parent pre tag
           parent = elem.find_parent('pre')
           if parent and 'data-language' in parent.attrs:
               language = parent['data-language']
               return f"CODE:{language}:{content}"

           # Handle inline code without language
           return f"CODE::{content}"

       except Exception as e:
           self.logger.error(f"Error processing inline code content: {str(e)}")
           return ''


    def _get_link_content(self, elem: Tag, rules: dict) -> str:
        """
        Process link element content.

        Args:
            elem: BeautifulSoup tag representing link element
            rules: Processing rules for the element

        Returns:
            str: Processed link content
        """
        try:
            # Skip processing if link is part of block elements
            if elem.find_parent(['li', 'p', 'blockquote']):
                return ''

            # Extract link attributes - handle as string
            href = elem.get('href', '')[0] if isinstance(elem.get('href'), list) else elem.get('href', '')
            text = elem.get_text(strip=True)

            # Handle Obsidian-style wiki-links
            if isinstance(href, str):  # Ensure href is string
                if href.startswith('[[') and href.endswith(']]'):
                    href = href[2:-2]  # Remove [[ ]]
                    # Split at pipe for aliased links [[target|alias]]
                    parts = href.split('|', 1)
                    if len(parts) > 1:
                        href = parts[0]
                        text = parts[1]

            # Handle special link types via rules
            link_type = 'default'
            if 'link_type_mapping' in rules:
                data_type = elem.get('data-type', ['default'])[0] if isinstance(elem.get('data-type'), list) else elem.get('data-type', 'default')
                link_type = rules['link_type_mapping'].get(data_type, 'default')

            return f"LINK:{link_type}:{href}:{text}"

        except Exception as e:
            self.logger.error(f"Error processing link content: {str(e)}")
            return ''

    def _get_paragraph_content(self, elem: Tag, rules: dict) -> str:
       """
       Process paragraph content with potential inline elements.

       Args:
           elem: BeautifulSoup tag representing paragraph
           rules: Processing rules for the element

       Returns:
           str: Processed paragraph content
       """
       try:
           parts = []
           for child in elem.children:
               if isinstance(child, Tag):
                   # Handle links
                   if child.name == 'a':
                       href = child.get('href', '')[0] if isinstance(child.get('href'), list) else child.get('href', '')
                       text = child.get_text(strip=True)
                       parts.append(f"LINK:{href}:{text}")
                   # Handle inline code
                   elif child.name == 'code':
                       code_content = child.get_text(strip=True)
                       parts.append(f"CODE::{code_content}")
                   # Handle other inline elements
                   else:
                       parts.append(child.get_text(strip=True))
               else:
                   # Handle text nodes
                   parts.append(str(child).strip())

           return ' '.join(filter(None, parts))

       except Exception as e:
           self.logger.error(f"Error processing paragraph content: {str(e)}")
           return ""

    def _get_blockquote_content(self, elem: Tag, rules: dict) -> str:
       """
       Process blockquote content with potential inline elements.

       Args:
           elem: BeautifulSoup tag representing blockquote
           rules: Processing rules for the element

       Returns:
           str: Processed blockquote content
       """
       try:
           parts = []
           for child in elem.children:
               if isinstance(child, Tag):
                   if child.name == 'a':
                       href = child.get('href', '')[0] if isinstance(child.get('href'), list) else child.get('href', '')
                       text = child.get_text(strip=True)
                       parts.append(f"LINK:{href}:{text}")
                   elif child.name == 'p':
                       # Get paragraph content directly
                       parts.append(child.get_text(strip=True))
                   elif child.name == 'code':
                       code_content = child.get_text(strip=True)
                       parts.append(f"CODE::{code_content}")
                   else:
                       parts.append(child.get_text(strip=True))
               else:
                   parts.append(str(child).strip())

           # Return formatted blockquote content
           return f"QUOTE:{' '.join(filter(None, parts))}"

       except Exception as e:
           self.logger.error(f"Error processing blockquote content: {str(e)}")
           return ""

    def _get_list_content(self, elem: Tag, rules: dict) -> str:
        """
        Process list content (ul/ol) with proper nesting.

        Args:
            elem: BeautifulSoup tag representing list element
            rules: Processing rules for the element

        Returns:
            str: Processed list content
        """
        try:
            items = []

            # Get list type
            list_type = 'ul' if elem.name == 'ul' else 'ol'

            # Process list items
            for li in elem.find_all('li', recursive=False):
                item_content = self._get_list_item_content(li)
                if item_content:
                    items.append(item_content)

            # Format with list type and items
            return f"LIST:{list_type}:{','.join(filter(None, items))}"

        except Exception as e:
            self.logger.error(f"Error processing list content: {str(e)}")
            return ""

    def _get_list_item_content(self, elem: Tag) -> str:
       """
       Process list item content with potential nested elements.

       Args:
           elem: BeautifulSoup tag representing list item

       Returns:
           str: Processed list item content
       """
       try:
           content = []

           for child in elem.children:
               if isinstance(child, Tag):
                   if child.name == 'a':
                       href = child.get('href', '')[0] if isinstance(child.get('href'), list) else child.get('href', '')
                       text = child.get_text(strip=True)
                       content.append(f"LINK:{href}:{text}")
                   elif child.name == 'code':
                       code_content = child.get_text(strip=True)
                       content.append(f"CODE::{code_content}")
                   elif child.name in ['ul', 'ol']:
                       nested_list = self._get_list_content(child, {})
                       content.append(nested_list)
                   else:
                       content.append(child.get_text(strip=True))
               else:
                   text = str(child).strip()
                   if text:
                       content.append(text)

           return ' '.join(filter(None, content))

       except Exception as e:
           self.logger.error(f"Error processing list item content: {str(e)}")
           return ""

    def _get_todo_content(self, elem: Tag, rules: dict) -> str:
       """
       Process todo item content (Obsidian-style).

       Args:
           elem: BeautifulSoup tag representing todo element
           rules: Processing rules for the element

       Returns:
           str: Processed todo content
       """
       try:
           # Get checkbox state
           text = elem.get_text().strip()
           is_checked = text.startswith('[x]')

           # Clean up the text content
           content = text[3:].strip() if text.startswith('[ ]') or text.startswith('[x]') else text

           # Format with state
           state = 'checked' if is_checked else 'unchecked'
           return f"TODO:{state}:{content}"

       except Exception as e:
           self.logger.error(f"Error processing todo content: {str(e)}")
           return ""

    def _get_code_content(self, elem: Tag, rules: dict) -> str:
        """
        Process code block content with language detection.

        Args:
            elem: BeautifulSoup tag representing code element
            rules: Processing rules for the element

        Returns:
            str: Processed code content
        """
        try:
            # Find code element if we're on a pre tag
            code = elem.find('code') if elem.name == 'pre' else elem
            if not code or not isinstance(code, Tag):
                return ""

            # Get language from class
            classes = code.attrs.get('class', [])
            language = ''
            for cls in classes:
                if cls.startswith('language-'):
                    language = cls.replace('language-', '')
                    break

            # Special handling for Mermaid diagrams
            if language == 'mermaid':
                return f"MERMAID:{code.get_text(strip=True)}"

            # Handle regular code blocks
            content = code.get_text()
            return f"CODE:{language}:{content}"

        except Exception as e:
            self.logger.error(f"Error processing code block content: {str(e)}")
            return ""

    def _get_image_content(self, elem: Tag, rules: dict) -> str:
       """
       Process image element content and attributes.

       Args:
           elem: BeautifulSoup tag representing image element
           rules: Processing rules for the element

       Returns:
           str: Processed image content with attributes
       """
       try:
           # Extract image attributes
           src = elem.get('src', '')[0] if isinstance(elem.get('src'), list) else elem.get('src', '')
           alt = elem.get('alt', '')[0] if isinstance(elem.get('alt'), list) else elem.get('alt', '')
           title = elem.get('title', '')[0] if isinstance(elem.get('title'), list) else elem.get('title', '')

           # Format attributes
           attrs = [
               f"src={src}",
               f"alt={alt}" if alt else "",
               f"title={title}" if title else ""
           ]

           # Return formatted image content
           return f"IMAGE:{':'.join(filter(None, attrs))}"

       except Exception as e:
           self.logger.error(f"Error processing image content: {str(e)}")
           return ""

    def _get_element_attributes(
        self,
        elem: Tag,
        rules: dict,
        html_metadata: Dict[str, Any]
    ) -> None:
        """
        Extract and process element attributes directly into html_metadata.

        Args:
            elem: The BeautifulSoup tag
            rules: Processing rules for this element type
            html_metadata: HTML metadata dict to update
        """
        try:
            # Generate ID
            raw_id = elem.get('id', '')
            element_id = raw_id[0] if isinstance(raw_id, list) else raw_id
            if not element_id:
                element_id = self.id_handler.generate_id(elem.name)
            html_metadata["attributes"]["id"] = str(element_id)

            # Get classes
            classes = self._get_element_classes(elem)
            html_metadata["classes"] = classes

            # Initialize attributes
            html_metadata["attributes"].update(rules.get('attributes', {}))

            # Process attribute mappings
            for md_attr, html_attr in rules.get('attribute_mapping', {}).items():
                if md_attr in elem.attrs:
                    attr_value = elem.attrs[md_attr]
                    html_metadata["attributes"][html_attr] = attr_value[0] if isinstance(attr_value, list) else attr_value

            # Process remaining attributes
            for attr, value in elem.attrs.items():
                if attr in {'id', 'class'}:
                    continue

                processed_value = value[0] if isinstance(value, list) else value

                if attr == 'href':
                    html_metadata["attributes"]['href'] = processed_value
                    if processed_value.startswith(('http://', 'https://')):
                        html_metadata["attributes"]['target'] = '_blank'
                        html_metadata["attributes"]['rel'] = 'noopener noreferrer'
                else:
                    attr_name = attr if attr in rules.get('allowed_attributes', []) else f'data-{attr}'
                    html_metadata["attributes"][attr_name] = processed_value

            # Add required attributes
            for required_attr in rules.get('required_attributes', []):
                if required_attr not in html_metadata["attributes"]:
                    self.logger.warning(
                        f"Missing required attribute '{required_attr}' "
                        f"for element: {elem.name}"
                    )
                    html_metadata["attributes"][required_attr] = ''

        except Exception as e:
            self.logger.error(f"Error extracting attributes: {str(e)}")
            html_metadata["attributes"] = {"id": str(self.id_handler.generate_id("error"))}
            html_metadata["classes"] = []

    def _get_element_classes(self, elem: Tag) -> List[str]:
       """
       Get combined classes for element based on processing rules and element attributes.

       Args:
           elem: BeautifulSoup tag

       Returns:
           List[str]: Combined list of CSS classes
       """
       try:
           # Get element type and rules
           element_type = self._get_element_type(elem)
           rules = self._processing_rules.get(element_type,
                                            self._processing_rules[MDElementType.UNKNOWN])

           # Start with default classes from rules
           classes = rules['default_classes'].copy()

           # Add classes from element's class attribute
           element_classes = elem.get('class', [])
           if isinstance(element_classes, str):
               # If a single class was provided as string
               classes.extend(element_classes.split())
           else:
               # If multiple classes were provided as list
               classes.extend(element_classes)

           # Add type-specific classes if applicable
           if 'type_class_mapping' in rules and 'type' in elem.attrs:
               type_value = elem.attrs['type']
               if isinstance(type_value, list):
                   type_value = type_value[0]
               type_class = rules['type_class_mapping'].get(
                   type_value,
                   rules['type_class_mapping'].get('default', '')
               )
               if type_class:
                   classes.append(type_class)

           # Add state-based classes
           if 'state' in elem.attrs:
               state = elem.attrs['state']
               if isinstance(state, list):
                   state = state[0]
               state_class = f"state-{state}"
               classes.append(state_class)

           # Remove duplicates while preserving order
           seen = set()
           unique_classes = [
               cls for cls in classes
               if cls not in seen and not seen.add(cls)
           ]

           self.logger.debug(
               f"Generated classes for {element_type.value}: {unique_classes}"
           )

           return unique_classes

       except Exception as e:
           self.logger.error(f"Error getting element classes: {str(e)}")
           return []

    def _default_element(self, source_path: Path) -> TrackedElement:
        """Return a default TrackedElement for unprocessable cases."""
        element = TrackedElement.from_discovery(
            path=source_path,
            element_type=ElementType.BODY,  # Use BODY for default paragraphs
            id_handler=self.id_handler
        )

        # Set basic metadata
        element.content = ""
        element.html_metadata.update({
            "attributes": {
                "id": self.id_handler.generate_id("default")
            },
            "classes": [],
            "context": {
                "parent_id": None,
                "level": None,
                "position": None
            }
        })
        element.metadata = {}
        element.state = ProcessingState.COMPLETED

        return element

    def _get_code_language(self, elem: Tag) -> Optional[str]:
       """
       Get code block language from element classes.

       Args:
           elem: BeautifulSoup tag representing code element

       Returns:
           Optional[str]: Language identifier or None if not found
       """
       try:
           if not isinstance(elem, Tag):
               return None

           classes = elem.attrs.get('class', [])
           for cls in classes:
               if cls.startswith('language-'):
                   return cls.replace('language-', '')

           return None

       except Exception as e:
           self.logger.error(f"Error getting code language: {str(e)}")
           return None

    def _validate_element(self, elem: Tag) -> bool:
       """
       Validate element structure.

       Args:
           elem: BeautifulSoup tag to validate

       Returns:
           bool: True if element is valid
       """
       try:
           if not isinstance(elem, Tag):
               return False

           # Get element type and rules
           element_type = self._get_element_type(elem)
           rules = self._processing_rules.get(element_type,
                                            self._processing_rules[MDElementType.UNKNOWN])

           # Validate required attributes
           if 'required_attributes' in rules:
               for attr in rules['required_attributes']:
                   if attr not in elem.attrs:
                       self.logger.warning(
                           f"Missing required attribute '{attr}' "
                           f"for {element_type.value} element"
                       )
                       return False

           # Element-specific validation
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
           # Verify tag name format (h1-h6)
           if not elem.name or not elem.name.startswith('h'):
               return False

           # Check level is 1-6
           level = int(elem.name[1])
           if not 1 <= level <= 6:
               return False

           # Verify content exists
           return bool(elem.get_text().strip())

       except (ValueError, AttributeError) as e:
           self.logger.error(f"Error validating heading: {str(e)}")
           return False

    def _validate_link(self, elem: Tag) -> bool:
       """Validate link element."""
       try:
           # Check for href attribute
           href = elem.attrs.get('href', '')
           if isinstance(href, list):
               href = href[0]

           if not href:
               self.logger.warning("Link element missing 'href' attribute")
               return False

           # Check for content
           if not elem.get_text().strip():
               self.logger.warning("Empty link content")
               return False

           return True

       except Exception as e:
           self.logger.error(f"Error validating link: {str(e)}")
           return False

    def _validate_image(self, elem: Tag) -> bool:
       """Validate image element."""
       try:
           # Check for src attribute
           src = elem.attrs.get('src', '')
           if isinstance(src, list):
               src = src[0]

           if not src:
               self.logger.warning("Image element missing 'src' attribute")
               return False

           # Check for alt text (accessibility)
           alt = elem.attrs.get('alt', '')
           if isinstance(alt, list):
               alt = alt[0]

           if not alt:
               self.logger.warning("Image element missing 'alt' attribute")
               return False

           return True

       except Exception as e:
           self.logger.error(f"Error validating image: {str(e)}")
           return False



    # ==========================================================================
    # METADATA EXTRACTION
    # ==========================================================================

    # PRIMARY METADATA EXTRACTION

    def _extract_yaml_frontmatter(self, content: str, level: str = 'topic') -> Dict[str, Any]:
        """
        Extract and normalize YAML frontmatter for maps or topics.

        Args:
            content: The raw markdown or MDITA content as a string.
            level: The context level ('map' or 'topic').

        Returns:
            A dictionary containing normalized metadata.
        """
        try:
            # Check if content starts with YAML frontmatter
            if not content.startswith('---'):
                return {}

            # Locate the end of the YAML block
            end_idx = content.find('---', 3)
            if end_idx == -1:
                return {}

            # Parse YAML block
            frontmatter = yaml.safe_load(content[3:end_idx]) or {}

            # Normalize metadata
            metadata = self._normalize_metadata(frontmatter, level)
            return metadata

        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML frontmatter: {str(e)}")
            return {}
        except Exception as e:
            self.logger.error(f"Error processing frontmatter: {str(e)}")
            return {}

    def _normalize_metadata(self, frontmatter: Dict[str, Any], level: str) -> Dict[str, Any]:
        """
        Normalize YAML frontmatter for maps and topics.

        Args:
            frontmatter: Parsed frontmatter dictionary.
            level: The context level ('map' or 'topic').

        Returns:
            Normalized metadata.
        """
        # Base metadata structure
        metadata = {
            'context': level,
            'content': {
                'title': frontmatter.get('title', 'Untitled'),
                'slug': frontmatter.get('slug') or self.id_handler.generate_id(frontmatter.get('title', 'Untitled')),
                'content_type': frontmatter.get('content-type', 'article'),
                'categories': frontmatter.get('categories', []),
                'keywords': frontmatter.get('keywords', []),
                'language': frontmatter.get('language', 'en-US'),
                'region': frontmatter.get('region', 'Global'),
                'abstract': frontmatter.get('abstract', ''),
            },
            'specialization': {
                'object': frontmatter.get('object', 'unknown'),
                'role': frontmatter.get('role', 'general'),
            },
            'publication': {
                'publication_date': frontmatter.get('publication-date'),
                'last_edited': frontmatter.get('last-edited'),
                'version': frontmatter.get('version', '1.0'),
                'revision_history': frontmatter.get('revision-history', []),
            },
            'contributors': {
                'authors': frontmatter.get('authors', []),
                'editor': frontmatter.get('editor'),
                'reviewer': frontmatter.get('reviewer'),
            },
            'delivery': frontmatter.get('delivery', {
                'channel_web': True,
                'channel_app': False,
                'channel_print': False,
            }),
            'media': frontmatter.get('media', {}),
            'conditions': frontmatter.get('conditions', {}),
            'analytics': frontmatter.get('analytics', {}),
        }

        # Handle map-level additions
        if level == 'map':
            metadata['features'] = frontmatter.get('features', {})
            metadata['audience'] = frontmatter.get('audience', [])
        elif level == 'topic':
            metadata['state'] = frontmatter.get('state', 'draft')

        return metadata



    # SECONDARY METADATA EXTRACTION

    def _extract_element_metadata(
        self,
        elem: Tag,
        document_metadata: Dict[str, Any],
        map_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract metadata for an individual element, including core attributes, conditional metadata,
        contextual information, feature flags, and element-specific metadata.

        Args:
            elem: BeautifulSoup tag to process.
            document_metadata: YAML frontmatter context (topic level).
            map_metadata: YAML frontmatter context (map level).

        Returns:
            A dictionary containing consolidated metadata for the element.
        """
        try:
            # Consolidate map and topic metadata for hierarchical inheritance
            context_metadata = {**map_metadata, **document_metadata} if map_metadata else document_metadata

            # Initialize metadata structure
            metadata = {
                'core': {
                    'id': elem.get('id') or self.id_handler.generate_id(str(elem.name)),
                    'tag': elem.name,
                    'class': ' '.join(elem.get('class', [])) if elem.get('class') else None,
                    'title': elem.get('title'),
                    'style': elem.get('style'),
                    'language': elem.get('lang', context_metadata.get('content', {}).get('language', 'en-US')),
                    'direction': elem.get('dir', 'ltr'),
                    'text': elem.get_text().strip() if elem.string else '',
                    'word_count': len(elem.get_text().split()) if elem.string else 0,
                    'has_children': bool(elem.find_all()),
                },
                'context': self._determine_element_context(elem, context_metadata),
                'processing': {
                    'state': elem.get('data-state', 'active'),
                    'priority': elem.get('data-priority', 'normal'),
                    'visibility': elem.get('data-visibility', 'show')
                },
                'conditions': {},
                'features': {},
                'custom': {},
                'specific': {},  # Placeholder for element-specific metadata
            }

            # Extract conditional metadata
            document_conditions = context_metadata.get('conditions', {})
            conditional_attrs = {
                'audience': elem.get('data-audience', document_conditions.get('audience', [])),
                'platform': elem.get('data-platform', []),
                'product': elem.get('data-product', []),
                'region': elem.get('data-region', context_metadata.get('content', {}).get('region', 'Global')),
                'version': elem.get('data-min-version', context_metadata.get('publication', {}).get('version'))
            }
            metadata['conditions'].update({
                condition: self._evaluate_condition(condition, value, context_metadata)
                for condition, value in conditional_attrs.items() if value
            })

            # Extract feature flags dynamically
            document_features = context_metadata.get('features', {})
            feature_attrs = {
                feature: self._evaluate_condition('feature', feature, context_metadata)
                for feature in document_features
            }
            metadata['features'].update(feature_attrs)

            # Extract custom attributes
            metadata['custom'] = {
                k: v for k, v in elem.attrs.items()
                if k.startswith('data-') and k not in {
                    'data-audience', 'data-platform', 'data-product',
                    'data-state', 'data-priority', 'data-visibility',
                    'data-interactive', 'data-subscription', 'data-show-line-numbers',
                    'data-mermaid', 'data-revision', 'data-importance',
                }
            }

            # STRATEGY-BASED element-specific metadata extraction
            element_type = self._get_element_type(elem)
            if element_type in self._metadata_extractors:
                metadata['specific'] = self._metadata_extractors[element_type](elem, context_metadata)

            # Include accessibility metadata if required
            if document_conditions.get('accessibility_compliant', False):
                metadata['core'].update({
                    'role': elem.get('role'),
                    'aria_label': elem.get('aria-label'),
                    'aria_describedby': elem.get('aria-describedby'),
                    'aria_hidden': elem.get('aria-hidden'),
                    'tabindex': elem.get('tabindex'),
                })

            return metadata

        except Exception as e:
            self.logger.error(f"Error extracting metadata for element '{elem.name}': {str(e)}")
            return {
                'error': str(e),
                'element': str(elem)
            }



    # METADATA EXTRACTOR STRATEGIES

    def _extract_image_metadata(self, elem: Tag, context_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata for an image element.

        Args:
            elem: BeautifulSoup tag representing the image.
            context_metadata: Consolidated map and topic-level metadata.

        Returns:
            A dictionary containing image-specific metadata.
        """
        figcaption = elem.parent.find('figcaption') if elem.parent and elem.parent.name == 'figure' else None
        return {
            'src': elem.get('src', ''),
            'alt': elem.get('alt', ''),
            'title': elem.get('title', ''),
            'width': elem.get('width'),
            'height': elem.get('height'),
            'loading': elem.get('loading', 'lazy'),
            'interactive': context_metadata.get('features', {}).get('interactive_media', False),
            'figure_id': self.id_handler.generate_id(f"fig-{elem.get('alt', '')}"),
            'caption': figcaption.get_text() if figcaption else ''
        }

    def _extract_link_metadata(self, elem: Tag, context_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata for a link element.

        Args:
            elem: BeautifulSoup tag representing the link.
            context_metadata: Consolidated map and topic-level metadata.

        Returns:
            A dictionary containing link-specific metadata.
        """
        href = str(elem.get('href', ''))
        return {
            'href': href,
            'title': elem.get('title'),
            'text': elem.get_text().strip(),
            'is_external': href.startswith(('http://', 'https://')),
            'is_conref': href.startswith('conref:'),
            'target': elem.get('target', '_blank' if href.startswith(('http://', 'https://')) else None),
            'rel': elem.get('rel', 'noopener noreferrer' if href.startswith(('http://', 'https://')) else None)
        }


    def _extract_code_metadata(self, elem: Tag, context_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata for a code block element.

        Args:
            elem: BeautifulSoup tag representing the code block.
            context_metadata: Consolidated map and topic-level metadata.

        Returns:
            A dictionary containing code block-specific metadata.
        """
        language = self._get_code_language(elem)
        return {
            'language': language,
            'line_count': len(elem.get_text().splitlines()),
            'is_mermaid': language == 'mermaid',
            'show_line_numbers': str(elem.get('data-show-line-numbers', 'true')).lower() == 'true',
            'highlight_lines': self._parse_highlight_lines(str(elem.get('data-highlight-lines', ''))),
            'caption': elem.get('data-caption'),
            'executable': str(elem.get('data-executable', 'false')).lower() == 'true'
        }


    def _extract_todo_metadata(self, elem: Tag, context_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata for a TODO item.

        Args:
            elem: BeautifulSoup tag representing the TODO item.
            context_metadata: Consolidated map and topic-level metadata.

        Returns:
            A dictionary containing TODO-specific metadata.
        """
        text = elem.get_text().strip()
        return {
            'is_checked': text.startswith('[x]'),
            'text': text[3:].strip() if text.startswith(('[x]', '[ ]')) else text,
            'priority': elem.get('data-priority', 'normal'),
            'due_date': elem.get('data-due-date'),
            'assigned_to': elem.get('data-assigned-to')
        }



    # STRATEGY-BASED CONTEXTUAL/SPECIALIZATION/CONDITIONAL METADATA PROCESSING

    def _determine_element_context(
        self,
        elem: Tag,
        context_metadata: Dict[str, Any]
    ) -> str:
        """
        Determine the context of an element based on its type and attributes.

        Args:
            elem: BeautifulSoup tag to process.
            context_metadata: Consolidated map and topic-level metadata.

        Returns:
            A string representing the context of the element.
        """
        try:
            # Mapping element types to context determination logic
            context_strategies = {
                'table': self._determine_table_context,
                'image': self._determine_image_context,
                'code': self._determine_code_context,
            }

            # Determine element type
            element_type = self._get_element_type(elem)

            # Apply strategy if defined, else default to 'default'
            context_strategy = context_strategies.get(element_type.value, lambda e, _: 'default')
            return context_strategy(elem, context_metadata)

        except Exception as e:
            self.logger.error(f"Error determining context for element '{elem.name}': {str(e)}")
            return 'default'

    # CONTEXT PROCESSING STRATEGIES

    def _determine_table_context(self, elem: Tag, _: Dict[str, Any]) -> str:
        """
        Determine the context for a table element based on its headers.

        Args:
            elem: BeautifulSoup tag representing the table.
            _: Placeholder for context metadata (not used here).

        Returns:
            A string representing the table's context.
        """
        thead = elem.find('thead')
        if thead and isinstance(thead, Tag):
            headers = [th.get_text().strip().lower() for th in thead.find_all('th') if isinstance(th, Tag)]
            if {'term', 'definition'}.issubset(headers):
                return 'glossary'
            if {'author', 'year', 'title'}.issubset(headers):
                return 'bibliography'
            if {'property', 'value'}.issubset(headers):
                return 'metadata'
        return 'default'


    def _determine_image_context(self, elem: Tag, _: Dict[str, Any]) -> str:
        """
        Determine the context for an image element based on its attributes.

        Args:
            elem: BeautifulSoup tag representing the image.
            _: Placeholder for context metadata (not used here).

        Returns:
            A string representing the image's context.
        """
        classes = elem.get('class', [])
        if 'figure' in classes:
            return 'figure'
        if 'diagram' in classes:
            return 'diagram'
        if 'screenshot' in classes:
            return 'screenshot'
        return 'default'

    def _determine_code_context(self, elem: Tag, _: Dict[str, Any]) -> str:
        """
        Determine the context for a code block element based on its attributes.

        Args:
            elem: BeautifulSoup tag representing the code block.
            _: Placeholder for context metadata (not used here).

        Returns:
            A string representing the code block's context.
        """
        classes = elem.get('class', [])
        if 'language-python' in classes:
            return 'python'
        if 'language-java' in classes:
            return 'java'
        if 'language-cpp' in classes:
            return 'cpp'
        return 'default'



    # STRATEGY-BASED METADATA EVALUATION AND VALIDATION

    def _evaluate_condition(
        self,
        condition: str,
        value: Any,
        context: Dict[str, Any]
    ) -> bool:
        """
        Dynamically evaluate a condition against a context using a strategy pattern.

        Args:
            condition: Type of condition (e.g., 'audience', 'platform', 'feature').
            value: Value to evaluate.
            context: Context dictionary for comparison.

        Returns:
            Boolean indicating whether the condition is satisfied.
        """
        try:
            strategies = {
                'version': self._strategy_version_condition,
                'audience': lambda v, c: bool(set(self._parse_condition_list(v)) & set(c.get('audience', []))),
                'platform': lambda v, c: bool(set(self._parse_condition_list(v)) & set(c.get('platform', []))),
                'region': lambda v, c: bool(set(self._parse_condition_list(v)) & set(c.get('region', []))),
                'product': lambda v, c: bool(set(self._parse_condition_list(v)) & set(c.get('product', []))),
                'feature': self._strategy_feature_condition,
                'default': lambda v, c: v == c.get(condition)
            }

            # Use the appropriate strategy or fallback to 'default'
            strategy = strategies.get(condition, strategies['default'])
            return strategy(value, context)

        except Exception as e:
            self.logger.error(f"Error evaluating condition '{condition}': {str(e)}")
            return True  # Default to true for safety

    # EVALUATION STRATEGIES

    def _strategy_version_condition(self, value: str, context: Dict[str, Any]) -> bool:
        """
        Strategy to evaluate version conditions.

        Args:
            value: The version condition value (e.g., "1.0-2.0").
            context: Context dictionary containing the current version.

        Returns:
            Boolean indicating whether the version condition is satisfied.
        """
        try:
            from packaging import version

            current_version = version.parse(context.get('version', '1.0'))
            if '-' in value:  # Handle version ranges (e.g., "1.0-2.0")
                min_version, _, max_version = value.partition('-')
                return version.parse(min_version) <= current_version <= version.parse(max_version or min_version)
            else:  # Single minimum version
                return version.parse(value) <= current_version
        except Exception as e:
            self.logger.error(f"Error evaluating version condition: {str(e)}")
            return True  # Default to true for safety

    def _strategy_feature_condition(self, feature: str, context: Dict[str, Any]) -> bool:
        """
        Strategy to evaluate feature flags.

        Args:
            feature: The feature to evaluate (e.g., 'index-numbers', 'toc').
            context: Context dictionary containing feature flags.

        Returns:
            Boolean indicating whether the feature is enabled.
        """
        try:
            # Check the feature flag in the context's 'features' section
            features = context.get('features', {})
            return features.get(feature, False)  # Default to False if the feature is not explicitly enabled
        except Exception as e:
            self.logger.error(f"Error evaluating feature condition for '{feature}': {str(e)}")
            return False


    # HELPER METHODS

    def _parse_condition_list(self, value: Union[str, List[str], None]) -> List[str]:
        """
        Parse condition lists into normalized strings.

        Args:
            value: The raw condition value, either a string, list, or None.

        Returns:
            A list of normalized strings.
        """
        if not value:
            return []
        if isinstance(value, str):
            return [v.strip() for v in value.split(',') if v.strip()]
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return []

    def _determine_reference_type(self, href: str) -> str:
        """Determine link reference type."""
        if href.startswith('conref:'):
            return 'conref'
        elif href.startswith(('http://', 'https://')):
            return 'external'
        elif href.startswith('#'):
            return 'local'
        elif href.startswith('[[') and href.endswith(']]'):
            return 'wiki'
        return 'internal'

    def _parse_highlight_lines(self, highlight_str: str) -> List[int]:
        """Parse code block line highlighting."""
        if not highlight_str:
            return []

        try:
            lines = []
            for part in highlight_str.split(','):
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    lines.extend(range(start, end + 1))
                else:
                    lines.append(int(part))
            return sorted(set(lines))
        except ValueError:
            return []




    # ==========================================================================
    # CLEANUP
    # ==========================================================================

    def cleanup(self) -> None:
       """Perform comprehensive cleanup of processor resources and state."""
       try:
           self.logger.debug("Starting Markdown element processor cleanup")

           # Reset core components
           self.id_handler = DITAIDHandler()

           # Clear processing caches
           self._clear_caches()

           # Reset element tracking
           self._clear_element_tracking()

           # Reset metadata tracking
           self._clear_metadata()

           # Reset state variables
           self._reset_state()

           self.logger.debug("Markdown element processor cleanup completed")

           # Validate cleanup
           if not self.validate_cleanup():
               raise RuntimeError("Cleanup validation failed")

       except Exception as e:
           self.logger.error(f"Markdown element processor cleanup failed: {str(e)}")
           raise

    def _clear_caches(self) -> None:
       """Clear all processor caches."""
       try:
           # Clear processed elements cache
           if hasattr(self, '_processed_elements'):
               self._processed_elements.clear()

           # Clear type mapping cache
           if hasattr(self, '_type_mapping_cache'):
               self._type_mapping_cache.clear()

           # Clear validation cache
           if hasattr(self, '_validation_cache'):
               self._validation_cache.clear()

           self.logger.debug("Cleared all processor caches")

       except Exception as e:
           self.logger.error(f"Error clearing caches: {str(e)}")
           raise

    def _clear_element_tracking(self) -> None:
       """Clear element tracking information."""
       try:
           # Clear tracked elements
           if hasattr(self, '_tracked_elements'):
               self._tracked_elements.clear()

           # Clear element hierarchy
           if hasattr(self, '_element_hierarchy'):
               self._element_hierarchy.clear()

           # Clear element references
           if hasattr(self, '_element_refs'):
               self._element_refs.clear()

           self.logger.debug("Cleared element tracking")

       except Exception as e:
           self.logger.error(f"Error clearing element tracking: {str(e)}")
           raise

    def _clear_metadata(self) -> None:
       """Clear metadata tracking and caches."""
       try:
           # Clear metadata cache
           if hasattr(self, '_metadata_cache'):
               self._metadata_cache.clear()

           # Clear custom metadata
           if hasattr(self, '_custom_metadata'):
               self._custom_metadata.clear()

           # Clear validation results
           if hasattr(self, '_metadata_validation'):
               self._metadata_validation.clear()

           self.logger.debug("Cleared metadata tracking")

       except Exception as e:
           self.logger.error(f"Error clearing metadata: {str(e)}")
           raise

    def _reset_state(self) -> None:
       """Reset all state variables to initial values."""
       try:
           # Reset processing state
           self._processing_depth = 0
           self._current_element_id = None

           # Reset flags
           self._processing_enabled = True
           self._validation_enabled = True

           # Reset counters
           self._processed_count = 0
           self._error_count = 0

           # Reset configuration
           self._config = {
               'strict_mode': False,
               'enable_caching': True,
               'enable_validation': True
           }

           self.logger.debug("Reset processor state")

       except Exception as e:
           self.logger.error(f"Error resetting state: {str(e)}")
           raise

    def validate_cleanup(self) -> bool:
       """Validate cleanup was successful."""
       try:
           # Check caches are empty
           caches_empty = (
               not hasattr(self, '_processed_elements') and
               not hasattr(self, '_type_mapping_cache') and
               not hasattr(self, '_validation_cache')
           )

           # Check tracking is reset
           tracking_reset = (
               not hasattr(self, '_tracked_elements') and
               not hasattr(self, '_element_hierarchy') and
               not hasattr(self, '_element_refs')
           )

           # Check metadata is cleared
           metadata_cleared = (
               not hasattr(self, '_metadata_cache') and
               not hasattr(self, '_custom_metadata') and
               not hasattr(self, '_metadata_validation')
           )

           # Check state is reset
           state_reset = (
               self._processing_depth == 0 and
               self._current_element_id is None and
               self._processing_enabled and
               self._validation_enabled
           )

           cleanup_successful = all([
               caches_empty,
               tracking_reset,
               metadata_cleared,
               state_reset
           ])

           self.logger.debug(f"Cleanup validation: {'successful' if cleanup_successful else 'failed'}")
           return cleanup_successful

       except Exception as e:
           self.logger.error(f"Error validating cleanup: {str(e)}")
           return False
