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
    MDElementInfo,
    MDElementContext,
    ElementAttributes,
    MDElementType,
    ProcessingPhase
)



class MarkdownElementProcessor:
    def __init__(self, content_processor: ContentProcessor):
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
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._custom_metadata: Dict[str, Dict[str, Any]] = {}
        self._metadata_validation: Dict[str, bool] = {}

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

    def process_element(self, elem: Tag, source_path: Optional[Path] = None) -> MDElementInfo:
        """
        Process a Markdown element with its processing rules.

        Args:
            elem: The BeautifulSoup tag to process
            source_path: Optional source file path for context

        Returns:
            MDElementInfo: Processed element information
        """
        try:
            # Get element type
            element_type = self._get_element_type(elem)

            # Get processing rules for this type
            rules = self._processing_rules.get(element_type,
                            self._processing_rules[MDElementType.UNKNOWN])

            # Create element context
            context = MDElementContext(
                parent_id=None,  # Will be set if parent exists
                element_type=elem.name,
                classes=rules['default_classes'],
                attributes=dict(elem.attrs),  # Convert attrs to dict
            )

            # Process attributes according to rules
            attributes = self._process_attributes(elem, rules)

            # Get element content
            content = self._get_element_content(elem)

            # Table specialization detection
            if elem.name == 'table':
                table_type = self._detect_table_specialization(elem)
                metadata = {
                    'table_info': {
                        'type': table_type,
                        'has_header': bool(elem.find('thead')),
                        'rows': len(elem.find_all('tr')),
                        'columns': len(elem.find_all('tr')[0].find_all(['td', 'th'])) if elem.find('tr') else 0
                    }
                }
                # Add specialized metadata based on type
                if table_type == 'bibliography':
                    metadata['table_info'].update({
                        'citation_format': elem.get('data-citation-format', 'apa'),
                        'sort_by': elem.get('data-sort', 'author')
                    })
                elif table_type == 'glossary':
                    metadata['table_info'].update({
                        'sort_by': elem.get('data-sort', 'term'),
                        'show_references': elem.get('data-show-refs', 'true') == 'true'
                    })
                elif table_type == 'metadata':
                    metadata['table_info'].update({
                        'visibility': elem.get('data-visibility', 'visible'),
                        'collapsible': elem.get('data-collapsible', 'false') == 'true'
                    })

            # Extract metadata
            metadata = self._extract_element_metadata(elem)

            # Apply any type-specific attribute mappings
            if 'attribute_mapping' in rules:
                for md_attr, html_attr in rules['attribute_mapping'].items():
                    if md_attr in elem.attrs:
                        attributes.custom_attrs[html_attr] = elem.attrs[md_attr]

            # Validate required attributes
            if 'required_attributes' in rules:
                for required_attr in rules['required_attributes']:
                    if required_attr not in attributes.custom_attrs:
                        self.logger.warning(
                            f"Missing required attribute '{required_attr}' "
                            f"for element type {element_type}"
                        )

            return MDElementInfo(
                type=element_type,
                content=content,
                attributes=attributes,
                context=context,
                metadata=metadata,
                level=int(elem.name[1]) if element_type == MDElementType.HEADING else None
            )

        except Exception as e:
            self.logger.error(f"Error processing Markdown element: {str(e)}")
            return self.content_processor.create_md_error_element(
                error=e,
                element_context=str(elem.name) if hasattr(elem, 'name') else None
            )

    def _process_attributes(
       self,
       elem: Tag,
       rules: Dict[str, Any]
    ) -> ElementAttributes:
       """
       Process element attributes according to rules.

       Args:
           elem: BeautifulSoup tag
           rules: Processing rules for the element type

       Returns:
           ElementAttributes: Processed attributes
       """
       try:
           # Generate ID
           element_id = elem.get('id', '')
           if isinstance(element_id, list):
               element_id = element_id[0]
           if not element_id:
               element_id = self.id_handler.generate_content_id(Path(str(elem.name)))

           # Get classes from rules and element
           classes = rules['default_classes'].copy()
           element_classes = elem.get('class', [])
           if isinstance(element_classes, str):
               classes.extend(element_classes.split())
           elif isinstance(element_classes, list):
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

           # Process custom attributes
           custom_attrs = {}

           # Add rule-defined attributes
           if 'attributes' in rules:
               custom_attrs.update(rules['attributes'])

           # Add element attributes with list handling
           for key, value in elem.attrs.items():
               if key not in {'id', 'class'}:
                   processed_value = value[0] if isinstance(value, list) else value
                   custom_attrs[key] = processed_value

           return ElementAttributes(
               id=str(element_id),
               classes=classes,
               custom_attrs=custom_attrs
           )

       except Exception as e:
           self.logger.error(f"Error processing attributes: {str(e)}")
           return ElementAttributes(
               id=str(self.id_handler.generate_id("error")),
               classes=[],
               custom_attrs={}
           )


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

    def _detect_table_specialization(self, table: Tag) -> str:
            """Detect table specialization based on context and attributes."""
            # Check explicit type attribute
            if table_type := table.get('data-table-type'):
                return table_type

            # Check classes
            classes = table.get('class', [])
            if isinstance(classes, str):
                classes = classes.split()

            specialization_classes = {
                'bibliography': {'bibliography', 'citations', 'references'},
                'glossary': {'glossary', 'terms', 'definitions'},
                'metadata': {'metadata', 'article-info', 'topic-meta'}
            }

            for spec_type, spec_classes in specialization_classes.items():
                if any(cls in classes for cls in spec_classes):
                    return spec_type

            # Content-based detection
            if table.find('thead'):
                headers = [th.get_text().lower() for th in table.find('thead').find_all('th')]
                if {'term', 'definition'}.issubset(headers):
                    return 'glossary'
                if {'author', 'year', 'title'}.issubset(headers):
                    return 'bibliography'
                if {'property', 'value'}.issubset(headers):
                    return 'metadata'

            return 'default'

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

    def _get_element_attributes(self, elem: Tag, rules: dict) -> ElementAttributes:
       """
       Extract and process element attributes based on processing rules.

       Args:
           elem: The BeautifulSoup tag
           rules: Processing rules for this element type

       Returns:
           ElementAttributes: Processed attributes for the element
       """
       try:
           # Generate or get element ID - handle potential list
           raw_id = elem.get('id', '')
           element_id = raw_id[0] if isinstance(raw_id, list) else raw_id
           if not element_id:
               element_id = self.id_handler.generate_content_id(Path(str(elem.name)))

           # Rest of implementation stays the same...
           classes = self._get_element_classes(elem)
           custom_attrs = {}

           if 'attributes' in rules:
               custom_attrs.update(rules['attributes'])

           if 'attribute_mapping' in rules:
               for md_attr, html_attr in rules['attribute_mapping'].items():
                   if md_attr in elem.attrs:
                       attr_value = elem.attrs[md_attr]
                       if isinstance(attr_value, list):
                           attr_value = attr_value[0]
                       custom_attrs[html_attr] = attr_value

           for attr, value in elem.attrs.items():
               if attr in {'id', 'class'}:
                   continue

               if isinstance(value, list):
                   value = value[0]

               if attr == 'href':
                   custom_attrs['href'] = value
                   if value.startswith(('http://', 'https://')):
                       custom_attrs['target'] = '_blank'
                       custom_attrs['rel'] = 'noopener noreferrer'
               else:
                   attr_name = attr if attr in rules.get('allowed_attributes', []) else f'data-{attr}'
                   custom_attrs[attr_name] = value

           if 'required_attributes' in rules:
               for required_attr in rules['required_attributes']:
                   if required_attr not in custom_attrs:
                       self.logger.warning(
                           f"Missing required attribute '{required_attr}' "
                           f"for element: {elem.name}"
                       )
                       custom_attrs[required_attr] = ''

           return ElementAttributes(
               id=str(element_id),  # Ensure string type
               classes=classes,
               custom_attrs=custom_attrs
           )

       except Exception as e:
           self.logger.error(f"Error extracting attributes: {str(e)}")
           return ElementAttributes(
               id=str(self.id_handler.generate_id("error")),  # Ensure string type
               classes=[],
               custom_attrs={}
           )

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

    def _default_element(self) -> MDElementInfo:
       """Return a default Markdown element for unprocessable cases."""
       return MDElementInfo(
           type=MDElementType.PARAGRAPH,
           content="",
           attributes=ElementAttributes(
               id=self.id_handler.generate_id("default"),
               classes=[],
               custom_attrs={}
           ),
           context=MDElementContext(
               parent_id=None,
               element_type="default",
               classes=[],
               attributes={}
           ),
           metadata={},
           level=None
       )

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

    def _extract_yaml_frontmatter(self, content: str) -> Dict[str, Any]:
        """
        Extract and validate YAML frontmatter from markdown content.
        Handles document-level metadata including:
        - Content info (title, slug, type)
        - Publication info (dates, version)
        - Authors and contributors
        - Categories and keywords
        - Media requirements
        - Delivery settings
        - Processing conditions
        """
        try:
            if not content.startswith('---'):
                return {}

            end_idx = content.find('---', 3)
            if end_idx == -1:
                return {}

            # Parse YAML block
            frontmatter = yaml.safe_load(content[3:end_idx])

            # Initialize metadata structure
            metadata = {
                'content': {
                    'title': frontmatter.get('title'),
                    'slug': frontmatter.get('slug'),
                    'content_type': frontmatter.get('content-type', 'article'),
                    'categories': frontmatter.get('categories', []),
                    'keywords': frontmatter.get('keywords', []),
                    'language': frontmatter.get('language', 'en-US'),
                    'region': frontmatter.get('region', 'Global'),
                    'abstract': frontmatter.get('abstract')
                },
                'publication': {
                    'publication_date': frontmatter.get('publication-date'),
                    'last_edited': frontmatter.get('last-edited'),
                    'version': frontmatter.get('version', '1.0'),
                    'status': frontmatter.get('status', 'draft'),
                    'revision_history': frontmatter.get('revision-history', [])
                },
                'contributors': {
                    'authors': [
                        {
                            'conref': author.get('conref'),
                            'name': author.get('name'),
                            'role': author.get('role', 'author')
                        }
                        for author in frontmatter.get('authors', [])
                    ],
                    'editor': frontmatter.get('editor'),
                    'reviewer': frontmatter.get('reviewer')
                },
                'delivery': {
                    'channel_web': frontmatter.get('delivery', {}).get('channel-web', True),
                    'channel_app': frontmatter.get('delivery', {}).get('channel-app', False),
                    'channel_print': frontmatter.get('delivery', {}).get('channel-print', False)
                },
                'media': {
                    'pdf_download': frontmatter.get('media', {}).get('pdf-download'),
                    'interactive_media': frontmatter.get('media', {}).get('interactive-media', False),
                    'video_embed': frontmatter.get('media', {}).get('video-embed'),
                    'simulation': frontmatter.get('media', {}).get('simulation')
                },
                'conditions': {
                    'audience': frontmatter.get('audience', []),
                    'subscription_required': frontmatter.get('features', {}).get('subscription-required', False),
                    'featured': frontmatter.get('features', {}).get('featured', False),
                    'accessibility_compliant': frontmatter.get('accessibility-compliant')
                },
                'analytics': frontmatter.get('analytics', {}),
                'specialization': {
                            'object': frontmatter.get('object', ''),
                            'role': frontmatter.get('role', ''),
                        },


            }

            # Validate required fields
            if not metadata['content']['title']:
                self.logger.warning("Missing required field: title")

            if not metadata['content']['slug']:
                # Generate slug from title if missing
                title = metadata['content']['title'] or ''
                metadata['content']['slug'] = re.sub(
                    r'[^\w\-]',
                    '-',
                    title.lower()
                ).strip('-')

            return metadata

        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML frontmatter: {str(e)}")
            return {}
        except Exception as e:
            self.logger.error(f"Error processing frontmatter: {str(e)}")
            return {}

    # SECONDARY METADATA EXTRACTION

    def _extract_element_metadata(
        self,
        elem: Tag,
        document_metadata: Dict[str, Any],
        map_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract element-level metadata with document context.
        Handles:
        - Element information
        - Content status
        - Conditional processing
        - Content features
        - Element specific data (images, code, etc.)

        Args:
            elem: BeautifulSoup tag to process
            document_metadata: YAML frontmatter context
        """
        try:
            element_type = self._get_element_type(elem)

            # Core element metadata
            metadata = {
                'element': {
                    'type': element_type.value,
                    'tag': elem.name,
                    'id': elem.get('id') or self.id_handler.generate_id(str(elem.name)),
                    'classes': elem.get('class', []),
                    'title': elem.get('title'),
                    'language': elem.get('lang', document_metadata.get('content', {}).get('language', 'en-US')),
                    'has_children': bool(elem.find_all()),
                    'has_text': bool(elem.string)
                },
                'content': {
                    'word_count': len(elem.get_text().split()) if elem.string else 0,
                    'text_direction': elem.get('dir', 'ltr'),
                    'is_block': self._is_block_element(elem)
                },
                'processing': {
                    'state': 'pending',
                    'visibility': True,  # Default to visible
                    'priority': elem.get('data-priority', 'normal')
                },
                'specialization': {
                    'object': document_metadata.get('specialization', {}).get('object', ''),
                    'role': document_metadata.get('specialization', {}).get('role', '')
                },
                'context': self._determine_element_context(elem),
                'conditions': self._extract_conditional_metadata(elem, document_metadata, map_metadata),
            }

            # Add conditional processing based on document settings
            if conditions := document_metadata.get('conditions'):
                metadata['conditions'] = {
                    'audience': elem.get('data-audience', conditions.get('audience', [])),
                    'subscription_required': elem.get(
                        'data-subscription',
                        conditions.get('subscription_required', False)
                    ),
                    'platform': elem.get('data-platform', []),
                    'product': elem.get('data-product', [])
                }

            # Handle element-specific metadata
            elif element_type == MDElementType.IMAGE:
                metadata['media'] = {
                    'src': elem.get('src', ''),
                    'alt': elem.get('alt', ''),
                    'title': elem.get('title', ''),
                    'width': elem.get('width'),
                    'height': elem.get('height'),
                    'loading': elem.get('loading', 'lazy'),
                    'interactive': document_metadata.get('media', {}).get('interactive_media', False)
                }


            elif element_type == MDElementType.CODE_BLOCK:
                metadata['code'] = {
                    'language': self._get_code_language(elem),
                    'line_count': len(elem.get_text().splitlines()),
                    'is_mermaid': bool(self._get_code_language(elem) == 'mermaid'),
                    'show_line_numbers': elem.get('data-show-line-numbers', 'true').lower() == 'true'
                }

            elif element_type == MDElementType.LINK:
                href = str(elem.get('href', ''))
                metadata['link'] = {
                    'href': href,
                    'title': elem.get('title'),
                    'is_external': href.startswith(('http://', 'https://')),
                    'is_conref': href.startswith('conref:'),
                    'target': elem.get('target', '_blank' if href.startswith(('http://', 'https://')) else None)

                }

            # Handle custom data attributes
            custom_attrs = {
                k: v for k, v in elem.attrs.items()
                if k.startswith('data-') and k not in {
                    'data-audience', 'data-subscription',
                    'data-platform', 'data-product',
                    'data-priority', 'data-show-line-numbers'
                }
            }
            if custom_attrs:
                metadata['custom'] = custom_attrs

            # Add accessibility information if required
            if document_metadata.get('conditions', {}).get('accessibility_compliant'):
                metadata['accessibility'] = {
                    'role': elem.get('role'),
                    'aria-label': elem.get('aria-label'),
                    'aria-describedby': elem.get('aria-describedby')
                }

            return metadata

        except Exception as e:
            self.logger.error(f"Error extracting element metadata: {str(e)}")
            return self.content_processor.create_md_error_element(
                error=e,
                element_context=str(elem.name)
            )


    def _extract_element_attributes(
        self,
        elem: Tag,
        document_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract and normalize element attributes with document context.

        Args:
            elem: BeautifulSoup tag to process
            document_metadata: YAML frontmatter context

        Returns:
            Dict containing:
            - Core attributes (id, class, etc.)
            - Conditional attributes (audience, platform, etc.)
            - Processing attributes (state, priority)
            - Feature flags
            - Custom data attributes
        """
        try:
            # Initialize base attributes structure
            attributes = {
                'core': {},
                'conditional': {},
                'processing': {},
                'features': {},
                'custom': {}
            }

            # Process core attributes with proper list handling
            core_attrs = {
                'id': elem.get('id'),
                'class': elem.get('class'),
                'title': elem.get('title'),
                'style': elem.get('style'),
                'lang': elem.get('lang', document_metadata.get('content', {}).get('language')),
                'dir': elem.get('dir', 'ltr')
            }

            # Clean and normalize core attributes
            for key, value in core_attrs.items():
                if value is not None:
                    # Handle list attributes (like classes)
                    if isinstance(value, list):
                        attributes['core'][key] = ' '.join(value)
                    else:
                        attributes['core'][key] = str(value)

            # Process conditional attributes based on document settings
            document_conditions = document_metadata.get('conditions', {})
            if document_conditions:
                conditional_attrs = {
                    'audience': elem.get('data-audience', document_conditions.get('audience', [])),
                    'platform': elem.get('data-platform', []),
                    'product': elem.get('data-product', []),
                    'revision': elem.get('data-revision'),
                    'importance': elem.get('data-importance', 'normal'),
                    'visibility': elem.get('data-visibility', 'show')
                }

                # Clean and normalize conditional attributes
                for key, value in conditional_attrs.items():
                    if value:
                        if isinstance(value, list):
                            attributes['conditional'][key] = [str(v) for v in value]
                        else:
                            attributes['conditional'][key] = str(value)

            # Process processing attributes
            processing_attrs = {
                'state': elem.get('data-state', 'active'),
                'priority': elem.get('data-priority', 'normal'),
                'processing-role': elem.get('data-processing-role', 'normal')
            }
            attributes['processing'].update(processing_attrs)

            # Process feature flags
            document_features = document_metadata.get('features', {})
            if document_features:
                feature_attrs = {
                    'interactive': elem.get('data-interactive', document_features.get('interactive_media', False)),
                    'subscription-required': elem.get('data-subscription', document_features.get('subscription_required', False)),
                    'show-line-numbers': str(elem.get('data-show-line-numbers', 'true')).lower() == 'true',
                    'mermaid': elem.get('data-mermaid', False)
                }
                attributes['features'].update(feature_attrs)

            # Handle accessibility attributes if compliance is required
            if document_metadata.get('conditions', {}).get('accessibility_compliant'):
                accessibility_attrs = {
                    'role': elem.get('role'),
                    'aria-label': elem.get('aria-label'),
                    'aria-describedby': elem.get('aria-describedby'),
                    'aria-hidden': elem.get('aria-hidden'),
                    'tabindex': elem.get('tabindex')
                }
                # Only add non-None accessibility attributes
                attributes['core'].update({k: v for k, v in accessibility_attrs.items() if v is not None})

            # Process remaining data-* attributes as custom
            custom_attrs = {
                k: v for k, v in elem.attrs.items()
                if k.startswith('data-') and k not in {
                    'data-audience', 'data-platform', 'data-product',
                    'data-state', 'data-priority', 'data-processing-role',
                    'data-interactive', 'data-subscription', 'data-show-line-numbers',
                    'data-mermaid', 'data-revision', 'data-importance', 'data-visibility'
                }
            }
            if custom_attrs:
                attributes['custom'].update(custom_attrs)

            return attributes

        except Exception as e:
            self.logger.error(f"Error extracting element attributes: {str(e)}")
            return {
                'core': {'error': str(e)},
                'conditional': {},
                'processing': {},
                'features': {},
                'custom': {}
            }

    def _extract_element_specific_metadata(
        self,
        elem: Tag,
        element_type: MDElementType,
        document_metadata: Dict[str, Any],
        specialization: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract element-specific metadata with type-based handling.

        Args:
            elem: BeautifulSoup tag to process
            element_type: Type of markdown element
            document_metadata: YAML frontmatter context

        Returns:
            Dict containing element-specific metadata
        """
        try:
            # Element type specific extraction
            if element_type == MDElementType.IMAGE:
                return {
                    'media': {
                        'src': elem.get('src', ''),
                        'alt': elem.get('alt', ''),
                        'title': elem.get('title', ''),
                        'width': elem.get('width'),
                        'height': elem.get('height'),
                        'loading': elem.get('loading', 'lazy'),
                        'interactive': document_metadata.get('media', {}).get('interactive_media', False),
                        'figure_id': self.id_handler.generate_id(f"fig-{elem.get('alt', '')}"),
                        'caption': elem.parent.find('figcaption').get_text() if elem.parent and elem.parent.name == 'figure' and elem.parent.find('figcaption') else ''
                    }
                }

            elif element_type == MDElementType.LINK:
                href = str(elem.get('href', ''))
                return {
                    'link': {
                        'href': href,
                        'title': elem.get('title'),
                        'text': elem.get_text().strip(),
                        'is_external': href.startswith(('http://', 'https://')),
                        'is_conref': href.startswith('conref:'),
                        'target': elem.get('target', '_blank' if href.startswith(('http://', 'https://')) else None),
                        'rel': elem.get('rel', 'noopener noreferrer' if href.startswith(('http://', 'https://')) else None),
                        'reference_type': self._determine_reference_type(href)
                    }
                }

            elif element_type == MDElementType.CODE_BLOCK:
                language = self._get_code_language(elem)
                return {
                    'code': {
                        'language': language,
                        'line_count': len(elem.get_text().splitlines()),
                        'is_mermaid': language == 'mermaid',
                        'show_line_numbers': str(elem.get('data-show-line-numbers', 'true')).lower() == 'true',
                        'highlight_lines': self._parse_highlight_lines(str(elem.get('data-highlight-lines', ''))),
                        'caption': elem.get('data-caption'),
                        'executable': str(elem.get('data-executable', 'false')).lower() == 'true'
                    }
                }

            elif element_type == MDElementType.HEADING:
                level = int(elem.name[1])  # h1 -> 1, h2 -> 2, etc.
                return {
                    'heading': {
                        'level': level,
                        'text': elem.get_text().strip(),
                        'is_topic_title': level == 1 and elem.get('class', []) == ['title'],
                        'numbering_enabled': document_metadata.get('features', {}).get('index-numbers', True),
                        'anchor_id': self.id_handler.generate_id(elem.get_text().strip()),
                        'toc_enabled': level <= 3 and document_metadata.get('features', {}).get('show-toc', True)
                    }
                }

            elif element_type == MDElementType.TABLE:
                return {
                    'table': {
                        'has_header': bool(elem.find('thead')),
                        'rows': len(elem.find_all('tr')),
                        'columns': len(elem.find_all('tr')[0].find_all(['td', 'th'])) if elem.find('tr') else 0,
                        'caption': elem.find('caption').get_text() if elem.find('caption') else '',
                        'responsive': 'table-responsive' in elem.get('class', []),
                        'bordered': 'table-bordered' in elem.get('class', [])
                    }
                }

            elif element_type == MDElementType.BLOCKQUOTE:
                return {
                    'quote': {
                        'text': elem.get_text().strip(),
                        'cite': elem.get('cite'),
                        'author': elem.find('cite').get_text() if elem.find('cite') else '',
                        'is_pullquote': 'pullquote' in elem.get('class', []),
                        'is_callout': 'callout' in elem.get('class', [])
                    }
                }

            elif element_type == MDElementType.TODO:
                text = elem.get_text().strip()
                return {
                    'todo': {
                        'is_checked': text.startswith('[x]'),
                        'text': text[3:].strip() if text.startswith(('[x]', '[ ]')) else text,
                        'priority': elem.get('data-priority', 'normal'),
                        'due_date': elem.get('data-due-date'),
                        'assigned_to': elem.get('data-assigned-to')
                    }
                }

            # Return empty dict for unsupported types
            return {}

        except Exception as e:
            self.logger.error(f"Error extracting specific metadata for {element_type}: {str(e)}")
            return {}

    def _determine_element_context(self, elem: Tag) -> str:
        element_type = self._get_element_type(elem)
        if element_type == MDElementType.TABLE:
            return self._determine_table_context(elem)
        elif element_type == MDElementType.IMAGE:
            return self._determine_image_context(elem)
        elif element_type == MDElementType.CODE_BLOCK:
            return self._determine_code_context(elem)
        # Add more cases for other element types as needed
        return 'default'

    def _determine_table_context(self, table: Tag) -> str:
        if table.find('thead'):
            headers = [th.get_text().lower() for th in table.find('thead').find_all('th')]
            if {'term', 'definition'}.issubset(headers):
                return 'glossary'
            if {'author', 'year', 'title'}.issubset(headers):
                return 'bibliography'
            if {'property', 'value'}.issubset(headers):
                return 'metadata'
        return 'default'

    def _determine_image_context(self, image: Tag) -> str:
        if 'figure' in image.get('class', []):
            return 'figure'
        if 'diagram' in image.get('class', []):
            return 'diagram'
        if 'screenshot' in image.get('class', []):
            return 'screenshot'
        return 'default'

    def _determine_code_context(self, code: Tag) -> str:
        if 'language-python' in code.get('class', []):
            return 'python'
        if 'language-java' in code.get('class', []):
            return 'java'
        if 'language-cpp' in code.get('class', []):
            return 'cpp'
        return 'default'

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

    def _extract_conditional_metadata(
        self,
        elem: Tag,
        document_metadata: Dict[str, Any],
        map_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        conditions = {}
        """
        Extract conditional processing metadata with document context.
        Handles DITA-style conditional processing including:
        - Audience/Platform/Product conditions
        - Feature flags and toggles
        - Processing instructions
        - Version control
        - Delivery channels

        Args:
            elem: BeautifulSoup tag to process
            document_metadata: YAML frontmatter context

        Returns:
            Dict containing conditional processing directives
        """
        try:
            # Get document-level conditions
            doc_conditions = document_metadata.get('conditions', {})
            doc_features = document_metadata.get('features', {})
            doc_delivery = document_metadata.get('delivery', {})

            conditions = {
                # Audience targeting
                'audience': {
                    'groups': self._parse_condition_list(
                        elem.get('data-audience', doc_conditions.get('audience', []))
                    ),
                    'subscription_required': elem.get(
                        'data-subscription',
                        doc_conditions.get('subscription_required', False)
                    ),
                    'expertise_level': elem.get('data-expertise', 'all')
                },

                # Platform/Product conditions
                'platform': {
                    'os': self._parse_condition_list(
                        elem.get('data-platform', [])
                    ),
                    'browser': self._parse_condition_list(
                        elem.get('data-browser', [])
                    ),
                    'device': self._parse_condition_list(
                        elem.get('data-device', [])
                    )
                },

                # Version control
                'version': {
                    'min_version': elem.get(
                        'data-min-version',
                        document_metadata.get('publication', {}).get('version')
                    ),
                    'max_version': elem.get('data-max-version'),
                    'revision': elem.get(
                        'data-revision',
                        document_metadata.get('publication', {}).get('revision')
                    )
                },

                # Delivery channels
                'delivery': {
                    'web': elem.get(
                        'data-web-delivery',
                        doc_delivery.get('channel_web', True)
                    ),
                    'app': elem.get(
                        'data-app-delivery',
                        doc_delivery.get('channel_app', False)
                    ),
                    'print': elem.get(
                        'data-print-delivery',
                        doc_delivery.get('channel_print', False)
                    )
                },

                # Feature toggles
                'features': {
                    'interactive': elem.get(
                        'data-interactive',
                        doc_features.get('interactive_media', False)
                    ),
                    'animated': elem.get('data-animated', False),
                    'executable': elem.get('data-executable', False)
                },

                # Processing instructions
                'processing': {
                    'visibility': elem.get('data-visibility', 'show'),
                    'importance': elem.get('data-importance', 'normal'),
                    'processing_role': elem.get('data-processing-role', 'normal'),
                    'render_mode': elem.get('data-render', 'default')
                },

                # Region/Language conditions
                'localization': {
                    'regions': self._parse_condition_list(
                        elem.get('data-region',
                        document_metadata.get('content', {}).get('region', ['Global']))
                    ),
                    'languages': self._parse_condition_list(
                        elem.get('data-language',
                        document_metadata.get('content', {}).get('language', ['en-US']))
                    )
                }
            }

            # Extract custom conditional attributes
            custom_conditions = {
                k.replace('data-condition-', ''): v
                for k, v in elem.attrs.items()
                if k.startswith('data-condition-')
            }
            if custom_conditions:
                conditions['custom'] = custom_conditions

            return conditions

        except Exception as e:
            self.logger.error(f"Error extracting conditional metadata: {str(e)}")
            return {
                'processing': {'visibility': 'show'},  # Safe default
                'error': str(e)
            }

    def _parse_condition_list(self, value: Union[str, List[str], None]) -> List[str]:
        """Parse condition list from string or list."""
        if not value:
            return []
        if isinstance(value, str):
            return [v.strip() for v in value.split(',') if v.strip()]
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return []

    def _evaluate_condition(
        self,
        condition: str,
        value: Any,
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate a single condition against context."""
        try:
            if condition == 'version':
                return self._evaluate_version_condition(value, context)
            elif condition in ['audience', 'platform', 'region']:
                return bool(set(self._parse_condition_list(value)) &
                            set(context.get(condition, [])))
            else:
                return value == context.get(condition)
        except Exception as e:
            self.logger.error(f"Error evaluating condition {condition}: {str(e)}")
            return True  # Safe default - show content on error

    def _evaluate_version_condition(
        self,
        version_value: str,
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate version-based condition."""
        try:
            from packaging import version
            doc_version = version.parse(context.get('version', '1.0'))

            if '-' in version_value:
                min_ver, max_ver = version_value.split('-')
                return (
                    version.parse(min_ver) <= doc_version <= version.parse(max_ver)
                    if max_ver else
                    version.parse(min_ver) <= doc_version
                )
            return version.parse(version_value) <= doc_version
        except Exception as e:
            self.logger.error(f"Error evaluating version condition: {str(e)}")
            return True



    # OLD METHODS TO REVISE


    def _is_root_element(self, elem: Tag) -> bool:
        """Check if element is a root-level element."""
        return elem.name in {'article', 'section', 'main'} or 'data-root' in elem.attrs

    def _extract_image_metadata(self, elem: Tag) -> Dict[str, Any]:
       """Extract image-specific metadata."""
       return {
           'media_info': {
               'src': elem.get('src', '')[0] if isinstance(elem.get('src'), list) else elem.get('src', ''),
               'alt': elem.get('alt', '')[0] if isinstance(elem.get('alt'), list) else elem.get('alt', ''),
               'title': elem.get('title', '')[0] if isinstance(elem.get('title'), list) else elem.get('title', ''),
               'width': elem.get('width'),
               'height': elem.get('height'),
               'loading': elem.get('loading', 'lazy')
           }
       }

    def _extract_link_metadata(self, elem: Tag) -> Dict[str, Any]:
       """Extract link metadata."""
       href = elem.get('href', '')[0] if isinstance(elem.get('href'), list) else elem.get('href', '')
       is_external = isinstance(href, str) and href.startswith(('http://', 'https://'))
       is_wiki = isinstance(href, str) and href.startswith('[[') and href.endswith(']]')

       return {
           'link_info': {
               'href': href,
               'title': elem.get('title'),
               'is_external': is_external,
               'is_wiki_link': is_wiki,
               'target': elem.get('target', '_blank' if is_external else None)
           }
       }

    def _extract_code_metadata(self, elem: Tag) -> Dict[str, Any]:
       """Extract code block metadata."""
       show_lines = elem.get('data-show-line-numbers', 'true')
       if isinstance(show_lines, list):
           show_lines = show_lines[0]

       return {
           'code_info': {
               'language': self._get_code_language(elem),
               'line_count': len(elem.get_text().splitlines()),
               'is_mermaid': bool(self._get_code_language(elem) == 'mermaid'),
               'show_line_numbers': str(show_lines).lower() == 'true'
           }
       }

    def _extract_todo_metadata(self, elem: Tag) -> Dict[str, Any]:
       """Extract todo item metadata."""
       text = elem.get_text().strip()
       return {
           'todo_info': {
               'is_checked': text.startswith('[x]'),
               'priority': elem.get('data-priority'),
               'due_date': elem.get('data-due-date'),
               'assigned_to': elem.get('data-assigned-to')
           }
       }


    def _get_element_path(self, elem: Tag) -> str:
       """Get element path similar to XPath."""
       try:
           path_parts = []
           for parent in elem.parents:
               if not isinstance(parent, Tag):
                   continue
               path_parts.append(parent.name)
           path_parts.reverse()
           path_parts.append(elem.name)
           return '/' + '/'.join(path_parts)
       except Exception as e:
           self.logger.error(f"Error getting element path: {str(e)}")
           return ""


    def _is_block_element(self, elem: Tag) -> bool:
       """Determine if element is a block-level element."""
       block_elements = {
           'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
           'ul', 'ol', 'pre', 'blockquote', 'table'
       }
       return elem.name in block_elements



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
