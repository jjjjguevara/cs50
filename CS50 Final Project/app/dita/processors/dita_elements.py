# app/dita/utils/dita_elements.py
from typing import Dict, Optional, List, Any
from datetime import datetime
from pathlib import Path
import logging
from lxml import etree
from app.dita.utils.id_handler import DITAIDHandler
from app.dita.models.types import (
    DITAElementType,
    DITAElementInfo,
    DITAElementContext,
    ElementAttributes,
    ElementType,
    ProcessingPhase
)
from app.dita.processors.content_processors import ContentProcessor

class DITAElementProcessor:
    """
    Processes DITA elements into structured information for transformation.
    Maps DITA XML elements to their HTML counterparts without transforming them.
    """
    def __init__(self, content_processor: ContentProcessor):
        self.logger = logging.getLogger(__name__)
        self.id_handler = DITAIDHandler()
        self._processed_elements: Dict[str, Any] = {}
        self.content_processor = content_processor
        self.current_phase = ProcessingPhase.DISCOVERY

        # Initialize processing caches
        self._processed_elements: Dict[str, Any] = {}
        self._memoized_results: Dict[str, Any] = {}
        self._type_mapping_cache: Dict[str, DITAElementType] = {}
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
        self._current_topic_id: Optional[str] = None
        self._current_element_id: Optional[str] = None
        self._processing_enabled: bool = True
        self._validation_enabled: bool = True
        self._processed_count: int = 0
        self._error_count: int = 0

        # Initialize configuration
        self._config = {
            'strict_mode': False,
            'enable_caching': True,
            'enable_validation': True
        }

        # Defines HTML output mappings for each DITA element type
        self._processing_rules: Dict[DITAElementType, Dict[str, Any]] = {
            # Structure elements
            DITAElementType.CONCEPT: {
                'html_tag': 'article',
                'default_classes': ['dita-concept', 'article-content'],
                'attributes': {'role': 'article'},
                'content_wrapper': 'div',
                'wrapper_classes': ['concept-content']
            },
            DITAElementType.TASK: {
                'html_tag': 'article',
                'default_classes': ['dita-task', 'article-content'],
                'attributes': {'role': 'article'},
                'content_wrapper': 'div',
                'wrapper_classes': ['task-content']
            },
            DITAElementType.REFERENCE: {
                'html_tag': 'article',
                'default_classes': ['dita-reference', 'article-content'],
                'attributes': {'role': 'article'},
                'content_wrapper': 'div',
                'wrapper_classes': ['reference-content']
            },

            # Block elements
            DITAElementType.SECTION: {
                'html_tag': 'section',
                'default_classes': ['dita-section'],
                'attributes': {'role': 'region'}
            },
            DITAElementType.PARAGRAPH: {
                'html_tag': 'p',
                'default_classes': ['dita-p']
            },
            DITAElementType.NOTE: {
                'html_tag': 'div',
                'default_classes': ['dita-note', 'alert'],
                'type_class_mapping': {
                    'warning': 'alert-warning',
                    'danger': 'alert-danger',
                    'tip': 'alert-info',
                    'note': 'alert-secondary'
                },
                'attributes': {'role': 'alert'}
            },
            DITAElementType.CODE_BLOCK: {
                'html_tag': 'pre',
                'default_classes': ['code-block', 'highlight'],
                'attributes': {
                    'spellcheck': 'false',
                    'translate': 'no'
                },
                'inner_tag': 'code',
                'inner_classes': ['language-{outputclass}']
            },

            # List elements
            DITAElementType.LIST: {
                'html_tag': 'ul',
                'default_classes': ['dita-ul']
            },
            DITAElementType.ORDERED_LIST: {
                'html_tag': 'ol',
                'default_classes': ['dita-ol']
            },
            DITAElementType.LIST_ITEM: {
                'html_tag': 'li',
                'default_classes': ['dita-li']
            },

            # Table elements
            DITAElementType.TABLE: {
                'html_tag': 'table',
                'default_classes': ['table', 'table-bordered'],
                'attributes': {
                    'role': 'grid',
                    'aria-label': 'Content Table'
                }
            },

            # Media elements
            DITAElementType.FIGURE: {
                'html_tag': 'figure',
                'default_classes': ['figure', 'dita-figure'],
                'inner_tag': 'figcaption',
                'caption_classes': ['figure-caption']
            },
            DITAElementType.IMAGE: {
                'html_tag': 'img',
                'default_classes': ['img-fluid', 'dita-image'],
                'required_attributes': ['src'],
                'attribute_mapping': {
                    'href': 'src',
                    'alt': 'alt'
                },
                'attributes': {
                    'loading': 'lazy',
                    'decoding': 'async'
                }
            },

            # Link elements
            DITAElementType.XREF: {
                'html_tag': 'a',
                'default_classes': ['dita-xref'],
                'required_attributes': ['href'],
                'attribute_mapping': {
                    'href': 'href',
                    'scope': 'data-scope',
                    'format': 'data-format'
                },
                'attributes': {
                    'target': '_self',
                    'rel': 'noopener'
                }
            },
            DITAElementType.LINK: {
                'html_tag': 'a',
                'default_classes': ['dita-link'],
                'required_attributes': ['href'],
                'attribute_mapping': {
                    'href': 'href',
                    'scope': 'data-scope',
                    'format': 'data-format'
                },
                'attributes': {
                    'target': '_blank',
                    'rel': 'noopener noreferrer'
                }
            },

            # Heading elements
            DITAElementType.TITLE: {
                'html_tag': 'h1',
                'default_classes': ['dita-title']
            },
            DITAElementType.SHORTDESC: {
                'html_tag': 'p',
                'default_classes': ['lead', 'dita-shortdesc']
            },

            # Task-specific elements
            DITAElementType.PREREQ: {
                'html_tag': 'div',
                'default_classes': ['prerequisites', 'alert', 'alert-warning'],
                'attributes': {'role': 'alert'}
            },
            DITAElementType.STEPS: {
                'html_tag': 'div',
                'default_classes': ['steps-container']
            },
            DITAElementType.STEP: {
                'html_tag': 'div',
                'default_classes': ['step'],
                'content_wrapper': 'div',
                'wrapper_classes': ['step-content']
            },
            DITAElementType.CMD: {
                'html_tag': 'div',
                'default_classes': ['command']
            },
            DITAElementType.INFO: {
                'html_tag': 'div',
                'default_classes': ['step-info']
            },
            DITAElementType.SUBSTEP: {
                'html_tag': 'div',
                'default_classes': ['substep']
            },
            DITAElementType.SUBSTEPS: {
                'html_tag': 'div',
                'default_classes': ['substeps-container']
            },

            # Inline elements
            DITAElementType.BOLD: {
                'html_tag': 'strong',
                'default_classes': ['dita-b']
            },
            DITAElementType.ITALIC: {
                'html_tag': 'em',
                'default_classes': ['dita-i']
            },
            DITAElementType.UNDERLINE: {
                'html_tag': 'u',
                'default_classes': ['dita-u']
            },
            DITAElementType.PHRASE: {
                'html_tag': 'span',
                'default_classes': ['dita-ph']
            },
            DITAElementType.QUOTE: {
                'html_tag': 'blockquote',
                'default_classes': ['dita-quote']
            },
            DITAElementType.CITE: {
                'html_tag': 'cite',
                'default_classes': ['dita-cite']
            },

            # Definition elements
            DITAElementType.DEFINITION: {
                'html_tag': 'dl',
                'default_classes': ['dita-dlentry']
            },
            DITAElementType.TERM: {
                'html_tag': 'dt',
                'default_classes': ['dita-dt']
            },

            # Metadata elements
            DITAElementType.METADATA: {
                'html_tag': 'div',
                'default_classes': ['metadata-section'],
                'attributes': {'aria-hidden': 'true'}
            },

            # Navigation elements
            DITAElementType.TOPICREF: {
                'html_tag': 'div',
                'default_classes': ['topic-ref']
            },
            DITAElementType.TOPICGROUP: {
                'html_tag': 'div',
                'default_classes': ['topic-group']
            },

            # Default for unknown elements
            DITAElementType.UNKNOWN: {
                'html_tag': 'div',
                'default_classes': ['dita-unknown']
            }
        }

    def set_processing_phase(self, phase: ProcessingPhase) -> None:
        """Update current processing phase."""
        self.current_phase = phase
        self.logger.debug(f"Updated processing phase to {phase.value}")

    def process_elements(self, xml_tree: etree._Element) -> List[DITAElementInfo]:
        """
        Process all elements in an XML tree.

        Args:
            xml_tree: The XML tree to process

        Returns:
            List[DITAElementInfo]: List of processed elements with HTML mapping rules
        """
        try:
            processed_elements = []

            # Process each element in the tree
            for elem in xml_tree.iter():
                if isinstance(elem.tag, str):  # Skip processing instructions/comments
                    # Get element type
                    element_type = self._get_element_type(elem)

                    # Get processing rules for this type
                    rules = self._processing_rules.get(element_type,
                                self._processing_rules[DITAElementType.UNKNOWN])

                    # Create element context with proper dict conversion
                    context = DITAElementContext(
                        parent_id=None,  # Will be set if parent exists
                        element_type=elem.tag,
                        classes=rules['default_classes'],
                        attributes=dict(elem.attrib),  # Convert to dict
                        topic_type=self._get_topic_type(elem),
                        is_body=self._is_body_element(elem)
                    )

                    # Process element with rules
                    element_info = self.process_element(elem, rules, context)
                    processed_elements.append(element_info)

            return processed_elements

        except Exception as e:
            self.logger.error(f"Error processing XML tree: {str(e)}")
            raise

    def process_element(
            self,
            elem: etree._Element,
            rules: Dict[str, Any],
            context: DITAElementContext
        ) -> DITAElementInfo:
        """
        Process a single DITA element with its processing rules.

        Args:
            elem: The element to process
            rules: Processing rules for this element type
            context: Element's processing context

        Returns:
            DITAElementInfo: Processed element information
        """
        try:
            # Get element type
            element_type = self._get_element_type(elem)

            # Process attributes according to rules
            attributes = self._process_attributes(elem, rules)

            # Get element content
            content = self._get_element_content(elem)

            # Process children if any
            children = [
                self.process_element(
                    child,
                    self._processing_rules.get(
                        self._get_element_type(child),
                        self._processing_rules[DITAElementType.UNKNOWN]
                    ),
                    context.replace(parent_id=attributes.id)
                )
                for child in elem
                if isinstance(child.tag, str)
            ]

            # Extract metadata
            metadata = self._get_element_metadata(elem)

            # Apply any type-specific attribute mappings
            if 'attribute_mapping' in rules:
                for dita_attr, html_attr in rules['attribute_mapping'].items():
                    if dita_attr in elem.attrib:
                        attributes.custom_attrs[html_attr] = elem.attrib[dita_attr]

            # Validate required attributes
            if 'required_attributes' in rules:
                for required_attr in rules['required_attributes']:
                    if required_attr not in attributes.custom_attrs:
                        self.logger.warning(
                            f"Missing required attribute '{required_attr}' "
                            f"for element type {element_type}"
                        )

            return DITAElementInfo(
                type=element_type,
                content=content,
                attributes=attributes,
                context=context,
                metadata=metadata,
                children=children
            )

        except Exception as e:
            self.logger.error(f"Error processing DITA element: {str(e)}")
            return self.content_processor.create_dita_error_element(
                error=e,
                element_context=str(elem.tag) if hasattr(elem, 'tag') else None
            )

    def _process_attributes(
       self,
       elem: etree._Element,
       rules: Dict[str, Any]
   ) -> ElementAttributes:
       try:
           # Convert attrib to dict for type safety
           elem_attrs = dict(elem.attrib)

           # Generate ID
           element_id = elem_attrs.get("id") or self.id_handler.generate_content_id(
               Path(str(elem.tag))
           )

           # Get classes from rules and element
           classes = rules['default_classes'].copy()
           if elem_attrs.get('class'):
               classes.extend(elem_attrs['class'].split())

           # Add type-specific classes if applicable
           if 'type_class_mapping' in rules and 'type' in elem_attrs:
               type_class = rules['type_class_mapping'].get(
                   elem_attrs['type'],
                   rules['type_class_mapping'].get('default', '')
               )
               if type_class:
                   classes.append(type_class)

           # Process custom attributes
           custom_attrs = {}
           # Add rule-defined attributes
           if 'attributes' in rules:
               custom_attrs.update(rules['attributes'])
           # Add element attributes
           custom_attrs.update(elem_attrs)

           return ElementAttributes(
               id=element_id,
               classes=classes,
               custom_attrs=custom_attrs
           )

       except Exception as e:
           self.logger.error(f"Error processing attributes: {str(e)}")
           return ElementAttributes(id="", classes=[], custom_attrs={})


    def _process_images(self, element_info: DITAElementInfo) -> DITAElementInfo:
        """
        Process image elements according to rules.

        Args:
            element_info: The image element information

        Returns:
            DITAElementInfo: Processed image information
        """
        try:
            if element_info.type != DITAElementType.IMAGE:
                return element_info

            # Get image rules
            rules = self._processing_rules[DITAElementType.IMAGE]

            # Process required src attribute
            src = element_info.attributes.get('href')  # DITA uses href for images
            if not src:
                self.logger.warning("Image element missing 'href' attribute")
                return self.content_processor.create_dita_error_element(
                    error=ValueError("Missing image source"),
                    element_context="image"
                )

            # Create figure wrapper if specified in rules
            if element_info.context.parent_id and rules.get('content_wrapper') == 'figure':
                # Get figure rules
                figure_rules = self._processing_rules[DITAElementType.FIGURE]

                # Create figure context
                figure_context = DITAElementContext(
                    parent_id=element_info.context.parent_id,
                    element_type='figure',
                    classes=figure_rules['default_classes'],
                    attributes={},
                    topic_type=element_info.context.topic_type,
                    is_body=element_info.context.is_body
                )

                # Create figure attributes
                figure_attrs = ElementAttributes(
                    id=self.id_handler.generate_id(f"fig-{element_info.attributes.id}"),
                    classes=figure_rules['default_classes'],
                    custom_attrs={}
                )

                # Add figcaption if alt text exists
                children = []
                if alt_text := element_info.attributes.get('alt'):
                    caption_context = DITAElementContext(
                        parent_id=figure_attrs.id,
                        element_type='figcaption',
                        classes=figure_rules.get('caption_classes', []),
                        attributes={},
                        topic_type=element_info.context.topic_type,
                        is_body=False
                    )

                    children.append(DITAElementInfo(
                        type=DITAElementType.PHRASE,  # Using PHRASE for figcaption
                        content=alt_text,
                        attributes=ElementAttributes(
                            id=self.id_handler.generate_id(f"figcaption-{element_info.attributes.id}"),
                            classes=figure_rules.get('caption_classes', []),
                            custom_attrs={}
                        ),
                        context=caption_context,
                        metadata={},
                        children=[]
                    ))

                # Return figure with image as child
                return DITAElementInfo(
                    type=DITAElementType.FIGURE,
                    content="",  # Figure container has no direct content
                    attributes=figure_attrs,
                    context=figure_context,
                    metadata=element_info.metadata,
                    children=[element_info] + children
                )

            # If no figure wrapper needed, return processed image
            return element_info

        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return self.content_processor.create_dita_error_element(
                error=e,
                element_context="image"
            )


    def _get_element_type(self, elem: etree._Element) -> DITAElementType:
        """
        Determine the DITA element type from XML element.

        Args:
            elem: XML element to analyze

        Returns:
            DITAElementType: The determined element type
        """
        try:
            # Get tag name, handling namespaces
            tag = etree.QName(elem).localname.lower()

            # Direct mapping for most elements
            try:
                return DITAElementType[tag.upper()]
            except KeyError:
                # Handle special cases and aliases
                special_cases = {
                    # Lists
                    'ul': DITAElementType.LIST,
                    'ol': DITAElementType.ORDERED_LIST,
                    'li': DITAElementType.LIST_ITEM,

                    # Code elements
                    'codeblock': DITAElementType.CODE_BLOCK,
                    'codeph': DITAElementType.CODE_PHRASE,

                    # Tasks
                    'taskbody': DITAElementType.TASKBODY,
                    'step': DITAElementType.STEP,
                    'cmd': DITAElementType.CMD,
                    'info': DITAElementType.INFO,
                    'substep': DITAElementType.SUBSTEP,
                    'substeps': DITAElementType.SUBSTEPS,

                    # Definitions
                    'dlentry': DITAElementType.DEFINITION,
                    'dt': DITAElementType.TERM,

                    # Inline elements
                    'b': DITAElementType.BOLD,
                    'i': DITAElementType.ITALIC,
                    'u': DITAElementType.UNDERLINE,
                    'ph': DITAElementType.PHRASE,
                    'q': DITAElementType.QUOTE,

                    # Images and figures
                    'fig': DITAElementType.FIGURE,
                    'image': DITAElementType.IMAGE,

                    # Links
                    'xref': DITAElementType.XREF,
                    'link': DITAElementType.LINK,
                }

                # Check special cases
                if tag in special_cases:
                    return special_cases[tag]

                # Check class attribute for DITA specialization
                if 'class' in elem.attrib:
                    class_tokens = elem.attrib['class'].split()

                    # Map DITA class specializations
                    class_mappings = {
                        'topic/body': DITAElementType.TASKBODY,
                        'task/step': DITAElementType.STEP,
                        'topic/p': DITAElementType.PARAGRAPH,
                        'topic/section': DITAElementType.SECTION,
                        'topic/title': DITAElementType.TITLE,
                        # Add more specialization mappings as needed
                    }

                    for token in class_tokens:
                        if token in class_mappings:
                            return class_mappings[token]

                # Log warning for unknown elements
                self.logger.warning(f"Unknown DITA element type: {tag}")
                return DITAElementType.UNKNOWN

        except Exception as e:
            self.logger.error(f"Error determining element type: {str(e)}")
            return DITAElementType.UNKNOWN


    def _get_element_content(self, elem: etree._Element) -> str:
        """
        Get the text content of an element based on its type and rules.

        Args:
            elem: The DITA XML element

        Returns:
            str: Processed content string for the element
        """
        try:
            # Get element type and rules
            element_type = self._get_element_type(elem)
            rules = self._processing_rules.get(element_type,
                                             self._processing_rules[DITAElementType.UNKNOWN])

            # Handle empty elements
            if elem.text is None and len(elem) == 0:
                return ""

            # Special content handling based on element type
            if element_type == DITAElementType.CODE_BLOCK:
                return self._get_code_content(elem, rules)

            elif element_type == DITAElementType.STEP:
                return self._get_step_content(elem, rules)

            elif element_type == DITAElementType.CMD:
                return self._get_command_content(elem, rules)

            elif element_type == DITAElementType.NOTE:
                return self._get_note_content(elem, rules)

            elif element_type == DITAElementType.IMAGE:
                return ""  # Images don't have text content

            elif element_type == DITAElementType.XREF:
                return self._get_xref_content(elem, rules)

            # Standard content processing
            content_parts = []

            # Add element's direct text if present
            if elem.text:
                content_parts.append(elem.text.strip())

            # Process child elements' content
            for child in elem:
                # Add child's tail text if present
                if child.tail:
                    content_parts.append(child.tail.strip())

                # Recursively get child content if it's a text-containing element
                if self._is_text_element(child):
                    child_content = self._get_element_content(child)
                    if child_content:
                        content_parts.append(child_content)

            # Join all content parts
            return ' '.join(filter(None, content_parts))

        except Exception as e:
            self.logger.error(f"Error extracting content from {etree.QName(elem).localname}: {str(e)}")
            return ""

    # CONTENT METHODS

    def _get_code_content(self, elem: etree._Element, rules: dict) -> str:
        """Get content for code blocks with proper handling."""
        try:
            # Get language from outputclass
            language = elem.get('outputclass', '')
            content = elem.text or ""

            # Preserve whitespace and indentation
            return content.rstrip()
        except Exception as e:
            self.logger.error(f"Error processing code content: {str(e)}")
            return ""

    def _get_step_content(self, elem: etree._Element, rules: dict) -> str:
        """Get content for task steps with proper structure."""
        try:
            parts = []
            cmd = elem.find('cmd')
            if cmd is not None and cmd.text:
                parts.append(cmd.text.strip())

            info = elem.find('info')
            if info is not None and info.text:
                parts.append(info.text.strip())

            return ' '.join(parts)
        except Exception as e:
            self.logger.error(f"Error processing step content: {str(e)}")
            return ""

    def _get_command_content(self, elem: etree._Element, rules: dict) -> str:
        """Get content for command elements."""
        try:
            return elem.text.strip() if elem.text else ""
        except Exception as e:
            self.logger.error(f"Error processing command content: {str(e)}")
            return ""

    def _get_note_content(self, elem: etree._Element, rules: dict) -> str:
        """Get content for note elements with type handling."""
        try:
            content = elem.text or ""
            note_type = elem.get('type', 'note')

            # Add any structured content inside the note
            for child in elem:
                if child.text:
                    content += " " + child.text.strip()
                if child.tail:
                    content += " " + child.tail.strip()

            return content.strip()
        except Exception as e:
            self.logger.error(f"Error processing note content: {str(e)}")
            return ""

    def _get_xref_content(self, elem: etree._Element, rules: dict) -> str:
        """Get content for cross-references."""
        try:
            # Use explicit link text if provided
            if elem.text:
                return elem.text.strip()

            # Otherwise use href as fallback
            return elem.get('href', '').strip()
        except Exception as e:
            self.logger.error(f"Error processing xref content: {str(e)}")
            return ""

    def _is_text_element(self, elem: etree._Element) -> bool:
        """Determine if an element can contain meaningful text content."""
        try:
            element_type = self._get_element_type(elem)
            return element_type not in {
                DITAElementType.IMAGE,
                DITAElementType.METADATA,
                DITAElementType.UNKNOWN
            }
        except Exception as e:
            self.logger.error(f"Error checking text element: {str(e)}")
            return False


    def _get_element_classes(self, elem: etree._Element) -> List[str]:
        """
        Get combined classes for element based on processing rules and element attributes.

        Args:
            elem: The DITA XML element

        Returns:
            List[str]: Combined list of CSS classes
        """
        try:
            # Get element type and rules
            element_type = self._get_element_type(elem)
            rules = self._processing_rules.get(element_type,
                                             self._processing_rules[DITAElementType.UNKNOWN])

            # Start with default classes from rules
            classes = rules['default_classes'].copy()

            # Add classes from element's class attribute
            if 'class' in elem.attrib:
                # Split and clean element classes
                element_classes = [
                    cls.strip()
                    for cls in elem.attrib['class'].split()
                    if cls.strip()
                ]
                classes.extend(element_classes)

            # Add classes from outputclass attribute
            if 'outputclass' in elem.attrib:
                output_classes = [
                    cls.strip()
                    for cls in elem.attrib['outputclass'].split()
                    if cls.strip()
                ]
                classes.extend(output_classes)

            # Add type-specific classes if applicable
            if 'type_class_mapping' in rules and 'type' in elem.attrib:
                type_class = rules['type_class_mapping'].get(
                    elem.attrib['type'],
                    rules['type_class_mapping'].get('default', '')
                )
                if type_class:
                    classes.append(type_class)

            # Add state-based classes
            if 'state' in elem.attrib:
                state_class = f"state-{elem.attrib['state']}"
                classes.append(state_class)

            # Add structural classes based on context
            if self._is_body_element(elem):
                classes.append('body-content')

            if parent_type := self._get_topic_type(elem):
                classes.append(f"{parent_type}-content")

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


    def _get_element_attributes(self, elem: etree._Element) -> ElementAttributes:
        """
        Extract and process element attributes based on processing rules.

        Args:
            elem: The DITA XML element

        Returns:
            ElementAttributes: Processed attributes for the element
        """
        try:
            # Get element type and rules
            element_type = self._get_element_type(elem)
            rules = self._processing_rules.get(element_type,
                                             self._processing_rules[DITAElementType.UNKNOWN])

            # Generate or get element ID
            element_id = elem.get("id") or self.id_handler.generate_content_id(
                Path(str(elem.tag))
            )

            # Get classes using _get_element_classes
            classes = self._get_element_classes(elem)

            # Initialize custom attributes dictionary
            custom_attrs = {}

            # Add default attributes from rules
            if 'attributes' in rules:
                custom_attrs.update(rules['attributes'])

            # Apply attribute mappings from rules
            if 'attribute_mapping' in rules:
                for dita_attr, html_attr in rules['attribute_mapping'].items():
                    if dita_attr in elem.attrib:
                        custom_attrs[html_attr] = elem.attrib[dita_attr]

            # Process remaining element attributes
            for attr, value in elem.attrib.items():
                # Skip already processed attributes
                if attr in {'id', 'class', 'outputclass'}:
                    continue

                # Handle special attributes based on element type
                if element_type == DITAElementType.IMAGE and attr == 'href':
                    custom_attrs['src'] = value
                elif element_type in {DITAElementType.XREF, DITAElementType.LINK} and attr == 'href':
                    custom_attrs['href'] = value
                    # Add target and rel attributes for external links
                    if value.startswith(('http://', 'https://')):
                        custom_attrs['target'] = '_blank'
                        custom_attrs['rel'] = 'noopener noreferrer'
                else:
                    # Add attribute with data- prefix for custom attributes
                    attr_name = attr if attr in rules.get('allowed_attributes', []) else f'data-{attr}'
                    custom_attrs[attr_name] = value

            # Add required attributes if missing
            if 'required_attributes' in rules:
                for required_attr in rules['required_attributes']:
                    if required_attr not in custom_attrs:
                        self.logger.warning(
                            f"Missing required attribute '{required_attr}' "
                            f"for element type {element_type}"
                        )
                        # Add placeholder for required attribute
                        custom_attrs[required_attr] = ''

            # Add accessibility attributes based on element type
            if 'accessibility_attributes' in rules:
                custom_attrs.update(rules['accessibility_attributes'])

            # Add role attribute if specified in rules
            if role := rules.get('role'):
                custom_attrs['role'] = role

            # Process conditional attributes
            if 'props' in elem.attrib:
                custom_attrs['data-props'] = elem.attrib['props']

            if 'platform' in elem.attrib:
                custom_attrs['data-platform'] = elem.attrib['platform']

            if 'product' in elem.attrib:
                custom_attrs['data-product'] = elem.attrib['product']

            # Log debug information
            self.logger.debug(
                f"Processed attributes for {element_type.value}: "
                f"id={element_id}, classes={classes}, "
                f"custom_attrs={custom_attrs}"
            )

            return ElementAttributes(
                id=element_id,
                classes=classes,
                custom_attrs=custom_attrs
            )

        except Exception as e:
            self.logger.error(f"Error extracting attributes: {str(e)}")
            return ElementAttributes(
                id=self.id_handler.generate_id("error"),
                classes=[],
                custom_attrs={}
            )

    def _get_topic_type(self, elem: etree._Element) -> Optional[str]:
        """
        Determine the topic type for an element based on its ancestry and context.

        Args:
            elem: The DITA XML element

        Returns:
            Optional[str]: Topic type if element is inside a topic, None otherwise
        """
        try:
            # Define valid topic types from DITAElementType
            topic_types = {
                DITAElementType.CONCEPT.value,
                DITAElementType.TASK.value,
                DITAElementType.REFERENCE.value,
                DITAElementType.TOPIC.value
            }

            # Check element itself first
            element_type = self._get_element_type(elem)
            if element_type.value in topic_types:
                return element_type.value

            # Traverse up the tree to find topic ancestor
            for ancestor in elem.iterancestors():
                # Get ancestor's tag name
                ancestor_tag = etree.QName(ancestor).localname

                # Check direct tag match
                if ancestor_tag in topic_types:
                    return ancestor_tag

                # Check class attribute for DITA specialization
                if 'class' in ancestor.attrib:
                    class_tokens = ancestor.attrib['class'].split()

                    # Map DITA class specializations to topic types
                    specialization_mapping = {
                        'concept/concept': DITAElementType.CONCEPT.value,
                        'task/task': DITAElementType.TASK.value,
                        'reference/reference': DITAElementType.REFERENCE.value,
                        'topic/topic': DITAElementType.TOPIC.value
                    }

                    for token in class_tokens:
                        if token in specialization_mapping:
                            return specialization_mapping[token]

            # Log debug information if no topic type found
            self.logger.debug(
                f"No topic type found for element {etree.QName(elem).localname}"
            )

            return None

        except Exception as e:
            self.logger.error(
                f"Error determining topic type for {etree.QName(elem).localname}: {str(e)}"
            )
            return None


    def _is_topic_element(self, elem: etree._Element) -> bool:
        """
        Check if element is a topic element.

        Args:
            elem: The DITA XML element

        Returns:
            bool: True if element is a topic element
        """
        try:
            element_type = self._get_element_type(elem)
            return element_type in {
                DITAElementType.CONCEPT,
                DITAElementType.TASK,
                DITAElementType.REFERENCE,
                DITAElementType.TOPIC
            }
        except Exception as e:
            self.logger.error(f"Error checking topic element: {str(e)}")
            return False



    def _is_body_element(self, elem: etree._Element) -> bool:
        """
        Determine if element is inside a body element of any topic type.

        Args:
            elem: The DITA XML element

        Returns:
            bool: True if element is inside a body element
        """
        try:
            # Define valid body types from DITAElementType
            body_elements = {
                'conbody',    # Concept body
                'taskbody',   # Task body
                'refbody',    # Reference body
                'body',       # Generic topic body
            }

            # Define body-specific mappings from class attributes
            body_class_mappings = {
                'topic/body': True,
                'concept/conbody': True,
                'task/taskbody': True,
                'reference/refbody': True
            }

            # Check element itself first
            element_tag = etree.QName(elem).localname
            if element_tag in body_elements:
                return True

            # Check class attribute for DITA specialization
            if 'class' in elem.attrib:
                class_tokens = elem.attrib['class'].split()
                if any(token in body_class_mappings for token in class_tokens):
                    return True

            # Traverse up the tree to find body ancestor
            for ancestor in elem.iterancestors():
                # Check ancestor tag
                ancestor_tag = etree.QName(ancestor).localname
                if ancestor_tag in body_elements:
                    return True

                # Check ancestor class attribute
                if 'class' in ancestor.attrib:
                    ancestor_classes = ancestor.attrib['class'].split()
                    if any(token in body_class_mappings for token in ancestor_classes):
                        return True

                # Stop traversing if we hit a topic element
                if self._is_topic_element(ancestor):
                    break

            # Log debug information
            self.logger.debug(
                f"Element {etree.QName(elem).localname} is not in a body element"
            )

            return False

        except Exception as e:
            self.logger.error(
                f"Error checking body element for {etree.QName(elem).localname}: {str(e)}"
            )
            return False

    def _get_containing_body(self, elem: etree._Element) -> Optional[etree._Element]:
        """
        Get the containing body element if any.

        Args:
            elem: The DITA XML element

        Returns:
            Optional[etree._Element]: The containing body element or None
        """
        try:
            # Check ancestors for body element
            for ancestor in elem.iterancestors():
                if self._is_body_element(ancestor):
                    return ancestor
            return None

        except Exception as e:
            self.logger.error(f"Error getting containing body: {str(e)}")
            return None

    def _get_body_type(self, body_elem: etree._Element) -> Optional[str]:
        """
        Get the specific type of body element.

        Args:
            body_elem: The body element

        Returns:
            Optional[str]: The body type or None
        """
        try:
            # Direct tag mapping
            tag = etree.QName(body_elem).localname
            body_types = {
                'conbody': 'concept',
                'taskbody': 'task',
                'refbody': 'reference',
                'body': 'topic'
            }

            if tag in body_types:
                return body_types[tag]

            # Check class attribute
            if 'class' in body_elem.attrib:
                class_tokens = body_elem.attrib['class'].split()
                class_mappings = {
                    'concept/conbody': 'concept',
                    'task/taskbody': 'task',
                    'reference/refbody': 'reference',
                    'topic/body': 'topic'
                }

                for token in class_tokens:
                    if token in class_mappings:
                        return class_mappings[token]

            return None

        except Exception as e:
            self.logger.error(f"Error getting body type: {str(e)}")
            return None


    # ==========================================================================
    # VALIDATION
    # ==========================================================================

    def _validate_element(self, elem: etree._Element) -> bool:
        """
        Validate element structure and required attributes based on processing rules.

        Args:
            elem: The DITA XML element

        Returns:
            bool: True if element is valid according to its rules
        """
        try:
            # Get element type and rules
            element_type = self._get_element_type(elem)
            rules = self._processing_rules.get(element_type,
                                             self._processing_rules[DITAElementType.UNKNOWN])

            # Validate based on element type
            validation_results = []

            # Check required attributes
            if 'required_attributes' in rules:
                for attr in rules['required_attributes']:
                    if attr not in elem.attrib:
                        self.logger.warning(
                            f"Missing required attribute '{attr}' "
                            f"for {element_type.value} element"
                        )
                        validation_results.append(False)

            # Element-specific validation
            if element_type == DITAElementType.IMAGE:
                validation_results.append(self._validate_image(elem))

            elif element_type == DITAElementType.XREF:
                validation_results.append(self._validate_xref(elem))

            elif element_type == DITAElementType.STEP:
                validation_results.append(self._validate_step(elem))

            elif element_type == DITAElementType.CODE_BLOCK:
                validation_results.append(self._validate_code_block(elem))

            # Check content requirements
            if 'requires_content' in rules and rules['requires_content']:
                if not self._has_content(elem):
                    self.logger.warning(
                        f"Empty content in {element_type.value} element "
                        f"that requires content"
                    )
                    validation_results.append(False)

            # Check parent requirements
            if 'allowed_parents' in rules:
                if not self._validate_parent(elem, rules['allowed_parents']):
                    self.logger.warning(
                        f"Invalid parent for {element_type.value} element"
                    )
                    validation_results.append(False)

            # If no specific validations were performed, consider element valid
            if not validation_results:
                return True

            # Element is valid if all validations passed
            return all(validation_results)

        except Exception as e:
            self.logger.error(
                f"Error validating element {etree.QName(elem).localname}: {str(e)}"
            )
            return False

    def _validate_image(self, elem: etree._Element) -> bool:
        """Validate image element."""
        try:
            # Check for href/src attribute
            if 'href' not in elem.attrib:
                self.logger.warning("Image element missing 'href' attribute")
                return False

            # Check for alt text (accessibility)
            if 'alt' not in elem.attrib:
                self.logger.warning("Image element missing 'alt' attribute")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating image element: {str(e)}")
            return False

    def _validate_xref(self, elem: etree._Element) -> bool:
        """Validate cross-reference element."""
        try:
            # Check for href attribute
            if 'href' not in elem.attrib:
                self.logger.warning("Xref element missing 'href' attribute")
                return False

            # Validate href format
            href = elem.attrib['href']
            if not href or href.isspace():
                self.logger.warning("Empty href in xref element")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating xref element: {str(e)}")
            return False

    def _validate_step(self, elem: etree._Element) -> bool:
        """Validate task step element."""
        try:
            # Check for required cmd element
            cmd = elem.find('cmd')
            if cmd is None:
                self.logger.warning("Step element missing required 'cmd' element")
                return False

            # Check for cmd content
            if not cmd.text or cmd.text.isspace():
                self.logger.warning("Empty cmd in step element")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating step element: {str(e)}")
            return False

    def _validate_code_block(self, elem: etree._Element) -> bool:
        """Validate code block element."""
        try:
            # Check for content
            if not elem.text or elem.text.isspace():
                self.logger.warning("Empty code block element")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating code block element: {str(e)}")
            return False

    def _has_content(self, elem: etree._Element) -> bool:
        """Check if element has meaningful content."""
        try:
            # Check direct text content
            if elem.text and not elem.text.isspace():
                return True

            # Check child elements
            if len(elem) > 0:
                return True

            # Check tail text
            if elem.tail and not elem.tail.isspace():
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking element content: {str(e)}")
            return False

    def _validate_parent(self, elem: etree._Element, allowed_parents: List[str]) -> bool:
        """Validate element's parent."""
        try:
            parent = elem.getparent()
            if parent is None:
                return False

            parent_type = self._get_element_type(parent)
            return parent_type.value in allowed_parents

        except Exception as e:
            self.logger.error(f"Error validating parent element: {str(e)}")
            return False


    # ==========================================================================
    # METADATA EXTRACTION
    # ==========================================================================

    def _get_element_metadata(self, elem: etree._Element) -> Dict[str, Any]:
        """Extract metadata from element with error handling."""
        # Initialize element_type outside try block
        element_type = DITAElementType.UNKNOWN

        try:
            # Get element type and rules
            element_type = self._get_element_type(elem)
            rules = self._processing_rules.get(element_type,
                                             self._processing_rules[DITAElementType.UNKNOWN])

            # Initialize metadata structure
            metadata: Dict[str, Any] = {
                'element_info': {
                    'type': element_type.value,
                    'tag': etree.QName(elem).localname,
                    'id': elem.get('id', ''),
                    'created': datetime.now().isoformat(),
                    'processing_phase': self.current_phase.value if hasattr(self, 'current_phase') else None
                },
                'content_info': {
                    'has_children': len(elem) > 0,
                    'has_text': bool(elem.text and not elem.text.isspace()),
                    'word_count': len(elem.text.split()) if elem.text else 0
                },
                'attributes': {},
                'context': {},
                'processing': {},
                'custom': {}
            }

            # Extract standard DITA metadata attributes
            standard_metadata = {
                'audience': elem.get('audience'),
                'platform': elem.get('platform'),
                'product': elem.get('product'),
                'rev': elem.get('rev'),
                'status': elem.get('status'),
                'importance': elem.get('importance'),
                'xml:lang': elem.get('{http://www.w3.org/XML/1998/namespace}lang')
            }
            metadata['attributes'].update({k: v for k, v in standard_metadata.items() if v is not None})

            # Add context information with proper parent type handling
            parent_elem = elem.getparent()
            parent_type = (
                self._get_element_type(parent_elem).value
                if parent_elem is not None
                else None
            )

            metadata['context'].update({
                'topic_type': self._get_topic_type(elem),
                'is_body': self._is_body_element(elem),
                'parent_type': parent_type,
                'path': self._get_element_path(elem)
            })

            # Add processing-specific metadata from rules
            if 'metadata' in rules:
                metadata['processing'].update(rules['metadata'])

            # Element-specific metadata extraction
            if element_type == DITAElementType.IMAGE:
                metadata.update(self._extract_image_metadata(elem))

            elif element_type == DITAElementType.XREF:
                metadata.update(self._extract_xref_metadata(elem))

            elif element_type == DITAElementType.NOTE:
                metadata.update(self._extract_note_metadata(elem))

            elif element_type == DITAElementType.CODE_BLOCK:
                metadata.update(self._extract_code_metadata(elem))

            # Extract custom metadata
            custom_metadata = self._extract_custom_metadata(elem)
            if custom_metadata:
                metadata['custom'].update(custom_metadata)

            # Add extensibility hooks
            metadata['_extensible'] = {
                'json_compatible': True,
                'sql_compatible': True,
                'conversion_hints': self._get_conversion_hints(element_type),
                'schema_version': '1.0'
            }

            self.logger.debug(f"Extracted metadata for {element_type.value}: {metadata}")
            return metadata

        except Exception as e:
                self.logger.error(f"Error extracting metadata: {str(e)}")
                return {
                    'element_info': {
                        'type': element_type.value,  # Now element_type is always defined
                        'error': str(e)
                    },
                    'attributes': {},
                    'context': {},
                    'processing': {},
                    'custom': {}
                }

    def _extract_image_metadata(self, elem: etree._Element) -> Dict[str, Any]:
        """Extract image-specific metadata."""
        return {
            'media_info': {
                'src': elem.get('href', ''),
                'alt': elem.get('alt', ''),
                'height': elem.get('height'),
                'width': elem.get('width'),
                'scale': elem.get('scale'),
                'scalefit': elem.get('scalefit'),
                'placement': elem.get('placement', 'inline')
            }
        }

    def _extract_xref_metadata(self, elem: etree._Element) -> Dict[str, Any]:
        """Extract cross-reference metadata."""
        return {
            'link_info': {
                'href': elem.get('href', ''),
                'scope': elem.get('scope', 'local'),
                'format': elem.get('format'),
                'type': elem.get('type'),
                'is_external': elem.get('scope') == 'external'
            }
        }

    def _extract_note_metadata(self, elem: etree._Element) -> Dict[str, Any]:
        """Extract note metadata."""
        return {
            'note_info': {
                'type': elem.get('type', 'note'),
                'importance': elem.get('importance', 'normal'),
                'rev': elem.get('rev')
            }
        }

    def _extract_code_metadata(self, elem: etree._Element) -> Dict[str, Any]:
        """Extract code block metadata."""
        return {
            'code_info': {
                'language': elem.get('outputclass', ''),
                'line_count': len(elem.text.splitlines()) if elem.text else 0,
                'is_executable': 'exec' in (elem.get('outputclass', '').split()),
                'show_line_numbers': elem.get('show-line-numbers', 'true').lower() == 'true'
            }
        }

    def _extract_custom_metadata(self, elem: etree._Element) -> Dict[str, Any]:
        """Extract custom metadata from data-* attributes."""
        return {
            k: v for k, v in elem.attrib.items()
            if k.startswith('data-') or k.startswith('custom-')
        }

    def _get_element_path(self, elem: etree._Element) -> str:
        """Get XPath-like path to element."""
        try:
            path_parts = []
            for ancestor in elem.iterancestors():
                path_parts.append(etree.QName(ancestor).localname)
            path_parts.reverse()
            path_parts.append(etree.QName(elem).localname)
            return '/' + '/'.join(path_parts)
        except Exception as e:
            self.logger.error(f"Error getting element path: {str(e)}")
            return ""

    def _get_conversion_hints(self, element_type: DITAElementType) -> Dict[str, Any]:
        """Get hints for metadata conversion to different formats."""
        return {
            'json': {
                'array_fields': ['attributes', 'classes'],
                'date_fields': ['created'],
                'nested_objects': ['element_info', 'content_info', 'media_info', 'link_info']
            },
            'sql': {
                'table_name': f'dita_{element_type.value}_metadata',
                'primary_key': 'element_info.id',
                'indexes': ['element_info.type', 'element_info.created'],
                'json_columns': ['attributes', 'custom']
            }
        }

    # Cleanup

    def cleanup(self) -> None:
        """
        Perform comprehensive cleanup of processor resources, caches, and state.
        Ensures no memory leaks or stale data between processing runs.
        """
        try:
            self.logger.debug("Starting DITA element processor cleanup")

            # Reset core components
            self.id_handler = DITAIDHandler()
            self._processed_elements.clear()

            # Clear processing caches
            self._clear_caches()

            # Reset element tracking
            self._clear_element_tracking()

            # Reset metadata tracking
            self._clear_metadata()

            # Reset state variables
            self._reset_state()

            self.logger.debug("DITA element processor cleanup completed successfully")

        except Exception as e:
            self.logger.error(f"DITA element processor cleanup failed: {str(e)}")
            raise

    def _clear_caches(self) -> None:
        """Clear all internal caches."""
        try:
            # Clear element processing cache
            self._processed_elements.clear()

            # Clear any memoized results
            if hasattr(self, '_memoized_results'):
                self._memoized_results.clear()

            # Clear type mapping caches
            if hasattr(self, '_type_mapping_cache'):
                self._type_mapping_cache.clear()

            # Clear validation result cache
            if hasattr(self, '_validation_cache'):
                self._validation_cache.clear()

            self.logger.debug("Cleared all processor caches")

        except Exception as e:
            self.logger.error(f"Error clearing caches: {str(e)}")
            raise

    def _clear_element_tracking(self) -> None:
        """Clear element tracking information."""
        try:
            # Clear processed elements tracking
            if hasattr(self, '_tracked_elements'):
                self._tracked_elements.clear()

            # Clear element hierarchy tracking
            if hasattr(self, '_element_hierarchy'):
                self._element_hierarchy.clear()

            # Clear element reference counting
            if hasattr(self, '_element_refs'):
                self._element_refs.clear()

            self.logger.debug("Cleared element tracking information")

        except Exception as e:
            self.logger.error(f"Error clearing element tracking: {str(e)}")
            raise

    def _clear_metadata(self) -> None:
        """Clear metadata tracking and caches."""
        try:
            # Clear metadata caches
            if hasattr(self, '_metadata_cache'):
                self._metadata_cache.clear()

            # Clear custom metadata tracking
            if hasattr(self, '_custom_metadata'):
                self._custom_metadata.clear()

            # Clear metadata validation results
            if hasattr(self, '_metadata_validation'):
                self._metadata_validation.clear()

            self.logger.debug("Cleared metadata tracking information")

        except Exception as e:
            self.logger.error(f"Error clearing metadata: {str(e)}")
            raise

    def _reset_state(self) -> None:
        """Reset all state variables to initial values."""
        try:
            # Reset processing state
            self._processing_depth = 0
            self._current_topic_id = None
            self._current_element_id = None

            # Reset flags
            self._processing_enabled = True
            self._validation_enabled = True

            # Reset counters
            self._processed_count = 0
            self._error_count = 0

            # Reset configuration to defaults
            self._config = {
                'strict_mode': False,
                'enable_caching': True,
                'enable_validation': True
            }

            self.logger.debug("Reset processor state to initial values")

        except Exception as e:
            self.logger.error(f"Error resetting state: {str(e)}")
            raise

    def validate_cleanup(self) -> bool:
        """
        Validate cleanup was successful.

        Returns:
            bool: True if cleanup was successful, False otherwise
        """
        try:
            # Check caches are empty
            caches_empty = (
                len(self._processed_elements) == 0
                and not hasattr(self, '_memoized_results')
                and not hasattr(self, '_type_mapping_cache')
                and not hasattr(self, '_validation_cache')
            )

            # Check tracking is reset
            tracking_reset = (
                not hasattr(self, '_tracked_elements')
                and not hasattr(self, '_element_hierarchy')
                and not hasattr(self, '_element_refs')
            )

            # Check metadata is cleared
            metadata_cleared = (
                not hasattr(self, '_metadata_cache')
                and not hasattr(self, '_custom_metadata')
                and not hasattr(self, '_metadata_validation')
            )

            # Check state is reset
            state_reset = (
                self._processing_depth == 0
                and self._current_topic_id is None
                and self._current_element_id is None
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
