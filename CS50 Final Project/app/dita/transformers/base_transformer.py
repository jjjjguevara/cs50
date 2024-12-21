# app/dita/transformers/base_transformer.py
from abc import ABC, abstractmethod
import logging
import inspect
from functools import partial
from typing import Callable, List, Dict, Optional, Union, Any, Tuple
from bs4 import BeautifulSoup, Tag
from pathlib import Path
import re

# Third-Party Libraries
from bs4 import BeautifulSoup, Tag

# Internal Utilities and Handlers
from ..models.types import (
    TrackedElement,
    ProcessedContent,
    ProcessorConfig,
    ProcessingContext,
    ProcessingPhase,
    ProcessingState,
    ElementType,
    ProcessingMetadata,
    LaTeXEquation,
    ProcessedEquation
)

from ..utils.html_helpers import HTMLHelper
from ..utils.heading import HeadingHandler
from ..utils.id_handler import DITAIDHandler
from ..utils.metadata import MetadataHandler
from app.dita.config_manager import DITAConfig

# LaTeX Handling
from app.dita.utils.latex.katex_renderer import KaTeXRenderer
from app.dita.utils.latex.latex_processor import LaTeXProcessor
from app.dita.utils.latex.latex_validator import LaTeXValidator


class BaseTransformer(ABC):
    def __init__(
        self,
        dita_root: Optional[Path] = None,
        config: Optional[DITAConfig] = None,
        processing_metadata: Optional[ProcessingMetadata] = None,
        id_handler: Optional[DITAIDHandler] = None,
        metadata_extractor: Optional[MetadataHandler] = None,
        html_helper: Optional[HTMLHelper] = None,
        heading_handler: Optional[HeadingHandler] = None,
        katex_renderer: Optional[KaTeXRenderer] = None,
        latex_processor: Optional[LaTeXProcessor] = None,
        latex_validator: Optional[LaTeXValidator] = None
    ):
        """
        Initializes the BaseTransformer with required utilities and configurations.

        Args:
            dita_root (Optional[Path]): The root path for DITA files (default: None).
            config (Optional[DITAConfig]): Configuration for feature toggles and settings.
            processing_metadata (Optional[ProcessingMetadata]): Metadata handler for processing elements.
            id_handler (Optional[DITAIDHandler]): ID handler for generating and resolving IDs.
            metadata_extractor (Optional[MetadataHandler]): Metadata extractor for elements.
            html_helper (Optional[HTMLHelper]): HTML helper for content finalization.
            heading_handler (Optional[HeadingHandler]): Handler for headings and hierarchy.
        """
        self.logger = logging.getLogger(__name__)


        # Configuration and metadata
        self.dita_root = dita_root
        self.config = config or DITAConfig()  # Initialize config from config_manager

        # Utilities
        self.id_handler = id_handler or DITAIDHandler()
        self.metadata_handler = MetadataHandler()
        self.html_helper = html_helper or HTMLHelper(dita_root)
        self.heading_handler = heading_handler or HeadingHandler(processing_metadata=self.processing_metadata)

        # LaTeX Utilities
        self.katex_renderer = katex_renderer or KaTeXRenderer()
        self.latex_processor = LaTeXProcessor()
        self.latex_validator = latex_validator or LaTeXValidator()

        # Content transformation strategies
        self._element_transformers = {

            # Titles
            ElementType.HEADING: self._transform_titles,
            ElementType.MAP_TITLE: self._transform_titles,

            # Tables and figures
            ElementType.TABLE: self._transform_table,
            ElementType.FIGURE: self.inject_image,

            # Block Elements
            ElementType.PARAGRAPH: self._transform_block,
            ElementType.NOTE: self._transform_block,
            ElementType.CODE_BLOCK: self._transform_block,
            ElementType.BLOCKQUOTE: self._transform_block,
            ElementType.CODE_PHRASE: self._transform_block,

            # Lists
            ElementType.UNORDERED_LIST: self._transform_list,
            ElementType.ORDERED_LIST: self._transform_list,
            ElementType.TODO_LIST: self._transform_list,
            ElementType.LIST_ITEM: self._transform_list,


            # Links and References
            ElementType.XREF: self._transform_link,
            ElementType.LINK: self._transform_link,

            # Inline Elements
            ElementType.BOLD: self._transform_emphasis,
            ElementType.ITALIC: self._transform_emphasis,
            ElementType.UNDERLINE: self._transform_emphasis,
            ElementType.CODE_PHRASE: self._transform_emphasis,


            # DITA Specific
            ElementType.TOPIC: self._transform_topic,
            ElementType.TASKBODY: self._transform_taskbody,
            ElementType.SHORTDESC: self._transform_shortdesc,
            ElementType.ABSTRACT: self._transform_abstract,
            ElementType.PREREQ: self._transform_prereq,
            ElementType.STEPS: self._transform_steps,
            ElementType.SUBSTEPS: self._transform_substeps,


            # Metadata & Structure
            ElementType.METADATA: self._transform_metadata,
            ElementType.TOPICREF: self._transform_topicref,

            # Fallback
            ElementType.UNKNOWN: self._transform_unknown

        }

        # Contextual/feature strategies (inject/add/swap)
        self._strategy_registry = {
            "inject_image": lambda c, m: c.features.get("image", False),
            "inject_video": lambda c, m: c.features.get("video", False),
            "inject_audio": lambda c, m: c.features.get("audio", False),
            "inject_iframe": lambda c, m: c.features.get("iframe", False),
            "add_latex": lambda c, m: c.features.get("latex", False),
            "add_topic_section": lambda c, m: c.features.get("topic_section", False),
            "add_heading_attributes": lambda c, m: (
                c.features.get("index_numbers", True) or
                c.features.get("anchor_links", True)
            ),
            "add_toc": lambda c, m: c.features.get("toc", False),
            "add_bibliography": lambda c, m: "bibliography" in m,
            "add_glossary": lambda c, m: "glossary" in m,
            "swap_topic_version": lambda c, m: c.features.get("swap_topic_version", False),
            "swap_topic_type": lambda c, m: c.features.get("swap_topic_type", False)
        }


        """
        Processing rules for DITA and Markdown element specialization.
        A processing rule is a dictionary that defines how an element should be transformed to HTML.
        It contains the following keys:
        - html_tag: The HTML tag to use for the element.
        - default_classes: A list of default classes to apply to the element.
        - attributes: A dictionary of attributes to apply to the element.
        - context_type: The context type for the element.
        - specializations: A dictionary of specializations and their processing rules.
        """
        self._processing_rules = {

            # Titles
            ElementType.MAP_TITLE: {
                   'html_tag': 'h1',
                   'default_classes': ['map-title', 'main-title', 'text-3xl', 'font-bold', 'mb-6'],
                   'attributes': {
                       'role': 'heading',
                       'aria-level': '1'
                   },
                   'context_type': 'title'
               },
            ElementType.HEADING: {
                'html_tag': 'h{level}',  # Level from metadata
                'default_classes': ['heading', 'topic-heading'],
                'level_classes': {
                    1: ['text-2xl', 'font-bold', 'mb-4'],
                    2: ['text-xl', 'font-bold', 'mb-3'],
                    3: ['text-lg', 'font-semibold', 'mb-2'],
                    4: ['text-base', 'font-semibold', 'mb-2'],
                    5: ['text-sm', 'font-medium', 'mb-1'],
                    6: ['text-sm', 'font-medium', 'mb-1']
                },
                'attributes': {
                    'role': 'heading'
                },
                'context_type': 'heading'
            },

            # Block Elements
            ElementType.PARAGRAPH: {
                'html_tag': 'p',
                'default_classes': ['prose'],
                'context_type': 'block'
            },
            ElementType.NOTE: {
                'html_tag': 'div',
                'default_classes': ['note', 'alert'],
                'type_classes': {
                    'warning': 'alert-warning',
                    'danger': 'alert-danger',
                    'tip': 'alert-info',
                    'note': 'alert-secondary',
                    'callout': 'callout'
                },
                'attributes': {'role': 'note'},
                'context_type': 'block'
            },
            ElementType.CODE_BLOCK: {
                'html_tag': 'pre',
                'default_classes': ['code-block', 'highlight'],
                'attributes': {
                    'spellcheck': 'false',
                    'translate': 'no'
                },
                'inner_tag': 'code',
                'inner_classes': ['language-{language}'],
                'context_type': 'block'
            },
            ElementType.BLOCKQUOTE: {
                'html_tag': 'blockquote',
                'default_classes': ['quote'],
                'context_type': 'block'
            },
            ElementType.CODE_PHRASE: {
                'html_tag': 'code',
                'default_classes': ['code-inline'],
                'context_type': 'inline'
            },

            # Tables
            ElementType.TABLE: {
                'html_tag': 'table',
                'default_classes': ['table', 'table-bordered'],
                'attributes': {'role': 'grid'},
                'specializations': {
                    'bibliography': {
                        'extra_classes': ['bibliography-table'],
                        'extra_attrs': {
                            'data-citation-format': '{citation_format}',
                            'data-sort': '{sort_by}',
                            'aria-label': 'Bibliography entries'
                        }
                    },
                    'glossary': {
                        'extra_classes': ['glossary-table'],
                        'extra_attrs': {
                            'data-sort': '{sort_by}',
                            'data-show-refs': '{show_references}',
                            'aria-label': 'Glossary terms and definitions'
                        }
                    },
                    'metadata': {
                        'extra_classes': ['metadata-table'],
                        'extra_attrs': {
                            'data-visibility': '{visibility}',
                            'data-collapsible': '{collapsible}',
                            'aria-label': 'Article metadata'
                        }
                    }
                }
            },
            ElementType.TABLE_HEADER: {
                'html_tag': 'th',
                'default_classes': ['table-header'],
                'attributes': {'scope': 'col'}
            },
            ElementType.TABLE_ROW: {
                'html_tag': 'tr',
                'default_classes': ['table-row']
            },
            ElementType.TABLE_CELL: {
                'html_tag': 'td',
                'default_classes': ['table-cell']
            },


            # Links
            ElementType.LINK: {
                'html_tag': 'a',
                'default_classes': ['link'],
                'external_attrs': {
                    'target': '_blank',
                    'rel': 'noopener noreferrer'
                },
                'context_type': 'inline'
            },
            ElementType.XREF: {
                "html_tag": "a",
                "classes": ["dita-xref"],
                "required_attrs": ["href"]
            },

            # Lists
            ElementType.UNORDERED_LIST: {
                'html_tag': 'ul',
                'default_classes': ['list-unordered'],
                'context_type': 'block'
            },
            ElementType.ORDERED_LIST: {
                'html_tag': 'ol',
                'default_classes': ['list-ordered'],
                'context_type': 'block'
            },
            ElementType.TODO_LIST: {
                'html_tag': 'li',
                'default_classes': ['todo-item', 'flex', 'items-center', 'gap-2'],
                'inner_tag': 'input',
                'inner_attrs': {
                    'type': 'checkbox',
                    'disabled': ''
                },
                'context_type': 'block'
            },

            # Emphasis elements
            ElementType.BOLD: {
                'html_tag': 'strong',
                'default_classes': ['font-bold'],
                'context_type': 'emphasis'
            },
            ElementType.ITALIC: {
                'html_tag': 'em',
                'default_classes': ['italic'],
                'context_type': 'emphasis'
            },
            ElementType.UNDERLINE: {
                'html_tag': 'u',
                'default_classes': ['underline'],
                'context_type': 'emphasis'
            },
            ElementType.HIGHLIGHT: {
                'html_tag': 'mark',
                'default_classes': ['bg-yellow-200'],
                'context_type': 'emphasis'
            },
            ElementType.STRIKETHROUGH: {
                'html_tag': 'del',
                'default_classes': ['line-through'],
                'context_type': 'emphasis'
            },
        }

    def transform(self, element: TrackedElement, context: ProcessingContext) -> ProcessedContent:
        """Transform content using centralized methods and strategy patterns."""
        try:
            # Convert to base HTML (handled by specific transformers)
            html_content = self._convert_to_html(element, context)

            # Transform using centralized methods
            if element.type in self._processing_rules:
                if element.type in [ElementType.HEADING, ElementType.MAP_TITLE]:
                    html_content = self._transform_titles(element)
                elif element.type in [ElementType.PARAGRAPH, ElementType.NOTE,
                                    ElementType.CODE_BLOCK, ElementType.BLOCKQUOTE,
                                    ElementType.CODE_PHRASE]:
                    html_content = self._transform_block(element)
                elif element.type in [ElementType.XREF, ElementType.LINK]:
                    html_content = self._transform_link(element)
                elif element.type in [ElementType.BOLD, ElementType.ITALIC,
                                    ElementType.UNDERLINE, ElementType.HIGHLIGHT,
                                    ElementType.STRIKETHROUGH]:
                    html_content = self._transform_emphasis(element)
                elif element.type in [ElementType.TABLE, ElementType.TABLE_HEADER,
                                    ElementType.TABLE_ROW, ElementType.TABLE_CELL]:
                    html_content = self._transform_table(element)
                elif element.type in [ElementType.UNORDERED_LIST, ElementType.ORDERED_LIST,
                                    ElementType.TODO_LIST, ElementType.LIST_ITEM]:
                    html_content = self._transform_list(element)

            # Apply transformation strategies
            html_content = self._apply_strategies(
                html=html_content,
                element=element,
                context=context,
                metadata=element.metadata
            )

            return ProcessedContent(
                html=html_content,
                element_id=element.id,
                metadata=element.metadata
            )

        except Exception as e:
            self.logger.error(f"Transform failed: {str(e)}")
            raise

    @abstractmethod
    def _convert_to_html(self, element: TrackedElement, context: ProcessingContext) -> str:
        """
        Initial HTML conversion to be implemented by specific transformers.
        DITA and Markdown transformers will handle their specific parsing here.
        """
        pass

    def _apply_strategies(
       self,
       html: str,
       element: TrackedElement,
       context: ProcessingContext,
       metadata: Dict[str, Any]
    ) -> str:
       """Apply feature transformation strategies."""
       try:
           soup = BeautifulSoup(html, "html.parser")

           for strategy in self._get_active_strategies(context, metadata):
               soup = strategy(soup, element, context)

           return str(soup)

       except Exception as e:
           self.logger.error(f"Strategy application failed: {str(e)}")
           return html

    def _get_active_strategies(
       self,
       context: ProcessingContext,
       metadata: Dict[str, Any]
    ) -> List[Callable]:
       try:
           # Define priorities as dict mapping
           priorities = {
               "_inject_": 1,
               "_append_": 2,
               "_swap_": 3
           }

           active_strategies = []

           # Get enabled strategies
           for strategy_name, condition in self._strategy_registry.items():
               if condition(context, metadata):
                   strategy = getattr(self, strategy_name, None)
                   if strategy:
                       active_strategies.append((
                           strategy,
                           next((p for prefix, p in priorities.items()
                                if strategy.__name__.startswith(prefix)), 99)
                       ))

           # Sort by priority and return strategies
           return [s for s, _ in sorted(active_strategies, key=lambda x: x[1])]

       except Exception as e:
           self.logger.error(f"Error getting active strategies: {str(e)}")
           return []

    def process_elements(self, soup: BeautifulSoup, element: TrackedElement, context: ProcessingContext) -> None:
        """
        Process content with strategies before transformation.

        Args:
            soup: Parsed HTML content.
            element: The current TrackedElement being processed.
            context: Current processing context.
        """
        try:
            # Log start of element processing
            self.logger.debug(f"Processing elements for {element.id}")

            # Apply active strategies
            strategies = self._get_active_strategies(context, element.metadata)
            for strategy in strategies:
                try:
                    # Execute each strategy with the current soup, element, and context
                    strategy(soup, element, context)
                    self.logger.debug(f"Applied strategy {strategy.__name__} for {element.id}")
                except Exception as strategy_error:
                    self.logger.error(f"Error in strategy {strategy.__name__}: {strategy_error}")
                    raise

        except Exception as e:
            self.logger.error(f"Error processing elements for {element.id}: {e}")
            raise


    ########################################
    # Injection strategies (for transform) #
    ########################################

    def add_latex(self, soup: BeautifulSoup, element: TrackedElement, context: ProcessingContext) -> None:
        """
        Inject LaTeX equations into the HTML content.

        Args:
            soup: Parsed HTML content.
            element: The current TrackedElement being processed.
            context: Current processing context.
        """
        try:
            latex_equations = self._extract_latex_equations(str(soup))

            for eq in latex_equations:
                if not eq.placeholder:
                    self.logger.warning(f"Skipping LaTeX equation with missing placeholder: {eq.id}")
                    continue

                if not self.latex_validator.validate_equation(eq):
                    self.logger.warning(f"Skipping invalid LaTeX equation (ID: {eq.id}): {eq.content}")
                    continue

                # Render LaTeX equation using KaTeX
                rendered = self.katex_renderer.render_equation(eq)

                # Replace placeholder in soup
                placeholder_node = soup.find(string=eq.placeholder)
                if placeholder_node:
                    rendered_element = BeautifulSoup(rendered, "html.parser")
                    placeholder_node.replace_with(rendered_element)
                else:
                    self.logger.warning(f"Placeholder for equation ID {eq.id} not found.")

        except Exception as e:
            self.logger.error(f"Error injecting LaTeX equations: {str(e)}")
            raise



    def add_heading_attributes(
        self, soup: BeautifulSoup, tracked_element: TrackedElement, context: ProcessingContext
    ) -> None:
        """
        Apply heading-specific attributes, IDs, numbering, and anchor links.

        Args:
            soup: Parsed HTML content.
            tracked_element: The current TrackedElement being processed.
            context: Current processing context.
        """
        try:
            headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])

            for element in headings:
                level = int(element.name[1])  # Extract heading level (1-6)

                # Process the heading using HeadingHandler
                heading_id, heading_text, heading_number = self.heading_handler.process_heading(
                    text=element.get_text(strip=True),
                    level=level,
                    is_topic_title=(level == 1 and tracked_element.metadata.get("is_title", False)),
                )

                # Update the heading element's ID
                element["id"] = heading_id

                # Conditionally append numbering based on context features
                if context.features.get("index_numbers", True) and heading_number:
                    heading_text = f"{heading_number} {heading_text}"

                # Update the heading text with or without numbering
                element.string = heading_text

                # Conditionally append anchor links
                if context.features.get("anchor_links", True):
                    anchor = soup.new_tag("a", href=f"#{heading_id}", **{"class": "heading-anchor"})
                    anchor.string = "¶"
                    element.append(anchor)

                # Update metadata with heading information
                if context.current_topic_id:
                    context.topic_metadata.setdefault(context.current_topic_id, {}).setdefault(
                        "headings", {}
                    )[heading_id] = {
                        "id": heading_id,
                        "text": heading_text,
                        "number": heading_number,
                        "level": level,
                        "topic_id": tracked_element.id,
                    }

        except Exception as e:
            self.logger.error(f"Error applying heading attributes: {str(e)}")
            raise


    def add_toc(self, soup: BeautifulSoup, element: TrackedElement, context: ProcessingContext) -> BeautifulSoup:
       """Append Table of Contents based on heading metadata."""
       try:
           # Get TOC metadata using metadata handler
           toc_metadata = self.metadata_handler.get_strategy_metadata(
               strategy="_append_toc",
               content_id=element.id
           )

           if not toc_metadata.get("data"):
               return soup

           # Create TOC container
           toc_nav = soup.new_tag("nav", attrs={
               "class": "table-of-contents",
               "aria-label": "Table of contents"
           })

           # Create TOC list
           toc_list = soup.new_tag("ul")

           # Track heading levels for nesting
           current_level = 0
           current_list = toc_list
           list_stack = []

           for heading in toc_metadata["data"]:
               heading_level = heading["level"]

               # Handle nesting
               if heading_level > current_level:
                   list_stack.append(current_list)
                   current_list = soup.new_tag("ul")
                   last_item = list_stack[-1].find_all("li", recursive=False)[-1]
                   last_item.append(current_list)
               elif heading_level < current_level:
                   for _ in range(current_level - heading_level):
                       if list_stack:
                           current_list = list_stack.pop()

               # Create heading entry
               item = soup.new_tag("li")
               link = soup.new_tag("a", href=f"#{heading['id']}")
               link.string = heading["text"]
               item.append(link)
               current_list.append(item)

               current_level = heading_level

           toc_nav.append(toc_list)

           # Insert TOC at start of body
           if body := soup.find("body"):
               body.insert(0, toc_nav)
           else:
               soup.insert(0, toc_nav)

           return soup

       except Exception as e:
           self.logger.error(f"Error appending TOC: {str(e)}")
           return soup


    def inject_image(self, soup: BeautifulSoup, element: TrackedElement, context: ProcessingContext) -> BeautifulSoup:
        """Inject image elements with attributes and figure wrappers."""
        try:
            metadata = self.metadata_handler.get_strategy_metadata(
                strategy="_inject_image",
                content_id=element.id
            )

            default_attrs = {
                'loading': 'lazy',
                'decoding': 'async',
                'class': 'img-fluid',
                'alt': 'Image placeholder',
                'width': 'auto',
                'height': 'auto'
            }

            for img in soup.find_all('img'):
                img_meta = next(
                    (m for m in metadata.get("data", [])
                     if m.get('element_id') == img.get('id')),
                    {}
                )

                # Apply defaults
                for attr, value in default_attrs.items():
                    if not img.get(attr):
                        img[attr] = value

                # Set src with path resolution
                if href := (img_meta.get('href') or img.get('src')):
                    img['src'] = self._get_media_path(href, element) or '/static/placeholder.png'

                # Apply metadata with validation
                for attr in ['alt', 'title', 'width', 'height', 'align']:
                    if value := img_meta.get(attr):
                        if attr in ['width', 'height']:
                            try:
                                float(value.rstrip('px%'))
                                img[attr] = value
                            except ValueError:
                                continue
                        else:
                            img[attr] = value

                # Handle figure wrapper
                if (alt := img.get('alt')) and not img.find_parent('figure'):
                    figure = soup.new_tag('figure', attrs={'class': 'figure'})
                    figcaption = soup.new_tag('figcaption', attrs={'class': 'figure-caption'})
                    figcaption.string = alt
                    img.wrap(figure)
                    figure.append(figcaption)

            return soup

        except Exception as e:
            self.logger.error(f"Error injecting images: {str(e)}")
            return soup

    def inject_video(self, soup: BeautifulSoup, element: TrackedElement, context: ProcessingContext) -> BeautifulSoup:
       """Inject video elements with controls and attributes."""
       try:
           metadata = self.metadata_handler.get_strategy_metadata(
               strategy="_inject_video",
               content_id=element.id
           )

           default_attrs = {
               'controls': '',
               'preload': 'metadata',
               'class': 'media-responsive',
               'width': 'auto',
               'height': 'auto'
           }

           for video in soup.find_all('video'):
               video_meta = next(
                   (m for m in metadata.get("data", [])
                    if m.get('element_id') == video.get('id')),
                   {}
               )

               # Apply defaults
               for attr, value in default_attrs.items():
                   if not video.get(attr):
                       video[attr] = value

               # Set src with path resolution
               if href := (video_meta.get('href') or video.get('src')):
                   video['src'] = self._get_media_path(href, element)

               # Apply metadata attributes
               for attr in ['width', 'height', 'poster', 'preload']:
                   if value := video_meta.get(attr):
                       video[attr] = value

               # Handle boolean attributes
               for attr in ['controls', 'autoplay', 'loop', 'muted']:
                   if video_meta.get(attr):
                       video[attr] = ''

           return soup

       except Exception as e:
           self.logger.error(f"Error injecting videos: {str(e)}")
           return soup

    def inject_audio(self, soup: BeautifulSoup, element: TrackedElement, context: ProcessingContext) -> BeautifulSoup:
       """Inject audio elements with controls and attributes."""
       try:
           metadata = self.metadata_handler.get_strategy_metadata(
               strategy="_inject_audio",
               content_id=element.id
           )

           default_attrs = {
               'controls': '',
               'preload': 'metadata',
               'class': 'media-responsive'
           }

           for audio in soup.find_all('audio'):
               audio_meta = next(
                   (m for m in metadata.get("data", [])
                    if m.get('element_id') == audio.get('id')),
                   {}
               )

               # Apply defaults
               for attr, value in default_attrs.items():
                   if not audio.get(attr):
                       audio[attr] = value

               # Set src with path resolution
               if href := (audio_meta.get('href') or audio.get('src')):
                   audio['src'] = self._get_media_path(href, element)

               # Handle boolean attributes
               for attr in ['controls', 'autoplay', 'loop', 'muted']:
                   if audio_meta.get(attr):
                       audio[attr] = ''

               # Apply metadata attributes
               if preload := audio_meta.get('preload'):
                   audio['preload'] = preload

           return soup

       except Exception as e:
           self.logger.error(f"Error injecting audio: {str(e)}")
           return soup

    def inject_iframe(self, soup: BeautifulSoup, element: TrackedElement, context: ProcessingContext) -> BeautifulSoup:
       """Inject iframe elements with security attributes."""
       try:
           metadata = self.metadata_handler.get_strategy_metadata(
               strategy="_inject_iframe",
               content_id=element.id
           )

           default_attrs = {
               'class': 'responsive-iframe',
               'width': '100%',
               'height': '400',
               'loading': 'lazy',
               'sandbox': 'allow-scripts allow-same-origin',
               'referrerpolicy': 'no-referrer'
           }

           for iframe in soup.find_all('iframe'):
               iframe_meta = next(
                   (m for m in metadata.get("data", [])
                    if m.get('element_id') == iframe.get('id')),
                   {}
               )

               # Apply defaults
               for attr, value in default_attrs.items():
                   if not iframe.get(attr):
                       iframe[attr] = value

               # Set src
               if href := (iframe_meta.get('href') or iframe.get('src')):
                   # Don't use _get_media_path since iframes typically use external URLs
                   iframe['src'] = href

               # Apply metadata attributes with validation
               for attr in ['width', 'height', 'sandbox', 'allow']:
                   if value := iframe_meta.get(attr):
                       iframe[attr] = value

               # Create wrapper for responsive behavior
               if not iframe.find_parent(class_='iframe-container'):
                   wrapper = soup.new_tag('div', attrs={
                       'class': 'iframe-container',
                       'style': 'position:relative;padding-bottom:56.25%;'
                   })
                   iframe.wrap(wrapper)

           return soup

       except Exception as e:
           self.logger.error(f"Error injecting iframes: {str(e)}")
           return soup

    # def add_bibliography(self, html_content: str, metadata: Dict[str, Any]) -> str:
    #     bibliography_data = metadata.get('bibliography', [])
    #     bibliography_html = self.html_helper.generate_bibliography(bibliography_data)
    #     return f"{html_content}\n{bibliography_html}"

    # def add_glossary(self, html_content: str, metadata: Dict[str, Any]) -> str:
    #     glossary_data = metadata.get('glossary', [])
    #     glossary_html = self.html_helper.generate_glossary(glossary_data)
    #     return f"{html_content}\n{glossary_html}"

    # def add_topic_section(self, html_content: str, section: str) -> str:
    #     # Implementation to inject specific topic section
    #     ...


    # def swap_topic_version(self, html_content: str, version: str) -> str:
    #     # Implementation to inject specific topic version
    #     ...

    ##############################
    # Common transformer methods #
    ##############################

    def _transform_titles(self, element: TrackedElement) -> str:
       """Transform titles and headings preserving hierarchy."""
       try:
           rules = self._processing_rules.get(element.type)
           if not rules:
               return element.content

           # Handle map titles
           if element.type == ElementType.MAP_TITLE:
               return self.html_helper.create_element(
                   tag=rules['html_tag'],
                   attrs={
                       'class': ' '.join(rules['default_classes']),
                       **rules['attributes']
                   },
                   content=element.content
               )

           # Handle headings
           level = element.metadata.get('heading_level', 1)

           # Get level-specific classes
           classes = [
               *rules['default_classes'],
               *rules['level_classes'].get(level, [])
           ]

           # Build attributes
           attrs = {
               'class': ' '.join(classes),
               **rules['attributes'],
               'aria-level': str(level)
           }

           # Allow transformation strategies to add numbers/anchors later
           if element.metadata.get('section_id'):
               attrs['data-section'] = element.metadata['section_id']

           return self.html_helper.create_element(
               tag=rules['html_tag'].format(level=level),
               attrs=attrs,
               content=element.content
           )

       except Exception as e:
           self.logger.error(f"Error transforming title/heading: {str(e)}")
           return element.content

    def _transform_block(self, element: TrackedElement) -> str:
       """Transform block-level elements using processing rules."""
       try:
           rules = self._processing_rules.get(element.type)
           if not rules:
               return element.content

           # Handle specific block types
           if element.type == ElementType.NOTE:
               # Handle note types including callouts
               specialization = element.metadata.get('specialization')
               type_class = rules['type_classes'].get(specialization, rules['type_classes']['note'])
               classes = [*rules['default_classes'], type_class]

               # Add DITA specialization if present
               if specialization:
                   classes.append(f"note-{specialization}")

           elif element.type == ElementType.CODE_BLOCK:
               # Handle code blocks with language
               language = element.metadata.get('language', '')
               inner_classes = [c.format(language=language) for c in rules['inner_classes']]

               inner_content = self.html_helper.create_element(
                   tag=rules['inner_tag'],
                   attrs={'class': ' '.join(inner_classes)},
                   content=element.content
               )
               return self.html_helper.create_element(
                   tag=rules['html_tag'],
                   attrs={
                       'class': ' '.join(rules['default_classes']),
                       **rules['attributes']
                   },
                   content=inner_content
               )

           elif element.type == ElementType.CODE_PHRASE:
               # Handle inline code
               return self.html_helper.create_element(
                   tag=rules['html_tag'],
                   attrs={'class': ' '.join(rules['default_classes'])},
                   content=element.content
               )

           # Default block handling
           return self.html_helper.create_element(
               tag=rules['html_tag'],
               attrs={
                   'class': ' '.join(rules['default_classes']),
                   **rules.get('attributes', {})
               },
               content=element.content
           )

       except Exception as e:
           self.logger.error(f"Error transforming block element: {str(e)}")
           return element.content

    def _transform_link(self, element: TrackedElement) -> str:
       """Transform pre-parsed link element to HTML using processing rules."""
       try:
           rules = self._processing_rules.get(element.type)
           if not rules:
               return element.content

           link_info = element.metadata["link_info"]
           href = link_info["href"]

           # Build classes
           classes = [
               *rules['default_classes'],
               link_info["link_type"],
               "external" if href.startswith(('http://', 'https://', 'ftp://', 'mailto:')) else "internal"
           ]

           attrs = {
               'class': ' '.join(filter(None, classes)),
               'href': href
           }

           # Add external attributes if external
           if "external" in attrs['class']:
               attrs.update(rules['external_attrs'])

           return self.html_helper.create_element(
               tag=rules['html_tag'],
               attrs=attrs,
               content=element.content
           )

       except Exception as e:
           self.logger.error(f"Error transforming link: {str(e)}")
           return element.content or ""

    def _transform_list(self, element: TrackedElement) -> str:
       """Transform list elements using processing rules."""
       try:
           rules = self._processing_rules.get(element.type)
           if not rules:
               return element.content

           if element.type == ElementType.TODO_LIST:
               # Create checkbox input
               inner_attrs = rules['inner_attrs'].copy()
               if element.metadata.get("checked"):
                   inner_attrs['checked'] = ''

               checkbox = self.html_helper.create_element(
                   tag=rules['inner_tag'],
                   attrs=inner_attrs,
                   content=""
               )

               return self.html_helper.create_element(
                   tag=rules['html_tag'],
                   attrs={'class': ' '.join(rules['default_classes'])},
                   content=f"{checkbox}{element.content}"
               )

           return self.html_helper.create_element(
               tag=rules['html_tag'],
               attrs={'class': ' '.join(rules['default_classes'])},
               content=element.content
           )

       except Exception as e:
           self.logger.error(f"Error transforming list: {str(e)}")
           return element.content

    def _transform_emphasis(self, element: TrackedElement) -> str:
       """Transform emphasis elements with nested emphasis support."""
       try:
           rules = self._processing_rules.get(element.type)
           if not rules:
               return element.content

           # Handle nested emphasis by preserving inner HTML
           content = element.content
           if element.metadata.get("has_nested_emphasis"):
               nested_elements = element.metadata.get("nested_elements", [])
               for nested in nested_elements:
                   nested_html = self._transform_emphasis(nested)
                   content = content.replace(f"{{emphasis-{nested.id}}}", nested_html)

           return self.html_helper.create_element(
               tag=rules['html_tag'],
               attrs={'class': ' '.join(rules['default_classes'])},
               content=content
           )

       except Exception as e:
           self.logger.error(f"Error transforming emphasis: {str(e)}")
           return element.content

    def _transform_table(self, element: TrackedElement) -> str:
        """Transform table elements using processing rules."""
        try:
            rules = self._processing_rules.get(element.type)
            if not rules:
                return element.content

            # Get table info from metadata
            table_info = element.metadata.get('table_info', {})

            # Start with default classes
            classes = rules['default_classes'].copy()

            # Handle table specializations
            if element.type == ElementType.TABLE:
                specialization = table_info.get('type', 'default')
                if spec_rules := rules['specializations'].get(specialization):
                    # Add specialization classes
                    classes.extend(spec_rules['extra_classes'])

                    # Build specialization attributes
                    spec_attrs = {}
                    for attr, value_template in spec_rules['extra_attrs'].items():
                        value = value_template.format(
                            citation_format=table_info.get('citation_format', 'apa'),
                            sort_by=table_info.get('sort_by', 'author'),
                            show_references=str(table_info.get('show_references', True)).lower(),
                            visibility=table_info.get('visibility', 'visible'),
                            collapsible=str(table_info.get('collapsible', False)).lower()
                        )
                        spec_attrs[attr] = value

                    # Add table metadata
                    if table_info.get('has_header'):
                        spec_attrs['data-has-header'] = 'true'
                    if rows := table_info.get('rows'):
                        spec_attrs['data-rows'] = str(rows)
                    if cols := table_info.get('columns'):
                        spec_attrs['data-columns'] = str(cols)

                    # Update attributes
                    attrs = {**rules.get('attributes', {}), **spec_attrs}
                else:
                    attrs = rules.get('attributes', {})
            else:
                # For other table elements, just use default attributes
                attrs = rules.get('attributes', {})

            # Add classes to attributes
            attrs['class'] = ' '.join(classes)

            return self.html_helper.create_element(
                tag=rules['html_tag'],
                attrs=attrs,
                content=element.content
            )

        except Exception as e:
            self.logger.error(f"Error transforming table element: {str(e)}")
            return element.content

    ##################
    # Helper methods
    #################

    def _get_media_path(self, href: str, element: TrackedElement) -> str:
        """Resolve media path relative to topic directory."""
        try:
            if href.startswith(('http://', 'https://', '/')):
                return href

            topic_path = Path(element.path)
            media_dir = topic_path.parent / 'media'
            media_path = (media_dir / href).resolve()

            if media_path.exists() and self.dita_root:
                try:
                    return f'/static/topics/{media_path.relative_to(self.dita_root)}'
                except ValueError:
                    self.logger.warning(f"Media path {media_path} not under DITA root")

            self.logger.warning(f"Media file not found: {media_path}")
            return ''
        except Exception as e:
            self.logger.error(f"Error resolving media path: {str(e)}")
            return ''

    def _finalize_html(self, soup: BeautifulSoup, element: TrackedElement, context: ProcessingContext) -> None:
        """
        Finalize HTML content by adding metadata and performing clean-up tasks.

        Args:
            soup: Parsed BeautifulSoup object.
            element: The TrackedElement being processed.
            context: The ProcessingContext for the transformation.
        """
        # Inject metadata as attributes or data-* fields
        for key, value in element.metadata.items():
            soup.attrs[f"data-{key}"] = value

        # Optional: Sanitize or validate final HTML structure
        self.logger.debug("Finalizing HTML content")


    ####################
    # LaTeX processing #
    ####################

    def _extract_latex_equations(self, content: str) -> List[LaTeXEquation]:
        """
        Extract LaTeX equations from content.

        Args:
            content: Content to process.

        Returns:
            List[LaTeXEquation]: Extracted equations.
        """
        try:
            equations = []

            # Extract block equations
            block_pattern = r'\$\$(.*?)\$\$'
            for i, match in enumerate(re.finditer(block_pattern, content, re.DOTALL)):
                placeholder = f"{{latex-block-{i}}}"
                equations.append(LaTeXEquation(
                    id=f"eq-block-{i}",
                    content=match.group(1).strip(),
                    is_block=True,
                    placeholder=placeholder
                ))
                content = content.replace(match.group(0), placeholder)

            # Extract inline equations
            inline_pattern = r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)'
            for i, match in enumerate(re.finditer(inline_pattern, content)):
                placeholder = f"{{latex-inline-{i}}}"
                equations.append(LaTeXEquation(
                    id=f"eq-inline-{i}",
                    content=match.group(1).strip(),
                    is_block=False,
                    placeholder=placeholder
                ))
                content = content.replace(match.group(0), placeholder)

            return equations

        except Exception as e:
            self.logger.error(f"Error extracting LaTeX equations: {str(e)}")
            return []

    def _process_latex_equations(self, soup: BeautifulSoup) -> None:
        """
        Process LaTeX equations in the content using the LaTeX pipeline.

        Args:
            soup: Parsed HTML content as a BeautifulSoup object.
        """
        try:
            # Extract LaTeX equations
            latex_equations = self._extract_latex_equations(str(soup))

            for eq in latex_equations:
                # Ensure the placeholder is not None
                if not eq.placeholder:
                    self.logger.warning(f"Skipping LaTeX equation with missing placeholder: {eq.id}")
                    continue

                # Validate the equation using LaTeXValidator
                if not self.latex_validator.validate_equation(eq):
                    self.logger.warning(f"Skipping invalid LaTeX equation (ID: {eq.id}): {eq.content}")
                    continue

                # Render LaTeX equation using KaTeX
                rendered = self.katex_renderer.render_equation(eq)

                # Replace placeholder with rendered output in soup
                placeholder = soup.find(text=re.escape(eq.placeholder))
                if placeholder:
                    rendered_element = BeautifulSoup(rendered, "html.parser")
                    placeholder.replace_with(rendered_element)

        except Exception as e:
            self.logger.error(f"Error processing LaTeX equations: {str(e)}")
            raise
