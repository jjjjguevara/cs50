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
        self.processing_metadata = processing_metadata or ProcessingMetadata(
            id="base-transformer-metadata",
            content_type=ElementType.DITAMAP,
            features={
                "latex": self.config.features.get("latex", True),
                "anchor_links": self.config.features.get("anchor_links", True),
                "index_numbers": self.config.features.get("index_numbers", True),
                "enable_xrefs": self.config.features.get("enable_xrefs", True),
                "toc": self.config.features.get("toc", True),
            },
        )

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
        #     # Block Elements
        #     "heading": self._transform_heading,
        #     "paragraph": self._transform_paragraph,
        #     "blockquote": self._transform_blockquote,
        #     "code_block": self._transform_code_block,
        #     "pre": self._transform_code_block,
        #     "table": self._transform_table,
        #     "figure": self._transform_figure,

        #     # Lists
        #     "ul": self._transform_unordered_list,
        #     "ol": self._transform_ordered_list,
        #     "li": self._transform_list_item,
        #     "dl": self._transform_definition_list,
        #     "dt": self._transform_term,
        #     "dd": self._transform_definition,

        #     # Inline Elements
        #     "link": self._transform_link,
        #     "image": self._transform_image,
        #     "code": self._transform_code_inline,
        #     "strong": self._transform_bold,
        #     "em": self._transform_italic,
        #     "u": self._transform_underline,

        #     # DITA Specific
        #     "concept": self._transform_concept,
        #     "task": self._transform_task,
        #     "reference": self._transform_reference,
        #     "topicref": self._transform_topicref
         }

        # Contextual/feature strategies (inject/append/swap)
        self._strategy_registry = {
           "_inject_latex": lambda c, m: c.features.get("latex", False),
           "_inject_media": lambda c, m: c.features.get("media", False),
           "_inject_topic_section": lambda c, m: c.features.get("topic_section", False),
           "_append_heading_attributes": lambda c, m: (
               c.features.get("index_numbers", True) or
               c.features.get("anchor_links", True)
           ),
           "_append_toc": lambda c, m: c.features.get("toc", False),
           "_append_bibliography": lambda c, m: "bibliography" in m,
           "_append_glossary": lambda c, m: "glossary" in m,
           "_swap_topic_version": lambda c, m: c.features.get("swap_topic_version", False),
           "_swap_topic_type": lambda c, m: c.features.get("swap_topic_type", False)
        }

        # Core element processing rules for DITA specializations
        self._processing_rules = {
            # Structure elements
            "concept": {
                "html_tag": "article",
                "classes": ["dita-concept", "article-content"],
                "attributes": {"role": "article"}
            },
            "task": {
                "html_tag": "article",
                "classes": ["dita-task", "article-content"],
                "attributes": {"role": "article"}
            },
            "reference": {
                "html_tag": "article",
                "classes": ["dita-reference", "article-content"],
                "attributes": {"role": "article"}
            },

            # Block elements
            "section": {
                "html_tag": "section",
                "classes": ["dita-section"],
                "attributes": {"role": "region"}
            },
            "paragraph": {
                "html_tag": "p",
                "classes": ["dita-p"]
            },
            "note": {
                "html_tag": "div",
                "classes": ["dita-note", "alert"],
                "attributes": {"role": "alert"},
                "type_classes": {
                    "warning": "alert-warning",
                    "danger": "alert-danger",
                    "tip": "alert-info"
                }
            },
            "code_block": {
                "html_tag": "pre",
                "classes": ["code-block", "highlight"],
                "attributes": {
                    "spellcheck": "false",
                    "translate": "no"
                }
            },

            # Lists
            "unordered_list": {
                "html_tag": "ul",
                "classes": ["dita-ul"]
            },
            "ordered_list": {
                "html_tag": "ol",
                "classes": ["dita-ol"]
            },
            "list_item": {
                "html_tag": "li",
                "classes": ["dita-li"]
            },

            # Tables
            "table": {
                "html_tag": "table",
                "classes": ["table", "table-bordered"],
                "attributes": {"role": "grid"}
            },

            # Media
            "image": {
                "html_tag": "img",
                "classes": ["img-fluid"],
                "required_attrs": ["src", "alt"],
                "attributes": {
                    "loading": "lazy",
                    "decoding": "async"
                }
            },
            "figure": {
                "html_tag": "figure",
                "classes": ["figure"],
                "inner_tag": "figcaption"
            },

            # Links
            "xref": {
                "html_tag": "a",
                "classes": ["dita-xref"],
                "required_attrs": ["href"]
            },
            "link": {
                "html_tag": "a",
                "classes": ["dita-link"],
                "attributes": {
                    "target": "_blank",
                    "rel": "noopener noreferrer"
                }
            },

            # Inline
            "bold": {
                "html_tag": "strong",
                "classes": ["dita-b"]
            },
            "italic": {
                "html_tag": "em",
                "classes": ["dita-i"]
            },
            "underline": {
                "html_tag": "u",
                "classes": ["dita-u"]
            }
        }

    def transform(self, element: TrackedElement, context: ProcessingContext) -> ProcessedContent:
       """Transform content with processing strategies."""
       try:
           html_content = self._convert_to_html(element, context)

           # Transform with element transformer
           if element.type in self._element_transformers:
               html_content = self._element_transformers[element.type](html_content)

           # Apply feature strategies
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
       """Convert element content to HTML. To be implemented by subclasses."""
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

    def _inject_latex(self, soup: BeautifulSoup, element: TrackedElement, context: ProcessingContext) -> None:
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



    def _append_heading_attributes(
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


    def _append_toc(self, soup: BeautifulSoup, element: TrackedElement, context: ProcessingContext) -> BeautifulSoup:
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

    def _inject_media(self, soup: BeautifulSoup, element: TrackedElement, context: ProcessingContext) -> BeautifulSoup:
        """
        Inject media elements with proper paths and attributes from keyref definitions.
        Media files are expected in topic's /media subdirectory unless explicitly defined.
        """
        try:
            media_metadata = self.metadata_handler.get_strategy_metadata(
                strategy="_inject_media",
                content_id=element.id
            )

            # Default attributes
            default_attrs = {
                'img': {
                    'loading': 'lazy',
                    'decoding': 'async',
                    'class': 'img-fluid',
                    'alt': 'Image placeholder',  # Accessibility default
                    'width': 'auto',
                    'height': 'auto'
                },
                'video': {
                    'controls': '',
                    'preload': 'metadata',
                    'class': 'media-responsive',
                    'width': 'auto',
                    'height': 'auto'
                },
                'audio': {
                    'controls': '',
                    'preload': 'metadata',
                    'class': 'media-responsive'
                }
            }

            for media_tag in ['img', 'video', 'audio']:
                for media_elem in soup.find_all(media_tag):
                    # Find matching metadata if exists
                    metadata = next(
                        (m for m in media_metadata.get("data", [])
                         if m.get('element_id') == media_elem.get('id')),
                        {}
                    )

                    # Apply default attributes first
                    for attr, value in default_attrs[media_tag].items():
                        if not media_elem.get(attr):
                            media_elem[attr] = value

                    # Handle src/href
                    href = metadata.get('href', media_elem.get('src', ''))
                    if href:
                        if not href.startswith(('http://', 'https://', '/')):
                            # Resolve relative to media directory
                            topic_path = Path(element.path)
                            media_dir = topic_path.parent / 'media'
                            media_path = (media_dir / href).resolve()

                            if media_path.exists():
                                if self.dita_root:
                                    try:
                                        href = f'/static/topics/{media_path.relative_to(self.dita_root)}'
                                    except ValueError:
                                        self.logger.warning(f"Media path {media_path} not under DITA root")
                            else:
                                self.logger.warning(f"Media file not found: {media_path}")
                                href = '/static/placeholder.png' if media_tag == 'img' else ''

                        media_elem['src'] = href

                    # Apply metadata attributes with validation
                    if media_elem.name == 'img':
                        if alt := (metadata.get('alt') or media_elem.get('alt')):
                            media_elem['alt'] = alt
                        if title := (metadata.get('linktext') or media_elem.get('title')):
                            media_elem['title'] = title

                    # Apply common attributes with validation
                    for attr in ['placement', 'scale', 'align', 'width', 'height']:
                        if value := metadata.get(attr):
                            # Validate numeric attributes
                            if attr in ['width', 'height', 'scale']:
                                try:
                                    float(value.rstrip('px%'))
                                    media_elem[attr] = value
                                except ValueError:
                                    continue
                            else:
                                media_elem[f'data-{attr}'] = value

                    # Handle classes
                    classes = set(filter(None, [
                        media_elem.get('class', ''),
                        default_attrs[media_tag]['class'],
                        metadata.get('outputclass', '')
                    ]))
                    media_elem['class'] = ' '.join(classes)

                    # Create figure wrapper for images with captions
                    if (media_elem.name == 'img' and
                        (alt := media_elem.get('alt')) and
                        not media_elem.find_parent('figure')):
                        figure = soup.new_tag('figure', attrs={'class': 'figure'})
                        figcaption = soup.new_tag('figcaption', attrs={'class': 'figure-caption'})
                        figcaption.string = alt
                        media_elem.wrap(figure)
                        figure.append(figcaption)

            return soup

        except Exception as e:
            self.logger.error(f"Error injecting media: {str(e)}")
            return soup


    # def _append_bibliography(self, html_content: str, metadata: Dict[str, Any]) -> str:
    #     bibliography_data = metadata.get('bibliography', [])
    #     bibliography_html = self.html_helper.generate_bibliography(bibliography_data)
    #     return f"{html_content}\n{bibliography_html}"

    # def _append_glossary(self, html_content: str, metadata: Dict[str, Any]) -> str:
    #     glossary_data = metadata.get('glossary', [])
    #     glossary_html = self.html_helper.generate_glossary(glossary_data)
    #     return f"{html_content}\n{glossary_html}"

    # def _inject_topic_version(self, html_content: str, version: str) -> str:
    #     # Implementation to inject specific topic version
    #     ...

    # def _inject_topic_section(self, html_content: str, section: str) -> str:
    #     # Implementation to inject specific topic section
    #     ...


    ##################
    # Helper methods
    #################

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


    #####################
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
