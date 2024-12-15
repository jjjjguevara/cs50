# app/dita/transformers/base_transformer.py
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


class BaseTransformer:
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
        self.config = config or DITAConfig()  # Initialize config from config_manager
        self.processing_metadata = processing_metadata or ProcessingMetadata(
            id="base-transformer-metadata",
            content_type=ElementType.DITAMAP,
            features={
                "process_latex": self.config.features.get("process_latex", True),
                "anchor_links": self.config.features.get("anchor_links", True),
                "index_numbers": self.config.features.get("index_numbers", True),
                "enable_cross_refs": self.config.features.get("enable_cross_refs", True),
                "show_toc": self.config.features.get("show_toc", True),
            },
        )

        # Utilities
        self.id_handler = id_handler or DITAIDHandler()
        self.metadata_extractor = metadata_extractor or MetadataHandler()
        self.html_helper = html_helper or HTMLHelper(dita_root)
        self.heading_handler = heading_handler or HeadingHandler(processing_metadata=self.processing_metadata)

        # LaTeX Utilities
        self.katex_renderer = katex_renderer or KaTeXRenderer()
        self.latex_processor = LaTeXProcessor()
        self.latex_validator = latex_validator or LaTeXValidator()


    def transform(
        self,
        element: TrackedElement,
        context: ProcessingContext,
        html_converter: Optional[Callable[[str, ProcessingContext], str]] = None
    ) -> ProcessedContent:
        """
        Centralized transformation method for various element types.

        Args:
            element: The element to be transformed.
            context: The processing context for the transformation.
            html_converter: Optional function to convert content to HTML.

        Returns:
            ProcessedContent: The transformed HTML and metadata.
        """
        try:
            # Metadata extraction
            metadata = self.metadata_extractor.extract_metadata(
                file_path=element.path, content_id=element.id
            )

            # HTML conversion
            html_content = (
                html_converter(element.content, context)
                if html_converter
                else self._default_html_converter(element.content, context)
            )

            # Parse and process HTML
            soup = BeautifulSoup(html_content, "html.parser")
            self.process_elements(soup, element, context)

            # Apply registered strategies
            html_content = self._apply_strategies(
                html=str(soup), element=element, context=context, metadata=metadata
            )

            # Finalize HTML content
            final_html = self.html_helper.process_final_content(html_content)

            # Update element state
            element.state = ProcessingState.COMPLETED
            element.phase = ProcessingPhase.ASSEMBLY

            return ProcessedContent(
                html=final_html, element_id=element.id, metadata=metadata
            )

        except Exception as e:
            self.logger.error(f"Error transforming element {element.id}: {str(e)}")
            raise


    import inspect

    def _apply_strategies(
        self,
        html: str,
        element: TrackedElement,
        context: ProcessingContext,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Apply active transformation strategies to the content.

        Args:
            html: Raw HTML content to transform.
            element: The element being processed.
            context: The processing context.
            metadata: Metadata for the element.

        Returns:
            str: Transformed HTML content.
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
            strategies = self._get_active_strategies(context, metadata)

            # Available arguments for strategies
            available_args = {
                "soup": soup,
                "element": element,
                "context": context,
                "metadata": metadata
            }

            # Execute each strategy
            for strategy in strategies:
                try:
                    strategy_params = inspect.signature(strategy).parameters
                    required_args = {
                        name: available_args[name]
                        for name in strategy_params
                        if name in available_args
                    }
                    strategy(**required_args)
                    self.logger.debug(f"Applied strategy: {strategy.__name__}")
                except Exception as e:
                    self.logger.error(f"Error applying strategy {strategy.__name__}: {e}")
                    raise

            return str(soup)

        except Exception as e:
            self.logger.error(f"Error applying strategies: {str(e)}")
            raise




    def _get_active_strategies(
        self,
        context: ProcessingContext,
        metadata: Dict[str, Any]
    ) -> List[Callable]:
        """
        Get list of active transformation strategies.

        Args:
            context: The processing context for the transformation.
            metadata: Metadata associated with the element.

        Returns:
            List[Callable]: List of strategy methods to apply.
        """
        strategies = []

        strategy_registry = {
            "_inject_latex": lambda: context.features.get("latex", False),
            "_append_heading_attributes": lambda: (
                context.features.get("index_numbers", True) or context.features.get("anchor_links", True)
            ),
            "_append_toc": lambda: context.features.get("toc", False),
            "_append_bibliography": lambda: "bibliography" in metadata,
            "_append_glossary": lambda: "glossary" in metadata,
        }

        for strategy_name, condition in strategy_registry.items():
            if condition():
                strategy = getattr(self, strategy_name, None)
                if strategy:
                    strategies.append(strategy)

        return strategies




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
