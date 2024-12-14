# app/dita/transformers/base_transformer.py
import logging
from typing import Callable
from typing import Dict, Optional, Union, List, Any
from bs4 import BeautifulSoup
from pathlib import Path
from ..models.types import TrackedElement, ProcessedContent, ProcessorConfig, ProcessingContext
from ..utils.html_helpers import HTMLHelper
from ..utils.heading import HeadingHandler
from ..utils.id_handler import DITAIDHandler
from ..utils.metadata import MetadataHandler


class BaseTransformer:
    def __init__(self, dita_root: Optional[Path] = None):
        """
        Initializes the BaseTransformer with required utilities.

        Args:
            dita_root (Optional[Path]): The root path for DITA files (default: None).
        """
        self.logger = logging.getLogger(__name__)
        self.heading_handler = HeadingHandler()
        self.id_handler = DITAIDHandler()
        self.metadata_extractor = MetadataHandler()
        self.html_helper = HTMLHelper(dita_root)

    def transform_topic(
            self,
            element: TrackedElement,
            context: ProcessingContext,
            html_converter: Optional[Callable[[str, ProcessingContext], str]] = None
        ) -> ProcessedContent:
            """Base transform method for all transformers."""
            raise NotImplementedError
            """
            Transform a parsed element into HTML.

            Args:
                parsed_element: The parsed element to transform
                context: Processing context for transformation
                html_converter: Optional function to convert content to HTML

            Returns:
                ProcessedContent: The transformed HTML and metadata
            """
            try:
                self.logger.info(f"Starting transformation for topic: {parsed_element.topic_path}")

                # Validate input
                if not parsed_element.content:
                    raise ValueError(f"ParsedElement {parsed_element.topic_path} has empty content.")

                # Extract metadata
                metadata = self.metadata_extractor.extract_metadata(
                    file_path=parsed_element.topic_path,
                    content_id=parsed_element.id
                )

                # Convert content to HTML using provided or default converter
                converter = html_converter or self._default_html_converter
                html_content = converter(parsed_element.content, context)

                # Process HTML content
                soup = BeautifulSoup(html_content, "html.parser")

                # Process elements
                self.process_elements(soup, parsed_element)

                # Finalize HTML
                final_html = self.html_helper.process_final_content(str(soup))

                # Validate output
                if not self._validate_html(final_html):
                    raise ValueError(f"HTML validation failed for {parsed_element.topic_path}")

                return ProcessedContent(
                    html=final_html,
                    element_id=parsed_element.id,
                    metadata={**metadata, **parsed_element.metadata}
                )

            except Exception as e:
                self.logger.error(f"Error transforming topic {parsed_element.topic_path}: {str(e)}")
                raise


    def process_elements(
            self,
            soup: BeautifulSoup,
            parsed_element: ParsedElement
        ) -> None:
            """
            Process elements in HTML content.

            Args:
                soup: The parsed HTML content
                parsed_element: The original parsed element
            """
            raise NotImplementedError("Subclasses must implement process_elements")

    def _default_html_converter(
        self,
        content: str,
        context: ProcessingContext
    ) -> str:
        """Default HTML conversion method."""
        return content

    def _validate_html(self, html_content: str) -> bool:
        """Validate HTML content."""
        return bool(html_content.strip() and "<html" not in html_content.lower())
