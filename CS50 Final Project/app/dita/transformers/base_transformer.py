# app/dita/transformers/base_transformer.py
import logging
from typing import Callable
from typing import Dict, Optional, Union, List, Any
from bs4 import BeautifulSoup
from pathlib import Path
from ..models.types import ParsedElement, ProcessedContent
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
        self.html_helper = HTMLHelper(dita_root)  # Initialize HTMLHelper with optional DITA root path


    def transform_topic(self, parsed_element: ParsedElement, to_html_func: Callable[[str], str]) -> ProcessedContent:
        """
        Transforms a parsed element into HTML with LaTeX support.

        Args:
            parsed_element (ParsedElement): The parsed element to transform.
            to_html_func (Callable[[str], str]): A callable that converts content to HTML.

        Returns:
            ProcessedContent: The transformed HTML and metadata.
        """
        try:
            self.logger.info(f"Starting transformation for topic: {parsed_element.topic_path}")

            # Validate input content
            if not parsed_element.content:
                raise ValueError(f"ParsedElement {parsed_element.topic_path} has empty content.")

            # Extract metadata
            metadata = self.metadata_extractor.extract_metadata(
                file_path=Path(parsed_element.topic_path),
                content_id=self.id_handler.generate_content_id(Path(parsed_element.topic_path)),
                heading_id=None
            )

            # Convert content to HTML
            html_content = to_html_func(parsed_element.content)

            # Parse and process the HTML
            soup = BeautifulSoup(html_content, "html.parser")

            # Process headings
            for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
                heading_id, numbered_heading = self.heading_handler.process_heading(
                    text=heading.text.strip(),
                    level=int(heading.name[1])
                )
                heading["id"] = heading_id
                heading.string = numbered_heading
                # Add pilcrow
                pilcrow = soup.new_tag('a', href=f"#{heading_id}", **{'class': 'pilcrow'})
                pilcrow.string = '¶'
                heading.append(pilcrow)

            # Process elements
            self.process_elements(soup, Path(parsed_element.topic_path))

            # Add KaTeX initialization if LaTeX is detected
            if soup.find_all('latex-equation'):
                katex_wrapper = soup.new_tag('div', attrs={'class': 'katex-wrapper'})
                katex_script = soup.new_tag('script', attrs={'type': 'text/javascript'})
                katex_script.string = """
                    document.addEventListener("DOMContentLoaded", function() {
                        renderMathInElement(document.body, {
                            delimiters: [
                                {left: "$$", right: "$$", display: true},
                                {left: "$", right: "$", display: false}
                            ],
                            throwOnError: false,
                            output: 'html',
                            displayMode: true
                        });
                    });
                """
                katex_wrapper.append(katex_script)
                soup.body.append(katex_wrapper) if soup.body else soup.append(katex_wrapper)

            # Process other elements (images, code blocks, etc.)
            # self._process_todo_lists(soup)
            # self._process_images(soup)
            # self._process_code_blocks(soup)

            # Finalize HTML content
            final_html = self.html_helper.process_final_content(str(soup))

            # Validate final HTML
            if not self.html_helper.validate_html(final_html):
                raise ValueError(f"HTML validation failed for {parsed_element.topic_path}")

            self.logger.info(f"Transformation completed successfully for topic: {parsed_element.topic_path}")

            return ProcessedContent(
                html=final_html,
                element_id=self.id_handler.generate_content_id(Path(parsed_element.topic_path)),
                metadata=metadata,
            )
        except Exception as e:
            self.logger.error(f"Error transforming topic {parsed_element.topic_path}: {str(e)}", exc_info=True)
            raise


    def process_elements(self, soup: BeautifulSoup, source_path: Path):
        """
        Process individual elements in the HTML. This method should be implemented by subclasses.

        Args:
            soup (BeautifulSoup): The parsed HTML content.
            source_path (Path): The source file path.
        """
        raise NotImplementedError("Subclasses must implement the process_elements method.")

    def _validate_html(self, html_content: str) -> bool:
        """
        Validate the final HTML content.

        Args:
            html_content (str): The HTML content to validate.

        Returns:
            bool: True if the HTML is valid, False otherwise.
        """
        # Basic validation: Ensure the content is non-empty and contains basic HTML structure
        return bool(html_content.strip() and "<html" not in html_content.lower())
