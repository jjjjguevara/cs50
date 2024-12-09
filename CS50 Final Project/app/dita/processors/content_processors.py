import logging
from pathlib import Path
from app.dita.transformers.dita_transform import DITATransformer
from app.dita.transformers.md_transform import MarkdownTransformer
from app.dita.models.types import ParsedElement, ProcessedContent


class ContentProcessor:
    """Processes content from various formats using the appropriate transformers."""

    def __init__(self, dita_root: Path, markdown_root: Path):
        """
        Initialize ContentProcessor with required paths.

        Args:
            dita_root (Path): Root directory for DITA content.
            markdown_root (Path): Root directory for Markdown content.
        """
        self.logger = logging.getLogger(__name__)
        self.dita_transformer = DITATransformer(dita_root)
        self.markdown_transformer = MarkdownTransformer(markdown_root)

    def process_content(self, parsed_element: ParsedElement) -> ProcessedContent:
        """
        Process content based on its type.

        Args:
            parsed_element (ParsedElement): The parsed element to process.

        Returns:
            ProcessedContent: The transformed content.
        """
        try:
            self.logger.debug(f"Processing content of type: {parsed_element.type}")

            if parsed_element.type == "dita":
                return self.dita_transformer.transform_topic(parsed_element)
            elif parsed_element.type == "markdown":
                return self.markdown_transformer.transform_topic(parsed_element)
            else:
                raise ValueError(f"Unsupported content type: {parsed_element.type}")

        except Exception as e:
            self.logger.error(f"Error processing content: {str(e)}", exc_info=True)
            return ProcessedContent(
                html=f"<div class='error'>Error processing content of type {parsed_element.type}</div>",
                element_id="",
                metadata=parsed_element.metadata,
            )
