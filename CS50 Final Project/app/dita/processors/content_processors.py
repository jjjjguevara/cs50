import logging
from typing import Optional, Dict, Any, Callable, List, Union, cast
from datetime import datetime
from pathlib import Path
from app.dita.utils.id_handler import DITAIDHandler
from app.dita.models.types import (
    ElementType,
    DITAElementInfo,
    MDElementInfo,
    ProcessedContent,
    ParsedElement,
    ProcessingContext,
    MapContext,
    ElementAttributes,
    DITAElementContext,
    MDElementContext,
    DITAElementType,
    MDElementType
)

from app.dita.processors.dita_elements import DITAElementProcessor
from app.dita.processors.md_elements import MarkdownElementProcessor
from app.dita.transformers.dita_transform import DITATransformer
from app.dita.transformers.md_transform import MarkdownTransformer


class ContentProcessor:
    def __init__(self, dita_root: Path, markdown_root: Path):
        self.logger = logging.getLogger(__name__)
        self.dita_root = dita_root
        self.markdown_root = markdown_root
        self.id_handler = DITAIDHandler()

        # Initialize processors with reference to this processor for error handling
        self.dita_element_processor = DITAElementProcessor(self)
        self.md_element_processor = MarkdownElementProcessor(self)

        # Initialize transformers
        self.dita_transformer = DITATransformer(dita_root)
        self.markdown_transformer = MarkdownTransformer(markdown_root)

    def process_content(self, parsed_element: ParsedElement) -> ProcessedContent:
        """Process content with appropriate transformer and element processor."""
        try:
            self.logger.debug(f"Processing content of type: {parsed_element.type}")

            # Create processing context
            processing_context = ProcessingContext(
                map_context=MapContext(
                    map_id=parsed_element.topic_id,
                    map_path=parsed_element.source_path,
                    metadata=parsed_element.metadata,
                    topic_order=[],  # Single topic processing
                    features={'process_latex': True},
                    type=parsed_element.type
                ),
                topics={}
            )

            if parsed_element.type == ElementType.MARKDOWN:
                return self.markdown_transformer.transform_topic(
                    parsed_element=parsed_element,
                    context=processing_context
                )
            elif parsed_element.type == ElementType.DITA:
                return self.dita_transformer.transform_topic(
                    parsed_element=parsed_element,
                    context=processing_context
                )
            else:
                raise ValueError(f"Unsupported content type: {parsed_element.type}")

        except Exception as e:
            self.logger.error(f"Error processing content: {str(e)}")
            return ProcessedContent(
                html=f"<div class='error'>Processing error: {str(e)}</div>",
                element_id=parsed_element.id,
                metadata=parsed_element.metadata
            )

    def create_error_element(
            self,
            error: Optional[Exception] = None,
            error_type: str = "processing_error",
            element_context: Optional[str] = None,
            element_type: ElementType = ElementType.UNKNOWN
        ) -> Union[DITAElementInfo, MDElementInfo]:
            """Create generic error element."""
            error_message = str(error) if error else "Unknown error"
            error_id = self.id_handler.generate_id(element_context or "error")

            # Common attributes
            attributes = ElementAttributes(
                id=error_id,
                classes=['processing-error'],
                custom_attrs={'error': error_message}
            )

            # Return type-specific error element
            if element_type in [ElementType.DITA, ElementType.DITAMAP]:
                return DITAElementInfo(
                    type=DITAElementType.UNKNOWN,
                    content="",
                    attributes=attributes,
                    context=DITAElementContext(
                        parent_id=None,
                        element_type="error",
                        classes=[],
                        attributes={}
                    ),
                    metadata={'error': error_message},
                    children=[]
                )
            else:
                return MDElementInfo(
                    type=MDElementType.UNKNOWN,
                    content="",
                    attributes=attributes,
                    context=MDElementContext(
                        parent_id=None,
                        element_type="error",
                        classes=[],
                        attributes={}
                    ),
                    metadata={'error': error_message},
                    level=None
                )

    def create_dita_error_element(
        self,
        error: Optional[Exception] = None,
        error_type: str = "dita_processing_error",
        element_context: Optional[str] = None
    ) -> DITAElementInfo:
        """Create DITA-specific error element."""
        return cast(DITAElementInfo, self.create_error_element(
            error=error,
            error_type=error_type,
            element_context=element_context,
            element_type=ElementType.DITA
        ))

    def create_md_error_element(
        self,
        error: Optional[Exception] = None,
        error_type: str = "markdown_processing_error",
        element_context: Optional[str] = None
    ) -> MDElementInfo:
        """Create Markdown-specific error element."""
        return cast(MDElementInfo, self.create_error_element(
            error=error,
            error_type=error_type,
            element_context=element_context,
            element_type=ElementType.MARKDOWN
        ))
