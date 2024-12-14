import logging
from typing import Optional, Dict, Any, Callable, List, Union, cast
from datetime import datetime
from pathlib import Path

from app.dita.models.types import (
    MDElementInfo,
    TrackedElement,
    ProcessingContext,
    ProcessedContent,
    ElementType,
    ProcessingPhase,
    DITAElementInfo,
    MDElementInfo,
    DITAElementType,
    ProcessingState
)

from app.dita.utils.id_handler import DITAIDHandler
from app.dita.processors.dita_elements import DITAElementProcessor
from app.dita.processors.md_elements import MarkdownElementProcessor
from app.dita.transformers.dita_transform import DITATransformer
from app.dita.transformers.md_transform import MarkdownTransformer



class ContentProcessor:
    def __init__(self, dita_root: Path, markdown_root: Path):
            """Initialize processor with transformers and logger."""
            self.logger = logging.getLogger(__name__)
            self.dita_root = dita_root
            self.markdown_root = markdown_root
            self.id_handler = DITAIDHandler()


            # Initialize transformers
            self.dita_transformer = DITATransformer(dita_root)
            self.markdown_transformer = MarkdownTransformer(markdown_root)

            self.logger.debug("Content processor initialized")


    def process_content(self, element: TrackedElement) -> ProcessedContent:
        """Process content with appropriate transformer and element processor."""
        try:
            self.logger.debug(f"Processing content of type: {element.type}")

            # Create processing context
            processing_context = ProcessingContext(
                map_id=element.parent_map_id or element.id,
                features={
                    "process_latex": True,
                    "number_headings": True,
                    "enable_cross_refs": True,
                    "process_artifacts": True,
                    "show_toc": True
                },
                # Set current topic if this is a topic
                current_topic_id=element.topic_id if element.topic_id else None,
                topic_order=[],  # Will be populated for maps
                map_metadata=element.metadata,
                topic_metadata={}
            )

            # Update element phase
            element.phase = ProcessingPhase.TRANSFORMATION
            element.state = ProcessingState.PROCESSING

            # Transform based on type
            if element.type == ElementType.MARKDOWN:
                content = self.markdown_transformer.transform_topic(
                    element=element,
                    context=processing_context
                )
            elif element.type == ElementType.DITA:
                content = self.dita_transformer.transform_topic(
                    element=element,
                    context=processing_context
                )
            else:
                raise ValueError(f"Unsupported content type: {element.type}")

            # Mark processing as complete
            element.state = ProcessingState.COMPLETED
            return content

        except Exception as e:
            self.logger.error(f"Error processing content: {str(e)}")
            element.set_error(str(e))
            return ProcessedContent(
                html=f"<div class='error'>Processing error: {str(e)}</div>",
                element_id=element.id,
                metadata=element.metadata
            )

    def create_error_element(
        self,
        error: Optional[Exception] = None,
        error_type: str = "processing_error",
        element_context: Optional[str] = None,
        element_type: ElementType = ElementType.UNKNOWN
    ) -> TrackedElement:
        """Create error element with proper tracking."""
        error_message = str(error) if error else "Unknown error"

        # Create tracked element for error
        error_element = TrackedElement(
            id=self.id_handler.generate_id(element_context or "error"),
            type=element_type,
            path=Path("error"),  # Placeholder path
            source_path=Path("error"),
            content="",  # Empty content for error

            # Set error state
            phase=ProcessingPhase.ERROR,
            state=ProcessingState.ERROR,

            # Add error metadata
            metadata={
                'error': error_message,
                'error_type': error_type,
                'context': element_context
            }
        )

        # Add error classes to HTML metadata
        error_element.html_metadata.update({
            "attributes": {'error': error_message},
            "classes": ['processing-error'],
            "context": {
                "parent_id": None,
                "level": None,
                "position": None
            }
        })

        # Set error info
        error_element.set_error(error_message)

        return error_element

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
    ) -> Dict[str, Any]:
        """Create Markdown-specific error element."""
        error_element = cast(MDElementInfo, self.create_error_element(
            error=error,
            error_type=error_type,
            element_context=element_context,
            element_type=ElementType.MARKDOWN
        ))
        return error_element.__dict__
