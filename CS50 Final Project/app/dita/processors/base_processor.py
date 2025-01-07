# app/dita/processors/base_processor.py

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Type
import logging

# Core managers
from ..event_manager import EventManager, EventType
from ..context_manager import ContextManager
from ..config.config_manager import ConfigManager
from ..metadata.metadata_manager import MetadataManager

# Utils
from ..utils.cache import ContentCache
from ..utils.id_handler import DITAIDHandler
from ..utils.html_helpers import HTMLHelper
from ..utils.heading import HeadingHandler
from ..utils.logger import DITALogger

# DTD imports
from ..dtd.dtd_mapper import DTDSchemaMapper
from ..dtd.dtd_validator import DTDValidator
from ..dtd.dtd_resolver import DTDResolver

# Types
from ..models.types import (
    ContentElement,
    ProcessedContent,
    ProcessingPhase,
    ProcessingState,
    ElementType,
    ProcessingMetadata,
    ProcessingContext,
    NavigationContext,
    ProcessingStatus,
    ContentScope
)

class BaseProcessor(ABC):
    """Base processor for content transformation pipeline."""

    class ProcessingStrategy(ABC):
        @abstractmethod
        def process(
            self,
            element: ContentElement,
            context: ProcessingContext,
            metadata: ProcessingMetadata,
            rules: Dict[str, Any]
        ) -> ProcessedContent:
            """Process element according to its type and rules."""
            pass

    class MapProcessingStrategy(ProcessingStrategy):
        def process(
            self,
            element: ContentElement,
            context: ProcessingContext,
            metadata: ProcessingMetadata,
            rules: Dict[str, Any]
        ) -> ProcessedContent:
            """Process map elements."""
            return ProcessedContent(
                element_id=element.id,
                html="",  # No transformation here
                metadata=metadata.transient_attributes,
                element_type=element.type
            )

    class TopicProcessingStrategy(ProcessingStrategy):
        def process(
            self,
            element: ContentElement,
            context: ProcessingContext,
            metadata: ProcessingMetadata,
            rules: Dict[str, Any]
        ) -> ProcessedContent:
            """Process topic elements."""
            return ProcessedContent(
                element_id=element.id,
                html="",  # No transformation here
                metadata=metadata.transient_attributes,
                element_type=element.type
            )

    class DefaultProcessingStrategy(ProcessingStrategy):
        def process(
            self,
            element: ContentElement,
            context: ProcessingContext,
            metadata: ProcessingMetadata,
            rules: Dict[str, Any]
        ) -> ProcessedContent:
            """Process unknown elements with default handling."""
            return ProcessedContent(
                element_id=element.id,
                html="",  # No transformation here
                metadata=metadata.transient_attributes,
                element_type=element.type
            )


    def __init__(
        self,
        event_manager: EventManager,
        context_manager: ContextManager,
        config_manager: ConfigManager,
        metadata_manager: MetadataManager,
        content_cache: ContentCache,
        logger: Optional[DITALogger] = None,
        id_handler: Optional[DITAIDHandler] = None,
        html_helper: Optional[HTMLHelper] = None,
        heading_handler: Optional[HeadingHandler] = None,
        dtd_path: Optional[Path] = None,
        dtd_mapper: Optional[DTDSchemaMapper] = None,
        dtd_validator: Optional[DTDValidator] = None
    ):
        # Core managers
        self.event_manager = event_manager
        self.context_manager = context_manager
        self.config_manager = config_manager
        self.metadata_manager = metadata_manager
        self.content_cache = content_cache

        # Utilities
        self.logger = logger or logging.getLogger(__name__)
        self.id_handler = id_handler or DITAIDHandler()
        self.html_helper = html_helper or HTMLHelper()
        self.heading_handler = heading_handler or HeadingHandler(event_manager=self.event_manager)

        # DTD processing components
        self.dtd_path = dtd_path
        self._dtd_mapper = dtd_mapper
        self._dtd_validator = dtd_validator
        # Only initialize DTD components if path is provided
        if self.dtd_path:
            self._init_dtd_components()
        else:
            self.logger.debug("No DTD path provided - DTD validation disabled")

        # Initialize state
        self._current_phase = ProcessingPhase.DISCOVERY

        # Initialize strategy registry
        self._strategies: Dict[ElementType, BaseProcessor.ProcessingStrategy] = {
            ElementType.DITAMAP: self.MapProcessingStrategy(),
            ElementType.TOPIC: self.TopicProcessingStrategy(),
            ElementType.UNKNOWN: self.DefaultProcessingStrategy()
        }

    def _init_dtd_components(self) -> None:
        """Initialize DTD processing components if not provided."""
        try:
            # Validate dtd_path exists
            if not self.dtd_path:
                raise ValueError("DTD path is required for DTD processing")

            # Initialize DTD resolver if needed
            if not hasattr(self, '_dtd_resolver'):
                self._dtd_resolver = DTDResolver(
                    base_path=self.dtd_path,  # Now we know it's not None
                    logger=self.logger if isinstance(self.logger, DITALogger) else None
                )

            # Initialize mapper if not provided
            if not self._dtd_mapper:
                self._dtd_mapper = DTDSchemaMapper(
                    logger=self.logger if isinstance(self.logger, DITALogger) else None
                )

            # Initialize validator if not provided
            if not self._dtd_validator:
                self._dtd_validator = DTDValidator(
                    resolver=self._dtd_resolver,
                    dtd_mapper=self._dtd_mapper,  # Use dtd_mapper directly
                    validation_manager=self.config_manager.validation_manager,
                    logger=self.logger if isinstance(self.logger, DITALogger) else None
                )

        except Exception as e:
            self.logger.error(f"Error initializing DTD components: {str(e)}")
            raise

    def _get_element_dtd(self, element_type: ElementType) -> Optional[Path]:
        """Get appropriate DTD file for element type."""
        if not self.dtd_path:
            return None

        # Map element types to DTD files
        dtd_mapping = {
            ElementType.DITAMAP: "map.dtd",
            ElementType.DITA: "topic.dtd",
            ElementType.TOPIC: "topic.dtd",
            ElementType.MAP: "map.dtd",
            # Add other mappings as needed
        }

        if dtd_file := dtd_mapping.get(element_type):
            dtd_path = self.dtd_path / dtd_file
            if dtd_path.exists():
                return dtd_path

        return None

    def register_strategy(
            self,
            element_type: ElementType,
            strategy: 'BaseProcessor.ProcessingStrategy'
        ) -> None:
            """Register a new processing strategy."""
            self._strategies[element_type] = strategy

    def _get_processing_strategy(
        self,
        element_type: ElementType
    ) -> 'BaseProcessor.ProcessingStrategy':
        """Get appropriate processing strategy for element type."""
        try:
            return self._strategies.get(
                element_type,
                self._strategies[ElementType.UNKNOWN]
            )
        except KeyError:
            raise ValueError(f"No strategy registered for element type: {element_type}")

    def process_element(self, element: ContentElement) -> ProcessedContent:
        """Process an element through the pipeline phases."""
        try:
            # Existing DTD validation
            if self.dtd_path and self._dtd_validator:
                element_dtd = self._get_element_dtd(element.type)
                if element_dtd:
                    validation_result = self._dtd_validator.validate_content(
                        content=element.content,
                        dtd_path=element_dtd,
                        context=None
                    )
                    # Add validation result to element metadata
                    element.metadata['dtd_validation'] = {
                        'is_valid': validation_result.is_valid,
                        'messages': [msg.message for msg in validation_result.messages]
                    }

            # Continue with normal processing
            strategy = self._get_processing_strategy(element.type)
            if not strategy:
                raise ValueError(f"No strategy found for element type: {element.type}")

            # Get or create element context
            context = self.context_manager.get_context(element.id)
            if not context:
                context = self.context_manager.register_context(
                    content_id=element.id,
                    element_type=element.type,
                    metadata=element.metadata
                )
                if not context:
                    raise ValueError(f"Failed to create context for element {element.id}")

            # Get processing rules
            rules = self.config_manager.get_processing_rules(element.type)

            # Process raw metadata
            raw_metadata = self.metadata_manager.process_metadata(
                element=element,
                context=context,
                phase=self._current_phase
            )

            # Convert to ProcessingMetadata instance
            metadata = ProcessingMetadata(
                content_id=element.id,
                content_type=element.type,
                content_scope=ContentScope.LOCAL,
            )
            metadata.transient_attributes = raw_metadata  # Store raw metadata

            # Process using appropriate strategy
            return strategy.process(element, context, metadata, rules)

        except Exception as e:
            self.logger.error(f"Error processing element {element.id}: {str(e)}")
            raise


    @abstractmethod
    def process_map(self, map_path: Path) -> ProcessedContent:
        """Process a DITA map or collection of topics."""
        pass

    @abstractmethod
    def process_topic(self, topic_path: Path) -> ProcessedContent:
        """Process a topic file (DITA or Markdown)."""
        pass
