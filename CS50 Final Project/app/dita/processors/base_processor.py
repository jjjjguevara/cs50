# app/dita/processors/base_processor.py

from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

# Event and Context Management
from ..event_manager import EventManager, EventType
from ..context_manager import ContextManager
from ..config_manager import ConfigManager

# Handlers and Utilities
from ..utils.metadata import MetadataHandler
from ..utils.id_handler import DITAIDHandler
from ..utils.heading import HeadingHandler
from ..utils.html_helpers import HTMLHelper
from ..utils.cache import ContentCache
from ..utils.logger import DITALogger

# Models and Types
from ..models.types import (
    TrackedElement,
    ProcessedContent,
    ProcessingPhase,
    ProcessingState,
    ElementType,
    ProcessingMetadata,
    ProcessingContext,
    ProcessingStateInfo,
    ContentScope
)

class BaseProcessor(ABC):
    """
    Base class for content processors implementing core processing logic.
    Handles orchestration between different components while maintaining clear
    separation of concerns.
    """

    def __init__(
        self,
        event_manager: EventManager,
        context_manager: ContextManager,
        config_manager: ConfigManager,
        metadata_handler: MetadataHandler,
        content_cache: ContentCache,
        logger: Optional[DITALogger] = None,
        id_handler: Optional[DITAIDHandler] = None,
        html_helper: Optional[HTMLHelper] = None,
        heading_handler: Optional[HeadingHandler] = None
    ):
        """
        Initialize processor with required components and handlers.

        Args:
            event_manager: Event management system
            context_manager: Context management system
            config_manager: Configuration management system
            metadata_handler: Metadata persistence handler
            content_cache: Content and processing cache
            logger: Optional logger instance
            id_handler: Optional ID generation handler
            html_helper: Optional HTML processing helper
            heading_handler: Optional heading hierarchy handler
        """
        # Core management systems
        self.event_manager = event_manager
        self.context_manager = context_manager
        self.config_manager = config_manager

        # Handlers
        self.metadata_handler = metadata_handler
        self.id_handler = id_handler or DITAIDHandler()
        self.config_manager = config_manager or ConfigManager()
        self.html_helper = html_helper or HTMLHelper()
        self.heading_handler = heading_handler or HeadingHandler(event_manager=event_manager)

        # Cache and logging
        self.content_cache = content_cache
        self.logger = logger or logging.getLogger(__name__)

        # Initialize processing state
        self._current_phase = ProcessingPhase.DISCOVERY
        self._processing_depth = 0
        self._tracked_elements: Dict[str, TrackedElement] = {}

        # Initialize feature flags from config
        self._feature_flags = self.config_manager.get_config('processing_features')

        # Register processor with event system
        self._register_event_handlers()


    def _register_event_handlers(self) -> None:
        """Register processor event handlers with event management system."""
        try:
            # Phase transitions
            self.event_manager.subscribe(
                EventType.PHASE_START,
                self._handle_phase_start
            )
            self.event_manager.subscribe(
                EventType.PHASE_END,
                self._handle_phase_end
            )

            # State changes
            self.event_manager.subscribe(
                EventType.STATE_CHANGE,
                self._handle_state_change
            )

            # Cache events
            self.event_manager.subscribe(
                EventType.CACHE_UPDATE,
                self._handle_cache_update
            )
            self.event_manager.subscribe(
                EventType.CACHE_INVALIDATE,
                self._handle_cache_invalidate
            )

        except Exception as e:
            self.logger.error(f"Error registering event handlers: {str(e)}")
            raise

    # app/dita/processors/base_processor.py

    def configure(self, config_updates: Optional[Dict[str, Any]] = None) -> None:
        try:
            # Get base configuration
            processor_config = self.config_manager.get_pipeline_config(
                pipeline_type="processor",
                context_id=None
            )

            # Update runtime configuration if needed
            if config_updates:
                self.config_manager.update_runtime_config(
                    updates=config_updates,
                    validate=True
                )

            # Get component features
            self._feature_flags = self.config_manager.get_component_features('processor')

            # Get processing rules
            self._processing_rules = {}
            for element_type in ElementType:
                rules = self.config_manager.get_processing_rules(
                    element_type=element_type,
                    context_id=None
                )
                if rules:
                    self._processing_rules[element_type] = rules

            # Configure cache
            cache_config = processor_config.get('cache', {})
            self.content_cache.max_size = cache_config.get('max_size', 1000)

            # Emit configuration event - handlers will configure themselves
            self.event_manager.emit(
                EventType.CONFIG_UPDATE,
                config=processor_config
            )

        except Exception as e:
            self.logger.error(f"Error configuring processor: {str(e)}")
            self.event_manager.emit(EventType.ERROR, error=str(e))
            raise

    def process_content(self, content: str) -> ProcessedContent:
        """
        Process raw content through the processing pipeline.

        Args:
            content: Raw content string to process

        Returns:
            ProcessedContent: Processed content with metadata
        """
        try:
            # Start processing phase
            element_id = self.id_handler.generate_id("content")
            self.event_manager.start_phase(element_id, ProcessingPhase.DISCOVERY)

            # Create processing context
            context = self.context_manager.register_context(
                content_id=element_id,
                context_type="content",
                metadata={}
            )

            # Classify content
            element_type = self.classify_element(content)

            # Create tracked element
            element = TrackedElement.from_discovery(
                path=Path("content"),
                element_type=element_type,
                id_handler=self.id_handler
            )
            element.content = content

            # Process through pipeline
            processed = self.process_element(element)

            # End processing phase
            self.event_manager.end_phase(element_id, ProcessingPhase.ASSEMBLY)

            return ProcessedContent(
                html=processed.html,
                element_id=element_id,
                metadata=processed.metadata
            )

        except Exception as e:
            self.logger.error(f"Error processing content: {str(e)}")
            self.event_manager.emit(EventType.ERROR, error=str(e))
            raise

    def process_element(self, element: TrackedElement) -> ProcessedElement:
        """
        Process a tracked element through the pipeline phases.

        Args:
            element: TrackedElement to process

        Returns:
            ProcessedElement: Processed element with HTML and metadata
        """
        try:
            # Validate element can be processed
            if not element.can_process():
                raise ValueError(f"Element {element.id} cannot be processed")

            # Start element processing
            self.event_manager.start_phase(element.id, ProcessingPhase.VALIDATION)

            # Get context
            context = self.context_manager.get_context(element.id)

            # Validate element
            if not self.validate_element(element):
                raise ValueError(f"Element {element.id} validation failed")

            # Process through phases
            element = self._process_discovery(element, context)
            element = self._process_validation(element, context)
            element = self._process_transformation(element, context)
            element = self._process_enrichment(element, context)
            element = self._process_assembly(element, context)

            # Update final state
            self.event_manager.update_element_state(
                element,
                ProcessingState.COMPLETED
            )

            return element

        except Exception as e:
            self.logger.error(f"Error processing element {element.id}: {str(e)}")
            element.set_error(str(e))
            self.event_manager.emit(EventType.ERROR, error=str(e))
            raise

    def classify_element(self, element: Any) -> ElementType:
        """Determine element type with caching."""
        try:
            # Generate cache key
            cache_key = f"type_{hash(str(element))}"

            # Check cache
            if cached_type := self.content_cache.get(cache_key):
                return cached_type

            # Determine type
            element_type = self._determine_element_type(element)

            # Cache result
            self.content_cache.set(
                cache_key,
                element_type,
                ElementType.UNKNOWN,
                ProcessingPhase.DISCOVERY
            )

            return element_type

        except Exception as e:
            self.logger.error(f"Error classifying element: {str(e)}")
            return ElementType.UNKNOWN

    def determine_context(self, element: Any) -> ProcessingContext:
        """Determine processing context for element."""
        try:
            # Get element ID
            element_id = getattr(element, 'id', self.id_handler.generate_id("context"))

            # Get existing context or create new
            if context := self.context_manager.get_context(element_id):
                return context

            # Create new context
            context = ProcessingContext(
                context_id=element_id,
                element_id=element_id,
                element_type=self.classify_element(element),
                state_info=ProcessingStateInfo(
                    phase=ProcessingPhase.DISCOVERY,
                    state=ProcessingState.PENDING,
                    element_id=element_id
                ),
                navigation=self.heading_handler.get_navigation_context(),
                scope=ContentScope.LOCAL
            )

            # Register context
            self.context_manager.register_context(
                element_id,
                context_type="element",
                metadata={}
            )

            return context

        except Exception as e:
            self.logger.error(f"Error determining context: {str(e)}")
            raise
