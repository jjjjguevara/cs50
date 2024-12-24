# app/dita/transformers/base_transformer.py

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type
from pathlib import Path

# Core managers
from ..event_manager import EventManager, EventType
from ..context_manager import ContextManager
from ..config_manager import ConfigManager
from ..key_manager import KeyManager

# Utils
from ..utils.cache import ContentCache
from ..utils.html_helpers import HTMLHelper
from ..utils.heading import HeadingHandler
from ..utils.logger import DITALogger

# Types
from ..models.types import (
    ProcessedContent,
    TrackedElement,
    ProcessingPhase,
    ProcessingState,
    ElementType,
    ProcessingMetadata,
    ProcessingContext,
    ValidationResult
)

class TransformationStrategy(ABC):
    """Base strategy for content transformation."""

    @abstractmethod
    def transform(
        self,
        element: TrackedElement,
        context: ProcessingContext,
        metadata: ProcessingMetadata,
        rules: Dict[str, Any]
    ) -> ProcessedContent:
        """Transform element according to strategy."""
        pass

class BaseTransformer(ABC):
    """Base transformer orchestrating content transformation pipeline."""

    def __init__(
        self,
        event_manager: EventManager,
        context_manager: ContextManager,
        config_manager: ConfigManager,
        key_manager: KeyManager,
        content_cache: ContentCache,
        html_helper: HTMLHelper,
        heading_handler: HeadingHandler,
        logger: Optional[DITALogger] = None
    ):
        # Core dependencies
        self.event_manager = event_manager
        self.context_manager = context_manager
        self.config_manager = config_manager
        self.key_manager = key_manager
        self.content_cache = content_cache
        self.html_helper = html_helper
        self.heading_handler = heading_handler
        self.logger = logger or logging.getLogger(__name__)

        # Initialize strategies
        self._strategies: Dict[ElementType, TransformationStrategy] = {}
        self._initialize_strategies()

        # Register for events
        self._register_event_handlers()

    def _initialize_strategies(self) -> None:
        """Initialize transformation strategies."""
        pass  # Implemented by subclasses

    def _register_event_handlers(self) -> None:
        """Register for transformation-related events."""
        self.event_manager.subscribe(
            EventType.PHASE_START,
            self._handle_phase_start
        )
        self.event_manager.subscribe(
            EventType.STATE_CHANGE,
            self._handle_state_change
        )

    def transform_content(
        self,
        element: TrackedElement,
        context: Optional[ProcessingContext] = None
    ) -> ProcessedContent:
        """
        Transform content with phase management and caching.

        Args:
            element: Element to transform
            context: Optional processing context

        Returns:
            ProcessedContent: Transformed content
        """
        try:
            # Get or create context
            if not context:
                context = self.context_manager.get_context(element.id)
                if not context:
                    context = self.context_manager.register_context(
                        content_id=element.id,
                        element_type=element.type,
                        metadata=element.metadata
                    )

            # Check cache
            cache_key = f"transform_{element.id}"
            if cached := self.content_cache.get_transformed_content(
                element.id,
                ProcessingPhase.TRANSFORMATION
            ):
                return cached

            # Start transformation phase
            self.event_manager.start_phase(
                element.id,
                ProcessingPhase.TRANSFORMATION
            )

            # Get transformation strategy
            strategy = self._get_strategy(element.type)
            if not strategy:
                raise ValueError(f"No strategy for type: {element.type}")

            # Get processing rules
            rules = self.config_manager.get_processing_rules(
                element.type,
                context
            )

            # Create processing metadata
            metadata = ProcessingMetadata(
                content_id=element.id,
                content_type=element.type,
                content_scope=context.scope
            )

            # Transform content
            transformed = strategy.transform(
                element,
                context,
                metadata,
                rules
            )

            # Cache result
            self.content_cache.register_transformed_content(
                element.id,
                ProcessingPhase.TRANSFORMATION,
                transformed
            )

            # End transformation phase
            self.event_manager.end_phase(
                element.id,
                ProcessingPhase.TRANSFORMATION
            )

            return transformed

        except Exception as e:
            self.logger.error(f"Transform error for {element.id}: {str(e)}")
            self.event_manager.update_element_state(
                element.id,
                ProcessingState.ERROR
            )
            raise

    def _get_strategy(
        self,
        element_type: ElementType
    ) -> Optional[TransformationStrategy]:
        """Get appropriate transformation strategy."""
        return self._strategies.get(element_type)

    @abstractmethod
    def enrich_content(
        self,
        content: ProcessedContent,
        context: ProcessingContext
    ) -> ProcessedContent:
        """
        Enrich transformed content with additional features.
        Implemented by specialized transformers.
        """
        pass

    def _handle_phase_start(self, **event_data: Any) -> None:
        """Handle phase start events."""
        try:
            element_id = event_data.get("element_id")
            phase = event_data.get("phase")

            if phase == ProcessingPhase.TRANSFORMATION:
                # Prepare for transformation
                self.content_cache.invalidate_pattern(
                    f"transform_{element_id}"
                )

        except Exception as e:
            self.logger.error(f"Error handling phase start: {str(e)}")

    def _handle_state_change(self, **event_data: Any) -> None:
        """Handle state change events."""
        try:
            element_id = event_data.get("element_id")
            state = event_data.get("state")

            if state == ProcessingState.ERROR:
                # Cleanup on error
                self.content_cache.invalidate_pattern(
                    f"transform_{element_id}"
                )

        except Exception as e:
            self.logger.error(f"Error handling state change: {str(e)}")
