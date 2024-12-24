# app/dita/processors/dita_processor.py

from pathlib import Path
from typing import Dict, List, Optional, Any, Type

# Base processor and strategy
from .base_processor import BaseProcessor

# Core managers
from ..event_manager import EventManager
from ..context_manager import ContextManager
from ..config_manager import ConfigManager
from ..metadata.metadata_manager import MetadataManager
from ..key_manager import KeyManager

# Utils
from ..utils.cache import ContentCache
from ..utils.id_handler import DITAIDHandler
from ..utils.html_helpers import HTMLHelper
from ..utils.heading import HeadingHandler
from ..utils.logger import DITALogger

# Types
from ..models.types import (
    TrackedElement,
    ProcessedContent,
    ProcessingPhase,
    ProcessingMetadata,
    ProcessingContext,
    ElementType,
    DITAElementType,
    DITAElementInfo,
    KeyDefinition
)

class DITAProcessor(BaseProcessor):
    """Processor for DITA-specific content."""

    def __init__(
        self,
        event_manager: EventManager,
        context_manager: ContextManager,
        config_manager: ConfigManager,
        metadata_manager: MetadataManager,
        key_manager: KeyManager,  # Additional for DITA key resolution
        content_cache: ContentCache,
        logger: Optional[DITALogger] = None,
        id_handler: Optional[DITAIDHandler] = None,
        html_helper: Optional[HTMLHelper] = None,
        heading_handler: Optional[HeadingHandler] = None
    ):
        super().__init__(
            event_manager=event_manager,
            context_manager=context_manager,
            config_manager=config_manager,
            metadata_manager=metadata_manager,
            content_cache=content_cache,
            logger=logger,
            id_handler=id_handler,
            html_helper=html_helper,
            heading_handler=heading_handler
        )

        # DITA-specific initialization
        self.key_manager = key_manager

        # Initialize DITA element strategies
        self._initialize_strategies()


    class DITATopicStrategy(BaseProcessor.ProcessingStrategy):
            """Strategy for DITA topic processing."""

            def __init__(self, processor: 'DITAProcessor'):
                self.processor = processor

            def process(
                self,
                element: TrackedElement,
                context: ProcessingContext,
                metadata: ProcessingMetadata,
                rules: Dict[str, Any]
            ) -> ProcessedContent:
                """Process DITA topic elements."""
                # Handle topic specialization
                if element.type == ElementType.TOPIC:
                    rules = self.processor._get_specialization_rules(rules)

                # Create topic metadata
                topic_metadata = {
                    "type": element.type.value,
                    "specialization": rules.get("specialization"),
                    "context": context.to_dict(),
                    "key_references": metadata.references
                }

                return ProcessedContent(
                    element_id=element.id,
                    html="",  # No transformation here
                    metadata=topic_metadata
                )

    class DITAMapStrategy(BaseProcessor.ProcessingStrategy):
        """Strategy for DITA map processing."""

        def __init__(self, processor: 'DITAProcessor'):
            self.processor = processor

        def process(
            self,
            element: TrackedElement,
            context: ProcessingContext,
            metadata: ProcessingMetadata,
            rules: Dict[str, Any]
        ) -> ProcessedContent:
            """Process DITA map elements."""
            # Handle map-specific metadata
            map_metadata = {
                "type": "map",
                "topics": element.topics,
                "key_references": metadata.references,
                "context": context.to_dict()
            }

            return ProcessedContent(
                element_id=element.id,
                html="",  # No transformation here
                metadata=map_metadata
            )

    def _initialize_strategies(self) -> None:
        """Initialize DITA-specific processing strategies."""
        self._strategies[ElementType.DITA] = self.DITATopicStrategy(self)
        self._strategies[ElementType.DITAMAP] = self.DITAMapStrategy(self)

    def process_map(self, map_path: Path) -> ProcessedContent:
        """Process a DITA map."""
        # Create tracked element for map
        element = TrackedElement.create_map(
            path=map_path,
            title="",  # Will be extracted during processing
            id_handler=self.id_handler
        )

        # Process using base processor's element processing
        return self.process_element(element)

    def process_topic(self, topic_path: Path) -> ProcessedContent:
        """Process a DITA topic."""
        # Create tracked element for topic
        element = TrackedElement.from_discovery(
            path=topic_path,
            element_type=ElementType.DITA,
            id_handler=self.id_handler
        )

        # Process using base processor's element processing
        return self.process_element(element)

    def _get_specialization_rules(self, base_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Get specialized processing rules from base rules."""
        try:
            specialized_rules = base_rules.copy()

            # Check for specialization in rules
            if specialization_info := base_rules.get("specialization"):
                specialization_type = specialization_info.get("type")
                if specialization_type:
                    # Get DITA element type
                    dita_type = DITAElementType(specialization_type)

                    # Get specialized rules
                    specialized_rules.update(
                        self.config_manager.get_dita_element_rules(dita_type)
                    )

            return specialized_rules

        except Exception as e:
            self.logger.error(f"Error getting specialization rules: {str(e)}")
            return base_rules

    def _process_keyref(self, element: TrackedElement, context: ProcessingContext) -> None:
        """Process key references in element."""
        try:
            # Get key definitions from context
            if key_refs := element.metadata.get("key_references"):
                # Get context map ID (either from parent or current if it's a map)
                context_map = element.parent_map_id or (
                    element.id if element.type == ElementType.DITAMAP else None
                )

                if not context_map:
                    self.logger.warning(f"No context map found for key resolution: {element.id}")
                    return

                # Resolve each key reference
                for key in key_refs:
                    resolved = self.key_manager.resolve_key(
                        key=key,
                        context_map=context_map
                    )
                    if resolved:
                        # Store resolved href from KeyDefinition
                        context.metadata_refs[key] = resolved.href if resolved.href else ""

        except Exception as e:
            self.logger.error(f"Error processing keyrefs: {str(e)}")
