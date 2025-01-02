# app/dita/transformers/dita_transformer.py

from typing import (
    Dict,
    List,
    Optional,
    Any,
    Type,
    Set,
    Tuple,
    Union,
    Generator
)
from bs4 import Tag
from pathlib import Path
import logging

# Base classes
from .base_transformer import BaseTransformer, TransformStrategy

# Core managers
from ..event_manager import EventManager, EventType
from ..context_manager import ContextManager
from ..config.config_manager import ConfigManager
from ..key_manager import KeyManager

# Utils
from ..utils.cache import ContentCache, CacheEntryType
from ..utils.html_helpers import HTMLHelper
from ..utils.heading import HeadingHandler
from ..utils.id_handler import DITAIDHandler, IDType
from ..utils.logger import DITALogger

# Types
from ..models.types import (
    ProcessedContent,
    TrackedElement,
    ProcessingPhase,
    ProcessingState,
    ElementType,
    DITAElementType,
    ProcessingMetadata,
    ProcessingContext,
    ValidationResult,
    ValidationMessage,
    ValidationSeverity,
    ContentScope,
    ProcessingRuleType,
    ProcessingStateInfo,
    KeyDefinition
)

class DITATransformStrategy(TransformStrategy):
    """Base strategy for DITA-specific transformations."""

    def __init__(self, transformer: 'DITATransformer'):
        self.transformer = transformer
        self.logger = transformer.logger


class DITATopicStrategy(DITATransformStrategy):
    """Strategy for DITA topic transformation."""

    def can_transform(
        self,
        element: TrackedElement,
        context: ProcessingContext
    ) -> bool:
        return True

    def transform(
        self,
        element: TrackedElement,
        context: ProcessingContext,
        metadata: ProcessingMetadata,
        config: Dict[str, Any]
    ) -> ProcessedContent:
        """Transform general DITA element."""
        try:
            # Use the new rule resolution system
            resolved_rule = self.transformer.config_manager.resolve_rule(
                element_type=element.type,
                rule_type=ProcessingRuleType.ELEMENT,
                context=context
            )

            rules = resolved_rule or {}

            # Create element with proper classes
            classes = ["dita-content"]
            if element.type == ElementType.DITAMAP:
                classes.append("dita-map")
            elif element.type == ElementType.DITA:
                classes.append("dita-topic")
            else:
                classes.append("dita-unknown")

            # Create element
            transformed = self.transformer.html_helper.create_element(
                tag=rules.get("html_tag", "div"),
                attrs={
                    "class": classes,
                    "id": element.id,
                    "data-type": element.type.value,
                    **rules.get("attributes", {})
                },
                content=element.content
            )

            return ProcessedContent(
                element_id=element.id,
                html=str(transformed),
                metadata=metadata.transient_attributes,
                element_type=element.type
            )

        except Exception as e:
            self.logger.error(f"Error transforming element: {str(e)}")
            raise

    def validate(
        self,
        element: TrackedElement,
        context: ProcessingContext
    ) -> ValidationResult:
        """Validate DITA topic."""
        messages = []

        # Validate required elements
        if not element.content:
            messages.append(
                ValidationMessage(
                    path=element.id,
                    message="Empty topic content",
                    severity=ValidationSeverity.ERROR,
                    code="empty_topic"
                )
            )

        # Validate specialization if present
        if specialization := self._get_specialization_type(element, context):
            if not self.transformer.specialization_rules.get(specialization.value):
                messages.append(
                    ValidationMessage(
                        path=element.id,
                        message=f"Unknown specialization: {specialization.value}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_specialization"
                    )
                )

        return ValidationResult(
            is_valid=not any(
                msg.severity == ValidationSeverity.ERROR
                for msg in messages
            ),
            messages=messages
        )

    def _get_specialization_type(
        self,
        element: TrackedElement,
        context: ProcessingContext
    ) -> Optional[DITAElementType]:
        """Get specialization type for element."""
        if element.id in self.transformer._specialized_elements:
            return self.transformer._specialized_elements[element.id]

        specialization = context.metadata_refs.get("specialization")
        if specialization:
            try:
                return DITAElementType(specialization)
            except ValueError:
                pass

        return None

    def _resolve_keys(
            self,
            element: TrackedElement,
            context: ProcessingContext
        ) -> None:
            """Resolve keys using transformer's key manager."""
            self.transformer._resolve_keys(element, context)

    def _transform_children(
        self,
        element: TrackedElement,
        context: ProcessingContext,
        metadata: ProcessingMetadata
    ) -> List[Union[Tag, str]]:
        """Transform child elements."""
        children = []
        # Parse element content into child elements
        child_elements = self.transformer.content_cache.get(
            f"children_{element.id}",
            entry_type=CacheEntryType.CONTENT
        )

        if not child_elements:
            # Parse content into child elements - this should be done by processor
            # and stored in cache, but for now we'll return empty list
            return []

        # Transform each child element
        for child_element in child_elements:
            if isinstance(child_element, TrackedElement):
                processed = self.transformer.transform_content(child_element, context)
                if processed and processed.html:
                    children.append(processed.html)
        return children

class DITAMapStrategy(DITATransformStrategy):
    """Strategy for DITA map transformation."""

    def can_transform(
        self,
        element: TrackedElement,
        context: ProcessingContext
    ) -> bool:
        return element.type == ElementType.DITAMAP

    def transform(
        self,
        element: TrackedElement,
        context: ProcessingContext,
        metadata: ProcessingMetadata,
        config: Dict[str, Any]
    ) -> ProcessedContent:
        """Transform DITA map."""
        try:
            # Update current map context
            self.transformer._current_map_id = element.id

            # Use the new rule resolution system
            resolved_rule = self.transformer.config_manager.resolve_rule(
                element_type=ElementType.MAP,
                rule_type=ProcessingRuleType.ELEMENT,
                context=context
            )

            rules = resolved_rule or {}

            # Create map container
            container = self.transformer.html_helper.create_element(
                tag=rules.get("html_tag", "div"),
                attrs={
                    "class": rules.get("default_classes", ["dita-map"]),
                    "id": element.id,
                    **rules.get("attributes", {})
                }
            )

            # Process topic refs
            topic_refs = self._process_topic_refs(element, context)

            # Create ToC if enabled
            if context.features.get("show_toc", True):
                toc = self._create_toc(element.id, context)
                children: List[Union[Tag, str]] = [toc]
                children.extend(topic_refs)
                container = self.transformer.html_helper.create_container(
                    tag=rules["html_tag"],
                    children=children,
                    attrs=container.attrs
                )
            else:
                container = self.transformer.html_helper.create_container(
                    tag=rules["html_tag"],
                    children=topic_refs,
                    attrs=container.attrs
                )

            return ProcessedContent(
                element_id=element.id,
                html=str(container),
                metadata=metadata.transient_attributes
            )

        except Exception as e:
            self.transformer.logger.error(f"Error transforming map: {str(e)}")
            raise

    def _create_toc(
        self,
        map_id: str,
        context: ProcessingContext
    ) -> str:
        """Create table of contents."""
        return self.transformer.html_helper.generate_toc(
            self.transformer.heading_handler.get_topic_headings(map_id)
        )

    def validate(
        self,
        element: TrackedElement,
        context: ProcessingContext
    ) -> ValidationResult:
        """Validate DITA map."""
        messages = []

        # Validate topic refs
        if not element.topics:
            messages.append(
                ValidationMessage(
                    path=element.id,
                    message="Map contains no topics",
                    severity=ValidationSeverity.WARNING,
                    code="empty_map"
                )
            )

        # Validate title
        if not element.title:
            messages.append(
                ValidationMessage(
                    path=element.id,
                    message="Map missing title",
                    severity=ValidationSeverity.ERROR,
                    code="missing_title"
                )
            )

        return ValidationResult(
            is_valid=not any(
                msg.severity == ValidationSeverity.ERROR
                for msg in messages
            ),
            messages=messages
        )

    def _process_topic_refs(
        self,
        element: TrackedElement,
        context: ProcessingContext
    ) -> List[Union[Tag, str]]:  # Updated return type
        """Process topic references in map."""
        topic_refs = []
        for topic_id in element.topics:
            topic_context = self.transformer.context_manager.get_context(topic_id)
            if not topic_context:
                continue
            processed = self.transformer.transform_content(
                TrackedElement.from_discovery(
                    path=Path(topic_id),
                    element_type=ElementType.TOPIC,
                    id_handler=self.transformer.id_handler
                ),
                topic_context
            )
            if processed and processed.html:
                topic_refs.append(processed.html)
        return topic_refs


class DITAElementStrategy(DITATransformStrategy):
    """Strategy for general DITA element transformation."""

    def can_transform(
        self,
        element: TrackedElement,
        context: ProcessingContext
    ) -> bool:
        return True  # Default strategy for any DITA element

    def transform(
        self,
        element: TrackedElement,
        context: ProcessingContext,
        metadata: ProcessingMetadata,
        config: Dict[str, Any]
    ) -> ProcessedContent:
        """Transform general DITA element."""
        try:
            # Use the new rule resolution system
            resolved_rule = self.transformer.config_manager.resolve_rule(
                element_type=element.type,
                rule_type=ProcessingRuleType.ELEMENT,
                context=context
            )

            rules = resolved_rule or {}

            # Create element
            transformed = self.transformer.html_helper.create_element(
                tag=rules.get("html_tag", "div"),
                attrs={
                    "class": rules.get("default_classes", ["dita-element"]),
                    "id": element.id,
                    **rules.get("attributes", {})
                },
                content=element.content
            )

            return ProcessedContent(
                element_id=element.id,
                html=str(transformed),
                metadata=metadata.transient_attributes
            )

        except Exception as e:
            self.transformer.logger.error(f"Error transforming element: {str(e)}")
            raise

    def validate(
        self,
        element: TrackedElement,
        context: ProcessingContext
    ) -> ValidationResult:
        """Validate DITA element."""
        return ValidationResult(is_valid=True, messages=[])

class DITATransformer(BaseTransformer):
    """
    DITA-specific transformer implementation.
    Handles DITA content transformation with specialization support.
    """

    def __init__(
        self,
        event_manager: EventManager,
        context_manager: ContextManager,
        config_manager: ConfigManager,
        key_manager: KeyManager,
        content_cache: ContentCache,
        html_helper: HTMLHelper,
        heading_handler: HeadingHandler,
        id_handler: DITAIDHandler,
        logger: Optional[DITALogger] = None,
        specialization_rules: Optional[Dict[str, Any]] = None
    ):
        """Initialize DITA transformer."""
        super().__init__(
            event_manager=event_manager,
            context_manager=context_manager,
            config_manager=config_manager,
            key_manager=key_manager,
            content_cache=content_cache,
            html_helper=html_helper,
            heading_handler=heading_handler,
            id_handler=id_handler,
            logger=logger
        )

        # DITA-specific configuration
        self.specialization_rules = specialization_rules or {}

        # Key resolution cache
        self._key_resolution_cache: Dict[str, KeyDefinition] = {}

        # Specialization tracking
        self._specialized_elements: Dict[str, DITAElementType] = {}

        # Transform state
        self._current_map_id: Optional[str] = None
        self._topic_hierarchy: Dict[str, List[str]] = {}

        # Initialize DITA-specific strategies
        self._initialize_strategies()  # Changed to use base class method name

    def _initialize_strategies(self) -> None:  # Changed from _initialize_dita_strategies
        """Initialize DITA-specific transformation strategies."""
        self._strategies = {
            ElementType.DITA: [DITAElementStrategy(self)],      # Add base DITA type
            ElementType.TOPIC: [DITATopicStrategy(self)],
            ElementType.DITAMAP: [DITAMapStrategy(self)],
            ElementType.UNKNOWN: [DITAElementStrategy(self)]
        }

    def register_strategy(
        self,
        element_type: ElementType,
        strategy: TransformStrategy
    ) -> None:
        """Register a DITA transformation strategy."""
        if element_type not in self._strategies:
            self._strategies[element_type] = []
        self._strategies[element_type].append(strategy)

    def transform_content(
        self,
        element: TrackedElement,
        context: Optional[ProcessingContext] = None
    ) -> ProcessedContent:
        try:
            # Get or create context
            if not context:
                ctx = self.context_manager.get_context(element.id)
                if not ctx:
                    ctx = self.context_manager.register_context(
                        content_id=element.id,
                        element_type=element.type,
                        metadata=element.metadata
                    )
            else:
                ctx = context

            if not ctx:
                raise ValueError(f"Could not create context for {element.id}")

            # Validate element
            validation_result = self._validate_element(element, ctx)
            if not validation_result.is_valid:
                raise ValueError(
                    f"Validation failed for {element.id}: "
                    f"{validation_result.messages[0].message}"
                )

            # Get appropriate strategies
            strategies = self._get_strategies(element.type)
            if not strategies:
                raise ValueError(f"No strategy found for {element.type}")

            # Find suitable strategy
            strategy = next(
                (s for s in strategies if s.can_transform(element, ctx)),
                None
            )
            if not strategy:
                raise ValueError(f"No suitable strategy for {element.id}")

            # Create processing metadata
            metadata = ProcessingMetadata(
                content_id=element.id,
                content_type=element.type,
                content_scope=ctx.scope
            )

            # Get transformation config
            config = self.config_manager.get_processing_rules(
                element.type,
                ctx
            )

            # Transform and enrich content
            transformed = strategy.transform(element, ctx, metadata, config)
            return self.enrich_content(transformed, ctx)

        except Exception as e:
            self.logger.error(f"Error transforming DITA content: {str(e)}")
            raise

    def _validate_element(
        self,
        element: TrackedElement,
        context: ProcessingContext
    ) -> ValidationResult:
        """Validate element before transformation."""
        try:
            strategies = self._get_strategies(element.type)
            for strategy in strategies:
                if strategy.can_transform(element, context):
                    return strategy.validate(element, context)
            return ValidationResult(
                is_valid=True,
                messages=[]
            )

        except Exception as e:
            self.logger.error(f"Error validating element: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[
                    ValidationMessage(
                        path=element.id,
                        message=str(e),
                        severity=ValidationSeverity.ERROR,
                        code="validation_error"
                    )
                ]
            )

    def _resolve_keys(
        self,
        element: TrackedElement,
        context: ProcessingContext
    ) -> None:
        """Resolve key references in element."""
        try:
            if key_refs := element.metadata.get("key_references"):
                current_map = self._current_map_id
                if not current_map:
                    self.logger.warning(f"No current map ID for key resolution: {element.id}")
                    return

                for key in key_refs:
                    if key not in self._key_resolution_cache:
                        resolved = self.key_manager.resolve_key(
                            key=key,
                            context_map=current_map
                        )
                        if resolved:
                            self._key_resolution_cache[key] = resolved
                            context.metadata_refs[key] = resolved.href or ""

        except Exception as e:
            self.logger.error(f"Error resolving keys: {str(e)}")
