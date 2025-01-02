# app/dita/transformers/markdown_transformer.py

from typing import Dict, List, Optional, Any, Union, Sequence
from pathlib import Path
import logging
from bs4 import BeautifulSoup, Tag
import re

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
from ..utils.heading import HeadingHandler, HeadingMetadata
from ..utils.id_handler import DITAIDHandler, IDType
from ..utils.logger import DITALogger

# Types
from ..models.types import (
    ProcessedContent,
    TrackedElement,
    ProcessingPhase,
    ProcessingState,
    ElementType,
    MDElementType,
    ProcessingMetadata,
    ProcessingContext,
    ValidationResult,
    ValidationMessage,
    ValidationSeverity,
    ContentScope,
    ProcessingStateInfo
)

class MarkdownTransformStrategy(TransformStrategy):
    """Base strategy for Markdown-specific transformations."""

    def __init__(self, transformer: 'MarkdownTransformer'):
        self.transformer = transformer

class MarkdownTopicStrategy(MarkdownTransformStrategy):
    """Strategy for Markdown topic transformation."""

    def can_transform(
        self,
        element: TrackedElement,
        context: ProcessingContext
    ) -> bool:
        return element.type == ElementType.MARKDOWN

    def transform(
        self,
        element: TrackedElement,
        context: ProcessingContext,
        metadata: ProcessingMetadata,
        config: Dict[str, Any]
    ) -> ProcessedContent:
        """Transform Markdown topic."""
        try:
            # Extract frontmatter if present
            frontmatter = self.transformer._extract_frontmatter(element)
            self.transformer._current_frontmatter = frontmatter

            # Create topic container
            container = self.transformer.html_helper.create_element(
                tag="article",
                attrs={
                    "class": ["markdown-topic"],
                    "id": element.id,
                    "data-topic-type": "markdown"
                }
            )

            # Transform content
            children = self._transform_content(element, context, metadata)
            container = self.transformer.html_helper.create_container(
                tag="article",
                children=children,
                attrs=container.attrs
            )

            return ProcessedContent(
                element_id=element.id,
                html=str(container),
                metadata={
                    **metadata.transient_attributes,
                    "frontmatter": frontmatter
                }
            )

        except Exception as e:
            self.transformer.logger.error(f"Error transforming markdown topic: {str(e)}")
            raise

    def validate(
        self,
        element: TrackedElement,
        context: ProcessingContext
    ) -> ValidationResult:
        """Validate Markdown topic."""
        messages = []

        # Validate content presence
        if not element.content:
            messages.append(
                ValidationMessage(
                    path=element.id,
                    message="Empty topic content",
                    severity=ValidationSeverity.ERROR,
                    code="empty_topic"
                )
            )

        # Validate frontmatter if required by context
        if context.features.get("require_frontmatter", False):
            frontmatter = self.transformer._extract_frontmatter(element)
            if not frontmatter:
                messages.append(
                    ValidationMessage(
                        path=element.id,
                        message="Missing required frontmatter",
                        severity=ValidationSeverity.ERROR,
                        code="missing_frontmatter"
                    )
                )

        return ValidationResult(
            is_valid=not any(
                msg.severity == ValidationSeverity.ERROR
                for msg in messages
            ),
            messages=messages
        )

    def _transform_content(
        self,
        element: TrackedElement,
        context: ProcessingContext,
        metadata: ProcessingMetadata
    ) -> List[Union[Tag, str]]:
        """Transform Markdown content."""
        children: List[Union[Tag, str]] = []

        # Get cached parsed content
        parsed_content = self.transformer.content_cache.get(
            f"parsed_{element.id}",
            entry_type=CacheEntryType.CONTENT
        )

        if not parsed_content:
            return children

        # Transform each block
        for block in parsed_content:
            if isinstance(block, TrackedElement):
                processed = self.transformer.transform_content(block, context)
                if processed and processed.html:
                    children.append(processed.html)

        return children

class MarkdownCalloutStrategy(MarkdownTransformStrategy):
    """Strategy for Markdown callout blocks."""

    def can_transform(
        self,
        element: TrackedElement,
        context: ProcessingContext
    ) -> bool:
        return element.type == ElementType.NOTE

    def transform(
        self,
        element: TrackedElement,
        context: ProcessingContext,
        metadata: ProcessingMetadata,
        config: Dict[str, Any]
    ) -> ProcessedContent:
        """Transform Markdown callout."""
        try:
            # Get callout type
            callout_type = element.metadata.get("callout_type", "note")

            # Create callout container
            container = self.transformer.html_helper.create_element(
                tag="div",
                attrs={
                    "class": [f"callout callout-{callout_type}"],
                    "id": element.id,
                    "role": "note"
                },
                content=element.content
            )

            return ProcessedContent(
                element_id=element.id,
                html=str(container),
                metadata=metadata.transient_attributes
            )

        except Exception as e:
            self.transformer.logger.error(f"Error transforming callout: {str(e)}")
            raise

    def validate(
        self,
        element: TrackedElement,
        context: ProcessingContext
    ) -> ValidationResult:
        """Validate Markdown callout."""
        messages = []

        # Validate callout type
        callout_type = element.metadata.get("callout_type")
        if callout_type not in {"note", "warning", "tip", "important", "caution"}:
            messages.append(
                ValidationMessage(
                    path=element.id,
                    message=f"Invalid callout type: {callout_type}",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_callout_type"
                )
            )

        return ValidationResult(
            is_valid=not any(
                msg.severity == ValidationSeverity.ERROR
                for msg in messages
            ),
            messages=messages
        )


class MarkdownTransformer(BaseTransformer):
    """
    Markdown-specific transformer implementation.
    Handles Markdown content transformation with frontmatter support.
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
        custom_syntax_rules: Optional[Dict[str, Any]] = None
    ):
        """Initialize Markdown transformer."""
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

        # Markdown-specific configuration
        self.custom_syntax_rules = custom_syntax_rules or {}

        # Frontmatter tracking
        self._frontmatter_cache: Dict[str, Dict[str, Any]] = {}

        # Custom syntax tracking
        self._custom_elements: Dict[str, MDElementType] = {}

        # Transform state
        self._current_topic_id: Optional[str] = None
        self._current_frontmatter: Optional[Dict[str, Any]] = None

        # Initialize Markdown-specific strategies
        self._initialize_strategies()  # Changed from _initialize_markdown_strategies

    def register_strategy(
        self,
        element_type: ElementType,
        strategy: TransformStrategy
    ) -> None:
        """Register a Markdown transformation strategy."""
        if element_type not in self._strategies:
            self._strategies[element_type] = []
        self._strategies[element_type].append(strategy)

    def _initialize_strategies(self) -> None:  # Changed method name
        """Initialize Markdown-specific transformation strategies."""
        self._strategies[ElementType.MARKDOWN] = [MarkdownTopicStrategy(self)]
        self._strategies[ElementType.NOTE] = [MarkdownCalloutStrategy(self)]

    def transform_content(
        self,
        element: TrackedElement,
        context: Optional[ProcessingContext] = None
    ) -> ProcessedContent:
        """Transform Markdown content."""
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
            return self.enrich_content(transformed, ctx)  # Added enrichment step

        except Exception as e:
            self.logger.error(f"Error transforming Markdown content: {str(e)}")
            raise

    def _extract_frontmatter(
        self,
        element: TrackedElement
    ) -> Optional[Dict[str, Any]]:
        """Extract frontmatter from Markdown content."""
        try:
            # Check cache first
            if element.id in self._frontmatter_cache:
                return self._frontmatter_cache[element.id]

            # Extract frontmatter
            content = element.content
            if not content.startswith('---\n'):
                return None

            # Find end of frontmatter
            end_idx = content.find('\n---\n', 4)
            if end_idx == -1:
                return None

            # Parse YAML frontmatter
            import yaml
            frontmatter_content = content[4:end_idx]
            frontmatter = yaml.safe_load(frontmatter_content)

            # Cache result
            self._frontmatter_cache[element.id] = frontmatter
            return frontmatter

        except Exception as e:
            self.logger.error(f"Error extracting frontmatter: {str(e)}")
            return None
