from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type, Set, Tuple, Protocol
from pathlib import Path
from bs4 import Tag, BeautifulSoup
import logging
import re

# Latex transformation
from ..utils.latex.latex_processor import LaTeXProcessor
from ..utils.latex.latex_validator import LaTeXValidator
from ..utils.latex.katex_renderer import KaTeXRenderer
from ..models.types import LaTeXEquation, ProcessedEquation

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
    ProcessingMetadata,
    ProcessingContext,
    ValidationResult,
    ValidationMessage,
    ValidationSeverity,
    ContentScope,
    ProcessingStateInfo
)

class TransformStrategy(ABC):
    """Base strategy for content transformation."""

    @abstractmethod
    def can_transform(
        self,
        element: TrackedElement,
        context: ProcessingContext
    ) -> bool:
        """
        Check if this strategy can transform the element.

        Args:
            element: Element to transform
            context: Processing context

        Returns:
            bool: True if strategy can handle the element
        """
        pass

    @abstractmethod
    def transform(
        self,
        element: TrackedElement,
        context: ProcessingContext,
        metadata: ProcessingMetadata,
        config: Dict[str, Any]
    ) -> ProcessedContent:
        """
        Transform element according to strategy.

        Args:
            element: Element to transform
            context: Processing context
            metadata: Processing metadata
            config: Transformation configuration

        Returns:
            ProcessedContent: Transformed content
        """
        pass

    @abstractmethod
    def validate(
        self,
        element: TrackedElement,
        context: ProcessingContext
    ) -> ValidationResult:
        """
        Validate element before transformation.

        Args:
            element: Element to validate
            context: Processing context

        Returns:
            ValidationResult: Validation result
        """
        pass


class ContentEnricher(Protocol):
    """Protocol for content enrichment strategies."""
    def can_enrich(self, content: ProcessedContent, context: ProcessingContext) -> bool:
        """Check if content can be enriched."""
        ...

    def enrich(
        self,
        content: ProcessedContent,
        context: ProcessingContext,
        config: Dict[str, Any]
    ) -> ProcessedContent:
        """Enrich content."""
        ...

class LaTeXEnricher(ContentEnricher):
    """LaTeX equation enrichment strategy."""

    def __init__(self, transformer: 'BaseTransformer'):
        self.transformer = transformer
        self.latex_processor = LaTeXProcessor()
        self.latex_validator = LaTeXValidator()
        self.katex_renderer = KaTeXRenderer()

    def can_enrich(self, content: ProcessedContent, context: ProcessingContext) -> bool:
        """Check if content contains LaTeX equations."""
        return '$$' in content.html or '$' in content.html

    def enrich(
        self,
        content: ProcessedContent,
        context: ProcessingContext,
        config: Dict[str, Any]
    ) -> ProcessedContent:
        """Enrich content with rendered LaTeX equations."""
        try:
            equations = self._extract_equations(content.html)
            processed_equations = self.latex_processor.process_equations(equations)

            # Replace equations in content
            html = content.html
            for processed_eq in processed_equations:
                placeholder = f'<latex-equation id="{processed_eq.id}"></latex-equation>'
                # Convert ProcessedEquation back to LaTeXEquation
                latex_eq = LaTeXEquation(
                    id=processed_eq.id,
                    content=processed_eq.original,
                    is_block=processed_eq.is_block
                )
                rendered = self.katex_renderer.render_equation(latex_eq)
                html = html.replace(placeholder, rendered)

            return ProcessedContent(
                element_id=content.element_id,
                html=html,
                metadata={
                    **(content.metadata or {}),
                    "equations": [eq.id for eq in equations]
                }
            )

        except Exception as e:
            self.transformer.logger.error(f"LaTeX enrichment failed: {str(e)}")
            return content

    def _extract_equations(self, html: str) -> List[LaTeXEquation]:
        """Extract LaTeX equations from HTML content."""
        equations = []

        # Extract block equations
        block_pattern = r'\$\$(.*?)\$\$'
        for idx, match in enumerate(re.finditer(block_pattern, html, re.DOTALL)):
            equations.append(LaTeXEquation(
                id=f'eq-block-{idx}',
                content=match.group(1).strip(),
                is_block=True
            ))

        # Extract inline equations
        inline_pattern = r'(?<!\$)\$(.*?)\$(?!\$)'
        for idx, match in enumerate(re.finditer(inline_pattern, html)):
            equations.append(LaTeXEquation(
                id=f'eq-inline-{idx}',
                content=match.group(1).strip(),
                is_block=False
            ))

        return equations

class MediaEnricher(ContentEnricher):
    """Media content enrichment strategy."""

    def __init__(self, transformer: 'BaseTransformer'):
        self.transformer = transformer

    def can_enrich(self, content: ProcessedContent, context: ProcessingContext) -> bool:
        """Check if content contains media elements."""
        return any(tag in content.html.lower() for tag in ['<img', '<video', '<audio', '<iframe'])

    def enrich(
        self,
        content: ProcessedContent,
        context: ProcessingContext,
        config: Dict[str, Any]
    ) -> ProcessedContent:
        """Enrich media elements with proper attributes and loading."""
        try:
            soup = BeautifulSoup(content.html, 'html.parser')

            # Process images
            for img in soup.find_all('img'):
                self._process_image(img, context, config)

            # Process videos
            for video in soup.find_all('video'):
                self._process_video(video, context, config)

            # Process audio
            for audio in soup.find_all('audio'):
                self._process_audio(audio, context, config)

            # Process iframes
            for iframe in soup.find_all('iframe'):
                self._process_iframe(iframe, context, config)

            return ProcessedContent(
                element_id=content.element_id,
                html=str(soup),
                metadata=content.metadata
            )

        except Exception as e:
            self.transformer.logger.error(f"Media enrichment failed: {str(e)}")
            return content

    def _process_image(
        self,
        img: Tag,
        context: ProcessingContext,
        config: Dict[str, Any]
    ) -> None:
        """Process image element."""
        img['loading'] = 'lazy'
        img['decoding'] = 'async'

        # Handle classes properly
        current_classes = img.get('class', [])
        if isinstance(current_classes, str):
            current_classes = current_classes.split()
        img['class'] = ' '.join(current_classes + ['img-fluid'])

        # Resolve source path - handle potential list type
        src = img.get('src')
        if src and isinstance(src, str):  # Ensure src is a string
            img['src'] = self.transformer.html_helper.resolve_image_path(
                src,
                Path(context.metadata_refs.get('topic_path', ''))
            )

    def _process_audio(
        self,
        audio: Tag,
        context: ProcessingContext,
        config: Dict[str, Any]
    ) -> None:
        """Process audio element."""
        audio['controls'] = ''
        audio['preload'] = 'metadata'

        # Add source if not present
        if not audio.find('source'):
            src = audio.get('src')
            if src:
                source = BeautifulSoup().new_tag('source')
                source['src'] = src
                source['type'] = 'audio/mpeg'  # Default to MP3
                audio.append(source)
                del audio['src']

    def _process_video(
        self,
        video: Tag,
        context: ProcessingContext,
        config: Dict[str, Any]
    ) -> None:
        """Process video element."""
        video['controls'] = ''
        video['preload'] = 'metadata'

        # Handle classes properly
        current_classes = video.get('class', [])
        if isinstance(current_classes, str):
            current_classes = current_classes.split()
        video['class'] = ' '.join(current_classes + ['video-fluid'])

    def _process_iframe(
        self,
        iframe: Tag,
        context: ProcessingContext,
        config: Dict[str, Any]
    ) -> None:
        """Process iframe element."""
        iframe['loading'] = 'lazy'

        # Handle classes properly
        current_classes = iframe.get('class', [])
        if isinstance(current_classes, str):
            current_classes = current_classes.split()
        iframe['class'] = ' '.join(current_classes + ['iframe-fluid'])

        # Security attributes
        iframe['sandbox'] = 'allow-scripts allow-same-origin'
        iframe['referrerpolicy'] = 'no-referrer'



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
        id_handler: DITAIDHandler,
        logger: Optional[DITALogger] = None
    ):
        """Initialize transformer."""
        # Core dependencies
        self.event_manager = event_manager
        self.context_manager = context_manager
        self.config_manager = config_manager
        self.key_manager = key_manager
        self.content_cache = content_cache
        self.html_helper = html_helper
        self.heading_handler = heading_handler
        self.id_handler = id_handler
        self.logger = logger or logging.getLogger(__name__)

        # Strategy registry
        self._strategies: Dict[ElementType, List[TransformStrategy]] = {}

        # State tracking
        self._processed_elements: Set[str] = set()
        self._active_transformations: Dict[str, ProcessingStateInfo] = {}

        # Feature flags
        self._feature_flags: Dict[str, bool] = {}

        # Initialize
        self._initialize_strategies()
        self._register_event_handlers()

        # Initialize enrichers
        self._enrichers: List[ContentEnricher] = [
            LaTeXEnricher(self),
            MediaEnricher(self)
        ]

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

    @abstractmethod
    def transform_content(
        self,
        element: TrackedElement,
        context: Optional[ProcessingContext] = None
    ) -> ProcessedContent:
        """Transform element through appropriate strategy."""
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

            # Transform using strategy and enrich content
            transformed = strategy.transform(element, ctx, metadata, config)
            return self.enrich_content(transformed, ctx)

        except Exception as e:
            self.logger.error(f"Error transforming content: {str(e)}")
            raise

    @abstractmethod
    def register_strategy(
        self,
        element_type: ElementType,
        strategy: TransformStrategy
    ) -> None:
        """
        Register a transformation strategy.

        Args:
            element_type: Type of element
            strategy: Strategy implementation
        """
        pass

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

    def _get_strategies(
        self,
        element_type: ElementType
    ) -> List[TransformStrategy]:
        """Get transformation strategies for element type."""
        return self._strategies.get(element_type, [])

    def _handle_phase_start(self, **event_data: Any) -> None:
        """Handle phase start events."""
        pass  # Implemented by subclasses

    def _handle_state_change(self, **event_data: Any) -> None:
        """Handle state change events."""
        pass  # Implemented by subclasses

    def enrich_content(
        self,
        content: ProcessedContent,
        context: ProcessingContext
    ) -> ProcessedContent:
        """Enrich content with LaTeX and media processing."""
        try:
            enriched = content
            config = self.config_manager.get_processing_rules(
                ElementType.UNKNOWN,  # Use default rules
                context
            )

            # Apply each enricher that can handle the content
            for enricher in self._enrichers:
                if enricher.can_enrich(enriched, context):
                    enriched = enricher.enrich(enriched, context, config)

            return enriched

        except Exception as e:
            self.logger.error(f"Content enrichment failed: {str(e)}")
            return content
