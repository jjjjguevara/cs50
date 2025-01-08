# app/dita/content_factory.py

from typing import Dict, List, Optional, Any, Union, Type
from pathlib import Path
import logging
from dataclasses import dataclass

# Core managers
from .event_manager import EventManager, EventType
from .context_manager import ContextManager
from .config.config_manager import ConfigManager
from .key_manager import KeyManager
from .metadata.metadata_manager import MetadataManager

# DTD components
from .dtd.dtd_mapper import DTDSchemaMapper
from .dtd.dtd_validator import DTDValidator
from .dtd.dtd_resolver import DTDResolver

# Processors
from .processors.dita_processor import DITAProcessor
from .processors.markdown_processor import MarkdownProcessor

# Transformers
from .transformers.dita_transformer import DITATransformer
from .transformers.markdown_transformer import MarkdownTransformer

# Utils
from .utils.cache import ContentCache, CacheEntryType
from .utils.html_helpers import HTMLHelper
from .utils.heading import HeadingHandler
from .utils.id_handler import DITAIDHandler, IDType
from .utils.logger import DITALogger


# Types
from .models.types import (
    ProcessedContent,
    ContentElement,
    ProcessingPhase,
    ProcessingState,
    ElementType,
    ProcessingMetadata,
    ProcessingContext,
    ValidationResult,
    ValidationMessage,
    ValidationSeverity,
    ContentScope
)

@dataclass
class AssemblyOptions:
    """Configuration options for content assembly."""
    add_toc: bool = True
    add_navigation: bool = True
    add_metadata: bool = True
    validate_output: bool = True
    minify_html: bool = False

class ContentFactory:
    """
    Main orchestrator for content processing and assembly.
    Coordinates processors and transformers to produce final HTML output.
    """

    def __init__(
        self,
        event_manager: EventManager,
        context_manager: ContextManager,
        config_manager: ConfigManager,
        key_manager: KeyManager,
        metadata_manager: MetadataManager,
        content_cache: ContentCache,
        html_helper: HTMLHelper,
        heading_handler: HeadingHandler,
        id_handler: DITAIDHandler,
        dtd_path: Optional[Path] = None,
        logger: Optional[DITALogger] = None
    ):
        """Initialize content factory."""
        # Core dependencies
        self.event_manager = event_manager
        self.context_manager = context_manager
        self.config_manager = config_manager
        self.key_manager = key_manager
        self.metadata_manager = metadata_manager
        self.content_cache = content_cache
        self.html_helper = html_helper
        self.heading_handler = heading_handler
        self.id_handler = id_handler
        self.logger = logger or DITALogger()
        self.dtd_path = dtd_path

        # Initialize DTD components if DTD path is provided
        if self.dtd_path:
            self._init_dtd_components()

        # Initialize processors
        self._init_processors()

        # Initialize transformers
        self._init_transformers()

        # State tracking
        self._current_entry_id: Optional[str] = None
        self._assembly_cache: Dict[str, str] = {}

    def _init_dtd_components(self) -> None:
        """Initialize DTD processing components."""
        try:
            # Validate DTD path
            if not self.dtd_path:
                raise ValueError("DTD path is required for DTD processing")

            # Initialize DTD resolver
            self._dtd_resolver = DTDResolver(
                base_path=self.dtd_path,  # Now we know it's not None
                logger=self.logger
            )

            # Initialize DTD mapper
            self._dtd_mapper = DTDSchemaMapper(
                logger=self.logger
            )

            # Initialize DTD validator
            self._dtd_validator = DTDValidator(
                resolver=self._dtd_resolver,
                dtd_mapper=self._dtd_mapper,
                validation_manager=self.config_manager.validation_manager,
                logger=self.logger
            )

        except Exception as e:
            self.logger.error(f"Error initializing DTD components: {str(e)}")
            raise

    def _init_processors(self) -> None:
        """Initialize content processors with DTD support."""
        # Create DITA processor args
        dita_processor_args = {
            "event_manager": self.event_manager,
            "context_manager": self.context_manager,
            "config_manager": self.config_manager,
            "metadata_manager": self.metadata_manager,
            "key_manager": self.key_manager,
            "content_cache": self.content_cache,
            "logger": self.logger,
            "id_handler": self.id_handler,
            "html_helper": self.html_helper,
            "heading_handler": self.heading_handler,
        }

        # Add DTD path only if it exists
        if self.dtd_path:
            dita_processor_args["dtd_path"] = self.dtd_path

        self._processors = {
            ElementType.DITA: DITAProcessor(**dita_processor_args),
            ElementType.MARKDOWN: MarkdownProcessor(
                event_manager=self.event_manager,
                context_manager=self.context_manager,
                config_manager=self.config_manager,
                metadata_manager=self.metadata_manager,
                content_cache=self.content_cache,
                logger=self.logger,
                id_handler=self.id_handler
            )
        }

    def _init_transformers(self) -> None:
        """Initialize content transformers."""
        self._transformers = {
            ElementType.DITA: DITATransformer(
                event_manager=self.event_manager,
                context_manager=self.context_manager,
                config_manager=self.config_manager,
                key_manager=self.key_manager,
                content_cache=self.content_cache,
                html_helper=self.html_helper,
                heading_handler=self.heading_handler,
                id_handler=self.id_handler,
                logger=self.logger,
                specialization_rules=self.config_manager.get_processing_rules(
                    ElementType.DITA,
                    None  # No context needed for specialization rules
                ).get('specializations', {})
            ),
            ElementType.MARKDOWN: MarkdownTransformer(
                event_manager=self.event_manager,
                context_manager=self.context_manager,
                config_manager=self.config_manager,
                key_manager=self.key_manager,
                content_cache=self.content_cache,
                html_helper=self.html_helper,
                heading_handler=self.heading_handler,
                id_handler=self.id_handler,
                logger=self.logger,
                custom_syntax_rules=self.config_manager.get_processing_rules(
                    ElementType.MARKDOWN
                )
            )
        }

    def process_entry(
        self,
        entry_path: Union[str, Path],
        options: Optional[AssemblyOptions] = None
    ) -> str:
        """Process an entry file with DTD validation."""
        try:
            # Convert path
            path = Path(entry_path)
            self.logger.debug(f"Processing file at path: {path}")

            # Verify file exists
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            self._current_entry_id = self.id_handler.generate_id(
                path.stem,
                IDType.MAP if path.suffix == '.ditamap' else IDType.TOPIC
            )

            # Check if DTD validation is enabled
            dtd_validation_enabled = self.config_manager.get_feature_state("dtd_validation")
            if dtd_validation_enabled and hasattr(self, '_dtd_validator'):
                # Get appropriate DTD file
                content_type = self._determine_content_type(path)
                if content_type in (ElementType.DITA, ElementType.DITAMAP):
                    dtd_file = "map.dtd" if content_type == ElementType.DITAMAP else "topic.dtd"
                    dtd_path = self.dtd_path / dtd_file if self.dtd_path else None

                    if dtd_path and dtd_path.exists():
                        # Perform DTD validation
                        with open(path, 'r') as f:
                            content = f.read()
                        validation_result = self._dtd_validator.validate_content(
                            content=content,
                            dtd_path=dtd_path
                        )
                        if not validation_result.is_valid:
                            # Log validation errors
                            for msg in validation_result.messages:
                                self.logger.error(f"DTD validation error: {msg.message}")
                            if self.config_manager.get_feature_state("strict_dtd_validation"):
                                raise ValueError("DTD validation failed")

            # Process content based on type
            processed = self._process_content(path)

            # Assemble final content
            assembled = self._assemble_content(
                processed,
                options or AssemblyOptions()
            )

            return assembled

        except Exception as e:
            self.logger.error(f"Error processing entry {entry_path}: {str(e)}")
            raise

    def _process_content(self, path: Path) -> ProcessedContent:
        """Process content with DTD awareness."""
        try:
            # Determine content type
            content_type = self._determine_content_type(path)
            processor = self._processors.get(content_type)
            if not processor:
                raise ValueError(f"No processor found for {content_type}")

            # Process based on type
            if content_type == ElementType.DITAMAP:
                return processor.process_map(path)
            else:
                return processor.process_topic(path)

        except Exception as e:
            self.logger.error(f"Error processing content: {str(e)}")
            raise

    def transform_content(
        self,
        element: ContentElement,
        context: Optional[ProcessingContext] = None
    ) -> ProcessedContent:
        """Transform processed content to HTML."""
        try:
            # Get transformer based on element type
            transformer = self._transformers.get(element.type)
            if not transformer:
                raise ValueError(f"No transformer found for {element.type}")

            # Transform content using appropriate transformer
            return transformer.transform_content(
                element=element,
                context=context
            )

        except Exception as e:
            self.logger.error(f"Error transforming content {element.path}: {str(e)}")
            raise

    def _determine_content_type(self, path: Path) -> ElementType:
        """Determine content type from file path."""
        if path.suffix == '.ditamap':
            return ElementType.DITA
        elif path.suffix == '.dita':
            return ElementType.DITA
        elif path.suffix == '.md':
            return ElementType.MARKDOWN
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

    def _assemble_content(
        self,
        content: ProcessedContent,
        options: AssemblyOptions
    ) -> str:
        """
        Assemble final HTML content with optional features.

        Args:
            content: Processed content
            options: Assembly options

        Returns:
            str: Final assembled HTML
        """
        try:
            # Create main container
            container = self.html_helper.create_element(
                tag="div",
                attrs={
                    "class": ["content-container"],
                    "id": f"content-{self._current_entry_id}",
                    "role": "main"
                }
            )

            # Add table of contents if enabled
            if options.add_toc and self._current_entry_id:  # Check for entry ID
                toc = self.html_helper.generate_toc(
                    self.heading_handler.get_topic_headings(self._current_entry_id)
                )
                container = self.html_helper.create_container(
                    tag="div",
                    children=[toc, content.html],
                    attrs=container.attrs
                )
            else:
                container = self.html_helper.create_container(
                    tag="div",
                    children=[content.html],
                    attrs=container.attrs
                )

            # Add metadata if enabled
            if options.add_metadata and content.metadata:  # Check for metadata
                metadata_html = self._generate_metadata_html(content.metadata)
                container = self.html_helper.create_container(
                    tag="div",
                    children=[metadata_html, str(container)],
                    attrs={"class": ["content-wrapper"]}
                )

            # Validate final output if enabled
            if options.validate_output:
                self._validate_output(str(container))

            # Minify if enabled
            if options.minify_html:
                return self._minify_html(str(container))

            return str(container)

        except Exception as e:
            self.logger.error(f"Error assembling content: {str(e)}")
            raise

    def _generate_metadata_html(
        self,
        metadata: Dict[str, Any]
    ) -> str:
        """Generate HTML for content metadata."""
        if not metadata:  # Early return for empty metadata
            return ""

        try:
            metadata_div = self.html_helper.create_element(
                tag="div",
                attrs={
                    "class": ["content-metadata"],
                    "hidden": "true",
                    "aria-hidden": "true"
                }
            )

            # Add metadata as data attributes
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata_div.attrs[f"data-{key}"] = str(value)

            return str(metadata_div)

        except Exception as e:
            self.logger.error(f"Error generating metadata HTML: {str(e)}")
            return ""

    def _validate_output(self, html: str) -> None:
        """Validate final HTML output."""
        result = self.html_helper.validate_html(html)
        if not result.is_valid:
            self.logger.warning(
                f"HTML validation warnings: {[msg.message for msg in result.messages]}"
            )

    def _minify_html(self, html: str) -> str:
        """Minify HTML content."""
        # Simple minification - could be enhanced
        return (
            html.replace('\n', '')
                .replace('\r', '')
                .replace('  ', ' ')
                .replace('> <', '><')
        )


    def cleanup(self) -> None:
        """Clean up factory resources."""
        try:
            self._assembly_cache.clear()
            self._current_entry_id = None

            # Clean up processors and transformers
            for processor in self._processors.values():
                processor.cleanup()
            for transformer in self._transformers.values():
                transformer.cleanup()

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise
