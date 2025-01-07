# app/dita/dita_parser.py

from pathlib import Path
from typing import Dict, List, Optional, Any, Type
from abc import ABC, abstractmethod
import logging

# Core managers
from ..event_manager import EventManager
from ..context_manager import ContextManager
from ..config.config_manager import ConfigManager
from ..metadata.metadata_manager import MetadataManager
from ..key_manager import KeyManager

# Processors
from .base_processor import BaseProcessor
from .dita_processor import DITAProcessor
from .markdown_processor import MarkdownProcessor

# Utils
from ..utils.cache import ContentCache
from ..utils.id_handler import DITAIDHandler
from ..utils.logger import DITALogger

# Types
from ..models.types import (
    ContentElement,
    ProcessedContent,
    ProcessingPhase,
    ProcessingState,
    ElementType,
    ProcessingMetadata,
    ProcessingContext,
    ProcessingRuleType,
    ValidationResult,
    DITAElementType
)

class ParsingStrategy(ABC):
    """Base strategy for file parsing."""

    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """Check if this strategy can parse the given file."""
        pass

    @abstractmethod
    def parse(
        self,
        file_path: Path,
        parser: 'DITAParser'
    ) -> ContentElement:
        """Parse file into a tracked element."""
        pass

class DITAMapStrategy(ParsingStrategy):
    """Strategy for parsing DITA maps."""

    def can_parse(self, file_path: Path) -> bool:
        """Check if file is a DITA map."""
        return file_path.suffix.lower() == '.ditamap'

    def parse(
        self,
        file_path: Path,
        parser: 'DITAParser'
    ) -> ContentElement:
        """Parse DITA map file."""
        try:
            # Create tracked element
            element = ContentElement.create_map(
                path=file_path,
                title="",  # Will be extracted during processing
                id_handler=parser.id_handler
            )

            # Load and parse content
            element.content = file_path.read_text()

            # Get processing rules using new rule resolution system
            resolved_rule = parser.config_manager.resolve_rule(
                element_type=ElementType.MAP,
                rule_type=ProcessingRuleType.ELEMENT,
                context=None  # During initial parsing we don't have a context yet
            )

            # Use resolved rules or fallback
            rules = resolved_rule or {
                "html_tag": "div",
                "default_classes": ["fallback-map"],
                "attributes": {
                    "data-type": "map",
                    "role": "article"
                }
            }

            # Log the rules being used
            parser.logger.debug(
                f"Processing rules for '{element.type.value}' during "
                f"{ProcessingPhase.DISCOVERY.value}: {rules}"
            )

            # Attach rules to element metadata for later processing
            element.metadata.update({
                "rules": rules,
                "element_type": element.type.value,
                "processing_phase": ProcessingPhase.DISCOVERY.value
            })

            return element

        except Exception as e:
            parser.logger.error(f"Error parsing map file {file_path}: {str(e)}")
            raise


class DITATopicStrategy(ParsingStrategy):
    """Strategy for parsing DITA topics."""

    def can_parse(self, file_path: Path) -> bool:
        """Check if file is a DITA topic."""
        return file_path.suffix.lower() == '.dita'

    def parse(
        self,
        file_path: Path,
        parser: 'DITAParser'
    ) -> ContentElement:
        """Parse DITA topic file."""
        # Create tracked element
        element = ContentElement.from_discovery(
            path=file_path,
            element_type=ElementType.DITA,
            id_handler=parser.id_handler
        )

        # Load and parse content
        element.content = file_path.read_text()
        return element

class MDITAMapStrategy(ParsingStrategy):
    """Strategy for parsing MDITA maps."""

    def can_parse(self, file_path: Path) -> bool:
        """Check if file is an MDITA map."""
        if not file_path.exists():
            return False

        if file_path.suffix.lower() == '.md':
            try:
                content = file_path.read_text()
                # Check for YAML frontmatter and list structure
                return (
                    content.startswith("---") and
                    "---" in content[3:] and
                    any(line.strip().startswith("- [") for line in content.splitlines())
                )
            except Exception:
                return False

        return False

    def parse(
        self,
        file_path: Path,
        parser: 'DITAParser'
    ) -> ContentElement:
        """Parse MDITA map file."""
        # Create tracked element
        element = ContentElement.create_map(
            path=file_path,
            title="",  # Will be extracted from frontmatter
            id_handler=parser.id_handler
        )

        # Load content
        element.content = file_path.read_text()

        # Set type to MARKDOWN for proper processing
        element.type = ElementType.MARKDOWN

        return element

class MarkdownTopicStrategy(ParsingStrategy):
    """Strategy for parsing Markdown topics."""

    def can_parse(self, file_path: Path) -> bool:
        """Check if file is a Markdown topic."""
        if file_path.suffix.lower() != '.md':
            return False

        # Make sure it's not an MDITA map
        if MDITAMapStrategy().can_parse(file_path):
            return False

        return True

    def parse(
        self,
        file_path: Path,
        parser: 'DITAParser'
    ) -> ContentElement:
        """Parse Markdown topic file."""
        # Create tracked element
        element = ContentElement.from_discovery(
            path=file_path,
            element_type=ElementType.MARKDOWN,
            id_handler=parser.id_handler
        )

        # Load and parse content
        element.content = file_path.read_text()
        return element



class DITAParser:
    """
    Main parsing orchestrator for content discovery and processing.
    Handles different file types through specialized strategies.
    """

    def __init__(
        self,
        event_manager: EventManager,
        context_manager: ContextManager,
        config_manager: ConfigManager,
        metadata_manager: MetadataManager,
        key_manager: KeyManager,
        content_cache: ContentCache,
        dtd_path: Path,
        logger: Optional[DITALogger] = None,
        id_handler: Optional[DITAIDHandler] = None
    ):
        # Core managers
        self.event_manager = event_manager
        self.context_manager = context_manager
        self.config_manager = config_manager
        self.metadata_manager = metadata_manager
        self.key_manager = key_manager
        self.content_cache = content_cache

        # Utils
        self.logger = logger or logging.getLogger(__name__)
        self.id_handler = id_handler or DITAIDHandler()

        # Initialize strategy registry
        self._strategies: List[ParsingStrategy] = []

        # Initialize processors
        self.dita_processor = DITAProcessor(
            event_manager=event_manager,
            context_manager=context_manager,
            config_manager=config_manager,
            metadata_manager=metadata_manager,
            key_manager=key_manager,
            content_cache=content_cache,
            dtd_path=dtd_path,  # Pass it through
            logger=logger,
            id_handler=id_handler
        )

        self.markdown_processor = MarkdownProcessor(
            event_manager=event_manager,
            context_manager=context_manager,
            config_manager=config_manager,
            metadata_manager=metadata_manager,
            content_cache=content_cache,
            logger=logger,
            id_handler=id_handler
        )

        # Initialize strategies after processors are ready
        self._initialize_strategies()

    def _initialize_strategies(self) -> None:
        """Initialize parsing strategies."""
        self._strategies.extend([
            DITAMapStrategy(),
            DITATopicStrategy(),
            MDITAMapStrategy(),  # Add MDITA map support
            MarkdownTopicStrategy()
        ])

    def parse_file(self, file_path: Path) -> ProcessedContent:
            """
            Parse a file using appropriate strategy and process it.
            Entry point for content processing.
            """
            try:
                # Find appropriate strategy
                strategy = self._get_strategy(file_path)
                if not strategy:
                    raise ValueError(f"No parsing strategy found for {file_path}")

                # Parse content into tracked element
                element = strategy.parse(file_path, self)

                # Process based on type
                if element.type == ElementType.DITAMAP:
                    return self.dita_processor.process_map(file_path)
                elif element.type == ElementType.DITA:
                    return self.dita_processor.process_topic(file_path)
                elif element.type == ElementType.MARKDOWN:
                    return self.markdown_processor.process_topic(file_path)
                else:
                    raise ValueError(f"Unsupported element type: {element.type}")

            except Exception as e:
                self.logger.error(f"Error parsing file {file_path}: {str(e)}")
                raise

    def _get_strategy(self, file_path: Path) -> Optional[ParsingStrategy]:
        """Get appropriate parsing strategy for file."""
        for strategy in self._strategies:
            if strategy.can_parse(file_path):
                return strategy
        return None
