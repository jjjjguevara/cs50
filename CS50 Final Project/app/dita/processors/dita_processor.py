# app/dita/processors/dita_processor.py

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Type, Union
from lxml import etree
from lxml.etree import _InputDocument

# Base processor and strategy
from .base_processor import BaseProcessor

# Core managers
from ..event_manager import EventManager
from ..context_manager import ContextManager
from ..config.config_manager import ConfigManager
from ..metadata.metadata_manager import MetadataManager
from ..key_manager import KeyManager

# Utils
from ..utils.cache import ContentCache, CacheEntryType
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
    ProcessingRuleType,
    ElementType,
    DITAElementType,
    DITAElementInfo,
    KeyDefinition
)

class DTDResolver(etree.Resolver):
    def __init__(self, dtd_path: Path):
        self.dtd_path = dtd_path
        super().__init__()

    def resolve(self, system_url: str, public_id: str, context) -> Optional[Union[etree._Element, etree._ElementTree]]:
        if system_url == "map.dtd":
            return self.resolve_filename(str(self.dtd_path / "map.dtd"), context)
        elif system_url == "topic.dtd":
            return self.resolve_filename(str(self.dtd_path / "topic.dtd"), context)
        return None

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
        dtd_path: Path,
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
        self.dtd_path = dtd_path  # Store DTD path
        self._current_context: Optional[ProcessingContext] = None

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


    def process_file(self, file_path: Path) -> TrackedElement:
        """Process a DITA file."""
        try:

            # Determine if it's a map or topic
            is_map = file_path.suffix == '.ditamap'

            # Create element
            element = TrackedElement.create_map(
                path=file_path,
                title="",  # Will be filled after parsing
                id_handler=self.id_handler
            ) if is_map else TrackedElement.from_discovery(
                path=file_path,
                element_type=ElementType.DITA,
                id_handler=self.id_handler
            )

            # Set type
            element.type = ElementType.DITAMAP if is_map else ElementType.DITA


            # Parse XML with DTD resolution
            from lxml import etree
            parser = etree.XMLParser(
                remove_blank_text=True,
                remove_comments=True,
                recover=True,
                resolve_entities=True,
                load_dtd=True,
                dtd_validation=False
            )

            # Add resolver with proper dtd_path
            parser.resolvers.add(DTDResolver(self.dtd_path))

            # Parse the XML content
            tree = etree.parse(str(file_path), parser)
            root = tree.getroot()

            if is_map:
                # Extract title
                title_elem = tree.find('.//title')
                if title_elem is not None and title_elem.text:
                    element.title = title_elem.text.strip()

                # Extract and process topic references
                for topicref in tree.xpath('//topicref[@href]'):
                    href = topicref.get('href').strip()
                    if href:
                        # Resolve topic path relative to map
                        topic_path = file_path.parent / href
                        if topic_path.exists():
                            topic_element = self.process_file(topic_path)
                            if topic_element:
                                element.topics.append(href)
                                self.content_cache.set(
                                    key=f"topic_{topic_element.id}",
                                    data=topic_element,
                                    entry_type=CacheEntryType.CONTENT,
                                    element_type=ElementType.DITA,
                                    phase=ProcessingPhase.DISCOVERY
                                )
            else:
                # For topics, extract title and content
                title_elem = tree.find('.//title')
                if title_elem is not None and title_elem.text:
                    element.title = title_elem.text.strip()

            # Store raw content without XML declaration
            element.content = etree.tostring(
                root,
                encoding='unicode',
                pretty_print=True,
                with_tail=True
            )

            return element

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            raise


    def process_map(self, map_path: Path) -> ProcessedContent:
        """Process a DITA map."""
        try:
            # Create tracked element for map
            element = TrackedElement.create_map(
                path=map_path,
                title="",  # Will be extracted during processing
                id_handler=self.id_handler
            )

            # Extract map metadata before processing
            metadata = {
                "content_type": "map",
                "file_path": str(map_path),
                "element_type": ElementType.DITAMAP.value,
                "creation_time": datetime.now().isoformat()
            }

            element.metadata = metadata

            # Process using base processor's element processing
            result = self.process_element(element)
            if not result:
                raise ValueError(f"Failed to process map: {map_path}")

            return result

        except Exception as e:
            self.logger.error(f"Error processing map {map_path}: {str(e)}")
            raise

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

                    # Use the new rule resolution system
                    resolved_rule = self.config_manager.resolve_rule(
                        element_type=ElementType(dita_type.value),
                        rule_type=ProcessingRuleType.SPECIALIZATION,  # Use specialization rule type
                        context=self._current_context  # Use current context from processor
                    )

                    # Update rules if resolution successful
                    if resolved_rule:
                        specialized_rules.update(resolved_rule)
                    else:
                        self.logger.warning(
                            f"No specialization rules resolved for {specialization_type} "
                            f"during {self._current_phase.value} phase"
                        )

                    # Log specialization resolution
                    self.logger.debug(
                        f"Resolved specialization rules for {specialization_type} "
                        f"during {self._current_phase.value} phase"
                    )

            return specialized_rules

        except Exception as e:
            self.logger.error(
                f"Error getting specialization rules during "
                f"{self._current_phase.value} phase: {str(e)}"
            )
            return base_rules  # Error fallback

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
