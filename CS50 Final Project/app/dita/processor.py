# Standard library imports
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import html
import traceback
import json
import os
import hashlib
import time

# XML/XSLT/YAML
from lxml import etree
import frontmatter

# Type aliases
HTMLString = str
XMLElement = Any

# Global config
from config import DITAConfig
from .config_manager import config_manager

# Global types
from .utils.types import (
    # Enums
    ElementType,
    ProcessingState,
    ProcessingPhase,

    # Content types
    DiscoveredMap,
    DiscoveredTopic,
    ParsedElement,
    TrackedElement,
    ProcessedContent,

    # Context types
    MapContext,
    TopicContext,
    ProcessingContext,

    # Processing types
    ParsedMap,
    ProcessingResult,
    ProcessingOptions,
    ProcessingError,

    # Feature types
    ContentFeatures,
    ProcessingFeatures,

    # Artifact types
    ArtifactReference,
    ProcessedArtifact,

    # Reference types
    HeadingReference,
    CrossReference,

    # Log types
    LogContext,

    # Type aliases
    PathLike,
    MetadataDict,
    HTMLString
)

# Third-party imports
from bs4 import BeautifulSoup, Tag
from lxml import etree
import markdown
import frontmatter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


# Local imports
from .utils.latex.katex_renderer import KaTeXRenderer
from .utils.dita_parser import DITAParser
from .utils.heading import HeadingHandler
from .utils.html_helpers import HTMLHelper
from .utils.metadata import MetadataHandler
from .utils.id_handler import DITAIDHandler
from .utils.dita_transform import DITATransformer
from .utils.markdown.md_transform import MarkdownTransformer
from .utils.logger import DITALogger




class DITAProcessor:
    """
    Main orchestrator class for DITA content processing.
    Talks to parsing, element tracking, and conditional processing pipelines.
    """
    def __init__(self, config: Optional[DITAConfig] = None):
        """
        Initialize processor with optional configuration.

        Args:
            config: Optional configuration object for processor.
        """
        self.logger = logging.getLogger(__name__)
        self._cache: Dict[str, Any] = {}
        self.current_phase = ProcessingPhase.DISCOVERY

        # Debug initialization
        self.logger.debug("Initializing DITAProcessor.")

        # Set application root
        self.app_root = Path(__file__).parent.parent

        # Initialize parser and HTML helper
        self.dita_parser = DITAParser()
        self.html_helper = HTMLHelper(self.app_root / 'dita')

        # Initialize with config if provided, otherwise use default settings
        if config:
            self.logger.debug("Configuring processor with provided config.")
            self._init_with_config(config)
        else:
            self.logger.debug("No config provided, initializing with defaults.")
            self._init_default()

        # Log final configuration state
        self.logger.debug(f"Final configuration: index_numbers_enabled={getattr(self, 'index_numbers_enabled', None)}")

        # Final log
        self.logger.debug("DITAProcessor fully initialized.")


    def _init_with_config(self, config: DITAConfig) -> None:
        """Initialize processor with provided configuration."""
        try:
            self.logger.debug("Initializing processor with config")

            # Initialize paths from config
            self.dita_root = config.paths.dita_root
            self.maps_dir = config.paths.maps_dir
            self.topics_dir = config.paths.topics_dir
            self.output_dir = config.paths.output_dir
            self.artifacts_dir = config.paths.artifacts_dir

            # Initialize handlers with config
            self._init_handlers(config)

            # Initialize processors with config
            self._init_processors(config)

            # Initialize processing options
            self.processing_options = ProcessingOptions(
                process_latex=config.processing.process_latex,
                number_headings=config.processing.number_headings,
                enable_cross_refs=config.processing.enable_cross_refs,
                process_artifacts=config.processing.process_artifacts,
                show_toc=config.processing.show_toc
            )

            self.current_context = None
            self.processed_maps = {}
            self.processed_topics = {}

        except Exception as e:
            self.logger.error(f"Processor initialization failed: {str(e)}")
            raise

    def _init_default(self) -> None:
        """Initialize processor with default settings."""
        self._init_paths()
        self._init_handlers(None)
        self._init_processors(None)
        self.current_context = None
        self.processed_maps = {}
        self.processed_topics = {}
        self.processing_options = ProcessingOptions()

    def _init_handlers(self, config: Optional[DITAConfig] = None) -> None:
        """
        Initialize handlers with optional configuration.
        """
        try:
            self.heading_handler = HeadingHandler()

            # Ensure heading numbering is enabled or disabled based on the configuration
            if config and hasattr(config.processing, 'number_headings'):
                self.heading_handler.set_numbering_enabled(config.processing.number_headings)
            else:
                self.heading_handler.set_numbering_enabled(False)

            # Initialize other handlers
            self.id_handler = DITAIDHandler()
            self.metadata_handler = MetadataHandler()

            # Configure handlers with config if available
            if config:
                if hasattr(self.id_handler, 'configure'):
                    self.id_handler.configure(config)
                if hasattr(self.metadata_handler, 'configure'):
                    self.metadata_handler.configure(config)

        except Exception as e:
            self.logger.error(f"Handler initialization failed: {str(e)}")
            raise


    def _init_processors(self, config: Optional[DITAConfig] = None) -> None:
        """Initialize processors with optional configuration."""
        try:
            self.logger.debug("Initializing processors.")

            # Initialize transformers
            self.dita_transformer = DITATransformer(self.dita_root)
            self.md_transformer = MarkdownTransformer(self.dita_root)

            if config:
                self._configure_transformers(config)

            self.logger.debug("Processors initialized successfully.")

        except Exception as e:
            self.logger.error(f"Processor initialization failed: {str(e)}")
            raise


    def _init_paths(self) -> None:
        """Initialize critical paths with defaults."""
        try:
            # Set up root paths
            self.dita_root = self.app_root / 'dita'

            # Set up content directories
            self.maps_dir = self.dita_root / 'maps'
            self.topics_dir = self.dita_root / 'topics'
            self.output_dir = self.dita_root / 'output'
            self.artifacts_dir = self.dita_root / 'artifacts'

            # Ensure directories exist
            self._init_directories()

            self.logger.debug("Paths initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing paths: {str(e)}")
            raise

    def _init_directories(self) -> None:
            """Create necessary directories if they don't exist."""
            try:
                # Define required directories
                required_dirs = [
                    self.maps_dir,
                    self.topics_dir / 'abstracts',
                    self.topics_dir / 'acoustics',
                    self.topics_dir / 'articles',
                    self.topics_dir / 'audio',
                    self.topics_dir / 'journals',
                    self.output_dir,
                    self.artifacts_dir
                ]

                # Create directories
                for directory in required_dirs:
                    directory.mkdir(parents=True, exist_ok=True)
                    self.logger.debug(f"Ensured directory exists: {directory}")

            except Exception as e:
                self.logger.error(f"Failed to create directories: {str(e)}")
                raise


    def _init_map_context(self, discovered_map: DiscoveredMap) -> MapContext:
        """
        Initialize the map context based on the discovered map.

        Args:
            discovered_map (DiscoveredMap): The discovered map.

        Returns:
            MapContext: The initialized map context.
        """
        try:
            # Extract map features, including heading numbering
            map_features = self._get_map_features(discovered_map.metadata)

            # Add heading numbering feature to map features
            if hasattr(self.processing_options, 'number_headings'):
                map_features['number_headings'] = self.processing_options.number_headings
                self.logger.debug(f"Map heading numbering set to: {self.processing_options.number_headings}")
            else:
                map_features['number_headings'] = False  # Default if undefined
                self.logger.debug("Map heading numbering defaulted to disabled.")

            # Create the MapContext
            return MapContext(
                map_id=discovered_map.id,
                map_path=discovered_map.path,
                metadata=discovered_map.metadata,
                topic_order=[topic.id for topic in discovered_map.topics],
                features=map_features,
                type=ElementType.DITAMAP
            )
        except Exception as e:
            self.logger.error(f"Error initializing map context: {str(e)}")
            raise




    def _get_map_features(self, metadata: Dict[str, Any]) -> Dict[str, bool]:
        """Get map-level processing features"""
        return {
            'process_latex': bool(metadata.get('process-latex', True)),
            'number_headings': bool(metadata.get('index-numbers', True)),
            'show_toc': bool(metadata.get('append-toc', True)),
            'enable_cross_refs': bool(metadata.get('enable-xrefs', True)),
            # More map-level features are added as needed
        }

    def _get_topic_features(self, metadata: Dict[str, Any]) -> Dict[str, bool]:
        """Get topic-level processing features"""
        return {
            'show_title': metadata.get('show_title', True),
            'process_artifacts': metadata.get('process_artifacts', True),
            'show_abstract': metadata.get('show_abstract', False),
            'enable_comments': metadata.get('enable_comments', False),
            # More topic-level features are added as needed
        }



    ####################
    # State management #
    ####################

    def _init_state(self) -> None:
        """Initialize processor state."""
        try:
            self.current_phase = ProcessingPhase.DISCOVERY
            self.heading_handler.init_state()
            self._reset_state()

        except Exception as e:
            self.logger.error(f"State initialization failed: {str(e)}")
            raise

    def _reset_state(self) -> None:
        """Reset processor state."""
        try:
            self.current_context = None
            self.processed_maps.clear()
            self.processed_topics.clear()

        except Exception as e:
            self.logger.error(f"State reset failed: {str(e)}")
            raise

    def _validate_state(self) -> bool:
        """
        Validate current processor state.

        Returns:
            True if state is valid
        """
        try:
            # Validate context
            if not self._validate_context():
                return False

            # Validate heading state
            if not hasattr(self.heading_handler, '_state'):
                return False

            # Validate phase
            if not hasattr(self, 'current_phase'):
                return False

            return True

        except Exception as e:
            self.logger.error(f"State validation failed: {str(e)}")
            return False




    ######################
    # Context processing #
    ######################

    def initialize_context(self, discovered_map: DiscoveredMap, reset: bool = False) -> None:
        """
        Initialize or reinitialize the processing context.

        Args:
            discovered_map (DiscoveredMap): The discovered map for which the context is being initialized.
            reset (bool): Whether this is a reinitialization (default: False).
        """
        try:
            self.logger.debug(f"{'Resetting' if reset else 'Initializing'} processing context")

            # Create MapContext
            map_context = MapContext(
                map_id=discovered_map.id,
                map_path=discovered_map.path,
                metadata=discovered_map.metadata,
                topic_order=[topic.id for topic in discovered_map.topics],
                features=self._get_map_features(discovered_map.metadata),
                type=ElementType.DITAMAP  # Type is now valid
            )

            # Create TopicContext objects
            topics = {
                topic.id: TopicContext(
                    topic_id=topic.id,
                    topic_path=topic.path,
                    parent_map_id=discovered_map.id,
                    metadata=topic.metadata,
                    features=self._get_topic_features(topic.metadata),
                    type=topic.type  # Ensure the type is passed here
                )
                for topic in discovered_map.topics
            }

            # Assign to current context
            self.current_context = ProcessingContext(
                map_context=map_context,
                topics=topics
            )

            self.logger.debug(f"Processing context {'reset' if reset else 'initialized'} successfully")
        except Exception as e:
            self.logger.error(f"Failed to {'reset' if reset else 'initialize'} context: {str(e)}")
            raise




    def _determine_content_type(self, path: Path) -> ElementType:
        """
        Determine the type of content based on the file extension.
        """
        if path.suffix == '.md':
            return ElementType.MARKDOWN
        elif path.suffix == '.dita':
            return ElementType.DITA
        else:
            self.logger.warning(f"Unknown content type for file: {path}")
            return ElementType.UNKNOWN



    def _update_transformation_context(self) -> None:
        self.logger.debug(f"Updating transformation context for phase: {self.current_phase}")
        """Update context for transformation phase."""
        try:
            if not self.current_context:
                raise ProcessingError(
                    error_type="context",
                    message="No context available for transformation update",
                    context="transformation_phase"
                )

            # Update topic states to processing
            for topic_id in self.current_context.map_context.topic_order:
                topic_context = self.current_context.topics.get(topic_id)
                if topic_context and topic_context.state == ProcessingState.PENDING:
                    topic_context.state = ProcessingState.PROCESSING

            self.logger.debug("Updated transformation context")

        except Exception as e:
            self.logger.error(f"Transformation context update failed: {str(e)}")
            raise

    def _update_assembly_context(self) -> None:
        """Update context for assembly phase."""
        try:
            if not self.current_context:
                raise ProcessingError(
                    error_type="context",
                    message="No context available for assembly update",
                    context="assembly_phase"
                )

            # Validate all topics are processed
            for topic_id in self.current_context.map_context.topic_order:
                topic_context = self.current_context.topics.get(topic_id)
                if topic_context and topic_context.state != ProcessingState.COMPLETED:
                    raise ProcessingError(
                        error_type="assembly",
                        message=f"Topic {topic_id} not fully processed",
                        context="assembly_phase"
                    )

            self.logger.debug("Updated assembly context")

        except Exception as e:
            self.logger.error(f"Assembly context update failed: {str(e)}")
            raise

    def _update_context_state(self, topic_id: str, state: ProcessingState) -> None:
        """
        Update processing state for a topic.

        Args:
            topic_id: ID of topic to update
            state: New processing state
        """
        try:
            if not self.current_context:
                raise ValueError("No current context available")

            if topic_id not in self.current_context.topics:
                raise ValueError(f"Topic {topic_id} not found in context")

            self.current_context.topics[topic_id].state = state
            self.logger.debug(f"Updated topic {topic_id} state to {state.value}")

        except Exception as e:
            self.logger.error(f"Failed to update context state: {str(e)}")
            raise

    def _get_topic_context(self, topic_id: str) -> Optional[TopicContext]:
        """
        Get topic context by ID.

        Args:
            topic_id: ID of topic to get

        Returns:
            TopicContext if found, None otherwise
        """
        try:
            if not self.current_context:
                return None

            return self.current_context.topics.get(topic_id)

        except Exception as e:
            self.logger.error(f"Failed to get topic context: {str(e)}")
            return None

    def _update_context_features(self, features: Dict[str, bool]) -> None:
        """
        Update context features during processing.

        Args:
            features: New feature flags to apply
        """
        try:
            if not self.current_context:
                raise ValueError("No current context available")

            # Update map-level features
            self.current_context.map_context.features.update(features)
            self.logger.debug("Updated context features")

        except Exception as e:
            self.logger.error(f"Failed to update context features: {str(e)}")
            raise

    def _process_discovered_content(self, discovered_map: DiscoveredMap) -> List[ProcessedContent]:
        """
        @deprecated Use run_transformation_phase instead.
        Process discovered content with context management.
        """
        try:

            if not self.current_context:
                        raise ValueError("No processing context available")

            processed_content = []

            # Check map metadata for heading numbering
            metadata = self.metadata_handler.extract_metadata(
                        discovered_map.path,  # Using path instead of source_path
                        content_id=discovered_map.id
                    )
            index_numbers = metadata.get('index-numbers', 'true').lower() == 'true'
            self.heading_handler.set_numbering_enabled(index_numbers)


            if not self._validate_context():
                raise ProcessingError(
                    error_type="processing",
                    message="Invalid processing context",
                    context=discovered_map.path
                )

            processed_content = []

            # Create tracked elements
            tracked_elements = self._track_elements(discovered_map)

            # Process elements using context management
            processed_content = self._process_elements(tracked_elements)

            return processed_content

        except Exception as e:
            self.logger.error(f"Error processing content: {str(e)}")
            raise



    def _update_context(self, phase: ProcessingPhase) -> None:
        """Update context for processing phase."""
        try:
            if not self.current_context:
                self.logger.debug("No context available, initializing new context")
                return  # Skip update if no context yet

            # Update phase-specific context
            if phase == ProcessingPhase.TRANSFORMATION:
                self._update_transformation_context()
            elif phase == ProcessingPhase.ASSEMBLY:
                self._update_assembly_context()

            self.logger.debug(f"Updated context for {phase.value} phase")

        except Exception as e:
            self.logger.error(f"Context update failed: {str(e)}")
            raise

    def _validate_context(self) -> bool:
        """
        Validate current processing context.

        Returns:
            True if context is valid
        """
        try:
            if not self.current_context:
                return False

            # Validate map context
            map_context = self.current_context.map_context
            if not map_context.map_path.exists():
                return False

            # Validate topic contexts
            for topic_id, topic_context in self.current_context.topics.items():
                if not topic_context.topic_path.exists():
                    return False
                if topic_id not in map_context.topic_order:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Context validation failed: {str(e)}")
            return False

    ########################
    # Map level processing #
    ########################

    def discover_map_content(self, map_path: Path) -> DiscoveredMap:
        """
        Discover and validate all content referenced in a DITA map.

        Args:
            map_path: Path to the ditamap file

        Returns:
            DiscoveredMap containing all found content

        Raises:
            ValueError: If map cannot be parsed or critical content is missing
        """
        try:
            self.logger.info(f"Beginning content discovery for map: {map_path}")

            # Generate map ID
            map_id = self.id_handler.generate_map_id(map_path)

            # Parse the map file
            tree = self.dita_parser.parse_file(map_path)
            if tree is None:
                raise ValueError(f"Failed to parse map file: {map_path}")

            # Extract map metadata
            map_metadata = self.metadata_handler.extract_metadata(map_path, map_id)

            # Get map title if present
            title_elem = tree.find(".//title")
            map_title = title_elem.text if title_elem is not None else None

            # Discover topics
            discovered_topics = []
            topic_refs = tree.xpath(".//topicref")

            self.logger.debug(f"Found {len(topic_refs)} topic references to process")

            for topicref in topic_refs:
                topic = self._discover_topic(topicref, map_path)
                if topic:
                    discovered_topics.append(topic)

            # Create discovery result
            discovered_map = DiscoveredMap(
                id=map_id,
                path=map_path,
                title=map_title,
                topics=discovered_topics,
                metadata=map_metadata
            )

            self.logger.info(
                f"Content discovery complete - Found {len(discovered_topics)} "
                f"valid topics in map {map_id}"
            )

            return discovered_map

        except Exception as e:
            self.logger.error(f"Error during content discovery: {str(e)}")
            raise

    def process_ditamap(self, ditamap_path: Path) -> Tuple[List[str], Dict[str, Any]]:
        """
        Process a DITA map file through the pipeline.

        Args:
            ditamap_path: Path to the .ditamap file.

        Returns:
            Tuple[List[str], Dict[str, Any]]: The transformed HTML content for all topics in the map
                                              and the metadata containing the `.ditamap` title.
        """
        self.logger.debug(f"Processing DITA map: {ditamap_path}")

        try:
            # Parse the map
            parsed_map = self.dita_parser.parse_map(ditamap_path)
            if not parsed_map or not isinstance(parsed_map, ParsedMap):
                raise ValueError(f"Failed to parse map: {ditamap_path}")
            self.logger.debug(f"Parsed map: {parsed_map}")

            # Set heading numbering based on metadata
            self.heading_handler.set_numbering_enabled(
                parsed_map.metadata.get("index-numbers", "false").lower() == "true"
            )
            self.logger.debug(f"Heading numbering enabled: {self.heading_handler.numbering_enabled}")

            # Extract title and metadata for the map
            map_title = parsed_map.title or "Untitled Map"
            combined_metadata = {
                'content_type': 'ditamap',
                'content_id': parsed_map.metadata.get('id', "unknown_map"),
                'processed_at': datetime.now().isoformat(),
                'page_title': map_title,
                **parsed_map.metadata
            }
            self.logger.debug(f"Combined metadata: {combined_metadata}")

            # Convert ParsedMap to DiscoveredMap
            discovered_map = DiscoveredMap(
                id=combined_metadata['content_id'],
                path=parsed_map.source_path,
                title=map_title,
                topics=[
                    DiscoveredTopic(
                        id=topic.id,
                        path=topic.topic_path,
                        type=topic.type,
                        href=str(topic.topic_path.relative_to(self.dita_root)) if self.dita_root in topic.topic_path.parents else str(topic.topic_path),
                        metadata=topic.metadata
                    )
                    for topic in parsed_map.topics
                ],
                metadata=parsed_map.metadata
            )
            self.logger.debug(f"Converted to DiscoveredMap: {discovered_map}")

            # Execute the pipeline
            transformed_contents = self.execute_pipeline(discovered_map)
            self.logger.debug(f"Processed DITA map with {len(transformed_contents)} topics.")

            return transformed_contents, combined_metadata

        except Exception as e:
            self.logger.error(f"Error processing DITA map {ditamap_path}: {str(e)}", exc_info=True)
            raise



    def _process_map_content(self, parsed_elements: List[ParsedElement]) -> List[str]:
        """
        Process map content by transforming each parsed element.

        Args:
            parsed_elements (List[ParsedElement]): List of parsed map elements.

        Returns:
            List[str]: List of transformed HTML content for each topic.
        """
        transformed_content = []
        for parsed in parsed_elements:
            try:
                self.logger.debug(f"Processing parsed element: {parsed.topic_id}")
                # Directly process the ParsedElement
                html_content = self._process_topic(parsed)
                if html_content:
                    transformed_content.append(html_content)
            except Exception as e:
                self.logger.error(f"Error processing parsed element {parsed.topic_id}: {str(e)}")

        return transformed_content




    def _assemble_final_html(self, content_list: List[ProcessedContent],
                            discovered_map: DiscoveredMap) -> str:
        """
        Assemble final HTML.

        @deprecated Use run_assembly_phase instead.
        """
        try:
            html_parts = [
                f'<div class="map-content" data-map-id="{discovered_map.id}">'
            ]

            # Add ToC if enabled - check for context first
            if self.current_context and \
               self.current_context.map_context.features.get('show_toc', True):
                html_parts.append(self._generate_toc())

            # Add content in order
            for content in content_list:
                html_parts.append(
                    f'<div class="topic-section" data-topic-id="{content.element_id}">'
                    f'{content.html}'
                    f'</div>'
                )

            html_parts.append('</div>')

            # Process final HTML
            final_html = self.html.process_final_content('\n'.join(html_parts))

            return final_html

        except Exception as e:
            self.logger.error(f"Error assembling final HTML: {str(e)}")
            raise


    def _process_map_title(self, title: str) -> str:
        """Process map title element."""
        try:
            return (
                f'<div class="map-title">'
                f'<h1 class="title display-4 mb-4">{title}</h1>'
                f'</div>'
            )
        except Exception as e:
            self.logger.error(f"Error processing map title: {str(e)}")
            return ""

    def reset_processor_state(self) -> None:
            """Reset all stateful components for new processing."""
            try:
                # Reset handlers
                self.heading_handler.reset()
                self.id_handler = DITAIDHandler()  # Reinitialize ID handler

                # Reset processing context
                self.current_context = None

                # Clear processing caches
                self.processed_maps.clear()
                self.processed_topics.clear()

                self.logger.debug("Reset processor state completed")

            except Exception as e:
                self.logger.error(f"Error resetting processor state: {str(e)}")


    ##########################
    # Topic level processing #
    ##########################

    def _discover_topic(self, topicref: etree._Element, map_path: Path) -> Optional[DiscoveredTopic]:
        """
        Discover and validate a single topic reference.

        Args:
            topicref: XML element containing the topic reference
            map_path: Path to parent map for resolution

        Returns:
            DiscoveredTopic if valid, None if topic should be skipped
        """
        try:
            href = topicref.get('href')
            if not href:
                self.logger.warning("Found topicref without href - skipping")
                return None

            # Resolve topic path
            topic_path = (map_path.parent / href).resolve()
            if not topic_path.exists():
                self.logger.warning(f"Topic file not found: {topic_path}")
                return None

            # Determine content type
            element_type = (
                ElementType.MARKDOWN
                if topic_path.suffix == '.md'
                else ElementType.DITA
            )

            # Generate topic ID
            topic_id = self.id_handler.generate_topic_id(topic_path, map_path)

            # Extract topic metadata
            topic_metadata = self.metadata_handler.extract_metadata(
                topic_path,
                topic_id
            )

            self.logger.debug(
                f"Discovered topic: {topic_id} ({element_type}) "
                f"at path: {topic_path}"
            )

            return DiscoveredTopic(
                id=topic_id,
                path=topic_path,
                type=element_type,
                href=href,
                metadata=topic_metadata
            )

        except Exception as e:
            self.logger.error(f"Error discovering topic: {str(e)}")
            return None

    def get_topic(self, topic_id: str) -> Optional[Path]:
            """
            Get topic path from ID.

            Args:
                topic_id: Can be:
                    - Simple ID (brownian-motion)
                    - With extension (brownian-motion.md)
                    - With subpath (articles/brownian-motion)
                    - Full ditamap reference (brownian-motion.ditamap)

            Returns:
                Optional[Path]: Path to topic file if found
            """
            try:
                self.logger.info(f"Looking for topic with ID: {topic_id}")

                # Handle .ditamap files first
                if topic_id.endswith('.ditamap'):
                    map_path = self.maps_dir / topic_id
                    if map_path.exists():
                        self.logger.info(f"Found map at: {map_path}")
                        return map_path

                # Clean the topic ID
                base_id = self._clean_topic_id(topic_id)

                # Search in maps directory first
                map_path = self.maps_dir / f"{base_id}.ditamap"
                if map_path.exists():
                    return map_path

                # Search in topics directories
                for subdir in ['articles', 'acoustics', 'audio', 'abstracts', 'journals']:
                    for ext in ['.dita', '.md']:
                        topic_path = self.topics_dir / subdir / f"{base_id}{ext}"
                        if topic_path.exists():
                            return topic_path

                self.logger.error(f"No topic found with ID: {topic_id}")
                return None

            except Exception as e:
                self.logger.error(f"Error getting topic path: {str(e)}")
                return None


    def _clean_topic_id(self, topic_id: str) -> str:
        """Clean and normalize topic ID"""
        # Remove file extensions
        for ext in ['.ditamap', '.dita', '.md']:
            topic_id = topic_id.replace(ext, '')

        # Normalize path separators
        topic_id = topic_id.replace('\\', '/')

        # Remove any leading/trailing separators
        return topic_id.strip('/')

    def _process_topic(self, topic: ParsedElement) -> str | None:
        """
        Process a single topic and return its transformed HTML content.

        Args:
            topic (ParsedElement): The parsed element to process.

        Returns:
            str | None: The transformed HTML content, or None if processing fails.
        """
        try:
            self.logger.debug(f"Processing topic: {topic.topic_id}")

            if topic.type == ElementType.MARKDOWN:
                processed = self.md_transformer.transform_topic(topic.topic_path)
            elif topic.type == ElementType.DITA:
                processed = self.dita_transformer.transform_topic(topic.topic_path)
            else:
                self.logger.warning(f"Unsupported topic type: {topic.type}")
                return None

            # Extract the HTML content from ProcessedContent
            if isinstance(processed, ProcessedContent):
                return processed.html
            return processed

        except Exception as e:
            self.logger.error(f"Error processing topic {topic.topic_id}: {str(e)}")
            return None



    def transform(self, content: ProcessedContent) -> str:
        """
        Transform processed content into its final HTML representation.

        Args:
            content (ProcessedContent): The content object containing HTML, metadata, and element ID.

        Returns:
            str: The fully transformed HTML string.
        """
        try:
            self.logger.debug("Starting transformation of processed content.")

            # Use HTMLHelper to render the HTML
            final_html = self.html_helper.render(content.html)

            self.logger.debug("Content transformation completed successfully.")
            return final_html

        except Exception as e:
            self.logger.error(f"Error transforming content: {str(e)}")
            raise



    def _is_safe_path(self, path: Path) -> bool:
        """
        Verify that the resolved path is within the project directory.
        Prevents directory traversal attacks.
        """
        try:
            # Resolve both paths to absolute paths
            project_root = self.app_root.resolve()
            resolved_path = path.resolve()

            # Check if resolved path is within project directory
            return str(resolved_path).startswith(str(project_root))

        except Exception as e:
            self.logger.error(f"Error checking path safety: {str(e)}")
            return False



    ############################
    # Element level processing #
    ############################

    def _track_elements(self, discovered_map: DiscoveredMap) -> List[TrackedElement]:
        """
        Create tracked elements from discovered content.

        Args:
            discovered_map: The discovered content to track

        Returns:
            List of tracked elements ready for processing
        """
        tracked = []
        try:
            # Tracks map title if present
            if discovered_map.title:
                tracked.append(TrackedElement(
                    id=discovered_map.id,
                    type=ElementType.MAP_TITLE,
                    path=discovered_map.path,
                    content=discovered_map.title,
                    metadata=discovered_map.metadata,
                    state=ProcessingState.PENDING
                ))

            # Track each topic
            for topic in discovered_map.topics:
                tracked.append(TrackedElement(
                    id=topic.id,
                    type=topic.type,
                    path=topic.path,  # Topic's actual path
                    content=str(topic.path),  # Stores path as string
                    metadata=topic.metadata,
                    state=ProcessingState.PENDING
                ))

            self.logger.debug(f"Created {len(tracked)} tracked elements")
            return tracked

        except Exception as e:
            self.logger.error(f"Error tracking elements: {str(e)}")
            raise

    def _process_elements(self, elements: List[TrackedElement]) -> List[ProcessedContent]:
        """Process elements using content routing system."""
        try:
            if not self._validate_context():
                raise ProcessingError(
                    error_type="processing",
                    message="Invalid processing context",
                    context="element_processing"
                )

            processed = []

            for element in elements:
                try:
                    # Route content through new routing system
                    processed_content = self._route_content(element)
                    processed.append(processed_content)

                except ProcessingError:
                    raise

            return processed

        except Exception as e:
            self.logger.error(f"Element processing failed: {str(e)}")
            raise

    def _get_element_topic_id(self, element: TrackedElement) -> Optional[str]:
        """Get topic ID for a TrackedElement."""
        if self.current_context and self.current_context.current_topic_id:
            return self.current_context.current_topic_id
        return None

    def _get_content_topic_id(self, content: ProcessedContent) -> Optional[str]:
        """Get topic ID for ProcessedContent."""
        if self.current_context and self.current_context.current_topic_id:
            return self.current_context.current_topic_id
        return None

    def _track_elements_with_context(self, elements: List[ParsedElement]) -> List[TrackedElement]:
        """Track elements with context information"""
        try:
            if not self.current_context:
                raise ValueError("No processing context available")

            tracked = []
            for element in elements:
                element_id = self.id_handler.generate_content_id(
                    Path(element.source_path)
                )

                # Update current topic ID in context if this is a topic element
                if element.type in [ElementType.DITA, ElementType.MARKDOWN]:
                    self.current_context.current_topic_id = element_id

                tracked.append(TrackedElement(
                    id=element_id,
                    type=element.type,
                    path=Path(element.source_path),
                    content=element.content,
                    metadata=element.metadata,
                    state=ProcessingState.PENDING
                ))

            return tracked

        except Exception as e:
            self.logger.error(f"Error tracking elements: {str(e)}")
            return []

    def _update_element_state(self, element: TrackedElement, state: ProcessingState) -> None:
        """
        Update processing state of tracked element.

        Args:
            element: The element to update
            state: New processing state
        """
        try:
            element.state = state
            self.logger.debug(f"Updated element {element.id} state to {state.value}")
        except Exception as e:
            self.logger.error(f"Error updating element state: {str(e)}")
            raise

    def _validate_element_state(self, element: TrackedElement) -> bool:
        """
        Validate if element is in valid state for processing.

        Args:
            element: Element to validate

        Returns:
            True if element can be processed
        """
        try:
            # Check if element should be processed based on context
            if not self.current_context:
                return True  # Default to processing if no context

            # Get topic features if this is a topic element
            if element.type in [ElementType.DITA, ElementType.MARKDOWN]:
                topic_context = self.current_context.topics.get(element.id)
                if topic_context:
                    # Check topic-level conditions
                    if not topic_context.features.get('show_topic', True):
                        return False

            # Map-level conditions from context
            map_context = self.current_context.map_context
            if not map_context:
                return True

            # Map-level validation goes here

            return True

        except Exception as e:
            self.logger.error(f"Error validating element state: {str(e)}")
            return False


    #################################################
    # Conditionally processed elements and features #
    #################################################

    def _filter_processed_content(self, content: List[ProcessedContent]) -> List[ProcessedContent]:
        """Filter out duplicate content."""
        seen = set()
        filtered = []

        for item in content:
            # Create a unique key for the content
            content_key = f"{item.element_id}:{item.html}"
            if content_key not in seen:
                seen.add(content_key)
                filtered.append(item)

        return filtered

    def _generate_topic_title(self, topic_context: TopicContext) -> str:
        """Generate topic title HTML"""
        title = topic_context.metadata.get('title', '')
        return (
            f'<h1 id="{topic_context.topic_id}" class="topic-title">'
            f'{title}'
            f'<a href="#{topic_context.topic_id}" class="heading-anchor">¶</a>'
            f'</h1>'
        )

    def _process_elements_with_context(self, elements: List[ParsedElement]) -> List[TrackedElement]:
        """Track elements with context information."""
        try:
            if not self.current_context:
                raise ValueError("No processing context available")

            tracked = []
            for element in elements:
                element_id = self.id_handler.generate_content_id(
                    Path(element.source_path)
                )

                # Update current topic ID in context if this is a topic element
                if element.type in [ElementType.DITA, ElementType.MARKDOWN]:
                    self.current_context.current_topic_id = element_id

                    topic_context = self._get_topic_context(element_id)
                    if topic_context:
                        tracked.append(TrackedElement(
                            id=element_id,
                            type=element.type,
                            path=Path(element.source_path),
                            content=element.content,
                            metadata=element.metadata,
                            state=ProcessingState.PENDING
                        ))

            return tracked

        except Exception as e:
            self.logger.error(f"Error tracking elements: {str(e)}")
            return []

    def _detect_features(self, content: str) -> ContentFeatures:
        """
        Detect features in content.

        Args:
            content: Content to analyze

        Returns:
            Detected features
        """
        try:
            features = ContentFeatures()

            # Detect LaTeX
            features.has_latex = '$$' in content or '$' in content

            # Detect code blocks
            features.has_code = '<pre>' in content or '<code>' in content

            # Detect tables
            features.has_tables = '<table' in content

            # Detect images
            features.has_images = '<img' in content

            # Detect cross-references
            features.has_xrefs = 'href=' in content

            # Detect artifacts
            # TODO: Implement artifact detection once component is ready
            features.has_artifacts = False  # Placeholder

            self.logger.debug(f"Detected features: {features}")
            return features

        except Exception as e:
            self.logger.error(f"Feature detection failed: {str(e)}")
            return ContentFeatures()

    def _generate_toc(self) -> str:
        """
        Generate table of contents.
        TODO: Implement proper TOC generation
        """
        try:
            if not self.current_context:
                return ''

            # Placeholder for TOC
            toc_parts = ['<nav class="dita-toc">']
            toc_parts.append('<div class="toc-placeholder">Table of Contents</div>')
            toc_parts.append('</nav>')

            return '\n'.join(toc_parts)

        except Exception as e:
            self.logger.error(f"TOC generation failed: {str(e)}")
            return ''

    def _process_artifacts(self, content: str) -> str:
        """
        Process artifacts in content.
        TODO: Implement proper artifact processing
        """
        try:
            # Placeholder for artifact processing
            return content

        except Exception as e:
            self.logger.error(f"Artifact processing failed: {str(e)}")
            return content

    def _update_processing_options(self, features: ContentFeatures) -> None:
        """
        Update processing options based on detected features.

        Args:
            features: Detected content features
        """
        try:
            if not self.current_context:
                return

            processing = ProcessingFeatures()

            # Update based on content features
            processing.needs_latex = features.has_latex
            processing.needs_artifacts = features.has_artifacts

            # TODO: Add more feature-based processing options

            # Update context features
            self.current_context.map_context.features.update({
                'process_latex': processing.needs_latex,
                'process_artifacts': processing.needs_artifacts,
                'show_toc': processing.needs_toc,
                'number_headings': processing.needs_heading_numbers
            })

        except Exception as e:
            self.logger.error(f"Processing options update failed: {str(e)}")


    #######################
    # Pipeline Management #
    #######################

    # Configuration
    def configure_processor(self, config: DITAConfig) -> None:
        """Configure processor with provided settings."""
        try:
            self.logger.debug("Configuring processor")

            # Configure paths
            self.dita_root = config.paths.dita_root
            self.maps_dir = config.paths.maps_dir
            self.topics_dir = config.paths.topics_dir

            # Convert DITAProcessingConfig to ProcessingOptions
            self.processing_options = ProcessingOptions(
                process_latex=config.processing.process_latex,
                number_headings=config.processing.number_headings,
                enable_cross_refs=config.processing.enable_cross_refs,
                process_artifacts=config.processing.process_artifacts,
                show_toc=config.processing.show_toc
            )

            # Configure components
            self._configure_transformers(config)
            self._configure_handlers(config)

            self.logger.debug("Processor configuration completed")

        except Exception as e:
            self.logger.error(f"Processor configuration failed: {str(e)}")
            raise

    def _configure_transformers(self, config: DITAConfig) -> None:
        """Configure all transformers."""
        try:
            self.logger.debug("Configuring transformers")

            # Configure DITA transformer
            if hasattr(self, 'dita_transformer'):
                self.dita_transformer.configure(config)

            # Configure Markdown transformer
            if hasattr(self, 'md_transformer'):
                self.md_transformer.configure(config)

            self.logger.debug("Transformers configuration completed")

        except Exception as e:
            self.logger.error(f"Transformer configuration failed: {str(e)}")
            raise

    def _configure_handlers(self, config: DITAConfig) -> None:
        """Configure all handlers."""
        try:
            self.logger.debug("Configuring handlers")

            # Configure each handler
            self.heading_handler.configure(config)
            self.id_handler.configure(config)
            self.html.configure_helper(config)
            self.metadata_handler.configure(config)

            self.logger.debug("Handlers configuration completed")

        except Exception as e:
            self.logger.error(f"Handlers configuration failed: {str(e)}")
            raise

    # Pipeline init

    def run_discovery_phase(self, map_path: Path) -> DiscoveredMap:
        """Initial content discovery phase."""
        try:
            self.logger.info(f"Starting discovery phase for map: {map_path}")
            self.transition_phase(ProcessingPhase.DISCOVERY)

            # Generate map ID
            map_id = self.id_handler.generate_map_id(map_path)
            self.logger.debug(f"Generated map ID: {map_id}")

            # Parse map file
            self.logger.debug("Calling dita_parser.parse_map")
            tree = self.dita_parser.parse_file(map_path)
            if tree is None:
                self.logger.error("parse_file returned None")
                raise ProcessingError(
                    error_type="discovery",
                    message="Failed to parse map file",
                    context=map_path
                )
            self.logger.debug("Successfully parsed map file")

            # Extract map metadata
            map_metadata = self.metadata_handler.extract_metadata(map_path, map_id)

            # Get map title
            title_elem = tree.find(".//title")
            map_title = title_elem.text if title_elem is not None else None

            # Discover topics
            discovered_topics = []
            topic_refs = tree.xpath(".//topicref")

            self.logger.debug(f"Found {len(topic_refs)} topic references")

            for topicref in topic_refs:
                topic = self._discover_topic(topicref, map_path)
                if topic:
                    discovered_topics.append(topic)

            discovered_map = DiscoveredMap(
                id=map_id,
                path=map_path,
                title=map_title,
                topics=discovered_topics,
                metadata=map_metadata
            )

            return discovered_map

        except Exception as e:
            self.logger.error(f"Discovery phase failed: {str(e)}")
            raise


    def execute_pipeline(self, discovered_map: DiscoveredMap) -> List[str]:
        """
        Execute the processing pipeline for a discovered map.

        Args:
            discovered_map: The DiscoveredMap object containing parsed topics.

        Returns:
            List[str]: The transformed HTML content for all topics in the map.
        """
        self.logger.debug("Executing pipeline.")

        try:
            # Initialize the map context
            map_context = self._init_map_context(discovered_map)
            self.logger.debug(f"Initialized map context: {map_context}")

            # Parse topics into ParsedElement objects
            parsed_elements = [
                ParsedElement(
                    id=topic.id,
                    topic_id=topic.id,
                    type=topic.type,
                    content=str(topic.path),  # Placeholder for content
                    topic_path=topic.path,
                    source_path=discovered_map.path,
                    metadata=topic.metadata
                )
                for topic in discovered_map.topics
            ]
            self.logger.debug(f"Parsed elements: {parsed_elements}")

            # Run the transformation phase
            transformed_contents = self.run_transformation_phase(parsed_elements)
            self.logger.debug(f"Transformed contents: {transformed_contents}")

            # Collect and return transformed HTML content
            html_outputs = [content.html for content in transformed_contents if content.html]
            self.logger.debug(f"Pipeline completed with {len(html_outputs)} outputs.")

            return html_outputs

        except Exception as e:
            self.logger.error(f"Error executing pipeline: {str(e)}", exc_info=True)
            raise




    # Phase handlers

    def run_validation_phase(self, discovered_map: DiscoveredMap) -> bool:
        """Validate discovered content."""
        try:
            self.transition_phase(ProcessingPhase.VALIDATION)

            # Initialize context
            self.initialize_context(discovered_map)

            # Validate context
            if not self._validate_context():
                return False

            # Validate map
            if not discovered_map.path.exists():
                self.logger.error(f"Map file not found: {discovered_map.path}")
                return False

            # Validate topics
            for topic in discovered_map.topics:
                if not topic.path.exists():
                    self.logger.error(f"Topic file not found: {topic.path}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Validation phase failed: {str(e)}")
            return False

    def run_transformation_phase(self, elements: List[ParsedElement]) -> List[ProcessedContent]:
        """Run the transformation phase for a list of parsed elements."""
        self.logger.debug("Starting transformation phase.")

        transformed_content = []
        for element in elements:
            self.logger.debug(f"Processing element: {element}")

            transformer = (
                self.md_transformer if element.type == ElementType.MARKDOWN else self.dita_transformer
            )

            try:
                processed = transformer.transform_topic(element.topic_path)
                transformed_content.append(processed)
                self.logger.debug(f"Successfully transformed element: {element.id}")

            except Exception as e:
                self.logger.error(f"Error transforming element {element.id}: {e}")

        self.logger.debug(f"Transformation phase completed for {len(transformed_content)} elements.")
        return transformed_content



    def run_enrichment_phase(self, content: List[ProcessedContent]) -> List[ProcessedContent]:
       """
       Enrich content with artifacts.

       Args:
           content: Content to enrich

       Returns:
           Enriched content
       """
       try:
           self.transition_phase(ProcessingPhase.ENRICHMENT)
           enriched_content = []

           # for item in content:
           #     if item.metadata.get('process_artifacts', True):
           #         if hasattr(self, 'artifact_renderer'):
           #             item.html = self.artifact_renderer.render_artifact(
           #                 item.html,
           #                 item.element_id
           #             )
           #     enriched_content.append(item)

           return enriched_content

       except Exception as e:
           self.logger.error(f"Enrichment phase failed: {str(e)}")
           raise

    def run_assembly_phase(self, processed_content):
        try:
            self.logger.debug("Starting assembly phase")
            seen_ids = set()
            unique_content = []

            for content in processed_content:
                if content.element_id not in seen_ids:
                    seen_ids.add(content.element_id)
                    unique_content.append(content)

            final_html = ''.join([content.html for content in unique_content])
            self.logger.debug(f"Assembly completed with {len(unique_content)} unique topics")
            return final_html
        except Exception as e:
            self.logger.error(f"Assembly phase failed: {str(e)}")
            raise




    def _initialize_assembly(self) -> None:
        """Initialize assembly phase state and resources."""
        try:
            self.logger.debug("Initializing assembly phase")
            if not self._validate_context():
                raise ProcessingError(
                    error_type="assembly",
                    message="Invalid context for assembly initialization",
                    context="assembly_phase"
                )

            # Reset any existing assembly state
            self._init_state()

        except Exception as e:
            self.logger.error(f"Assembly initialization failed: {str(e)}")
            raise

    def _assemble_content(self, content_list: List[ProcessedContent]) -> str:
        """
        Assemble all processed content into final HTML structure.

        Args:
            content_list: List of processed content to assemble.

        Returns:
            Assembled HTML content.
        """
        try:
            if not self.current_context:
                raise ProcessingError(
                    error_type="assembly",
                    message="No context available for assembly",
                    context="content_assembly"
                )

            self.logger.debug("Starting content assembly.")
            html_parts = ['<div class="dita-content">']

            # Process content in topic order
            for topic_id in self.current_context.map_context.topic_order:
                self.logger.debug(f"Assembling content for topic ID: {topic_id}")

                topic_content = next(
                    (c for c in content_list if c.element_id == topic_id),
                    None
                )

                if topic_content:
                    assembled_topic = self._assemble_topic_content(topic_content)
                    html_parts.append(assembled_topic)
                else:
                    self.logger.warning(f"No content found for topic ID: {topic_id}")

            html_parts.append('</div>')

            # Process final HTML
            final_html = self.html_helper.process_final_content('\n'.join(html_parts))
            self.logger.debug("Content assembly completed successfully.")
            return final_html

        except Exception as e:
            self.logger.error(f"Content assembly failed: {str(e)}", exc_info=True)
            raise


    def _assemble_topic_content(self, topic_content: ProcessedContent) -> str:
        """
        Assemble topic level content.

        Args:
            topic_content: Processed topic content

        Returns:
            Assembled topic HTML
        """
        try:
            topic_classes = ['topic-section']
            if topic_metadata := topic_content.metadata:
                if topic_type := topic_metadata.get('topic_type'):
                    topic_classes.append(f'topic-type-{topic_type}')

            return self.html.wrap_content(
                topic_content.html,
                'div',
                [*topic_classes, f'topic-{topic_content.element_id}']
            )

        except Exception as e:
            self.logger.error(f"Topic assembly failed: {str(e)}")
            return ""

    def _assemble_section_content(self, section_content: ProcessedContent) -> str:
        """
        Assemble section level content.

        Args:
            section_content: Processed section content

        Returns:
            Assembled section HTML
        """
        try:
            return self.html.wrap_content(
                section_content.html,
                'section',
                ['dita-section', f'section-{section_content.element_id}']
            )

        except Exception as e:
            self.logger.error(f"Section assembly failed: {str(e)}")
            return ""


    # Pipeline orchestration
    def validate_phase(self, phase: ProcessingPhase) -> bool:
        """
        Validate if pipeline can transition to phase.

        Args:
            phase: Phase to validate

        Returns:
            True if phase transition is valid
        """
        try:
            if not hasattr(self, 'current_phase'):
                self.current_phase = ProcessingPhase.DISCOVERY
                return True

            # Define valid phase transitions
            valid_transitions = {
                ProcessingPhase.DISCOVERY: [ProcessingPhase.VALIDATION, ProcessingPhase.TRANSFORMATION],
                ProcessingPhase.VALIDATION: [ProcessingPhase.TRANSFORMATION],
                ProcessingPhase.TRANSFORMATION: [ProcessingPhase.ENRICHMENT, ProcessingPhase.ASSEMBLY],
                ProcessingPhase.ENRICHMENT: [ProcessingPhase.ASSEMBLY],
                ProcessingPhase.ASSEMBLY: []  # Terminal phase
            }

            return phase in valid_transitions.get(self.current_phase, [])

        except Exception as e:
            self.logger.error(f"Phase validation failed: {str(e)}")
            return False

    def transition_phase(self, new_phase: ProcessingPhase) -> None:
        """
        Transition pipeline to new phase.

        Args:
            new_phase: Phase to transition to
        """
        try:
            if not self.validate_phase(new_phase):
                self.logger.warning(
                    f"Invalid phase transition from {self.current_phase.value if hasattr(self, 'current_phase') else 'None'} "
                    f"to {new_phase.value}"
                )
                # Instead of raising an error, we'll set the phase and continue

            self.current_phase = new_phase
            self.logger.info(f"Pipeline transitioned to {new_phase.value} phase")

        except Exception as e:
            self.logger.error(f"Phase transition failed: {str(e)}")
            raise



    ######################
    # Heading processing #
    ######################

    def _init_heading_state(self) -> None:
        """Initialize heading state for new processing."""
        try:
            self.heading_handler.init_state()
            self.logger.debug("Initialized heading state")
        except Exception as e:
            self.logger.error(f"Failed to initialize heading state: {str(e)}")
            raise

    def _manage_heading_state(self, element_type: ElementType) -> None:
        """
        Manage heading state based on content type.

        Args:
            element_type: Type of content being processed
        """
        try:
            if element_type in [ElementType.DITA, ElementType.MARKDOWN]:
                # Save current state before processing new topic
                self.heading_handler.save_state()

                # Reset section counters but maintain H1
                self.heading_handler.reset_section()

            self.logger.debug(f"Updated heading state for {element_type.value}")
        except Exception as e:
            self.logger.error(f"Failed to manage heading state: {str(e)}")
            raise

    def _validate_heading_state(self) -> bool:
        """
        Validate current heading state.

        Returns:
            True if heading state is valid
        """
        try:
            # Check if we have a current context
            if not self.current_context:
                self.logger.warning("No context available for heading validation")
                return False

            # Validate topic order matches heading hierarchy
            topic_order = self.current_context.map_context.topic_order
            current_h1 = self.heading_handler._state.current_h1

            if len(topic_order) != current_h1:
                self.logger.warning(
                    f"Heading count mismatch: {current_h1} H1s "
                    f"for {len(topic_order)} topics"
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"Failed to validate heading state: {str(e)}")
            return False



    #######################
    # Final HTML assembly #
    #######################

    def _assemble_html(self, processed_contents: List[ProcessedContent]) -> str:
        """
        Assemble final HTML with proper heading structure.

        Args:
            processed_contents: List of processed content to assemble

        Returns:
            Assembled HTML content
        """
        try:
            if not self.current_context:
                raise ValueError("No processing context available")

            html_parts = ['<div class="dita-content">']

            # Process content in topic order
            for topic_id in self.current_context.map_context.topic_order:
                topic_content = next(
                    (content for content in processed_contents
                    if content.element_id == topic_id),
                    None
                )

                if topic_content:
                    # Create topic container
                    html_parts.append(
                        f'<div class="topic-section" '
                        f'data-topic-id="{topic_id}">'
                    )

                    # Add processed content
                    html_parts.append(topic_content.html)
                    html_parts.append('</div>')
                else:
                    self.logger.warning(f"Missing content for topic: {topic_id}")

            html_parts.append('</div>')

            # Validate final heading state
            if not self._validate_heading_state():
                self.logger.warning("Final heading validation failed")

            # Process final HTML
            final_html = self.html.process_final_content('\n'.join(html_parts))

            return final_html

        except Exception as e:
            self.logger.error(f"Error assembling HTML: {str(e)}")
            raise ProcessingError(
                error_type="assembly",
                message=f"HTML assembly failed: {str(e)}",
                context="html_assembly"
            )


    def _assemble_html_with_context(self, processed_contents: List[ProcessedContent]) -> str:
        """Assemble HTML with context-aware features"""
        try:
            if not self.current_context:
                raise ValueError("No processing context available")

            map_context = self.current_context.map_context

            html_parts = [
                f'<div class="dita-content" '
                f'data-map-id="{map_context.map_id}">'
            ]

            # Add map-level features
            if map_context.features['show_toc']:
                html_parts.append(self._generate_toc())

            # Process contents in topic order
            for topic_id in map_context.topic_order:
                topic_contents = [
                    content for content in processed_contents
                    if self._get_content_topic_id(content) == topic_id  # Use new method
                ]

                topic_context = self.current_context.topics[topic_id]
                if topic_context.features['show_title']:
                    html_parts.append(self._generate_topic_title(topic_context))

                for content in topic_contents:
                    html_parts.append(content.html)

            html_parts.append('</div>')

            # Apply final HTML processing
            final_html = self.html.process_final_content('\n'.join(html_parts))

            return final_html

        except Exception as e:
            self.logger.error(f"Error assembling HTML: {str(e)}")
            return self._create_error_html(str(e), Path("HTML Assembly"))


    #################################
    # Transformer Routes Management #
    #################################


    def _route_content(self, element: TrackedElement) -> ProcessedContent:
        """Route content to appropriate transformer."""
        try:
            if element.type == ElementType.MAP_TITLE:
                # Handle map title directly
                content = self._process_map_title(element.content)
                return ProcessedContent(
                    html=content,
                    element_id=element.id,
                    metadata=element.metadata
                )

            # Get transformer for other content types
            transformer = self._get_transformer(element.type)
            if not transformer:
                raise ProcessingError(
                    error_type="routing",
                    message=f"No transformer found for type: {element.type.value}",
                    element_id=element.id,
                    context=element.path
                )

            # Transform content
            if element.type in [ElementType.DITA, ElementType.MARKDOWN]:
                content = transformer.transform_topic(element.path)
            else:
                raise ProcessingError(
                    error_type="routing",
                    message=f"Unsupported content type: {element.type.value}",
                    element_id=element.id,
                    context=element.path
                )

            return ProcessedContent(
                html=content,
                element_id=element.id,
                metadata=element.metadata
            )

        except Exception as e:
            self.logger.error(f"Content routing failed: {str(e)}")
            raise

    def _get_transformer(self, element_type: ElementType) -> Any:
        """
        Get appropriate transformer for content type.

        Args:
            element_type: Type of content

        Returns:
            Transformer instance
        """
        try:
            transformers = {
                ElementType.DITA: self.dita_transformer,
                ElementType.MARKDOWN: self.md_transformer
            }
            return transformers.get(element_type)

        except Exception as e:
            self.logger.error(f"Error getting transformer: {str(e)}")
            return None

    def _validate_transformer(self, transformer: Any, element_type: ElementType) -> bool:
        """
        Validate transformer availability and compatibility.

        Args:
            transformer: Transformer instance to validate
            element_type: Expected content type

        Returns:
            True if transformer is valid
        """
        try:
            if not transformer:
                return False

            # Validate transformer has required method
            if not hasattr(transformer, 'transform_topic'):
                self.logger.error(
                    f"Transformer for {element_type.value} missing transform_topic method"
                )
                return False

            # Additional validation could be added here
            # e.g., checking for specific transformer capabilities

            return True

        except Exception as e:
            self.logger.error(f"Transformer validation failed: {str(e)}")
            return False




    ################################
    # Error handling and Debugging #
    ################################

    def _handle_processing_error(self,
                               error: Union[ProcessingError, Exception],
                               phase: ProcessingPhase,
                               context: Optional[Dict[str, Any]] = None) -> None:
        """
        Centralized error handling for processing pipeline.

        Args:
            error: The error that occurred
            phase: Processing phase where error occurred
            context: Additional context information
        """
        try:
            # Create log context
            log_context = LogContext(
                phase=phase,
                map_id=self.current_context.map_context.map_id if self.current_context else None,
                topic_id=self.current_context.current_topic_id if self.current_context else None
            )

            # If it's already a ProcessingError, log it directly
            if isinstance(error, ProcessingError):
                self.logger.error(
                    f"Processing error in {phase.value}: {error.message}\n"
                    f"Context: {error.context}"
                )
            else:
                self.logger.error(
                    f"Error during {phase.value}: {str(error)}\n"
                    f"Context: {context}"
                )
        except Exception as e:
            self.logger.error(f"Error in error handler: {str(e)}")

    def _log_phase_transition(self,
                             from_phase: ProcessingPhase,
                             to_phase: ProcessingPhase) -> None:
        """
        Log phase transitions with state information.

        Args:
            from_phase: Current phase
            to_phase: Target phase
        """
        try:
            debug_info = self.get_debug_info()

            log_context = LogContext(
                phase=to_phase,
                map_id=self.current_context.map_context.map_id if self.current_context else None,
                topic_id=self.current_context.current_topic_id if self.current_context else None
            )

            self.logger.debug(
                        f"Phase transition: {from_phase.value} -> {to_phase.value}\n"
                        f"Debug info: {debug_info}"
                    )

        except Exception as e:
            self.logger.error(f"Failed to log phase transition: {str(e)}")

    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get current debug information about processor state.

        Returns:
            Dictionary containing debug information
        """
        try:
            debug_info = {
                'has_context': self.current_context is not None,
                'current_phase': self.current_phase.value if hasattr(self, 'current_phase') else None,
                'heading_state': {
                    'current_h1': self.heading_handler._state.current_h1,
                    'counters': self.heading_handler._state.counters.copy()
                }
            }

            if self.current_context:
                debug_info.update({
                    'map_id': self.current_context.map_context.map_id,
                    'current_topic': self.current_context.current_topic_id,
                    'topic_count': len(self.current_context.topics),
                    'features': self.current_context.map_context.features
                })

            return debug_info

        except Exception as e:
            self.logger.error(f"Failed to get debug info: {str(e)}")
            return {'error': str(e)}


    def _create_error_html(self, error: str, context: Union[str, Path]) -> str:
        """
        @deprecated Must use generate_error_response instead.
        """
        self.logger.warning("Using deprecated _create_error_html method")
        return self.generate_error_response(
            ProcessingError(
                error_type="processing",
                message=str(error),
                context=str(context)
            )
        )

    def generate_error_response(self, error: ProcessingError) -> str:
        """
        Generate standardized HTML error response.

        Args:
            error: Processing error to display

        Returns:
            HTML error response
        """
        try:
            error_classes = {
                'discovery': 'bg-yellow-50 border-yellow-500 text-yellow-700',
                'validation': 'bg-orange-50 border-orange-500 text-orange-700',
                'transformation': 'bg-red-50 border-red-500 text-red-700',
                'assembly': 'bg-purple-50 border-purple-500 text-purple-700'
            }

            css_class = error_classes.get(error.error_type, 'bg-red-50 border-red-500 text-red-700')

            html = f"""
            <div class="error-container p-4 {css_class} border-l-4 rounded-lg my-4">
                <div class="flex items-center mb-2">
                    <h3 class="font-bold text-lg">{error.error_type.title()} Error</h3>
                </div>
                <p class="mb-2">{error.message}</p>
                <div class="text-sm">
                    <p>Location: {error.context}</p>
                    {f'<p>Element: {error.element_id}</p>' if error.element_id else ''}
                </div>

                {(f'<details class="mt-4 text-sm">'
                  f'<summary>Technical Details</summary>'
                  f'<pre class="mt-2 p-2 bg-gray-50 rounded">{error.stacktrace}</pre>'
                  f'</details>') if error.stacktrace else ''}
            </div>
            """

            return html

        except Exception as e:
            self.logger.error(f"Failed to generate error response: {str(e)}")
            return f"""
            <div class="error-container p-4 bg-red-50 border-l-4 border-red-500 text-red-700">
                <h3 class="font-bold">Error</h3>
                <p>Failed to generate error response</p>
            </div>
            """


    ################################
    # Cleanup and Waste Management #
    ################################

    def cleanup(self) -> None:
        """Perform comprehensive cleanup of processor resources and states."""
        try:
            self.logger.info("Starting processor cleanup")

            # Clean up in order
            self._cleanup_state()
            self._cleanup_resources()
            self._cleanup_handlers()
            self._reset_transformers()

            self.logger.info("Processor cleanup completed")

        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")
            raise

    def _cleanup_state(self) -> None:
        """Clean up processor state."""
        try:
            # Reset processing context
            self.current_context = None

            # Clear processed content
            self.processed_maps.clear()
            self.processed_topics.clear()

            # Reset phase state
            if hasattr(self, 'current_phase'):
                delattr(self, 'current_phase')

            self.logger.debug("State cleanup completed")

        except Exception as e:
            self.logger.error(f"State cleanup failed: {str(e)}")
            raise

    def _cleanup_resources(self) -> None:
        """Clean up processor resources."""
        try:
            # Clear any cached data
            if hasattr(self, '_cache'):
                self._cache.clear()

            # Reset any file handles or resources
            # Currently no direct file handles, but placeholder for future use

            self.logger.debug("Resource cleanup completed")

        except Exception as e:
            self.logger.error(f"Resource cleanup failed: {str(e)}")
            raise

    def _cleanup_handlers(self) -> None:
        """Clean up and reset all handlers."""
        try:
            # Reset heading handler
            self.heading_handler.reset()

            # Reset ID handler
            self.id_handler = DITAIDHandler()

            # Reset HTML helper
            self.html = HTMLHelper(self.dita_root)

            # Reset metadata handler
            self.metadata_handler = MetadataHandler()

            self.logger.debug("Handler cleanup completed")

        except Exception as e:
            self.logger.error(f"Handler cleanup failed: {str(e)}")
            raise

    def _reset_transformers(self) -> None:
        """Reset transformers to initial state."""
        try:
            # Reset DITA transformer
            if hasattr(self, 'dita_transformer'):
                self.dita_transformer = DITATransformer(self.dita_root)

            # Reset Markdown transformer
            if hasattr(self, 'md_transformer'):
                self.md_transformer = MarkdownTransformer(self.dita_root)

            self.logger.debug("Transformer reset completed")

        except Exception as e:
            self.logger.error(f"Transformer reset failed: {str(e)}")
            raise
