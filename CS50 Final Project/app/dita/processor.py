# Standard library imports
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
import html
import traceback
import json
import os
import hashlib
import time

# XML parsing imports
from lxml import etree
import frontmatter

# Global types
from .utils.types import (
    # Enums
    ElementType,
    ProcessingState,

    # Content types
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

    # Artifact types
    ArtifactReference,
    ProcessedArtifact,

    # Reference types
    HeadingReference,
    CrossReference,

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

# Type aliases
HTMLString = str
XMLElement = Any

class DITAProcessor:
    """
    Main orchestrator for DITA content processing.
    Handles parsing, element tracking, and conditional processing pipelines.
    """
    def __init__(self) -> None:
        try:
            # Initialize core components
            self.logger = logging.getLogger(__name__)
            self._init_paths()
            self._init_handlers()
            self._init_processors()
            self.current_context: Optional[ProcessingContext] = None
            self.processed_maps: Dict[str, MapContext] = {}
            self.processed_topics: Dict[str, TopicContext] = {}
            self.processing_options: ProcessingOptions = ProcessingOptions()
            self.logger.info("DITA Processor initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize DITA processor: {str(e)}")
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


    def _init_paths(self) -> None:
        """Initialize critical paths"""
        try:
            # Set up root paths
            self.app_root = Path(__file__).parent.parent
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

    def _init_handlers(self) -> None:
        """Initialize core handlers"""
        self.id_handler = DITAIDHandler()
        self.heading_handler = HeadingHandler()
        self.metadata_handler = MetadataHandler()
        self.html = HTMLHelper()

    def _init_processors(self) -> None:
        """Initialize processing pipelines"""
        # Parser
        self.dita_parser = DITAParser()

        # DITA Pipeline
        self.dita_transformer = DITATransformer(self.dita_root)

        # Markdown Pipeline
        self.md_transformer = MarkdownTransformer(self.dita_root)

        # LaTeX Pipeline
        self.latex_renderer = KaTeXRenderer()

        # Artifact Pipeline
        # self.artifact_parser = ArtifactParser(self.dita_root)
        # self.artifact_renderer = ArtifactRenderer(self.artifacts_dir)



    def _init_map_context(self, map_path: Path) -> MapContext:
            """Initialize context for map processing"""
            map_id = self.id_handler.generate_map_id(map_path)
            metadata = self.metadata_handler.extract_metadata(map_path, map_id)

            return MapContext(
                map_id=map_id,
                map_path=map_path,
                metadata=metadata,
                topic_order=[],
                features=self._get_map_features(metadata)
            )

    def _create_processing_context(
            self,
            map_context: MapContext,
            parsed_elements: List[ParsedElement]
        ) -> ProcessingContext:
            """Create complete processing context"""
            topics: Dict[str, TopicContext] = {}

            for element in parsed_elements:
                if element.type == 'topic':
                    topic_id = self.id_handler.generate_topic_id(
                        Path(element.source_path),
                        map_context.map_path
                    )

                    topics[topic_id] = TopicContext(
                        topic_id=topic_id,
                        topic_path=Path(element.source_path),
                        parent_map_id=map_context.map_id,
                        metadata=element.metadata,
                        features=self._get_topic_features(element.metadata)
                    )

                    map_context.topic_order.append(topic_id)

            return ProcessingContext(
                map_context=map_context,
                topics=topics
            )

    def _get_map_features(self, metadata: Dict[str, Any]) -> Dict[str, bool]:
            """Get map-level processing features"""
            return {
                'process_latex': metadata.get('process_latex', True),
                'number_headings': metadata.get('number_headings', True),
                'show_toc': metadata.get('show_toc', True),
                'enable_cross_refs': metadata.get('enable_cross_refs', True),
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


    ########################
    # Map level processing #
    ########################

    def process_ditamap(self, map_path: Path) -> str:
        """Main entry point for ditamap processing."""
        try:
            self.logger.info(f"Processing DITA map: {map_path}")
            self.reset_processor_state()

            # Initial parse for content discovery
            parsed_map = self.dita_parser.parse_map(map_path)
            if not parsed_map:
                raise ValueError("Failed to parse map")

            # Initialize context before tracking elements
            self.current_context = ProcessingContext(
                map_context=MapContext(
                    map_id=self.id_handler.generate_map_id(map_path),
                    map_path=map_path,
                    metadata=parsed_map.metadata,
                    topic_order=[],
                    features=self._get_map_features(parsed_map.metadata)
                ),
                topics={},
                current_topic_id=None  # Initialize as None
            )

            # Track elements with initialized context
            tracked_elements = self._track_elements_with_context(parsed_map.topics)

            # Process elements
            processed_content = self._process_elements_with_context(tracked_elements)

            return self._assemble_html_with_context(processed_content)

        except Exception as e:
            self.logger.error(f"Error processing map {map_path}: {str(e)}")
            return self._create_error_html(str(e), map_path)

    def _process_map_content(self, parsed_map: ParsedMap) -> str:
            """Process content discovered in map."""
            try:
                # Initialize contexts before using them
                self.current_context = ProcessingContext(
                    map_context=MapContext(
                        map_id=self.id_handler.generate_map_id(parsed_map.source_path),
                        map_path=parsed_map.source_path,
                        metadata=parsed_map.metadata,
                        topic_order=[],
                        features=self._get_map_features(parsed_map.metadata)
                    ),
                    topics={}
                )

                # Process title if exists
                html_parts = []
                if parsed_map.title:
                    html_parts.append(self._process_map_title(parsed_map.title))

                # Process each topic
                for topic in parsed_map.topics:
                    if processed_topic := self._process_topic(topic):
                        html_parts.append(processed_topic)
                        if self.current_context and self.current_context.map_context:
                            self.current_context.map_context.topic_order.append(topic.id)

                # Store processed map
                if self.current_context and self.current_context.map_context:
                    self.processed_maps[self.current_context.map_context.map_id] = \
                        self.current_context.map_context

                # Return combined content
                return self._assemble_html_with_context(html_parts)

            except Exception as e:
                self.logger.error(f"Error processing map content: {str(e)}")
                return self._create_error_html(str(e), parsed_map.source_path)

    def _process_topic(self, topic: ParsedElement) -> Optional[str]:
            """Process individual topic based on its type."""
            try:
                if not self.current_context:
                    raise ValueError("No processing context available")

                # Create topic context
                topic_context = TopicContext(
                    topic_id=topic.id,
                    topic_path=topic.source_path,
                    parent_map_id=self.current_context.map_context.map_id,
                    metadata=topic.metadata,
                    features=self._get_topic_features(topic.metadata),
                    state=ProcessingState.PENDING
                )

                # Store topic context
                self.current_context.topics[topic.id] = topic_context
                self.current_context.current_topic_id = topic.id

                # Route to appropriate processor based on type
                if topic.type == ElementType.DITA:
                    return self.dita_transformer.transform_topic(topic.source_path)
                elif topic.type == ElementType.MARKDOWN:
                    return self.md_transformer.process_topic(topic.source_path)  # Changed method name
                else:
                    raise ValueError(f"Unsupported topic type: {topic.type}")

            except Exception as e:
                self.logger.error(f"Error processing topic {topic.id}: {str(e)}")
                return None

    def _process_map_title(self, title: str) -> str:
        """Process map title."""
        title_id = self.id_handler.generate_content_id(Path(title))
        return (
            f'<div id="{title_id}" class="map-title">'
            f'<div class="title display-4 mb-4">{title}</div>'
            f'</div>'
        )

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


    def transform(self, path: Union[str, Path]) -> str:
        content_path: Optional[Path] = None
        try:
            content_path = Path(path) if isinstance(path, str) else path

            if not self._is_safe_path(content_path):
                raise ValueError(f"Invalid path: {content_path}")

            self.logger.debug(f"Processing file: {content_path}")

            # For ditamaps, we need to process their referenced content
            if content_path.suffix == '.ditamap':
                self.logger.debug("Processing ditamap...")
                # Parse the map to get the actual content file
                tree = self.dita_parser.parse_file(content_path)
                if tree is None:
                    raise ValueError(f"Failed to parse map file: {content_path}")

                # Find the first topicref
                topicref = tree.find(".//topicref")
                if topicref is not None and (href := topicref.get('href')):
                    # Get the content file path
                    content_file = (content_path.parent / href).resolve()
                    self.logger.debug(f"Found content file in map: {content_file}")

                    # Process based on content file type
                    if content_file.suffix == '.md':
                        content = self.md_transformer.transform_topic(content_file)
                    elif content_file.suffix == '.dita':
                        content = self.dita_transformer.transform_topic(content_file)
                    else:
                        raise ValueError(f"Unsupported content file type: {content_file.suffix}")

                    self.logger.debug(f"Processed content length: {len(content)}")
                    return content
                else:
                    raise ValueError("No content reference found in map")

            # Direct file processing
            elif content_path.suffix == '.md':
                content = self.md_transformer.transform_topic(content_path)
                self.logger.debug(f"Markdown transformed content length: {len(content)}")
                return content
            elif content_path.suffix == '.dita':
                content = self.dita_transformer.transform_topic(content_path)
                self.logger.debug(f"DITA transformed content length: {len(content)}")
                return content
            else:
                raise ValueError(f"Unsupported file type: {content_path.suffix}")

        except Exception as e:
            self.logger.error(f"Transformation error: {str(e)}")
            error_path = content_path if content_path else Path(str(path))
            return self._create_error_html(str(e), error_path)

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

    def _track_elements(self, elements: List[ParsedElement]) -> List[TrackedElement]:
        """Track elements and assign IDs"""
        tracked = []
        for element in elements:
            element_id = self.id_handler.generate_content_id(
                Path(element.source_path)
            )
            tracked.append(TrackedElement(
                id=element_id,
                type=element.type,
                content=element.content,
                metadata=element.metadata
            ))
        return tracked

    def _process_elements(self, elements: List[TrackedElement]) -> List[ProcessedContent]:
        """Process elements through appropriate pipelines"""
        processed = []
        for element in elements:
            # Apply processing conditions here
            if self._should_process_element(element):
                processed_content = self._route_to_pipeline(element)
                processed.append(processed_content)
        return processed

    def _should_process_element(self, element: TrackedElement) -> bool:
            """Determine if element should be processed based on context"""
            if not self.current_context:
                return True  # Default to processing if no context

            # Get current topic context if available
            topic_id = self._get_element_topic_id(element)
            if topic_id:
                topic_context = self.current_context.topics.get(topic_id)
                if topic_context:
                    # Check topic-level conditions
                    if element.type == 'latex' and not topic_context.features['process_latex']:
                        return False
                    if element.type == 'artifact' and not topic_context.features['process_artifacts']:
                        return False

            # Check map-level conditions
            map_context = self.current_context.map_context
            if element.type == 'latex' and not map_context.features['process_latex']:
                return False

            return True

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

    def _route_to_pipeline(self, element: TrackedElement) -> ProcessedContent:
        """Route element to appropriate processing pipeline"""
        try:
            element_path = Path(element.content)

            if element.type == ElementType.MARKDOWN:
                html_content = self.md_transformer.transform_topic(element_path)
                return ProcessedContent(
                    html=html_content,
                    element_id=element.id
                )
            elif element.type == ElementType.DITA:
                html_content = self.dita_transformer.transform_topic(element_path)
                return ProcessedContent(
                    html=html_content,
                    element_id=element.id
                )
            elif element.type == ElementType.LATEX:
                html_content = self.latex_renderer.render(element.content)
                return ProcessedContent(
                    html=html_content,
                    element_id=element.id
                )
            else:
                raise ValueError(f"Unknown element type: {element.type}")
        except Exception as e:
            self.logger.error(f"Error processing element {element.id}: {str(e)}")
            return ProcessedContent(
                html=self._create_error_html(str(e), Path(element.id)),
                element_id=element.id
            )

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
                    content=element.content,
                    metadata=element.metadata,
                    state=ProcessingState.PENDING
                ))

            return tracked

        except Exception as e:
            self.logger.error(f"Error tracking elements: {str(e)}")
            return []

    def _assemble_html(self, processed_contents: List[ProcessedContent]) -> str:
        """Assemble final HTML with proper heading structure"""
        try:
            html_parts = ['<div class="dita-content">']

            # Process headings
            for content in processed_contents:
                self.heading_handler.process_content_headings(content)
                html_parts.append(content.html)

            html_parts.append('</div>')

            # Apply final HTML processing
            final_html = self.html.process_final_content('\n'.join(html_parts))

            return final_html

        except Exception as e:
            self.logger.error(f"Error assembling HTML: {str(e)}")
            return self._create_error_html(str(e), "HTML Assembly")



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


    ####################################
    # Conditionally processed elements #
    ####################################

    def _process_elements_with_context(self, elements: List[TrackedElement]) -> List[ProcessedContent]:
        """Process elements with context information"""
        processed = []
        for element in elements:
            if self._should_process_element(element):
                processed_content = self._route_to_pipeline(element)
                processed.append(processed_content)
        return processed

    def _generate_toc(self) -> str:
        """Generate table of contents"""
        if not self.current_context:
            return ''

        toc_parts = ['<nav class="dita-toc">']
        for topic_id in self.current_context.map_context.topic_order:
            topic = self.current_context.topics.get(topic_id)
            if topic and topic.features.get('show_title', True):
                toc_parts.append(
                    f'<div class="toc-entry">'
                    f'<a href="#{topic_id}">{topic.metadata.get("title", "")}</a>'
                    f'</div>'
                )
        toc_parts.append('</nav>')
        return '\n'.join(toc_parts)

    def _generate_topic_title(self, topic_context: TopicContext) -> str:
        """Generate topic title HTML"""
        title = topic_context.metadata.get('title', '')
        return (
            f'<h1 id="{topic_context.topic_id}" class="topic-title">'
            f'{title}'
            f'<a href="#{topic_context.topic_id}" class="heading-anchor">¶</a>'
            f'</h1>'
        )



    ##################
    # Error handling #
    ##################

    def _create_error_html(self, error: str, context: Union[Path, str]) -> str:
        """Create error message HTML"""
        context_str = str(context)
        return f"""
            <div class="error-container bg-red-50 border-l-4 border-red-500 p-4 my-4">
                <h3 class="text-lg font-medium text-red-800">Processing Error</h3>
                <p class="text-red-700">{error}</p>
                <div class="mt-2 text-sm text-red-600">
                    <p>Error in: {context_str}</p>
                </div>
            </div>
        """
