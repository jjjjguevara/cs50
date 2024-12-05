# app/dita/utils/dita_transform.py
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import re
import html
from bs4 import BeautifulSoup, Tag
from lxml import etree

# Local imports
from .html_helpers import HTMLHelper
from .heading import HeadingHandler
from .metadata import MetadataHandler
from .id_handler import DITAIDHandler
from .dita_parser import DITAParser
from .dita_elements import DITAContentProcessor
from ..artifacts.parser import ArtifactParser
from ..artifacts.renderer import ArtifactRenderer

class DITATransformer:
    """Handles DITA content transformation to HTML."""

    def __init__(self, dita_root: Path):
        self.logger = logging.getLogger(__name__)
        self.app_root = Path(__file__).parent.parent
        self.dita_root = dita_root
        self.maps_dir = dita_root / 'maps'
        self.topics_dir = dita_root / 'topics'
        self.artifacts_dir = dita_root / 'artifacts'
        self.dita_root = self.app_root / 'dita'


        # Initialize utilities
        self.html = HTMLHelper()
        self.heading_handler = HeadingHandler()
        self.metadata_handler = MetadataHandler()
        self.id_handler = DITAIDHandler()
        self.dita_parser = DITAParser()
        self.dita_processor = DITAContentProcessor()

        # Initialize artifact handlers
        self.artifact_parser = ArtifactParser(self.dita_root)
        self.artifact_renderer = ArtifactRenderer(self.artifacts_dir)


    def transform_map(self, map_path: Path) -> str:
        try:
            self.logger.debug(f"\n{'='*50}")
            self.logger.debug(f"Starting map transformation for {map_path}")

            # Reset state at start of map
            self.heading_handler.reset()
            self.logger.debug(f"Initial heading state: {self.heading_handler.counters}")

            map_id = self.id_handler.generate_map_id(map_path)

            # Parse artifacts
            artifacts = self._process_map_artifacts(map_path)

            tree = self.dita_parser.parse_file(map_path)
            if tree is None:
                raise ValueError(f"Failed to parse map file: {map_path}")

            html_content = [f'<div class="map-content" data-map-id="{map_id}">']

            # Handle map title
            title_elem = tree.find(".//title")
            if title_elem is not None and title_elem.text:
                title_id = self.id_handler.generate_content_id(Path(title_elem.text))
                html_content.append(
                    f'<div id="{title_id}" class="map-title">'
                    f'<div class="title display-4 mb-4">{title_elem.text}</div>'
                    f'</div>'
                )

            # Process topics
            topic_refs = tree.xpath(".//topicref")
            self.logger.debug(f"Found {len(topic_refs)} topicrefs to process")

            for topicref in topic_refs:
                href = topicref.get('href')
                if not href:
                    continue

                topic_path = Path(map_path).parent / href
                if not topic_path.exists() or not topic_path.suffix == '.dita':
                    continue

                self.logger.debug(f"Before processing topic {href}:")
                self.logger.debug(f"H1 counter: {self.heading_handler.counters['h1']}")
                self.heading_handler.start_new_section()
                self.logger.debug(f"After start_new_section. Counters: {self.heading_handler.counters}")

                topic_id = self.id_handler.generate_topic_id(topic_path, map_path)
                html_content.append(f'<div class="topic-section" data-topic-id="{topic_id}">')
                content = self._transform_dita_topic(topic_path)
                html_content.append(content)
                html_content.append('</div>')

            html_content.append('</div>')
            combined_content = '\n'.join(html_content)

            # Inject artifacts if any
            if artifacts:
                combined_content = self._inject_map_artifacts(combined_content, artifacts)

            return combined_content

        except Exception as e:
            self.logger.error(f"Error transforming map {map_path}: {str(e)}")
            return self._create_error_html(e, map_path)

    def transform_topic(self, topic_path: Path) -> str:
        """Transform single topic to HTML."""
        try:
            self.logger.info(f"Transforming topic: {topic_path}")
            self.heading_handler.reset()
            return self._transform_dita_topic(topic_path)

        except Exception as e:
            self.logger.error(f"Error transforming topic {topic_path}: {str(e)}")
            return self._create_error_html(e, topic_path)

    def _process_map_artifacts(self, map_path: Path) -> List[Dict[str, Any]]:
            """Process and validate map artifacts."""
            artifacts = self.artifact_parser.parse_artifact_references(map_path)
            if artifacts:
                self.logger.info(f"Found artifacts: {artifacts}")
                for artifact in artifacts:
                    if target := artifact.get('target'):
                        self.heading_handler.register_existing_id(target, target)
                        self.logger.debug(f"Registered target heading: {target}")
            return artifacts

    def _generate_map_html(
        self,
        tree: etree._Element,
        map_id: str,
        map_path: Path,
        artifacts: List[Dict[str, Any]]
    ) -> str:
        """Generate HTML content for map."""
        html_content = [f'<div class="map-content" data-map-id="{map_id}">']

        # Process title
        html_content.extend(self._process_map_title(tree))

        # Process topics
        html_content.extend(self._process_map_topics(tree, map_path))

        html_content.append('</div>')
        return '\n'.join(html_content)

    def _process_map_title(self, tree: etree._Element) -> List[str]:
        """Process map title section."""
        title_parts = []
        title_elem = tree.find(".//title")
        if title_elem is not None and title_elem.text:
            # Create a Path-like object for the title
            title_path = Path(title_elem.text)
            title_id = self.id_handler.generate_content_id(title_path)
            title_parts.append(
                f'<div id="{title_id}" class="map-title">'
                f'<h1 class="display-4 mb-4">{title_elem.text}</h1>'
                f'</div>'
            )
        return title_parts

    def _process_map_topics(self, tree: etree._Element, map_path: Path) -> List[str]:
        """Process map topics."""
        topic_parts = []
        for topicref in tree.xpath(".//topicref"):
            href = topicref.get('href')
            if not href:
                continue

            self.logger.info(f"Processing topicref: {href}")
            topic_path = self.resolve_path(map_path, href)

            if not topic_path or not topic_path.exists():
                self.logger.error(f"Could not resolve topic for href: {href}")
                topic_parts.append(self._create_topic_error_message(href))
                continue

            topic_parts.extend(self._process_single_topic(topic_path, map_path))

        return topic_parts

    def _process_single_topic(self, topic_path: Path, map_path: Path) -> List[str]:
        """Process a single topic."""
        topic_parts = []
        topic_id = self.id_handler.generate_topic_id(topic_path, map_path)

        # Extract metadata
        metadata = self.metadata_handler.extract_metadata(topic_path, content_id=topic_id)

        # Start topic section
        topic_parts.append(f'<div class="topic-section" data-topic-id="{topic_id}">')

        # Process topic content
        topic_content = self.transform_topic(topic_path)

        # Add metadata features
        features = self.metadata_handler.get_toggleable_features(metadata)
        if features.get('show_journal_table'):
            topic_parts.append(self._generate_journal_table(metadata))
        if features.get('show_abstract'):
            topic_parts.append(self._generate_abstract(metadata))

        # Add content and close section
        topic_parts.append(topic_content)
        topic_parts.append('</div>')

        return topic_parts

    def _create_topic_error_message(self, href: str) -> str:
        """Create error message for missing topic."""
        return (
            f'<div class="error-message">'
            f'Topic not found: {href}'
            f'</div>'
        )

    def _inject_map_artifacts(self, content: str, artifacts: List[Dict[str, Any]]) -> str:
        """Inject artifacts into map content."""
        self.logger.info("Processing artifacts...")
        soup = BeautifulSoup(content, 'html.parser')

        for artifact in artifacts:
            artifact_path = self.dita_root / artifact['href'].lstrip('../')
            target_id = artifact['target']

            target_elem = self.html.find_target_element(soup, target_id)
            if not target_elem:
                self.logger.warning(f"Target element not found: {target_id}")
                continue

            self.logger.info(f"Processing artifact: {artifact_path}")
            artifact_html = self.artifact_renderer.render_artifact(
                artifact_path,
                target_id
            )

            new_content = BeautifulSoup(artifact_html, 'html.parser')
            target_elem.insert_after(new_content)

        return str(soup)

    def _generate_journal_table(self, metadata: Dict[str, Any]) -> str:
        """Generate HTML table for journal metadata."""
        table_html = ['<div class="journal-metadata">']
        table_html.append('<table class="metadata-table">')

        fields = [
            ('journal', 'Journal'),
            ('doi', 'DOI'),
            ('publication-date', 'Published'),
            ('authors', 'Authors'),
            ('institution', 'Institution')
        ]

        for key, label in fields:
            if value := metadata.get(key):
                if isinstance(value, list):
                    value = ', '.join(value)
                table_html.append(
                    f'<tr><th>{label}</th><td>{value}</td></tr>'
                )

        table_html.append('</table></div>')
        return '\n'.join(table_html)

    def _generate_abstract(self, metadata: Dict[str, Any]) -> str:
        """Generate HTML for abstract section."""
        if abstract := metadata.get('abstract'):
            return (
                f'<div class="abstract-section">'
                f'<h2>Abstract</h2>'
                f'<p>{abstract}</p>'
                f'</div>'
            )
        return ''

    def _transform_dita_topic(self, topic_path: Path) -> str:
        """Transform DITA topic to HTML with consistent heading handling."""
        try:
            tree = self.dita_parser.parse_file(topic_path)
            if tree is None:
                raise ValueError(f"Failed to parse DITA topic: {topic_path}")

            self.logger.debug(f"Starting DITA topic transformation: {topic_path}")
            html_parts = []

            # Process title (H1)
            title_elem = tree.find('.//title')
            if title_elem is not None and title_elem.text:
                # Pass our heading handler to the DITA processor
                self.dita_processor.heading_handler = self.heading_handler

                self.logger.debug(f"Before processing DITA title. Heading state: {self.heading_handler.counters}")
                content = self.dita_processor.process_element(tree, topic_path)
                self.logger.debug(f"After processing DITA title. Heading state: {self.heading_handler.counters}")
                html_parts.append(content)

            return '\n'.join(html_parts)

        except Exception as e:
            self.logger.error(f"Error transforming DITA topic {topic_path}: {str(e)}")
            raise

    def _process_main_title(self, tree: etree._Element, html_parts: List[str]) -> None:
        """Process the main title of a DITA topic."""
        title_elem = tree.find('.//title')
        if title_elem is not None and title_elem.text:
            title_id = self.id_handler.generate_content_id(Path(title_elem.text))

            # Process title - removed LaTeX processing
            heading_id, formatted_text = self.heading_handler.process_heading(
                title_elem.text,  # Direct text use
                1,  # Level 1
                is_map_title=True  # Only map titles should be unnumbered
            )

            html_parts.append(
                f'<h1 id="{title_id}" class="topic-title">'
                f'{formatted_text}'
                f'<a href="#{title_id}" class="heading-anchor">¶</a>'
                f'</h1>'
            )

    def _process_body_content(self, tree: etree._Element, html_parts: List[str], current_h1: int) -> None:
        """Process the body content of a DITA topic."""
        body_elem = self._get_body_element(tree)
        if body_elem is not None:
            for elem in body_elem:
                if elem.tag == 'section':
                    self._process_section(elem, html_parts, current_h1)
                else:
                    self._process_non_section_element(elem, html_parts)

    def _get_body_element(self, tree: etree._Element) -> Optional[etree._Element]:
        """Get the appropriate body element based on topic type."""
        root_tag = etree.QName(tree).localname
        body_tag = {
            'concept': './/conbody',
            'task': './/taskbody',
            'reference': './/refbody',
            'topic': './/body'
        }.get(root_tag, './/body')

        return tree.find(body_tag)

    def _process_section(self, section_elem: etree._Element, html_parts: List[str], current_h1: int) -> None:
        """Process a section element."""
        self._process_section_title(section_elem, html_parts, current_h1)
        self._process_section_content(section_elem, html_parts)

    def _process_section_title(self, section_elem: etree._Element, html_parts: List[str], current_h1: int) -> None:
        """Process a section title."""
        section_title = section_elem.find('title')
        if section_title is not None and section_title.text:
            # Removed LaTeX processing, use text directly
            self.heading_handler._current_h1 = current_h1
            heading_id, formatted_text = self.heading_handler.process_heading(
                section_title.text,  # Direct text use
                2  # Section titles are H2
            )

            html_parts.append(
                f'<h2 id="{heading_id}" class="section-title">'
                f'{formatted_text}'
                f'<a href="#{heading_id}" class="heading-anchor">¶</a>'
                f'</h2>'
            )

    def _process_section_content(self, section_elem: etree._Element, html_parts: List[str]) -> None:
        """Process the content within a section."""
        for child in section_elem:
            if child.tag != 'title':  # Skip title as we've already processed it
                content = self.dita_processor.process_element(child)
                html_parts.append(content)

    def _process_non_section_element(self, elem: etree._Element, html_parts: List[str]) -> None:
        """Process a non-section element."""
        # Remove LaTeX processing, pass element directly to DITA processor
        content = self.dita_processor.process_element(elem)
        html_parts.append(content)

    def resolve_path(self, base_path: Path, href: str) -> Optional[Path]:
        """Resolve relative path from base path."""
        try:
            if href.startswith('../'):
                return (base_path.parent / href).resolve()
            return (self.topics_dir / href).resolve()
        except Exception as e:
            self.logger.error(f"Error resolving path: {str(e)}")
            return None

    def _create_error_html(self, error: Exception, context: Path) -> str:
        """Create error message HTML."""
        return f"""
        <div class="error-container bg-red-50 border-l-4 border-red-500 p-4 rounded-lg my-4">
            <h3 class="text-lg font-medium text-red-800">Processing Error</h3>
            <p class="text-red-700">{str(error)}</p>
            <div class="mt-2 text-sm text-red-600">
                <p>Error occurred processing: {context}</p>
            </div>
        </div>
        """
