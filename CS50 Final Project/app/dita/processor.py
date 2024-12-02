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


# Third-party imports
from bs4 import BeautifulSoup, Tag
from lxml import etree
import markdown
import frontmatter

# Local imports
from .utils.html_helpers import HTMLHelper
from .utils.heading import HeadingHandler
from .utils.metadata import MetadataHandler
from .utils.id_handler import DITAIDHandler
from .artifacts.parser import ArtifactParser
from .artifacts.renderer import ArtifactRenderer
from .utils.dita_elements import DITAContentProcessor

# Type aliases
HTMLString = str
XMLElement = Any  # or more specifically: etree._Element


class DITAProcessor:
    """
    DITA content processor for HTML transformation.
    Handles DITA, Markdown, and interactive artifact processing.
    """
    ##### 1. INITIALIZATION #####
    def __init__(self) -> None:
        """
        Initialize processor with utilities and configurations.
        Sets up file paths, parsers, and processing utilities.
        """
        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Initialize paths
        self.app_root = Path(__file__).parent.parent
        self.dita_root = self.app_root / 'dita'
        self.maps_dir = self.dita_root / 'maps'
        self.topics_dir = self.dita_root / 'topics'
        self.output_dir = self.dita_root / 'output'
        self.artifacts_dir = self.dita_root / 'artifacts'

        # Ensure critical directories exist
        self._init_directories()

        # Initialize utilities
        self.dita_elements = DITAContentProcessor()
        self.id_handler = DITAIDHandler()
        self.heading_handler = HeadingHandler()
        self.metadata_handler = MetadataHandler()
        self.html = HTMLHelper()


        # Initialize artifact handlers with proper paths
        self.artifact_parser = ArtifactParser(self.dita_root)
        self.artifact_renderer = ArtifactRenderer(self.artifacts_dir)

        # Initialize XML parser with safety features
        self.parser = etree.XMLParser(
            recover=True,
            remove_blank_text=True,
            resolve_entities=False,
            dtd_validation=False,
            load_dtd=False,
            no_network=True
        )

        # Initialize Markdown processor with extensions
        self.md = markdown.Markdown(extensions=[
            'fenced_code',
            'tables',
            'meta',
            'attr_list'
        ])

        self.logger.info("DITA Processor initialized successfully")

    def _init_directories(self) -> None:
        """Create necessary directories if they don't exist."""
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

        for directory in required_dirs:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Ensured directory exists: {directory}")
            except Exception as e:
                self.logger.error(f"Failed to create directory {directory}: {e}")
                raise

    def _validate_setup(self) -> bool:
        """
        Validate processor setup and required dependencies.
        Returns True if everything is properly configured.
        """
        try:
            # Check critical directories
            if not all(p.exists() for p in [
                self.maps_dir,
                self.topics_dir,
                self.artifacts_dir
            ]):
                self.logger.error("Critical directories missing")
                return False

            # Validate utilities
            if not all([
                self.heading_handler,
                self.metadata_handler,
                self.id_handler,
                self.artifact_parser,
                self.artifact_renderer
            ]):
                self.logger.error("One or more utilities not properly initialized")
                return False

            # Test XML parser
            test_xml = "<test>content</test>"
            if etree.fromstring(test_xml, self.parser) is None:
                self.logger.error("XML parser not functioning properly")
                return False

            self.logger.info("Processor setup validated successfully")
            return True

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return False



    def list_topics(self) -> List[Dict[str, Any]]:
            """List all available topics"""
            try:
                topics = []
                # Search in known topic directories
                for subdir in ['articles', 'acoustics', 'audio', 'abstracts', 'journals']:
                    topic_dir = self.topics_dir / subdir
                    if not topic_dir.exists():
                        continue

                    # Look for both .dita and .md files
                    for ext in ['.dita', '.md']:
                        for topic_file in topic_dir.glob(f'*{ext}'):
                            try:
                                topic_info = {
                                    'id': topic_file.stem,
                                    'path': str(topic_file.relative_to(self.dita_root)),
                                    'type': subdir,
                                    'format': ext.lstrip('.'),
                                    'title': self._get_topic_title(topic_file)
                                }
                                topics.append(topic_info)
                            except Exception as e:
                                self.logger.error(f"Error processing topic {topic_file}: {str(e)}")

                return topics

            except Exception as e:
                self.logger.error(f"Error listing topics: {str(e)}")
                return []

    def _get_topic_title(self, topic_path: Path) -> str:
            """
            Extract title from topic file.
            Returns the filename stem if no title is found.
            """
            try:
                if topic_path.suffix == '.md':
                    with open(topic_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        post = frontmatter.loads(content)
                        # Use get() with default value
                        return post.metadata.get('title') or topic_path.stem

                else:  # .dita file
                    tree = etree.parse(str(topic_path), self.parser)
                    title_elem = tree.find('.//title')
                    # Use conditional expression to handle None
                    return title_elem.text if title_elem is not None and title_elem.text else topic_path.stem

            except Exception as e:
                self.logger.error(f"Error getting topic title: {str(e)}")
                return topic_path.stem

    def list_maps(self) -> List[Dict[str, Any]]:
        """List all available DITA maps"""
        try:
            maps = []
            if self.maps_dir.exists():
                for map_file in self.maps_dir.glob('*.ditamap'):
                    try:
                        self.logger.info(f"Processing map file: {map_file}")

                        with open(map_file, 'r', encoding='utf-8') as f:
                            content = f.read()

                        tree = etree.fromstring(content.encode('utf-8'), self.parser)

                        # Get the map title
                        title_elem = tree.find(".//title")
                        title = title_elem.text if title_elem is not None and title_elem.text else map_file.stem

                        # Process topic groups
                        groups = []
                        for topicgroup in tree.findall(".//topicgroup"):
                            navtitle = topicgroup.find(".//navtitle")
                            group_title = navtitle.text if navtitle is not None and navtitle.text else "Untitled Group"

                            topics = []
                            for topicref in topicgroup.findall(".//topicref"):
                                href = topicref.get('href')
                                if href:
                                    topic_id = Path(href).stem
                                    topics.append({
                                        'id': topic_id,
                                        'href': href
                                    })

                            groups.append({
                                'navtitle': group_title,
                                'topics': topics
                            })

                        maps.append({
                            'id': map_file.stem,
                            'title': title,
                            'groups': groups
                        })

                    except Exception as e:
                        self.logger.error(f"Error processing map {map_file}: {e}")
                        maps.append({
                            'id': map_file.stem,
                            'title': map_file.stem,
                            'error': str(e)
                        })

            return maps

        except Exception as e:
            self.logger.error(f"Error listing maps: {str(e)}")
            return []


    ##### 2. PRIMARY ENTRY POINTS #####

    def transform(self, input_path: Path) -> HTMLString:
            """
            Main entry point for transforming any DITA content to HTML.

            Args:
                input_path: Path to the input file (.dita, .md, or .ditamap)

            Returns:
                HTMLString: Transformed HTML content

            The transformation process:
            1. Validates input and determines content type
            2. Extracts metadata and generates IDs
            3. Processes content based on type
            4. Injects artifacts if present
            5. Applies final formatting
            """
            try:
                    self.logger.info(f"Starting transformation of: {input_path}")

                    # Validate input path
                    if not input_path.exists():
                        error_msg = f"File not found: {input_path}"
                        self.logger.error(error_msg)
                        return self.handle_error(FileNotFoundError(error_msg), input_path)

                    # Generate content ID
                    content_id = self.id_handler.generate_content_id(input_path)
                    self.logger.debug(f"Generated content ID: {content_id}")

                    # Extract metadata
                    metadata = self.metadata_handler.extract_metadata(
                        input_path,
                        content_id=content_id
                    )
                    self.logger.debug(f"Extracted metadata: {metadata}")

                    # Reset heading handler for new document
                    self.heading_handler.reset()

                    # Initialize html_content
                    html_content: HTMLString = ""

                    # Process based on file type
                    if input_path.suffix == '.ditamap':
                        # Parse artifacts first
                        artifacts = self.artifact_parser.parse_artifact_references(input_path)
                        self.logger.info(f"Found artifacts: {artifacts}")

                        # Pre-register artifact target headings
                        for artifact in artifacts:
                            if target := artifact.get('target'):
                                self.heading_handler.register_existing_id(target, target)
                                self.logger.debug(f"Registered target heading: {target}")

                        # Transform map content
                        html_content = self.transform_map(input_path)

                        # Inject artifacts if any
                        if artifacts:
                            html_content = self.inject_artifacts(html_content, artifacts)
                            self.logger.info("Artifacts injected into content")
                    else:  # .dita or .md files
                        # Initialize section counters
                        section_counters = {
                            'h1': 0, 'h2': 0, 'h3': 0,
                            'h4': 0, 'h5': 0, 'h6': 0
                        }

                        # Process single topic
                        html_content = self.process_topic(
                            topic_path=input_path,
                            section_counters=section_counters
                        )

                    # Ensure we have content
                    if not html_content:
                        return self.handle_error(
                            ValueError("No content generated"),
                            input_path
                        )

                    # Apply final formatting
                    soup = BeautifulSoup(html_content, 'html.parser')
                    content_div = self.html.ensure_wrapper(soup)

                    # Add metadata attributes
                    content_div['data-content-id'] = content_id
                    content_div['data-content-type'] = input_path.suffix.lstrip('.')

                    # Add metadata features
                    features = self.metadata_handler.get_toggleable_features(metadata)
                    for feature, value in features.items():
                        content_div[f'data-{feature}'] = str(value).lower()

                    return str(soup)

            except Exception as e:
                self.logger.error(f"Error transforming {input_path}: {str(e)}")
                return self.handle_error(e, input_path)

    def _format_final_html(self, content: HTMLString, metadata: Dict[str, Any]) -> HTMLString:
        """
        Apply final HTML formatting and metadata attributes.
        Uses HTMLHelper for consistent HTML manipulation.
        """
        try:
            soup = BeautifulSoup(content, 'html.parser')

            # Get or create content wrapper
            content_div = self.html.ensure_wrapper(soup)

            # Add metadata-based classes
            if metadata.get('is_journal_entry'):
                self.html.add_class(content_div, 'journal-entry')

            if metadata.get('has_interactive_content'):
                self.html.add_class(content_div, 'interactive')

            # Add feature-based classes
            for feature, value in metadata.items():
                if isinstance(value, bool) and value:
                    feature_class = f"has-{feature.replace('_', '-')}"
                    self.html.add_class(content_div, feature_class)

            # Add metadata attributes
            self.html.set_data_attributes(content_div, {
                'processed': 'true',
                'processed-time': datetime.now().isoformat()
            })

            self.logger.debug("Final HTML formatting applied successfully")
            return str(soup)

        except Exception as e:
            self.logger.error(f"Error in final formatting: {str(e)}")
            # Return original content if formatting fails
            return content


    def transform_map(self, map_path: Path) -> HTMLString:
            """
            Transform DITA map and its referenced topics to HTML.

            Args:
                map_path: Path to the .ditamap file

            Returns:
                HTMLString: Combined HTML content with proper structure

            Processing steps:
            1. Parse map structure
            2. Initialize section counters
            3. Process each topic reference
            4. Combine content with proper hierarchy
            """
            try:
                self.logger.info(f"Transforming map: {map_path}")
                map_id = self.id_handler.generate_map_id(map_path)

                # Reset counters for new map
                self.heading_handler.reset()
                section_counters = {'h1': 0, 'h2': 0, 'h3': 0, 'h4': 0, 'h5': 0, 'h6': 0}

                # Parse map XML
                tree = self.parse_dita(map_path)
                html_content = ['<div class="map-content">']

                # Process map title
                title_elem = tree.find(".//title")
                if title_elem is not None and title_elem.text:
                    title_id = self.heading_handler.generate_heading_id(title_elem.text)
                    html_content.append(
                        f'<h1 id="{title_id}" class="map-title">'
                        f'{title_elem.text}'
                        f'<a href="#{title_id}" class="heading-anchor" '
                        f'aria-label="Link to this heading">¶</a>'
                        f'</h1>'
                    )

                # Process each topicref
                for topicref in tree.xpath(".//topicref"):
                    href = topicref.get('href')
                    if href:
                        self.logger.info(f"Processing topicref: {href}")
                        topic_path = self.resolve_path(map_path, href)

                        if topic_path and topic_path.exists():
                            # Generate topic ID
                            topic_id = self.id_handler.generate_topic_id(topic_path, map_path)

                            # Extract topic metadata
                            metadata = self.metadata_handler.extract_metadata(
                                topic_path,
                                content_id=topic_id
                            )

                            # Start topic section
                            html_content.append(
                                f'<div class="topic-section" '
                                f'data-topic-id="{topic_id}">'
                            )

                            # Process topic content
                            topic_content = self.process_topic(
                                topic_path,
                                section_counters
                            )

                            # Add metadata-based features
                            features = self.metadata_handler.get_toggleable_features(metadata)
                            if features.get('show_journal_table'):
                                html_content.append(self._generate_journal_table(metadata))
                            if features.get('show_abstract'):
                                html_content.append(self._generate_abstract(metadata))

                            # Add main content
                            html_content.append(topic_content)
                            html_content.append('</div>')  # Close topic section
                        else:
                            self.logger.error(f"Could not resolve topic for href: {href}")
                            html_content.append(
                                f'<div class="error-message">'
                                f'Topic not found: {href}'
                                f'</div>'
                            )

                html_content.append('</div>')  # Close map-content

                # Combine and format final HTML
                combined_html = '\n'.join(html_content)

                # Final processing
                soup = BeautifulSoup(combined_html, 'html.parser')

                # Add map metadata
                soup = BeautifulSoup(combined_html, 'html.parser')

                # Get map metadata
                map_metadata = self.metadata_handler.extract_metadata(
                    map_path,
                    content_id=map_id
                )

                # Use HTMLHelper to safely set attributes
                content_div = self.html.ensure_wrapper(soup, wrapper_class='map-content')

                # Prepare attributes
                map_attributes = {
                    'map-id': map_id,
                    'map-type': map_metadata.get('type', 'standard'),
                    'processed-at': datetime.now().isoformat()
                }

                # Add other metadata attributes
                for key, value in map_metadata.items():
                    if isinstance(value, (str, int, bool)):
                        map_attributes[key] = value

                # Set all attributes safely
                self.html.set_data_attributes(content_div, map_attributes)

                return str(soup)

            except Exception as e:
                self.logger.error(f"Error transforming map {map_path}: {str(e)}")
                return self.handle_error(e, map_path)




    def _generate_journal_table(self, metadata: Dict[str, Any]) -> HTMLString:
        """Generate HTML table for journal metadata"""
        try:
            table_html = ['<div class="journal-metadata">']
            table_html.append('<table class="metadata-table">')

            # Define fields to display in order
            fields = [
                ('journal', 'Journal'),
                ('doi', 'DOI'),
                ('publication-date', 'Published'),
                ('authors', 'Authors'),
                ('institution', 'Institution'),
                ('citation', 'Citation')
            ]

            for key, label in fields:
                if value := metadata.get(key):
                    if isinstance(value, list):
                        value = ', '.join(value)
                    table_html.append(
                        f'<tr>'
                        f'<th class="metadata-label">{html.escape(label)}</th>'
                        f'<td class="metadata-value">{html.escape(str(value))}</td>'
                        f'</tr>'
                    )

            table_html.append('</table></div>')
            return '\n'.join(table_html)

        except Exception as e:
            self.logger.error(f"Error generating journal table: {str(e)}")
            return ''

    def _generate_abstract(self, metadata: Dict[str, Any]) -> HTMLString:
        """Generate HTML for abstract section"""
        try:
            if abstract := metadata.get('abstract'):
                return (
                    f'<div class="abstract-section">'
                    f'<h2 class="abstract-title">Abstract</h2>'
                    f'<div class="abstract-content">'
                    f'{html.escape(abstract)}'
                    f'</div>'
                    f'</div>'
                )

            if shortdesc := metadata.get('shortdesc'):
                return (
                    f'<div class="abstract-section">'
                    f'<div class="abstract-content short-description">'
                    f'{html.escape(shortdesc)}'
                    f'</div>'
                    f'</div>'
                )

            return ''

        except Exception as e:
            self.logger.error(f"Error generating abstract: {str(e)}")
            return ''



    ##### 3. CORE PROCESSING #####

    def process_topic(
        self,
        topic_path: Path,
        section_counters: Dict[str, int]
    ) -> HTMLString:
        """
        Process single topic with section numbering and formatting.

        Args:
            topic_path: Path to topic file (.dita or .md)
            section_counters: Dictionary tracking section numbers for h1-h6

        Returns:
            HTMLString: Processed HTML content with proper heading structure
        """
        try:
            self.logger.info(f"Processing topic: {topic_path}")

            # Handle different file types
            if topic_path.suffix == '.md':
                return self._process_markdown_topic(topic_path, section_counters)
            elif topic_path.suffix == '.dita':
                return self._process_dita_topic(topic_path, section_counters)
            else:
                raise ValueError(f"Unsupported file type: {topic_path.suffix}")

        except Exception as e:
            self.logger.error(f"Error processing topic {topic_path}: {str(e)}")
            return self.handle_error(e, topic_path)

    def _process_markdown_topic(self, topic_path: Path,
                              section_counters: Dict[str, int]) -> HTMLString:
        """Process Markdown topic file"""
        try:
            with open(topic_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse frontmatter and content
            post = frontmatter.loads(content)
            html_content = self.md.convert(post.content)

            # Process HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # Process all headings
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                level = int(heading.name[1])  # Get heading level number
                original_text = heading.get_text(strip=True)

                # Generate heading ID and formatted text
                heading_id, formatted_text = self.heading_handler.process_heading(
                    original_text,
                    level,
                    section_counters
                )

                # Update heading
                heading['id'] = heading_id
                heading.string = formatted_text

                # Add anchor link
                anchor = soup.new_tag(
                    'a',
                    href=f'#{heading_id}',
                    class_='heading-anchor',
                    **{'aria-label': 'Link to this heading'}
                )
                anchor.string = '¶'
                heading.append(anchor)

                # Add appropriate heading classes
                heading['class'] = self._get_heading_classes(level)

            return str(soup)

        except Exception as e:
            self.logger.error(f"Error processing markdown topic: {str(e)}")
            raise

    def _process_dita_topic(self, topic_path: Path,
                           section_counters: Dict[str, int]) -> HTMLString:
        """Process DITA topic file"""
        try:
            # Parse DITA XML
            tree = self.parse_dita(topic_path)
            html_content = ['<div class="dita-content">']

            # Process title
            title_elem = tree.find(".//title")
            if title_elem is not None and title_elem.text:
                heading_id, formatted_text = self.heading_handler.process_heading(
                    title_elem.text,
                    1,  # Main title is always h1
                    section_counters
                )
                html_content.append(
                    f'<h1 id="{heading_id}" class="{self._get_heading_classes(1)}">'
                    f'{formatted_text}'
                    f'<a href="#{heading_id}" class="heading-anchor" '
                    f'aria-label="Link to this heading">¶</a>'
                    f'</h1>'
                )

            # Process sections
            for section in tree.xpath(".//section"):
                # Process section title
                section_title = section.find("title")
                if section_title is not None and section_title.text:
                    heading_id, formatted_text = self.heading_handler.process_heading(
                        section_title.text,
                        2,  # Section titles are h2
                        section_counters
                    )
                    html_content.append(
                        f'<h2 id="{heading_id}" class="{self._get_heading_classes(2)}">'
                        f'{formatted_text}'
                        f'<a href="#{heading_id}" class="heading-anchor" '
                        f'aria-label="Link to this heading">¶</a>'
                        f'</h2>'
                    )

                # Process section content
                for elem in section:
                    if elem.tag != 'title':  # Skip title as it's already processed
                        html_content.append(self._process_dita_element(elem))

            html_content.append('</div>')
            return '\n'.join(html_content)

        except Exception as e:
            self.logger.error(f"Error processing DITA topic: {str(e)}")
            raise

    def _process_dita_element(self, elem: etree._Element) -> HTMLString:
        """Process individual DITA elements to HTML"""
        tag = etree.QName(elem).localname

        if tag == 'p' and elem.text:
            return f'<p class="mb-4">{elem.text}</p>'

        elif tag == 'ul':
            items = [
                f'<li class="mb-2">{item.text}</li>'
                for item in elem.findall('li')
                if item.text
            ]
            return f'<ul class="list-disc ml-6 mb-4">{"".join(items)}</ul>'

        elif tag == 'ol':
            items = [
                f'<li class="mb-2">{item.text}</li>'
                for item in elem.findall('li')
                if item.text
            ]
            return f'<ol class="list-decimal ml-6 mb-4">{"".join(items)}</ol>'

        elif tag == 'codeblock':
            return (
                f'<pre class="bg-gray-100 p-4 rounded-lg mb-4">'
                f'<code>{elem.text if elem.text else ""}</code>'
                f'</pre>'
            )

        return ''  # Return empty string for unhandled elements

    def _get_heading_classes(self, level: int) -> str:
        """Get appropriate classes for heading level"""
        classes = {
            1: "text-2xl font-bold mb-4",
            2: "text-xl font-bold mt-6 mb-3",
            3: "text-lg font-bold mt-4 mb-2",
            4: "text-base font-bold mt-3 mb-2",
            5: "text-sm font-bold mt-2 mb-1",
            6: "text-xs font-bold mt-2 mb-1"
        }
        return classes.get(level, "")



    def process_content(self, content_path: Path) -> HTMLString:
        """
        Process raw content to HTML with proper structure and formatting.

        Args:
            content_path: Path to content file (.dita, .md, or content string)

        Returns:
            HTMLString: Processed HTML with consistent structure

        This method handles the core content processing:
        1. Content parsing and cleaning
        2. Image and link processing
        3. Code block formatting
        4. Cross-reference resolution
        """
        try:
                self.logger.info(f"Processing content from: {content_path}")

                if not content_path.exists():
                    raise FileNotFoundError(f"Content file not found: {content_path}")

                # Read content
                with open(content_path, 'r', encoding='utf-8') as f:
                    raw_content = f.read()

                # Process based on content type
                if content_path.suffix == '.md':
                    return self._process_markdown_content(raw_content, content_path)
                elif content_path.suffix == '.dita':
                    tree = etree.fromstring(raw_content.encode('utf-8'), self.parser)
                    # Use DITAContentProcessor directly
                    return self.dita_elements.process_element(tree, content_path)
                else:
                    raise ValueError(f"Unsupported content type: {content_path.suffix}")

        except Exception as e:
            self.logger.error(f"Error processing content: {str(e)}")
            return self.handle_error(e, content_path)

    def _process_markdown_content(self, content: str, source_path: Path) -> HTMLString:
        """Process Markdown content to HTML"""
        try:
            # Parse frontmatter and content
            post = frontmatter.loads(content)

            # Convert markdown to HTML
            html_content = self.md.convert(post.content)

            return html_content

        except Exception as e:
            self.logger.error(f"Error processing markdown content: {str(e)}")
            raise


    def inject_artifacts(self, html_content: str, artifacts: List[Dict[str, Any]]) -> HTMLString:
        """Inject interactive artifacts into HTML content."""
        try:
            self.logger.info("Starting artifact injection")
            soup = BeautifulSoup(html_content, 'html.parser')
            successfully_injected = []
            failed_artifacts = []

            for artifact in artifacts:
                target_elem = None  # Initialize before try block
                try:
                    # Extract artifact info
                    artifact_path = self.dita_root / artifact['href'].lstrip('../')
                    component_name = artifact['name']
                    target_id = artifact['target']

                    self.logger.info(f"Processing artifact: {component_name} at {artifact_path}")

                    # Validate artifact
                    if not artifact_path.exists():
                        raise ValueError(f"Artifact file not found: {artifact_path}")

                    # Find target element using HTMLHelper
                    target_elem = self.html.find_target_element(soup, target_id)
                    if not target_elem:
                        raise ValueError(f"Target element not found: {target_id}")

                    # Create isolated artifact container
                    artifact_html = self.artifact_renderer.render_artifact(
                        artifact_path,
                        target_id
                    )

                    if not artifact_html:
                        raise ValueError(f"Failed to render artifact: {artifact_path}")

                    # Insert artifact
                    artifact_soup = BeautifulSoup(artifact_html, 'html.parser')
                    target_elem.insert_after(artifact_soup)

                    successfully_injected.append(component_name)
                    self.logger.info(f"Successfully injected artifact: {component_name}")

                except Exception as e:
                    self.logger.error(f"Failed to inject artifact {artifact.get('name')}: {str(e)}")
                    failed_artifacts.append({
                        'name': artifact.get('name'),
                        'error': str(e)
                    })
                    if target_elem:  # Now safely checked
                        error_html = self.html.create_artifact_error_message(artifact, str(e))
                        error_soup = BeautifulSoup(error_html, 'html.parser')
                        target_elem.insert_after(error_soup)

            # Add injection status
            self.html.add_injection_status(soup, successfully_injected, failed_artifacts)

            return str(soup)

        except Exception as e:
            self.logger.error(f"Critical error in artifact injection: {str(e)}")
            return self.html.add_critical_error_message(html_content, str(e))




    def _wrap_with_error_boundary(self, artifact_html: str, component_name: str) -> str:
        """Wrap artifact HTML with error boundary and loading states"""
        instance_id = f"artifact-{component_name}-{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

        return f"""
        <div class="artifact-container mb-8" data-artifact-id="{instance_id}">
            <!-- Error Boundary -->
            <div class="artifact-error-boundary hidden">
                <!-- Error message content -->
            </div>

            <!-- Loading State -->
            <div class="artifact-loading hidden">
                <!-- Loading indicator -->
            </div>

            <!-- Artifact Content -->
            <div class="artifact-content">
                {artifact_html}
            </div>

            <!-- Error handling script -->
            <script>
                // Error handling and retry logic
            </script>
        </div>
        """




    ##### 4. PATH HANDLING #####

    def resolve_path(self, map_path: Path, href: str) -> Optional[Path]:
        """
        Resolve topic reference path relative to map file.

        Args:
            map_path: Path to the source DITA map
            href: Reference path from topicref or artifact
                Can be:
                - Relative to map (../topics/example.md)
                - Direct path (topics/example.dita)
                - Cross-map reference (othermap.ditamap#topic-id)

        Returns:
            Optional[Path]: Resolved absolute path to the referenced file
        """
        try:
            self.logger.info(f"Resolving path: {href} relative to: {map_path}")

            # Handle cross-map references
            if '#' in href:
                href, topic_id = href.split('#', 1)
                self.logger.debug(f"Split cross-map reference - File: {href}, Topic ID: {topic_id}")

            # Clean the href
            cleaned_href = href.strip().replace('\\', '/')

            # Try different resolution strategies
            resolved_path = None

            # Strategy 1: Resolve relative to map
            if cleaned_href.startswith('..'):
                try:
                    resolved_path = (map_path.parent / cleaned_href).resolve()
                    self.logger.debug(f"Resolved relative to map: {resolved_path}")
                except Exception as e:
                    self.logger.debug(f"Failed to resolve relative to map: {e}")

            # Strategy 2: Resolve from topics directory
            if not resolved_path or not resolved_path.exists():
                try:
                    # Remove any '../topics/' prefix
                    topic_path = cleaned_href.replace('../topics/', '').replace('topics/', '')
                    resolved_path = (self.topics_dir / topic_path).resolve()
                    self.logger.debug(f"Resolved from topics directory: {resolved_path}")
                except Exception as e:
                    self.logger.debug(f"Failed to resolve from topics directory: {e}")

            # Strategy 3: Check maps directory for .ditamap files
            if not resolved_path or not resolved_path.exists():
                if cleaned_href.endswith('.ditamap'):
                    try:
                        resolved_path = (self.maps_dir / cleaned_href).resolve()
                        self.logger.debug(f"Resolved from maps directory: {resolved_path}")
                    except Exception as e:
                        self.logger.debug(f"Failed to resolve from maps directory: {e}")

            # Strategy 4: Search in subdirectories
            if not resolved_path or not resolved_path.exists():
                try:
                    # Search in known content directories
                    for content_dir in [
                        self.topics_dir / 'articles',
                        self.topics_dir / 'acoustics',
                        self.topics_dir / 'audio',
                        self.topics_dir / 'abstracts',
                        self.topics_dir / 'journals'
                    ]:
                        candidate = (content_dir / cleaned_href).resolve()
                        if candidate.exists():
                            resolved_path = candidate
                            self.logger.debug(f"Found in subdirectory: {resolved_path}")
                            break
                except Exception as e:
                    self.logger.debug(f"Failed to search subdirectories: {e}")

            # Validate resolved path
            if resolved_path and resolved_path.exists():
                # Security check: ensure path is within project directory
                if self._is_safe_path(resolved_path):
                    self.logger.info(f"Successfully resolved path to: {resolved_path}")
                    return resolved_path
                else:
                    self.logger.error(f"Path resolution attempted directory traversal: {resolved_path}")
                    return None
            else:
                self.logger.error(f"Could not resolve path for href: {href}")
                return None

        except Exception as e:
            self.logger.error(f"Error resolving path {href}: {str(e)}")
            return None

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

    def _normalize_path(self, path: str) -> str:
        """
        Normalize path string to consistent format.
        """
        return path.replace('\\', '/').strip()


    def get_topic(self, topic_id: str) -> Optional[Path]:
        """
        Get topic path from ID.

        Args:
            topic_id: Identifier for the topic
                Can be:
                - Simple ID (brownian-motion)
                - With extension (brownian-motion.md)
                - With subpath (articles/brownian-motion)
                - Full ditamap reference (brownian-motion.ditamap)

        Returns:
            Optional[Path]: Path to the topic file if found
        """
        try:
            self.logger.info(f"Looking for topic with ID: {topic_id}")

            # Handle .ditamap files first (direct map references)
            if topic_id.endswith('.ditamap'):
                map_path = self.maps_dir / topic_id
                if map_path.exists():
                    self.logger.info(f"Found map at: {map_path}")
                    return map_path

            # Clean the topic ID
            base_id = self._clean_topic_id(topic_id)
            self.logger.debug(f"Cleaned topic ID: {base_id}")

            # Split path components
            path_parts = base_id.split('/')
            filename = path_parts[-1]
            subdirs = path_parts[:-1]

            self.logger.debug(f"Searching for: filename={filename}, subdirs={subdirs}")

            # Search strategies in priority order
            search_strategies = [
                self._search_maps,          # Check maps first
                self._search_direct_path,   # Try direct path
                self._search_topics_tree,   # Search in topics tree
                self._search_by_metadata    # Search by metadata matching
            ]

            for strategy in search_strategies:
                if found_path := strategy(filename, subdirs):
                    self.logger.info(f"Found topic at: {found_path}")
                    return found_path

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

    def _search_maps(self, filename: str, subdirs: List[str]) -> Optional[Path]:
        """Search in maps directory"""
        try:
            # Check direct ditamap
            map_path = self.maps_dir / f"{filename}.ditamap"
            if map_path.exists():
                return map_path

            # Check in map subdirectories
            if subdirs:
                subdir_path = self.maps_dir.joinpath(*subdirs) / f"{filename}.ditamap"
                if subdir_path.exists():
                    return subdir_path

            return None
        except Exception as e:
            self.logger.debug(f"Map search error: {str(e)}")
            return None

    def _search_direct_path(self, filename: str, subdirs: List[str]) -> Optional[Path]:
        """Try direct path resolution"""
        try:
            # Check both .dita and .md extensions
            for ext in ['.dita', '.md']:
                # Build full path
                if subdirs:
                    full_path = self.topics_dir.joinpath(*subdirs) / f"{filename}{ext}"
                else:
                    full_path = self.topics_dir / f"{filename}{ext}"

                if full_path.exists():
                    return full_path

            return None
        except Exception as e:
            self.logger.debug(f"Direct path search error: {str(e)}")
            return None

    def _search_topics_tree(self, filename: str, subdirs: List[str]) -> Optional[Path]:
        """Search through topics directory tree"""
        try:
            # Known content directories
            content_dirs = [
                self.topics_dir / 'articles',
                self.topics_dir / 'acoustics',
                self.topics_dir / 'audio',
                self.topics_dir / 'abstracts',
                self.topics_dir / 'journals'
            ]

            # Search in each content directory
            for content_dir in content_dirs:
                # Check with subdirs if provided
                if subdirs:
                    search_dir = content_dir.joinpath(*subdirs)
                else:
                    search_dir = content_dir

                # Try both extensions
                for ext in ['.dita', '.md']:
                    file_path = search_dir / f"{filename}{ext}"
                    if file_path.exists():
                        return file_path

            return None
        except Exception as e:
            self.logger.debug(f"Topics tree search error: {str(e)}")
            return None

    def _search_by_metadata(self, filename: str, subdirs: List[str]) -> Optional[Path]:
        """Search topics using metadata matching"""
        try:
            # This could be enhanced to search by:
            # - Topic titles
            # - Keywords
            # - Categories
            # - Custom metadata fields

            # For now, we'll just do a basic search through all potential files
            for ext in ['.dita', '.md']:
                # Recursive search through topics directory
                for path in self.topics_dir.rglob(f"*{ext}"):
                    try:
                        # Check if metadata matches
                        if self._check_topic_metadata(path, filename):
                            return path
                    except Exception as e:
                        self.logger.debug(f"Metadata check error for {path}: {str(e)}")
                        continue

            return None
        except Exception as e:
            self.logger.debug(f"Metadata search error: {str(e)}")
            return None

    def _check_topic_metadata(self, path: Path, search_term: str) -> bool:
        """Check if topic metadata matches search term"""
        try:
            # Extract metadata based on file type
            if path.suffix == '.md':
                with open(path, 'r', encoding='utf-8') as f:
                    metadata = frontmatter.load(f).metadata
            else:  # .dita
                tree = etree.parse(str(path), self.parser)
                metadata = self.metadata_handler.extract_metadata(path, '')

            # Check various metadata fields
            searchable_fields = ['title', 'id', 'keywords', 'categories']

            for field in searchable_fields:
                value = metadata.get(field)
                if value:
                    if isinstance(value, str) and search_term.lower() in value.lower():
                        return True
                    elif isinstance(value, list) and any(
                        search_term.lower() in str(v).lower() for v in value
                    ):
                        return True

            return False
        except Exception as e:
            self.logger.debug(f"Metadata check error: {str(e)}")
            return False



    ##### 5. PARSING AND ERROR HANDLING #####


    def parse_dita(self, input_path: Path) -> etree._Element:
        """
        Parse DITA file into XML tree with validation and error handling.

        Args:
            input_path: Path to DITA file (.dita or .ditamap)

        Returns:
            etree._Element: Parsed XML tree

        Raises:
            ValueError: If file doesn't exist or has invalid content
            etree.XMLSyntaxError: If XML parsing fails
        """
        try:
            self.logger.info(f"Parsing DITA file: {input_path}")

            if not input_path.exists():
                raise ValueError(f"File not found: {input_path}")

            # Read content
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                raise ValueError(f"Empty file: {input_path}")

            # First try strict parsing to catch validation errors
            try:
                strict_parser = etree.XMLParser(
                    remove_blank_text=True,
                    resolve_entities=False,
                    dtd_validation=False,
                    load_dtd=False,
                    no_network=True,
                    recover=False  # Strict mode
                )
                tree = etree.fromstring(content.encode('utf-8'), strict_parser)
                self.logger.debug("Strict parsing successful")

            except etree.XMLSyntaxError as e:
                # Log the validation error details
                self._log_xml_error(e, content, input_path)

                # Fallback to recovery mode
                self.logger.info("Attempting parsing with recovery mode")
                tree = etree.fromstring(content.encode('utf-8'), self.parser)

                if tree is None:
                    raise ValueError("Failed to parse XML even in recovery mode")

            # Validate DITA structure
            doctype = self._get_doctype(tree)
            if not self._validate_dita_structure(tree, doctype):
                self.logger.warning(f"Invalid DITA structure in {input_path}")
                # Continue processing but log the issue

            self.logger.info(f"Successfully parsed: {input_path}")
            return tree

        except Exception as e:
            self.logger.error(f"Error parsing DITA file {input_path}: {str(e)}")
            raise

    def _log_xml_error(self, error: etree.XMLSyntaxError,
                       content: str, file_path: Path) -> None:
        """Log detailed XML error information"""
        self.logger.error(f"XML Syntax Error in {file_path.name}:")
        self.logger.error(f"Error: {error.msg}")
        self.logger.error(f"Line {error.lineno}, Column {error.offset}")

        # Show context around the error
        lines = content.splitlines()
        if 0 <= error.lineno - 1 < len(lines):
            # Show a few lines before and after the error
            start = max(0, error.lineno - 3)
            end = min(len(lines), error.lineno + 2)

            self.logger.error("Context:")
            for i in range(start, end):
                prefix = "-> " if i == error.lineno - 1 else "   "
                self.logger.error(f"{prefix}{i + 1}: {lines[i]}")

                if i == error.lineno - 1:
                    # Point to the error position
                    self.logger.error("   " + " " * (error.offset + 2) + "^")

    def _get_doctype(self, tree: etree._Element) -> str:
        """
        Determine DITA document type from root element.
        Returns one of: 'topic', 'concept', 'task', 'reference', 'map'
        """
        root_tag = etree.QName(tree).localname

        # Map root tags to document types
        doctypes = {
            'topic': 'topic',
            'concept': 'concept',
            'task': 'task',
            'reference': 'reference',
            'map': 'map'
        }

        return doctypes.get(root_tag, 'unknown')

    def _validate_dita_structure(self, tree: etree._Element, doctype: str) -> bool:
        """
        Validate basic DITA structure based on document type.
        Returns True if structure is valid.
        """
        try:
            # Check for required elements based on doctype
            if doctype == 'map':
                # Maps must have title
                if tree.find('.//title') is None:
                    self.logger.warning("Map missing required title element")
                    return False

            else:  # Regular topics
                # All topics must have title and body/conbody/taskbody
                if tree.find('.//title') is None:
                    self.logger.warning("Topic missing required title element")
                    return False

                # Check for appropriate body element
                body_tags = {
                    'topic': './/body',
                    'concept': './/conbody',
                    'task': './/taskbody',
                    'reference': './/refbody'
                }

                if doctype in body_tags:
                    if tree.find(body_tags[doctype]) is None:
                        self.logger.warning(f"Missing required {body_tags[doctype]} element")
                        return False

            # Additional structural checks could be added here

            return True

        except Exception as e:
            self.logger.error(f"Error validating DITA structure: {str(e)}")
            return False

    def _validate_references(self, tree: etree._Element) -> List[str]:
        """
        Validate references (hrefs, conrefs, etc.) in the DITA content.
        Returns list of validation errors.
        """
        errors = []

        try:
            # Check href attributes
            for elem in tree.xpath('//*[@href]'):
                href = elem.get('href')
                if href:
                    # Resolve the reference
                    if not self._is_valid_reference(href):
                        errors.append(f"Invalid reference: {href}")

            # Check conref attributes
            for elem in tree.xpath('//*[@conref]'):
                conref = elem.get('conref')
                if conref:
                    if not self._is_valid_reference(conref):
                        errors.append(f"Invalid conref: {conref}")

        except Exception as e:
            self.logger.error(f"Error validating references: {str(e)}")
            errors.append(f"Reference validation error: {str(e)}")

        return errors

    def _is_valid_reference(self, ref: str) -> bool:
        """Check if a reference is valid"""
        try:
            # Handle different reference types
            if ref.startswith('http'):
                # External reference - could add URL validation
                return True

            elif '#' in ref:
                # Internal reference with fragment
                path, fragment = ref.split('#', 1)
                if path:
                    # Check if target file exists
                    return self.resolve_path(self.dita_root, path) is not None
                return True  # Same-file reference

            else:
                # Simple file reference
                return self.resolve_path(self.dita_root, ref) is not None

        except Exception as e:
            self.logger.debug(f"Reference validation error: {str(e)}")
            return False



    def handle_error(self, error: Exception, context: Union[Path, str]) -> HTMLString:
        """
        Create HTML error message with detailed context and recovery options.
        """
        try:
            self.logger.error(f"Handling error in context {context}: {str(error)}")
            error_info = self._get_error_info(error)

            # Generate error HTML based on error type
            if isinstance(error, etree.XMLSyntaxError):
                if isinstance(context, Path):
                    return self._format_xml_error(error, context)
                else:
                    return self._format_general_error(
                        error,
                        error_info,
                        f"XML Syntax Error in content: {context}"
                    )

            elif isinstance(error, FileNotFoundError):
                context_path = Path(context) if isinstance(context, str) else context
                return self._format_missing_file_error(error, context_path)

            elif isinstance(error, ValueError):
                return self._format_validation_error(error, context)

            else:
                return self._format_general_error(error, error_info, context)

        except Exception as e:
            self.logger.critical(f"Error in error handler: {str(e)}")
            return self._format_fallback_error(e)

    def _get_error_info(self, error: Exception) -> Dict[str, Any]:
        """Extract detailed information from error"""
        return {
            'type': error.__class__.__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat(),
            'details': getattr(error, 'details', None)
        }

    def _format_xml_error(self, error: etree.XMLSyntaxError, context: Path) -> HTMLString:
        """Format XML parsing error with line highlighting"""
        try:
            with open(context, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.splitlines()
            error_line = error.lineno - 1

            # Get context lines
            start = max(0, error_line - 2)
            end = min(len(lines), error_line + 3)

            # Format code snippet
            code_lines = []
            for i in range(start, end):
                line_class = "error-line" if i == error_line else ""
                code_lines.append(
                    f'<div class="code-line {line_class}">'
                    f'<span class="line-number">{i + 1}</span>'
                    f'<span class="line-content">{html.escape(lines[i])}</span>'
                    f'</div>'
                )

                # Add error indicator
                if i == error_line:
                    code_lines.append(
                        f'<div class="error-indicator">'
                        f'{"&nbsp;" * error.offset}^</div>'
                    )

            return f"""
            <div class="bg-red-50 border-l-4 border-red-500 p-4 rounded-lg my-4">
                <div class="flex items-center mb-2">
                    <svg class="h-5 w-5 text-red-400 mr-2" viewBox="0 0 20 20" fill="currentColor">
                        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/>
                    </svg>
                    <h3 class="text-lg font-medium text-red-800">XML Syntax Error</h3>
                </div>
                <div class="mt-2">
                    <p class="text-red-700 mb-2">{html.escape(str(error))}</p>
                    <div class="text-sm text-red-600 mb-2">
                        <span>File: {context.name}</span>
                        <span class="ml-4">Line: {error.lineno}</span>
                        <span class="ml-4">Column: {error.offset}</span>
                    </div>
                    <div class="bg-gray-800 text-white p-4 rounded-lg font-mono text-sm overflow-x-auto">
                        {"".join(code_lines)}
                    </div>
                </div>
                <div class="mt-4">
                    <button onclick="window.location.reload()" class="mt-4 px-4 py-2 bg-red-100 text-red-700 rounded hover:bg-red-200 transition-colors">
                        Retry
                    </button>
                </div>
            </div>
            """

        except Exception as e:
            self.logger.error(f"Error formatting XML error: {str(e)}")
            return self._format_fallback_error(error)

    def _format_missing_file_error(self, error: FileNotFoundError,
                                 context: Union[Path, str]) -> HTMLString:
        """Format file not found error with helpful suggestions"""
        context_str = str(context)
        return f"""
        <div class="bg-red-50 border-l-4 border-red-500 p-4 rounded-lg my-4">
            <div class="flex items-center mb-2">
                <svg class="h-5 w-5 text-red-400 mr-2" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"/>
                </svg>
                <h3 class="text-lg font-medium text-red-800">File Not Found</h3>
            </div>
            <div class="mt-2">
                <p class="text-red-700">The file could not be found: {html.escape(context_str)}</p>
                <div class="mt-4">
                    <h4 class="text-sm font-medium text-red-800">Possible Solutions:</h4>
                    <ul class="list-disc ml-6 mt-2 text-sm text-red-700">
                        <li>Check if the file exists in the correct location</li>
                        <li>Verify the file path in your references</li>
                        <li>Ensure proper file permissions</li>
                    </ul>
                </div>
            </div>
        </div>
        """

    def _format_validation_error(self, error: ValueError, context: Union[Path, str]) -> HTMLString:
        """Format validation error with details"""
        context_str = str(context)
        return f"""
        <div class="bg-red-50 border-l-4 border-red-500 p-4 rounded-lg my-4">
            <div class="flex items-center mb-2">
                <svg class="h-5 w-5 text-red-400 mr-2" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>
                </svg>
                <h3 class="text-lg font-medium text-red-800">Validation Error</h3>
            </div>
            <div class="mt-2">
                <p class="text-red-700">{html.escape(str(error))}</p>
                <div class="text-sm text-red-600 mt-2">
                    <strong>Context:</strong> {html.escape(context_str)}
                </div>
            </div>
        </div>
        """

    def _format_general_error(self, error: Exception,
                             error_info: Dict[str, Any],
                             context: Any) -> HTMLString:
        """Format general error with debug information"""
        return f"""
        <div class="error-container">
            <div class="error-header">
                <svg class="error-icon" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>
                </svg>
                <h3 class="error-title">Processing Error</h3>
            </div>
            <div class="error-details">
                <p class="error-message">{html.escape(str(error))}</p>
                <div class="error-context">
                    <strong>Error Type:</strong> {error_info['type']}<br>
                    <strong>Context:</strong> {html.escape(str(context))}
                </div>
                {self._format_debug_info(error_info) if self._is_debug_mode() else ''}
            </div>
            <div class="error-footer">
                <button onclick="window.location.reload()" class="retry-button">
                    Retry
                </button>
            </div>
        </div>
        """

    def _format_fallback_error(self, error: Exception) -> HTMLString:
        """Format minimal fallback error when main error handling fails"""
        return f"""
        <div class="error-container">
            <div class="error-header">
                <h3 class="error-title">Critical Error</h3>
            </div>
            <div class="error-details">
                <p class="error-message">
                    An unexpected error occurred while processing the content.
                    {html.escape(str(error))}
                </p>
            </div>
        </div>
        """

    def _format_debug_info(self, error_info: Dict[str, Any]) -> HTMLString:
        """Format debug information for development"""
        return f"""
        <div class="debug-info">
            <details>
                <summary>Debug Information</summary>
                <pre class="traceback">{html.escape(error_info['traceback'])}</pre>
                <div class="timestamp">
                    Error occurred at: {error_info['timestamp']}
                </div>
            </details>
        </div>
        """

    def _is_debug_mode(self) -> bool:
        """Check if debug mode is enabled"""
        return os.environ.get('FLASK_ENV') == 'development'
