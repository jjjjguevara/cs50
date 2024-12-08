from packaging.metadata import Metadata
# app/dita/utils/dita_parser.py

from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
import re
from lxml import etree
import logging
import frontmatter
from bs4 import BeautifulSoup
from .types import (
    ElementType,
    ParsedElement,
    DiscoveredTopic,
    ParsedMap,
    LaTeXEquation,
    PathLike,
    MetadataDict,
    ParsedElement,
)

from app.dita.utils.metadata import MetadataHandler


class DITAParser:
    """Comprehensive parser for DITA, Markdown, and LaTeX content."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.heading_tracker = HeadingTracker()
        self.parser = etree.XMLParser(
            recover=True,
            remove_blank_text=True,
            resolve_entities=False,
            dtd_validation=False,
            load_dtd=False,
            no_network=True
        )

        self.metadata_handler = MetadataHandler()

    ### MAP PARSING ###

    def _process_metadata_value(self, name: str, value: str) -> Union[bool, str]:
        """Process metadata value based on name."""
        if name in ['index-numbers', 'append-toc']:
            return value.lower() == 'true'
        return value

    def parse_map(self, map_path: Path) -> ParsedMap:
            """Parse a .ditamap file and return a ParsedMap."""
            try:
                # Read the file content
                if not map_path.exists():
                    raise FileNotFoundError(f"Map file not found: {map_path}")
                with open(map_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                # Parse as XML using BeautifulSoup
                soup = BeautifulSoup(content, 'xml')

                # Extract the map title (if available)
                title_element = soup.find('title')
                title = title_element.text.strip() if title_element else None

                # Collect topics
                topics: List[ParsedElement] = []
                for topicref in soup.find_all('topicref'):
                    href = topicref.get('href')
                    if not href:
                        continue  # Skip topicrefs without hrefs

                    topic_path = map_path.parent / href
                    if not topic_path.exists():
                        self.logger.warning(f"Referenced topic not found: {topic_path}")
                        continue

                    element_type = (
                        ElementType.DITA if topic_path.suffix == '.dita' else ElementType.MARKDOWN
                    )
                    topics.append(
                        ParsedElement(
                            id=topic_path.stem,
                            topic_id=topic_path.stem,
                            type=element_type,
                            content=topic_path.read_text(encoding='utf-8'),
                            topic_path=topic_path,
                            source_path=map_path.parent,
                            metadata={},  # Add metadata extraction if needed
                        )
                    )

                # Return a ParsedMap object
                return ParsedMap(
                    title=title,
                    topics=topics,
                    metadata={"id": map_path.stem},
                    source_path=map_path,
                )

            except Exception as e:
                self.logger.error(f"Error parsing map {map_path}: {str(e)}", exc_info=True)
                raise

    def _discover_topics(self, soup: BeautifulSoup, base_path: Path) -> List[ParsedElement]:
            """Discover topics from the .ditamap."""
            topicrefs = soup.find_all('topicref')
            parsed_elements = []

            for topicref in topicrefs:
                href = topicref.get('href')
                if not href:
                    self.logger.warning("Skipping topicref with no href.")
                    continue

                topic_path = base_path / href
                if not topic_path.exists():
                    self.logger.warning(f"Referenced topic not found: {href}")
                    continue

                element_type = ElementType.DITA if topic_path.suffix == '.dita' else ElementType.MARKDOWN
                with open(topic_path, 'r', encoding='utf-8') as topic_file:
                    content = topic_file.read()

                parsed_element = ParsedElement(
                    id=topic_path.stem,
                    topic_id=topic_path.stem,
                    type=element_type,
                    content=content,
                    topic_path=topic_path,
                    source_path=topic_path.parent,
                    metadata={}
                )
                parsed_elements.append(parsed_element)

            return parsed_elements


    def _discover_topic(self, topic_path: Path, map_path: Path) -> Optional[ParsedElement]:
        """
        Discover and parse a single topic into a ParsedElement.

        Args:
            topic_path: Path to the topic file.
            map_path: Path to the parent map file.

        Returns:
            ParsedElement object or None if discovery fails.
        """
        try:
            if not topic_path.exists():
                self.logger.warning(f"Topic file not found: {topic_path}")
                return None

            element_type = (
                ElementType.MARKDOWN if topic_path.suffix == '.md' else ElementType.DITA
            )

            topic_metadata = self.metadata_handler.extract_metadata(topic_path, topic_path.stem)

            return ParsedElement(
                id=f"{map_path.stem}-{topic_path.stem}",  # Unique ID combining map and topic
                topic_id=topic_path.stem,  # Topic's identifier
                type=element_type,
                content=str(topic_path),  # Path as placeholder for content
                topic_path=topic_path,  # Path to the topic file
                source_path=map_path,  # Parent map file path
                metadata=topic_metadata
            )
        except Exception as e:
            self.logger.error(f"Error discovering topic: {str(e)}")
            return None


    def _parse_topicref(self, topicref: etree._Element, map_path: Path) -> Optional[ParsedElement]:
        """
        Parse a single topicref element into a ParsedElement.

        Args:
            topicref: XML element for the topic reference.
            map_path: Path to the parent map file.

        Returns:
            ParsedElement object or None if parsing fails.
        """
        try:
            href = topicref.get('href')
            if not href:
                return None

            topic_path = (map_path.parent / href).resolve()
            if not topic_path.exists():
                self.logger.warning(f"Topic file not found: {topic_path}")
                return None

            element_type = (
                ElementType.MARKDOWN if topic_path.suffix == '.md' else ElementType.DITA
            )

            topic_metadata = self.metadata_handler.extract_metadata(topic_path, topic_path.stem)

            return ParsedElement(
                id=f"{map_path.stem}-{topic_path.stem}",  # Unique ID combining map and topic
                topic_id=topic_path.stem,  # Topic's identifier
                type=element_type,
                content=str(topic_path),  # Path as placeholder for content
                topic_path=topic_path,  # Path to the topic file
                source_path=map_path,  # Parent map file path
                metadata=topic_metadata
            )
        except Exception as e:
            self.logger.error(f"Error parsing topicref: {str(e)}")
            return None



    ### DITA/XML PARSING ###
    def parse_file(self, path: Path) -> Optional[etree._Element]:
        """Parse DITA file into XML tree."""
        try:
            self.logger.debug(f"Parsing DITA file: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.logger.debug(f"Read file content, length: {len(content)}")

            tree = etree.fromstring(content.encode('utf-8'), self.parser)

            # Check for metadata specifically
            metadata = tree.find(".//metadata")
            if metadata is not None:
                self.logger.debug("Found metadata element")
                for othermeta in metadata.findall(".//othermeta"):
                    name = othermeta.get('name')
                    content = othermeta.get('content')
                    self.logger.debug(f"Metadata found: {name}={content}")
            else:
                self.logger.debug("No metadata element found")

            return tree

        except Exception as e:
            self.logger.error(f"Error parsing DITA file {path}: {str(e)}")
            return None

    def parse_content(self, content: str) -> Optional[etree._Element]:
        """Parse DITA content string into XML tree."""
        try:
            return etree.fromstring(content.encode('utf-8'), self.parser)
        except Exception as e:
            self.logger.error(f"Error parsing DITA content: {str(e)}")
            return None

    ### MARKDOWN PARSING ###
    def parse_markdown(self, path: Path) -> Tuple[str, MetadataDict]:
        """Parse markdown file with YAML frontmatter."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)
            return post.content, post.metadata
        except Exception as e:
            self.logger.error(f"Error parsing markdown {path}: {str(e)}")
            return "", {}

    ### LATEX PARSING ###
    def find_latex_equations(self, content: str) -> List[LaTeXEquation]:
        """Find LaTeX equations in content."""
        equations = []
        # Match both block and inline equations
        patterns = {
            'block': r'\$\$(.*?)\$\$',
            'inline': r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)'
        }

        for eq_type, pattern in patterns.items():
            for match in re.finditer(pattern, content, flags=re.DOTALL):
                equation_id = f"eq-{len(equations)}"
                equations.append(LaTeXEquation(
                    id=equation_id,
                    content=match.group(1).strip(),
                    is_block=(eq_type == 'block'),
                    metadata={
                        'position': match.span(),
                        'original': match.group(0)
                    }
                ))
                self.logger.debug(
                    f"Found {eq_type} equation {equation_id}: "
                    f"{match.group(1)[:50]}..."
                )

        return equations

    ### VALIDATION ###
    def validate_content(self, content: str) -> List[str]:
        """Validate DITA content and return any errors."""
        errors = []
        try:
            # Try parsing with strict parser first
            strict_parser = etree.XMLParser(recover=False)
            etree.fromstring(content.encode('utf-8'), strict_parser)
        except etree.XMLSyntaxError as e:
            errors.append(f"Line {e.lineno}, Column {e.offset}: {e.msg}")
            # Get the problematic line
            lines = content.splitlines()
            if 0 <= e.lineno - 1 < len(lines):
                errors.append(f"Line content: {lines[e.lineno - 1]}")
        return errors

    ### HELPER METHODS ###
    def _determine_element_type(self, path: Path) -> ElementType:
        """Determine element type from file extension."""
        suffix = path.suffix.lower()
        if suffix == '.dita':
            return ElementType.DITA
        elif suffix == '.md':
            return ElementType.MARKDOWN
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _resolve_topic_path(self, href: Path, map_path: Path) -> Optional[Path]:
        """Resolve topic path relative to map location."""
        try:
            if href.is_absolute():
                return href
            return (map_path.parent / href).resolve()
        except Exception as e:
            self.logger.error(f"Error resolving path: {str(e)}")
            return None

    def _extract_map_metadata(self, tree: etree._Element) -> MetadataDict:
        """Extract metadata from map."""
        metadata: MetadataDict = {}
        try:
            # Extract map-level metadata
            if metadata_elem := tree.find(".//metadata"):
                # Add metadata extraction logic here
                pass
        except Exception as e:
            self.logger.error(f"Error extracting map metadata: {str(e)}")
        return metadata

    def _extract_topicref_metadata(self, topicref: etree._Element) -> MetadataDict:
        """Extract metadata from topicref."""
        metadata: MetadataDict = {}
        try:
            # Get navtitle
            if navtitle := topicref.find(".//navtitle"):
                metadata['navtitle'] = navtitle.text

            # Get processing attributes
            for attr in ['processing-role', 'format', 'scope']:
                if value := topicref.get(attr):
                    metadata[attr] = value

        except Exception as e:
            self.logger.error(f"Error extracting topicref metadata: {str(e)}")
        return metadata
