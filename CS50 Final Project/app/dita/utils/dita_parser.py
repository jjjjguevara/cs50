# app/dita/utils/dita_parser.py

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
    ParsedMap,
    LaTeXEquation,
    PathLike,
    MetadataDict
)

class DITAParser:
    """Comprehensive parser for DITA, Markdown, and LaTeX content."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.parser = etree.XMLParser(
            recover=True,
            remove_blank_text=True,
            resolve_entities=False,
            dtd_validation=False,
            load_dtd=False,
            no_network=True
        )

    ### MAP PARSING ###
    def parse_map(self, map_path: Path) -> Optional[ParsedMap]:
        """Initial parse of ditamap for content discovery."""
        try:
            self.logger.info(f"Parsing map: {map_path}")

            tree = self.parse_file(map_path)
            if tree is None:
                return None

            # Get topics
            topics = []
            for topicref in tree.xpath(".//topicref"):
                href = topicref.get('href')
                if href:
                    topic_path = Path(href)
                    # Log the topic path being processed
                    self.logger.debug(f"Processing topicref href: {href}")

                    # Get absolute path
                    full_path = (map_path.parent / topic_path).resolve()
                    self.logger.debug(f"Resolved topic path: {full_path}")

                    if full_path.exists():
                        element_type = ElementType.MARKDOWN if topic_path.suffix == '.md' else ElementType.DITA

                        # For markdown files, try to read content
                        if element_type == ElementType.MARKDOWN:
                            with open(full_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                self.logger.debug(f"Markdown content length: {len(content)}")

                        topics.append(ParsedElement(
                            id=topic_path.stem,
                            type=element_type,
                            content=str(full_path),
                            source_path=full_path,
                            metadata={}
                        ))
                        self.logger.debug(f"Added topic: {topic_path.stem} ({element_type})")
                    else:
                        self.logger.warning(f"Topic file not found: {full_path}")

            return ParsedMap(
                title=None,
                topics=topics,
                metadata={},
                source_path=map_path
            )

        except Exception as e:
            self.logger.error(f"Error parsing map {map_path}: {str(e)}")
            return None

    def _parse_topicref(self, topicref: etree._Element, map_path: Path) -> Optional[ParsedElement]:
        """Parse individual topicref element."""
        try:
            href = topicref.get('href')
            if not href:
                return None

            topic_path = self._resolve_topic_path(Path(href), map_path)
            if not topic_path or not topic_path.exists():
                self.logger.warning(f"Topic not found: {href}")
                return None

            # Generate ID for topic
            topic_id = f"topic-{href.replace('/', '-').replace('.', '-')}"

            # Determine content type
            element_type = self._determine_element_type(topic_path)

            # Get basic metadata from topicref
            metadata = self._extract_topicref_metadata(topicref)

            return ParsedElement(
                id=topic_id,  # Add ID
                type=element_type,
                content=str(topic_path),
                source_path=topic_path,
                metadata=metadata
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
            return self.parse_content(content)
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
