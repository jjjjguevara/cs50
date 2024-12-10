from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import logging
from app.dita.utils.id_handler import DITAIDHandler
from app.dita.utils.metadata import MetadataHandler
from ..models.types import (
    DITAElementType,
    DITAElementInfo,
    ElementType,
    ParsedElement,
    ParsedMap,
    ProcessedContent,
    ElementAttributes,
    DITAElementContext,
    HeadingContext,
    ParsedMap,
    ProcessingError
)
from ..utils.heading import HeadingHandler
from app_config import DITAConfig
import xml.etree.ElementTree as ET
from lxml import etree





class DITAParser:
    """Parser for DITA maps and topics, handling headings and metadata extraction."""

    def __init__(self, config: Optional[DITAConfig] = None):
        """
        Initialize the DITAParser with an optional configuration.

        Args:
            config: Optional configuration object for the parser.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or DITAConfig()
        self.heading_handler = HeadingHandler()
        self.id_handler = DITAIDHandler()
        self.metadata = MetadataHandler()
        self.heading_handler = HeadingHandler()


    def parse_ditamap(self, map_path: Path) -> ParsedMap:
        """
        Parse a DITA map, extracting topics, metadata, and title.

        Args:
            map_path: Path to the DITA map file.

        Returns:
            A ParsedMap object representing the parsed map.
        """
        try:
            self.logger.debug(f"Parsing DITA map: {map_path}")

            if not map_path.exists() or not map_path.is_file():
                raise FileNotFoundError(f"DITA map not found: {map_path}")

            # Load and parse XML content
            dita_map_content = map_path.read_text(encoding="utf-8")
            root = ET.fromstring(dita_map_content)

            # Extract title
            title_elem = root.find("title")
            title = title_elem.text if title_elem is not None else "Untitled"

            # Extract metadata including index-numbers setting
            metadata = {
                "id": map_path.stem,
                "content_type": "ditamap",
                "processed_at": datetime.now().isoformat()
            }
            metadata_elem = root.find("metadata")
            if metadata_elem is not None:
                for othermeta in metadata_elem.findall("othermeta"):
                    name = othermeta.get("name")
                    content = othermeta.get("content")
                    if name and content:
                        metadata[name] = content # keep as string

            # Parse topics
            topics = []
            for topicref in root.findall(".//topicref"):
                href = topicref.get("href")
                if not href:
                    continue

                topic_id = topicref.get("id") or Path(href).stem
                topic_path = map_path.parent / href

                # Determine correct element type based on file extension
                element_type = (
                    ElementType.MARKDOWN if href.endswith('.md')
                    else ElementType.DITA if href.endswith('.dita')
                    else ElementType.UNKNOWN
                )

                topics.append(
                    ParsedElement(
                        id=topic_id,
                        topic_id=topic_id,
                        type=element_type,
                        content="",
                        topic_path=topic_path,
                        source_path=map_path,
                        metadata={}
                    )
                )

            parsed_map = ParsedMap(
                title=title,
                topics=topics,
                metadata=metadata,
                source_path=map_path
            )

            self.logger.debug(f"Parsed map: {parsed_map}")
            return parsed_map

        except Exception as e:
            self.logger.error(f"Error parsing DITA map: {str(e)}", exc_info=True)
            raise ProcessingError(
                error_type="parsing",
                message=str(e),
                context=str(map_path)
            )

    def parse_topic(self, topic_path: Path) -> ParsedElement:
        """Parse a topic file into a ParsedElement."""
        try:
            # Read the file content
            content = topic_path.read_text()

            # Determine the element type
            element_type = (
                ElementType.MARKDOWN if topic_path.suffix == '.md'
                else ElementType.DITA
            )

            # Detect LaTeX content
            has_latex = self._detect_latex(content)

            # Create metadata
            metadata = {
                'has_latex': has_latex,
                'latex_blocks': self._count_latex_blocks(content) if has_latex else 0,
                'content_type': element_type.value
            }

            # Return ParsedElement without preprocessing
            return ParsedElement(
                id=self.id_handler.generate_id(str(topic_path)),
                topic_id=topic_path.stem,
                type=element_type,
                content=content,  # Pass raw content
                topic_path=topic_path,
                source_path=topic_path,
                metadata=metadata
            )
        except Exception as e:
            self.logger.error(f"Error parsing topic {topic_path}: {str(e)}")
            raise

    def _detect_latex(self, content: str) -> bool:
            """Detect if content contains LaTeX equations."""
            import re
            latex_patterns = [
                r'\$\$(.*?)\$\$',  # Display math
                r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)'  # Inline math
            ]
            return any(re.search(pattern, content, re.DOTALL) for pattern in latex_patterns)

    def _count_latex_blocks(self, content: str) -> dict:
        """Count LaTeX blocks in content."""
        import re
        display_count = len(re.findall(r'\$\$(.*?)\$\$', content, re.DOTALL))
        inline_count = len(re.findall(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', content))
        return {
            'display': display_count,
            'inline': inline_count
        }


    def parse_markdown_file(self, file_path: str) -> ParsedElement:
        """
        Parse a Markdown file and extract raw YAML metadata and content.

        Args:
            file_path: Path to the Markdown file as a string.

        Returns:
            ParsedElement containing the raw metadata and Markdown content.
        """
        try:
            file_path_obj = Path(file_path)  # Convert to Path object

            # Normalize the file path to avoid redundant directory structure
            base_dir = file_path_obj.parent
            file_path_obj = base_dir / file_path_obj.name

            with open(file_path_obj, 'r', encoding='utf-8') as f:
                file_content = f.read()  # Read file content as a string

            # Extract YAML metadata
            metadata, content = self.metadata.extract_yaml_metadata(file_content)

            # Return metadata and raw content
            return ParsedElement(
                id=self.id_handler.generate_content_id(file_path_obj),
                topic_id=file_path_obj.stem,
                type=ElementType.MARKDOWN,
                content=content,  # Pass raw Markdown content
                topic_path=file_path_obj,  # Ensure Path object
                source_path=file_path_obj,  # Ensure Path object
                metadata=metadata,  # Pass raw YAML metadata
            )
        except Exception as e:
            self.logger.error(f"Error parsing Markdown file {file_path}: {e}")
            raise

    def parse_xml_content(self, content: str) -> etree._Element:
        """Parse raw XML content into etree."""
        try:
            content_bytes = content.encode('utf-8') if isinstance(content, str) else content
            parser = etree.XMLParser(remove_blank_text=True)
            return etree.fromstring(content_bytes, parser=parser)
        except Exception as e:
            self.logger.error(f"Error parsing XML content: {str(e)}")
            raise

    def _extract_topics(self, dita_map_content: str, map_path: Path) -> List[ParsedElement]:
        try:
            root = ET.fromstring(dita_map_content)
            topics = []

            for topicref in root.findall(".//topicref"):
                href = topicref.get("href")
                if not href:
                    continue

                topic_id = topicref.get("id") or Path(href).stem
                self.logger.debug(f"Extracted topic: id={topic_id}, href={href}")

                # Normalize the path to ensure no redundancy
                # If the href is relative, it should be resolved relative to the map_path's parent directory
                topic_path = map_path.parent / href

                # Ensure the path is normalized without redundant segments
                topic_path = topic_path.resolve()

                # If the topic_path has unexpected directories, normalize them
                if topic_path.parts[-2] == "maps" and topic_path.parts[-3] == "app":
                    topic_path = Path(*topic_path.parts[3:])

                topics.append(
                    ParsedElement(
                        id=topic_id,
                        topic_id=topic_id,
                        type=ElementType.DITAMAP,
                        content="",
                        topic_path=topic_path,
                        source_path=map_path,
                        metadata={},
                    )
                )
            return topics
        except ET.ParseError as e:
            self.logger.error(f"XML parsing error for {map_path}: {str(e)}", exc_info=True)
            raise


    def _extract_elements(self, topic_content: str, topic_path: Path) -> List[ParsedElement]:
        """
        Extract elements (including headings) from topic content.

        Args:
            topic_content: Raw content of the topic file.
            topic_path: Path to the topic file.

        Returns:
            A list of ParsedElement objects representing the topic content.
        """
        try:
            self.logger.debug(f"Extracting elements from topic: {topic_path}")

            elements = []
            self.heading_handler.init_state()  # Initialize heading state for this topic

            # Normalize the topic path to avoid redundancies
            base_dir = topic_path.parent
            topic_path = base_dir / topic_path.name  # Resolve file path

            # Process each line to extract potential elements
            for i, line in enumerate(topic_content.splitlines(), start=1):
                # Skip empty lines
                if not line.strip():
                    continue

                # Determine heading level or treat as body content
                if line.startswith("#"):  # Example: Markdown-style headings
                    heading_level = line.count("#")
                    heading_text = line.lstrip("#").strip()

                    heading_id, processed_heading = self.heading_handler.process_heading(
                        text=heading_text, level=heading_level
                    )

                    element = ParsedElement(
                        id=heading_id,
                        topic_id=f"topic-{topic_path.stem}",
                        type=ElementType.HEADING,
                        content=processed_heading,
                        topic_path=topic_path,
                        source_path=topic_path,
                        metadata={"heading_level": heading_level},
                    )
                    elements.append(element)

                else:  # Treat as body content
                    body_id = self.id_handler.generate_id(f"line-{i}-{topic_path.stem}")
                    element = ParsedElement(
                        id=body_id,
                        topic_id=f"topic-{topic_path.stem}",
                        type=ElementType.BODY,
                        content=line.strip(),
                        topic_path=topic_path,
                        source_path=topic_path,
                        metadata={},
                    )
                    elements.append(element)

            self.logger.debug(f"Extracted {len(elements)} elements from topic: {topic_path}")
            return elements

        except Exception as e:
            self.logger.error(f"Error extracting elements from topic {topic_path}: {str(e)}", exc_info=True)
            raise ProcessingError(
                error_type="element_extraction",
                message=f"Failed to extract elements from topic: {str(e)}",
                context=str(topic_path),
            )
