from datetime import datetime
from pathlib import Path
import sqlite3
from contextlib import contextmanager
import logging
import json
from typing import List, Dict, Optional, Generator, Any
from app.dita.utils.id_handler import DITAIDHandler
from app.dita.utils.metadata import MetadataHandler
from ..models.types import (
    TrackedElement,
    DITAElementType,
    DITAElementInfo,
    ElementType,
    ProcessingPhase,
    ProcessingState,
    ProcessedContent,
    ProcessingError,
    ProcessingMetadata
)
from ..utils.heading import HeadingHandler
from app_config import DITAConfig
import xml.etree.ElementTree as ET
from lxml import etree


class DITAParser:
    """Parser for DITA maps and topics, handling headings and metadata extraction."""

    def __init__(
        self,
        db_path: Path,
        root_path: Path,
        config: Optional[DITAConfig] = None,
        metadata: Optional[ProcessingMetadata] = None,
        id_handler: Optional[DITAIDHandler] = None,
    ):
        """
        Initialize the DITAParser with an optional configuration.

        Args:
            db_path: Path to the SQLite database.
            root_path: Root path for DITA content.
            config: Optional configuration object for the parser.
            metadata: Optional processing metadata for element tracking.
            id_handler: Optional ID handler instance for generating IDs.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or DITAConfig()
        self.root_path = root_path
        self.db_path = db_path

        # Initialize metadata
        self.metadata = metadata or ProcessingMetadata(
            id="parser-metadata",
            content_type=ElementType.DITAMAP,
            features={
                "index_numbers": True,
                "toc": True,
                "enable_cross_refs": True,
                "number_headings": True,
            },
        )

        # Initialize ID handler
        self.id_handler = id_handler or DITAIDHandler()

        # Initialize metadata and heading handlers
        self.metadata_handler = MetadataHandler()
        self.heading_handler = HeadingHandler(processing_metadata=self.metadata)

        # Initialize the database connection
        self._init_db()



    def _init_db(self) -> None:
        """Initialize database connection and schema."""
        try:
            with self._get_db() as conn:
                schema_path = Path(__file__).parent / "metadata.sql"
                schema = schema_path.read_text()
                conn.executescript(schema)
                self.logger.debug("Database schema initialized")
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            raise

    @contextmanager
    def _get_db(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection with context management."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise
        finally:
            conn.close()

    def parse_ditamap(self, map_path: Path) -> TrackedElement:
       """Parse a DITA map and extract metadata and key definitions."""
       try:
           self.logger.debug(f"Parsing DITA map: {map_path}")
           if not map_path.exists():
               raise FileNotFoundError(f"DITA map not found: {map_path}")

           # Generate map ID and initialize metadata
           map_element = TrackedElement.create_map(
               path=map_path,
               title=self._extract_map_title(map_path),
               id_handler=self.id_handler
           )

           # Extract base metadata
           map_metadata = self.metadata_handler.extract_metadata(map_path, map_element.id)

           # Extract and store key definitions
           key_definitions = self._extract_key_definitions(map_element)
           map_metadata["key_definitions"] = key_definitions

           # Enrich metadata with relational context and keys
           map_metadata = self._enrich_map_metadata(map_metadata)
           map_element.metadata = map_metadata

           # Store complete metadata
           self._store_map_metadata(map_element)

           return map_element

       except Exception as e:
           self.logger.error(f"Error parsing DITA map: {str(e)}")
           raise

    def parse_topic(self, topic_path: Path, map_metadata: Dict[str, Any]) -> TrackedElement:
        """Parse a DITA topic and extract metadata."""
        try:
            self.logger.debug(f"Parsing DITA topic: {topic_path}")
            if not topic_path.exists():
                raise FileNotFoundError(f"DITA topic not found: {topic_path}")

            # Generate topic ID and initialize metadata
            topic_element = TrackedElement.from_discovery(
                path=topic_path,
                element_type=ElementType.DITA,
                id_handler=self.id_handler
            )
            topic_metadata = self.metadata_handler.extract_metadata(topic_path, topic_element.id, map_metadata=map_metadata)

            # Enrich metadata with relational context
            topic_metadata = self._enrich_topic_metadata(topic_metadata, map_metadata)
            topic_element.metadata = topic_metadata

            # Store metadata persistently
            self._store_topic_metadata(topic_element)

            return topic_element

        except Exception as e:
            self.logger.error(f"Error parsing DITA topic: {str(e)}")
            raise

    def _extract_key_definitions(self, map_element: TrackedElement) -> Dict[str, Any]:
        """Extract key definitions from DITA map."""
        try:
            tree = etree.parse(str(map_element.path))
            keydefs = {}

            for keydef in tree.xpath(".//keydef"):
                key = keydef.get("keys")
                if not key:
                    continue

                keydefs[key] = {
                    "href": keydef.get("href"),
                    "alt": keydef.get("alt"),
                    "placement": keydef.get("placement"),
                    "scale": keydef.get("scale"),
                    "props": keydef.get("props"),
                    "audience": keydef.get("audience"),
                    "platform": keydef.get("platform"),
                    "product": keydef.get("product"),
                    "otherprops": keydef.get("otherprops"),
                    "conref": keydef.get("conref"),
                    "keyref": keydef.get("keyref"),
                    "rev": keydef.get("rev"),
                    "outputclass": keydef.get("outputclass"),
                    "align": keydef.get("align"),
                    "scalefit": keydef.get("scalefit"),
                    "width": keydef.get("width"),
                    "height": keydef.get("height")
                }

                # Extract topicmeta if present
                topicmeta = keydef.find("topicmeta")
                if topicmeta is not None:
                    keydefs[key].update({
                        "linktext": getattr(topicmeta.find("linktext"), "text", None),
                        "searchtitle": getattr(topicmeta.find("searchtitle"), "text", None)
                    })

            return keydefs

        except Exception as e:
            self.logger.error(f"Error extracting key definitions: {str(e)}")
            return {}

    def _store_map_metadata(self, map_element: TrackedElement) -> None:
            """Store enriched map metadata in the database."""
            try:
                self.metadata_handler.store_metadata(map_element.id, map_element.metadata)
            except Exception as e:
                self.logger.error(f"Error storing map metadata: {str(e)}")
                raise

    def _store_topic_metadata(self, topic_element: TrackedElement) -> None:
        """Store enriched topic metadata in the database."""
        try:
            self.metadata_handler.store_metadata(topic_element.id, topic_element.metadata)
        except Exception as e:
            self.logger.error(f"Error storing topic metadata: {str(e)}")
            raise

    def _enrich_map_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
            """Enrich map metadata with relational and feature information."""
            metadata.setdefault("related_topics", [])
            metadata.setdefault("prerequisites", [])
            metadata.setdefault("feature_flags", {
                "enable_toc": True,
                "enable_cross_refs": True
            })
            return metadata

    def _enrich_topic_metadata(self, metadata: Dict[str, Any], map_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich topic metadata with relational and feature information."""
        metadata.setdefault("related_topics", map_metadata.get("related_topics", []))
        metadata.setdefault("prerequisites", map_metadata.get("prerequisites", []))
        metadata.setdefault("feature_flags", map_metadata.get("feature_flags", {}))
        return metadata


    def _extract_map_title(self, map_path: Path) -> str:
        """Extract the title of a DITA map from its content."""
        try:
            # Parse the DITA map file
            with map_path.open(encoding="utf-8") as file:
                tree = etree.parse(file)
                title_element = tree.find(".//title")  # Locate the title element
                if title_element is not None and title_element.text:
                    return title_element.text.strip()  # Safely strip if text exists
            return "Untitled Map"  # Default title if no title element or text found
        except Exception as e:
            self.logger.error(f"Error extracting title from {map_path}: {str(e)}")
            return "Untitled Map"

    def _determine_element_type(self, path: str) -> ElementType:
        """Determine element type from file path."""
        if path.endswith('.md'):
            return ElementType.MARKDOWN
        elif path.endswith('.dita'):
            return ElementType.DITA
        return ElementType.UNKNOWN


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

    def parse_markdown_file(self, file_path: str) -> TrackedElement:
        """Parse a Markdown file and extract YAML metadata and content."""
        try:
            file_path_obj = Path(file_path).resolve()

            # Create initial tracked element
            element = TrackedElement.from_discovery(
                path=file_path_obj,
                element_type=ElementType.MARKDOWN,
                id_handler=self.id_handler
            )
            element.topic_id = file_path_obj.stem  # Assign topic ID separately

            # Read and parse content
            with open(file_path_obj, 'r', encoding='utf-8') as f:
                file_content = f.read()

            # Extract YAML metadata
            metadata, content = self.metadata_handler.extract_yaml_metadata(file_content)
            element.content = content
            element.metadata.update(metadata)

            # Update element state
            element.phase = ProcessingPhase.VALIDATION

            # Store in database
            self._store_topic_metadata(element)

            return element

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

    def _extract_topics(self, dita_map_content: str, map_path: Path) -> List[TrackedElement]:
        """Extract topic elements from DITA map content."""
        try:
            root = ET.fromstring(dita_map_content)
            topics = []

            for topicref in root.findall(".//topicref"):
                if href := topicref.get("href"):
                    topic_id = topicref.get("id") or Path(href).stem
                    self.logger.debug(f"Extracted topic: id={topic_id}, href={href}")

                    # Normalize topic path
                    topic_path = (map_path.parent / href).resolve()

                    # Create tracked element for topic
                    topic = TrackedElement.from_discovery(
                        path=topic_path,
                        element_type=self._determine_element_type(href),
                        id_handler=self.id_handler
                    )
                    topic.topic_id = topic_id  # Assign topic ID separately

                    # Set relationship to map
                    topic.parent_map_id = map_path.stem
                    topic.href = href
                    topic.sequence_number = len(topics)  # Order in map

                    # Extract any topicref metadata
                    topic.metadata.update({
                        'type': topicref.get('type', 'topic'),
                        'scope': topicref.get('scope', 'local'),
                        'format': topicref.get('format', None),
                        'processing-role': topicref.get('processing-role', 'normal')
                    })

                    # Store in database
                    self._store_topic_metadata(topic)

                    topics.append(topic)

            return topics

        except ET.ParseError as e:
            self.logger.error(f"XML parsing error for {map_path}: {str(e)}", exc_info=True)
            raise


    def _extract_elements(self, topic_content: str, topic_path: Path) -> List[TrackedElement]:
        """Extract and track elements from topic content."""
        try:
            self.logger.debug(f"Extracting elements from topic: {topic_path}")
            elements = []
            self.heading_handler.init_state()

            topic_path = topic_path.resolve()
            current_section = None

            with self._get_db() as conn:
                # Start element extraction
                for i, line in enumerate(topic_content.splitlines(), start=1):
                    if not line.strip():
                        continue

                    if line.startswith("#"):  # Headings
                        level = line.count("#")
                        text = line.lstrip("#").strip()

                        # Create heading element
                        element = TrackedElement.from_discovery(
                            path=topic_path,
                            element_type=ElementType.HEADING,
                            id_handler=self.id_handler
                        )

                        # Process heading
                        heading_id, processed_text = self.heading_handler.process_heading(
                            text=text, level=level
                        )

                        element.id = heading_id
                        element.content = processed_text
                        element.metadata.update({
                            "heading_level": level,
                            "original_text": text,
                            "sequence_number": len(elements)
                        })

                        # Update section tracking
                        current_section = element.id

                        # Store heading in database
                        conn.execute("""
                            INSERT INTO topic_elements (
                                element_id, topic_id, element_type,
                                parent_element_id, sequence_num, content_hash
                            ) VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            element.id,
                            topic_path.stem,
                            "heading",
                            None,
                            element.metadata["sequence_number"],
                            hash(element.content)
                        ))

                        # Store in heading index
                        conn.execute("""
                            INSERT INTO heading_index (
                                id, topic_id, map_id, text,
                                level, sequence_number, path_fragment
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            element.id,
                            topic_path.stem,
                            element.parent_map_id,
                            text,
                            level,
                            str(self.heading_handler._state.counters[f'h{level}']),
                            element.id
                        ))

                    else:  # Body content
                        element = TrackedElement.from_discovery(
                            path=topic_path,
                            element_type=ElementType.BODY,
                            id_handler=self.id_handler
                        )

                        element.content = line.strip()
                        element.metadata.update({
                            "line_number": i,
                            "section_id": current_section,
                            "sequence_number": len(elements)
                        })

                        # Store body element in database
                        conn.execute("""
                            INSERT INTO topic_elements (
                                element_id, topic_id, element_type,
                                parent_element_id, sequence_num, content_hash
                            ) VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            element.id,
                            topic_path.stem,
                            "body",
                            current_section,
                            element.metadata["sequence_number"],
                            hash(element.content)
                        ))

                    # Store element context
                    conn.execute("""
                        INSERT INTO element_context (
                            element_id, context_type, parent_context,
                            level, xpath
                        ) VALUES (?, ?, ?, ?, ?)
                    """, (
                        element.id,
                        element.type.value,
                        current_section,
                        element.metadata.get("heading_level", 0),
                        f"/topic/{topic_path.stem}/{element.type.value}[{len(elements)}]"
                    ))

                    elements.append(element)

            self.logger.debug(f"Extracted {len(elements)} elements from topic: {topic_path}")
            return elements

        except Exception as e:
            self.logger.error(f"Error extracting elements: {str(e)}", exc_info=True)
            raise ProcessingError(
                error_type="element_extraction",
                message=f"Failed to extract elements: {str(e)}",
                context=str(topic_path)
            )
