from datetime import datetime
from pathlib import Path
import sqlite3
from contextlib import contextmanager
import logging
import json
from typing import List, Dict, Optional, Generator
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
    ProcessingError
)
from ..utils.heading import HeadingHandler
from app_config import DITAConfig
import xml.etree.ElementTree as ET
from lxml import etree


class DITAParser:
    """Parser for DITA maps and topics, handling headings and metadata extraction."""

    def __init__(self, db_path: Path, root_path: Path, config: Optional[DITAConfig] = None):
        """
        Initialize the DITAParser with an optional configuration.

        Args:
            config: Optional configuration object for the parser.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or DITAConfig()
        self.root_path = root_path
        self.db_path = db_path
        self._init_db()
        self.heading_handler = HeadingHandler()
        self.id_handler = DITAIDHandler()
        self.metadata = MetadataHandler()
        self.heading_handler = HeadingHandler()


    def _init_db(self) -> None:
            """Initialize database connection and schema."""
            try:
                with self._get_db() as conn:
                    # Read and execute schema
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

    def _store_map_metadata(self, map_element: TrackedElement) -> None:
            """Store map metadata in database."""
            try:
                with self._get_db() as conn:
                    # Insert into maps table
                    conn.execute("""
                        INSERT INTO maps (
                            map_id, title, file_path, version, status,
                            language, toc_enabled, index_numbers_enabled,
                            context_root, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(map_id) DO UPDATE SET
                            title=excluded.title,
                            updated_at=excluded.updated_at
                    """, (
                        map_element.id,
                        map_element.title,
                        str(map_element.path),
                        map_element.metadata.get('version', '1.0'),
                        map_element.metadata.get('status', 'draft'),
                        map_element.metadata.get('language', 'en'),
                        map_element.metadata.get('toc_enabled', True),
                        map_element.metadata.get('index_numbers_enabled', True),
                        str(map_element.path.parent),
                        map_element.created_at.isoformat(),
                        map_element.last_updated.isoformat() if map_element.last_updated else None
                    ))

                    # Store processing context
                    conn.execute("""
                        INSERT INTO processing_contexts (
                            content_id, content_type, phase, state,
                            features, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        map_element.id,
                        'map',
                        map_element.phase.value,
                        map_element.state.value,
                        json.dumps(map_element.metadata.get('features', {})),
                        map_element.created_at.isoformat()
                    ))

                    # Store feature flags
                    for name, value in map_element.metadata.items():
                        if name.startswith('feature_'):
                            conn.execute("""
                                INSERT INTO content_flags (
                                    content_id, name, value, scope
                                ) VALUES (?, ?, ?, ?)
                            """, (
                                map_element.id,
                                name.replace('feature_', ''),
                                str(value),
                                'map'
                            ))

            except Exception as e:
                self.logger.error(f"Error storing map metadata: {str(e)}")
                raise

    def _store_topic_metadata(self, topic_element: TrackedElement) -> None:
        """Store topic metadata in database."""
        try:
            with self._get_db() as conn:
                # Get topic type ID
                cur = conn.execute("""
                    SELECT type_id FROM topic_types
                    WHERE name = ?
                """, (topic_element.metadata.get('topic_type', 'topic'),))
                type_id = cur.fetchone()[0]

                # Insert into topics table
                conn.execute("""
                    INSERT INTO topics (
                        id, title, path, type_id, content_type,
                        short_desc, parent_topic_id, root_map_id,
                        specialization_type, created_at, updated_at,
                        published_version, status, language
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        updated_at=excluded.updated_at,
                        status=excluded.status
                """, (
                    topic_element.id,
                    topic_element.title,
                    str(topic_element.path),
                    type_id,
                    topic_element.type.value,
                    topic_element.metadata.get('short_desc'),
                    topic_element.metadata.get('parent_topic_id'),
                    topic_element.parent_map_id,
                    topic_element.metadata.get('specialization_type'),
                    topic_element.created_at.isoformat(),
                    topic_element.last_updated.isoformat() if topic_element.last_updated else None,
                    topic_element.metadata.get('version'),
                    topic_element.metadata.get('status', 'draft'),
                    topic_element.metadata.get('language', 'en')
                ))

                # Store context hierarchy
                if topic_element.parent_map_id:
                    conn.execute("""
                        INSERT INTO context_hierarchy (
                            map_id, topic_id, parent_id, level,
                            sequence_num, context_path
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        topic_element.parent_map_id,
                        topic_element.id,
                        topic_element.metadata.get('parent_topic_id'),
                        topic_element.metadata.get('level', 1),
                        topic_element.sequence_number or 0,
                        f"{topic_element.parent_map_id}/{topic_element.id}"
                    ))

                # Store references
                if refs := topic_element.metadata.get('references', []):
                    for ref in refs:
                        conn.execute("""
                            INSERT INTO content_references (
                                ref_id, source_id, target_id,
                                ref_type, text, href
                            ) VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            self.id_handler.generate_id(f"ref-{ref['target']}"),
                            topic_element.id,
                            ref['target'],
                            ref.get('type', 'internal'),
                            ref.get('text'),
                            ref['href']
                        ))

        except Exception as e:
            self.logger.error(f"Error storing topic metadata: {str(e)}")
            raise


    def parse_ditamap(self, map_path: Path) -> TrackedElement:
        """Parse a DITA map into a tracked element."""
        try:
            self.logger.debug(f"Parsing DITA map: {map_path}")

            if not map_path.exists() or not map_path.is_file():
                raise FileNotFoundError(f"DITA map not found: {map_path}")

            # Create initial map element
            map_element = TrackedElement.create_map(
                path=map_path,
                title="Untitled",  # Will be updated
                id_handler=self.id_handler
            )

            # Load and parse XML content
            dita_map_content = map_path.read_text(encoding="utf-8")
            root = ET.fromstring(dita_map_content)

            # Extract title
            if title_elem := root.find("title"):
                map_element.title = title_elem.text

            # Extract metadata including feature flags
            map_element.metadata.update({
                "content_type": "ditamap",
                "processed_at": datetime.now().isoformat(),
                "version": "1.0",  # Default version
                "language": "en",   # Default language
                "status": "draft",  # Default status
            })

            # Process metadata element
            if metadata_elem := root.find("metadata"):
                for othermeta in metadata_elem.findall("othermeta"):
                    name = othermeta.get("name")
                    content = othermeta.get("content")
                    if name and content:
                        map_element.metadata[name] = content

            # Parse topics
            for topicref in root.findall(".//topicref"):
                if href := topicref.get("href"):
                    topic_id = topicref.get("id") or Path(href).stem
                    topic_path = map_path.parent / href

                    # Create topic element
                    topic_element = TrackedElement.from_discovery(
                        path=topic_path,
                        element_type=self._determine_element_type(href),
                        id_handler=self.id_handler,
                        topic_id=topic_id
                    )
                    topic_element.parent_map_id = map_element.id
                    topic_element.href = href

                    # Track in map
                    map_element.topics.append(topic_element.id)
                    map_element.by_type.setdefault(
                        topic_element.type.value,
                        []
                    ).append(topic_element.id)

                    # Store in database
                    self._store_topic_metadata(topic_element)

            # Store map metadata in database
            self._store_map_metadata(map_element)

            map_element.phase = ProcessingPhase.VALIDATION
            return map_element

        except Exception as e:
            self.logger.error(f"Error parsing DITA map: {str(e)}", exc_info=True)
            raise ProcessingError(
                error_type="parsing",
                message=str(e),
                context=str(map_path)
            )

    def parse_topic(self, topic_path: Path) -> TrackedElement:
        """Parse a topic file into a TrackedElement."""
        try:
            # Create topic element
            topic_element = TrackedElement.from_discovery(
                path=topic_path,
                element_type=self._determine_element_type(str(topic_path)),
                id_handler=self.id_handler
            )

            # Read content
            topic_element.content = topic_path.read_text()

            # Extract metadata
            topic_element.metadata.update({
                'has_latex': self._detect_latex(topic_element.content),
                'content_type': topic_element.type.value,
                'status': 'draft',
                'language': 'en',
                'version': '1.0'
            })

            # Add LaTeX info if needed
            if topic_element.metadata['has_latex']:
                topic_element.metadata['latex_blocks'] = self._count_latex_blocks(topic_element.content)

            # Store in database
            self._store_topic_metadata(topic_element)

            topic_element.phase = ProcessingPhase.VALIDATION
            return topic_element

        except Exception as e:
            self.logger.error(f"Error parsing topic {topic_path}: {str(e)}")
            raise

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
                id_handler=self.id_handler,
                topic_id=file_path_obj.stem
            )

            # Read and parse content
            with open(file_path_obj, 'r', encoding='utf-8') as f:
                file_content = f.read()

            # Extract YAML metadata
            metadata, content = self.metadata.extract_yaml_metadata(file_content)
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
                        id_handler=self.id_handler,
                        topic_id=topic_id
                    )

                    # Set relationship to map
                    topic.parent_map_id = map_path.stem
                    topic.href = href
                    topic.sequence_number = len(topics)  # Order in map

                    # Extract any topicref metadata
                    topic.metadata.update({
                        'type': topicref.get('type', 'topic'),
                        'scope': topicref.get('scope', 'local'),
                        'format': topicref.get('format'),
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
