from pathlib import Path
import sqlite3
import logging
import re
from typing import List, Dict, Optional, Generator, Any
from contextlib import contextmanager
from lxml import etree
import xml.etree.ElementTree as ET

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

from ..utils.id_handler import DITAIDHandler
from ..utils.metadata import MetadataHandler
from ..utils.heading import HeadingHandler
from ..event_manager import EventManager
from ..config_manager import DITAConfig
from ..utils.cache import ContentCache

class DITAParser:
    """Parser for DITA maps and topics with event-driven processing."""

    def __init__(
        self,
        db_path: Path,
        root_path: Path,
        event_manager: EventManager,
        content_cache: ContentCache,
        config: Optional[DITAConfig] = None,
        metadata: Optional[ProcessingMetadata] = None,
        id_handler: Optional[DITAIDHandler] = None,
    ):
        """
        Initialize parser with event management and caching.

        Args:
            db_path: Path to SQLite database
            root_path: Root path for DITA content
            event_manager: Event management system
            content_cache: Content caching system
            config: Optional configuration settings
            metadata: Optional processing metadata
            id_handler: Optional ID generation handler
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or DITAConfig()
        self.root_path = root_path
        self.db_path = db_path

        # Initialize managers
        self.event_manager = event_manager
        self.content_cache = content_cache
        self.id_handler = id_handler or DITAIDHandler()
        self.metadata_handler = MetadataHandler(event_manager=event_manager, content_cache=content_cache, config=config)
        self.heading_handler = HeadingHandler(event_manager=event_manager)


        # Initialize the database
        self._init_db()

        # Initialize database connection
        self._conn: Optional[sqlite3.Connection] = None

        # Custom element extraction patterns
        self._element_extractors = {
           # Headings
           r"^#{1,6}\s": {
               "type": ElementType.HEADING,
               "extract": lambda match, line: {
                   "level": len(match.group().strip()),
                   "content": line.lstrip("#").strip()
               }
           },

           # Links
           r"\[\[(.+?)\]\]": {
               "type": ElementType.XREF,
               "extract": lambda match, line: {
                   "href": match.group(1),
                   "content": match.group(1).split("|")[-1] if "|" in match.group(1) else match.group(1),
                   "link_type": "wikilink"
               }
           },
           r"\[([^\]]+)\]\(([^)]+)\)": {
               "type": ElementType.LINK,
               "extract": lambda match, line: {
                   "href": match.group(2),
                   "content": match.group(1),
                   "link_type": "markdown"
               }
           },
           r"<xref\s+href=[\"']([^\"']+)[\"']": {
               "type": ElementType.XREF,
               "extract": lambda match, line: {
                   "href": match.group(1),
                   "content": line,
                   "link_type": "dita"
               }
           },

           # Obsidian-style Callouts
           r"^>\s*\[!(\w+)\]": {  # Matches > [!type]
               "type": ElementType.NOTE,
               "extract": lambda match, line: {
                   "content": line.split("]", 1)[1].strip() if "]" in line else "",
                   "specialization": match.group(1).lower(),
                   "note_type": "callout"
               }
           },

            # Obsidian-style TODO list
           r"^(?:[-*+]\s)?\[([x ])\]": {
               "type": ElementType.TODO_LIST,
               "extract": lambda match, line: {
                   "content": line.split("]", 1)[1].strip(),
                   "checked": match.group(1).lower() == "x",
                   "list_type": "todo"
               }
           }
        }

    def _init_db(self) -> None:
        """Initialize database connection and schema."""
        try:
            if self._conn is None:
                self._conn = sqlite3.connect(str(self.db_path))
                self._conn.row_factory = sqlite3.Row

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


    ##########################################################################
    # Content parsing methods
    ##########################################################################


    def parse_ditamap(self, map_path: Path) -> TrackedElement:
        """Parse a DITA map with event tracking and caching."""
        # Initialize element before try block
        map_element = TrackedElement.create_map(
            path=map_path,
            title="",  # Temporary title
            id_handler=self.id_handler
        )

        try:
            self.logger.debug(f"Parsing DITA map: {map_path}")
            if not map_path.exists():
                raise FileNotFoundError(f"DITA map not found: {map_path}")

            # Check cache first
            cache_key = f"map_{map_path.stem}"
            if cached_map := self.content_cache.get(cache_key):
                self.logger.debug(f"Retrieved cached map: {map_path}")
                return cached_map

            # Update title after extraction
            map_element.title = self._extract_map_title(map_path)

            # Start discovery phase
            self.event_manager.start_phase(map_element.id, ProcessingPhase.DISCOVERY)
            self.event_manager.update_element_state(map_element, ProcessingState.PROCESSING)

            # Extract base metadata
            map_metadata = self.metadata_handler.extract_metadata(
                map_path,
                map_element.id
            )

            # Extract key definitions
            key_definitions = self._extract_key_definitions(map_element)
            map_metadata["key_definitions"] = key_definitions

            # Enrich metadata
            map_metadata = self._enrich_map_metadata(map_metadata)
            map_element.metadata = map_metadata

            # Cache the processed map
            self.content_cache.set(
                cache_key,
                map_element,
                ElementType.DITAMAP,
                ProcessingPhase.DISCOVERY
            )

            # Complete discovery phase
            self.event_manager.update_element_state(map_element, ProcessingState.COMPLETED)
            self.event_manager.end_phase(map_element.id, ProcessingPhase.DISCOVERY)

            return map_element

        except Exception as e:
            self.logger.error(f"Error parsing DITA map: {str(e)}")
            self.event_manager.update_element_state(map_element, ProcessingState.ERROR)
            raise

    def parse_topic(self, topic_path: Path, map_metadata: Dict[str, Any]) -> TrackedElement:
        """Parse a DITA topic with event tracking and caching."""
        # Initialize element before try block
        topic_element = TrackedElement.from_discovery(
            path=topic_path,
            element_type=ElementType.DITA,
            id_handler=self.id_handler
        )

        try:
            self.logger.debug(f"Parsing DITA topic: {topic_path}")
            if not topic_path.exists():
                raise FileNotFoundError(f"DITA topic not found: {topic_path}")

            # Check cache
            cache_key = f"topic_{topic_path.stem}"
            if cached_topic := self.content_cache.get(cache_key):
                self.logger.debug(f"Retrieved cached topic: {topic_path}")
                return cached_topic

            # Start discovery phase
            self.event_manager.start_phase(topic_element.id, ProcessingPhase.DISCOVERY)
            self.event_manager.update_element_state(topic_element, ProcessingState.PROCESSING)

            # Extract metadata
            topic_metadata = self.metadata_handler.extract_metadata(
                topic_path,
                topic_element.id,
                map_metadata=map_metadata
            )

            # Enrich metadata
            topic_metadata = self._enrich_topic_metadata(topic_metadata, map_metadata)
            topic_element.metadata = topic_metadata

            # Parse topic content and extract elements
            content = topic_path.read_text(encoding='utf-8')
            elements = self._extract_elements(content, topic_path)

            # Track extracted elements
            for element in elements:
                self.event_manager.track_element(element)
                topic_element.order.append(element.id)

                # Group elements by type
                element_type = str(element.type)
                if element_type not in topic_element.by_type:
                    topic_element.by_type[element_type] = []
                topic_element.by_type[element_type].append(element.id)

            # Cache processed topic
            self.content_cache.set(
                cache_key,
                topic_element,
                ElementType.DITA,
                ProcessingPhase.DISCOVERY
            )

            # Complete discovery phase
            self.event_manager.update_element_state(topic_element, ProcessingState.COMPLETED)
            self.event_manager.end_phase(topic_element.id, ProcessingPhase.DISCOVERY)

            return topic_element

        except Exception as e:
            self.logger.error(f"Error parsing DITA topic: {str(e)}")
            self.event_manager.update_element_state(topic_element, ProcessingState.ERROR)
            raise

    def parse_markdown_file(self, file_path: str) -> TrackedElement:
        """Parse Markdown content with event tracking and caching."""
        # Initialize element before try block
        file_path_obj = Path(file_path).resolve()
        element = TrackedElement.from_discovery(
            path=file_path_obj,
            element_type=ElementType.MARKDOWN,
            id_handler=self.id_handler
        )

        try:
            cache_key = f"md_{file_path_obj.stem}"

            # Check cache
            if cached_content := self.content_cache.get(cache_key):
                self.logger.debug(f"Retrieved cached markdown: {file_path}")
                return cached_content

            # Start discovery phase
            self.event_manager.start_phase(element.id, ProcessingPhase.DISCOVERY)
            self.event_manager.update_element_state(element, ProcessingState.PROCESSING)

            # Parse content
            with open(file_path_obj, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract YAML metadata
            metadata, content = self.metadata_handler.extract_yaml_metadata(content)
            element.content = content
            element.metadata.update(metadata)

            # Cache processed content
            self.content_cache.set(
                cache_key,
                element,
                ElementType.MARKDOWN,
                ProcessingPhase.DISCOVERY
            )

            # Complete discovery phase
            self.event_manager.update_element_state(element, ProcessingState.COMPLETED)
            self.event_manager.end_phase(element.id, ProcessingPhase.DISCOVERY)

            return element

        except Exception as e:
            self.logger.error(f"Error parsing Markdown file {file_path}: {e}")
            self.event_manager.update_element_state(element, ProcessingState.ERROR)
            raise



    def parse_xml_content(self, content: str) -> etree._Element:
        """
        Parse raw XML content into etree with caching.

        Args:
            content: XML content string

        Returns:
            Parsed etree Element
        """
        try:
            # Generate cache key from content hash
            cache_key = f"xml_{hash(content)}"

            # Check cache first
            if cached := self.content_cache.get(cache_key):
                return cached

            # Parse if not cached
            content_bytes = content.encode('utf-8') if isinstance(content, str) else content
            parser = etree.XMLParser(remove_blank_text=True)
            parsed = etree.fromstring(content_bytes, parser=parser)

            # Cache parsed content
            self.content_cache.set(
                cache_key,
                parsed,
                ElementType.DITA,
                ProcessingPhase.DISCOVERY,
                ttl=3600  # Cache for 1 hour
            )

            return parsed

        except Exception as e:
            self.logger.error(f"Error parsing XML content: {str(e)}")
            raise

    ##########################################################################
    # Metadata extraction methods
    ##########################################################################

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
        """Enrich map metadata with defaults."""
        metadata.setdefault("related_topics", [])
        metadata.setdefault("prerequisites", [])
        metadata.setdefault("feature_flags", {
            "enable_toc": True,
            "enable_xrefs": True
        })
        return metadata

    def _enrich_topic_metadata(
        self,
        metadata: Dict[str, Any],
        map_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enrich topic metadata with map context."""
        metadata.setdefault("related_topics", map_metadata.get("related_topics", []))
        metadata.setdefault("prerequisites", map_metadata.get("prerequisites", []))
        metadata.setdefault("feature_flags", map_metadata.get("feature_flags", {}))
        return metadata


    def _extract_map_title(self, map_path: Path) -> str:
        """Extract map title with caching."""
        try:
            cache_key = f"title_{map_path.stem}"

            if cached := self.content_cache.get(cache_key):
                return cached

            with map_path.open(encoding="utf-8") as file:
                tree = etree.parse(file)
                title_element = tree.find(".//title")
                title = title_element.text.strip() if title_element is not None and title_element.text else "Untitled Map"

            # Cache title
            self.content_cache.set(
                cache_key,
                title,
                ElementType.DITAMAP,
                ProcessingPhase.DISCOVERY,
                ttl=3600
            )

            return title

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
        """
        Detect LaTeX equations with caching.

        Args:
            content: Content to check

        Returns:
            True if LaTeX equations found
        """
        try:
            cache_key = f"latex_detect_{hash(content)}"

            if cached := self.content_cache.get(cache_key):
                return cached

            latex_patterns = [
                r'\$\$(.*?)\$\$',  # Display math
                r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)'  # Inline math
            ]

            result = any(re.search(pattern, content, re.DOTALL) for pattern in latex_patterns)

            # Cache result briefly
            self.content_cache.set(
                cache_key,
                result,
                ElementType.LATEX,
                ProcessingPhase.DISCOVERY,
                ttl=300  # Cache for 5 minutes
            )

            return result

        except Exception as e:
            self.logger.error(f"Error detecting LaTeX: {str(e)}")
            return False

    def _count_latex_blocks(self, content: str) -> dict:
        """
        Count LaTeX blocks with caching.

        Args:
            content: Content to analyze

        Returns:
            Dict with display and inline counts
        """
        try:
            cache_key = f"latex_count_{hash(content)}"

            if cached := self.content_cache.get(cache_key):
                return cached

            display_count = len(re.findall(r'\$\$(.*?)\$\$', content, re.DOTALL))
            inline_count = len(re.findall(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', content))

            result = {
                'display': display_count,
                'inline': inline_count
            }

            # Cache the counts
            self.content_cache.set(
                cache_key,
                result,
                ElementType.LATEX,
                ProcessingPhase.DISCOVERY,
                ttl=300  # Cache for 5 minutes
            )

            return result

        except Exception as e:
            self.logger.error(f"Error counting LaTeX blocks: {str(e)}")
            return {'display': 0, 'inline': 0}


    def _extract_topics(
        self,
        dita_map_content: str,
        map_path: Path
    ) -> List[TrackedElement]:
        """
        Extract topics from map with event tracking.

        Args:
            dita_map_content: Map content string
            map_path: Path to map file

        Returns:
            List of topic TrackedElements
        """
        try:
            root = ET.fromstring(dita_map_content)
            topics = []

            for topicref in root.findall(".//topicref"):
                if href := topicref.get("href"):
                    topic_id = topicref.get("id") or Path(href).stem

                    # Generate cache key for topic
                    cache_key = f"topic_{topic_id}"

                    # Check cache first
                    if cached_topic := self.content_cache.get(cache_key):
                        topics.append(cached_topic)
                        continue

                    # Process new topic
                    topic_path = (map_path.parent / href).resolve()

                    topic = TrackedElement.from_discovery(
                        path=topic_path,
                        element_type=self._determine_element_type(href),
                        id_handler=self.id_handler
                    )

                    # Start topic processing
                    self.event_manager.start_phase(topic.id, ProcessingPhase.DISCOVERY)
                    self.event_manager.update_element_state(topic, ProcessingState.PROCESSING)

                    # Set topic attributes
                    topic.topic_id = topic_id
                    topic.parent_map_id = map_path.stem
                    topic.href = href
                    topic.sequence_number = len(topics)

                    # Extract metadata
                    topic.metadata.update({
                        'type': topicref.get('type', 'topic'),
                        'scope': topicref.get('scope', 'local'),
                        'format': topicref.get('format', None),
                        'processing-role': topicref.get('processing-role', 'normal')
                    })

                    # Cache topic
                    self.content_cache.set(
                        cache_key,
                        topic,
                        ElementType.DITA,
                        ProcessingPhase.DISCOVERY
                    )

                    # Complete topic processing
                    self.event_manager.update_element_state(topic, ProcessingState.COMPLETED)
                    self.event_manager.end_phase(topic.id, ProcessingPhase.DISCOVERY)

                    topics.append(topic)

            return topics

        except ET.ParseError as e:
            self.logger.error(f"XML parsing error for {map_path}: {str(e)}")
            raise


    def _extract_elements(
        self,
        content: str,
        topic_path: Path
    ) -> List[TrackedElement]:
        """Extract elements with heading tracking."""
        try:
            elements = []
            current_section = None

            with self._get_db() as conn:
                for i, line in enumerate(content.splitlines(), start=1):
                    if not line.strip():
                        continue

                    # Match line against strategies
                    element = None
                    for pattern, strategy in self._element_extractors.items():
                        if match := re.match(pattern, line):
                            # Create element
                            element = TrackedElement.from_discovery(
                                path=topic_path,
                                element_type=strategy["type"],
                                id_handler=self.id_handler
                            )

                            # Track element processing
                            if not self.event_manager.track_element(element):
                                continue

                            # Extract data using strategy
                            extracted = strategy["extract"](match, line)
                            element.content = extracted.pop("content", line.strip())
                            element.metadata.update(extracted)

                            # Let HeadingHandler track the hierarchy
                            if strategy["type"] == ElementType.HEADING:
                                current_section = element.id
                                level = extracted.get("level", 1)
                                is_topic_title = (level == 1 and current_section is None)
                                self.heading_handler.track_heading(element, level, is_topic_title)

                            # Store element
                            self._store_element(
                                conn,
                                element,
                                current_section,
                                len(elements)
                            )

                            elements.append(element)
                            break

                    # Default to body element if no match
                    if not element:
                        element = self._create_body_element(
                            topic_path,
                            line.strip(),
                            current_section,
                            len(elements)
                        )
                        elements.append(element)

                return elements

        except Exception as e:
            self.logger.error(f"Error extracting elements: {str(e)}")
            raise

    def _store_element(
        self,
        conn: sqlite3.Connection,
        element: TrackedElement,
        parent_id: Optional[str],
        sequence_num: int
    ) -> None:
        """Store element with proper event tracking."""
        try:
            # Store in topic_elements
            conn.execute("""
                INSERT INTO topic_elements (
                    element_id, topic_id, element_type,
                    parent_element_id, sequence_num, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                element.id,
                element.topic_id,
                element.type.value,
                parent_id,
                sequence_num,
                hash(element.content)
            ))

            # Store context
            conn.execute("""
                INSERT INTO element_context (
                    element_id, context_type, parent_context,
                    level, xpath
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                element.id,
                element.type.value,
                parent_id,
                element.metadata.get("level", 0),
                f"/topic/{element.topic_id}/{element.type.value}[{sequence_num}]"
            ))

        except Exception as e:
            self.logger.error(f"Error storing element {element.id}: {str(e)}")
            raise

    def _create_body_element(
        self,
        topic_path: Path,
        content: str,
        parent_id: Optional[str],
        sequence_num: int
    ) -> TrackedElement:
        """Create body element with event tracking."""
        try:
            element = TrackedElement.from_discovery(
                path=topic_path,
                element_type=ElementType.BODY,
                id_handler=self.id_handler
            )

            # Track element
            if not self.event_manager.track_element(element):
                return element

            element.content = content

            # Store element
            with self._get_db() as conn:
                self._store_element(conn, element, parent_id, sequence_num)

            return element

        except Exception as e:
            self.logger.error(f"Error creating body element: {str(e)}")
            raise


    def _extract_key_definitions(self, map_element: TrackedElement) -> Dict[str, Any]:
        """
        Extract key definitions with event tracking.

        Args:
            map_element: The map TrackedElement

        Returns:
            Dictionary of key definitions
        """
        try:
            # Generate cache key
            cache_key = f"keydef_{map_element.id}"

            # Check cache
            if cached := self.content_cache.get(cache_key):
                return cached

            tree = etree.parse(str(map_element.path))
            keydefs = {}

            for keydef in tree.xpath(".//keydef"):
                key = keydef.get("keys")
                if not key:
                    continue

                # Track each keydef element
                keydef_element = TrackedElement.from_discovery(
                    path=map_element.path,
                    element_type=ElementType.UNKNOWN,
                    id_handler=self.id_handler
                )
                self.event_manager.track_element(keydef_element)

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

            # Cache key definitions
            self.content_cache.set(
                cache_key,
                keydefs,
                ElementType.DITAMAP,
                ProcessingPhase.DISCOVERY,
                ttl=3600
            )

            return keydefs

        except Exception as e:
            self.logger.error(f"Error extracting key definitions: {str(e)}")
            return {}


    def cleanup(self) -> None:
        """Clean up resources and reset state."""
        try:
            # Close database connection
            if self._conn is not None:
                self._conn.close()
                self._conn = None

            # Clear caches
            self.content_cache.clear()

            # Reset event tracking
            self.event_manager.clear_tracked_elements()

            self.logger.debug("Parser cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise
