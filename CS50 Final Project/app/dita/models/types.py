# app/dita/utils/types.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, TypedDict, Optional, Any, Set, Union, TYPE_CHECKING
from pathlib import Path
from uuid import uuid4
if TYPE_CHECKING:
    from app.dita.utils.id_handler import DITAIDHandler


# Type aliases
MetadataDict = Dict[str, Any]
PathLike = Union[str, Path]
HTMLString = str
IDType = str

class ElementType(Enum):
    """Types of elements that can be parsed"""
    DITA = "dita"
    DITAMAP = "ditamap"
    MARKDOWN = "markdown"
    LATEX = "latex"
    ARTIFACT = "artifact"
    TOPIC = "topic"
    MAP_TITLE = "map_title"
    HEADING = "heading"
    TITLE = "title"
    BODY = "body"
    UNORDERED_LIST = "ul"
    ORDERED_LIST = "ol"
    LIST_ITEM = "li"
    CODE_BLOCK = "codeblock"
    CODE_PHRASE = "codeph"
    FIGURE = "fig"
    IMAGE = "image"
    XREF = "xref"
    LINK = "link"
    SHORTDESC = "shortdesc"
    ABSTRACT = "abstract"
    PREREQ = "prereq"
    STEPS = "steps"
    DEFINITION = "dlentry"
    TERM = "term"
    BOLD = "b"
    ITALIC = "i"
    UNDERLINE = "u"
    PHRASE = "ph"
    QUOTE = "q"
    PRE = "pre"
    CITE = "cite"
    METADATA = "metadata"
    TOPICREF = "topicref"
    UNKNOWN = "unknown"

# Enums for validation
class ProcessingPhase(Enum):
    """Phases of the processing pipeline"""
    DISCOVERY = "discovery"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    ENRICHMENT = "enrichment" # LaTeX, artifacts
    ASSEMBLY = "assembly"
    ERROR = "error"

class ProcessingState(Enum):
    """Processing states."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"



@dataclass
class TrackedElement:
    """Element tracking throughout the entire processing pipeline."""
    # Core identification
    id: str
    type: ElementType
    path: Path
    source_path: Path
    content: str

    # Map/Topic specific
    title: Optional[str] = None
    topic_id: Optional[str] = None  # Context tracking
    parent_map_id: Optional[str] = None  # Context tracking
    href: Optional[str] = None  # For topic references in maps
    topics: List[str] = field(default_factory=list)  # For maps only
    sequence_number: Optional[int] = None  # For ordering

    # Hierarchy tracking
    order: List[str] = field(default_factory=list)
    hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    by_type: Dict[str, List[str]] = field(default_factory=dict)

    # Error tracking
    last_error: Optional[str] = None
    # last_updated: Optional[datetime] = None  # Deprecated: Use metadata or database for persistent tracking
    created_at: datetime = field(default_factory=datetime.now)

    # Processing state
    phase: ProcessingPhase = ProcessingPhase.DISCOVERY
    state: ProcessingState = ProcessingState.PENDING
    previous_state: Optional[ProcessingState] = None
    processing_attempts: int = 0

    # Timestamps & Error tracking
    # created_at: datetime = field(default_factory=datetime.now)  # Already defined above
    # last_updated: Optional[datetime] = None  # Deprecated (handled dynamically elsewhere)
    # last_error: Optional[str] = None  # Redundant; already declared above

    # Metadata & Rendering
    metadata: Dict[str, Any] = field(default_factory=dict)  # Persistent metadata
    html_metadata: Dict[str, Any] = field(default_factory=lambda: {
        "attributes": {},
        "classes": [],
        "context": {
            "parent_id": None,
            "level": None,
            "position": None
        },
        "features": {}
    })

    # New Fields
    context: Dict[str, Any] = field(default_factory=dict)  # Processing context (e.g., TOC, cross-references)
    feature_flags: Dict[str, bool] = field(default_factory=dict)  # Feature toggles like heading numbering, TOC
    custom_metadata: Dict[str, Any] = field(default_factory=dict)  # Future extensibility for user-defined metadata

    # Methods
    def parse_content(self) -> None:
        """Load and parse content (replaces ParsedElement functionality)"""
        if self.phase != ProcessingPhase.DISCOVERY:
            raise ValueError("Can only parse in DISCOVERY phase")

        self.state = ProcessingState.PROCESSING
        self.content = self.path.read_text()
        self.metadata = self._extract_metadata()

        self.state = ProcessingState.COMPLETED
        self.phase = ProcessingPhase.VALIDATION

    def _extract_metadata(self) -> Dict[str, Any]:
        """Extract metadata based on content type"""
        # Default empty metadata
        return {}

    def get_attribute(self, key: str) -> Optional[str]:
        return self.html_metadata["attributes"].get(key)

    def add_class(self, class_name: str) -> None:
        if class_name not in self.html_metadata["classes"]:
            self.html_metadata["classes"].append(class_name)

    def advance_phase(self, new_phase: ProcessingPhase) -> None:
        """Advance to next processing phase."""
        self.phase = new_phase
        self.state = ProcessingState.PENDING
        self.metadata["last_updated"] = datetime.now()  # Use metadata for transient timestamps

    def update_state(self, new_state: ProcessingState) -> None:
        """Update processing state."""
        self.previous_state = self.state
        self.state = new_state
        self.metadata["last_updated"] = datetime.now()  # Use metadata for transient timestamps

    def increment_attempts(self) -> None:
        """Increment processing attempts counter."""
        self.processing_attempts += 1
        self.metadata["last_updated"] = datetime.now()  # Use metadata for transient timestamps

    def set_error(self, error: str) -> None:
        """Record processing error."""
        self.last_error = error
        self.metadata["last_updated"] = datetime.now()  # Use metadata for transient timestamps

    def can_process(self) -> bool:
        """Check if element can be processed in current phase."""
        return (
            self.state != ProcessingState.ERROR and
            self.processing_attempts < 3 and
            self.state != ProcessingState.COMPLETED
        )

    @classmethod
    def create_map(cls, path: Path, title: str, id_handler: DITAIDHandler) -> "TrackedElement":
        """
        Create a TrackedElement for a DITA map.

        Args:
            path: Path to the DITA map file.
            title: Title of the map.
            id_handler: The DITAIDHandler instance for ID generation.

        Returns:
            A `TrackedElement` representing the map.
        """
        try:
            map_id = id_handler.generate_id(
                base=path.stem,
                element_type="map"  # Prepend "map" for consistent ID generation
            )
            return cls(
                id=map_id,
                type=ElementType.DITAMAP,
                path=path,
                source_path=path,
                content="",  # Content will be parsed later
                title=title
            )
        except Exception as e:
            raise ValueError(f"Error creating map TrackedElement: {str(e)}")


    @classmethod
    def from_discovery(cls, path: Path, element_type: ElementType, id_handler: DITAIDHandler) -> "TrackedElement":
        """Create a TrackedElement during the discovery phase."""
        return cls(
            id=id_handler.generate_id(path.stem),
            type=element_type,
            path=path,
            source_path=path,
            content=""  # Content will be loaded later
        )


@dataclass
class ProcessingMetadata:
    """Core metadata needed for rendering and transformation"""
    # Core identifiers and types
    id: str
    content_type: ElementType
    base_type: Optional[str] = None
    parent_id: Optional[str] = None
    language: str = "en"

    # Use TrackedElement for element tracking
    elements: Dict[str, TrackedElement] = field(default_factory=dict)
    current_element_id: Optional[str] = None

    def add_element(self, element: TrackedElement) -> None:
        """Add tracked element"""
        self.elements[element.id] = element

    def get_current_element(self) -> Optional[TrackedElement]:
        """Get current element being processed"""
        if self.current_element_id:
            return self.elements.get(self.current_element_id)
        return None

    @property
    def element_order(self) -> List[str]:
        """Get ordered list of element IDs"""
        return [
            element.id for element in sorted(
                self.elements.values(),
                key=lambda e: e.sequence_number or 0
            )
        ]

    def get_elements_by_type(self, element_type: ElementType) -> List[TrackedElement]:
        """Get elements of specific type"""
        return [
            element for element in self.elements.values()
            if element.type == element_type
        ]

    def get_child_elements(self, parent_id: str) -> List[TrackedElement]:
        """Get child elements of a parent"""
        return [
            element for element in self.elements.values()
            if element.parent_map_id == parent_id
        ]

    # Processing flags
    features: Dict[str, bool] = field(default_factory=lambda: {
        "index_numbers": True,
        "anchor_links": True,
        "toc": True,
        "latex": False,
        "artifacts": False,
        "xrefs": True
    })

    # Context for rendering
    map_metadata: Dict[str, Any] = field(default_factory=lambda: {
        "title": "",
        "topic_order": [],
        "current_topic": None
    })

    # Rendering attributes
    attributes: Dict[str, Any] = field(default_factory=lambda: {
        "classes": [],
        "id": "",
        "custom_attrs": {}
    })

    # Content flags
    display_flags: Dict[str, bool] = field(default_factory=lambda: {
        "visible": True,
        "enabled": True,
        "expanded": True
    })

    # Cache references during processing
    references: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "headings": {},  # id -> {text, level}
        "xrefs": {}     # id -> {href, text}
    })



    def add_heading(self, heading_id: str, text: str, level: int) -> None:
        """Cache heading for link resolution"""
        self.references["headings"][heading_id] = {
            "text": text,
            "level": level
        }

    def resolve_reference(self, ref_id: str) -> Optional[str]:
        """Get href for reference"""
        return self.references["xrefs"].get(ref_id, {}).get("href")


    def should_render(self) -> bool:
        """Single decision point for rendering"""
        return self.display_flags.get("visible", True)

    @property
    def has_latex(self) -> bool:
        return self.features.get("latex", False)

    @property
    def should_number_headings(self) -> bool:
        return self.features.get("number_headings", True)


@dataclass
class ProcessingContext:
    """Core processing context with minimal metadata."""
    # Map-level metadata
    map_id: str
    features: Dict[str, bool] = field(default_factory=lambda: {
        "index_numbers": True,
        "anchor_links": True,
        "toc": True,
        "latex": False,
        "artifacts": False,
        "enable_cross_refs": True,
    })

    # Current processing state
    current_topic_id: Optional[str] = None
    topic_order: List[str] = field(default_factory=list)

    # Metadata
    map_metadata: Dict[str, Any] = field(default_factory=dict)
    topic_metadata: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "headings": {},  # Default headings metadata
    })

    def register_element(self, element: TrackedElement, id_handler: DITAIDHandler) -> None:
        """Register element with proper ID handling"""
        if element.type == ElementType.DITAMAP:
            self.map_id = element.id
        elif self.current_topic_id:
            # For elements within topics, use topic prefix
            element.id = id_handler.generate_id(
                element.content[:20],
                prefix=self.current_topic_id
            )

    def get_topic_metadata(self) -> Optional[Dict[str, Any]]:
        if self.current_topic_id:
            return self.topic_metadata.get(self.current_topic_id)
        return None

    def set_topic(self, topic_id: str, metadata: Dict[str, Any]) -> None:
        self.current_topic_id = topic_id
        self.topic_metadata[topic_id] = metadata



@dataclass
class ProcessedContent:
    """Processed element content"""
    html: str
    element_id: str
    heading_level: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Topic:
    """Topic for processing pipeline."""
    id: str
    path: Path
    content_type: ElementType  # DITA or MARKDOWN
    parent_map_id: Optional[str] = None
    base_type: Optional[str] = None   # Added from TopicType
    processing_metadata: ProcessingMetadata = field(default_factory=lambda: ProcessingMetadata(
        id="",
        content_type=ElementType.UNKNOWN
    ))

    # Optional rendering metadata
    title: Optional[str] = None
    short_desc: Optional[str] = None

    def __post_init__(self):
        self.processing_metadata.id = self.id
        self.processing_metadata.content_type = self.content_type

    @property
    def template_path(self) -> Optional[Path]:
        """Get template path based on content type and base type"""
        if self.base_type:
            return Path(f"templates/{self.base_type}/{self.content_type.value}.html")
        return Path(f"templates/{self.content_type.value}.html")

    @property
    def is_specialized(self) -> bool:
        """Check if topic is specialized from base type"""
        return self.base_type is not None

@dataclass
class Map:
    """Single consolidated map class"""
    id: str
    path: Path
    title: str
    topics: List[str]  # Just topic IDs
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None




# Heading tracking types
@dataclass
class HeadingState:
    """State for heading tracking."""
    current_h1: int = 0
    counters: Dict[str, int] = field(default_factory=lambda: {f"h{i}": 0 for i in range(1, 7)})
    used_ids: Set[str] = field(default_factory=set)

    def increment(self, level: int) -> None:
        """Increment counter for a given heading level."""
        if level == 1:
            self.current_h1 += 1
            self.counters['h1'] = self.current_h1
        else:
            self.counters[f'h{level}'] += 1
        # Reset lower-level counters
        for l in range(level + 1, 7):
            self.counters[f'h{l}'] = 0

    def current_heading_number(self) -> str:
        """Return current heading number as a string."""
        return '.'.join(str(self.counters[f'h{l}']) for l in range(1, 7) if self.counters[f'h{l}'] > 0)



# Artifact types
@dataclass
class ArtifactReference:
    """Artifact reference information"""
    id: str
    href: str
    target: str
    type: str
    metadata: Dict[str, Any]

@dataclass
class ProcessedArtifact:
    """Processed artifact content"""
    id: str
    html: str
    target: str
    metadata: Dict[str, Any]

# LaTeX processing types
@dataclass
class LaTeXEquation:
    """Represents a LaTeX equation."""
    id: str
    content: str
    is_block: bool
    placeholder: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessedEquation:
    id: str
    original: str
    rendered: str
    placeholder: str
    is_block: bool


# Error types
@dataclass
class ProcessingError(Exception):
    """Custom error for processing failures"""
    def __init__(
        self,
        error_type: str,
        message: str,
        context: Union[str, Path],
        element_id: Optional[str] = None,
        stacktrace: Optional[str] = None
    ):
        self.error_type = error_type
        self.message = message
        self.context = context
        self.element_id = element_id
        self.stacktrace = stacktrace
        super().__init__(self.message)

@dataclass
class LogContext:
    phase: ProcessingPhase
    element_id: Optional[str] = None
    topic_id: Optional[str] = None
    map_id: Optional[str] = None

# Processing pipeline types

@dataclass
class DITAProcessingConfig:
    """Configuration for DITA processing."""
    enable_debug: bool = False
    show_toc: bool = True
    enable_cross_refs: bool = True
    process_latex: bool = False




@dataclass
class DITAParserConfig:
    """Configuration for DITA parser."""
    enable_strict_mode: bool = False
    parse_external_references: bool = True
    support_legacy_dita: bool = False
    allowed_extensions: list[str] = field(default_factory=lambda: ['.dita', '.ditamap', '.md'])


# Configuration
@dataclass
class ParserConfig:
    """Parser configuration"""
    validate_dtd: bool = False
    resolve_entities: bool = False
    load_dtd: bool = False
    remove_blank_text: bool = True

@dataclass
class PathConfig:
    """Path configuration"""
    dita_root: Path
    maps_dir: Path
    topics_dir: Path
    output_dir: Path
    artifacts_dir: Path
    media_dir: Path

@dataclass
class ProcessorConfig:
    """Main processor configuration"""
    parser: ParserConfig
    paths: PathConfig
    latex_config: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        'enabled': False,
        'settings': {
            'macros': {},
            'throw_on_error': False,
            'output_mode': 'html'
        }
    })




class DITAElementType(Enum):
    """Types of DITA elements"""
    CONCEPT = "concept"
    TASK = "task"
    REFERENCE = "reference"
    TOPIC = "topic"
    MAP = "map"
    SECTION = "section"
    PARAGRAPH = "p"
    NOTE = "note"
    TABLE = "table"
    LIST = "ul"
    LIST_ITEM = "li"
    ORDERED_LIST = "ol"
    CODE_BLOCK = "codeblock"
    BLOCKQUOTE = "blockquote"
    CODE_PHRASE = "codeph"
    FIGURE = "fig"
    IMAGE = "image"
    XREF = "xref"
    LINK = "link"
    TITLE = "title"
    SHORTDESC = "shortdesc"
    ABSTRACT = "abstract"
    PREREQ = "prereq"
    STEPS = "steps"
    STEP = "step"
    SUBSTEP = "substep"
    SUBSTEPS = "substeps"
    DEFINITION = "dlentry"
    TERM = "term"
    BOLD = "b"
    ITALIC = "i"
    UNDERLINE = "u"
    PHRASE = "ph"
    QUOTE = "q"
    PRE = "pre"
    CITE = "cite"
    METADATA = "metadata"
    TOPICREF = "topicref"
    TOPICGROUP = "topicgroup"
    UL = "ul"
    LI = "li"
    CMD = "cmd"
    INFO = "info"
    TASKBODY = "taskbody"
    WARNING = "warning"
    UNKNOWN = "unknown"


@dataclass
class DITAElementInfo:
    """Information about a DITA element"""
    type: DITAElementType
    content: str
    children: List
    metadata: Dict[str, Any]

class MDElementType(Enum):
    """Types of Markdown elements"""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    CALLOUT = "callout"
    LIST = "list"
    LIST_ITEM = "list_item"
    LINK = "link"
    UNORDERED_LIST = "unordered_list"
    ORDERED_LIST = "ordered_list"
    IMAGE = "image"
    CODE_BLOCK = "code_block"
    CODE_PHRASE = "code_phrase"
    BLOCKQUOTE = "blockquote"
    TABLE = "table"
    TABLE_HEADER = "table_header"
    TABLE_ROW = "table_row"
    TABLE_CELL = "table_cell"
    BACKLINK = "backlink"
    FOOTNOTE = "footnote"
    ITALIC = "italic"
    BOLD = "bold"
    UNDERLINE = "underline"
    STRIKETHROUGH = "strikethrough"
    TODO = "todo"
    YAML_METADATA = "yaml_metadata"
    UNKNOWN = "unknown"


@dataclass
class MDElementInfo:
    """Information about a Markdown element"""
    type: MDElementType
    content: str
    metadata: Dict[str, Any]
    level: Optional[int] = None
    specialization: Optional[Dict[str, Any]] = None
