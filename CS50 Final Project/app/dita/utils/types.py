# app/dita/utils/types.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Set
from pathlib import Path
from enum import Enum

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
    MAP_TITLE = "map_title"
    UNKNOWN = "unknown"

class ProcessingPhase(Enum):
    """Phases of the processing pipeline"""
    DISCOVERY = "discovery"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    ENRICHMENT = "enrichment" # LaTeX, artifacts
    ASSEMBLY = "assembly"

class ProcessingState(Enum):
    """States for processing pipeline"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class ParsedElement:
    """Element returned by parser"""
    id: str
    topic_id: str
    type: ElementType
    content: str
    topic_path: Path
    source_path: Path
    metadata: Dict[str, Any]

@dataclass
class ParsedMap:
    """Container for parsed map information"""
    title: Optional[str]
    topics: List[ParsedElement]
    metadata: Dict[str, Any]
    source_path: Path


@dataclass
class ProcessedContent:
    """Processed element content"""
    html: str
    element_id: str
    heading_level: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DiscoveredTopic:
    """Represents a discovered topic file"""
    id: str
    path: Path
    type: ElementType
    href: str  # Original href from map
    metadata: Dict[str, Any]

@dataclass
class DiscoveredMap:
    """Represents discovered map content"""
    id: str
    path: Path
    title: Optional[str]
    topics: List[DiscoveredTopic]
    metadata: Dict[str, Any]

# Context tracking types
@dataclass
class MapContext:
    """Tracking context for map processing"""
    map_id: str
    map_path: Path
    metadata: Dict[str, Any]
    topic_order: List[str]  # List of topic IDs in order
    features: Dict[str, bool]  # Map-level features/conditions
    type: ElementType

@dataclass
class TopicContext:
    """Tracking context for topic processing"""
    topic_id: str
    topic_path: Path
    type: ElementType
    parent_map_id: str
    metadata: Dict[str, Any]
    features: Dict[str, bool]  # Topic-level features/conditions
    state: ProcessingState = ProcessingState.PENDING

@dataclass
class ProcessingContext:
    """Complete processing context"""
    map_context: MapContext
    topics: Dict[str, TopicContext]  # topic_id -> TopicContext
    current_topic_id: Optional[str] = None

# MD types
@dataclass
class MDElementContext:
    """Context for markdown element processing"""
    parent_id: Optional[str]
    element_type: str
    classes: List[str]
    attributes: Dict[str, str]
    topic_path: Optional[Path] = None

# Feature types

@dataclass
@dataclass
class ContentFeatures:
    """Detected content features"""
    has_latex: bool = False
    has_code: bool = False
    has_tables: bool = False
    has_images: bool = False
    has_xrefs: bool = False
    has_artifacts: bool = False
    # Future features can be added here

@dataclass
class ProcessingFeatures:
    """Processing requirements"""
    needs_heading_numbers: bool = True
    needs_toc: bool = True
    needs_artifacts: bool = False
    needs_latex: bool = False
    # Future processing features can be added here

# Heading tracking types
@dataclass
class HeadingState:
    """State for heading tracking."""
    current_h1: int = 0
    counters: Dict[str, int] = field(default_factory=lambda: {
        'h1': 0, 'h2': 0, 'h3': 0, 'h4': 0, 'h5': 0, 'h6': 0
    })
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

@dataclass
class HeadingContext:
    """Context for heading processing"""
    parent_id: Optional[str] = None
    level: int = 1
    is_topic_title: bool = False


# Reference tracking types
@dataclass
class HeadingReference:
    """Heading reference information"""
    id: str
    text: str
    level: int
    topic_id: Optional[str] = None
    map_id: Optional[str] = None

@dataclass
class CrossReference:
    """Cross-reference information"""
    source_id: str
    target_id: str
    text: Optional[str] = None
    type: str = "internal"  # "internal", "external", "web"

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
    """LaTeX equation information"""
    id: str
    content: str
    is_block: bool
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ProcessedEquation:
    """Processed LaTeX equation"""
    id: str
    html: str
    original: str
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
class ProcessingResult:
    """Result of processing operation"""
    content: Optional[ProcessedContent] = None
    error: Optional[ProcessingError] = None
    metadata: Optional[Dict[str, Any]] = None
    state: ProcessingState = ProcessingState.COMPLETED

@dataclass
class ProcessingOptions:
    """Options for processing pipeline"""
    process_latex: bool = True
    number_headings: bool = True
    enable_cross_refs: bool = True
    process_artifacts: bool = True
    show_toc: bool = True
    features: Optional[Dict[str, bool]] = None


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
    processing: ProcessingOptions
    features: ProcessingFeatures


## Element tracking
@dataclass
class ElementAttributes:
    """Base attributes for any element"""
    id: str
    classes: List[str]
    custom_attrs: Dict[str, str]


class DITAElementType(Enum):
    """Types of DITA elements"""
    CONCEPT = "concept"
    TASK = "task"
    REFERENCE = "reference"
    TOPIC = "topic"
    SECTION = "section"
    PARAGRAPH = "p"
    NOTE = "note"
    TABLE = "table"
    LIST = "ul"
    LIST_ITEM = "li"
    ORDERED_LIST = "ol"
    CODE_BLOCK = "codeblock"
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
    DEFINITION = "dlentry"
    TERM = "term"
    BOLD = "b"
    ITALIC = "i"
    UNDERLINE = "u"
    PHRASE = "ph"
    QUOTE = "q"
    PRE = "pre"
    CITE = "cite"

@dataclass
class DITAElementContext:
    """Context for DITA element processing"""
    parent_id: Optional[str]
    element_type: str
    classes: List[str]
    attributes: Dict[str, str]
    topic_type: Optional[str] = None  # For tracking if inside concept/task/reference
    is_body: bool = False  # For tracking if inside conbody/taskbody/refbody
    topic_path: Optional[Path] = None

@dataclass
class DITAElementInfo:
    """Information about a DITA element"""
    type: DITAElementType
    content: str
    attributes: ElementAttributes
    context: DITAElementContext
    metadata: Dict[str, Any]

class MDElementType(Enum):
    """Types of Markdown elements"""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    LIST_ITEM = "list_item"
    CODE = "code"
    BLOCKQUOTE = "blockquote"
    LINK = "link"
    IMAGE = "image"
    TABLE = "table"
    EMPHASIS = "emphasis"
    STRONG = "strong"
    INLINE_CODE = "inline_code"

@dataclass
class MDElementInfo:
    """Information about a Markdown element"""
    type: MDElementType
    content: str
    attributes: ElementAttributes
    context: MDElementContext
    metadata: Dict[str, Any]

@dataclass
class TrackedElement:
    """Element with comprehensive tracking information."""
    # Core identification
    id: str

    # Content info
    type: ElementType
    path: Path
    content: str  # Actual content or title for map titles

    # Processing metadata
    metadata: Dict[str, Any]

    # Processing state tracking
    state: ProcessingState = ProcessingState.PENDING
    previous_state: Optional[ProcessingState] = None
    processing_attempts: int = 0

    # Context tracking
    parent_map_id: Optional[str] = None
    sequence_number: Optional[int] = None  # For ordering

    # Timestamp tracking
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: Optional[datetime] = None

    # Error tracking
    last_error: Optional[str] = None

    def update_state(self, new_state: ProcessingState) -> None:
        """Update element processing state with timestamp."""
        self.previous_state = self.state
        self.state = new_state
        self.last_updated = datetime.now()

    def increment_attempts(self) -> None:
        """Increment processing attempts counter."""
        self.processing_attempts += 1
        self.last_updated = datetime.now()

    def set_error(self, error: str) -> None:
        """Record processing error."""
        self.last_error = error
        self.last_updated = datetime.now()

    def can_process(self) -> bool:
        """Check if element can be processed."""
        return (
            self.state != ProcessingState.ERROR and
            self.processing_attempts < 3 and  # Max attempts
            self.state != ProcessingState.COMPLETED
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'type': self.type.value,
            'path': str(self.path),
            'state': self.state.value,
            'attempts': self.processing_attempts,
            'created': self.created_at.isoformat(),
            'updated': self.last_updated.isoformat() if self.last_updated else None,
            'error': self.last_error,
            'metadata': self.metadata
        }
