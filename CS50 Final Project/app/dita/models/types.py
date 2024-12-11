# app/dita/utils/types.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any, Set, Union
from pathlib import Path


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

class ProcessingState(Enum):
    """Processing states."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class ProcessingFeatures:
    """Processing requirements"""
    needs_heading_numbers: bool = True
    needs_toc: bool = True
    needs_artifacts: bool = False
    needs_latex: bool = False
    latex_settings: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        'macros': {
            "\\N": "\\mathbb{N}",
            "\\R": "\\mathbb{R}"
        },
        'throw_on_error': False,
        'output_mode': 'html'
    })

class ContentScope(Enum):
    """Scope for conditional attributes."""
    GLOBAL = "global"
    MAP = "map"
    TOPIC = "topic"
    ELEMENT = "element"

class ContentItem:
    """Base class for DITA content items (topics and maps)."""
    id: str
    file_path: Path
    title: str
    version: str = "1.0"
    status: str = "draft"
    language: str = "en"
    content_hash: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Topic(ContentItem):
    """Topic content model."""
    type_id: int
    short_desc: Optional[str] = None
    parent_topic_id: Optional[str] = None
    root_map_id: Optional[str] = None
    specialization_type: Optional[str] = None
    content_type: str = "dita"  # 'dita' or 'markdown'

@dataclass
class Map(ContentItem):
    """Map content model."""
    toc_enabled: bool = True
    index_numbers_enabled: bool = True
    context_root: Optional[str] = None


@dataclass
class MDElementContext:
    """Context for markdown element processing"""
    parent_id: Optional[str]
    element_type: str
    classes: List[str]
    attributes: Dict[str, str]
    topic_path: Optional[Path] = None



@dataclass
class ElementContext:
    """Fine-grained context for element processing."""
    element_id: str
    topic_id: str
    element_type: str
    context_type: str  # body, abstract, prereq, etc.
    parent_context: Optional[str]
    level: int
    xpath: str
    conditions: Dict[str, Any] = field(default_factory=dict)
    processing_features: ProcessingFeatures = field(default_factory=ProcessingFeatures)



# Topic Type Requirements
@dataclass
class TopicTypeRequirement:
    """Requirements for DITA topic type validation."""
    id: int
    type_id: int
    element_name: str
    min_occurs: int = 1
    max_occurs: Optional[int] = None  # None means unbounded
    parent_element: Optional[str] = None
    description: Optional[str] = None

@dataclass
class TopicType:
    """Represents a DITA topic type with its requirements."""
    id: int
    name: str
    base_type: Optional[str]
    description: str
    schema_file: Optional[str]
    is_custom: bool = False
    requirements: List[TopicTypeRequirement] = field(default_factory=list)


# Relationship Types
@dataclass
class TopicRelationship:
    """Represents relationships between topics."""
    relationship_id: int
    source_topic_id: str
    target_topic_id: str
    relationship_type: str
    weight: int = 0
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class TopicElement:
    """Represents an element within a topic."""
    element_id: str
    topic_id: str
    element_type: str
    parent_element_id: Optional[str]
    sequence_num: int
    content_hash: Optional[str]
    context: Optional[ElementContext] = None
    created_at: datetime = field(default_factory=datetime.now)




@dataclass
class HeadingContext:
    """Context for heading processing"""
    parent_id: Optional[str] = None
    level: int = 1
    is_topic_title: bool = False


@dataclass
class BaseContext:
    """Base context information."""
    id: str
    content_type: str  # 'topic' or 'map'
    processing_phase: ProcessingPhase
    processing_state: ProcessingState
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TopicContext:
    """Context for topic processing."""
    base: BaseContext
    map_id: str
    parent_id: Optional[str]
    level: int
    sequence_num: int
    heading_number: Optional[str]
    context_path: str
    topic_type: TopicType
    processing_features: ProcessingFeatures = field(default_factory=ProcessingFeatures)

@dataclass
class MapContext:
    """Context for map processing."""
    base: BaseContext  # base contains metadata
    topic_order: List[str]
    root_context: str
    conditions: Dict[str, Any] = field(default_factory=dict)
    features: ProcessingFeatures = field(default_factory=ProcessingFeatures)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingContext:
    """Complete processing context combining map and topic contexts."""
    map_context: MapContext
    topics: Dict[str, TopicContext]  # topic_id -> TopicContext
    current_topic_id: Optional[str] = None

    def get_current_topic_context(self) -> Optional[TopicContext]:
        if self.current_topic_id:
            return self.topics.get(self.current_topic_id)
        return None

    def set_current_topic(self, topic_id: str) -> None:
        if topic_id in self.topics:
            self.current_topic_id = topic_id





# Conditional Processing Types
@dataclass
class ConditionalAttribute:
    """Represents a conditional processing attribute."""
    attribute_id: int
    name: str
    description: Optional[str]
    is_toggle: bool
    context_dependent: bool
    scope: str  # 'global', 'map', 'topic', or 'element'

@dataclass
class ConditionalValue:
    """Represents a value for a conditional attribute."""
    value_id: int
    attribute_id: int
    value: str
    description: Optional[str]

@dataclass
class ContentCondition:
    """Represents a condition applied to content."""
    condition_id: int
    content_id: str
    attribute_id: int
    value_id: int
    content_type: str  # 'topic', 'map', or 'element'

# Content Processing Types
@dataclass
class ProcessingMetadata:
    """Metadata for content processing."""
    processing_id: int
    content_id: str
    content_type: str
    process_latex: bool = False
    enable_cross_refs: bool = True
    show_toc: bool = True
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)



@dataclass
class ParsedElement:
    """Element returned by parser"""
    id: str
    topic_id: str
    type: ElementType
    content: str  # Raw Markdown content
    topic_path: Path
    source_path: Path
    metadata: Dict[str, Any]  # Raw YAML/XML metadata

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

# Feature types


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
class DITAProcessingConfig:
    """Configuration for DITA processing."""
    enable_debug: bool = False
    show_toc: bool = True
    enable_cross_refs: bool = True
    process_latex: bool = False

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
    features: ProcessingFeatures = field(default_factory=ProcessingFeatures)

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
    processing: ProcessingOptions
    features: ProcessingFeatures
    latex_config: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        'enabled': False,
        'settings': {
            'macros': {},
            'throw_on_error': False,
            'output_mode': 'html'
        }
    })

## Element tracking
@dataclass
class ElementAttributes:
    """Base attributes for any element"""
    id: str
    classes: List[str]
    custom_attrs: Dict[str, str]

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Allows dictionary-like access to custom attributes.

        Args:
            key: The key to retrieve.
            default: The default value if the key does not exist.

        Returns:
            The value of the attribute or the default value.
        """
        return self.custom_attrs.get(key, default)



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
class DITAElementContext:
    """Context for DITA element processing"""
    parent_id: Optional[str]
    element_type: str
    classes: List[str]
    attributes: Dict[str, str]
    topic_type: Optional[str] = None  # For tracking if inside concept/task/reference
    is_body: bool = False  # For tracking if inside conbody/taskbody/refbody
    topic_path: Optional[Path] = None

    def replace(self, **changes) -> 'DITAElementContext':
        """Return a new instance with specified fields replaced."""
        return DITAElementContext(**{**self.__dict__, **changes})

@dataclass
class DITAElementInfo:
    """Information about a DITA element"""
    type: DITAElementType
    content: str
    attributes: ElementAttributes
    context: DITAElementContext
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
    attributes: ElementAttributes
    context: MDElementContext
    metadata: Dict[str, Any]
    level: Optional[int] = None  # Added level attribute

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
