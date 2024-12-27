# app/dita/utils/types.py

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, TypedDict, Optional, Any, Set, Union, TYPE_CHECKING
from pathlib import Path
from uuid import uuid4
if TYPE_CHECKING:
    from ..utils.id_handler import DITAIDHandler



# Type aliases
MetadataDict = Dict[str, Any]
PathLike = Union[str, Path]
HTMLString = str

class ElementType(Enum):
    """Types of elements that can be parsed"""
    # Core types
    DITA = "dita"
    DITAMAP = "ditamap"
    MAP = "map"
    TOPIC = "topic"
    MARKDOWN = "markdown"

    # Structural elements
    DEFAULT = "default"          # Default fallback type
    HEADING = "heading"
    TITLE = "title"
    MAP_TITLE = "map_title"
    BODY = "body"
    SECTION = "section"
    SPECIALIZATIONS = "specializations"
    BLOCK = "block"

    # Block elements
    PARAGRAPH = "paragraph"
    CODE_BLOCK = "code_block"
    BLOCKQUOTE = "blockquote"
    NOTE = "note"

    # List elements (aligned with JSON)
    UNORDERED_LIST = "unordered"
    ORDERED_LIST = "ordered"
    LIST_ITEM = "item"

    # Inline elements (aligned with JSON)
    CODE_PHRASE = "codeph"
    BOLD = "bold"
    ITALIC = "italic"
    UNDERLINE = "underline"
    PHRASE = "phrase"
    HIGHLIGHT = "highlight"
    STRIKETHROUGH = "strikethrough"
    QUOTE = "quote"
    CITE = "cite"

    # Media elements
    FIGURE = "figure"
    IMAGE = "image"

    # Link elements
    XREF = "xref"
    LINK = "link"

    # Table elements
    TABLE = "table"
    TABLE_HEADER = "table_header"
    TABLE_ROW = "table_row"
    TABLE_CELL = "table_cell"

    # Special elements
    INNER = "inner"
    SHORTDESC = "shortdesc"
    ABSTRACT = "abstract"
    PREREQ = "prereq"
    STEPS = "steps"
    STEP = "step"
    SUBSTEP = "substep"
    SUBSTEPS = "substeps"
    DEFINITION = "definition"
    TERM = "term"
    METADATA = "metadata"
    TOPICREF = "topicref"
    CONCEPT = "concept"
    TASK = "task"
    REFERENCE = "reference"
    BASE = "base"
    CMD = "cmd"
    INFO = "info"
    DLENTRY = "dlentry"
    TOPICGROUP = "topicgroup"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, element_type: str) -> 'ElementType':
        """Convert string to ElementType with fallback."""
        try:
            return cls(element_type)
        except ValueError:
            return cls.UNKNOWN

    @classmethod
    def is_block_element(cls, element_type: 'ElementType') -> bool:
        """Check if element type is a block element."""
        block_elements = {
            cls.PARAGRAPH, cls.CODE_BLOCK, cls.BLOCKQUOTE,
            cls.NOTE, cls.SECTION, cls.FIGURE
        }
        return element_type in block_elements

    @classmethod
    def is_inline_element(cls, element_type: 'ElementType') -> bool:
        """Check if element type is an inline element."""
        inline_elements = {
            cls.CODE_PHRASE, cls.BOLD, cls.ITALIC, cls.UNDERLINE,
            cls.PHRASE, cls.HIGHLIGHT, cls.STRIKETHROUGH, cls.QUOTE
        }
        return element_type in inline_elements

class IDType(Enum):
    """Types of IDs that require different patterns."""
    MAP = "map"
    TOPIC = "topic"
    HEADING = "heading"
    ARTIFACT = "artifact"
    FIGURE = "figure"
    TABLE = "table"
    FORMULA = "formula"
    EQUATION = "equation"
    CITATION = "citation"
    SECTION = "section"
    APPENDIX = "appendix"
    SUPPLEMENTAL = "supplemental"
    REFERENCE = "reference"
    HTML_ELEMENT = "element"
    CACHE_ENTRY = "cache"
    METADATA = "meta"
    EVENT = "event"
    STATE = "state"
    TRANSFORM = "transform"
    VALIDATION = "validation"

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
class ProcessingStateInfo:
    """
    Comprehensive state information for element processing.
    Tracks phase, state, history, and relationships.
    """
    # Core identification
    element_id: str
    phase: ProcessingPhase
    state: ProcessingState
    parent_id: Optional[str] = None

    # History tracking
    previous_state: Optional[ProcessingState] = None
    previous_phase: Optional[ProcessingPhase] = None
    timestamp: datetime = field(default_factory=datetime.now)

    # Error tracking
    error_message: Optional[str] = None
    error_count: int = 0

    # Processing metadata
    attempts: int = 0
    duration: Optional[float] = None
    cache_hits: int = 0

    # State flags
    is_locked: bool = False
    needs_reprocessing: bool = False
    is_cached: bool = False

    # Relations
    dependent_ids: List[str] = field(default_factory=list)
    blocking_ids: List[str] = field(default_factory=list)

    def update(
        self,
        new_state: Optional[ProcessingState] = None,
        new_phase: Optional[ProcessingPhase] = None,
        error: Optional[str] = None
    ) -> None:
        """Update state with history tracking."""
        if new_state:
            self.previous_state = self.state
            self.state = new_state

        if new_phase:
            self.previous_phase = self.phase
            self.phase = new_phase

        if error:
            self.error_message = error
            self.error_count += 1

        self.timestamp = datetime.now()

    def increment_attempt(self) -> None:
        """Increment processing attempt counter."""
        self.attempts += 1
        self.timestamp = datetime.now()

    def can_process(self) -> bool:
        """Check if element can be processed."""
        return (
            not self.is_locked and
            self.attempts < 3 and
            self.state != ProcessingState.ERROR and
            self.state != ProcessingState.COMPLETED
        )

    @property
    def has_error(self) -> bool:
        """Check if element has errors."""
        return bool(self.error_message) or self.error_count > 0

    @property
    def is_complete(self) -> bool:
        """Check if processing is complete."""
        return self.state == ProcessingState.COMPLETED

    @property
    def is_processing(self) -> bool:
        """Check if element is currently processing."""
        return self.state == ProcessingState.PROCESSING

    @property
    def is_blocked(self) -> bool:
        """Check if element is blocked by dependencies."""
        return bool(self.blocking_ids)

class ProcessingRuleType(Enum):
    """Types of processing rules with hierarchical validation."""
    ELEMENT = "element"           # Basic element processing
    TRANSFORMATION = "transform"  # Content transformation
    VALIDATION = "validation"     # Content validation
    SPECIALIZATION = "special"    # Content specialization
    ENRICHMENT = "enrichment"     # Content enrichment
    PUBLICATION = "publication"   # Publication-specific
    TITLES = "titles"             # Title processing
    BLOCKS = "blocks"             # Block element processing
    TABLES = "tables"             # Table processing
    EMPHASIS = "emphasis"         # Emphasis processing
    LINKS = "links"               # Link processing
    HEADINGS = "headings"         # Heading processing
    LISTS = "lists"               # List processing
    MEDIA = "media"               # Media processing
    INLINE = "inline"             # Inline element processing
    CODE_BLOCK = "code_block"     # Code block processing
    BLOCK = "block"               # Block element processing
    STRUCTURE = "structure"       # Structure processing
    TASK_ELEMENTS = "task_elements" # Task elements processing
    DEFINITION = "definition"     # Definition list entry processing
    METADATA = "metadata"         # Metadata processing
    NAVIGATION = "navigation"     # Navigation element processing
    DEFAULT = "default"           # Default processing

    @classmethod
    def validate_rule_type(cls, rule_type: str) -> bool:
        """Validate if a rule type is valid."""
        return rule_type in [member.value for member in cls]

    @classmethod
    def get_parent_type(cls, rule_type: str) -> Optional[str]:
        """Get parent rule type if exists."""
        hierarchy = {
            "titles": "element",
            "blocks": "element",
            "tables": "blocks",
            "emphasis": "inline",
            "links": "element",
            "headings": "titles",
            "lists": "blocks",
            "media": "element",
            "inline": "element",
            "definition": "blocks",
            "metadata": "element",
            "navigation": "element",
            "default": "element"
        }
        return hierarchy.get(rule_type)


class ContentType(Enum):
    DITA = "dita"
    MARKDOWN = "markdown"
    MAP = "map"
    TOPIC = "topic"
    UNKNOWN = "unknown"


@dataclass
class LogContext:
    """Context for structured logging."""
    operation_id: str
    operation_type: str
    content_id: str
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class YAMLFrontmatter:
    """Structured YAML frontmatter data."""
    feature_flags: Dict[str, bool]
    relationships: Dict[str, List[str]]
    context: Dict[str, Any]
    raw_data: Dict[str, Any]

@dataclass
class KeyDefinition:
    """DITA key definition structure."""
    key: str
    href: Optional[str]
    scope: str
    processing_role: str
    metadata: Dict[str, Any]
    source_map: str


##################################
# Metadata validation and schemas
##################################

@dataclass
class MetadataField:
    name: str
    value: Any
    content_type: ContentType
    source_id: str
    heading_id: Optional[str] = None
    timestamp: datetime = datetime.now()

@dataclass
class MetadataTransaction:
    """Represents a metadata update transaction."""
    content_id: str
    updates: Dict[str, Any]
    is_committed: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    conflicts: List[str] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from updates with a default fallback."""
        return self.updates.get(key, default)


class ValidationSeverity(Enum):
    """Validation message severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationMessage:
    """Represents a validation message."""
    path: str
    message: str
    severity: ValidationSeverity
    code: str

@dataclass
class ValidationResult:
    """Results of metadata validation."""
    is_valid: bool
    messages: List[ValidationMessage] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class MetadataState:
    """Tracks metadata state during processing."""
    content_id: str
    phase: ProcessingPhase
    state: ProcessingState
    timestamp: datetime = field(default_factory=datetime.now)
    cached: bool = False
    validation_state: Optional[ValidationResult] = None
    key_references: List[str] = field(default_factory=list)
    metadata_refs: Dict[str, str] = field(default_factory=dict)


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
    def create_map(cls,
        path: Path,
        title: str,
        id_handler: 'DITAIDHandler'
    ) -> "TrackedElement":
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
                id_type=IDType.MAP  # Use proper IDType enum
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
    def from_discovery(cls, path: Path, element_type: ElementType, id_handler: 'DITAIDHandler') -> "TrackedElement":
        """Create a TrackedElement during the discovery phase."""
        return cls(
            id=id_handler.generate_id(
                base=path.stem,
                id_type=IDType.TOPIC if element_type == ElementType.TOPIC else IDType.MAP
            ),
            type=element_type,
            path=path,
            source_path=path,
            content=""  # Content will be loaded later
        )


class ContentScope(Enum):
    """Content scope types for context tracking."""
    LOCAL = "local"      # Content within current topic
    PEER = "peer"        # Content in related topics
    EXTERNAL = "external"  # Content from external sources
    GLOBAL = "global"    # Content available everywhere

class ContentRelationType(Enum):
    """Types of content relationships."""
    PARENT = "parent"
    CHILD = "child"
    PREREQ = "prerequisite"
    RELATED = "related"
    REFERENCE = "reference"
    CONREF = "conref"
    KEYREF = "keyref"

@dataclass
class ContentRelationship:
    """Represents a relationship between content elements."""
    source_id: str
    target_id: str
    relation_type: ContentRelationType
    scope: ContentScope
    metadata: Dict[str, Any] = field(default_factory=dict)
    key_refs: Set[str] = field(default_factory=set)

@dataclass
class NavigationContext:
    """Navigation and structural context."""
    path: List[str]  # Hierarchical path
    level: int
    sequence: int
    parent_id: Optional[str]
    root_map: str
    siblings: List[str] = field(default_factory=list)

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update navigation context fields.

        Args:
            updates: Dictionary of fields to update
        """
        if new_path := updates.get('path'):
            self.path = new_path
        if 'level' in updates:
            self.level = updates['level']
        if 'sequence' in updates:
            self.sequence = updates['sequence']
        if 'parent_id' in updates:
            self.parent_id = updates['parent_id']
        if 'root_map' in updates:
            self.root_map = updates['root_map']
        if new_siblings := updates.get('siblings'):
            self.siblings = new_siblings


@dataclass
class ProcessingContext:
    """
    Processing context with relationship and scope awareness.
    Replaces old ProcessingContext implementation.
    """
    # Core identification
    context_id: str
    element_id: str
    element_type: ElementType

    # Processing state
    state_info: ProcessingStateInfo

    # Structural context
    navigation: NavigationContext
    scope: ContentScope

    # Relationships
    relationships: List[ContentRelationship] = field(default_factory=list)
    active_keys: Set[str] = field(default_factory=set)

    # Feature and condition context
    features: Dict[str, bool] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)

    # Metadata access (references, not storage)
    metadata_refs: Dict[str, str] = field(default_factory=dict)

    # New metadata fields
    metadata_state: MetadataState = field(default_factory=lambda: MetadataState(
        content_id="",
        phase=ProcessingPhase.DISCOVERY,
        state=ProcessingState.PENDING
    ))
    key_refs: Set[str] = field(default_factory=set)
    metadata_cache: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
            """Get context attribute with fallback."""
            return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for validation."""
        return {
            'pipeline_features': self.features.get('pipeline', {}),
            'content_type': self.element_type.value,
            'component': self.metadata_refs.get('component'),
            'features': self.features
        }

    def update_state(
        self,
        new_phase: Optional[ProcessingPhase] = None,
        new_state: Optional[ProcessingState] = None
    ) -> None:
        """Update processing state."""
        if new_phase:
            self.state_info.phase = new_phase
        if new_state:
            self.state_info.state = new_state
        self.state_info.timestamp = datetime.now()

@dataclass
class ElementReference:
    """NEW: Reference information for an element."""
    id: str
    type: ElementType
    text: str
    level: Optional[int] = None
    href: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessingMetadata:
    """
    Redesigned: Focuses on transient metadata during processing.
    Removes feature flags and element tracking (moved to other components).
    """
    # Core identification
    content_id: str
    content_type: ElementType
    content_scope: ContentScope
    language: str = "en"

    # Processing state
    processing_state: ProcessingStateInfo = field(default_factory=lambda: ProcessingStateInfo(
            element_id="",
            phase=ProcessingPhase.DISCOVERY,
            state=ProcessingState.PENDING
        ))

    # Cache for processing
    references: Dict[str, ElementReference] = field(default_factory=dict)
    transient_attributes: Dict[str, Any] = field(default_factory=dict)
    _reference_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)


    def add_reference(
        self,
        element_id: str,
        element_type: ElementType,
        text: str,
        **kwargs: Any
    ) -> None:
        """Add element reference with optional attributes."""
        self.references[element_id] = ElementReference(
            id=element_id,
            type=element_type,
            text=text,
            **kwargs
        )

    def get_reference(
        self,
        ref_id: str
    ) -> Optional[ElementReference]:
        """Get element reference by ID."""
        return self.references.get(ref_id)

    def add_transient_attribute(
        self,
        key: str,
        value: Any,
        scope: str = "global"
    ) -> None:
        """Add transient processing attribute."""
        if scope not in self.transient_attributes:
            self.transient_attributes[scope] = {}
        self.transient_attributes[scope][key] = value

    def get_transient_attribute(
        self,
        key: str,
        scope: str = "global"
    ) -> Optional[Any]:
        """Get transient processing attribute."""
        return self.transient_attributes.get(scope, {}).get(key)

    def cache_reference(
        self,
        ref_type: str,
        ref_id: str,
        data: Dict[str, Any]
    ) -> None:
        """Cache reference data for processing."""
        if ref_type not in self._reference_cache:
            self._reference_cache[ref_type] = {}
        self._reference_cache[ref_type][ref_id] = data

    def get_cached_reference(
        self,
        ref_type: str,
        ref_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached reference data."""
        return self._reference_cache.get(ref_type, {}).get(ref_id)

    def clear_cache(self) -> None:
        """Clear reference cache."""
        self._reference_cache.clear()

    def update_state(
        self,
        new_state: Optional[ProcessingState] = None,
        new_phase: Optional[ProcessingPhase] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Update processing state.

        Args:
            new_state: Optional new state
            new_phase: Optional new phase
            error: Optional error message
        """
        if new_state is not None:
            self.processing_state.state = new_state
        if new_phase is not None:
            self.processing_state.phase = new_phase
        if error is not None:
            self.processing_state.error_message = error
        self.processing_state.timestamp = datetime.now()

    def cleanup(self) -> None:
        """Cleanup transient data."""
        self.clear_cache()
        self.transient_attributes.clear()



@dataclass
class ProcessedContent:
    """Processed element content"""
    html: str
    element_id: str
    heading_level: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TopicType:
    """NEW: Topic type information."""
    name: str
    base_type: Optional[str] = None
    schema: Optional[str] = None
    specialization_rules: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Topic:
    """Topic with enhanced metadata and type handling."""
    # Core identification
    id: str
    path: Path
    content_type: ElementType  # DITA or MARKDOWN

    # Structural information
    parent_map_id: Optional[str] = None
    topic_type: TopicType = field(default_factory=lambda: TopicType(name="topic"))

    # Processing metadata
    processing_metadata: ProcessingMetadata = field(default_factory=lambda: ProcessingMetadata(
        content_id="",
        content_type=ElementType.UNKNOWN,
        content_scope=ContentScope.LOCAL
    ))

    # Rendering metadata
    title: Optional[str] = None
    short_desc: Optional[str] = None

    def __post_init__(self):
        """Initialize metadata after creation."""
        self.processing_metadata.content_id = self.id
        self.processing_metadata.content_type = self.content_type

        # If topic is specialized, set base type
        if self.topic_type.base_type:
            self.processing_metadata.transient_attributes["base_type"] = self.topic_type.base_type

    @property
    def template_path(self) -> Path:
        """Get template path based on content type and specialization."""
        base_path = "templates"

        if self.topic_type.base_type:
            return Path(f"{base_path}/{self.topic_type.base_type}/{self.content_type.value}.html")

        if self.topic_type.name != "topic":
            return Path(f"{base_path}/{self.topic_type.name}/{self.content_type.value}.html")

        return Path(f"{base_path}/{self.content_type.value}.html")

    @property
    def is_specialized(self) -> bool:
        """Check if topic is a specialization."""
        return self.topic_type.base_type is not None or self.topic_type.name != "topic"

    @property
    def specialization_type(self) -> Optional[str]:
        """Get specialization type if specialized."""
        if self.is_specialized:
            return self.topic_type.base_type or self.topic_type.name
        return None

    def add_transient_attribute(self, key: str, value: Any) -> None:
        """Add transient processing attribute."""
        self.processing_metadata.add_transient_attribute(key, value, scope="topic")

    def get_transient_attribute(self, key: str) -> Optional[Any]:
        """Get transient processing attribute."""
        return self.processing_metadata.get_transient_attribute(key, scope="topic")

    def update_state(
        self,
        phase: Optional[ProcessingPhase] = None,
        state: Optional[ProcessingState] = None
    ) -> None:
        """
        Update topic processing state.

        Args:
            phase: New processing phase
            state: New processing state
        """
        # First update the state
        if state is not None:
            self.processing_metadata.update_state(new_state=state)

        # Then update the phase if provided
        if phase is not None:
            self.processing_metadata.processing_state.phase = phase

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
    HIGHLIGHT = "highlight"
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
