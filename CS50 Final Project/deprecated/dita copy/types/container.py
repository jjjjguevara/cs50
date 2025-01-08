"""Container types for DITA processing system."""
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from ..types import (
    Phase,
    State,
    Scope,
    Status,
    FeatureType,
    ContentType,
    ElementType,
    Environment,
    CacheStrategy,
    Language,
    Audience,
    Author,
    Contributor,
    Funding,
    Citation,
    Publication,
    GitMeta,
    Analytics,
    Category,
    Subcategory,
    SecurityLevel,
    Rights,
    User,
    TopicType,
    MapType,
    Mode,
    ValidationLevel
)


@dataclass
class Message:
    """System message container."""
    path: str
    message: str
    severity: State
    code: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Event:
    """Event container."""
    id: str
    type: str
    target: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    state: State = State.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class Context:
    """Content context container."""
    id: str
    type: ContentType
    values: Dict[str, Any] = field(default_factory=dict)
    references: Dict[str, List[str]] = field(default_factory=dict)
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    collections: Dict[str, Set[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Metadata:
    """Core processing metadata container."""
    # Core Identity
    id: str
    type: ContentType  # DITA, Markdown, etc.
    status: Status

    # Processing Context
    phase: Phase = Phase.DISCOVERY
    state: State = State.PENDING
    scope: Scope = Scope.LOCAL

    # Processing Controls
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    dita_props: Dict[str, Any] = field(default_factory=dict)
    processing_rules: Dict[str, Any] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)

    # Cache & State
    cache_strategy: CacheStrategy = CacheStrategy.DEFAULT
    processing_history: List[Dict[str, Any]] = field(default_factory=list)
    validation_status: Dict[str, Any] = field(default_factory=dict)

    # Security & Access
    security_level: SecurityLevel = SecurityLevel.PUBLIC
    access_rights: Rights = Rights.DEFAULT
    permissions: User = User.GUEST

    def add_processing_event(self, phase: Phase, state: State, message: Optional[str] = None) -> None:
        """Record a processing event."""
        self.processing_history.append({
            "phase": phase,
            "state": state,
            "timestamp": datetime.now(),
            "message": message
        })
        self.phase = phase
        self.state = state

@dataclass
class Feature:
    """Feature container."""
    # Core Identity
    id: str
    name: str
    type: FeatureType
    scope: Scope

    # Content
    content: Union[str, Dict[str, Any]]
    target: str

    # Integration
    position: Optional[str] = None
    wrapper: Optional["Wrapper"] = None
    events: List[Event] = field(default_factory=list)

    # Optional configurations
    template: Optional[str] = None
    placeholder: Optional[str] = None

    # Behavior
    enabled: bool = True
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    processing_instructions: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)

    # Presentation
    style: Dict[str, Any] = field(default_factory=dict)
    format: Optional[str] = None
    numbering: Optional[Dict[str, Any]] = None
    layout: Optional[Dict[str, Any]] = None

    # Metadata
    description: Optional[str] = None
    version: str = "1.0"
    author: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: Optional[datetime] = None
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RuleSet:
    """Rule set container."""
    id: str
    type: str
    patterns: List[Dict[str, Any]]
    conditions: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Cache:
    """Cache container."""
    key: str
    value: Any
    type: str
    strategy: CacheStrategy = CacheStrategy.DEFAULT
    expiry: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Config:
    """Configuration container."""
    id: str
    type: str
    settings: Dict[str, Any]
    scope: Scope = Scope.GLOBAL
    environment: Environment = Environment.DEVELOPMENT
    overrides: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Wrapper:
    """Context wrapper for content processing."""
    # Core Identity
    id: str

    # Processing Context
    context: Context  # Base processing context
    metadata: Metadata  # Processing metadata

    # Content Classification
    category: Category
    subcategory: Optional[Subcategory] = None

    # Processing Controls
    feature: Optional[Feature] = None  # Active feature configuration
    ruleset: Optional[RuleSet] = None  # Active processing rules

    # Reference Resolution
    references: Dict[str, List[str]] = field(default_factory=dict)  # type -> [ref_ids]
    key_refs: Dict[str, str] = field(default_factory=dict)  # key -> resolved_value

    # Inheritance Chain
    parent_wrapper: Optional["Wrapper"] = None
    inheritance_chain: List[str] = field(default_factory=list)  # List of parent wrapper ids

    # Processing Configuration
    processing_mode: Mode = Mode.STRICT
    validation_level: ValidationLevel = ValidationLevel.STRICT
    cache_strategy: CacheStrategy = CacheStrategy.DEFAULT

    # State Tracking
    processing_stack: List[Dict[str, Any]] = field(default_factory=list)
    validation_results: List[Dict[str, Any]] = field(default_factory=list)

    # Custom Data
    custom_data: Dict[str, Any] = field(default_factory=dict)

    def add_reference(self, ref_type: str, ref_id: str) -> None:
        """Add a reference to track."""
        self.references.setdefault(ref_type, []).append(ref_id)

    def add_key_ref(self, key: str, value: str) -> None:
        """Add a key reference resolution."""
        self.key_refs[key] = value

    def get_inherited_rules(self) -> Dict[str, Any]:
        """Get rules including inherited ones."""
        rules = {}
        if self.parent_wrapper:
            rules.update(self.parent_wrapper.get_inherited_rules())
        if self.ruleset:
            rules.update({rule['id']: rule for rule in self.ruleset.patterns})
        return rules

    def push_state(self, phase: Phase, state: Dict[str, Any]) -> None:
        """Push processing state to stack."""
        self.processing_stack.append({
            'phase': phase,
            'state': state,
            'timestamp': datetime.now()
        })

    def pop_state(self) -> Optional[Dict[str, Any]]:
        """Pop processing state from stack."""
        return self.processing_stack.pop() if self.processing_stack else None

    def add_validation_result(self, phase: Phase, result: Dict[str, Any]) -> None:
        """Add validation result."""
        self.validation_results.append({
            'phase': phase,
            'result': result,
            'timestamp': datetime.now()
        })

class Topic:
    """DITA Topic container."""
    # Core Identity
    id: str
    title: str
    type: TopicType
    status: Status

    # Content Classification
    category: Category
    subcategory: Optional[Subcategory] = None

    # Academic Context
    publication: Optional[Publication] = None  # Contains journal, publisher, etc.
    authors: List[Author] = field(default_factory=list)
    contributors: List[Contributor] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    funding: List[Funding] = field(default_factory=list)

    # Content Context
    abstract: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    language: Language = Language.EN
    audience: List[Audience] = field(default_factory=list)

    # Hierarchy & References
    parent_id: Optional[str] = None
    root_map_id: Optional[str] = None
    references: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    # Navigation
    level: int = 0
    sequence: int = 0
    siblings: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)

    # Version Control
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: Optional[datetime] = None
    revision_history: List[Dict[str, Any]] = field(default_factory=list)
    git_metadata: Optional[GitMeta] = None

    # Analytics
    analytics: Optional[Analytics] = None

    # Processing Metadata
    metadata: Metadata = field(default_factory=lambda: Metadata(
        id="",  # Will be set in post_init
        type=ContentType.DITA,
        status=Status.DRAFT
    ))

    def __post_init__(self):
        """Initialize metadata with topic id."""
        if not self.metadata.id:
            self.metadata.id = self.id

@dataclass
class Map:
    """DITA Map container."""
    # Core Identity
    id: str
    title: str
    type: MapType
    status: Status

    # Map Structure
    topics: List[Topic] = field(default_factory=list)
    topic_refs: Dict[str, List[str]] = field(default_factory=dict)  # topic_id -> [ref_ids]
    relationships: Dict[str, List[str]] = field(default_factory=dict)  # topic_id -> [related_ids]

    # Navigation Structure
    toc_entries: List[Dict[str, Any]] = field(default_factory=list)
    hierarchy: Dict[str, List[str]] = field(default_factory=dict)  # parent_id -> [child_ids]
    order: List[str] = field(default_factory=list)  # Ordered list of topic ids

    # Publication Context
    publication: Optional[Publication] = None
    publication_date: Optional[datetime] = None
    authors: List[Author] = field(default_factory=list)
    contributors: List[Contributor] = field(default_factory=list)

    # Map Properties
    keys: Dict[str, Any] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    reltables: List[Dict[str, Any]] = field(default_factory=list)

    # Version Control
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: Optional[datetime] = None
    revision_history: List[Dict[str, Any]] = field(default_factory=list)
    git_metadata: Optional[GitMeta] = None

    # Analytics
    analytics: Optional[Analytics] = None

    # Processing Metadata
    metadata: Metadata = field(default_factory=lambda: Metadata(
        id="",  # Will be set in post_init
        type=ContentType.DITA,
        status=Status.DRAFT
    ))

    def __post_init__(self):
        """Initialize metadata with map id."""
        if not self.metadata.id:
            self.metadata.id = self.id

    def add_topic(self, topic: Topic, parent_id: Optional[str] = None) -> None:
        """Add topic to map with proper hierarchy."""
        self.topics.append(topic)
        self.order.append(topic.id)
        if parent_id:
            self.hierarchy.setdefault(parent_id, []).append(topic.id)
            topic.parent_id = parent_id
            topic.root_map_id = self.id
            # Update topic's level based on parent
            parent_level = next((t.level for t in self.topics if t.id == parent_id), -1)
            topic.level = parent_level + 1

    def get_topic_hierarchy(self, topic_id: str) -> List[str]:
        """Get full path from root to topic."""
        path = []
        current = topic_id
        while current:
            path.insert(0, current)
            current = next((t.parent_id for t in self.topics if t.id == current), None)
        return path

@dataclass
class Content:
    """Content container for processing pipeline."""
    # Core Identity
    id: str
    type: ElementType      # More granular than ContentType - specific element type
    content_type: ContentType  # High-level type (DITA, Markdown, etc.)

    # Source Information
    path: Path
    title: Optional[str]
    content: str          # Raw content

    # Processing State
    phase: Phase
    state: State

    # Processing Context
    wrapper: "Wrapper"

    # Content Objects - Only one should be populated based on type
    topic: Optional[Topic] = None        # If type is a topic-type element
    map: Optional[Map] = None           # If type is a map-type element
    element: Optional[Dict[str, Any]] = None  # For other element types

    # Processing Artifacts
    transformed_content: Optional[str] = None  # Content after transformation
    rendered_content: Optional[str] = None    # Final rendered output

    def __post_init__(self):
        """Validate content structure."""
        if self.type in {
            ElementType.TOPIC, ElementType.CONCEPT,
            ElementType.TASK, ElementType.REFERENCE
        }:
            if not self.topic:
                raise ValueError(f"Topic content required for type {self.type}")
        elif self.type in {ElementType.MAP, ElementType.BOOKMAP}:
            if not self.map:
                raise ValueError(f"Map content required for type {self.type}")
        elif not self.element:
            raise ValueError(f"Element content required for type {self.type}")
