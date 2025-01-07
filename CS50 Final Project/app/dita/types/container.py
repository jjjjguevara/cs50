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
    CitationFormat,
    Language,
    Platform,
    Audience,
    SubscriptionLevel,
    Source,
    Author,
    Contributor,
    Funding,
    Affiliation,
    Citation,
    Publication,
    GitMeta,
    Journal
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
    """Metadata container for scientific publishing."""

    # Core Identifiers
    id: str
    title: str

    # Academic Identifiers
    doi: Optional[str] = None

    # Journal Information
    journal: Optional[Journal] = None

    # Version Control
    version: str = "1.0"
    status: Status = Status.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: Optional[datetime] = None
    revision_history: List[str] = field(default_factory=list)

    # Basic Metadata
    description: Optional[str] = None
    language: Language = Language.EN
    keywords: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)

    # Authorship & Attribution
    authors: List[Author] = field(default_factory=list)
    contributors: List[Contributor] = field(default_factory=list)
    corresponding_author: Optional[Author] = None
    affiliations: List[Affiliation] = field(default_factory=list)

    # Research Context
    research_fields: List[str] = field(default_factory=list)
    methodology: Optional[str] = None
    data_availability: Optional[str] = None

    # Funding & Support
    funding: List[Funding] = field(default_factory=list)
    acknowledgements: Optional[str] = None

    # Ethics & Compliance
    ethics_statement: Optional[str] = None
    irb_number: Optional[str] = None
    approval_date: Optional[datetime] = None

    # Citations & References
    citations: List[Citation] = field(default_factory=list)
    references: List[Citation] = field(default_factory=list)
    citation_formats: CitationFormat = CitationFormat.MLA

    # Publications
    publications: List[Publication] = field(default_factory=list)

    # Content Management
    parent_id: Optional[str] = None
    root_map: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    related_content: List[str] = field(default_factory=list)

    # Access Control
    platform: List[Platform] = field(default_factory=list)
    audience: List[Audience] = field(default_factory=list)
    subscription_level: Optional[SubscriptionLevel] = None
    access_rights: List[str] = field(default_factory=list)
    permissions: Optional[dict] = field(default_factory=dict)

    # GitHub Integration
    github_metadata: Optional[GitMeta] = None  # e.g., trace of repository actions

    # SEO & Discovery
    meta_description: Optional[str] = None
    meta_keywords: List[str] = field(default_factory=list)
    open_graph: Optional[dict] = None
    twitter_card: Optional[dict] = None
    schema_org: Optional[dict] = None

    # Analytics
    views: int = 0
    downloads: int = 0
    citations_count: int = 0
    altmetrics: dict = field(default_factory=dict)

    # Publication Details
    publisher: Optional[str] = None
    publication_date: Optional[datetime] = None
    publication_status: Optional[str] = None
    peer_review_status: Optional[str] = None

    # Processing Metadata
    processing_history: List[str] = field(default_factory=list)
    validation_status: Optional[dict] = None
    transformation_info: Optional[dict] = None

    # Custom & Transient Metadata
    custom_metadata: dict = field(default_factory=dict)
    transient: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate and process metadata after initialization."""
        # Ensure datetime fields are datetime objects
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.modified_at, str):
            self.modified_at = datetime.fromisoformat(self.modified_at)
        if isinstance(self.publication_date, str):
            self.publication_date = datetime.fromisoformat(self.publication_date)

    @property
    def citation_count(self) -> int:
        """Get total citation count."""
        return len(self.citations)

    @property
    def is_published(self) -> bool:
        """Check if content is published."""
        return self.status == Status.PUBLISHED

    @property
    def has_peer_review(self) -> bool:
        """Check if content has been peer-reviewed."""
        return bool(self.peer_review_status)

    @property
    def author_count(self) -> int:
        """Get the number of authors."""
        return len(self.authors)

    def get_primary_author(self) -> Optional[Author]:
        """Get primary author information."""
        return self.authors[0] if self.authors else None

    def get_citation(self, format: CitationFormat = CitationFormat.MLA) -> Optional[str]:
        """Get a formatted citation in the specified format."""
        for citation in self.citations:
            if citation.format == format:
                return citation.citation_text
        return None

    def update_analytics(self, views: int = 0, downloads: int = 0) -> None:
        """Update analytics counts."""
        self.views += views
        self.downloads += downloads

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
    wrapper: Optional[str] = None
    events: List[str] = field(default_factory=list)

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
    """Container wrapper."""
    id: str
    context: Context
    metadata: Metadata
    feature: Optional[Feature] = None
    ruleset: Optional[RuleSet] = None
    custom_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Content:
    """Main content container."""
    id: str
    type: ElementType
    content_type: ContentType
    path: Path
    title: Optional[str]
    content: str
    phase: Phase
    state: State
    wrapper: Wrapper
