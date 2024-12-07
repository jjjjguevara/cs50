# app/models.py
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Set, Union
from enum import Enum

class ContentType(Enum):
    """Content type enumeration."""
    DITA = "dita"
    MARKDOWN = "markdown"
    MAP = "map"

class ContentStatus(Enum):
    """Content status enumeration."""
    DRAFT = "draft"
    REVIEW = "review"
    PUBLISHED = "published"
    ARCHIVED = "archived"

@dataclass
class Author:
    """Author metadata."""
    name: str
    email: Optional[str] = None
    affiliation: Optional[str] = None
    orcid: Optional[str] = None

@dataclass
class Citation:
    """Citation metadata."""
    doi: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    authors: List[str] = field(default_factory=list)
    title: Optional[str] = None
    url: Optional[str] = None

@dataclass
class TopicMetadata:
    """Comprehensive topic metadata."""
    title: str
    description: Optional[str] = None
    authors: List[Author] = field(default_factory=list)
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    version: str = "1.0"
    status: ContentStatus = ContentStatus.DRAFT
    keywords: Set[str] = field(default_factory=set)
    citations: List[Citation] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    content_type: ContentType = ContentType.DITA
    language: str = "en"
    abstract: Optional[str] = None
    review_status: Optional[str] = None
    review_date: Optional[datetime] = None
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MapMetadata:
    """Map metadata."""
    title: str
    description: Optional[str] = None
    author: Optional[Author] = None
    version: str = "1.0"
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    topics: List[str] = field(default_factory=list)
    status: ContentStatus = ContentStatus.DRAFT
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Topic:
    """Topic model."""
    id: str
    path: Path
    type: ContentType
    metadata: TopicMetadata
    content_hash: str
    last_modified: datetime

@dataclass
class Map:
    """DITA map model."""
    id: str
    path: Path
    topics: List[Topic]
    metadata: MapMetadata
    last_modified: datetime

@dataclass
class ProcessedContent:
    """Model for processed content."""
    id: str
    title: str
    html: str
    metadata: Union[TopicMetadata, MapMetadata]
    processed_at: datetime
    source_hash: str

@dataclass
class ContentCache:
    """Cache for processed content."""
    id: str
    content: ProcessedContent
    created_at: datetime
    expires_at: datetime
