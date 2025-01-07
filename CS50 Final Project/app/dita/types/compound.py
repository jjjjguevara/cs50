from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from .subjects import Category, Subcategory
from .base import (
    AffiliateType,
    AffiliateRank,
    DepartmentType,
    DepartmentRank,
    OrganizationBand,
    Initiative,
    ProjectRole,
    FundingSource,
    CitationFormat,
    CitationType,
    PublicationType,
    SecurityLevel,
    Status,
    User,
    Language,
    GitAction
)

@dataclass
class Affiliation:
    title: str
    type: AffiliateType
    rank: AffiliateRank
    department: Optional[str] = None # Department who engages with this affiliation.
    address: Optional[str] = None
    website: Optional[str] = None
    country: Optional[str] = None


@dataclass
class Person: # for internal users, contributors, workers, etc
    id: str
    firstname: str
    lastname: str
    department: "Department"
    rank: DepartmentRank
    band: OrganizationBand
    user: User = User.GUEST
    role: ProjectRole = ProjectRole.CONTRIBUTOR
    email: Optional[str] = None
    country: Optional[str] = None
    orcid: Optional[str] = None
    languages: List[Language] = field(default_factory=list)
    affiliations: List[Affiliation] = field(default_factory=list)

@dataclass
class Department:
    id: str
    title: str
    manager: Person
    type: DepartmentType
    secrecy: SecurityLevel
    initiatives: List[Initiative] = field(default_factory=list)


@dataclass
class Project:
    id: str
    title: str
    description: str
    initiative: Initiative
    status: Status
    manager: Person
    secrecy: SecurityLevel
    cohort: List[Person] = field(default_factory=list)
    funding: List["Funding"] = field(default_factory=list)
    url: Optional[str] = None

@dataclass
class Affiliate:
    id: str
    title: str
    type: AffiliateType
    rank: AffiliateRank
    department: Department # Department who engages with this affiliate.
    secrecy: SecurityLevel
    address: Optional[str] = None
    website: Optional[str] = None
    country: Optional[str] = None

@dataclass
class Author:
    id: str
    name: str
    email: Optional[str] = None
    orcid: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    affiliations: List[Affiliation] = field(default_factory=list)

@dataclass
class Contributor:
    id: str
    name: str
    email: Optional[str] = None
    orcid: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    affiliations: List[Affiliation] = field(default_factory=list)

@dataclass
class Funding:
    id: str
    funder: Affiliate
    source: FundingSource
    status: Status
    grant_number: Optional[str] = None
    acknowledgements: Optional[str] = None
    approval_date: Optional[datetime] = None
    recipients: List[Person] = field(default_factory=list)
    initiatives: List[Initiative] = field(default_factory=list)
    affiliations: List[Affiliation] = field(default_factory=list)

@dataclass
class Journal:
    id: str
    title: str
    category: Category
    subcategory: Optional[Subcategory] = None
    abbreviation: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    country: Optional[str] = None
    website: Optional[str] = None

@dataclass
class Publisher:
    id: str
    name: str
    category: Category
    subcategory: Optional[Subcategory] = None
    email: Optional[str] = None
    website: Optional[str] = None
    affiliations: List[Affiliation] = field(default_factory=list)

@dataclass
class Citation:
    id: str
    title: str
    citation_text: str
    type: CitationType
    category: Category
    subcategory: Optional[Subcategory] = None
    format: CitationFormat = CitationFormat.MLA
    authors: List[Author] = field(default_factory=list)
    journal: Optional[Journal] = None
    doi: Optional[str] = None
    issn: Optional[str] = None
    isbn: Optional[str] = None
    url: Optional[str] = None
    languages: List[Language] = field(default_factory=list)
    publication_date: Optional[datetime] = None

@dataclass
class Publication:
    id: str
    title: str
    type: PublicationType
    status: Status
    category: Category
    subcategory: Optional[Subcategory] = None
    authors: List[Author] = field(default_factory=list)
    description: Optional[str] = None
    journal: Optional[Journal] = None
    publisher: Optional[Publisher] = None
    edition: Optional[str] = None
    collection: Optional[str] = None
    doi: Optional[str] = None
    issn: Optional[str] = None
    isbn: Optional[str] = None
    url: Optional[str] = None
    languages: List[Language] = field(default_factory=list)
    version: str = "1.0.0"
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    publication_date: Optional[datetime] = None
    revision_history: List[str] = field(default_factory=list)

@dataclass
class Analytics:
    # Existing SEO & Discovery
    meta_description: Optional[str] = None
    meta_keywords: List[str] = field(default_factory=list)
    open_graph: Optional[dict] = None
    twitter_card: Optional[dict] = None
    schema_org: Optional[dict] = None

    # New SEO Fields
    canonical_url: Optional[str] = None
    headings: List[str] = field(default_factory=list)
    schema_properties: Optional[dict] = None
    alternate_languages: List[Language] = field(default_factory=list)
    keyword_clusters: Dict[str, List[str]] = field(default_factory=dict)
    last_updated: Optional[datetime] = None
    primary_author: Optional[Author] = None
    social_engagement: Dict[str, int] = field(default_factory=dict)

    # Existing Analytics
    views: int = 0
    downloads: int = 0
    citations_count: int = 0
    altmetrics: dict = field(default_factory=dict)

    # Additional Analytics
    session_count: int = 0
    average_time_on_page: float = 0.0
    scroll_depth: float = 0.0
    ctr: float = 0.0
    conversion_actions: List[str] = field(default_factory=list)
    media_engagement: Dict[str, int] = field(default_factory=dict)
    user_demographics: Dict[str, Any] = field(default_factory=dict)
    referral_sources: List[str] = field(default_factory=list)
    engagement_path: List[str] = field(default_factory=list)
    error_count: int = 0

    # Future-Ready Fields
    annotated_entities: List[Dict[str, Any]] = field(default_factory=list)
    user_feedback: List[Dict[str, Any]] = field(default_factory=list)
    oa_metrics: Dict[str, int] = field(default_factory=dict)

@dataclass
class GitMeta:
    action: GitAction
    repository: str
    branch: Optional[str] = None
    commits: List[str] = field(default_factory=list)

    @property
    def trace(self) -> str:
        """Compose a trace string with repository, branch, action, and timestamp."""
        timestamp = datetime.now().isoformat()
        branch_part = self.branch if self.branch else "default"
        return f"{self.repository}/{branch_part}/{self.action.name}/{timestamp}"
