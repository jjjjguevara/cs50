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
    Priority,
    Source,
    User,
    Language,
    GitAction
)


@dataclass
class Affiliate: # A person or a group that engages with a project, almost always external
    id: str
    name: str
    job_title: str
    affiliation: "Affiliation"
    department: "Department" # Department who engages with this affiliate.
    endorser: "Person" # Sponsor of this affiliate
    clearance: SecurityLevel


@dataclass
class Affiliation: # Group of attributes of an affiliate
    id: str
    title: str
    type: AffiliateType
    rank: AffiliateRank # Ranks are not job titles, but rather their stake at our organization
    status: Status
    secrecy: SecurityLevel
    endorser: "Person" # The owner of this affiliation
    department: Optional[str] = None # Department who endorses this affiliation.
    address: Optional[str] = None
    website: Optional[str] = None
    country: Optional[str] = None
    est: Optional[datetime] = None # Date this affiliation started


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
class Department: # A division within our organization
    id: str
    title: str
    manager: Person
    type: DepartmentType
    secrecy: SecurityLevel
    initiatives: List[Initiative] = field(default_factory=list)


@dataclass
class Project: # A collection of tasks, always backs an initiative
    id: str
    title: str
    description: str
    initiative: Initiative
    status: Status
    manager: Person
    secrecy: SecurityLevel = SecurityLevel.RESTRICTED
    priority: Priority = Priority.LOW
    tasks: List["Task"] = field(default_factory=list) # Tasks within a project
    cohort: List[Person] = field(default_factory=list) # People working on a project
    project_start: Optional[datetime] = None
    project_end: Optional[datetime] = None
    funding: List["Funding"] = field(default_factory=list) # Budget source for a project
    url: Optional[str] = None

@dataclass
class Task: # A single task within a project
    id: str
    title: str
    description: str
    status: Status = Status.OPEN
    priority: Priority = Priority.LOW
    secrecy: SecurityLevel = SecurityLevel.INTERNAL
    deadline: Optional[datetime] = None
    url: Optional[str] = None





@dataclass
class Author:
    id: str
    name: str
    email: Optional[str] = None
    orcid: Optional[str] = None
    roles: List[ProjectRole] = field(default_factory=list) # Must include AUTHOR
    affiliations: List[Affiliation] = field(default_factory=list)

@dataclass
class Contributor:
    id: str
    name: str
    role: ProjectRole
    email: Optional[str] = None
    orcid: Optional[str] = None
    affiliations: List[Affiliation] = field(default_factory=list)
    contribution_date: Optional[datetime] = None
    review_status: Optional[Status] = None  # For reviewers

@dataclass
class Funding:
    id: str
    funder: Affiliate
    source: FundingSource
    status: Status
    approver: Person # Person who approves the funding
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
    source: Optional[Source] = None # The source of the citation
    source_rank: Optional[str] = None # A ranking number for the source, based on a TBD logic
    authors: List[Author] = field(default_factory=list)
    journal: Optional[Journal] = None
    doi: Optional[str] = None
    issn: Optional[str] = None
    isbn: Optional[str] = None
    url: Optional[str] = None
    accessed: Optional[datetime] = None # Date the citation was last accessed
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
