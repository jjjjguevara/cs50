"""Base types for DITA processing system."""
from enum import Enum, Flag, auto

class Phase(Enum):
    """Processing phases."""
    DISCOVERY = "discovery"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    ENRICHMENT = "enrichment"
    ASSEMBLY = "assembly"
    ERROR = "error"

class State(Enum):
    """Processing states."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    CACHED = "cached"
    INVALIDATED = "invalidated"
    BLOCKED = "blocked"

class Scope(Enum):
    """Processing scopes."""
    LOCAL = "local"
    PEER = "peer"
    EXTERNAL = "external"
    GLOBAL = "global"
    INHERITED = "inherited"
    SPECIALIZED = "specialized"

class Status(Enum):
    """Content status."""
    OPEN = "open"
    DRAFT = "draft"
    REVIEW = "review"
    REQUESTED = "requested"
    APPROVED = "approved"
    PUBLISHED = "published"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"
    EXPERIMENTAL = "experimental"
    ACTIVE = "active"
    INACTIVE = "inactive"

class Mode(Enum):
    """Processing modes."""
    STRICT = "strict"
    PERMISSIVE = "permissive"
    DISCOVERY = "discovery"
    DEBUG = "debug"
    PRODUCTION = "production"
    DEVELOPMENT = "development"

class ContextType(Enum):
    DOCUMENT = "document"
    REFERENCE = "reference"
    NAVIGATION = "navigation"
    PROCESSING = "processing"
    SPECIALIZATION = "specialization"
    RESOLUTION = "resolution"
    FEATURE = "feature"
    VALIDATION = "validation"
    PUBLISHING = "publishing"
    METADATA = "metadata"

class Attribute(Enum):
    """Attribute types."""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    ENUM = "enum"
    ID = "id"
    IDREF = "idref"
    IDREFS = "idrefs"
    NMTOKEN = "nmtoken"
    NMTOKENS = "nmtokens"
    ENTITY = "entity"
    ENTITIES = "entities"
    NOTATION = "notation"
    XML = "xml"
    CDATA = "cdata"
    DTD = "dtd"
    YAML = "yaml"
    JSON = "json"

class Action(Enum):
    """Actions to use with Strategies and DITA Props."""
    INCLUDE = "include"
    EXCLUDE = "exclude"
    PASSTHROUGH = "passthrough"
    FLAG = "flag"
    SKIP = "skip"
    MERGE = "merge"
    NOMERGE = "nomerge"
    CASCADE = "cascade"
    OVERRIDE = "override"
    ADD = "add"
    SELECT = "select"
    SPECIALIZE = "specialize"
    INHERIT = "inherit"
    NAVIGATE = "navigate"
    FILTER = "filter"
    REMOVE = "remove"
    COMBINE = "combine"
    SPLIT = "split"
    CLUSTER = "cluster"
    REBUILD = "rebuild"
    CHUNK = "chunk"
    RESOLVE = "resolve"
    APPEND = "append"
    REBASE = "rebase"
    IMPOSE = "impose"
    KEEP = "keep"
    VALIDATE = "validate"
    IDENTIFY = "identify"
    CLASSIFY = "classify"
    EXTRACT = "extract"
    APPLY = "apply"
    TRANSFORM = "transform"
    PROCESS = "process"
    GENERATE = "generate"
    ASSEMBLE = "assemble"
    FINALIZE = "finalize"
    CUSTOM = "custom"

class Role(Flag):
    """Relationship types."""
    ANCESTOR = auto()
    CHILD = auto()
    COUSIN = auto()
    DESCENDANT = auto()
    FRIEND = auto()
    NEXT = auto()
    OTHER = auto()
    PARENT = auto()
    PREVIOUS = auto()
    SIBLING = auto()

class Collection(Enum):
    """Collection types."""
    UNORDERED = "unordered"
    SEQUENCE = "sequence"
    CHOICE = "choice"
    FAMILY = "family"
    ALL = "all"


class Linking(Enum):
    """Linking types."""
    TARGETONLY = "targetonly"
    SOURCEANDTARGET = "sourceandtarget"
    SOURCEONLY = "sourceonly"
    NORMAL = "normal"
    NONE = "none"

class ElementType(Enum):
    """Element types."""
    # Core types
    MAP = "map"
    TOPIC = "topic"
    CONCEPT = "concept"
    TASK = "task"
    REFERENCE = "reference"
    GLOSSARY = "glossary"
    BOOKMAP = "bookmap"

    # Structural elements
    HEADING = "heading"
    TITLE = "title"
    MAP_TITLE = "map_title"
    SHORTDESC = "shortdesc"
    BODY = "body"
    SECTION = "section"
    ABSTRACT = "abstract"
    DEFAULT = "default"
    SPECIALIZATION = "specialization"
    TOOLTIP = "tooltip"

    # Block elements
    PARAGRAPH = "paragraph"
    CODE_BLOCK = "codeblock"
    PRE = "pre"
    QUOTE = "quote"
    NOTE = "note"
    WARNING = "warning"
    TIP = "tip"
    IMPORTANT = "important"
    CALLOUT = "callout"
    TASKBODY = "taskbody"


    # List elements
    UNORDERED_LIST = "ul"
    ORDERED_LIST = "ol"
    LIST_ITEM = "li"
    DEFINITION_LIST = "dl"
    DEFINITION_TERM = "dt"
    DEFINITION_DESC = "dd"
    TODO_LIST = "todo"

    # Table elements
    TABLE = "table"
    TABLE_HEAD = "thead"
    TABLE_BODY = "tbody"
    TABLE_ROW = "tr"
    TABLE_HEADER = "th"
    TABLE_DATA = "td"

    # Inline elements
    LINK = "link"
    XREF = "xref"
    TOPICREF = "topicref"
    KEYWORD = "keyword"
    TERM = "term"
    PHRASE = "ph"
    TEXT = "text"
    BOLD = "b"
    ITALIC = "i"
    UNDERLINE = "u"
    HIGHLIGHT = "highlight"
    STRIKETHROUGH = "strikethrough"
    CODE = "code"
    SUBSCRIPT = "sub"
    SUPERSCRIPT = "sup"

    # Media elements
    IMAGE = "image"
    FIGURE = "fig"
    VIDEO = "video"
    AUDIO = "audio"
    MEDIA = "media"

    # Special elements
    ARTIFACT = "artifact"
    EQUATION = "equation"
    FORMULA = "formula"
    COMPONENT = "component"
    PLACEHOLDER = "placeholder"

    # Task-specific elements
    STEPS = "steps"
    STEP = "step"
    CMD = "cmd"
    INFO = "info"
    SUBSTEPS = "substeps"
    CHOICES = "choices"
    CHOICE = "choice"

    # Reference elements
    PROPERTIES = "properties"
    PROPERTY = "property"
    PROPTYPE = "proptype"
    PROPVALUE = "propvalue"
    PROPDESC = "propdesc"

class ContentType(Enum):
    """Content types."""
    DITA = "dita"
    MARKDOWN = "markdown"
    XML = "xml"
    HTML = "html"
    TEXT = "text"
    PDF = "pdf"
    EPUB = "epub"
    YAML = "yaml"
    JSON = "json"
    OTHER = "other"


class TopicType(Enum):
    """DITA Topic types."""
    TASK = "task"                 # Step-by-step instructions.
    REFERENCE = "reference"       # Structured information, such as tables or glossaries.
    CONCEPT = "concept"           # Explanation of ideas or concepts.
    GENERAL = "general"           # General-purpose topic.
    TROUBLESHOOTING = "troubleshooting" # Guidance on resolving issues or errors.
    HOWTO = "howto"               # Instructions for a specific task.
    FAQ = "faq"                   # Frequently asked questions.
    GUIDE = "guide"               # Provides directions for a task.
    TUTORIAL = "tutorial"         # Provides instructions for a task.
    OVERVIEW = "overview"         # Introduces a topic or concept.
    CONTENT = "content"           # Provides information about a topic.
    POLICY = "policy"             # Describes a policy.
    TASKGROUP = "taskgroup"       # Groups a set of tasks as a unit.
    SUBJECTSCHEME = "subjectscheme" # Defines controlled values for metadata.
    CLASSIFICATION = "classification" # Maps topics based on classifications.


class MapType(Enum):
    """DITA Map types."""
    STANDARD = "standard"         # Basic DITA map.
    BOOKMAP = "bookmap"           # Used for publishing books with chapters and sections.
    INDEX = "index"
    GLOSSARY = "glossary"
    HIERARCHY = "hierarchy"
    SUBJECTSCHEME = "subjectscheme" # Defines controlled values for metadata.
    CLASSIFICATION = "classification" # Maps topics based on classifications.
    LEARNINGMAP = "learningmap"   # Organizes topics for e-learning.
    KNOWLEDGEMAP = "knowledgemap" # Structures knowledge topics and their relationships.

class Expanse(Enum):
    """Expanse types."""
    COLUMN = "column"
    PAGE = "page"
    SPREAD = "spread"
    TEXTLINE = "textline"

class Platform(Enum):
    """Platforms."""
    LINUX = "linux"
    WINDOWS = "windows"
    MACOS = "macos"
    IOS = "ios"
    ANDROID = "android"
    WEB = "web"
    MOBILE = "mobile"

class Product(Enum):
    """Products."""
    HEX21 = "hex21"
    PERSONAL = "personal"
    ACADEMIC = "journal"
    GOVERNMENT = "government"
    FINANCIAL = "financial"
    EDUCATIONAL = "educational"
    ENTERPRISE = "enterprise"
    INDEPENDENT = "independent"
    NEWS = "news"

class PublicationType(Flag):
    """Publication types."""
    ARTICLE = auto()
    MAGAZINE = auto()
    JOURNAL = auto()
    REPORT = auto()
    BOOK = auto()
    TRANSCRIPT = auto()
    PODCAST = auto()
    ISSUE = auto()
    BRIEF = auto()
    NOTE = auto()
    BLOG = auto()
    PREPRINT = auto()
    CONFERENCE = auto()
    WORKSHOP = auto()
    PATENT = auto()
    DATASET = auto()
    WEBSITE = auto()
    DOCUMENTATION = auto()
    ERRATA = auto()

class SubscriptionLevel(Enum):
    """Subscription levels."""
    FREE = "free"
    PREMIUM = "premium"
    FOUNDER = "founder"
    CORPORATE = "corporate"

class Policy(Enum):
    """Policy types."""
    PRIVACY_POLICY = "privacy_policy"  # Guidelines on data collection and usage
    TERMS_OF_SERVICE = "terms_of_service"  # Rules governing service usage
    USAGE_RESTRICTIONS = "usage_restrictions"  # Limitations on how content can be used
    LICENSE_AGREEMENTS = "license_agreements"  # Terms under which content can be used or shared

class License(Enum):
    """License types."""
    CC_BY = "cc-by"  # Creative Commons Attribution
    CC_BY_SA = "cc-by-sa"  # Creative Commons Attribution-ShareAlike
    CC_BY_NC = "cc-by-nc"  # Creative Commons Attribution-NonCommercial
    CC_BY_NC_SA = "cc-by-nc-sa"  # Creative Commons Attribution-NonCommercial-ShareAlike
    MIT = "mit"  # MIT License
    CC0 = "cc0"  # Public Domain Dedication)
    GPL = "gpl"  # GNU General Public License

class Audience(Flag):
    """Audiences."""
    PUBLIC = auto()
    DEVELOPERS = auto()
    ADMINISTRATORS = auto()
    TECHNICAL_WRITERS = auto()
    MANAGERS = auto()
    EDUCATORS = auto()
    STUDENTS = auto()
    RESEARCHERS = auto()
    GOVERNMENT = auto()
    CORPORATIONS = auto()
    NONPROFITS = auto()
    FREELANCERS = auto()
    MEDIA = auto()
    CONSUMERS = auto()

class Source(Flag):
    """Source types."""
    INTERNAL = auto()
    EXTERNAL = auto()
    INDEPENDENT = auto()
    PRIMARY = auto()         # Original, firsthand data or evidence.
    SECONDARY = auto()       # Analysis, interpretation, or summary of primary sources.
    TERTIARY = auto()        # Summaries or compilations of secondary sources (e.g., encyclopedias).
    QUALITATIVE = auto()     # Non-numerical, descriptive data or sources.
    QUANTITATIVE = auto()    # Numerical or statistical data.
    SCHOLARLY = auto()       # Academic and peer-reviewed works.
    POPULAR = auto()         # Non-academic, general audience works.
    GOVERNMENT = auto()      # Official documents or publications from government entities.
    CORPORATE = auto()       # Reports or publications from businesses or organizations.
    MEDIA = auto()           # News articles, broadcasts, or other media outlets.
    LEGAL = auto()           # Legal cases, statutes, or regulatory documents.
    STATISTICAL = auto()     # Data or statistics from reputable sources.
    HISTORICAL = auto()      # Historical documents or archives.
    ANONYMOUS = auto()       # Anonymous sources.

class AffiliateType(Enum):
    """Affiliation types."""
    INTERNAL = "internal"
    INDIVIDUAL = "individual"
    ORGANIZATION = "organization"
    INSTITUTION = "institution"
    DEPARTMENT = "department"
    FACULTY = "faculty"
    UNIVERSITY = "university"
    COLLEGE = "college"
    SCHOOL = "school"
    PUBLISHER = "publisher"              # Academic or professional publisher.
    COMMUNITY = "community"              # Community-based organizations.
    INSTITUTE = "institute"              # Research institutes or centers.
    THINK_TANK = "think_tank"            # Research institutes or centers.
    RESEARCH_LAB = "research_lab"        # Specialized research facilities or labs.
    RESEARCH_CENTER = "research_center"  # Specialized research facilities or labs.
    NON_PROFIT = "non_profit"            # Charitable or non-governmental organizations.
    CORPORATION = "corporation"          # Private or public businesses.
    GOVERNMENT = "government"            # Governmental agencies or bodies.
    MILITARY = "military"                # Armed forces or defense entities.
    HOSPITAL = "hospital"                # Medical institutions.
    CLINIC = "clinic"                    # Smaller or specialized medical facilities.
    INDUSTRY = "industry"                # Sector-specific industrial organizations.
    INTERNATIONAL = "international"      # Global organizations or entities.
    CITY = "city"                        # Municipal-level affiliation.
    STATE = "state"                      # Regional or state-level affiliation.
    TRIBE = "tribe"                      # Indigenous or tribal group affiliation.
    RELIGIOUS = "religious"              # Churches, temples, or other religious groups.
    NETWORK = "network"                  # Collaborative groups or networks.
    OTHER = "other"                      # Any other unspecified type.

class AffiliateRank(Flag):
    """Affiliation ranks."""
    FOUNDER = auto()
    PARTNER = auto()
    MEMBER = auto()
    COOPERATOR = auto()
    CONTRIBUTOR = auto()
    SPONSOR = auto()
    SUPPORTER = auto()
    TOKENHOLDER = auto()
    SHAREHOLDER = auto()
    CONTRACTOR = auto()
    SUPPLIER = auto()
    ASSOCIATE = auto()
    CLIENT = auto()
    CUSTOMER = auto()
    VISITOR = auto()

class DepartmentType(Enum):
    """Departments."""
    ADMINISTRATION = "administration"
    MANAGEMENT = "management"
    BOARD = "board"
    ADVISORY = "advisory"
    OPERATIONS = "operations"
    PRODUCTION = "production"
    RESEARCH = "research"
    DEVELOPMENT = "development"
    TRAINING = "training"
    SALES = "sales"
    MARKETING = "marketing"
    SUPPORT = "support"
    FINANCE = "finance"
    ACCOUNTING = "accounting"
    QUALITY = "quality"
    LEGAL = "legal"
    CRM = "crm"
    HR = "hr"
    IT = "it"

class DepartmentRank(Flag):
    """Department ranks."""
    MANAGER = auto()
    DIRECTOR = auto()
    EXECUTIVE = auto()
    SENIOR_EXECUTIVE = auto()
    CHIEF_EXECUTIVE = auto()
    LEAD = auto()
    LEADER = auto()
    VP = auto()
    PRESIDENT = auto()
    CHAIR = auto()
    CHAIRMAN = auto()
    SECRETARY = auto()
    ADMINISTRATOR = auto()
    ADVISOR = auto()

class OrganizationBand(Enum):
    """Organization bands."""
    DEFAULT = "default"
    FIRST = "first"
    SECOND = "second"
    THIRD = "third"
    FOURTH = "fourth"
    FIFTH = "fifth"
    SIXTH = "sixth"
    SEVENTH = "seventh"

class FundingSource(Flag):
    """Funding sources."""
    INTERNAL = auto()
    EXTERNAL = auto()
    GRANT = auto()
    PRIVATE = auto()
    PUBLIC = auto()
    CROWDFUNDED = auto()
    GOVERNMENT = auto()
    SCHOLARSHIP = auto()
    MENTORSHIP = auto()
    INVESTOR = auto()
    VENTURE = auto()
    SAVINGS = auto()
    LOAN = auto()
    LOTTERY = auto()

class Initiative(Flag):
    """Strategic initiatives."""
    GLOBAL = auto()
    OUTREACH = auto()
    ENGAGEMENT = auto()
    INNOVATION = auto()
    RESEARCH = auto()
    TRANSFORMATION = auto()
    DEVELOPMENT = auto()
    MEDIA = auto()
    LEADERSHIP = auto()
    INDEXING = auto()
    DIVULGATION = auto()
    CONSULTING = auto()
    EVANGELISM = auto()
    GROWTH = auto()
    AUDIT = auto()
    CI = auto()
    CD = auto()

class ProjectRole(Flag): # For contributors, external and internal
    """Project roles."""
    AUTHOR = auto()
    CONTRIBUTOR = auto()
    EDITOR = auto()
    REVIEWER = auto()
    ADMINISTRATOR = auto()
    LEAD = auto()
    TESTER = auto()
    RESEARCHER = auto()
    SME = auto()
    ASSISTANT = auto()
    CONSULTANT = auto()

class User(Flag): # For internal users of the app only
    """User types for permission assignment."""
    ADMIN = auto()
    EDITOR = auto()
    AUTHOR = auto()
    REVIEWER = auto()
    CONTRIBUTOR = auto()
    GUEST = auto()

class Rights(Flag): # The permissions of a User
    NONE = 0
    MASTER = auto()
    READONLY = auto()
    READWRITE = auto()
    DELETE = auto()
    SHARE = auto()
    DEFAULT = READONLY | SHARE
    ALL = MASTER | READONLY | READWRITE | DELETE | SHARE


class FeatureType(Enum):
    """Feature types."""
    ARTIFACT = "artifact"
    LATEX = "latex"
    UI_COMPONENT = "ui_component"
    NUMBERING = "numbering"
    FORMAT = "format"
    LAYOUT = "layout"
    NAVIGATION = "navigation"
    TOC = "toc"
    INDEX = "index"
    SEARCH = "search"
    FILTER = "filter"
    SORT = "sort"
    PAGINATION = "pagination"
    THEME = "theme"
    TEMPLATE = "template"

class ReferenceType(Enum):
    """Reference types."""
    KEYREF = "keyref"
    CONREF = "conref"
    XREF = "xref"
    HREF = "href"
    TOPICREF = "topicref"
    MAPREF = "mapref"
    CITE = "cite"
    LINK = "link"
    EXTERNAL = "external"

class CitationFormat(Enum):
    """Citation formats."""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    CUSTOM = "custom"

class CitationType(Enum):
    """Types of citations."""
    BOOK = "book"
    ARTICLE = "article"
    PAPER = "paper"
    REPORT = "report"
    THESIS = "thesis"
    PATENT = "patent"
    WEBSITE = "website"
    DATASET = "dataset"
    SOFTWARE = "software"

class RuleType(Enum):
    """Rule types."""
    INSTRUCTIONS = "instructions"
    SETTINGS = "settings"
    PREFERENCES = "preferences"
    CONDITIONS = "conditions"
    RECIPE = "recipe"
    COOKBOOK = "cookbook"
    RESTRICTION = "restriction"
    COMMANDMENT = "commandment"
    AMENDMENT = "amendment"
    FILTERS = "filters"

class ValidationLevel(Enum):
    """Validation levels."""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    FULL = "full"
    CUSTOM = "custom"

class CacheStrategy(Enum):
    """Cache strategies."""
    LRU = "lru"
    FIFO = "fifo"
    LIFO = "lifo"
    TTL = "ttl"
    NONE = "none"
    DEFAULT = "default"
    CUSTOM = "custom"

class OutputFormat(Enum):
    """Output formats."""
    HTML = "html"
    PDF = "pdf"
    XML = "xml"
    JSON = "json"
    MARKDOWN = "markdown"
    PLAINTEXT = "plaintext"

class Priority(Enum):
    """Priority levels."""
    HIGHEST = "highest"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    LOWEST = "lowest"

class Environment(Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class ProcessingRole(Enum):
    """Processing roles."""
    NORMAL = "normal"
    RESOURCE_ONLY = "resource-only"
    TEMPLATE = "template"
    SYSTEM = "system"

class SecurityLevel(Enum):
    """Security levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PRIVATE = "private"

class Language(Flag):
    """Language codes."""
    EN = auto()
    ES = auto()
    FR = auto()
    DE = auto()
    IT = auto()
    PT = auto()

class GitAction(Enum):
    """Git actions"""
    ADD = "add"  # Stage changes for commit
    COMMIT = "commit"  # Commit staged changes
    PUSH = "push"  # Push local commits to a remote repository
    PULL = "pull"  # Fetch and merge changes from the remote repository
    MERGE = "merge"  # Merge branches together
    BRANCH = "branch"  # Create, list, or delete branches
    CHECKOUT = "checkout"  # Switch branches or restore working tree files
    STATUS = "status"  # Show the working tree status
    LOG = "log"  # Show commit logs
    DIFF = "diff"  # Show changes between commits, commit and working tree, etc.
    FETCH = "fetch"  # Download objects and refs from another repository
    REBASE = "rebase"  # Reapply commits on top of another base tip
    STASH = "stash"  # Stash changes in a dirty working directory
    RESET = "reset"  # Reset current HEAD to a specified state
    TAG = "tag"  # Create, list, delete or verify a tag object signed with GPG
    REMOTE = "remote"  # Manage set of tracked repositories
    INIT = "init"  # Create an empty Git repository or reinitialize an existing one
    CLONE = "clone"  # Clone a repository into a new directory
    RENAME = "rename"  # Rename a branch or file
