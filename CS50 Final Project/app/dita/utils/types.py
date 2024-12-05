# app/dita/utils/types.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
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
    MARKDOWN = "markdown"
    LATEX = "latex"
    ARTIFACT = "artifact"

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
    type: ElementType
    content: str
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
class TrackedElement:
    """Element with tracking information"""
    id: str
    type: ElementType
    content: str
    metadata: Dict[str, Any]
    state: ProcessingState = ProcessingState.PENDING

@dataclass
class ProcessedContent:
    """Processed element content"""
    html: str
    element_id: str
    heading_level: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

# Context tracking types
@dataclass
class MapContext:
    """Tracking context for map processing"""
    map_id: str
    map_path: Path
    metadata: Dict[str, Any]
    topic_order: List[str]  # List of topic IDs in order
    features: Dict[str, bool]  # Map-level features/conditions

@dataclass
class TopicContext:
    """Tracking context for topic processing"""
    topic_id: str
    topic_path: Path
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
class ProcessingError:
    """Processing error information"""
    error_type: str
    message: str
    context: Union[str, Path]
    element_id: Optional[str] = None
    stacktrace: Optional[str] = None

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
