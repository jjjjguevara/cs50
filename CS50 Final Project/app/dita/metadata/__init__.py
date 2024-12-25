"""
Metadata management system for DITA content processing.
Handles both transient and persistent metadata with event-driven updates.
"""

from .metadata_manager import MetadataManager
from .storage import MetadataStorage
from .extractor import MetadataExtractor
from dita.utils.heading import HeadingMetadata

# Export main interface
__all__ = ['MetadataManager']
