"""
Metadata management system for DITA content processing.
Handles both transient and persistent metadata with event-driven updates.
"""

from .metadata_manager import MetadataManager
from .storage import MetadataStorage
from .extractor import MetadataExtractor
from ..utils.heading import HeadingMetadata  # Fixed relative import

# Export main interface
__all__ = ['MetadataManager']
