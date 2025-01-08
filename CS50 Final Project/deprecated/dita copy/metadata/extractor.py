"""Content metadata extraction for DITA processing."""
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from pathlib import Path
import yaml
from datetime import datetime
from lxml import etree
import re
import logging
if TYPE_CHECKING:
    from ..config.config_manager import ConfigManager

from ..models.types import (
    ContentElement,
    ProcessingContext,
    ProcessingPhase,
    ElementType,
    ContentType,
    YAMLFrontmatter,
    ValidationResult
)
from ..config.config_manager import ConfigManager
from ..utils.cache import ContentCache, CacheEntryType
from ..utils.logger import DITALogger

class MetadataExtractor:
    """Extracts and processes metadata from different content types."""

    def __init__(
        self,
        cache: 'ContentCache',
        config_manager: 'ConfigManager',
        logger: Optional['DITALogger'] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.cache = cache
        self.config_manager = config_manager

        # Initialize extractors
        self._extractors = {
            ElementType.DITA: self._extract_dita_metadata,
            ElementType.MARKDOWN: self._extract_markdown_metadata,
            ElementType.MAP: self._extract_map_metadata
        }

    def extract_metadata(
        self,
        element: ContentElement,
        context: ProcessingContext,
        phase: ProcessingPhase
    ) -> Dict[str, Any]:
        """Extract metadata based on element type and context."""
        try:
            # Check cache first
            cache_key = f"metadata_{element.id}_{phase.value}"
            if cached := self.cache.get(
                key=cache_key,
                entry_type=CacheEntryType.METADATA
            ):
                return cached

            # Get extractor
            extractor = self._extractors.get(element.type)
            if not extractor:
                return {}

            # Extract base metadata
            metadata = extractor(element, context)

            # Add standard fields
            metadata.update({
                'element_id': element.id,
                'element_type': element.type.value,
                'processing_phase': phase.value,
                'context_path': context.navigation.path,
                'extracted_at': datetime.now().isoformat()
            })

            # Get feature flags from config
            features = self.config_manager.get_feature_state(element.type.value)
            if features:
                metadata['feature_flags'] = features

            # Cache result
            self.cache.set(
                key=cache_key,
                data=metadata,
                entry_type=CacheEntryType.METADATA,
                element_type=element.type,
                phase=phase
            )

            return metadata

        except Exception as e:
            self.logger.error(f"Error extracting metadata: {str(e)}")
            return {}

    def _extract_dita_metadata(
        self,
        element: ContentElement,
        context: ProcessingContext
    ) -> Dict[str, Any]:
        """Extract metadata from DITA content."""
        try:
            tree = etree.parse(str(element.path))
            metadata = {}

            # Extract prolog metadata
            if prolog := tree.find(".//prolog"):
                metadata.update(self._process_prolog(prolog))

            # Extract attributes from topic or map
            root = tree.getroot()
            for attr, value in root.attrib.items():
                metadata[attr] = value

            # Extract specific elements based on type
            if element.type == ElementType.DITA:
                metadata.update(self._extract_topic_metadata(root))
            elif element.type == ElementType.MAP:
                metadata.update(self._extract_map_metadata(root))

            return metadata

        except Exception as e:
            self.logger.error(f"Error extracting DITA metadata: {str(e)}")
            return {}

    def _process_prolog(self, prolog: etree._Element) -> Dict[str, Any]:
        """Process DITA prolog metadata."""
        metadata = {}

        # Author metadata
        if author := prolog.find(".//author"):
            metadata['author'] = author.text

        # Extract all metadata fields
        for meta in prolog.findall(".//metadata//*"):
            if meta.tag not in ['author']:  # Skip already processed
                if meta.text and meta.text.strip():
                    metadata[meta.tag] = meta.text.strip()

        # Audience metadata
        for audience in prolog.findall(".//audience"):
            if 'audience' not in metadata:
                metadata['audience'] = []
            metadata['audience'].append({
                'type': audience.get('type'),
                'job': audience.get('job'),
                'experience': audience.get('experience')
            })

        return metadata

    def _extract_markdown_metadata(
        self,
        element: ContentElement,
        context: ProcessingContext
    ) -> Dict[str, Any]:
        """Extract metadata from Markdown content."""
        try:
            content = element.path.read_text(encoding='utf-8')

            # Extract YAML frontmatter
            if content.startswith('---'):
                try:
                    end = content.index('---', 3)
                    yaml_content = content[3:end]
                    frontmatter = yaml.safe_load(yaml_content) or {}

                    return {
                        'title': frontmatter.get('title'),
                        'author': frontmatter.get('author'),
                        'date': frontmatter.get('date'),
                        'tags': frontmatter.get('tags', []),
                        'categories': frontmatter.get('categories', []),
                        'feature_flags': frontmatter.get('features', {}),
                        'custom_metadata': {
                            k: v for k, v in frontmatter.items()
                            if k not in {'title', 'author', 'date', 'tags',
                                       'categories', 'features'}
                        }
                    }
                except yaml.YAMLError:
                    self.logger.warning("Invalid YAML frontmatter")
                    return {}

            return {}

        except Exception as e:
            self.logger.error(f"Error extracting Markdown metadata: {str(e)}")
            return {}

    def _extract_map_metadata(self, root: etree._Element) -> Dict[str, Any]:
        """Extract metadata specific to maps."""
        metadata = {
            'topics': [],
            'keydefs': [],
            'reltables': []
        }

        # Extract topicref information
        for topicref in root.findall(".//topicref"):
            if href := topicref.get('href'):
                metadata['topics'].append({
                    'href': href,
                    'type': topicref.get('type', 'topic'),
                    'format': topicref.get('format'),
                    'scope': topicref.get('scope', 'local'),
                    'processing-role': topicref.get('processing-role', 'normal')
                })

        # Extract keydef information
        for keydef in root.findall(".//keydef"):
            if keys := keydef.get('keys'):
                metadata['keydefs'].append({
                    'keys': keys,
                    'href': keydef.get('href'),
                    'scope': keydef.get('scope', 'local')
                })

        # Extract reltable information
        for reltable in root.findall(".//reltable"):
            metadata['reltables'].append({
                'id': reltable.get('id'),
                'type': reltable.get('type')
            })

        return metadata

    def _extract_topic_metadata(self, root: etree._Element) -> Dict[str, Any]:
            """Extract metadata specific to topics."""
            metadata = {}

            # Extract title
            if title := root.find(".//title"):
                metadata['title'] = title.text.strip() if title.text else ""

            # Extract shortdesc
            if shortdesc := root.find(".//shortdesc"):
                metadata['shortdesc'] = shortdesc.text.strip() if shortdesc.text else ""

            # Extract abstract
            if abstract := root.find(".//abstract"):
                metadata['abstract'] = "".join(abstract.itertext()).strip()

            # Extract topic attributes
            metadata.update({
                'topic_type': root.tag,
                'id': root.get('id', ''),
                'class': root.get('class', ''),
                'domains': root.get('domains', '')
            })

            return metadata
