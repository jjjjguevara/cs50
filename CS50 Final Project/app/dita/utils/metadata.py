from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import yaml
import sqlite3
from lxml import etree
from lxml.etree import _Element
import frontmatter
import yaml
import json
import re
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from app.dita.models.types import PathLike, ElementType
# Global config
from app_config import DITAConfig

class ContentType(Enum):
    DITA = "dita"
    MARKDOWN = "markdown"
    MAP = "map"
    TOPIC = "topic"
    UNKNOWN = "unknown"

@dataclass
class MetadataField:
    name: str
    value: Any
    content_type: ContentType
    source_id: str
    heading_id: Optional[str] = None
    timestamp: datetime = datetime.now()

class MetadataHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._conn = self._init_db_connection()
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        self.parser = etree.XMLParser(
            recover=True,
            remove_blank_text=True,
            resolve_entities=False,
            dtd_validation=False,
            load_dtd=False,
            no_network=True
        )

    def _init_db_connection(self) -> sqlite3.Connection:
            """Initialize SQLite connection with proper settings."""
            try:
                db_path = Path('metadata.db')  # You might want to make this configurable
                conn = sqlite3.connect(str(db_path))
                conn.row_factory = sqlite3.Row
                return conn
            except Exception as e:
                self.logger.error(f"Failed to initialize database connection: {str(e)}")
                raise

    def configure(self, config: DITAConfig) -> None:
        """Configure metadata handler."""
        try:
            if hasattr(self, '_conn'):
                self._conn.close()

            # Use config's metadata DB path
            self._conn = sqlite3.connect(str(config.metadata_db_path))
            self._conn.row_factory = sqlite3.Row

            self.logger.debug("Configuring metadata handler")
            # Add any configuration-specific settings here
            self.logger.debug("Metadata handler configuration completed")
        except Exception as e:
            self.logger.error(f"Metadata handler configuration failed: {str(e)}")
            raise

    def extract_metadata(
            self,
            file_path: Path,
            content_id: str,
            heading_id: Optional[str] = None,
            map_metadata: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """
            Extract metadata for a given file, handling both DITA and Markdown content.

            Args:
                file_path: Path to the content file.
                content_id: Unique identifier for the content.
                heading_id: Optional heading for scoped metadata.
                map_metadata: Parent map metadata for context.

            Returns:
                Dict containing extracted metadata.
            """
            try:
                content_type = self._determine_content_type(file_path)

                if content_type == ContentType.MARKDOWN:
                    metadata = self._extract_markdown_metadata(file_path, content_id, heading_id, map_metadata)
                elif content_type in {ContentType.DITA, ContentType.MAP}:
                    metadata = self._extract_dita_metadata(file_path, content_id, heading_id, map_metadata)
                else:
                    self.logger.warning(f"Unsupported content type for {file_path}")
                    return {}

                # Enrich metadata with feature flags and context
                metadata = self._enrich_metadata(metadata, map_metadata)

                return metadata

            except Exception as e:
                self.logger.error(f"Error extracting metadata from {file_path}: {str(e)}")
                return {}

    def _enrich_metadata(self, metadata: Dict[str, Any], map_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enrich metadata with default values and feature flags.

        Args:
            metadata: Extracted metadata.
            map_metadata: Parent map metadata for context.

        Returns:
            Enriched metadata.
        """
        enriched_metadata = metadata.copy()

        # Add feature flags
        enriched_metadata.setdefault("feature_flags", {
            "enable_toc": True,
            "enable_cross_refs": True,
            "enable_heading_numbering": True
        })

        # Add relational metadata defaults
        enriched_metadata.setdefault("prerequisites", [])
        enriched_metadata.setdefault("related_topics", [])

        # Merge parent map metadata if provided
        if map_metadata:
            enriched_metadata["context"] = map_metadata.get("context", {})
            enriched_metadata["feature_flags"].update(map_metadata.get("feature_flags", {}))

        # Validate and finalize
        self._validate_metadata(enriched_metadata)
        return enriched_metadata

    def _validate_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Validate metadata fields for consistency and completeness.

        Args:
            metadata: Metadata to validate.

        Raises:
            ValueError if validation fails.
        """
        required_fields = ["title", "feature_flags", "context"]
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Missing required metadata field: {field}")

        # Additional validation for feature flags
        if not isinstance(metadata.get("feature_flags"), dict):
            raise ValueError("Feature flags must be a dictionary.")

    def store_metadata(
        self, content_id: str, metadata: Dict[str, Any]
    ) -> None:
        """
        Store metadata in the database.

        Args:
            content_id: Unique identifier for the content.
            metadata: Metadata to store.
        """
        try:
            with self._conn as conn:
                conn.execute("""
                    INSERT INTO metadata_store (content_id, metadata)
                    VALUES (?, ?)
                    ON CONFLICT(content_id) DO UPDATE SET
                    metadata = excluded.metadata
                """, (content_id, json.dumps(metadata)))

        except Exception as e:
            self.logger.error(f"Error storing metadata for {content_id}: {str(e)}")
            raise

    def _determine_content_type(self, file_path: Path) -> ContentType:
        """
        Determine content type based on file extension.

        Args:
            file_path: Path to the content file.

        Returns:
            ContentType enum.
        """
        ext = file_path.suffix.lower()
        if ext in {".dita", ".ditamap"}:
            return ContentType.DITA if ext == ".dita" else ContentType.MAP
        elif ext == ".md":
            return ContentType.MARKDOWN
        else:
            return ContentType.UNKNOWN


    def extract_yaml_metadata(self, content: str) -> Tuple[Dict[str, Any], str]:
        """
        Extract YAML metadata from the beginning of the content if present.

        Args:
            content: The content string to extract metadata from.

        Returns:
            A tuple containing the extracted metadata (dict) and the remaining content.
        """
        match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
        if match:
            try:
                metadata = yaml.safe_load(match.group(1))  # Parse YAML block
                remaining_content = content[len(match.group(0)):]  # Strip YAML block
                return metadata, remaining_content
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML metadata: {e}")
        return {}, content

    def add_index_entry(
            self,
            topic_id: str,
            term: str,
            entry_type: str,
            target_id: Optional[str] = None
        ) -> None:
            """Add an index entry."""
            try:
                cur = self._conn.cursor()
                cur.execute("""
                    INSERT INTO index_entries (topic_id, term, type, target_id)
                    VALUES (?, ?, ?, ?)
                """, (topic_id, term, entry_type, target_id))
                self._conn.commit()

            except Exception as e:
                self.logger.error(f"Error adding index entry: {str(e)}")

    def add_conditional_attribute(
        self,
        name: str,
        attr_type: str,
        scope: str,
        description: Optional[str] = None
    ) -> None:
        """Add a conditional processing attribute."""
        try:
            cur = self._conn.cursor()
            cur.execute("""
                INSERT INTO conditional_attributes
                (name, attribute_type, scope, description)
                VALUES (?, ?, ?, ?)
            """, (name, attr_type, scope, description))
            self._conn.commit()

        except Exception as e:
            self.logger.error(f"Error adding conditional attribute: {str(e)}")


    def _extract_dita_metadata(
        self,
        file_path: Path,
        content_id: str,
        heading_id: Optional[str],
        map_metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        try:
            tree = etree.parse(str(file_path), self.parser)
            metadata = {
                'content_type': (ContentType.MAP.value
                               if file_path.suffix == '.ditamap'
                               else ContentType.DITA.value),
                'content_id': content_id,
                'heading_id': heading_id,
                'source_file': str(file_path),
                'processed_at': datetime.now().isoformat()
            }

            for othermeta in tree.xpath('.//othermeta'):
                name = othermeta.get('name')
                content = othermeta.get('content')
                if name and content:
                    if name in ['index-numbers', 'append-toc']:
                        metadata[name] = content.lower() == 'true'
                    else:
                        metadata[name] = content

            metadata.update(map_metadata or {})

            return metadata

        except Exception as e:
            self.logger.error(f"Error processing DITA metadata: {str(e)}")
            return {}

    def _extract_markdown_metadata(
        self,
        file_path: Path,
        content_id: str,
        heading_id: Optional[str],
        map_metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)

            metadata = post.metadata
            metadata.update({
                'content_type': ContentType.MARKDOWN.value,
                'content_id': content_id,
                'heading_id': heading_id,
                'source_file': str(file_path),
                'processed_at': datetime.now().isoformat()
            })

            metadata['has_bibliography'] = metadata.get('bibliography', False)

            metadata.update(map_metadata or {})

            return metadata

        except Exception as e:
            self.logger.error(f"Error processing markdown metadata: {str(e)}")
            return {}


    def _check_metadata_flag(self, prolog: _Element, flag_name: str) -> bool:
        """Check if a metadata flag is set to true"""
        for meta in prolog.findall('.//othermeta'):
            name = meta.get('name')
            content = meta.get('content')
            if name == flag_name and isinstance(content, str):
                return content.lower() == 'true'
        return False

    def prepare_for_database(self, metadata: Dict[str, Any]) -> List[MetadataField]:
        """Convert metadata dict to database-ready format"""
        fields = []
        content_type = ContentType(metadata.pop('content_type'))
        content_id = metadata.pop('content_id')
        heading_id = metadata.pop('heading_id', None)

        for name, value in metadata.items():
            # Convert complex values to JSON strings
            if isinstance(value, (list, dict)):
                value = json.dumps(value)

            fields.append(MetadataField(
                name=name,
                value=value,
                content_type=content_type,
                source_id=content_id,
                heading_id=heading_id
            ))

        return fields

    def get_toggleable_features(self, metadata: Dict[str, Any]) -> Dict[str, bool]:
        """Extract toggleable features from metadata"""
        return {
            'show_journal_table': metadata.get('is_journal_entry', False),
            'show_bibliography': metadata.get('has_bibliography', False),
            'show_abstract': 'abstract' in metadata,
            # Additional toggleable features go here
        }

    def cleanup(self) -> None:
            """Clean up metadata handler resources and state."""
            try:
                if hasattr(self, '_conn'):
                    self._conn.close()

                self.logger.debug("Starting metadata handler cleanup")

                # Clear cached metadata
                self._metadata_cache.clear()

                self.logger.debug("Metadata handler cleanup completed")

            except Exception as e:
                self.logger.error(f"Metadata handler cleanup failed: {str(e)}")
                raise
