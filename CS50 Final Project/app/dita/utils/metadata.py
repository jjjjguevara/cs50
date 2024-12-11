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
    DITAMAP = "ditamap"
    MARKDOWN = "markdown"

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

    def extract_metadata(self,
                         file_path: PathLike,
                         content_id: str,
                         heading_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract metadata from any supported content type.
        """
        try:
            # Convert file_path to Path object if it's not already
            if isinstance(file_path, str):
                file_path = Path(file_path)

            # Ensure file_path is a valid Path
            assert isinstance(file_path, Path), f"Expected Path, got {type(file_path).__name__}"

            # Determine content type
            content_type = self._determine_content_type(file_path)

            if content_type == ElementType.MARKDOWN:
                return self._extract_markdown_metadata(file_path, content_id, heading_id)
            elif content_type in (ElementType.DITA, ElementType.DITAMAP):
                return self._extract_dita_metadata(file_path, content_id, heading_id)
            else:
                self.logger.warning(f"Unsupported content type for {file_path}")
                return {}

        except Exception as e:
            self.logger.error(f"Error extracting metadata from {file_path}: {str(e)}")
            return {}


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


    def _determine_content_type(self, file_path: Path) -> ContentType:
        """Determine content type from file extension"""
        suffix = file_path.suffix.lower()
        if suffix == '.md':
            return ContentType.MARKDOWN
        elif suffix == '.dita':
            return ContentType.DITA
        elif suffix == '.ditamap':
            return ContentType.DITAMAP
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _extract_markdown_metadata(self,
                                 file_path: Path,
                                 content_id: str,
                                 heading_id: Optional[str]) -> Dict[str, Any]:
        """Extract metadata from markdown frontmatter"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)

            metadata = post.metadata
            # Add standard fields
            metadata.update({
                'content_type': ContentType.MARKDOWN.value,
                'content_id': content_id,
                'heading_id': heading_id,
                'source_file': str(file_path),
                'processed_at': datetime.now().isoformat()
            })

            # Handle special flags
            metadata['has_bibliography'] = metadata.get('bibliography', False)

            return metadata

        except Exception as e:
            self.logger.error(f"Error processing markdown metadata: {str(e)}")
            return {}

    def _extract_dita_metadata(self,
                             file_path: Path,
                             content_id: str,
                             heading_id: Optional[str]) -> Dict[str, Any]:
        """Extract metadata from DITA XML"""
        try:
            tree = etree.parse(str(file_path), self.parser)
            metadata = {
                'content_type': (ContentType.DITAMAP.value
                               if file_path.suffix == '.ditamap'
                               else ContentType.DITA.value),
                'content_id': content_id,
                'heading_id': heading_id,
                'source_file': str(file_path),
                'processed_at': datetime.now().isoformat()
            }

            # Process othermeta elements
            for othermeta in tree.xpath('.//othermeta'):
                name = othermeta.get('name')
                content = othermeta.get('content')
                if name and content:
                    # Convert specific flags to booleans
                    if name in ['index-numbers', 'append-toc']:
                        metadata[name] = content.lower() == 'true'
                    else:
                        metadata[name] = content

            return metadata

        except Exception as e:
            self.logger.error(f"Error processing DITA metadata: {str(e)}")
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
