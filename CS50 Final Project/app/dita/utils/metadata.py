from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
import yaml
from lxml import etree
from lxml.etree import _Element
import frontmatter
import json
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

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
        self.parser = etree.XMLParser(
            recover=True,
            remove_blank_text=True,
            resolve_entities=False,
            dtd_validation=False,
            load_dtd=False,
            no_network=True
        )

    def extract_metadata(self,
                        file_path: Path,
                        content_id: str,
                        heading_id: Optional[str] = None) -> Dict[str, Any]:
        """Extract metadata from any supported content type"""
        try:
            content_type = self._determine_content_type(file_path)

            if content_type == ContentType.MARKDOWN:
                return self._extract_markdown_metadata(file_path, content_id, heading_id)
            elif content_type in (ContentType.DITA, ContentType.DITAMAP):
                return self._extract_dita_metadata(file_path, content_id, heading_id)
            else:
                self.logger.warning(f"Unsupported content type for {file_path}")
                return {}

        except Exception as e:
            self.logger.error(f"Error extracting metadata from {file_path}: {str(e)}")
            return {}

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

            # Process prolog metadata
            prolog = tree.find('.//prolog')
            if prolog is not None:
                # Authors
                authors = [author.text for author in prolog.findall('.//author') if author.text]
                if authors:
                    metadata['authors'] = authors

                # Institution
                institution = prolog.find('.//institution')
                if institution is not None and institution.text:
                    metadata['institution'] = institution.text

                # Categories
                categories = [cat.text for cat in prolog.findall('.//category') if cat.text]
                if categories:
                    metadata['categories'] = categories

                # Keywords
                keywords = [kw.text for kw in prolog.findall('.//keyword') if kw.text]
                if keywords:
                    metadata['keywords'] = keywords

                # Other metadata
                for othermeta in prolog.findall('.//othermeta'):
                    name = othermeta.get('name')
                    content = othermeta.get('content')
                    if name and content:
                        metadata[name] = content

                # Special flags - add here, inside the prolog check
                metadata['is_journal_entry'] = self._check_metadata_flag(prolog, 'journal-entry')

            # Abstract
            abstract = tree.find('.//abstract/shortdesc')
            if abstract is not None and abstract.text:
                metadata['abstract'] = abstract.text

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
