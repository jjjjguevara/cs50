from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor
from markdown.core import Markdown
import re
import yaml
import logging
from typing import List, Dict, Any, Optional

class YAMLMetadataPreprocessor(Preprocessor):
    """Process YAML front matter in markdown files"""

    YAML_REGEX = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)

    def __init__(self, md: Markdown, extension: 'YAMLMetadataExtension'):
        super().__init__(md)
        self.logger = logging.getLogger(__name__)
        self._extension = extension

    def run(self, lines: List[str]) -> List[str]:
        """Extract YAML front matter and store in markdown metadata"""
        try:
            # Join lines to process YAML block
            content = '\n'.join(lines)
            yaml_match = self.YAML_REGEX.match(content)

            if yaml_match:
                # Extract and parse YAML
                yaml_content = yaml_match.group(1)
                metadata = yaml.safe_load(yaml_content) or {}

                # Store metadata in extension
                self._extension.set_metadata(metadata)

                # Return content without YAML front matter
                return self.YAML_REGEX.sub('', content).split('\n')

            return lines
        except Exception as e:
            self.logger.error(f"Error processing YAML front matter: {e}")
            return lines

class YAMLMetadataExtension(Extension):
    """Extension for processing YAML front matter"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._metadata: Dict[str, Any] = {}

    def extendMarkdown(self, md: Markdown) -> None:
        self._metadata = {}
        md.preprocessors.register(
            YAMLMetadataPreprocessor(md, self),
            'yaml_metadata',
            priority=1
        )

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """Set metadata from preprocessor"""
        self._metadata = metadata

    def get_metadata(self) -> Dict[str, Any]:
        """Get the current metadata"""
        return self._metadata
