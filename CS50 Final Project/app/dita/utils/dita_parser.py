from pathlib import Path
from typing import Optional, List
from lxml import etree
import logging

class DITAParser:
    """Handles DITA XML parsing with validation and error handling."""

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

    def parse_file(self, path: Path) -> Optional[etree._Element]:
        """Parse DITA file into XML tree."""
        try:
            self.logger.debug(f"Parsing DITA file: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.parse_content(content)
        except Exception as e:
            self.logger.error(f"Error parsing DITA file {path}: {str(e)}")
            return None

    def parse_content(self, content: str) -> Optional[etree._Element]:
        """Parse DITA content string into XML tree."""
        try:
            return etree.fromstring(content.encode('utf-8'), self.parser)
        except Exception as e:
            self.logger.error(f"Error parsing DITA content: {str(e)}")
            return None

    def validate_content(self, content: str) -> List[str]:
        """Validate DITA content and return any errors."""
        errors = []
        try:
            # Try parsing with strict parser first
            strict_parser = etree.XMLParser(recover=False)
            etree.fromstring(content.encode('utf-8'), strict_parser)
        except etree.XMLSyntaxError as e:
            errors.append(f"Line {e.lineno}, Column {e.offset}: {e.msg}")
            # Get the problematic line
            lines = content.splitlines()
            if 0 <= e.lineno - 1 < len(lines):
                errors.append(f"Line content: {lines[e.lineno - 1]}")
        return errors
