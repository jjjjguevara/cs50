# app/dita/artifacts/parser.py
from pathlib import Path
from typing import List, Dict
import logging
from lxml import etree

# Initialize module-level logger
logger = logging.getLogger(__name__)

class ArtifactParser:
    def __init__(self, dita_root: Path):
        self.dita_root = dita_root
        # Initialize instance logger
        self.logger = logging.getLogger(__name__)

        self.parser = etree.XMLParser(
            recover=True,
            remove_blank_text=True,
            resolve_entities=False,
            dtd_validation=False,
            load_dtd=False,
            no_network=True
        )

    def parse_artifact_references(self, ditamap_path: Path) -> List[Dict]:
        """Parse artifact references from DITAMAP"""
        try:
            self.logger.info(f"Parsing artifacts from: {ditamap_path}")

            # Read and validate XML first
            with open(ditamap_path, 'r', encoding='utf-8') as f:
                content = f.read()

            try:
                # First try to parse without recovery to get validation errors
                strict_parser = etree.XMLParser(recover=False)
                etree.fromstring(content.encode('utf-8'), strict_parser)
            except etree.XMLSyntaxError as e:
                self.logger.error(f"XML Syntax Error in {ditamap_path.name}:")
                self.logger.error(f"Line {e.lineno}, Column {e.offset}")
                self.logger.error(f"Error: {e.msg}")
                # Print the problematic line
                lines = content.splitlines()
                if 0 <= e.lineno - 1 < len(lines):
                    self.logger.error(f"Problematic line: {lines[e.lineno - 1]}")
                    self.logger.error(" " * (e.offset - 1) + "^")
                return []

            # If validation passes, parse with recovery for processing
            tree = etree.parse(ditamap_path, self.parser)
            artifacts = []

            # Find all artifact data elements
            data_elements = tree.xpath('//data[@name="artifacts"]')
            self.logger.debug(f"Found {len(data_elements)} artifact containers")

            for container in data_elements:
                for data in container.xpath('.//data'):
                    name = data.get('name')
                    href = data.get('href')
                    target = data.get('target-heading')

                    if href:  # Only process if href is present
                        artifact = {
                            'name': name,
                            'href': href,
                            'target': target
                        }
                        self.logger.info(f"Found artifact: {artifact}")
                        artifacts.append(artifact)

            if not artifacts:
                self.logger.warning(f"No artifacts found in {ditamap_path}")
                # Dump XML for debugging
                self.logger.debug(f"DITAMAP content:\n{etree.tostring(tree, pretty_print=True).decode()}")

            return artifacts

        except Exception as e:
            self.logger.error(f"Error parsing artifacts: {e}", exc_info=True)
            return []

    def validate_ditamap(self, content: str) -> List[str]:
        """Validate DITAMAP content and return any errors"""
        errors = []
        try:
            parser = etree.XMLParser(recover=False)
            etree.fromstring(content.encode('utf-8'), parser)
        except etree.XMLSyntaxError as e:
            errors.append(f"Line {e.lineno}, Column {e.offset}: {e.msg}")
            # Get the problematic line
            lines = content.splitlines()
            if 0 <= e.lineno - 1 < len(lines):
                errors.append(f"Line content: {lines[e.lineno - 1]}")
        return errors

    def _log_xml_error(self, error: etree.XMLSyntaxError, content: str) -> None:
        """Helper method to log XML errors with context"""
        self.logger.error(f"XML Error: {error.msg}")
        self.logger.error(f"Line {error.lineno}, Column {error.offset}")

        lines = content.splitlines()
        if 0 <= error.lineno - 1 < len(lines):
            self.logger.error("Context:")
            # Show a few lines before and after the error
            start = max(0, error.lineno - 3)
            end = min(len(lines), error.lineno + 2)
            for i in range(start, end):
                prefix = "-> " if i == error.lineno - 1 else "   "
                self.logger.error(f"{prefix}{i + 1}: {lines[i]}")
                if i == error.lineno - 1:
                    self.logger.error("   " + " " * (error.offset + 2) + "^")
