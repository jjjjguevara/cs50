from markdown.extensions import Extension
from markdown.treeprocessors import Treeprocessor
from xml.etree import ElementTree as etree
import logging

class DITATreeprocessor(Treeprocessor):
    """Process the markdown AST to add DITA semantics"""

    def __init__(self, md):
        super().__init__(md)
        self.logger = logging.getLogger(__name__)

    def run(self, root):
        """Transform elements to match DITA semantics"""
        try:
            self.process_headers(root)
            self.process_sections(root)
            self.process_lists(root)
            return root
        except Exception as e:
            self.logger.error(f"Error processing DITA semantics: {e}")
            return root

    def process_headers(self, element):
        """Process headers to match DITA structure"""
        for header in element.findall(".//h1"):
            header.set('class', 'dita-title')
        for header in element.findall(".//h2"):
            header.set('class', 'dita-section-title')

    def process_sections(self, element):
        """Process sections to match DITA structure"""
        for section in element.findall(".//section"):
            section.set('class', 'dita-section')

    def process_lists(self, element):
        """Process lists to match DITA structure"""
        for ul in element.findall(".//ul"):
            ul.set('class', 'dita-ul')
        for ol in element.findall(".//ol"):
            ol.set('class', 'dita-ol')

class DITAExtension(Extension):
    """Extension for adding DITA semantics to markdown output"""

    def extendMarkdown(self, md):
        md.treeprocessors.register(
            DITATreeprocessor(md),
            'dita',
            priority=100
        )
