import os
from pathlib import Path
from lxml import etree
import logging
from typing import Optional, List, Dict

class DITAProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Get the absolute path to the app directory
        self.app_root = Path(__file__).parent.parent
        self.dita_root = self.app_root / 'dita'
        self.maps_dir = self.dita_root / 'maps'
        self.topics_dir = self.dita_root / 'topics'
        self.output_dir = self.dita_root / 'output'
        self.xsl_dir = self.dita_root / 'xsl'

        # Initialize XML parser
        self.parser = etree.XMLParser(
            recover=True,
            remove_blank_text=True,
            resolve_entities=False,
            dtd_validation=False,
            load_dtd=False,
            no_network=True
        )

        # Configure logging
        logging.basicConfig(level=logging.INFO)

        # Ensure directories exist
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.maps_dir,
            self.topics_dir / 'abstracts',
            self.topics_dir / 'acoustics',
            self.topics_dir / 'articles',
            self.topics_dir / 'audio',
            self.topics_dir / 'journals',
            self.output_dir,
            self.xsl_dir
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _extract_content_preview(self, tree) -> Optional[str]:
        """Extract a preview of the topic content"""
        try:
            # Try different content locations based on topic type
            content_paths = [
                './/body/p',
                './/conbody/p',
                './/taskbody/p',
                './/abstract/p',
                './/shortdesc'
            ]

            for path in content_paths:
                elements = tree.xpath(path)
                if elements:
                    return ' '.join(elem.text for elem in elements if elem.text)

            return None
        except Exception as e:
            self.logger.error(f"Error extracting content preview: {str(e)}")
            return None

    def _transform_list(self, list_elem) -> str:
        """Helper method to transform lists to HTML"""
        tag = etree.QName(list_elem).localname
        html = [f'<{tag} class="list-disc ml-6 mb-4">']

        for item in list_elem.findall('.//*[local-name()="li"]'):
            if item.text:
                html.append(f'<li class="mb-2">{item.text}</li>')

        html.append(f'</{tag}>')
        return '\n'.join(html)

    def transform_to_html(self, input_path: Path) -> Optional[str]:
        """Transform DITA content to HTML"""
        try:
            self.logger.info(f"Transforming {input_path} to HTML")

            # Read the DITA content
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse the content
            doc = etree.fromstring(content.encode('utf-8'), self.parser)

            # Basic HTML transformation
            html_content = ['<div class="dita-content">']

            # Add title
            title_elem = doc.find('.//*[local-name()="title"]')
            if title_elem is not None and title_elem.text:
                html_content.append(f'<h1 class="text-2xl font-bold mb-4">{title_elem.text}</h1>')

            # Add metadata if available
            prolog = doc.find('.//*[local-name()="prolog"]')
            if prolog is not None:
                html_content.append('<div class="metadata mb-4">')
                # Add authors
                authors = prolog.findall('.//*[local-name()="author"]')
                if authors:
                    html_content.append('<div class="authors">')
                    for author in authors:
                        if author.text:
                            html_content.append(f'<span class="author">{author.text}</span>')
                    html_content.append('</div>')
                html_content.append('</div>')

            # Add main content
            for element in doc.iter():
                tag = etree.QName(element).localname
                if tag == 'p' and element.text:
                    html_content.append(f'<p class="mb-4">{element.text}</p>')
                elif tag in ['ul', 'ol']:
                    html_content.append(self._transform_list(element))

            html_content.append('</div>')

            result = '\n'.join(html_content)

            self.logger.info("HTML transformation completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Transformation error: {e}")
            return f'<div class="error">Error displaying content: {str(e)}</div>'

    def list_topics(self) -> List[Dict[str, str]]:
        """List all available topics"""
        topics = []
        subdirs = ['abstracts', 'acoustics', 'articles', 'audio', 'journals']

        self.logger.info(f"Searching for topics in: {self.topics_dir}")

        for subdir in subdirs:
            subdir_path = self.topics_dir / subdir
            if subdir_path.exists():
                self.logger.info(f"Checking directory: {subdir_path}")
                for topic_file in subdir_path.glob('*.dita'):
                    try:
                        self.logger.info(f"Found file: {topic_file}")

                        # Read the file content
                        with open(topic_file, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # Parse the content
                        tree = etree.fromstring(content.encode('utf-8'), self.parser)

                        # Find the title element - simplified approach
                        title_elem = None
                        for elem in tree.iter():
                            tag_name = etree.QName(elem).localname
                            if tag_name == 'title':
                                title_elem = elem
                                break

                        # Get basic content for preview
                        content_preview = self._extract_content_preview(tree)

                        # Create topic data with fallbacks
                        topic_data = {
                            'id': topic_file.stem,
                            'title': title_elem.text if title_elem is not None else topic_file.stem,
                            'path': str(topic_file.relative_to(self.dita_root)),
                            'type': subdir,
                            'fullPath': str(topic_file),
                            'preview': content_preview or "No content available",
                            'hasContent': bool(content_preview)
                        }

                        self.logger.info(f"Adding topic: {topic_data}")
                        topics.append(topic_data)

                    except Exception as e:
                        self.logger.error(f"Error processing {topic_file}: {str(e)}")
                        # Add the topic even if there are errors
                        topics.append({
                            'id': topic_file.stem,
                            'title': topic_file.stem,
                            'path': str(topic_file.relative_to(self.dita_root)),
                            'type': subdir,
                            'fullPath': str(topic_file),
                            'preview': "Error loading content",
                            'hasContent': False,
                            'error': str(e)
                        })

        self.logger.info(f"Total topics found: {len(topics)}")
        return topics

    def get_topic_path(self, topic_id: str) -> Optional[Path]:
        """Get the full path for a topic by its ID"""
        self.logger.info(f"Looking for topic with ID: {topic_id}")

        for subdir in ['acoustics', 'articles', 'audio', 'abstracts', 'journals']:
            potential_path = self.topics_dir / subdir / f"{topic_id}.dita"
            self.logger.info(f"Checking path: {potential_path}")

            if potential_path.exists():
                self.logger.info(f"Found topic at: {potential_path}")
                return potential_path

        self.logger.error(f"No topic found with ID: {topic_id}")
        return None
