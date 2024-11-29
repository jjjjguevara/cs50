from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union, cast, Callable
from pathlib import Path
from lxml import etree
import logging
import frontmatter
from bs4 import BeautifulSoup
import markdown
from markdown.extensions import fenced_code, tables, meta

# Type aliases and generics
HTMLString = str
HTMLElements = List[str]
XMLElement = Any
MarkdownContent = TypeVar('MarkdownContent', bound=Dict[str, Any])

class DITAProcessor:
    # Class variable type annotations
    logger: logging.Logger
    app_root: Path
    dita_root: Path
    maps_dir: Path
    topics_dir: Path
    output_dir: Path
    xsl_dir: Path
    parser: etree.XMLParser

    def __init__(self) -> None:
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

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

        self.md = markdown.Markdown(extensions=[
            'fenced_code',
            'tables',
            'meta',
            'attr_list'
        ])

        # Ensure directories exist
        self._create_directories()

    def create_topic(self, title: str, content: str, topic_type: str = "concept") -> Optional[Path]:
        """Create a new DITA topic file"""
        try:
            # Create topic ID from title (sanitize the title for filename)
            topic_id = "".join(c for c in title.lower() if c.isalnum() or c in ('-', '_')).replace(' ', '-')

            # Determine appropriate subdirectory based on topic type or content
            subdir = 'articles'  # default
            if any(keyword in topic_id for keyword in ['acoustics', 'room', 'sound']):
                subdir = 'acoustics'
            elif any(keyword in topic_id for keyword in ['audio', 'microphone', 'recording']):
                subdir = 'audio'
            elif topic_type == 'abstract':
                subdir = 'abstracts'
            elif topic_type == 'journal':
                subdir = 'journals'

            # Construct full topic path
            topic_dir = self.topics_dir / subdir
            topic_path = topic_dir / f"{topic_id}.dita"

            self.logger.info(f"Creating topic in directory: {topic_dir}")

            # Ensure directory exists
            topic_dir.mkdir(parents=True, exist_ok=True)

            # Create DITA content with proper structure based on topic type
            if topic_type == "concept":
                topic_content = f"""<?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE concept PUBLIC "-//OASIS//DTD DITA Concept//EN" "concept.dtd">
    <concept id="{topic_id}">
        <title>{title}</title>
        <conbody>
            <p>{content}</p>
        </conbody>
    </concept>"""
            elif topic_type == "task":
                topic_content = f"""<?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE task PUBLIC "-//OASIS//DTD DITA Task//EN" "task.dtd">
    <task id="{topic_id}">
        <title>{title}</title>
        <taskbody>
            <context>
                <p>{content}</p>
            </context>
        </taskbody>
    </task>"""
            elif topic_type == "abstract":
                topic_content = f"""<?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE topic PUBLIC "-//OASIS//DTD DITA Topic//EN" "topic.dtd">
    <topic id="{topic_id}">
        <title>{title}</title>
        <abstract>
            <shortdesc>{content}</shortdesc>
        </abstract>
    </topic>"""
            elif topic_type == "journal":
                topic_content = f"""<?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE topic PUBLIC "-//OASIS//DTD DITA Topic//EN" "topic.dtd">
    <topic id="{topic_id}">
        <title>{title}</title>
        <prolog>
            <metadata>
                <keywords>
                    <keyword>journal</keyword>
                </keywords>
            </metadata>
        </prolog>
        <body>
            <p>{content}</p>
        </body>
    </topic>"""
            else:  # default to topic type
                topic_content = f"""<?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE topic PUBLIC "-//OASIS//DTD DITA Topic//EN" "topic.dtd">
    <topic id="{topic_id}">
        <title>{title}</title>
        <body>
            <p>{content}</p>
        </body>
    </topic>"""

            # Write content to file
            with open(topic_path, 'w', encoding='utf-8') as f:
                f.write(topic_content)

            self.logger.info(f"Created topic at: {topic_path}")
            return topic_path

        except Exception as e:
            self.logger.error(f"Error creating topic: {e}")
            return None


    def _create_directories(self) -> None:
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

    def _parse_markdown(self, content: str) -> Tuple[Dict[str, Any], str]:
        """Parse Markdown content with YAML front matter"""
        try:
            # Parse front matter and content
            post = frontmatter.loads(content)

            # Convert markdown to HTML
            html_content = self.md.convert(post.content)

            return post.metadata, html_content
        except Exception as e:
            self.logger.error(f"Error parsing markdown: {e}")
            return {}, ""

    def _transform_list(self, list_elem: XMLElement) -> str:
        """Helper method to transform lists to HTML"""
        tag = etree.QName(list_elem).localname
        html = [f'<{tag} class="list-disc ml-6 mb-4">']

        for item in list_elem.iter():
            if etree.QName(item).localname == 'li' and item.text:
                html.append(f'<li class="mb-2">{item.text}</li>')

        html.append(f'</{tag}>')
        return '\n'.join(html)

    def transform_to_html(self, input_path: Path) -> HTMLString:
        """Transform DITA content to HTML"""
        try:
            # Handle different file types
            if input_path.suffix == '.md':
                with open(input_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                metadata, html_content = self._parse_markdown(content)
                return self._generate_markdown_html(metadata, html_content)
            elif input_path.suffix == '.ditamap':  # Add this condition
                return self._transform_map_to_html(input_path)
            else:
                self.logger.info(f"Transforming {input_path} to HTML")
                doc = self._parse_dita_file(input_path)
                return self._generate_html_content(doc, input_path)
        except Exception as e:
            return self._create_error_html(e, input_path)

    def _transform_map_to_html(self, map_path: Path) -> HTMLString:
        """Transform DITA map and its referenced topics to HTML with numbered headings"""
        try:
            with open(map_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = etree.fromstring(content.encode('utf-8'), self.parser)
            html_content = ['<div class="map-content">']

            # Handle main title (no numbering)
            title_elem = tree.find(".//title")
            if title_elem is not None and title_elem.text:
                html_content.append(f'<h1 class="content-title">{title_elem.text}</h1>')

            # Initialize section counters
            section_numbers = {
                'h1': 0,
                'h2': 0,
                'current_h1': None
            }

            # First pass: collect all topics and their headings
            all_topics = []
            for topicgroup in tree.findall(".//topicgroup"):
                for topicref in topicgroup.findall(".//topicref"):
                    href = topicref.get('href')
                    if href:
                        topic_path = self._resolve_topic_path(map_path, href)
                        if topic_path and topic_path.exists():
                            all_topics.append(topic_path)

            # Second pass: process topics with consistent numbering
            for topic_path in all_topics:
                html_content.append('<div class="content-section">')
                topic_content = self._transform_topic_with_numbering(topic_path, section_numbers)
                html_content.append(topic_content)
                html_content.append('</div>')

            html_content.append('</div>')
            return '\n'.join(html_content)

        except Exception as e:
            self.logger.error(f"Error transforming map to HTML: {str(e)}")
            return self._create_error_html(e, map_path)

    def _transform_topic_with_numbering(self, topic_path: Path, section_numbers: Dict[str, Any]) -> HTMLString:
        """Transform a topic to HTML with numbered headings"""
        try:
            if topic_path.suffix == '.md':
                with open(topic_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                metadata, html_content = self._parse_markdown(content)

                # Create soup object for consistent handling
                soup = BeautifulSoup(html_content, 'html.parser')

                # Wrap content in markdown-content div if not already wrapped
                if not soup.find('div', class_='markdown-content'):
                    new_div = soup.new_tag('div', attrs={'class': 'markdown-content'})
                    for tag in soup.contents[:]:
                        new_div.append(tag.extract())
                    soup.append(new_div)

                # Find all headings and number them consistently
                h1_elements = soup.find_all('h1')
                h2_elements = soup.find_all('h2')

                # First h1 gets the next section number
                if h1_elements:
                    section_numbers['h1'] += 1
                    section_numbers['h2'] = 0
                    section_numbers['current_h1'] = section_numbers['h1']
                    first_h1 = h1_elements[0]
                    original_text = first_h1.get_text()
                    new_text = f"{section_numbers['h1']}. {original_text}"
                    new_tag = soup.new_tag('h1', attrs={'class': 'text-2xl font-bold mb-4'})
                    new_tag.string = new_text
                    first_h1.replace_with(new_tag)

                # Number all h2 elements
                for h2 in h2_elements:
                    section_numbers['h2'] += 1
                    original_text = h2.get_text()
                    new_text = f"{section_numbers['current_h1']}.{section_numbers['h2']}. {original_text}"
                    new_tag = soup.new_tag('h2', attrs={'class': 'text-xl font-bold mt-6 mb-3'})
                    new_tag.string = new_text
                    h2.replace_with(new_tag)

                return str(soup)
            else:
                doc = self._parse_dita_file(topic_path)
                return self._generate_numbered_html_content(doc, section_numbers)

        except Exception as e:
            self.logger.error(f"Error transforming topic with numbering: {str(e)}")
            return self._create_error_html(e, topic_path)

    def _generate_numbered_html_content(self, doc: XMLElement, section_numbers: Dict[str, Any]) -> HTMLString:
        """Generate HTML content with numbered headings"""
        html_content = ['<div class="dita-content">']

        # Handle h1 (main topic title)
        title_elem = self._find_first_element(doc, 'title')
        if title_elem is not None and title_elem.text:
            section_numbers['h1'] += 1
            section_numbers['h2'] = 0  # Reset h2 counter
            section_numbers['current_h1'] = section_numbers['h1']
            numbered_title = f"{section_numbers['h1']}. {title_elem.text}"
            html_content.append(f'<h1 class="text-2xl font-bold mb-4">{numbered_title}</h1>')

        # Add metadata if present
        html_content.extend(self._transform_metadata(doc))

        # Process sections with numbering
        for elem in doc.iter():
            tag = etree.QName(elem).localname
            if tag == 'section':
                section_title = self._find_first_element(elem, 'title')
                if section_title is not None and section_title.text:
                    section_numbers['h2'] += 1
                    numbered_section = f"{section_numbers['current_h1']}.{section_numbers['h2']}. {section_title.text}"
                    html_content.append(f'<h2 class="text-xl font-bold mt-6 mb-3">{numbered_section}</h2>')

            elif tag == 'p' and elem.text:
                html_content.append(f'<p class="mb-4">{elem.text}</p>')
            elif tag == 'shortdesc' and elem.text:
                html_content.append(f'<p class="text-lg text-gray-600 mb-6">{elem.text}</p>')
            elif tag in ['ul', 'ol']:
                html_content.append(self._transform_list(elem))

        html_content.append('</div>')
        return '\n'.join(html_content)

    def _add_numbering_to_html(self, html_content: str, section_numbers: Dict[str, Any]) -> HTMLString:
        """Add numbering to HTML content from Markdown"""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Process h1 headings
        for h1 in soup.find_all('h1'):
            section_numbers['h1'] += 1
            section_numbers['h2'] = 0
            section_numbers['current_h1'] = section_numbers['h1']
            h1.string = f"{section_numbers['h1']}. {h1.text}"

        # Process h2 headings
        for h2 in soup.find_all('h2'):
            section_numbers['h2'] += 1
            h2.string = f"{section_numbers['current_h1']}.{section_numbers['h2']}. {h2.text}"

        return str(soup)

    def _resolve_topic_path(self, map_path: Path, href: str) -> Optional[Path]:
        """Resolve a topic reference path relative to the map file"""
        try:
            # Remove any leading ../ from href
            clean_href = href.replace('../', '')

            # Try to find the topic in the topics directory
            topic_path = self.topics_dir / clean_href

            if topic_path.exists():
                return topic_path

            # If not found, try relative to map location
            relative_path = map_path.parent / href
            if relative_path.exists():
                return relative_path

            return None
        except Exception as e:
            self.logger.error(f"Error resolving topic path: {str(e)}")
            return None

    def _generate_markdown_html(self, metadata: Dict[str, Any], content: str) -> HTMLString:
        """Generate HTML from markdown content with metadata"""
        html_content = ['<div class="markdown-content">']

        # Add title if present
        if 'title' in metadata:
            html_content.append(
                f'<h1 class="text-2xl font-bold mb-4">{metadata["title"]}</h1>'
            )

        # Add metadata table if present
        if metadata:
            html_content.append('<div class="metadata mb-6 bg-gray-50 p-4 rounded-lg">')
            html_content.append('<table class="min-w-full">')
            for key, value in metadata.items():
                if key != 'title':  # Skip title as it's already shown
                    html_content.append(f'<tr><td class="font-semibold">{key}</td>')
                    if isinstance(value, list):
                        html_content.append(f'<td>{", ".join(value)}</td>')
                    else:
                        html_content.append(f'<td>{value}</td>')
                    html_content.append('</tr>')
            html_content.append('</table></div>')

        # Add main content
        html_content.append(content)
        html_content.append('</div>')

        return '\n'.join(html_content)

    def _parse_dita_file(self, input_path: Path) -> XMLElement:
        """Parse DITA file into XML tree"""
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return cast(XMLElement, etree.fromstring(content.encode('utf-8'), self.parser))

    def _generate_html_content(self, doc: XMLElement, input_path: Path) -> HTMLString:
        """Generate HTML content from parsed DITA document"""
        html_content = ['<div class="dita-content">']

        # Add title
        html_content.extend(self._transform_title(doc))

        # Add metadata
        html_content.extend(self._transform_metadata(doc))

        # Add main content
        html_content.extend(self._transform_main_content(doc))

        html_content.append('</div>')
        result = '\n'.join(html_content)

        self.logger.info("HTML transformation completed successfully")
        return result

    def _transform_title(self, doc: XMLElement) -> HTMLElements:
        """Transform title element to HTML"""
        html_content = []
        title_elem = self._find_first_element(doc, 'title')

        if title_elem is not None and title_elem.text:
            html_content.append(
                f'<h1 class="text-2xl font-bold mb-4">{title_elem.text}</h1>'
            )

        return html_content

    def _transform_metadata(self, doc: XMLElement) -> HTMLElements:
        """Transform metadata elements to HTML"""
        html_content = []
        prolog = self._find_first_element(doc, 'prolog')

        if prolog is not None:
            html_content.append('<div class="metadata mb-6 bg-gray-50 p-4 rounded-lg border border-gray-200">')

            # Add table structure
            html_content.append('<table class="min-w-full">')
            html_content.append('<tbody>')

            # Add authors
            authors = [
                author.text for author in prolog.iter()
                if (etree.QName(author).localname == 'author' and author.text)
            ]
            if authors:
                html_content.append('<tr>')
                html_content.append('<td class="py-2 px-4 font-semibold">Authors</td>')
                html_content.append(f'<td class="py-2 px-4">{", ".join(authors)}</td>')
                html_content.append('</tr>')

            # Add institution if present
            institution = self._find_first_element(prolog, 'institution')
            if institution is not None and institution.text:
                html_content.append('<tr>')
                html_content.append('<td class="py-2 px-4 font-semibold">Institution</td>')
                html_content.append(f'<td class="py-2 px-4">{institution.text}</td>')
                html_content.append('</tr>')

            # Add categories
            categories = [
                cat.text for cat in prolog.iter()
                if (etree.QName(cat).localname == 'category' and cat.text)
            ]
            if categories:
                html_content.append('<tr>')
                html_content.append('<td class="py-2 px-4 font-semibold">Categories</td>')
                html_content.append('<td class="py-2 px-4">')
                for category in categories:
                    html_content.append(f'<span class="inline-block bg-blue-100 text-blue-800 px-2 py-1 rounded mr-2 text-sm">{category}</span>')
                html_content.append('</td>')
                html_content.append('</tr>')

            # Add keywords
            keywords = [
                kw.text for kw in prolog.iter()
                if (etree.QName(kw).localname == 'keyword' and kw.text)
            ]
            if keywords:
                html_content.append('<tr>')
                html_content.append('<td class="py-2 px-4 font-semibold">Keywords</td>')
                html_content.append('<td class="py-2 px-4">')
                for keyword in keywords:
                    html_content.append(f'<span class="inline-block bg-gray-100 text-gray-800 px-2 py-1 rounded mr-2 text-sm">{keyword}</span>')
                html_content.append('</td>')
                html_content.append('</tr>')

            # Add other metadata
            for othermeta in prolog.iter():
                if etree.QName(othermeta).localname == 'othermeta':
                    name = othermeta.get('name')
                    content = othermeta.get('content')
                    if name and content:
                        html_content.append('<tr>')
                        html_content.append(f'<td class="py-2 px-4 font-semibold">{name.title()}</td>')
                        html_content.append(f'<td class="py-2 px-4">{content}</td>')
                        html_content.append('</tr>')

            html_content.append('</tbody>')
            html_content.append('</table>')
            html_content.append('</div>')

        return html_content

    def _transform_authors(self, prolog: XMLElement) -> HTMLElements:
        """Transform author elements to HTML"""
        html_content = []
        authors = [
            author.text for author in prolog.iter()
            if (etree.QName(author).localname == 'author' and author.text)
        ]

        if authors:
            html_content.extend([
                '<div class="authors">',
                '<span class="font-semibold">Authors: </span>',
                ', '.join(authors),
                '</div>'
            ])

        return html_content

    def _transform_keywords(self, prolog: XMLElement) -> HTMLElements:
        """Transform keyword elements to HTML"""
        html_content = []
        keywords = [
            keyword.text for keyword in prolog.iter()
            if (etree.QName(keyword).localname == 'keyword' and keyword.text)
        ]

        if keywords:
            html_content.extend([
                '<div class="keywords mt-2">',
                '<span class="font-semibold">Keywords: </span>',
                ', '.join(keywords),
                '</div>'
            ])

        return html_content

    def _transform_main_content(self, doc: XMLElement) -> HTMLElements:
        """Transform main content elements to HTML"""
        html_content = []

        for elem in doc.iter():
            tag = etree.QName(elem).localname
            if tag == 'p' and elem.text:
                html_content.append(
                    f'<p class="mb-4">{elem.text}</p>'
                )
            elif tag == 'shortdesc' and elem.text:
                html_content.append(
                    f'<p class="text-lg text-gray-600 mb-6">{elem.text}</p>'
                )
            elif tag in ['ul', 'ol']:
                html_content.append(self._transform_list(elem))
            elif tag == 'section':
                html_content.extend(self._transform_section(elem))

        return html_content

    def _transform_section(self, section_elem: XMLElement) -> HTMLElements:
        """Transform section element to HTML"""
        html_content = []
        section_title = self._find_first_element(section_elem, 'title')

        if section_title is not None and section_title.text:
            html_content.append(
                f'<h2 class="text-xl font-bold mt-6 mb-3">{section_title.text}</h2>'
            )

        return html_content

    def _find_first_element(self, elem: XMLElement, tag_name: str) -> Optional[XMLElement]:
        """Find first element with given tag name"""
        for child in elem.iter():
            if etree.QName(child).localname == tag_name:
                return child
        return None

    def _create_error_html(self, error: Exception, input_path: Path) -> HTMLString:
        """Create HTML error message"""
        self.logger.error(f"Transformation error: {error}")
        return f"""
        <div class="error-container p-4 bg-red-50 border-l-4 border-red-500 text-red-700">
            <h3 class="font-bold">Error Loading Content</h3>
            <p>{str(error)}</p>
            <p class="text-sm mt-2">File: {input_path}</p>
        </div>
        """

    def list_topics(self) -> List[Dict[str, str]]:
        """List all available topics"""
        topics = []
        subdirs = ['abstracts', 'acoustics', 'articles', 'audio', 'journals', 'reference']

        self.logger.info(f"Searching for topics in: {self.topics_dir}")

        for subdir in subdirs:
            subdir_path = self.topics_dir / subdir
            if subdir_path.exists():
                self.logger.info(f"Checking directory: {subdir_path}")

                # Handle DITA files
                for topic_file in subdir_path.glob('*.dita'):
                    try:
                        self.logger.info(f"Found DITA file: {topic_file}")
                        with open(topic_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        tree = etree.fromstring(content.encode('utf-8'), self.parser)

                        title_elem = None
                        for elem in tree.iter():
                            if etree.QName(elem).localname == 'title':
                                title_elem = elem
                                break

                        content_preview = self._extract_content_preview(tree)

                        topic_data = {
                            'id': topic_file.stem,
                            'title': title_elem.text if title_elem is not None else topic_file.stem,
                            'path': str(topic_file.relative_to(self.dita_root)),
                            'type': subdir,
                            'format': 'dita',
                            'fullPath': str(topic_file),
                            'preview': content_preview or "No content available",
                            'hasContent': bool(content_preview)
                        }
                        self.logger.info(f"Adding DITA topic: {topic_data}")
                        topics.append(topic_data)
                    except Exception as e:
                        self.logger.error(f"Error processing DITA file {topic_file}: {str(e)}")
                        topics.append(self._create_error_topic(topic_file, subdir, str(e)))

                # Handle Markdown files
                for topic_file in subdir_path.glob('*.md'):
                    try:
                        self.logger.info(f"Found Markdown file: {topic_file}")
                        with open(topic_file, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # Parse frontmatter and content
                        post = frontmatter.loads(content)
                        metadata = post.metadata

                        topic_data = {
                            'id': topic_file.stem,
                            'title': metadata.get('title', topic_file.stem),
                            'path': str(topic_file.relative_to(self.dita_root)),
                            'type': metadata.get('type', subdir),
                            'format': 'markdown',
                            'fullPath': str(topic_file),
                            'preview': post.content[:200] + "..." if post.content else "No content available",
                            'hasContent': bool(post.content),
                            'metadata': metadata
                        }
                        self.logger.info(f"Adding Markdown topic: {topic_data}")
                        topics.append(topic_data)
                    except Exception as e:
                        self.logger.error(f"Error processing Markdown file {topic_file}: {str(e)}")
                        topics.append(self._create_error_topic(topic_file, subdir, str(e)))

        self.logger.info(f"Total topics found: {len(topics)}")
        return topics

    def _create_error_topic(self, topic_file: Path, subdir: str, error: str) -> Dict[str, Any]:
        """Create an error topic entry"""
        return {
            'id': topic_file.stem,
            'title': topic_file.stem,
            'path': str(topic_file.relative_to(self.dita_root)),
            'type': subdir,
            'format': 'dita' if topic_file.suffix == '.dita' else 'markdown',
            'fullPath': str(topic_file),
            'preview': "Error loading content",
            'hasContent': False,
            'error': error
        }

    def get_topic_path(self, topic_id: str) -> Optional[Path]:
        """Get the full path for a topic by its ID"""
        self.logger.info(f"Looking for topic with ID: {topic_id}")

        # First check if this is a .ditamap file
        if topic_id.endswith('.ditamap'):
            map_path = self.maps_dir / topic_id
            if map_path.exists():
                self.logger.info(f"Found map at: {map_path}")
                return map_path
            # Remove .ditamap extension and continue search
            topic_id = topic_id.replace('.ditamap', '')

        # Remove any file extension from topic_id
        topic_base = topic_id.replace('.md', '').replace('.dita', '')

        # Check in maps directory first for .ditamap
        map_path = self.maps_dir / f"{topic_base}.ditamap"
        if map_path.exists():
            self.logger.info(f"Found map at: {map_path}")
            return map_path

        # Then check in topic directories
        for subdir in ['acoustics', 'articles', 'audio', 'abstracts', 'journals', 'reference']:
            subdir_path = self.topics_dir / subdir

            # Check for .dita file
            dita_path = subdir_path / f"{topic_base}.dita"
            self.logger.info(f"Checking DITA path: {dita_path}")
            if dita_path.exists():
                self.logger.info(f"Found topic at: {dita_path}")
                return dita_path

            # Check for .md file
            md_path = subdir_path / f"{topic_base}.md"
            self.logger.info(f"Checking MD path: {md_path}")
            if md_path.exists():
                self.logger.info(f"Found topic at: {md_path}")
                return md_path

        self.logger.error(f"No topic found with ID: {topic_id}")
        return None

    def _extract_content_preview(self, tree: XMLElement) -> Optional[str]:
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

    def list_maps(self) -> List[Dict[str, Any]]:
        """List all available DITA maps"""
        maps = []
        self.logger.info(f"Searching for maps in: {self.maps_dir}")

        if self.maps_dir.exists():
            for map_file in self.maps_dir.glob('*.ditamap'):
                try:
                    self.logger.info(f"Processing map file: {map_file}")

                    with open(map_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    tree = etree.fromstring(content.encode('utf-8'), self.parser)

                    # Get the map title
                    title_elem = tree.find(".//title")
                    title = title_elem.text if title_elem is not None else map_file.stem

                    # Process topic groups
                    groups = []
                    for topicgroup in tree.findall(".//topicgroup"):
                        navtitle = topicgroup.find(".//navtitle")
                        group_title = navtitle.text if navtitle is not None else "Untitled Group"

                        topics = []
                        for topicref in topicgroup.findall(".//topicref"):
                            href = topicref.get('href')
                            if href:
                                topic_id = Path(href).stem
                                topics.append({
                                    'id': topic_id,
                                    'href': href
                                })

                        groups.append({
                            'navtitle': group_title,
                            'topics': topics
                        })

                    maps.append({
                        'id': map_file.stem,
                        'title': title,
                        'groups': groups
                    })

                except Exception as e:
                    self.logger.error(f"Error processing map {map_file}: {e}")
                    maps.append({
                        'id': map_file.stem,
                        'title': map_file.stem,
                        'error': str(e)
                    })

        return maps

    def generate_toc(self, topic_path: Path) -> List[Dict[str, Any]]:
        """Generate table of contents from topic"""
        try:
            if not topic_path or not topic_path.exists():
                return []

            with open(topic_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = etree.fromstring(content.encode('utf-8'), self.parser)
            if tree is None:
                return []

            toc = []
            # Process headers recursively
            for section in tree.xpath('//*[contains(local-name(), "section")]') or []:
                title_elem = section.find('.//title')
                if title_elem is not None and title_elem.text:
                    section_id = section.get('id', '') or f"section-{len(toc)}"
                    toc_item = {
                        'id': section_id,
                        'title': title_elem.text,
                        'level': len(section.xpath('ancestor::*[contains(local-name(), "section")]')) + 1,
                        'children': []
                    }
                    toc.append(toc_item)

            return toc
        except Exception as e:
            self.logger.error(f"Error generating TOC: {str(e)}")
            return []

    def get_topic_metadata(self, topic_path: Path) -> Dict[str, Any]:
        """Extract metadata from topic"""
        try:
            if not topic_path or not topic_path.exists():
                return {}

            with open(topic_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if topic_path.suffix == '.md':
                # Handle Markdown front matter
                post = frontmatter.loads(content)
                return post.metadata or {}

            tree = etree.fromstring(content.encode('utf-8'), self.parser)
            if tree is None:
                return {}

            metadata = {}

            # Extract prolog metadata
            prolog = tree.find('.//prolog')
            if prolog is not None:
                # ... rest of metadata extraction ...
                pass

            # Get title
            title_elem = tree.find('.//title')
            if title_elem is not None and title_elem.text:
                metadata['title'] = title_elem.text

            return metadata
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {str(e)}")
            return {}

    def search_topics(self, query: str) -> List[Dict[str, Any]]:
        """Search through topics"""
        results = []
        try:
            topics = self.list_topics()
            for topic in topics:
                topic_path = self.get_topic_path(topic['id'])
                if topic_path:
                    with open(topic_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Simple text search (can be enhanced with better search algorithms)
                    if query.lower() in content.lower():
                        # Get a preview of the matching content
                        preview = self._extract_preview(content, query)
                        results.append({
                            'id': topic['id'],
                            'title': topic['title'],
                            'preview': preview,
                            'type': topic.get('type', 'topic')
                        })

            return results
        except Exception as e:
            self.logger.error(f"Error searching topics: {str(e)}")
            return []

    def _extract_preview(self, content: str, query: str, chars: int = 200) -> str:
        """Extract a preview of the content around the search query"""
        try:
            lower_content = content.lower()
            query_pos = lower_content.find(query.lower())

            if query_pos != -1:
                start = max(0, query_pos - chars // 2)
                end = min(len(content), query_pos + len(query) + chars // 2)
                preview = content[start:end]

                if start > 0:
                    preview = f"...{preview}"
                if end < len(content):
                    preview = f"{preview}..."

                return preview
            return ""
        except Exception:
            return ""
