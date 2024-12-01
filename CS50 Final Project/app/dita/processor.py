import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union, cast, Callable
from .artifacts.parser import ArtifactParser
from .artifacts.renderer import ArtifactRenderer
from .utils.heading import HeadingIDGenerator
from .citations import parse_citations
import frontmatter
import markdown
from bs4 import BeautifulSoup
from lxml import etree
from markdown.extensions import fenced_code, meta, tables


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
    md: markdown.Markdown
    artifact_parser: ArtifactParser
    artifact_renderer: ArtifactRenderer
    heading_generator: HeadingIDGenerator

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

        self.artifact_parser = ArtifactParser(self.dita_root)
        self.artifact_renderer = ArtifactRenderer(self.dita_root / 'artifacts')
        self.heading_generator = HeadingIDGenerator()

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


    # Adds artifacts to article sections based on DITA coordinates
    def _inject_artifacts(self, html_content: str, artifacts: List[Dict]) -> str:
        """Inject artifacts into HTML content at specified locations"""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Add required scripts if not already present
        head = soup.find('head')
        if head:
            # Add React scripts if not present
            react_scripts = [
                'https://unpkg.com/react@18/umd/react.development.js',
                'https://unpkg.com/react-dom@18/umd/react-dom.development.js'
            ]
            for script_src in react_scripts:
                if not soup.find('script', src=script_src):
                    script_tag = soup.new_tag('script', src=script_src)
                    head.append(script_tag)

        # Inject artifacts
        for artifact in artifacts:
            target = soup.find(id=artifact['target_heading'])
            if target:
                artifact_html = self.artifact_renderer.render_artifact(
                    self.dita_root / artifact['source']
                )
                target.insert_after(BeautifulSoup(artifact_html, 'html.parser'))

        return str(soup)

    def transform_to_html(self, input_path: Path) -> HTMLString:
            """Transform DITA content to HTML, with artifact handling"""
            try:
                # Handle different file types
                if input_path.suffix == '.md':
                    with open(input_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    metadata, html_content = self._parse_markdown(content)
                    return self._generate_markdown_html(metadata, html_content, input_path)
                elif input_path.suffix == '.ditamap':
                    artifacts = self.artifact_parser.parse_artifact_references(input_path)
                    html_content = self._transform_map_to_html(input_path)
                    return self._inject_artifacts(html_content, artifacts)
                else:
                    self.logger.info(f"Transforming {input_path} to HTML")
                    doc = self._parse_dita_file(input_path)
                    return self._generate_html_content(doc, input_path)
            except Exception as e:
                return self._create_error_html(e, input_path)

    def _transform_map_to_html(self, map_path: Path) -> HTMLString:
        """Transform DITA map and its referenced topics to HTML"""
        try:
            # Log map transformation start
            self.logger.info(f"Transforming map: {map_path}")

            # Get map ID without extension
            map_id = map_path.stem

            with open(map_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = etree.fromstring(content.encode('utf-8'), self.parser)
            html_content = ['<div class="map-content">']

            # Handle main title
            title_elem = tree.find(".//title")
            if title_elem is not None and title_elem.text:
                html_content.append(f'<h1 class="content-title">{title_elem.text}</h1>')

            # Initialize section counters and reference counter
            section_numbers = {
                'h1': 0,
                'h2': 0,
                'h3': 0,
                'current_h1': None
            }
            reference_counter = 1
            all_references = []

            # Process all topics and collect references
            for topicgroup in tree.findall(".//topicgroup"):
                for topicref in topicgroup.findall(".//topicref"):
                    href = topicref.get('href')
                    if href:
                        self.logger.info(f"Processing topicref with href: {href}")
                        topic_path = self._resolve_topic_path(map_path, href)
                        if topic_path and topic_path.exists():
                            html_content.append('<div class="content-section">')

                            # Pass map_id to _transform_topic_with_numbering
                            topic_content, topic_references = self._transform_topic_with_numbering(
                                topic_path,
                                section_numbers,
                                reference_counter,
                                map_id  # Pass map ID here
                            )

                            # Update reference counter and collect references
                            if topic_references:
                                for ref in topic_references:
                                    all_references.append(ref)
                                    reference_counter = max(reference_counter, int(ref['id'])) + 1

                            html_content.append(topic_content)
                            html_content.append('</div>')
                        else:
                            self.logger.error(f"Could not resolve topic for href: {href}")

            html_content.append('</div>')
            return '\n'.join(html_content)

        except Exception as e:
            self.logger.error(f"Error transforming map to HTML: {str(e)}")
            return self._create_error_html(e, map_path)


    def _transform_topic_with_numbering(
        self,
        topic_path: Path,
        section_numbers: Dict[str, Any],
        start_ref_number: int,
        current_map_id: Optional[str] = None  # Fixed type hint
    ) -> Tuple[HTMLString, List[Dict[str, str]]]:
        """Transform a topic to HTML with numbered headings"""
        try:
            references = []
            if topic_path.suffix == '.md':
                with open(topic_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                metadata, html_content = self._parse_markdown(content)
                soup = BeautifulSoup(html_content, 'html.parser')

                # Process images
                for img in soup.find_all('img'):
                    src = img.get('src', '')
                    if src:
                        self.logger.info(f"Processing image src: {src}")
                        img_path = (topic_path.parent / src).resolve()
                        try:
                            relative_path = img_path.relative_to(self.topics_dir)
                            new_src = f'/static/topics/{relative_path}'
                            self.logger.info(f"Transformed image path: {new_src}")
                            img['src'] = new_src
                            img['class'] = 'max-w-full h-auto'
                        except ValueError as e:
                            self.logger.warning(f"Image path {img_path} is not within topics directory: {e}")

                # Process internal links
                for link in soup.find_all('a'):
                    href = link.get('href', '')
                    if href and '#' in href:
                        file_part, heading_part = href.split('#', 1)
                        if file_part:
                            link['href'] = f'/entry/{current_map_id}#{heading_part}'
                        else:
                            link['href'] = f'#{heading_part}'
                        self.logger.info(f"Transformed link: {href} -> {link['href']}")

                # Initialize h3 counter in section_numbers if not present
                if 'h3' not in section_numbers:
                    section_numbers['h3'] = 0

                # Process headings
                headings = soup.find_all(['h1', 'h2', 'h3'])
                for heading in headings:
                    original_text = heading.get_text(strip=True)
                    heading_id = self._generate_id(original_text)
                    new_text = original_text

                    if heading.name == 'h1':
                        section_numbers['h1'] += 1
                        section_numbers['h2'] = 0
                        section_numbers['h3'] = 0
                        section_numbers['current_h1'] = section_numbers['h1']
                        new_text = f"{section_numbers['h1']}. {original_text}"
                    elif heading.name == 'h2':
                        section_numbers['h2'] += 1
                        section_numbers['h3'] = 0
                        new_text = f"{section_numbers['current_h1']}.{section_numbers['h2']}. {original_text}"
                    elif heading.name == 'h3':
                        section_numbers['h3'] += 1
                        new_text = f"{section_numbers['current_h1']}.{section_numbers['h2']}.{section_numbers['h3']}. {original_text}"

                    heading['id'] = heading_id
                    heading.string = new_text

                    if heading.name == 'h1':
                        heading['class'] = 'text-2xl font-bold mb-4'
                    elif heading.name == 'h2':
                        heading['class'] = 'text-xl font-bold mt-6 mb-3'
                    elif heading.name == 'h3':
                        heading['class'] = 'text-lg font-bold mt-4 mb-2'

                    anchor = soup.new_tag('a', attrs={
                        'href': f'#{heading_id}',
                        'class': 'heading-anchor',
                        'aria-label': 'Link to this heading'
                    })
                    anchor.string = '¶'
                    heading.append(anchor)

                if not soup.find('div', class_='markdown-content'):
                    wrapper = soup.new_tag('div', attrs={'class': 'markdown-content'})
                    for tag in soup.contents[:]:
                        wrapper.append(tag.extract())
                    soup.append(wrapper)

                return str(soup), references

            else:
                doc = self._parse_dita_file(topic_path)
                content = self._generate_numbered_html_content(doc, section_numbers, start_ref_number)
                return content, references

        except Exception as e:
            self.logger.error(f"Error transforming topic with numbering: {str(e)}")
            return self._create_error_html(e, topic_path), []



    def _validate_json(self, json_str: str) -> bool:
        """Validate JSON string"""
        try:
            json.loads(json_str)
            return True
        except Exception as e:
            self.logger.error(f"Invalid JSON: {e}")
            self.logger.error(f"JSON string: {json_str}")
            return False


    def _generate_numbered_html_content(self, doc: XMLElement, section_numbers: Dict[str, Any], start_ref_number: int) -> HTMLString:
        """Generate HTML content with numbered headings"""
        try:

            html_content = ['<div class="dita-content">']

            # Handle h1 (main topic title)
            title_elem = self._find_first_element(doc, 'title')
            if title_elem is not None and title_elem.text:
                section_numbers['h1'] += 1
                section_numbers['h2'] = 0
                section_numbers['current_h1'] = section_numbers['h1']
                numbered_title = f"{section_numbers['h1']}. {title_elem.text}"
                heading_id = self._generate_id(numbered_title)
                html_content.append(
                    f'<h1 id="{heading_id}" class="text-2xl font-bold mb-4">'
                    f'<span class="heading-text">{numbered_title}</span>'  # Wrap the text in span
                    f'<a href="#{heading_id}" class="heading-anchor" aria-label="Link to this heading">¶</a>'
                    f'</h1>'
                )

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
                        heading_id = self._generate_id(numbered_section)
                        html_content.append(
                            f'<h2 id="{heading_id}" class="text-xl font-bold mt-6 mb-3">'
                            f'<span class="heading-text">{numbered_section}</span>'  # Wrap the text in span
                            f'<a href="#{heading_id}" class="heading-anchor" aria-label="Link to this heading">¶</a>'
                            f'</h2>'
                        )
                elif tag == 'p' and elem.text:
                    html_content.append(f'<p class="mb-4">{elem.text}</p>')
                elif tag == 'shortdesc' and elem.text:
                    html_content.append(f'<p class="text-lg text-gray-600 mb-6">{elem.text}</p>')
                elif tag in ['ul', 'ol']:
                    html_content.append(self._transform_list(elem))

            html_content.append('</div>')
            return '\n'.join(html_content)

        except Exception as e:
            self.logger.error(f"Error generating numbered HTML content: {str(e)}")
            return f"""
            <div class="error-container p-4 bg-red-50 border-l-4 border-red-500 text-red-700">
                <h3 class="font-bold">Error Processing Content</h3>
                <p>{str(e)}</p>
            </div>
            """

    def _generate_id(self, text: str) -> str:
        """Generate a URL-friendly ID from text"""
        return text.lower().replace(' ', '-').replace('.', '-').replace('(', '').replace(')', '').replace(',', '')

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
            self.logger.info(f"Resolving topic path for href: {href} relative to map: {map_path}")

            # Handle relative paths starting with ../
            if href.startswith('../'):
                # Start from the map directory and resolve the relative path
                resolved_path = (map_path.parent / href).resolve()
                self.logger.info(f"Resolved relative path to: {resolved_path}")

                if resolved_path.exists():
                    self.logger.info(f"Found topic file at: {resolved_path}")
                    return resolved_path

            # Also try direct path from topics directory
            clean_href = href.replace('../topics/', '')
            topic_path = self.topics_dir / clean_href
            self.logger.info(f"Trying direct topics path: {topic_path}")

            if topic_path.exists():
                self.logger.info(f"Found topic file at: {topic_path}")
                return topic_path

            self.logger.error(f"Could not resolve topic path for href: {href}")
            return None
        except Exception as e:
            self.logger.error(f"Error resolving topic path: {str(e)}")
            return None

    def _generate_markdown_html(self, metadata: Dict[str, Any], content: str, current_path: Optional[Path] = None) -> HTMLString:
        """Generate HTML from markdown content with metadata"""
        html_content = ['<div class="markdown-content">']

        # Create a BeautifulSoup object for the markdown content
        soup = BeautifulSoup(content, 'html.parser')

        # Process images if current_path is provided
        if current_path:
                    self.logger.info(f"Processing markdown from path: {current_path}")
                    for img in soup.find_all('img'):
                        src = img.get('src', '')
                        if src:
                            self.logger.info(f"Found image with src: {src}")
                            # Convert the image path to be relative to the topics directory
                            img_path = (current_path.parent / src).resolve()
                            self.logger.info(f"Resolved full image path: {img_path}")
                            try:
                                relative_path = img_path.relative_to(self.topics_dir)
                                new_src = f'/static/topics/{relative_path}'
                                self.logger.info(f"Setting new image src to: {new_src}")
                                img['src'] = new_src
                                img['class'] = 'max-w-full h-auto'
                            except ValueError as e:
                                self.logger.warning(f"Image path {img_path} is not within topics directory: {e}")

        # Find the first h1 or create one from metadata
        h1 = soup.find('h1')
        if not h1 and 'title' in metadata:
            heading_id = self._generate_id(metadata['title'])
            html_content.append(
                f'<h1 id="{heading_id}" class="text-2xl font-bold mb-4">'
                f'<span class="heading-text">{metadata["title"]}</span>'
                f'<a href="#{heading_id}" class="heading-anchor" aria-label="Link to this heading">¶</a>'
                f'</h1>'
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

        # Process the main content with proper heading structure
        # If we created an h1 from metadata, we want to preserve that structure
        if 'title' in metadata:
            # Remove any existing h1 elements to avoid duplication
            for h1 in soup.find_all('h1'):
                h1.name = 'h2'  # Convert to h2 to maintain hierarchy
                heading_id = self._generate_id(h1.text)
                h1['id'] = heading_id
                h1['class'] = 'text-xl font-bold mt-6 mb-3'
                # Add anchor link
                anchor = soup.new_tag('a', attrs={
                    'href': f'#{heading_id}',
                    'class': 'heading-anchor',
                    'aria-label': 'Link to this heading'
                })
                anchor.string = '¶'
                h1.append(anchor)

        # Process all remaining headings
        for heading in soup.find_all(['h2', 'h3', 'h4', 'h5', 'h6']):
            heading_id = self._generate_id(heading.text)
            heading['id'] = heading_id
            if heading.name == 'h2':
                heading['class'] = 'text-xl font-bold mt-6 mb-3'
            # Add anchor link
            anchor = soup.new_tag('a', attrs={
                'href': f'#{heading_id}',
                'class': 'heading-anchor',
                'aria-label': 'Link to this heading'
            })
            anchor.string = '¶'
            heading.append(anchor)

        # Add the processed content
        html_content.append(str(soup))
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

        # Remove any file extension from topic_id
        topic_base = topic_id.replace('.md', '').replace('.dita', '')

        # Handle subdirectories in topic_id
        topic_parts = topic_base.split('/')
        topic_filename = topic_parts[-1]
        subdirs = topic_parts[:-1]  # This will be empty if no subdirectories

        self.logger.info(f"Searching for topic: {topic_filename} in subdirs: {subdirs}")

        # Check in maps directory first for .ditamap
        map_path = self.maps_dir / f"{topic_filename}.ditamap"
        if map_path.exists():
            self.logger.info(f"Found map at: {map_path}")
            return map_path

        # Then check in topic directories
        for subdir in ['acoustics', 'articles', 'audio', 'abstracts', 'journals', 'reference']:
            # Start with the base topic directory
            search_dir = self.topics_dir / subdir

            # Add any subdirectories from the topic_id
            if subdirs:
                search_dir = search_dir.joinpath(*subdirs)

            # Only proceed if this directory exists
            if not search_dir.exists():
                continue

            self.logger.info(f"Searching in directory: {search_dir}")

            # Check for .dita file
            dita_path = search_dir / f"{topic_filename}.dita"
            self.logger.info(f"Checking DITA path: {dita_path}")
            if dita_path.exists():
                self.logger.info(f"Found topic at: {dita_path}")
                return dita_path

            # Check for .md file
            md_path = search_dir / f"{topic_filename}.md"
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
