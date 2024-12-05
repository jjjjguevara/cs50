# app/dita/utils/markdown/md_transform.py

from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
import markdown
import frontmatter
from bs4 import BeautifulSoup, Tag

from ..id_handler import DITAIDHandler
from ..heading import HeadingHandler
from ..metadata import MetadataHandler
from ..html_helpers import HTMLHelper
from .md_elements import MarkdownContentProcessor

class MarkdownTransformer:
    """
    Handles transformation of Markdown content to HTML with DITA-compliant structure.
    """

    logger: logging.Logger
    root_path: Path
    id_handler: DITAIDHandler
    heading_handler: HeadingHandler
    metadata_handler: MetadataHandler
    html_helper: HTMLHelper
    md_processor: MarkdownContentProcessor

    def __init__(self, root_path: Path):
        self.logger = logging.getLogger(__name__)
        self.root_path = root_path
        self.id_handler = DITAIDHandler()
        self.heading_handler = HeadingHandler()
        self.metadata_handler = MetadataHandler()
        self.html_helper = HTMLHelper()

        # Initialize markdown converter
        self.markdown = markdown.Markdown(extensions=[
            'fenced_code',
            'tables',
            'attr_list'
        ])

        # Initialize markdown processor - no LaTeX
        self.md_processor = MarkdownContentProcessor()

    def process_topic(self, topic_path: Path) -> str:
        """Transform markdown topic to HTML."""
        try:
            self.logger.info(f"Processing markdown topic: {topic_path}")
            return self.transform_content(topic_path)
        except Exception as e:
            self.logger.error(f"Error processing markdown topic: {str(e)}")
            return self._create_error_html(str(e), topic_path)

    def transform_topic(self, topic_path: Path) -> str:
        """Transform a markdown topic to HTML."""
        try:
            self.logger.info(f"Transforming markdown topic: {topic_path}")

            # Read the markdown file
            with open(topic_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
                self.logger.info(f"Raw markdown content length: {len(markdown_content)}")
                self.logger.info(f"Raw markdown sample: {markdown_content[:200]}")

            # Parse frontmatter and content
            post = frontmatter.loads(markdown_content)
            self.logger.info(f"Frontmatter: {post.metadata}")
            self.logger.info(f"Content length: {len(post.content)}")

            # First convert markdown to HTML
            html_content = self.markdown.convert(post.content)
            self.logger.info(f"Initial HTML conversion length: {len(html_content)}")
            self.logger.info(f"Initial HTML sample: {html_content[:200]}")

            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # Log all found elements
            all_elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'blockquote', 'pre', 'ul', 'ol'])
            self.logger.info(f"Found {len(all_elements)} elements to process")
            for idx, elem in enumerate(all_elements):
                self.logger.info(f"Element {idx}: {elem.name} - {elem.get_text()[:50]}")

            # Process each element
            processed_html = []
            for elem in all_elements:
                if isinstance(elem, Tag):
                    self.logger.info(f"Processing element: {elem.name}")
                    processed = self.md_processor.process_element(elem, topic_path)
                    if processed:
                        processed_html.append(processed)
                        self.logger.info(f"Processed {elem.name}: {processed[:100]}")
                    else:
                        self.logger.warning(f"Element processing returned empty: {elem.name}")

            # Combine processed content
            content = '\n'.join(processed_html)
            self.logger.info(f"Final processed content length: {len(content)}")
            self.logger.info(f"Final content sample: {content[:200]}")

            # Return within required wrapper structure
            map_id = self.id_handler.generate_content_id(topic_path)
            final_html = f"""
            <div class="content-wrapper">
                <div class="dita-content" data-map-id="{map_id}">
                    <nav class="dita-toc"></nav>
                    {content}  <!-- THIS IS WHERE THE CONTENT GOES -->
                </div>
            </div>
            """
            self.logger.info(f"Final HTML length: {len(final_html)}")
            self.logger.debug(f"Final HTML structure:\n{final_html}")
            return final_html.strip()  # Add strip() to remove extra whitespace

        except Exception as e:
            self.logger.error(f"Error transforming markdown topic: {str(e)}", exc_info=True)
            return self._create_error_html(str(e), topic_path)

    def transform_content(self, content_path: Path) -> str:
        """Transform markdown content to HTML"""
        try:
            self.logger.info(f"Transforming markdown content: {content_path}")

            # Reset state for new transformation
            self.heading_handler.reset()

            # Read and parse markdown content
            with open(content_path, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)

            # Extract metadata
            metadata = self._process_metadata(post.metadata, content_path)

            # Process content
            html_content = self._process_content(post.content, content_path)

            # Apply metadata-based styling
            final_content = self._apply_metadata_styling(html_content, metadata)

            return final_content

        except Exception as e:
            self.logger.error(f"Error transforming markdown content: {str(e)}")
            return self._create_error_html(str(e), content_path)

    def _process_metadata(self, metadata: Dict[str, Any],
                         source_path: Path) -> Dict[str, Any]:
        """Process markdown metadata"""
        try:
            # Extract standard metadata
            processed_metadata = {
                'title': metadata.get('title', ''),
                'description': metadata.get('description', ''),
                'author': metadata.get('author', ''),
                'date': metadata.get('date', ''),
                'tags': metadata.get('tags', []),
                'source_path': str(source_path)
            }

            # Process DITA-specific metadata
            if 'dita' in metadata:
                dita_meta = metadata['dita']
                processed_metadata.update({
                    'topic_type': dita_meta.get('type', 'topic'),
                    'product': dita_meta.get('product', ''),
                    'platform': dita_meta.get('platform', ''),
                    'audience': dita_meta.get('audience', ''),
                })

            return processed_metadata

        except Exception as e:
            self.logger.error(f"Error processing metadata: {str(e)}")
            return {}

    def _process_content(self, content: str, source_path: Path) -> str:
        """Process markdown content to HTML"""
        try:
            # Parse markdown into BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')

            # Process elements
            processed_html = []
            for elem in soup.children:
                if isinstance(elem, Tag):
                    processed_html.append(
                        self.md_processor.process_element(elem, source_path)
                    )

            return '\n'.join(processed_html)

        except Exception as e:
            self.logger.error(f"Error processing content: {str(e)}")
            return self._create_error_html(str(e), source_path)

    def _apply_metadata_styling(self, content: str,
                              metadata: Dict[str, Any]) -> str:
        """Apply metadata-based styling to content"""
        try:
            soup = BeautifulSoup(content, 'html.parser')

            # Get or create main wrapper
            wrapper = self.html_helper.ensure_wrapper(soup)

            # Add topic type class
            topic_type = metadata.get('topic_type', 'topic')
            self.html_helper.add_class(wrapper, f'md-{topic_type}')

            # Add metadata attributes
            self.html_helper.set_data_attributes(wrapper, {
                'topic-type': topic_type,
                'author': metadata.get('author', ''),
                'date': metadata.get('date', ''),
                'tags': ','.join(metadata.get('tags', [])),
            })

            return str(soup)

        except Exception as e:
            self.logger.error(f"Error applying metadata styling: {str(e)}")
            return content

    def _create_error_html(self, error: str, context: Union[Path, str]) -> str:
        """Create HTML error message"""
        return f"""
            <div class="error-container bg-red-50 border-l-4 border-red-500 p-4 my-4">
                <h3 class="text-lg font-medium text-red-800">Markdown Processing Error</h3>
                <p class="text-red-700">{error}</p>
                <div class="mt-2 text-sm text-red-600">
                    <p>Error in: {context}</p>
                </div>
            </div>
        """

    def reset_state(self) -> None:
        """Reset all stateful components"""
        self.heading_handler.reset()
        self.id_handler = DITAIDHandler()
