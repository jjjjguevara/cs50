from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import frontmatter
import markdown
import logging
from .renderer import DITAHTMLRenderer
from .extensions.dita_extensions import DITAExtension
from .extensions.yaml_metadata import YAMLMetadataExtension


class MarkdownParser:
    """Parser for Markdown files with DITA semantics"""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.renderer = DITAHTMLRenderer()

        # Create the YAML metadata extension instance
        self.yaml_ext = YAMLMetadataExtension()

        # Initialize Markdown with custom extensions
        self.md = markdown.Markdown(
            extensions=[
                'fenced_code',
                'tables',
                'attr_list',
                DITAExtension(),
                self.yaml_ext
            ],
            output_format='html'
        )
    def parse_file(self, file_path: Path) -> Tuple[Dict[str, Any], str]:
        """Parse a Markdown file and return metadata and HTML content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.parse_content(content)
        except Exception as e:
            self.logger.error(f"Error parsing markdown file {file_path}: {e}")
            return {}, ""

    def parse_content(self, content: str) -> Tuple[Dict[str, Any], str]:
        """Parse Markdown content and render with consistent metadata"""
        try:
            # Parse front matter
            post = frontmatter.loads(content)
            metadata = post.metadata or {}

            # Convert markdown to HTML using custom renderer
            html_content = self.md.convert(post.content)

            # Render metadata in DITA style
            metadata_html = self.renderer.render_metadata(metadata)

            # Combine metadata and content
            final_html = f"""
            <div class="dita-content">
                <h1 class="text-2xl font-bold mb-4">{metadata.get('title', '')}</h1>
                {metadata_html}
                <div class="topic-content">
                    {html_content}
                </div>
            </div>
            """

            return metadata, final_html
        except Exception as e:
            self.logger.error(f"Error parsing markdown content: {e}")
            return {}, ""

    def validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate required DITA metadata"""
        required_fields = ['title', 'type']
        return all(field in metadata for field in required_fields)
