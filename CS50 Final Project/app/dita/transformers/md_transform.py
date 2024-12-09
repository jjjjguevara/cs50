# app/dita/transformers/md_transform.py

import logging
from pathlib import Path
from typing import Optional
import html
import markdown
from bs4 import BeautifulSoup
from ..models.types import (
    ParsedElement,
    ProcessedContent,
    MDElementInfo,
    MDElementType,
    HeadingContext
)
from app_config import DITAConfig
from ..utils.html_helpers import HTMLHelper
from ..utils.heading import HeadingHandler
from app.dita.processors.md_elements import MarkdownContentProcessor
from app.dita.utils.latex.latex_processor import LaTeXProcessor


class MarkdownTransformer:
    """Transforms Markdown content to HTML using element definitions."""

    def __init__(self, root_path: Path):
        self.logger = logging.getLogger(__name__)
        self.root_path = root_path
        self.html_helper = HTMLHelper(root_path)
        self.heading_handler = HeadingHandler()
        self.content_processor = MarkdownContentProcessor()
        self.latex_processor = LaTeXProcessor()

        # Update extensions configuration
        self.extensions = [
            'markdown.extensions.fenced_code',
            'markdown.extensions.tables',
            'app.dita.utils.markdown.latex_extension'
        ]

        self.extension_configs = {
            'app.dita.utils.markdown.latex_extension': {
                'use_dollar_delimiter': True,
                'process_latex': True
            }
        }

    def configure(self, config: DITAConfig):
            """
            Apply additional configuration settings.
            """
            self.logger.debug("Configuring MarkdownTransformer")
            # Example: Add custom configuration logic
            self.some_markdown_setting = getattr(config, 'markdown_setting', None)

    def transform_topic(self, parsed_element: ParsedElement) -> ProcessedContent:
        try:
            self.logger.debug(f"Transforming Markdown topic: {parsed_element.topic_path}")
            content = parsed_element.content
            html_parts = []

            # Start new topic
            self.heading_handler.start_new_topic()

            # Handle YAML frontmatter
            if content.startswith('---'):
                end_index = content.find('---', 3)
                if end_index != -1:
                    frontmatter = content[0:end_index+3]
                    content = content[end_index+3:].strip()
                    html_parts.append(
                        f'<div class="code-block-wrapper">'
                        f'<div class="code-label">yaml</div>'
                        f'<pre class="code-block yaml-frontmatter"><code>{html.escape(frontmatter)}</code></pre>'
                        f'</div>'
                    )

            # Convert Markdown to HTML
            html_content = markdown.markdown(content, extensions=self.extensions)
            soup = BeautifulSoup(html_content, "html.parser")

            # Process all elements
            self._process_todo_lists(soup)
            self._process_images(soup)
            self._process_code_blocks(soup)

            # Process headings with proper hierarchy
            first_heading = True
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                level = int(heading.name[1])
                heading_id, numbered_heading = self.heading_handler.process_heading(
                    text=heading.text.strip(),
                    level=level,
                    is_topic_title=first_heading
                )
                heading['id'] = heading_id
                heading.string = numbered_heading
                pilcrow = soup.new_tag('a', href=f"#{heading_id}", **{'class': 'pilcrow'})
                pilcrow.string = '¶'
                heading.append(pilcrow)
                first_heading = False

            # Combine parts and wrap in container
            final_html = (
                '<div class="markdown-content">'
                f'{"".join(html_parts)}{str(soup)}'
                '</div>'
            )

            return ProcessedContent(
                html=final_html,
                element_id=parsed_element.id,
                metadata=parsed_element.metadata
            )
        except Exception as e:
                self.logger.error(f"Error transforming Markdown topic: {str(e)}", exc_info=True)
                return ProcessedContent(
                    html=f"<div class='error'>Error processing topic {parsed_element.id}: {str(e)}</div>",
                    element_id=parsed_element.id,
                    metadata={}
                )

    def _process_todo_lists(self, soup: BeautifulSoup) -> None:
        """Process markdown todo lists into HTML checkboxes."""
        for li in soup.find_all('li'):
            text = li.get_text()
            if text.startswith('[ ]') or text.startswith('[x]'):
                is_checked = text.startswith('[x]')
                text = text[3:].strip()  # Remove the checkbox markdown

                # Create checkbox input
                checkbox = soup.new_tag('input', type='checkbox')
                if is_checked:
                    checkbox['checked'] = ''

                # Create label
                label = soup.new_tag('label')
                label.string = text

                # Replace content
                li.clear()
                li['class'] = li.get('class', []) + ['todo-item']
                li.append(checkbox)
                li.append(label)


    def _process_images(self, soup: BeautifulSoup) -> None:
        """Process markdown images."""
        try:
            for img in soup.find_all('img'):
                src = img.get('src', '')
                if not src:
                    continue

                # Handle relative paths
                if not src.startswith(('http://', 'https://')):
                    if src.startswith('media/'):
                        # For topic-specific media files
                        img['src'] = f"/dita/topics/cs50/{src}"
                    else:
                        # For other media files
                        img['src'] = f"/dita/topics/{src}"

                # Add responsive classes
                img['class'] = (img.get('class', []) + ['img-fluid']).strip()

                # Add figure wrapper if there's alt text
                if img.get('alt'):
                    figure = soup.new_tag('figure', attrs={'class': 'figure'})
                    figcaption = soup.new_tag('figcaption', attrs={'class': 'figure-caption'})
                    figcaption.string = img['alt']
                    img.wrap(figure)
                    figure.append(figcaption)

        except Exception as e:
            self.logger.error(f"Error processing images: {str(e)}")

    def _process_html_elements(self, soup: BeautifulSoup) -> None:
        """
        Processes individual HTML elements using element processors.

        Args:
            soup (BeautifulSoup): Parsed HTML content.
        """
        for tag in soup.find_all():
            element_info = self.content_processor.process_element(tag)
            if element_info.type == MDElementType.HEADING:
                tag["id"] = self.heading_handler.generate_id(tag.text)

    def _process_code_blocks(self, soup: BeautifulSoup) -> None:
        """Process code blocks with consistent styling."""
        for pre in soup.find_all('pre'):
            code = pre.find('code')
            if not code:
                continue

            # Get language class
            lang_class = next((c for c in code.get('class', [])
                              if c.startswith('language-')), '')
            language = lang_class.replace('language-', '') if lang_class else ''

            # Create wrapper
            wrapper = soup.new_tag('div', attrs={'class': 'code-block-wrapper'})
            pre.wrap(wrapper)

            # Add language label
            if language:
                label = soup.new_tag('div', attrs={'class': 'code-label'})
                label.string = language
                wrapper.insert(0, label)

            # Style pre and code
            pre['class'] = pre.get('class', []) + ['code-block', 'highlight']
            if language:
                pre['data-language'] = language
                code['class'] = code.get('class', []) + [f'language-{language}']
