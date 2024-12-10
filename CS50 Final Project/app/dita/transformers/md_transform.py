# app/dita/transformers/md_transform.py

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Callable
from bs4 import NavigableString, Tag
import html
import markdown
from bs4 import BeautifulSoup
from ..models.types import (
    ParsedElement,
    ProcessedContent,
    MDElementInfo,
    MDElementType,
    HeadingContext,
    ProcessingContext
)
from app_config import DITAConfig
from ..utils.html_helpers import HTMLHelper
from ..utils.heading import HeadingHandler
from app.dita.processors.md_elements import MarkdownContentProcessor
from app.dita.utils.latex.latex_processor import LaTeXProcessor
from app.dita.transformers.base_transformer import BaseTransformer


class MarkdownTransformer(BaseTransformer):
    def __init__(self, root_path: Path):
        super().__init__(root_path)
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

        # Fix extension configs to match LaTeXExtension's expected config
        self.extension_configs = {
            'app.dita.utils.markdown.latex_extension': {
                'enable_numbering': False,
                'enable_references': False,
                'preserve_delimiters': True
            }
        }
    def configure(self, config: DITAConfig):
            """
            Apply additional configuration settings.
            """
            self.logger.debug("Configuring MarkdownTransformer")
            # Example: Add custom configuration logic
            self.some_markdown_setting = getattr(config, 'markdown_setting', None)

    def transform_topic(
            self,
            parsed_element: ParsedElement,
            context: ProcessingContext,
            html_converter: Optional[Callable[[str, ProcessingContext], str]] = None
        ) -> ProcessedContent:
        """
        Transform Markdown topic to HTML with proper context handling.

        Args:
            parsed_element: The parsed element to transform
            context: Processing context for transformation
            to_html_func: Optional custom HTML conversion function

        Returns:
            ProcessedContent: The transformed content with metadata
        """
        try:
            self.logger.debug(f"Transforming Markdown topic: {parsed_element.topic_path}")

            # Start new topic for heading tracking
            self.heading_handler.start_new_topic()

            # Initialize content parts
            content = parsed_element.content
            html_parts = []

            # Process frontmatter if present
            if content.startswith('---'):
                end_index = content.find('---', 3)
                if end_index != -1:
                    frontmatter = content[0:end_index+3]
                    content = content[end_index+3:].strip()
                    html_parts.append(
                        f'<div class="code-block-wrapper">'
                        f'<div class="code-label">yaml</div>'
                        f'<pre class="code-block yaml-frontmatter">'
                        f'<code>{html.escape(frontmatter)}</code></pre>'
                        f'</div>'
                    )

            # Use markdown-specific converter if none provided
            converter = html_converter or self._convert_markdown_to_html

            # Call base transformer with our converter
            return super().transform_topic(
                parsed_element=parsed_element,
                context=context,
                html_converter=converter
            )

            try:
                # Convert content with context
                html_content = html_converter(content, context)

                # Parse with BeautifulSoup for processing
                soup = BeautifulSoup(html_content, 'html.parser')

                # Process elements
                self.process_elements(soup, parsed_element.topic_path)

                # Add frontmatter if present
                if html_parts:
                    frontmatter_div = BeautifulSoup('\n'.join(html_parts), 'html.parser')
                    soup.insert(0, frontmatter_div)

                # Wrap in container with necessary classes
                wrapper = soup.new_tag('div')
                wrapper['class'] = 'markdown-content'
                if parsed_element.metadata.get('has_latex'):
                    wrapper['class'] += ' katex-content'

                # Move all content to wrapper
                for child in soup.children:
                    wrapper.append(child)

                # Create final processed content
                return ProcessedContent(
                    html=str(wrapper),
                    element_id=parsed_element.id,
                    metadata={
                        **parsed_element.metadata,
                        'content_type': 'markdown',
                        'processed_at': datetime.now().isoformat(),
                        'topic_path': str(parsed_element.topic_path)
                    }
                )

            except Exception as e:
                self.logger.error(f"Error in HTML conversion: {str(e)}")
                raise

        except Exception as e:
            self.logger.error(f"Error transforming Markdown topic: {str(e)}")
            return ProcessedContent(
                html=f"<div class='error'>Error processing topic {parsed_element.id}: {str(e)}</div>",
                element_id=parsed_element.id,
                metadata={
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'processed_at': datetime.now().isoformat()
                }
            )


    def process_elements(
            self,
            soup: BeautifulSoup,
            parsed_element: ParsedElement
        ) -> None:
        """
        Process HTML elements using the Markdown content processor.
        Delegates actual element processing to MarkdownContentProcessor.

        Args:
            soup: BeautifulSoup object containing parsed HTML
            topic_path: Path to the source topic file
        """
        try:
            for element in soup.find_all(True):
                element_info = self.content_processor.process_element(
                    element,
                    parsed_element.topic_path
                )

                # Apply processed attributes
                if element_info.attributes.id:
                    element['id'] = element_info.attributes.id
                if element_info.attributes.classes:
                    element['class'] = ' '.join(element_info.attributes.classes)
                for key, value in element_info.attributes.custom_attrs.items():
                    element[key] = value

                # Handle specific element types
                if element_info.type == MDElementType.HEADING:
                    self._apply_heading_attributes(
                        element=element,
                        level=element_info.level or int(element.name[1]),
                        heading_info=element_info
                    )

        except Exception as e:
            self.logger.error(f"Error processing elements: {str(e)}")
            raise


    def _apply_heading_attributes(
        self,
        element: Tag,
        level: int,
        heading_info: MDElementInfo
    ) -> None:
        """
        Apply heading-specific attributes and numbering.

        Args:
            element: The heading element
            level: Heading level (1-6)
            heading_info: Processed heading information
        """
        try:
            # Get heading ID and numbered text from heading handler
            heading_id, numbered_heading = self.heading_handler.process_heading(
                text=element.text.strip(),
                level=level
            )

            # Update element
            element['id'] = heading_id
            element.string = numbered_heading

            # Add pilcrow (section marker)
            soup = BeautifulSoup('', 'html.parser')
            pilcrow = soup.new_tag('a', href=f"#{heading_id}")
            pilcrow['class'] = 'pilcrow'
            pilcrow.string = '¶'
            element.append(pilcrow)

        except Exception as e:
            self.logger.error(f"Error applying heading attributes: {str(e)}")
            raise

    def _apply_latex_attributes(self, element: Tag, element_info: MDElementInfo) -> None:
        """Apply LaTeX-specific attributes."""
        try:
            is_block = element_info.context.element_type == 'block'
            element['class'] = 'katex-display' if is_block else 'katex-inline'
            content = element.string or ''
            if is_block and not content.startswith('$$'):
                element.string = f'$${content}$$'
            elif not is_block and not content.startswith('$'):
                element.string = f'${content}$'
        except Exception as e:
            self.logger.error(f"Error applying LaTeX attributes: {str(e)}")


    def _process_latex_equations(self, soup: BeautifulSoup) -> None:
        """Process LaTeX equations for KaTeX rendering."""
        try:
            # Process block equations (already wrapped by LaTeX extension)
            for equation in soup.find_all('div', class_='math-block'):
                # Ensure proper KaTeX display mode class
                equation['class'] = 'katex-display'
                # Preserve original LaTeX content
                content = equation.string.strip()
                if not content.startswith('$$'):
                    equation.string = f'$${content}$$'

            # Process inline equations (already wrapped by LaTeX extension)
            for equation in soup.find_all('span', class_='math-inline'):
                # Ensure proper KaTeX inline class
                equation['class'] = 'katex-inline'
                # Preserve original LaTeX content
                content = equation.string.strip()
                if not content.startswith('$'):
                    equation.string = f'${content}$'

            # Add data attributes for debugging if needed
            for eq in soup.find_all(['div', 'span'], class_=['katex-display', 'katex-inline']):
                eq['data-latex-original'] = eq.string.strip()
                eq['data-processed'] = 'true'

            self.logger.debug(
                f"Processed {len(soup.find_all('div', class_='katex-display'))} display equations and "
                f"{len(soup.find_all('span', class_='katex-inline'))} inline equations"
            )

        except Exception as e:
            self.logger.error(f"Error processing LaTeX equations: {str(e)}")
            raise

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


    def _convert_markdown_to_html(
            self,
            content: str,
            context: ProcessingContext
        ) -> str:
            """Convert Markdown content to HTML."""
            return markdown.markdown(
                content,
                extensions=self.extensions,
                extension_configs=self.extension_configs
            )

    def _convert_to_html(self, content: str) -> str:
        """Custom HTML conversion with LaTeX handling."""
        try:
            # Convert Markdown to HTML with proper config
            html_content = markdown.markdown(
                content,
                extensions=self.extensions,
                extension_configs=self.extension_configs,
                output_format='html'
            )

            soup = BeautifulSoup(html_content, 'html.parser')

            # Process equations (rest of the method remains the same)
            for block_eq in soup.find_all('div', class_='katex-block'):
                content = block_eq.string
                if content:
                    # Ensure proper delimiters
                    if not content.startswith('$$'):
                        content = f'$${content}$$'
                    block_eq.string = content
                    block_eq['class'] = 'katex-display'

            for inline_eq in soup.find_all('span', class_='katex-inline'):
                content = inline_eq.string
                if content:
                    # Ensure proper delimiters
                    if not content.startswith('$'):
                        content = f'${content}$'
                    inline_eq.string = content

            # Create wrapper div with proper class
            wrapper = soup.new_tag('div')
            wrapper['class'] = 'katex-content'

            # Move all content into wrapper
            for child in soup.children:
                wrapper.append(child)

            # Replace soup contents with wrapped version
            soup.clear()
            soup.append(wrapper)

            return str(soup)
        except Exception as e:
            self.logger.error(f"Error converting to HTML: {str(e)}")
            raise
