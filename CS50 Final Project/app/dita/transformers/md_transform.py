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
    ProcessingContext,
    ProcessedContent
)
from app_config import DITAConfig
from ..utils.html_helpers import HTMLHelper
from ..utils.heading import HeadingHandler
from app.dita.processors.content_processors import ContentProcessor
from app.dita.processors.md_elements import MarkdownElementProcessor
from app.dita.utils.latex.latex_processor import LaTeXProcessor
from app.dita.utils.latex.latex_processor import KaTeXRenderer
from app.dita.transformers.base_transformer import BaseTransformer


class MarkdownTransformer(BaseTransformer):
    def __init__(self, root_path: Path):
        super().__init__(root_path)
        self.logger = logging.getLogger(__name__)
        self.root_path = root_path
        self.html_helper = HTMLHelper(root_path)
        self.heading_handler = HeadingHandler()

        # Initialize content processor first
        content_processor = ContentProcessor(
            dita_root=root_path,
            markdown_root=root_path
        )

        # Pass content processor to markdown element processor
        self.content_processor = MarkdownElementProcessor(content_processor)

        # Initialize LaTeX pipeline components
        self.latex_processor = LaTeXProcessor()
        self.katex_renderer = KaTeXRenderer()

        # Initialize markdown with core extensions only
        self.extensions = [
            'markdown.extensions.fenced_code',
            'markdown.extensions.tables'
        ]

        # Element transformers mapping
        self._element_transformers = {
            MDElementType.HEADING: self._transform_heading,
            MDElementType.PARAGRAPH: self._transform_paragraph,
            MDElementType.LINK: self._transform_link,
            MDElementType.IMAGE: self._transform_image,
            MDElementType.CODE_BLOCK: self._transform_code_block,
            MDElementType.BLOCKQUOTE: self._transform_blockquote,
            MDElementType.TODO: self._transform_todo,
            MDElementType.FOOTNOTE: self._transform_footnote,
            MDElementType.YAML_METADATA: self._transform_metadata,
            MDElementType.UNORDERED_LIST: self._transform_list,
            MDElementType.ORDERED_LIST: self._transform_list,
            MDElementType.LIST_ITEM: self._transform_list_item,
            MDElementType.TABLE: self._transform_table,
            MDElementType.BOLD: self._transform_bold,
            MDElementType.ITALIC: self._transform_italic
        }


    def configure(self, config: DITAConfig) -> None:
        """Configure transformer with settings and handlers."""
        try:
            self.logger.debug("Configuring MarkdownTransformer")

            # Configure handlers
            self.heading_handler.set_numbering_enabled(config.number_headings)
            self.html_helper.configure_helper(config)
            self.id_handler.configure(config)
            self.metadata_handler.configure(config)

            # Configure processing features
            self.enable_latex = config.process_latex
            self.enable_numbering = config.number_headings
            self.enable_cross_refs = config.enable_cross_refs
            self.show_toc = config.show_toc

            self.logger.debug(
                f"Configured with features: latex={self.enable_latex}, "
                f"numbering={self.enable_numbering}, cross_refs={self.enable_cross_refs}"
            )

        except Exception as e:
            self.logger.error(f"MarkdownTransformer configuration failed: {str(e)}")
            raise

    def transform_topic(
            self,
            parsed_element: ParsedElement,
            context: ProcessingContext,
            html_converter: Optional[Callable[[str, ProcessingContext], str]] = None
        ) -> ProcessedContent:
        """Transform parsed Markdown to HTML with full pipeline integration."""
        try:
            self.logger.debug(f"Transforming Markdown topic: {parsed_element.topic_path}")

            # Extract metadata
            metadata = self.metadata_handler.extract_metadata(
                parsed_element.topic_path,
                parsed_element.id
            )

            # Convert Markdown to initial HTML
            html_content = self._convert_markdown_to_html(parsed_element.content, context)

            # Parse with BeautifulSoup for processing
            soup = BeautifulSoup(html_content, 'html.parser')

            # Process headings first
            self.heading_handler.start_new_topic()
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                level = int(heading.name[1])
                heading_id, numbered_text = self.heading_handler.process_heading(
                    heading.get_text(),
                    level,
                    is_topic_title=(level == 1 and 'title' in heading.get('class', []))
                )
                heading['id'] = heading_id
                heading.string = numbered_text

            # Process other elements
            self.process_elements(soup, parsed_element)

            # Process LaTeX if enabled
            if self.enable_latex and metadata.get('has_latex'):
                html_content = self._process_latex(str(soup))
                soup = BeautifulSoup(html_content, 'html.parser')

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


    def _transform_heading(self, element_info: MDElementInfo) -> str:
        level = element_info.level or 1
        heading_id, numbered_text = self.heading_handler.process_heading(
            element_info.content, level
        )
        return f'<h{level} id="{heading_id}">{numbered_text}<a href="#{heading_id}" class="heading-anchor">¶</a></h{level}>'

    def _transform_link(self, element_info: MDElementInfo) -> str:
        href = element_info.attributes.custom_attrs.get('href', '')
        classes = ' '.join(element_info.attributes.classes)
        target = '_blank' if href.startswith(('http://', 'https://')) else None
        target_attr = f' target="{target}"' if target else ''
        return f'<a href="{href}" class="{classes}"{target_attr}>{element_info.content}</a>'

    def _transform_image(self, element_info: MDElementInfo) -> str:
        src = element_info.attributes.custom_attrs.get('src', '')
        alt = element_info.attributes.custom_attrs.get('alt', '')
        title = element_info.attributes.custom_attrs.get('title', '')
        classes = ' '.join(['img-fluid', *element_info.attributes.classes])

        img_html = f'<img src="{src}" alt="{alt}" class="{classes}"'
        if title:
            img_html += f' title="{title}"'
        img_html += ' />'

        if alt:  # Wrap in figure if there's alt text
            return f'<figure class="figure">{img_html}<figcaption class="figure-caption">{alt}</figcaption></figure>'
        return img_html

    def _transform_code_block(self, element_info: MDElementInfo) -> str:
        language = element_info.metadata.get('code_info', {}).get('language', '')
        content = html.escape(element_info.content)

        if language == 'mermaid':
            return f'<div class="mermaid">{content}</div>'

        lang_label = f'<div class="code-label">{language}</div>' if language else ''
        return (
            f'<div class="code-block-wrapper">'
            f'{lang_label}'
            f'<pre class="code-block" data-language="{language}">'
            f'<code class="language-{language}">{content}</code>'
            f'</pre></div>'
        )

    def _transform_todo(self, element_info: MDElementInfo) -> str:
        is_checked = element_info.metadata.get('todo_info', {}).get('is_checked', False)
        checked_attr = 'checked' if is_checked else ''
        return (
            f'<div class="todo-item">'
            f'<input type="checkbox" {checked_attr} disabled />'
            f'<label>{element_info.content}</label>'
            f'</div>'
        )

    def _transform_list(self, element_info: MDElementInfo) -> str:
        tag = 'ol' if element_info.type == MDElementType.ORDERED_LIST else 'ul'
        classes = ' '.join(element_info.attributes.classes)
        return f'<{tag} class="{classes}">{element_info.content}</{tag}>'


    def _process_latex(self, content: str) -> str:
        """Process LaTeX equations in content."""
        try:
            # Extract equations
            block_pattern = r'\$\$(.*?)\$\$'
            inline_pattern = r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)'

            equations: List[LaTeXEquation] = []

            # Process block equations
            for i, match in enumerate(re.finditer(block_pattern, content, re.DOTALL)):
                equations.append(LaTeXEquation(
                    id=f'eq-block-{i}',
                    content=match.group(1).strip(),
                    is_block=True
                ))

            # Process inline equations
            for i, match in enumerate(re.finditer(inline_pattern, content)):
                equations.append(LaTeXEquation(
                    id=f'eq-inline-{i}',
                    content=match.group(1).strip(),
                    is_block=False
                ))

            # Process equations through LaTeX pipeline
            processed = self.latex_processor.process_equations(equations)

            # Replace equations in content
            for equation in processed:
                if equation.is_block:
                    content = content.replace(
                        f'$${equation.original}$$',
                        self.katex_renderer.render_equation(equation)
                    )
                else:
                    content = content.replace(
                        f'${equation.original}$',
                        self.katex_renderer.render_equation(equation)
                    )

            return content

        except Exception as e:
            self.logger.error(f"LaTeX processing failed: {str(e)}")
            return content
