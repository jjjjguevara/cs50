# app/dita/transformers/md_transform.py

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Callable
from bs4 import NavigableString, Tag
import html
import markdown
from bs4 import BeautifulSoup, Tag
from ..models.types import (
    MapContext,
    ParsedElement,
    ProcessedContent,
    MDElementInfo,
    MDElementType,
    HeadingContext,
    ProcessingContext,
    HeadingReference,
    LaTeXEquation,
    ProcessingError
)

from app_config import DITAConfig
from ..utils.html_helpers import HTMLHelper
from ..utils.heading import HeadingHandler
from ..utils.metadata import MetadataHandler
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
        self.metadata_handler = MetadataHandler()

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
            MDElementType.LINK: self._transform_link,
            MDElementType.IMAGE: self._transform_image,
            MDElementType.TODO: self._transform_todo,
            MDElementType.ORDERED_LIST: self._transform_list,
            MDElementType.UNORDERED_LIST: self._transform_list,
            MDElementType.HEADING: self._transform_heading,
            MDElementType.CODE_BLOCK: self._transform_code_block,
            MDElementType.PARAGRAPH: self._transform_paragraph,
            MDElementType.BLOCKQUOTE: self._transform_blockquote,
            MDElementType.FOOTNOTE: self._transform_footnote,
            MDElementType.YAML_METADATA: self._transform_metadata,
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

            # Add table of contents if enabled
            if self.show_toc:
                html_content = self.html_helper.render_headings(str(soup), True)
            else:
                html_content = str(soup)

            # Process final HTML
            final_html = self.html_helper.process_final_content(html_content)

            return ProcessedContent(
                html=final_html,
                element_id=parsed_element.id,
                metadata={
                    **metadata,
                    'processed_at': datetime.now().isoformat()
                }
            )

        except Exception as e:
            self.logger.error(f"Error transforming topic: {str(e)}")
            raise

    def process_elements(
       self,
       soup: BeautifulSoup,
       parsed_element: ParsedElement
    ) -> None:
       """
       Process HTML elements using the Markdown content processor.

       Args:
           soup: BeautifulSoup object containing parsed HTML
           parsed_element: Original parsed element for context
       """
       try:
           # Process elements by type using content processor
           for element in soup.find_all(True):
               element_info = self.content_processor.process_element(
                   element,
                   source_path=parsed_element.topic_path
               )

               # Apply processed attributes
               if element_info.attributes.id:
                   element['id'] = element_info.attributes.id

               if element_info.attributes.classes:
                   element['class'] = ' '.join(element_info.attributes.classes)

               for key, value in element_info.attributes.custom_attrs.items():
                   element[key] = value

               # Transform element based on type
               if element_info.type in self._element_transformers:
                   transform_method = self._element_transformers[element_info.type]
                   transformed_html = transform_method(element_info)

                   if transformed_html:
                       new_element = BeautifulSoup(transformed_html, 'html.parser')
                       element.replace_with(new_element)

               # Process special elements
               if element.name == 'img':
                   src = element.get('src', '')
                   if src:
                       element['src'] = self.html_helper.resolve_image_path(
                           src,
                           parsed_element.topic_path
                       )

               # Process code elements
               elif element.name == 'code':
                   parent = element.find_parent('pre')
                   if parent:
                       self._process_code_block(element, parent)
                   else:
                       self._process_code_phrase(element)

               # Process table elements
               elif element.name in ['th', 'td', 'tr']:
                   self._process_table_element(element, element_info.type)

       except Exception as e:
           self.logger.error(f"Error processing elements: {str(e)}")
           raise

    def _apply_heading_attributes(
        self,
        element: Tag,
        level: int,
        heading_info: MDElementInfo,
        context: ProcessingContext
    ) -> None:
        """
        Apply heading-specific attributes and numbering.

        Args:
            element: The heading element
            level: Heading level (1-6)
            heading_info: Processed heading information
            context: Current processing context
        """
        try:
            # Get heading ID and numbered text
            heading_id, numbered_heading = self.heading_handler.process_heading(
                text=element.get_text().strip(),
                level=level,
                is_topic_title=(level == 1 and heading_info.metadata.get('is_title', False))
            )

            # Update element attributes
            element['id'] = heading_id
            element.string = numbered_heading

            # Add heading classes
            default_classes = ['heading', f'heading-{level}']
            if heading_info.attributes.classes:
                default_classes.extend(heading_info.attributes.classes)
            element['class'] = ' '.join(default_classes)

            # Create and add anchor link
            soup = BeautifulSoup('', 'html.parser')
            anchor = soup.new_tag('a', href=f"#{heading_id}", attrs={'class': 'heading-anchor'})
            anchor.string = '¶'
            element.append(anchor)

            # Track cross-reference information
            self.heading_handler.add_heading_reference(
                HeadingReference(
                    id=heading_id,
                    text=numbered_heading,
                    level=level,
                    topic_id=heading_info.context.parent_id,
                    map_id=context.map_context.base.id if context and hasattr(context, 'map_context') else None
                )
            )

        except Exception as e:
            self.logger.error(f"Error applying heading attributes: {str(e)}")
            raise


    def _apply_latex_attributes(self, element: Tag, element_info: MDElementInfo) -> None:
        """
        Apply LaTeX-specific attributes to equation elements.

        Args:
            element: BeautifulSoup tag representing equation
            element_info: Processed element information
        """
        try:
            from app.dita.models.types import LaTeXEquation

            is_block = element_info.context.element_type == 'block'
            display_class = 'katex-display' if is_block else 'katex-inline'

            # Set element classes
            classes = [display_class]
            if element_info.attributes.classes:
                classes.extend(element_info.attributes.classes)
            element['class'] = ' '.join(classes)

            # Get equation content
            content = element.string or ''

            # Add proper delimiters if missing
            if is_block and not content.startswith('$$'):
                content = f'$${content}$$'
            elif not is_block and not content.startswith('$'):
                content = f'${content}$'

            # Create LaTeX equation object
            equation = LaTeXEquation(
                id=element_info.attributes.id,
                content=content.strip('$'),
                is_block=is_block,
                metadata=element_info.metadata
            )

            # Store for later processing
            if not hasattr(self, '_latex_equations'):
                self._latex_equations = []
            self._latex_equations.append(equation)

            # Store original content
            element['data-latex-original'] = content
            element['data-equation-id'] = equation.id

        except Exception as e:
            self.logger.error(f"Error applying LaTeX attributes: {str(e)}")

    def _process_latex_equations(self, soup: BeautifulSoup) -> None:
        """Process LaTeX equations for KaTeX rendering."""
        try:
            if not hasattr(self, '_latex_equations') or not self._latex_equations:
                return

            # Process collected equations through LaTeX pipeline
            processed_equations = self.latex_processor.process_equations(self._latex_equations)

            # Replace equations in soup with rendered versions
            for processed in processed_equations:
                element = soup.find(attrs={'data-equation-id': processed.id})
                if element:
                    # Create LaTeXEquation from processed data
                    latex_equation = LaTeXEquation(
                        id=processed.id,
                        content=processed.original,
                        is_block=processed.is_block,
                        metadata={}
                    )
                    rendered = self.katex_renderer.render_equation(latex_equation)
                    new_element = BeautifulSoup(rendered, 'html.parser')
                    element.replace_with(new_element)

            # Clear processed equations
            self._latex_equations = []

        except Exception as e:
            self.logger.error(f"Error processing LaTeX equations: {str(e)}")


    def _process_todo_lists(self, soup: BeautifulSoup) -> None:
       """Process markdown todo lists into HTML checkboxes."""
       try:
           for li in soup.find_all('li'):
               text = li.get_text().strip()
               if text.startswith('[ ]') or text.startswith('[x]'):
                   # Create new list item with todo styling
                   is_checked = text.startswith('[x]')
                   text = text[3:].strip()  # Remove checkbox markdown

                   # Create checkbox input
                   checkbox = soup.new_tag('input', type='checkbox')
                   if is_checked:
                       checkbox['checked'] = ''
                   checkbox['disabled'] = ''  # Make checkbox read-only

                   # Create label
                   label = soup.new_tag('label')
                   label.string = text

                   # Clear and update list item
                   li.clear()
                   li['class'] = ' '.join(['todo-item', *li.get('class', [])])
                   li.append(checkbox)
                   li.append(label)

       except Exception as e:
           self.logger.error(f"Error processing todo lists: {str(e)}")


    def _process_images(self, soup: BeautifulSoup, topic_path: Path) -> None:
       """
       Process markdown images with proper path resolution.
       Images are expected in topic's media subdirectory.
       """
       try:
           topic_dir = topic_path.parent
           media_dir = topic_dir / 'media'

           for img in soup.find_all('img'):
               src = img.get('src', '')
               if not src:
                   continue

               if not src.startswith(('http://', 'https://')):
                   # For topic-specific media files
                   if src.startswith('media/'):
                       src = src.replace('media/', '')

                   # Construct path relative to media directory
                   img_path = (media_dir / src).resolve()

                   if img_path.exists():
                       # Make path relative to dita_root for serving
                       relative_path = img_path.relative_to(self.root_path)
                       img['src'] = f'/static/topics/{relative_path}'
                   else:
                       self.logger.warning(f"Image not found: {img_path}")

               # Add responsive classes
               img['class'] = ' '.join(['img-fluid', *img.get('class', [])])

               # Add figure wrapper if there's alt text
               if alt_text := img.get('alt'):
                   # Create figure wrapper
                   figure = soup.new_tag('figure')
                   figure['class'] = 'figure'

                   # Create caption
                   figcaption = soup.new_tag('figcaption')
                   figcaption['class'] = 'figure-caption'
                   figcaption.string = alt_text

                   # Wrap image and add caption
                   img.wrap(figure)
                   figure.append(figcaption)

       except Exception as e:
           self.logger.error(f"Error processing images: {str(e)}")


    def _process_code_block(self, code_elem: Tag, pre_elem: Tag) -> None:
        """Handle code block elements."""
        language = ''
        for cls in code_elem.get('class', []):
            if cls.startswith('language-'):
                language = cls.replace('language-', '')
                break

        if language:
            pre_elem['data-language'] = language
            code_elem['class'] = f'language-{language}'

    def _process_code_phrase(self, element: Tag) -> None:
        """Handle inline code elements."""
        element['class'] = ' '.join(['code-inline', *element.get('class', [])])

    def _process_table_element(self, element: Tag, element_type: MDElementType) -> None:
        """Handle table-specific elements."""
        if element_type == MDElementType.TABLE_HEADER:
            element['scope'] = 'col'
            element['class'] = ' '.join(['th', *element.get('class', [])])
        elif element_type == MDElementType.TABLE_CELL:
            element['class'] = ' '.join(['td', *element.get('class', [])])
        elif element_type == MDElementType.TABLE_ROW:
            element['class'] = ' '.join(['tr', *element.get('class', [])])


    def _process_code_blocks(self, soup: BeautifulSoup) -> None:
       """
       Process code blocks with consistent styling and language handling.
       """
       try:
           for pre in soup.find_all('pre'):
               code = pre.find('code')
               if not code:
                   continue

               # Get language class
               lang_class = None
               for cls in code.get('class', []):
                   if cls.startswith('language-'):
                       lang_class = cls
                       break

               language = lang_class.replace('language-', '') if lang_class else ''

               # Create wrapper div
               wrapper = soup.new_tag('div', attrs={'class': 'code-block-wrapper'})
               pre.wrap(wrapper)

               # Add language label if specified
               if language:
                   label = soup.new_tag('div', attrs={'class': 'code-label'})
                   label.string = language
                   wrapper.insert(0, label)

                   # Add language attributes
                   pre['data-language'] = language
                   pre['class'] = ' '.join(['code-block', 'highlight', *pre.get('class', [])])
                   if not lang_class in code.get('class', []):
                       code['class'] = ' '.join([f'language-{language}', *code.get('class', [])])

               # Handle Mermaid diagrams specially
               if language == 'mermaid':
                   pre['class'] = ' '.join(['mermaid', *pre.get('class', [])])
                   wrapper['class'] = ' '.join(['mermaid-wrapper', *wrapper.get('class', [])])

               # Add copy button
               copy_btn = soup.new_tag('button', attrs={
                   'class': 'copy-code-button',
                   'aria-label': 'Copy code to clipboard'
               })
               copy_btn.string = 'Copy'
               wrapper.insert(1, copy_btn)

       except Exception as e:
           self.logger.error(f"Error processing code blocks: {str(e)}")

    def _process_html_elements(self, soup: BeautifulSoup, context: ProcessingContext) -> None:
       """
       Process individual HTML elements using element processors.

       Args:
           soup (BeautifulSoup): Parsed HTML content.
           context (ProcessingContext): Current processing context
       """
       try:
           for tag in soup.find_all(True):
               element_info = self.content_processor.process_element(tag)

               # Apply element attributes
               if element_info.attributes.id:
                   tag['id'] = element_info.attributes.id

               if element_info.attributes.classes:
                   existing_classes = tag.get('class', [])
                   if isinstance(existing_classes, str):
                       existing_classes = existing_classes.split()
                   tag['class'] = ' '.join([*existing_classes, *element_info.attributes.classes])

               # Process custom attributes
               for key, value in element_info.attributes.custom_attrs.items():
                   tag[key] = value

               # Special handling for element types
               if element_info.type == MDElementType.HEADING:
                   self._apply_heading_attributes(tag, element_info.level or 1, element_info, context)
               elif element_info.type == MDElementType.CODE_BLOCK:
                   # Find parent pre tag if it exists
                   pre_parent = tag.find_parent('pre')
                   if pre_parent:
                       self._process_code_block(tag, pre_parent)
                   else:
                       self.logger.warning(f"Code block element without pre parent: {tag}")
               elif element_info.type == MDElementType.CODE_PHRASE:
                   self._process_code_phrase(tag)

       except Exception as e:
           self.logger.error(f"Error processing HTML elements: {str(e)}")

    def _convert_markdown_to_html(
           self,
           content: str,
           context: ProcessingContext
       ) -> str:
       """
       Convert Markdown content to HTML with proper context.

       Args:
           content: Raw markdown content
           context: Processing context

       Returns:
           str: Initial HTML conversion
       """
       try:
           # Convert using core extensions
           html_content = markdown.markdown(
               content,
               extensions=self.extensions
           )

           # Initial parse
           soup = BeautifulSoup(html_content, 'html.parser')

           # Process elements with context
           self._process_html_elements(soup, context)

           # Handle special processors
           self._process_images(soup, context.get_current_topic_context().base.file_path)
           self._process_todo_lists(soup)
           self._process_code_blocks(soup)

           return str(soup)

       except Exception as e:
           self.logger.error(f"Error converting markdown to HTML: {str(e)}")
           return f"<div class='error'>Error converting content: {str(e)}</div>"

    def _convert_to_html(self, content: str) -> str:
       """
       Custom HTML conversion with LaTeX and element handling.

       Args:
           content: Content to convert

       Returns:
           str: Processed HTML content
       """
       try:
           # Process LaTeX if needed
           if '$$' in content or '$' in content:
               equations = self._extract_latex_equations(content)
               if equations:
                   content = self._process_latex(content, equations)

           # Convert Markdown
           html_content = markdown.markdown(
               content,
               extensions=self.extensions
           )

           soup = BeautifulSoup(html_content, 'html.parser')

           # Create wrapper with proper class
           wrapper = soup.new_tag('div')
           wrapper['class'] = 'markdown-content'
           if '$$' in content or '$' in content:
               wrapper['class'] = ' '.join([wrapper['class'], 'katex-content'])

           # Move all content into wrapper
           for child in soup.children:
               wrapper.append(child)

           return str(wrapper)

       except Exception as e:
           self.logger.error(f"Error converting to HTML: {str(e)}")
           raise ProcessingError(
               error_type="conversion",
               message=f"HTML conversion failed: {str(e)}",
               context="markdown_conversion"
           )


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
