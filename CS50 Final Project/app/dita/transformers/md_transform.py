# app/dita/transformers/md_transform.py


import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Callable, List
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
import re
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
            MDElementType.TODO: self._transform_list,
            MDElementType.ORDERED_LIST: self._transform_list,
            MDElementType.UNORDERED_LIST: self._transform_list,
            MDElementType.LIST_ITEM: self._transform_list,
            MDElementType.HEADING: self._transform_heading,
            MDElementType.BLOCKQUOTE: self._transform_blockquote,
            MDElementType.CODE_BLOCK: self._transform_code_block,
            MDElementType.PARAGRAPH: self._transform_paragraph,
            MDElementType.FOOTNOTE: self._transform_footnote,
            MDElementType.TABLE: self._transform_table,
            MDElementType.TABLE_HEADER: self._transform_table,
            MDElementType.TABLE_ROW: self._transform_table,
            MDElementType.TABLE_CELL: self._transform_table,
            MDElementType.BOLD: self._transform_emphasis,
            MDElementType.ITALIC: self._transform_emphasis,
            MDElementType.UNDERLINE: self._transform_emphasis,
            MDElementType.STRIKETHROUGH: self._transform_emphasis,
            MDElementType.YAML_METADATA: self._transform_metadata,

            # METADATA USAGE
            # The transform metadata method should prepare the metadata to be served to another transformer method
            # For example, methods which inject metadata into HTML content objects for the front end:
            # `article metadata` tables, `Bibliography` tables, `Glossary` tables, etc.
            # The output of this method should therefore be a dictionary of metadata key-value pairs

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
            html_content = self._transform_markdown_to_html(parsed_element.content, context)

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
               # Get element info from content processor
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

               # Handle special path resolution for images
               elif element_info.type == MDElementType.IMAGE:
                   src = element.get('src', '')
                   if src:
                       element['src'] = self.html_helper.resolve_image_path(
                           src,
                           parsed_element.topic_path
                       )

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

    def _transform_heading(self, element_info: MDElementInfo) -> str:
           """
           Transform a heading element to HTML.

           Args:
               element_info: Processed heading information

           Returns:
               str: Transformed HTML
           """
           try:
               level = element_info.level or 1
               heading_id = element_info.attributes.id
               content = element_info.content
               classes = ' '.join(['heading', f'heading-{level}', *element_info.attributes.classes])

               return (
                   f'<h{level} id="{heading_id}" class="{classes}">'
                   f'{content}'
                   f'<a href="#{heading_id}" class="heading-anchor">¶</a>'
                   f'</h{level}>'
               )

           except Exception as e:
               self.logger.error(f"Error transforming heading: {str(e)}")
               return ""

    def _transform_paragraph(self, element_info: MDElementInfo) -> str:
       """
       Transform paragraph elements to HTML.

       Args:
           element_info: Processed paragraph information

       Returns:
           str: Transformed HTML
       """
       try:
           content = html.escape(element_info.content)
           classes = ' '.join(['paragraph', *element_info.attributes.classes])

           return (
               f'<p class="{classes}">'
               f'{content}'
               f'</p>'
           )

       except Exception as e:
           self.logger.error(f"Error transforming paragraph: {str(e)}")
           return ""

    def _transform_footnote(self, element_info: MDElementInfo) -> str:
       """
       Transform footnote elements to HTML, supporting Obsidian-style footnotes.
       Handles both inline references and footnote content sections.

       Args:
           element_info: Processed footnote information

       Returns:
           str: Transformed HTML
       """
       try:
           content = html.escape(element_info.content)
           footnote_id = element_info.attributes.id
           base_classes = ['footnote', *element_info.attributes.classes]

           # Handle inline reference (^1)
           if element_info.metadata.get('is_reference', False):
               ref_number = element_info.metadata.get('footnote_number', '')
               ref_classes = ' '.join(['footnote-ref', *base_classes])

               return (
                   f'<sup class="{ref_classes}">'
                   f'<a href="#fn-{footnote_id}" '
                   f'id="fnref-{footnote_id}" '
                   f'data-footnote-ref '
                   f'aria-describedby="footnote-label">'
                   f'{ref_number}'
                   f'</a>'
                   f'</sup>'
               )

           # Handle footnote section [[1]]
           section_classes = ' '.join(['footnote-section', *base_classes])
           return (
               f'<section id="fn-{footnote_id}" '
               f'class="{section_classes}" '
               f'data-footnote '
               f'role="doc-endnote">'
               f'{content}'
               f'<a href="#fnref-{footnote_id}" '
               f'class="footnote-backref" '
               f'data-footnote-backref '
               f'aria-label="Back to reference {footnote_id}">↩</a>'
               f'</section>'
           )

       except Exception as e:
           self.logger.error(f"Error transforming footnote: {str(e)}")
           return ""

    def _transform_emphasis(self, element_info: MDElementInfo) -> str:
       """
       Transform emphasis elements to HTML.
       Handles bold, italic, underline and strikethrough, including nested formatting.

       Args:
           element_info: Processed emphasis information

       Returns:
           str: Transformed HTML
       """
       try:
           content = html.escape(element_info.content)

           # Map element types to HTML tags and classes
           emphasis_map = {
               MDElementType.BOLD: ('strong', 'bold'),
               MDElementType.ITALIC: ('em', 'italic'),
               MDElementType.UNDERLINE: ('u', 'underline'),
               MDElementType.STRIKETHROUGH: ('del', 'strikethrough')
           }

           if element_info.type not in emphasis_map:
               return content

           tag, base_class = emphasis_map[element_info.type]
           classes = ' '.join([base_class, *element_info.attributes.classes])

           return f'<{tag} class="{classes}">{content}</{tag}>'

       except Exception as e:
           self.logger.error(f"Error transforming emphasis element: {str(e)}")
           return ""

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

    def _transform_image(self, element_info: MDElementInfo) -> str:
       """
       Transform an image element to HTML.

       Args:
           element_info: Processed image information

       Returns:
           str: Transformed HTML
       """
       try:
           src = element_info.attributes.custom_attrs.get('src', '')
           alt = element_info.attributes.custom_attrs.get('alt', '')
           title = element_info.attributes.custom_attrs.get('title', '')
           classes = ' '.join(['img-fluid', *element_info.attributes.classes])

           # Build image tag
           img = f'<img src="{src}" alt="{alt}" class="{classes}"'
           if title:
               img += f' title="{title}"'
           img += ' />'

           # Wrap in figure if there's alt text
           if alt:
               return (
                   f'<figure class="figure">'
                   f'{img}'
                   f'<figcaption class="figure-caption">{alt}</figcaption>'
                   f'</figure>'
               )

           return img

       except Exception as e:
           self.logger.error(f"Error transforming image: {str(e)}")
           return ""


    def _transform_table(self, element_info: MDElementInfo) -> str:
        """
        Transform table elements to HTML.
        Handles tables, headers, rows, and cells.

        Args:
            element_info: Processed table information

        Returns:
            str: Transformed HTML
        """
        try:
            content = html.escape(element_info.content)

            # Map element types to HTML configuration
            table_map = {
                MDElementType.TABLE: {
                    'tag': 'table',
                    'base_class': 'markdown-table',
                    'attrs': {'role': 'grid'}
                },
                MDElementType.TABLE_HEADER: {
                    'tag': 'th',
                    'base_class': 'table-header',
                    'attrs': {'scope': 'col'}
                },
                MDElementType.TABLE_ROW: {
                    'tag': 'tr',
                    'base_class': 'table-row',
                    'attrs': {}
                },
                MDElementType.TABLE_CELL: {
                    'tag': 'td',
                    'base_class': 'table-cell',
                    'attrs': {}
                }
            }

            if element_info.type not in table_map:
                return content

            config = table_map[element_info.type]
            tag = config['tag']
            classes = ' '.join([config['base_class'], *element_info.attributes.classes])

            # Build attributes string
            attrs = ' '.join([
                f'{key}="{value}"'
                for key, value in {
                    **config['attrs'],
                    **element_info.attributes.custom_attrs,
                    'class': classes
                }.items()
            ])

            return f'<{tag} {attrs}>{content}</{tag}>'

        except Exception as e:
            self.logger.error(f"Error transforming table element: {str(e)}")
            return ""

    def _transform_code_block(self, element_info: MDElementInfo) -> str:
       """
       Transform code elements (both blocks and phrases) to HTML.

       Args:
           element_info: Processed code element information

       Returns:
           str: Transformed HTML
       """
       try:
           # Handle inline code phrase
           if element_info.type == MDElementType.CODE_PHRASE:
               content = html.escape(element_info.content)
               classes = ' '.join(['code-inline', *element_info.attributes.classes])
               return f'<code class="{classes}">{content}</code>'

           # Handle code blocks
           content = html.escape(element_info.content)
           language = element_info.metadata.get('code_info', {}).get('language', '')

           # Special handling for Mermaid
           if language == 'mermaid':
               return (
                   f'<div class="mermaid-wrapper">'
                   f'<div class="mermaid">{content}</div>'
                   f'</div>'
               )

           # Build standard code block
           return (
               f'<div class="code-block-wrapper">'
               f'{f"<div class=\"code-label\">{language}</div>" if language else ""}'
               f'<pre class="code-block highlight" data-language="{language}">'
               f'<code class="language-{language}">{content}</code>'
               f'</pre>'
               f'<button class="copy-code-button" aria-label="Copy code">Copy</button>'
               f'</div>'
           )

       except Exception as e:
           self.logger.error(f"Error transforming code element: {str(e)}")
           return ""

    def _transform_blockquote(self, element_info: MDElementInfo) -> str:
       """
       Transform a blockquote element to HTML.

       Args:
           element_info: Processed blockquote information

       Returns:
           str: Transformed HTML
       """
       try:
           content = html.escape(element_info.content)
           classes = ' '.join(['blockquote', *element_info.attributes.classes])

           return (
               f'<blockquote class="{classes}">'
               f'<p>{content}</p>'
               f'</blockquote>'
           )

       except Exception as e:
           self.logger.error(f"Error transforming blockquote: {str(e)}")
           return ""


    def _process_html_elements(self, soup: BeautifulSoup, context: ProcessingContext) -> None:
       """
       Process individual HTML elements using element processors.

       Args:
           soup (BeautifulSoup): Parsed HTML content.
           context (ProcessingContext): Current processing context
       """
       try:
           for tag in soup.find_all(True):
               # Get element info from content processor
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

               # Transform element based on type
               if element_info.type in self._element_transformers:
                   transform_method = self._element_transformers[element_info.type]
                   transformed_html = transform_method(element_info)

                   if transformed_html:
                       new_element = BeautifulSoup(transformed_html, 'html.parser')
                       tag.replace_with(new_element)

       except Exception as e:
           self.logger.error(f"Error processing HTML elements: {str(e)}")

    def _transform_markdown_to_html(
           self,
           content: str,
           context: ProcessingContext
       ) -> str:
       """
       Transform Markdown content to HTML with full processing pipeline.

       Args:
           content: Raw markdown content
           context: Processing context

       Returns:
           str: Processed HTML content
       """
       try:
           # Initial markdown to HTML conversion
           html_content = markdown.markdown(
               content,
               extensions=self.extensions
           )

           # Parse into BeautifulSoup
           soup = BeautifulSoup(html_content, 'html.parser')

           # Extract and process LaTeX if present
           if '$$' in content or '$' in content:
               self._latex_equations = self._extract_latex_equations(content)
               if self._latex_equations:
                   self._process_latex_equations(soup)

           # Process all elements through transformer pipeline
           self._process_html_elements(soup, context)

           # Create wrapper with proper classes
           wrapper = soup.new_tag('div')
           classes = ['markdown-content']
           if self._latex_equations:
               classes.append('katex-content')
           wrapper['class'] = ' '.join(classes)

           # Move processed content into wrapper
           for child in soup.children:
               wrapper.append(child)

           return str(wrapper)

       except Exception as e:
           self.logger.error(f"Error transforming markdown to HTML: {str(e)}")
           raise ProcessingError(
               error_type="transformation",
               message=f"HTML transformation failed: {str(e)}",
               context="markdown_transformation"
           )

    def _transform_link(self, element_info: MDElementInfo) -> str:
       """
       Transform a link element to HTML.

       Args:
           element_info: Processed link information

       Returns:
           str: Transformed HTML
       """
       try:
           href = element_info.attributes.custom_attrs.get('href', '')
           classes = ' '.join(['markdown-link', *element_info.attributes.classes])

           # Add target and rel for external links
           target = ''
           rel = ''
           if href.startswith(('http://', 'https://')):
               target = ' target="_blank"'
               rel = ' rel="noopener noreferrer"'

           return f'<a href="{href}" class="{classes}"{target}{rel}>{element_info.content}</a>'

       except Exception as e:
           self.logger.error(f"Error transforming link: {str(e)}")
           return ""



    def _transform_list(self, element_info: MDElementInfo) -> str:
        """
        Transform list elements to HTML.
        Handles ordered, unordered and todo lists.

        Args:
            element_info: Processed list information

        Returns:
            str: Transformed HTML
        """
        try:
            content = html.escape(element_info.content)

            # Handle todo list items
            if element_info.type == MDElementType.TODO:
                is_checked = element_info.metadata.get('todo_info', {}).get('is_checked', False)
                classes = ' '.join(['todo-item', *element_info.attributes.classes])
                return (
                    f'<li class="{classes}">'
                    f'<input type="checkbox" {"checked" if is_checked else ""} disabled />'
                    f'<label>{content}</label>'
                    f'</li>'
                )

            # Handle list items
            if element_info.type == MDElementType.LIST_ITEM:
                classes = ' '.join(['list-item', *element_info.attributes.classes])
                return f'<li class="{classes}">{content}</li>'

            # Handle ordered/unordered lists
            tag = 'ol' if element_info.type == MDElementType.ORDERED_LIST else 'ul'
            base_class = 'ordered-list' if tag == 'ol' else 'unordered-list'
            classes = ' '.join([base_class, *element_info.attributes.classes])

            return (
                f'<{tag} class="{classes}">'
                f'{content}'
                f'</{tag}>'
            )

        except Exception as e:
            self.logger.error(f"Error transforming list element: {str(e)}")
            return ""



    ##########################################################################
    # Metadata processing
    ##########################################################################


    def _transform_metadata(self, element_info: MDElementInfo) -> str:
        """
        Transform YAML metadata into injectable HTML components.
        Prepares metadata for specialized transformers.

        Args:
            element_info: Processed metadata information

        Returns:
            str: Transformed HTML that can be used by other transformers
        """
        try:
            metadata = element_info.metadata

            # Don't render metadata directly in content
            # Instead, prepare it for other transformers
            if metadata.get('content'):
                article_meta = {
                    'title': metadata['content'].get('title'),
                    'authors': metadata['content'].get('authors', []),
                    'abstract': metadata['content'].get('abstract'),
                    'keywords': metadata['content'].get('keywords', [])
                }

                # Store for article metadata transformer
                if not hasattr(self, '_article_metadata'):
                    self._article_metadata = {}
                self._article_metadata.update(article_meta)

            # Handle bibliography data
            if metadata.get('citations'):
                if not hasattr(self, '_bibliography_data'):
                    self._bibliography_data = []
                self._bibliography_data.extend(metadata['citations'])

            # Handle glossary entries
            if metadata.get('glossary'):
                if not hasattr(self, '_glossary_entries'):
                    self._glossary_entries = []
                self._glossary_entries.extend(metadata['glossary'])

            # Return empty string since we don't want to inject metadata directly
            return ""

        except Exception as e:
            self.logger.error(f"Error transforming metadata: {str(e)}")
            return ""



    ##########################################################################
    # LaTeX processing methods
    ##########################################################################


    def _extract_latex_equations(self, content: str) -> List[LaTeXEquation]:
        """
        Extract LaTeX equations from content.

        Args:
            content: Content to process

        Returns:
            List[LaTeXEquation]: Extracted equations
        """
        try:
            equations = []

            # Extract block equations
            block_pattern = r'\$\$(.*?)\$\$'
            for i, match in enumerate(re.finditer(block_pattern, content, re.DOTALL)):
                equations.append(LaTeXEquation(
                    id=f'eq-block-{i}',
                    content=match.group(1).strip(),
                    is_block=True,
                    metadata={}
                ))

            # Extract inline equations
            inline_pattern = r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)'
            for i, match in enumerate(re.finditer(inline_pattern, content)):
                equations.append(LaTeXEquation(
                    id=f'eq-inline-{i}',
                    content=match.group(1).strip(),
                    is_block=False,
                    metadata={}
                ))

            return equations

        except Exception as e:
            self.logger.error(f"Error extracting LaTeX equations: {str(e)}")
            return []

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
