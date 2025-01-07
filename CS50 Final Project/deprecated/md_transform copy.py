# app/dita/transformers/md_transform.py
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Callable, List, Dict, Any, TypedDict
from bs4 import NavigableString, Tag
import html
import markdown
from bs4 import BeautifulSoup, Tag
from functools import partial
from ..models.types import (
    ContentElement,
    ProcessingState,
    ProcessedContent,
    MDElementInfo,
    MDElementType,
    ElementType,
    ProcessingContext,
    LaTeXEquation,
    ProcessingError,
    ProcessedEquation,
    ProcessingPhase
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
        self.heading_handler = HeadingHandler(processing_metadata=self.processing_metadata)
        self.metadata_handler = MetadataHandler()

        # Initialize content processor first
        content_processor = ContentProcessor(
            dita_root=root_path,
            markdown_root=root_path
        )

        # Pass content processor to markdown element processor
        self.content_processor = MarkdownElementProcessor(content_processor, document_metadata={}, map_metadata={})

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
            # MDElementType.HEADING: self._transform_heading,
            MDElementType.CALLOUT: self._transform_blockquote,
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
            self.enable_toc = config.show_toc

            self.logger.debug(
                f"Configured with features: latex={self.enable_latex}, "
                f"numbering={self.enable_numbering}, cross_refs={self.enable_cross_refs}"
            )

        except Exception as e:
            self.logger.error(f"MarkdownTransformer configuration failed: {str(e)}")
            raise

    def transform_topic(
       self,
       element: ContentElement,
       context: ProcessingContext,
       html_converter: Optional[Callable[[str, ProcessingContext], str]] = None
    ) -> ProcessedContent:
       """Transform markdown to HTML with processing context."""
       try:
           # Extract metadata using element's path and ID
           metadata = self.metadata_handler.extract_metadata(element.path, element.id)

           # Transform content
           if html_converter:
               html_content = html_converter(element.content, context)
           else:
               html_content = self._transform_markdown_to_html(element, context)

           # Process with BeautifulSoup
           soup = BeautifulSoup(html_content, 'html.parser')
           self.process_elements(soup, element)

           # Apply registered strategies
           html_content = self._apply_strategies(
               html=str(soup),
               element=element,
               context=context,
               metadata=metadata
           )

           # Final processing
           final_html = self.html_helper.process_final_content(html_content)

           # Update element state
           element.state = ProcessingState.COMPLETED
           element.phase = ProcessingPhase.ASSEMBLY

           return ProcessedContent(
               html=final_html,
               element_id=element.id,
               metadata=metadata
           )

       except Exception as e:
           element.set_error(str(e))
           self.logger.error(f"Error transforming markdown: {str(e)}")
           return ProcessedContent(
               html=f"<div class='error'>Transform error: {str(e)}</div>",
               element_id=element.id,
               metadata={}
           )

    def _transform_markdown_to_html(
        self,
        element: ContentElement,
        context: ProcessingContext
    ) -> str:
        """
        Transform Markdown content to HTML with full processing pipeline.

        Args:
            element: ContentElement containing markdown content
            context: Processing context

        Returns:
            str: Processed HTML content
        """
        try:
            # Initial markdown to HTML conversion
            html_content = markdown.markdown(
                element.content,
                extensions=self.extensions
            )

            # Parse into BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # Handle LaTeX if present and enabled
            if context.features.get("process_latex"):
                if '$$' in element.content or '$' in element.content:
                    latex_equations = self.latex_processor.process_equations(
                        self._extract_latex_equations(element.content)
                    )
                    if latex_equations:
                        element.metadata["has_latex"] = True
                        self._process_latex_equations(soup, latex_equations)

            # Process all elements through transformer pipeline
            self.process_elements(soup, element)

            # Create wrapper with proper classes and metadata
            wrapper = soup.new_tag('div')
            wrapper_classes = ['markdown-content']

            # Add conditional classes
            if element.metadata.get("has_latex"):
                wrapper_classes.append('katex-content')
            if element.metadata.get("has_bibliography"):
                wrapper_classes.append('bibliography-content')
            if context.features.get("enable_cross_refs"):
                wrapper_classes.append('cross-refs-enabled')

            wrapper['class'] = ' '.join(wrapper_classes)

            # Add metadata attributes
            for key, value in element.metadata.get("content_attributes", {}).items():
                wrapper[f"data-{key}"] = str(value)

            # Move processed content into wrapper
            for child in soup.children:
                wrapper.append(child)

            return str(wrapper)

        except Exception as e:
            self.logger.error(f"Error transforming markdown to HTML: {str(e)}")
            raise ProcessingError(
                error_type="transformation",
                message=f"HTML transformation failed: {str(e)}",
                context=str(element.path)
            )


    ##########################################################################
    # Transformer methods and final processing
    ##########################################################################


    def process_elements(
        self,
        soup: BeautifulSoup,
        parsed_element: ContentElement
    ) -> None:
        """Process soup elements using our tracking and metadata infrastructure."""
        try:
            document_metadata = self.metadata_handler.extract_metadata(
                parsed_element.path,
                parsed_element.id
            )

            for html_elem in soup.find_all(True):
                # Process element with full context
                processed_element = self.content_processor.process_element(
                    html_elem,
                    source_path=parsed_element.path,
                    document_metadata=document_metadata,
                    map_metadata=parsed_element.metadata
                )

                # Apply metadata and attributes
                if processed_element.html_metadata["attributes"].get("id"):
                    html_elem["id"] = processed_element.html_metadata["attributes"]["id"]

                if processed_element.html_metadata["classes"]:
                    html_elem["class"] = " ".join(processed_element.html_metadata["classes"])

                # Handle image elements specially
                if processed_element.type == MDElementType.IMAGE:
                    if transformed_html := self._transform_image(processed_element):
                        new_element = BeautifulSoup(transformed_html, "html.parser")
                        html_elem.replace_with(new_element)
                    continue

                # Apply element-specific transformations
                if processed_element.type.value in MDElementType.__members__:
                    md_type = MDElementType(processed_element.type.value)
                    if md_type in self._element_transformers:
                        transform_method = self._element_transformers[md_type]
                        if transformed_html := transform_method(processed_element):
                            new_element = BeautifulSoup(transformed_html, "html.parser")
                            html_elem.replace_with(new_element)

        except Exception as e:
            self.logger.error(f"Error processing elements: {str(e)}")
            raise ProcessingError(
                error_type="transformation",
                message=str(e),
                context=str(parsed_element.path)
            )



    def _transform_paragraph(self, tracked_element: ContentElement) -> str:
        """
        Transform paragraph elements to HTML.

        Args:
            tracked_element: Processed element information

        Returns:
            str: Transformed HTML
        """
        try:
            content = html.escape(tracked_element.content)
            classes = ' '.join(['paragraph', *tracked_element.html_metadata["classes"]])

            return (
                f'<p class="{classes}">'
                f'{content}'
                f'</p>'
            )

        except Exception as e:
            self.logger.error(f"Error transforming paragraph: {str(e)}")
            return ""

    def _transform_footnote(self, tracked_element: ContentElement) -> str:
       """
       Transform footnote elements to HTML, supporting Obsidian-style footnotes.
       Handles both inline references and footnote content sections.

       Args:
           tracked_element: Processed footnote information

       Returns:
           str: Transformed HTML
       """
       try:
           content = html.escape(tracked_element.content)
           footnote_id = tracked_element.html_metadata["attributes"].get("id", "")
           base_classes = ['footnote', *tracked_element.html_metadata["classes"]]

           # Handle inline reference (^1)
           if tracked_element.metadata.get('is_reference', False):
               ref_number = tracked_element.metadata.get('footnote_number', '')
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

    def _transform_emphasis(self, tracked_element: ContentElement) -> str:
       """
       Transform emphasis elements to HTML.
       Handles bold, italic, underline and strikethrough, including nested formatting.

       Args:
           tracked_element: Processed element information

       Returns:
           str: Transformed HTML
       """
       try:
           content = html.escape(tracked_element.content)

           # Map element types to HTML tags and classes
           emphasis_map = {
               'bold': ('strong', 'bold'),
               'italic': ('em', 'italic'),
               'underline': ('u', 'underline'),
               'strikethrough': ('del', 'strikethrough')
           }

           # Get element type value
           element_type = tracked_element.type.value
           if element_type not in emphasis_map:
               return content

           tag, base_class = emphasis_map[element_type]
           classes = ' '.join([base_class, *tracked_element.html_metadata["classes"]])

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

    def _transform_image(self, tracked_element: ContentElement) -> str:
        """
        Transform an image element to HTML.

        Args:
            tracked_element: Processed image information

        Returns:
            str: Transformed HTML
        """
        try:
            # Get base attributes
            attrs = tracked_element.html_metadata["attributes"]
            src = attrs.get('src', '')
            alt = attrs.get('alt', '')
            title = attrs.get('title', '')

            # Handle conditional images
            if conditional_src := tracked_element.metadata.get('conditional_image', {}).get('src'):
                src = conditional_src

            # Process source path
            if src and not src.startswith(('http://', 'https://')):
                src = self.html_helper.resolve_image_path(src, tracked_element.path)

            # Build classes
            classes = ['img-fluid', 'md-image', *tracked_element.html_metadata["classes"]]

            # Build image tag
            img_attrs = [
                f'src="{src}"',
                f'alt="{alt}"',
                f'class="{" ".join(classes)}"',
                'loading="lazy"'
            ]

            if title:
                img_attrs.append(f'title="{title}"')

            img = f'<img {" ".join(img_attrs)} />'

            # Add figure wrapper if there's alt text
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


    def _transform_table(self, tracked_element: ContentElement) -> str:
        """Transform table elements to HTML."""
        content = html.escape(tracked_element.content)
        table_info = tracked_element.metadata.get('table_info', {})

        table_map = {
            'table': {
                'tag': 'table',
                'base_class': 'markdown-table',
                'attrs': {'role': 'grid'}
            },
            'table_header': {
                'tag': 'th',
                'base_class': 'table-header',
                'attrs': {'scope': 'col'}
            },
            'table_row': {
                'tag': 'tr',
                'base_class': 'table-row',
                'attrs': {}
            },
            'table_cell': {
                'tag': 'td',
                'base_class': 'table-cell',
                'attrs': {}
            }
        }

        element_type = tracked_element.type.value
        if element_type not in table_map:
            return content

        config = table_map[element_type]
        tag = config['tag']

        specialization = table_info.get('type', 'default')
        base_classes = [
            config['base_class'],
            f'table-{specialization}',
            *tracked_element.html_metadata["classes"]
        ]

        # Build final attributes
        final_attrs = config['attrs'].copy()

        # Handle specializations
        if specialization in ('bibliography', 'glossary', 'metadata'):
            if specialization == 'bibliography':
                final_attrs.update({
                    'data-citation-format': table_info.get('citation_format', 'apa'),
                    'data-sort': table_info.get('sort_by', 'author'),
                    'aria-label': 'Bibliography entries'
                })
            elif specialization == 'glossary':
                final_attrs.update({
                    'data-sort': table_info.get('sort_by', 'term'),
                    'data-show-refs': str(table_info.get('show_references', True)).lower(),
                    'aria-label': 'Glossary terms and definitions'
                })
            else:  # metadata
                final_attrs.update({
                    'data-visibility': table_info.get('visibility', 'visible'),
                    'data-collapsible': str(table_info.get('collapsible', False)).lower(),
                    'aria-label': 'Article metadata'
                })

        # Add table info attributes
        if table_info.get('has_header'):
            final_attrs['data-has-header'] = 'true'
        if rows := table_info.get('rows'):
            final_attrs['data-rows'] = str(rows)
        if cols := table_info.get('columns'):
            final_attrs['data-columns'] = str(cols)

        # Add custom attributes from html_metadata
        final_attrs.update(tracked_element.html_metadata["attributes"])
        final_attrs['class'] = ' '.join(base_classes)

        # Build attribute string
        attrs_str = ' '.join(f'{k}="{v}"' for k, v in final_attrs.items())

        return f'<{tag} {attrs_str}>{content}</{tag}>'

    def _transform_code_block(self, tracked_element: ContentElement) -> str:
        """Transform code elements (blocks and phrases) to HTML."""
        try:
            content = html.escape(tracked_element.content)
            element_type = tracked_element.type.value

            # Handle inline code phrase
            if element_type == 'code_phrase':
                classes = ' '.join(['code-inline', *tracked_element.html_metadata["classes"]])
                return f'<code class="{classes}">{content}</code>'

            # Handle code blocks
            language = tracked_element.metadata.get('code_info', {}).get('language', '')

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

    def _transform_blockquote(self, tracked_element: ContentElement) -> str:
        """Transform blockquote element to HTML."""
        try:
            content = html.escape(tracked_element.content)
            classes = ' '.join(['blockquote', *tracked_element.html_metadata["classes"]])

            return (
                f'<blockquote class="{classes}">'
                f'<p>{content}</p>'
                f'</blockquote>'
            )

        except Exception as e:
            self.logger.error(f"Error transforming blockquote: {str(e)}")
            return ""


    def _transform_link(self, tracked_element: ContentElement) -> str:
        """
        Transform a link element to HTML.

        Args:
            tracked_element: Processed link information

        Returns:
            str: Transformed HTML
        """
        try:
            attrs = tracked_element.html_metadata["attributes"]
            href = attrs.get('href', '')
            classes = ['markdown-link', *tracked_element.html_metadata["classes"]]

            # Build link attributes
            link_attrs = {
                'href': href,
                'class': ' '.join(classes)
            }

            # Handle external links
            if href.startswith(('http://', 'https://')):
                link_attrs['target'] = '_blank'
                link_attrs['rel'] = 'noopener noreferrer'

            # Build HTML attributes string
            attrs_str = ' '.join(f'{k}="{v}"' for k, v in link_attrs.items())

            return f'<a {attrs_str}>{tracked_element.content}</a>'

        except Exception as e:
            self.logger.error(f"Error transforming link: {str(e)}")
            return ""

    def _transform_list(self, tracked_element: ContentElement) -> str:
        """
        Transform list elements to HTML.
        Handles ordered, unordered and todo lists.

        Args:
            tracked_element: Processed list information

        Returns:
            str: Transformed HTML
        """
        try:
            content = html.escape(tracked_element.content)
            element_type = tracked_element.type.value

            # Handle todo list items
            if element_type == 'todo':
                is_checked = tracked_element.metadata.get('todo_info', {}).get('is_checked', False)
                classes = ['todo-item', *tracked_element.html_metadata["classes"]]
                return (
                    f'<li class="{" ".join(classes)}">'
                    f'<input type="checkbox" {"checked" if is_checked else ""} disabled />'
                    f'<label>{content}</label>'
                    f'</li>'
                )

            # Handle list items
            if element_type == 'list_item':
                classes = ['list-item', *tracked_element.html_metadata["classes"]]
                return f'<li class="{" ".join(classes)}">{content}</li>'

            # Handle ordered/unordered lists
            is_ordered = element_type == 'ordered_list'
            tag = 'ol' if is_ordered else 'ul'
            base_class = 'ordered-list' if is_ordered else 'unordered-list'
            classes = [base_class, *tracked_element.html_metadata["classes"]]

            # Add any additional attributes
            attrs = tracked_element.html_metadata["attributes"].copy()
            attrs['class'] = ' '.join(classes)
            attrs_str = ' '.join(f'{k}="{v}"' for k, v in attrs.items())

            return f'<{tag} {attrs_str}>{content}</{tag}>'

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
        metadata = element_info.metadata

        if metadata.get('content'):
            article_meta = {
                'title': metadata['content'].get('title'),
                'authors': metadata['content'].get('authors', []),
                'abstract': metadata['content'].get('abstract'),
                'keywords': metadata['content'].get('keywords', [])
            }
            if not hasattr(self, '_article_metadata'):
                self._article_metadata = {}
            self._article_metadata.update(article_meta)

        if metadata.get('citations'):
            if not hasattr(self, '_bibliography_data'):
                self._bibliography_data = []
            self._bibliography_data.extend(metadata['citations'])

        if metadata.get('glossary'):
            if not hasattr(self, '_glossary_entries'):
                self._glossary_entries = []
            self._glossary_entries.extend(metadata['glossary'])

        return ""
