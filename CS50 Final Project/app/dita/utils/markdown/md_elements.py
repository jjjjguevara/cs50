# app/dita/utils/markdown/md_elements.py

from enum import Enum
from typing import Dict, Optional, Union, List, Any
from pathlib import Path
import html
import logging
from bs4 import BeautifulSoup, Tag
import re

from ..id_handler import DITAIDHandler
from ..heading import HeadingHandler

class NoteType(Enum):
    NOTE = 'note'
    WARNING = 'warning'
    DANGER = 'danger'
    TIP = 'tip'


class MarkdownContentProcessor:
    """Processes Markdown elements into HTML with DITA-compliant styling"""

    # Define class attributes explicitly
    logger: logging.Logger
    heading_handler: HeadingHandler
    id_handler: DITAIDHandler


    def __init__(self) -> None:
        """
        Initialize the Markdown content processor.
        """
        self.logger = logging.getLogger(__name__)
        self.heading_handler = HeadingHandler()
        self.id_handler = DITAIDHandler()

    def process_element(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process any Markdown element based on its type."""
        try:
            # Map tags to their processing methods
            processors = {
                # Headers
                'h1': self._process_h1,
                'h2': self._process_h2,
                'h3': self._process_h3,
                'h4': self._process_h4,
                'h5': self._process_h5,
                'h6': self._process_h6,

                # Block elements
                'p': self._process_paragraph,
                'blockquote': self._process_blockquote,
                'pre': self._process_pre,
                'code': self._process_code,

                # Lists
                'ul': self._process_unordered_list,
                'ol': self._process_ordered_list,
                'li': self._process_list_item,

                # Inline elements
                'strong': self._process_strong,
                'em': self._process_emphasis,
                'a': self._process_link,
                'img': self._process_image,

                # Table elements
                'table': self._process_table,
                'thead': self._process_table_head,
                'tbody': self._process_table_body,
                'tr': self._process_table_row,
                'th': self._process_table_header,
                'td': self._process_table_data,

                # Definition lists
                'dl': self._process_definition_list,
                'dt': self._process_definition_term,
                'dd': self._process_definition_description,

                # Additional MDITA elements
                'div': self._process_div,
                'span': self._process_span,
                'figure': self._process_figure,
                'figcaption': self._process_figcaption,
            }

            # Get the processor method for this element type
            processor = processors.get(elem.name)
            if processor:
                result = processor(elem, source_path)
                self.logger.info(f"Processed {elem.name} result length: {len(result) if result else 0}")
                return result
            else:
                self.logger.warning(f"No processor found for element: {elem.name}")
                return ''

        except Exception as e:
            self.logger.error(f"Error processing element {elem.name}: {str(e)}")
            return ''

    # ===========================
    # Header Elements
    # ===========================

    def _process_h1(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process h1 element (DITA equivalent: topic title)"""
        try:
            heading_id, formatted_text = self.heading_handler.process_heading(
                elem.text or "",
                1
            )
            return (
                f'<h1 id="{heading_id}" class="topic-title">'
                f'{formatted_text}'
                f'<a href="#{heading_id}" class="heading-anchor">¶</a>'
                f'</h1>'
            )
        except Exception as e:
            self.logger.error(f"Error processing h1: {str(e)}")
            return ''

    def _process_h2(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process h2 element (DITA equivalent: section title)"""
        try:
            heading_id, formatted_text = self.heading_handler.process_heading(
                elem.text or "",
                2
            )
            return (
                f'<h2 id="{heading_id}" class="section-title">'
                f'{formatted_text}'
                f'<a href="#{heading_id}" class="heading-anchor">¶</a>'
                f'</h2>'
            )
        except Exception as e:
            self.logger.error(f"Error processing h2: {str(e)}")
            return ''

    def _process_h3(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process h3 element"""
        try:
            heading_id, formatted_text = self.heading_handler.process_heading(
                elem.text or "",
                3
            )
            return (
                f'<h3 id="{heading_id}" class="subsection-title">'
                f'{formatted_text}'
                f'<a href="#{heading_id}" class="heading-anchor">¶</a>'
                f'</h3>'
            )
        except Exception as e:
            self.logger.error(f"Error processing h3: {str(e)}")
            return ''

    def _process_h4(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process h4 element"""
        try:
            heading_id, formatted_text = self.heading_handler.process_heading(
                elem.text or "",
                4
            )
            return (
                f'<h4 id="{heading_id}" class="subsubsection-title">'
                f'{formatted_text}'
                f'<a href="#{heading_id}" class="heading-anchor">¶</a>'
                f'</h4>'
            )
        except Exception as e:
            self.logger.error(f"Error processing h4: {str(e)}")
            return ''

    def _process_h5(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process h5 element"""
        try:
            heading_id, formatted_text = self.heading_handler.process_heading(
                elem.text or "",
                5
            )
            return (
                f'<h5 id="{heading_id}" class="title-level5">'
                f'{formatted_text}'
                f'<a href="#{heading_id}" class="heading-anchor">¶</a>'
                f'</h5>'
            )
        except Exception as e:
            self.logger.error(f"Error processing h5: {str(e)}")
            return ''

    def _process_h6(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process h6 element"""
        try:
            heading_id, formatted_text = self.heading_handler.process_heading(
                elem.text or "",
                6
            )
            return (
                f'<h6 id="{heading_id}" class="title-level6">'
                f'{formatted_text}'
                f'<a href="#{heading_id}" class="heading-anchor">¶</a>'
                f'</h6>'
            )
        except Exception as e:
            self.logger.error(f"Error processing h6: {str(e)}")
            return ''

    # ===========================
    # Block Elements
    # ===========================

    def _process_paragraph(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process paragraph element"""
        try:
            content = self._process_inline_content(elem)
            return f'<p class="md-p mb-4">{content}</p>'
        except Exception as e:
            self.logger.error(f"Error processing paragraph: {str(e)}")
            return ''

    def _process_blockquote(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process blockquote element"""
        try:
            content = self._process_inline_content(elem)
            return (
                f'<blockquote class="md-quote border-l-4 border-gray-300 '
                f'pl-4 my-4 italic">{content}</blockquote>'
            )
        except Exception as e:
            self.logger.error(f"Error processing blockquote: {str(e)}")
            return ''

    def _normalize_classes(self, classes_attr: Union[str, List[str], None]) -> List[str]:
            """Normalize class attribute to list of strings"""
            if classes_attr is None:
                return []
            if isinstance(classes_attr, str):
                return [classes_attr]
            return [str(c) for c in classes_attr if c is not None]

    def _get_language_class(self, code_elem: Tag) -> str:
        """Safely extract language class from code element"""
        try:
            if not isinstance(code_elem, Tag):
                return ""

            # Get raw classes attribute
            classes_attr = code_elem.get('class', [])

            # Normalize to List[str]
            classes = self._normalize_classes(classes_attr)

            # Find first language class
            language_classes = [
                c.replace('language-', '')
                for c in classes
                if isinstance(c, str) and c.startswith('language-')
            ]

            return language_classes[0] if language_classes else ""

        except Exception as e:
            self.logger.error(f"Error extracting language class: {str(e)}")
            return ""

    def _get_safe_text(self, elem: Optional[Tag]) -> str:
        """Safely get text content from element"""
        try:
            if not elem:
                return ""
            return elem.text if elem.text else ""
        except Exception as e:
            self.logger.error(f"Error getting text content: {str(e)}")
            return ""

    def _process_pre(self, elem: Tag, source_path: Optional[Path] = None) -> str:
            """Process pre element"""
            try:
                code_elem = elem.find('code')
                if not isinstance(code_elem, Tag):
                    return (
                        f'<pre class="md-pre bg-gray-50 p-4 rounded-lg mb-4 '
                        f'font-mono text-sm overflow-x-auto">'
                        f'{html.escape(self._get_safe_text(elem))}</pre>'
                    )

                lang = self._get_language_class(code_elem)
                lang_class = f" language-{lang}" if lang else ""
                content = self._get_safe_text(code_elem)

                return (
                    f'<pre class="md-pre bg-gray-50 p-4 rounded-lg mb-4 '
                    f'font-mono text-sm overflow-x-auto{lang_class}">'
                    f'{html.escape(content)}'
                    f'</pre>'
                )
            except Exception as e:
                self.logger.error(f"Error processing pre: {str(e)}")
                return ''

    def _process_code(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process code element"""
        try:
            if not elem or not isinstance(elem, Tag):
                return ''

            parent = elem.parent
            if not parent or not isinstance(parent, Tag):
                return ''

            if parent.name == 'pre':
                return html.escape(self._get_safe_text(elem))

            return (
                f'<code class="md-code bg-gray-100 px-1 rounded '
                f'font-mono text-sm">{html.escape(self._get_safe_text(elem))}</code>'
            )
        except Exception as e:
            self.logger.error(f"Error processing code: {str(e)}")
            return ''

    # ===========================
    # List Elements
    # ===========================

    def _process_unordered_list(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process unordered list element"""
        try:
            items = []
            for item in elem.find_all('li', recursive=False):
                items.append(self.process_element(item, source_path))
            return (
                f'<ul class="md-ul list-disc ml-6 mb-4">'
                f'{"".join(items)}'
                f'</ul>'
            )
        except Exception as e:
            self.logger.error(f"Error processing unordered list: {str(e)}")
            return ''

    def _process_ordered_list(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process ordered list element"""
        try:
            items = []
            for item in elem.find_all('li', recursive=False):
                items.append(self.process_element(item, source_path))
            return (
                f'<ol class="md-ol list-decimal ml-6 mb-4">'
                f'{"".join(items)}'
                f'</ol>'
            )
        except Exception as e:
            self.logger.error(f"Error processing ordered list: {str(e)}")
            return ''

    def _process_list_item(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process list item element"""
        try:
            content = self._process_inline_content(elem)
            return f'<li class="md-li mb-2">{content}</li>'
        except Exception as e:
            self.logger.error(f"Error processing list item: {str(e)}")
            return ''

    # ===========================
    # Inline Elements
    # ===========================

    def _process_strong(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process strong element"""
        try:
            content = self._process_inline_content(elem)
            return f'<strong class="md-strong font-bold">{content}</strong>'
        except Exception as e:
            self.logger.error(f"Error processing strong: {str(e)}")
            return ''

    def _process_emphasis(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process emphasis element"""
        try:
            content = self._process_inline_content(elem)
            return f'<em class="md-em italic">{content}</em>'
        except Exception as e:
            self.logger.error(f"Error processing emphasis: {str(e)}")
            return ''

    def _process_link(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process link element"""
        try:
            href = elem.get('href', '')
            if isinstance(href, list):
                href = href[0] if href else ''

            title = elem.get('title', '')
            if isinstance(title, list):
                title = title[0] if title else ''

            content = self._process_inline_content(elem)

            # Handle internal links
            if href.startswith('#'):
                heading_id = self.id_handler.generate_content_id(
                    Path(href[1:].replace('/', '-'))
                )
                href = f"#{heading_id}"

            title_attr = f' title="{html.escape(title)}"' if title else ''
            return (
                f'<a href="{html.escape(href)}" class="md-a text-blue-600 '
                f'hover:text-blue-800"{title_attr}>{content}</a>'
            )
        except Exception as e:
            self.logger.error(f"Error processing link: {str(e)}")
            return ''

    def _process_image(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process image element"""
        try:
            src = elem.get('src', '')
            alt = elem.get('alt', '')
            title = elem.get('title', '')

            if not src:
                return ''

            img_attrs = {
                'src': src,
                'alt': alt,
                'class': 'md-img max-w-full h-auto'
            }

            if title:
                img_attrs['title'] = title

            attr_str = ' '.join(f'{k}="{html.escape(v)}"' for k, v in img_attrs.items())
            return f'<img {attr_str}>'
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return ''


# ===========================
    # Table Elements
    # ===========================

    def _process_table(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process table element"""
        try:
            table_id = self.id_handler.generate_content_id(Path("table"))
            return (
                f'<div class="md-table-wrapper overflow-x-auto mb-8" id="{table_id}">'
                f'<table class="min-w-full divide-y divide-gray-200">'
                f'{self._process_table_children(elem)}'
                f'</table></div>'
            )
        except Exception as e:
            self.logger.error(f"Error processing table: {str(e)}")
            return ''

    def _process_table_head(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process table head element"""
        try:
            return (
                f'<thead class="bg-gray-50">'
                f'{self._process_table_children(elem)}'
                f'</thead>'
            )
        except Exception as e:
            self.logger.error(f"Error processing table head: {str(e)}")
            return ''

    def _process_table_body(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process table body element"""
        try:
            return (
                f'<tbody class="bg-white divide-y divide-gray-200">'
                f'{self._process_table_children(elem)}'
                f'</tbody>'
            )
        except Exception as e:
            self.logger.error(f"Error processing table body: {str(e)}")
            return ''

    def _process_table_row(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process table row element"""
        try:
            return (
                f'<tr class="hover:bg-gray-50">'
                f'{self._process_table_children(elem)}'
                f'</tr>'
            )
        except Exception as e:
            self.logger.error(f"Error processing table row: {str(e)}")
            return ''

    def _process_table_header(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process table header cell element"""
        try:
            content = self._process_inline_content(elem)
            align = elem.get('align', 'left')
            return (
                f'<th scope="col" class="px-6 py-3 text-{align} '
                f'text-xs font-medium text-gray-500 uppercase tracking-wider">'
                f'{content}</th>'
            )
        except Exception as e:
            self.logger.error(f"Error processing table header: {str(e)}")
            return ''

    def _process_table_data(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process table data cell element"""
        try:
            content = self._process_inline_content(elem)
            align = elem.get('align', 'left')
            return (
                f'<td class="px-6 py-4 whitespace-nowrap text-sm '
                f'text-gray-500 text-{align}">{content}</td>'
            )
        except Exception as e:
            self.logger.error(f"Error processing table data: {str(e)}")
            return ''

    # ===========================
    # Definition List Elements
    # ===========================

    def _process_definition_list(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process definition list element"""
        try:
            content = self._process_definition_list_children(elem)
            return (
                f'<dl class="md-dl space-y-4 mb-4">'
                f'{content}'
                f'</dl>'
            )
        except Exception as e:
            self.logger.error(f"Error processing definition list: {str(e)}")
            return ''

    def _process_definition_term(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process definition term element"""
        try:
            content = self._process_inline_content(elem)
            return f'<dt class="md-dt font-bold">{content}</dt>'
        except Exception as e:
            self.logger.error(f"Error processing definition term: {str(e)}")
            return ''

    def _process_definition_description(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process definition description element"""
        try:
            content = self._process_inline_content(elem)
            return f'<dd class="md-dd ml-4">{content}</dd>'
        except Exception as e:
            self.logger.error(f"Error processing definition description: {str(e)}")
            return ''

    # ===========================
    # Additional MDITA Elements
    # ===========================

    def _process_div(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process div element (for MDITA extended profile)"""
        try:
            content = self._process_inline_content(elem)
            class_name = elem.get('class', '')

            # Get data-class attribute and handle it directly
            if elem.get('data-class') == 'note':
                return self._process_note(elem)
            elif elem.get('data-class') == 'keydef':
                return self._process_keydef(elem)

            return f'<div class="md-div {class_name}">{content}</div>'
        except Exception as e:
            self.logger.error(f"Error processing div: {str(e)}")
            return ''

    def _process_span(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process span element (for MDITA extended profile)"""
        try:
            content = self._process_inline_content(elem)
            class_name = elem.get('class', '')

            # Handle footnotes directly from class attribute
            if 'fn' in (class_name if isinstance(class_name, str) else ''):
                return self._process_footnote(elem)

            # Handle data-class based components if needed
            if elem.get('data-class') == 'linktext':
                return f'<var class="md-linktext">{content}</var>'

            # Default span processing
            return f'<span class="md-span {class_name}">{content}</span>'

        except Exception as e:
            self.logger.error(f"Error processing span: {str(e)}")
            return ''

    def _process_figure(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process figure element"""
        try:
            content = self._process_figure_children(elem)
            return f'<figure class="md-figure mb-4">{content}</figure>'
        except Exception as e:
            self.logger.error(f"Error processing figure: {str(e)}")
            return ''

    def _process_figcaption(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process figure caption element"""
        try:
            content = self._process_inline_content(elem)
            return f'<figcaption class="text-center text-sm mt-2">{content}</figcaption>'
        except Exception as e:
            self.logger.error(f"Error processing figcaption: {str(e)}")
            return ''

    # ===========================
    # Special Elements
    # ===========================

    def _process_note(self, elem: Tag, source_path: Optional[Path] = None) -> str:
            """Process note element"""
            try:
                content = self._process_inline_content(elem)

                # Safely get note type
                note_type_attr = elem.get('type', 'note')
                note_type_str = note_type_attr[0] if isinstance(note_type_attr, list) else str(note_type_attr)

                # Convert to enum for type safety
                try:
                    note_type = NoteType(note_type_str.lower())
                except ValueError:
                    note_type = NoteType.NOTE

                # Define note classes with strict typing
                note_classes: Dict[NoteType, str] = {
                    NoteType.NOTE: 'bg-blue-50 border-blue-500',
                    NoteType.WARNING: 'bg-yellow-50 border-yellow-500',
                    NoteType.DANGER: 'bg-red-50 border-red-500',
                    NoteType.TIP: 'bg-green-50 border-green-500'
                }

                # Safely get class
                classes = note_classes.get(note_type, note_classes[NoteType.NOTE])

                return (
                    f'<div class="md-note border-l-4 p-4 mb-4 {classes}">'
                    f'{content}'
                    f'</div>'
                )
            except Exception as e:
                self.logger.error(f"Error processing note: {str(e)}")
                return ''

    def _process_keydef(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process keydef element"""
        try:
            content = self._process_inline_content(elem)
            return f'<div class="md-keydef hidden">{content}</div>'
        except Exception as e:
            self.logger.error(f"Error processing keydef: {str(e)}")
            return ''

    def _process_footnote(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process footnote element"""
        try:
            content = self._process_inline_content(elem)
            footnote_id = self.id_handler.generate_content_id(Path("footnote"))
            return (
                f'<sup class="md-footnote"><a href="#{footnote_id}" '
                f'id="fnref:{footnote_id}">[{footnote_id}]</a></sup>'
                f'<span class="footnote-content hidden" id="fn:{footnote_id}">'
                f'{content}</span>'
            )
        except Exception as e:
            self.logger.error(f"Error processing footnote: {str(e)}")
            return ''

    # ===========================
    # Helper Methods
    # ===========================

    def _process_inline_content(self, elem: Tag) -> str:
            """Process inline content of an element"""
            try:
                if elem.text:
                    # Escape regular text
                    return html.escape(elem.text)

                content = []
                for child in elem.children:
                    if isinstance(child, Tag):
                        content.append(self.process_element(child))
                    else:
                        content.append(html.escape(str(child)))
                return ''.join(content)

            except Exception as e:
                self.logger.error(f"Error processing inline content: {str(e)}")
                return ''

    def _process_table_children(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process table children elements"""
        try:
            content = []
            for child in elem.children:
                if isinstance(child, Tag):
                    content.append(self.process_element(child))
            return ''.join(content)
        except Exception as e:
            self.logger.error(f"Error processing table children: {str(e)}")
            return ''

    def _process_definition_list_children(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process definition list children elements"""
        try:
            content = []
            for child in elem.children:
                if isinstance(child, Tag):
                    content.append(self.process_element(child))
            return ''.join(content)
        except Exception as e:
            self.logger.error(f"Error processing definition list children: {str(e)}")
            return ''

    def _process_figure_children(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Process figure children elements"""
        try:
            content = []
            for child in elem.children:
                if isinstance(child, Tag):
                    content.append(self.process_element(child))
            return ''.join(content)
        except Exception as e:
            self.logger.error(f"Error processing figure children: {str(e)}")
            return ''

    def _is_latex(self, text: str) -> bool:
        """Check if text contains LaTeX content"""
        return bool(re.search(r'(?:\$\$|\$).*(?:\$\$|\$)', text))

    def _process_default(self, elem: Tag, source_path: Optional[Path] = None) -> str:
        """Default processor for unhandled elements"""
        try:
            return f'<!-- Unhandled Markdown element: {elem.name} -->'
        except Exception as e:
            self.logger.error(f"Error in default processor: {str(e)}")
            return ''
