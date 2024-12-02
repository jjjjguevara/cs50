# app/dita/utils/dita_elements.py
from typing import Dict, List, Optional, Any
from pathlib import Path
from lxml import etree
import html
import logging
from bs4 import BeautifulSoup, Tag

class DITAContentProcessor:
    """Processes DITA elements into HTML with consistent styling"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def process_element(self, elem: etree._Element, source_path: Optional[Path] = None) -> str:
        """
        Process any DITA element based on its tag.
        Main entry point for element processing.
        """
        try:
            tag = etree.QName(elem).localname

            # Map tags to their processing methods
            processors = {
                'xref': self._process_xref,
                'image': self._process_image,
                'p': self._process_paragraph,
                'ul': self._process_unordered_list,
                'ol': self._process_ordered_list,
                'li': self._process_list_item,
                'codeblock': self._process_codeblock,
                'note': self._process_note,
                'table': self._process_table,
                'section': self._process_section,
                'title': self._process_title,
                'fig': self._process_figure,
                'dlentry': self._process_definition_list_entry,
                'term': self._process_term,
                'ph': self._process_phrase,
                'b': self._process_bold,
                'i': self._process_italic,
                'u': self._process_underline,
                'pre': self._process_preformatted,
                'codeph': self._process_code_phrase,
                'cite': self._process_citation,
                'q': self._process_quote,
                'abstract': self._process_abstract,
                'shortdesc': self._process_short_description,
                'related-links': self._process_related_links,
            }

            processor = processors.get(tag, self._process_default)
            return processor(elem, source_path) if source_path else processor(elem)

        except Exception as e:
            self.logger.error(f"Error processing element {etree.QName(elem).localname}: {str(e)}")
            return ''

    def _process_xref(self, elem: etree._Element, source_path: Path) -> str:
        """Process cross-reference links"""
        try:
            href = elem.get('href', '')
            if not href:
                return ''

            # TODO: Implement proper path resolution
            return (
                f'<a href="{html.escape(href)}" '
                f'class="dita-xref text-blue-600 hover:text-blue-800">'
                f'{elem.text or href}'
                f'</a>'
            )
        except Exception as e:
            self.logger.error(f"Error processing xref: {str(e)}")
            return ''

    def _process_image(self, elem: etree._Element, source_path: Path) -> str:
        """Process image elements"""
        try:
            href = elem.get('href', '')
            alt = elem.get('alt', '')
            title = elem.get('title', '')

            if not href:
                return ''

            # TODO: Implement proper image path resolution
            img_attrs = {
                'src': href,
                'alt': alt,
                'class': 'dita-image max-w-full h-auto'
            }

            if title:
                img_attrs['title'] = title

            attr_str = ' '.join(f'{k}="{html.escape(v)}"' for k, v in img_attrs.items())

            if title:
                return (
                    f'<figure class="dita-figure mb-4">'
                    f'<img {attr_str}>'
                    f'<figcaption class="text-center text-sm mt-2">{html.escape(title)}</figcaption>'
                    f'</figure>'
                )
            return f'<img {attr_str}>'

        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return ''


    def _process_paragraph(self, elem: etree._Element) -> str:
        """Process <p> elements"""
        try:
            text = self._get_text_content(elem)
            return f'<p class="dita-p mb-4">{text}</p>'
        except Exception as e:
            self.logger.error(f"Error processing paragraph: {str(e)}")
            return ''

    def _process_unordered_list(self, elem: etree._Element) -> str:
        """Process <ul> elements"""
        try:
            items = [
                self._process_list_item(item)
                for item in elem.findall('li')
            ]
            return (
                f'<ul class="dita-ul list-disc ml-6 mb-4">'
                f'{"".join(items)}'
                f'</ul>'
            )
        except Exception as e:
            self.logger.error(f"Error processing unordered list: {str(e)}")
            return ''

    def _process_ordered_list(self, elem: etree._Element) -> str:
        """Process <ol> elements"""
        try:
            items = [
                self._process_list_item(item)
                for item in elem.findall('li')
            ]
            return (
                f'<ol class="dita-ol list-decimal ml-6 mb-4">'
                f'{"".join(items)}'
                f'</ol>'
            )
        except Exception as e:
            self.logger.error(f"Error processing ordered list: {str(e)}")
            return ''

    def _process_list_item(self, elem: etree._Element) -> str:
        """Process <li> elements"""
        try:
            text = self._get_text_content(elem)
            return f'<li class="dita-li mb-2">{text}</li>'
        except Exception as e:
            self.logger.error(f"Error processing list item: {str(e)}")
            return ''

    def _process_codeblock(self, elem: etree._Element) -> str:
        """Process <codeblock> elements"""
        try:
            text = self._get_text_content(elem)
            return (
                f'<pre class="dita-codeblock bg-gray-100 p-4 rounded-lg mb-4">'
                f'<code>{text}</code>'
                f'</pre>'
            )
        except Exception as e:
            self.logger.error(f"Error processing codeblock: {str(e)}")
            return ''

    def _process_note(self, elem: etree._Element) -> str:
        """Process <note> elements"""
        try:
            text = self._get_text_content(elem)
            note_type = elem.get('type', 'note')
            note_classes = {
                'note': 'bg-blue-50 border-blue-500',
                'warning': 'bg-yellow-50 border-yellow-500',
                'danger': 'bg-red-50 border-red-500',
                'tip': 'bg-green-50 border-green-500'
            }
            classes = note_classes.get(note_type, note_classes['note'])
            return (
                f'<div class="dita-note border-l-4 p-4 mb-4 {classes}">'
                f'<p class="note-content">{text}</p>'
                f'</div>'
            )
        except Exception as e:
            self.logger.error(f"Error processing note: {str(e)}")
            return ''

    def _process_section(self, elem: etree._Element) -> str:
        """Process <section> elements"""
        try:
            content = []
            for child in elem:
                content.append(self.process_element(child))
            return (
                f'<section class="dita-section mb-8">'
                f'{"".join(content)}'
                f'</section>'
            )
        except Exception as e:
            self.logger.error(f"Error processing section: {str(e)}")
            return ''

    def _process_title(self, elem: etree._Element) -> str:
        """Process <title> elements"""
        try:
            text = self._get_text_content(elem)
            return f'<h2 class="dita-title text-xl font-bold mb-4">{text}</h2>'
        except Exception as e:
            self.logger.error(f"Error processing title: {str(e)}")
            return ''

    def _process_figure(self, elem: etree._Element) -> str:
        """Process <fig> elements"""
        try:
            content = []
            title = elem.find('title')
            if title is not None:
                content.append(f'<figcaption class="text-center text-sm mt-2">{title.text}</figcaption>')
            return (
                f'<figure class="dita-figure mb-4">'
                f'{"".join(content)}'
                f'</figure>'
            )
        except Exception as e:
            self.logger.error(f"Error processing figure: {str(e)}")
            return ''

    def _process_definition_list_entry(self, elem: etree._Element) -> str:
        """Process <dlentry> elements"""
        try:
            term = elem.find('dt')
            definition = elem.find('dd')
            return (
                f'<div class="dita-dlentry mb-4">'
                f'<dt class="font-bold">{term.text if term is not None else ""}</dt>'
                f'<dd class="ml-4">{definition.text if definition is not None else ""}</dd>'
                f'</div>'
            )
        except Exception as e:
            self.logger.error(f"Error processing definition list entry: {str(e)}")
            return ''

    def _process_term(self, elem: etree._Element) -> str:
        """Process <term> elements"""
        try:
            text = self._get_text_content(elem)
            return f'<span class="dita-term font-medium">{text}</span>'
        except Exception as e:
            self.logger.error(f"Error processing term: {str(e)}")
            return ''

    def _process_phrase(self, elem: etree._Element) -> str:
        """Process <ph> elements"""
        try:
            text = self._get_text_content(elem)
            output_class = elem.get('outputclass', '')
            class_mapping = {
                'highlight': 'bg-yellow-100',
                'emphasis': 'italic',
                'strong': 'font-bold',
                'code': 'font-mono bg-gray-100 px-1 rounded'
            }
            extra_class = class_mapping.get(output_class, '')
            return f'<span class="dita-ph {extra_class}">{text}</span>'
        except Exception as e:
            self.logger.error(f"Error processing phrase: {str(e)}")
            return ''

    def _process_bold(self, elem: etree._Element) -> str:
        """Process <b> elements"""
        try:
            return f'<strong class="dita-bold font-bold">{elem.text if elem.text else ""}</strong>'
        except Exception as e:
            self.logger.error(f"Error processing bold: {str(e)}")
            return ''

    def _process_italic(self, elem: etree._Element) -> str:
        """Process <i> elements"""
        try:
            return f'<em class="dita-italic italic">{elem.text if elem.text else ""}</em>'
        except Exception as e:
            self.logger.error(f"Error processing italic: {str(e)}")
            return ''

    def _process_underline(self, elem: etree._Element) -> str:
        """Process <u> elements"""
        try:
            return f'<span class="dita-underline underline">{elem.text if elem.text else ""}</span>'
        except Exception as e:
            self.logger.error(f"Error processing underline: {str(e)}")
            return ''

    def _process_preformatted(self, elem: etree._Element) -> str:
        """Process <pre> elements"""
        try:
            return (
                f'<pre class="dita-pre bg-gray-50 p-4 rounded-lg mb-4 '
                f'font-mono text-sm overflow-x-auto">'
                f'{html.escape(elem.text) if elem.text else ""}'
                f'</pre>'
            )
        except Exception as e:
            self.logger.error(f"Error processing preformatted: {str(e)}")
            return ''

    def _process_code_phrase(self, elem: etree._Element) -> str:
        """Process <codeph> elements"""
        try:
            return (
                f'<code class="dita-codeph bg-gray-100 px-1 rounded font-mono text-sm">'
                f'{html.escape(elem.text) if elem.text else ""}</code>'
            )
        except Exception as e:
            self.logger.error(f"Error processing code phrase: {str(e)}")
            return ''

    def _process_citation(self, elem: etree._Element) -> str:
        """Process <cite> elements"""
        try:
            return f'<cite class="dita-cite italic">{elem.text if elem.text else ""}</cite>'
        except Exception as e:
            self.logger.error(f"Error processing citation: {str(e)}")
            return ''

    def _process_quote(self, elem: etree._Element) -> str:
        """Process <q> elements"""
        try:
            return (
                f'<blockquote class="dita-quote border-l-4 border-gray-300 '
                f'pl-4 my-4 italic">{elem.text if elem.text else ""}</blockquote>'
            )
        except Exception as e:
            self.logger.error(f"Error processing quote: {str(e)}")
            return ''

    def _process_abstract(self, elem: etree._Element) -> str:
        """Process <abstract> elements"""
        try:
            content = []
            for child in elem:
                content.append(self.process_element(child))
            return (
                f'<div class="dita-abstract bg-gray-50 p-6 rounded-lg mb-8 '
                f'border border-gray-200">'
                f'{"".join(content)}'
                f'</div>'
            )
        except Exception as e:
            self.logger.error(f"Error processing abstract: {str(e)}")
            return ''

    def _process_short_description(self, elem: etree._Element) -> str:
        """Process <shortdesc> elements"""
        try:
            return (
                f'<p class="dita-shortdesc text-lg text-gray-600 mb-6 '
                f'leading-relaxed">{elem.text if elem.text else ""}</p>'
            )
        except Exception as e:
            self.logger.error(f"Error processing short description: {str(e)}")
            return ''

    def _get_text_content(self, elem: etree._Element) -> str:
        """Safely get text content from element"""
        return html.escape(elem.text) if elem.text else ""

    def _process_related_links(self, elem: etree._Element) -> str:
        """Process <related-links> elements"""
        try:
            links = []
            for link in elem.findall('.//link'):
                href = link.get('href', '')
                if href:
                    title = link.find('linktext')
                    # Handle potential None value for link text
                    link_text = title.text if title is not None and title.text else href
                    links.append(
                        f'<li><a href="{html.escape(href)}" '
                        f'class="text-blue-600 hover:text-blue-800">'
                        f'{html.escape(link_text)}</a></li>'
                    )

            if links:
                return (
                    f'<div class="dita-related-links mt-8 border-t pt-4">'
                    f'<h2 class="text-lg font-bold mb-4">Related Links</h2>'
                    f'<ul class="list-disc ml-6 space-y-2">{"".join(links)}</ul>'
                    f'</div>'
                )
            return ''

        except Exception as e:
            self.logger.error(f"Error processing related links: {str(e)}")
            return ''

    def _process_table(self, elem: etree._Element) -> str:
        """Process <table> elements"""
        try:
            table_html = [
                '<div class="dita-table-wrapper overflow-x-auto mb-8">',
                '<table class="min-w-full divide-y divide-gray-200">'
            ]

            # Process title if present
            title = elem.find('title')
            if title is not None and title.text:
                table_html.insert(0,
                    f'<div class="text-center font-bold mb-2">{title.text}</div>'
                )

            # Process header
            thead = elem.find('.//thead')
            if thead is not None:
                table_html.append('<thead class="bg-gray-50">')
                for row in thead.findall('.//row'):
                    table_html.append('<tr>')
                    for entry in row.findall('entry'):
                        table_html.append(
                            f'<th scope="col" class="px-6 py-3 text-left '
                            f'text-xs font-medium text-gray-500 uppercase tracking-wider">'
                            f'{entry.text if entry.text else ""}</th>'
                        )
                    table_html.append('</tr>')
                table_html.append('</thead>')

            # Process body
            tbody = elem.find('.//tbody')
            if tbody is not None:
                table_html.append(
                    '<tbody class="bg-white divide-y divide-gray-200">'
                )
                for row in tbody.findall('.//row'):
                    table_html.append(
                        '<tr class="hover:bg-gray-50">'
                    )
                    for entry in row.findall('entry'):
                        table_html.append(
                            f'<td class="px-6 py-4 whitespace-nowrap text-sm '
                            f'text-gray-500">{entry.text if entry.text else ""}</td>'
                        )
                    table_html.append('</tr>')
                table_html.append('</tbody>')

            table_html.extend(['</table>', '</div>'])
            return '\n'.join(table_html)

        except Exception as e:
            self.logger.error(f"Error processing table: {str(e)}")
            return ''

    def _process_default(self, elem: etree._Element) -> str:
        """Default processor for unhandled elements"""
        try:
            return f'<!-- Unhandled DITA element: {etree.QName(elem).localname} -->'
        except Exception as e:
            self.logger.error(f"Error in default processor: {str(e)}")
            return ''
