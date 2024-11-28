from typing import Dict, Any, List, Optional
import logging

class DITAHTMLRenderer:
    """Custom HTML renderer for DITA-compatible output"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def render_metadata(self, metadata: Dict[str, Any]) -> str:
        """Render metadata table with consistent styling"""
        html = ['<div class="metadata mb-6 bg-gray-50 p-4 rounded-lg border border-gray-200">']
        html.append('<table class="min-w-full">')
        html.append('<tbody>')

        # Process metadata fields
        if metadata.get('authors'):
            html.append('<tr>')
            html.append('<td class="py-2 px-4 font-semibold">Authors</td>')
            html.append(f'<td class="py-2 px-4">{", ".join(metadata["authors"])}</td>')
            html.append('</tr>')

        # Other metadata fields
        standard_fields = [
            ('journal', 'Journal'),
            ('doi', 'DOI'),
            ('publication-date', 'Publication Date'),
            ('citation', 'Citation'),
            ('institution', 'Institution'),
            ('type', 'Type')
        ]

        for field, label in standard_fields:
            if metadata.get(field):
                html.append('<tr>')
                html.append(f'<td class="py-2 px-4 font-semibold">{label}</td>')
                html.append(f'<td class="py-2 px-4">{metadata[field]}</td>')
                html.append('</tr>')

        # Categories with styling
        if metadata.get('categories'):
            html.append('<tr>')
            html.append('<td class="py-2 px-4 font-semibold">Categories</td>')
            html.append('<td class="py-2 px-4">')
            for category in metadata['categories']:
                html.append(f'<span class="category-tag">{category}</span>')
            html.append('</td>')
            html.append('</tr>')

        # Keywords with styling
        if metadata.get('keywords'):
            html.append('<tr>')
            html.append('<td class="py-2 px-4 font-semibold">Keywords</td>')
            html.append('<td class="py-2 px-4">')
            for keyword in metadata['keywords']:
                html.append(f'<span class="keyword-tag">{keyword}</span>')
            html.append('</td>')
            html.append('</tr>')

        html.append('</tbody>')
        html.append('</table>')
        html.append('</div>')

        return '\n'.join(html)

    def render_content(self, content: str, metadata: Dict[str, Any]) -> str:
        """Render the main content with consistent styling"""
        html = ['<div class="dita-content">']  # Use dita-content for both types

        # Add title
        if metadata.get('title'):
            html.append(f'<h1 class="text-2xl font-bold mb-4">{metadata["title"]}</h1>')

        # Add metadata
        html.append(self.render_metadata(metadata))

        # Add main content
        html.append(f'<div class="topic-content">{content}</div>')

        html.append('</div>')
        return '\n'.join(html)
