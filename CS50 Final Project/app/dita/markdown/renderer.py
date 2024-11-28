from typing import Dict, Any, List, Optional
import logging

class DITAHTMLRenderer:
    """Custom HTML renderer for DITA-compatible output"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def render_metadata(self, metadata: Dict[str, Any]) -> str:
        """Render metadata section"""
        html = ['<div class="metadata mb-6 bg-gray-50 p-4 rounded-lg">']
        html.append('<table class="min-w-full">')

        # Process each metadata field
        for key, value in metadata.items():
            if key and value:  # Skip empty values
                html.append('<tr>')
                html.append(f'<th class="py-2 px-4 text-left">{key.title()}</th>')

                # Handle different value types
                if isinstance(value, list):
                    html.append(f'<td class="py-2 px-4">{", ".join(value)}</td>')
                else:
                    html.append(f'<td class="py-2 px-4">{value}</td>')

                html.append('</tr>')

        html.append('</table>')
        html.append('</div>')
        return '\n'.join(html)

    def render_content(self, content: str, metadata: Dict[str, Any]) -> str:
        """Render main content with DITA semantics"""
        topic_type = metadata.get('type', 'topic')

        html = [f'<div class="dita-content dita-{topic_type}">']

        # Add title if present
        if 'title' in metadata:
            html.append(f'<h1 class="text-2xl font-bold mb-4">{metadata["title"]}</h1>')

        # Add metadata section
        html.append(self.render_metadata(metadata))

        # Add main content
        html.append(content)

        html.append('</div>')
        return '\n'.join(html)
