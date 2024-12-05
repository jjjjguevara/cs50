import katex
from typing import Optional
import logging

class KaTeXRenderer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def render(self, latex: str, block: bool = False) -> str:
        try:
            escaped_latex = (
                latex.strip()
                .replace('"', '&quot;')
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
            )

            element_type = 'div' if block else 'span'
            class_name = 'math-block' if block else 'math-tex'

            return (
                f'<{element_type} class="{class_name}" '
                f'data-latex-source="{escaped_latex}">'
                f'{escaped_latex}'
                f'</{element_type}>'
            )
        except Exception as e:
            self.logger.error(f"KaTeX rendering error: {str(e)}")
            return f'<span class="latex-error">{latex}</span>'
