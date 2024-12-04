import katex
from typing import Optional
import logging

class KaTeXRenderer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def render(self, latex: str, block: bool = False) -> str:
        try:
            # Escape special characters in the latex attribute
            escaped_latex = latex.replace('"', '&quot;').replace("'", "&#39;")

            if block:
                return f'<div class="math-block" data-latex-source="{escaped_latex}">{latex}</div>'
            return f'<span class="math-tex" data-latex-source="{escaped_latex}">{latex}</span>'
        except Exception as e:
            self.logger.error(f"KaTeX rendering error: {str(e)}")
            return f'<span class="latex-error">{latex}</span>'
