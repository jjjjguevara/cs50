# app/dita/utils/latex/katex_renderer.py
import logging
from typing import Optional
from app.dita.models.types import LaTeXEquation

class KaTeXRenderer:
    """Renders LaTeX equations to HTML using KaTeX."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def render_equation(self, equation: LaTeXEquation) -> str:
        """
        Render LaTeX equation to HTML.
        """
        try:
            # Prepare equation class based on type
            equation_class = "katex-display" if equation.is_block else "katex-inline"

            # Use proper KaTeX delimiters
            if equation.is_block:
                return f'<div class="{equation_class}" data-equation-id="{equation.id}">$${self._escape_html(equation.content)}$$</div>'
            else:
                return f'<span class="{equation_class}" data-equation-id="{equation.id}">${self._escape_html(equation.content)}$</span>'

        except Exception as e:
            self.logger.error(f"Failed to render equation {equation.id}: {str(e)}")
            raise ValueError(f"Equation rendering failed: {str(e)}")

    def _escape_html(self, content: str) -> str:
        """Escape HTML special characters."""
        return (
            content.replace('&', '&amp;')
                  .replace('<', '&lt;')
                  .replace('>', '&gt;')
                  .replace('"', '&quot;')
                  .replace("'", '&#39;')
        )
