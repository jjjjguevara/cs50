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

        Args:
            equation: Equation to render

        Returns:
            HTML string with rendered equation

        Raises:
            ValueError: If rendering fails
        """
        try:
            # Prepare equation class based on type
            equation_class = "display-equation" if equation.is_block else "inline-equation"

            # Create KaTeX compatible HTML
            html = (
                f'<div class="katex-equation {equation_class}" '
                f'data-equation-id="{equation.id}">'
                f'{self._escape_html(equation.content)}'
                f'</div>' if equation.is_block else
                f'<span class="katex-equation {equation_class}" '
                f'data-equation-id="{equation.id}">'
                f'{self._escape_html(equation.content)}'
                f'</span>'
            )

            return html

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
