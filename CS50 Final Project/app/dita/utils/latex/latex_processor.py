import re
from typing import List, Dict, Optional, Any
from bs4 import BeautifulSoup
import logging

from ..interfaces.latex_definitions import LaTeXProcessor
from .katex_renderer import KaTeXRenderer
from .latex_validator import LaTeXValidator

class DitaLaTeXProcessor(LaTeXProcessor):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.renderer = KaTeXRenderer()
        self.validator = LaTeXValidator()

    def process_equation(self, latex: str, block: bool = False) -> str:
        """Process a single equation."""
        if self.validate_equation(latex):
            return self.renderer.render(latex, block)
        else:
            self.logger.error(f"Invalid LaTeX equation: {latex[:50]}")
            return f'<span class="latex-error">{latex}</span>'

    def process_equations(self, equations: List[Dict[str, Any]]) -> List[str]:
        """Process a batch of equations."""
        return [self.process_equation(eq['latex'], eq.get('block', True))
                for eq in equations]

    def validate_equation(self, latex: str) -> bool:
        """Validate equation."""
        return self.validator.validate_equation(latex)
