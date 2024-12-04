import re
from typing import List, Dict, Optional
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
        self.patterns = {
            'block': r'\$\$(.*?)\$\$',
            'inline': r'\$(.*?)\$',
            'display': r'\\\[(.*?)\\\]',
            'inline_tex': r'\\\((.*?)\\\)'
        }

    def process_content(self, content: str) -> str:
            try:
                equations = self.extract_equations(content)
                processed_content = content

                for eq in equations:
                    rendered = self.process_equation(eq['latex'], eq['type'] == 'block')
                    processed_content = processed_content.replace(eq['full'], rendered)

                return processed_content

            except Exception as e:
                self.logger.error(f"LaTeX processing error: {str(e)}")
                return content

    def process_equation(self, latex: str, block: bool = False) -> str:
        try:
            clean_latex = latex.strip()
            escaped_latex = (
                clean_latex
                .replace('"', '&quot;')
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
            )

            elem_type = 'div' if block else 'span'
            class_name = 'math-block' if block else 'math-tex'

            # Use a placeholder that won't be affected by markdown processing
            return f'LATEX_EQUATION_{class_name}_{escaped_latex}_END_LATEX'

        except Exception as e:
            self.logger.error(f"LaTeX equation processing error: {str(e)}")
            return f'<span class="latex-error">{latex}</span>'

    def extract_equations(self, content: str) -> List[Dict[str, str]]:
        equations = []

        # Find block equations ($$...$$)
        block_pattern = r'\$\$(.*?)\$\$'
        for match in re.finditer(block_pattern, content, re.DOTALL):
            equations.append({
                'type': 'block',
                'latex': match.group(1).strip(),
                'full': match.group(0)
            })
            self.logger.debug(f"Found block equation: {match.group(1)[:50]}")

        # Find inline equations ($...$)
        inline_pattern = r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)'
        for match in re.finditer(inline_pattern, content):
            equations.append({
                'type': 'inline',
                'latex': match.group(1).strip(),
                'full': match.group(0)
            })
            self.logger.debug(f"Found inline equation: {match.group(1)[:50]}")

        return equations


    def validate_equation(self, latex: str) -> bool:
        return self.validator.validate_equation(latex)

    def post_process_content(self, content: str) -> str:
            """Convert placeholders back to HTML after markdown processing"""
            def replace_placeholder(match):
                class_name = match.group(1)
                latex = match.group(2)
                elem_type = 'div' if class_name == 'math-block' else 'span'
                return f'<{elem_type} class="{class_name}" data-latex-source="{latex}">{latex}</{elem_type}>'

            pattern = r'LATEX_EQUATION_(math-(?:block|tex))_(.+?)_END_LATEX'
            return re.sub(pattern, replace_placeholder, content)
