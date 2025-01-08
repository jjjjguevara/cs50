from typing import Dict, List, Optional, Any
import re
import logging
import html
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor
from markdown.postprocessors import Postprocessor
from app.dita.utils.logger import DITALogger

from app.dita.models.types import LaTeXEquation, ProcessedEquation
from ..latex.latex_processor import LaTeXProcessor

class LaTeXPreprocessor(Preprocessor):
    def __init__(self, md, config: Dict[str, Any]):
        super().__init__(md)
        self.logger = logging.getLogger(__name__)
        self.equations: Dict[str, LaTeXEquation] = {}
        self.equation_count = 0

    def run(self, lines: List[str]) -> List[str]:
        try:
            content = '\n'.join(lines)

            # Process block equations
            content = self._process_block_equations(content)

            # Process inline equations
            content = self._process_inline_equations(content)

            return content.split('\n')
        except Exception as e:
            self.logger.error(f"LaTeX preprocessing failed: {str(e)}")
            return lines

    def _process_block_equations(self, content: str) -> str:
        pattern = r'\$\$(.*?)\$\$'

        def replace_block(match):
            eq_content = match.group(1).strip()
            self.equation_count += 1
            eq_id = f'eq-{self.equation_count}'

            # Store equation with original delimiters
            self.equations[eq_id] = LaTeXEquation(
                id=eq_id,
                content=eq_content,
                is_block=True
            )

            # Return block with original content
            return f'<div class="katex-block">{match.group(0)}</div>'

        return re.sub(pattern, replace_block, content, flags=re.DOTALL)

    def _process_inline_equations(self, content: str) -> str:
        pattern = r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)'

        def replace_inline(match):
            eq_content = match.group(1).strip()
            self.equation_count += 1
            eq_id = f'eq-{self.equation_count}'

            # Store equation with original delimiters
            self.equations[eq_id] = LaTeXEquation(
                id=eq_id,
                content=eq_content,
                is_block=False
            )

            # Return inline with original content
            return f'<span class="katex-inline">{match.group(0)}</span>'

        return re.sub(pattern, replace_inline, content)


class LaTeXPostprocessor(Postprocessor):
    """Postprocessor to replace equation placeholders with rendered content."""

    def __init__(self, md, equations: Optional[List[LaTeXEquation]] = None):
        super().__init__(md)
        self.logger = logging.getLogger(__name__)
        self.equations: Dict[str, LaTeXEquation] = {}
        if equations:
            self.equations = {eq.id: eq for eq in equations}

    def run(self, text: str) -> str:
        """
        Replace equation placeholders with rendered LaTeX.

        Args:
            text: Content with equation placeholders

        Returns:
            Content with rendered equations
        """
        try:
            def replace_equation(match):
                equation_id = match.group(1)
                equation = self.equations.get(equation_id)
                if equation:
                    content = equation.content
                    if equation.is_block:
                        return f'<div class="katex-display" data-equation-id="{equation_id}">$${content}$$</div>'
                    return f'<span class="katex-inline" data-equation-id="{equation_id}">${content}$</span>'
                return match.group(0)

            pattern = r'<latex-equation id="([^"]+)"[^>]*></latex-equation>'
            return re.sub(pattern, replace_equation, text)

        except Exception as e:
            self.logger.error(f"LaTeX postprocessing failed: {str(e)}")
            return text


class LaTeXExtension(Extension):
    """Markdown extension for LaTeX equation processing."""

    def __init__(self, **kwargs):
        # Define valid config options
        self.config = {
            'enable_numbering': [False, 'Enable equation numbering'],
            'enable_references': [False, 'Enable equation references'],
            'preserve_delimiters': [True, 'Preserve LaTeX delimiters']
        }
        self.preprocessor = None
        self.postprocessor = None

        # Initialize parent with our config
        super().__init__(**kwargs)

    def extendMarkdown(self, md):
        """Add LaTeX processors to Markdown pipeline."""
        try:
            # Create preprocessor with our config
            self.preprocessor = LaTeXPreprocessor(md, self.getConfigs())
            md.preprocessors.register(self.preprocessor, 'latex', 175)

            # Create postprocessor
            self.postprocessor = LaTeXPostprocessor(md, [])
            md.postprocessors.register(self.postprocessor, 'latex', 175)

            # Register the extension instance
            md.registerExtension(self)

        except Exception as e:
            logging.error(f"Error extending Markdown with LaTeX: {str(e)}")
            raise

    def reset(self):
        """Reset the extension state."""
        if self.preprocessor:
            self.preprocessor.equations.clear()
        if self.postprocessor:
            self.postprocessor.equations.clear()

    def update_equations(self):
        """Update postprocessor equations from preprocessor."""
        if self.preprocessor and self.postprocessor:
            self.postprocessor.equations = {
                eq.id: eq for eq in self.preprocessor.equations.values()
            }


def makeExtension(**kwargs):
    """Create LaTeX extension."""
    return LaTeXExtension(**kwargs)
