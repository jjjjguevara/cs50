from typing import Dict, List, Optional, Any
import re
import logging
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor
from markdown.postprocessors import Postprocessor

from app.dita.models.types import LaTeXEquation, ProcessedEquation
from ..latex.latex_processor import LaTeXProcessor

class LaTeXPreprocessor(Preprocessor):
    """Preprocessor to identify and track LaTeX equations in Markdown."""

    def __init__(self, md, config: Dict[str, Any]):
        super().__init__(md)
        self.logger = logging.getLogger(__name__)
        self.equations: List[LaTeXEquation] = []
        self.equation_count = 0

    def generate_equation_id(self) -> str:
        """Generate unique equation ID."""
        self.equation_count += 1
        return f"eq-{self.equation_count}"

    def run(self, lines: List[str]) -> List[str]:
        """
        Process Markdown content to find LaTeX equations.

        Args:
            lines: List of content lines

        Returns:
            Processed lines with equation placeholders
        """
        try:
            new_lines = []
            content = '\n'.join(lines)

            # Find block equations
            content = self._process_block_equations(content)

            # Find inline equations
            content = self._process_inline_equations(content)

            return content.split('\n')

        except Exception as e:
            self.logger.error(f"LaTeX preprocessing failed: {str(e)}")
            return lines

    def _process_block_equations(self, content: str) -> str:
        """Process block LaTeX equations."""
        try:
            pattern = r'\$\$(.*?)\$\$'

            def replace_block(match):
                equation_content = match.group(1).strip()
                equation_id = self.generate_equation_id()

                # Create equation object
                equation = LaTeXEquation(
                    id=equation_id,
                    content=equation_content,
                    is_block=True
                )

                self.equations.append(equation)
                return f'<latex-equation id="{equation_id}"></latex-equation>'

            return re.sub(pattern, replace_block, content, flags=re.DOTALL)

        except Exception as e:
            self.logger.error(f"Block equation processing failed: {str(e)}")
            return content

    def _process_inline_equations(self, content: str) -> str:
        """Process inline LaTeX equations."""
        try:
            pattern = r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)'

            def replace_inline(match):
                equation_content = match.group(1).strip()
                equation_id = self.generate_equation_id()

                # Create equation object
                equation = LaTeXEquation(
                    id=equation_id,
                    content=equation_content,
                    is_block=False
                )

                self.equations.append(equation)
                return f'<latex-equation id="{equation_id}"></latex-equation>'

            return re.sub(pattern, replace_inline, content)

        except Exception as e:
            self.logger.error(f"Inline equation processing failed: {str(e)}")
            return content

class LaTeXPostprocessor(Postprocessor):
    """Postprocessor to replace equation placeholders with rendered content."""

    def __init__(self, md, equations: List[LaTeXEquation]):
        super().__init__(md)
        self.logger = logging.getLogger(__name__)
        self.equations = equations
        self.latex_processor = LaTeXProcessor()

    def run(self, text: str) -> str:
        """
        Replace equation placeholders with rendered LaTeX.

        Args:
            text: Content with equation placeholders

        Returns:
            Content with rendered equations
        """
        try:
            # Process all equations
            processed_equations = self.latex_processor.process_equations(self.equations)

            # Create lookup for replacement
            equation_lookup = {eq.id: eq for eq in processed_equations}

            # Replace placeholders
            def replace_equation(match):
                equation_id = match.group(1)
                if equation_id in equation_lookup:
                    return equation_lookup[equation_id].html
                return match.group(0)

            pattern = r'<latex-equation id="([^"]+)"></latex-equation>'
            return re.sub(pattern, replace_equation, text)

        except Exception as e:
            self.logger.error(f"LaTeX postprocessing failed: {str(e)}")
            return text

class LaTeXExtension(Extension):
    """Markdown extension for LaTeX equation processing."""

    def __init__(self, **kwargs):
        self.config = {
            'enable_numbering': [False, 'Enable equation numbering'],
            'enable_references': [False, 'Enable equation references']
        }
        super().__init__(**kwargs)

    def extendMarkdown(self, md):
        """Add LaTeX processors to Markdown pipeline."""
        # Create preprocessor
        latex_preprocessor = LaTeXPreprocessor(md, self.getConfigs())
        md.preprocessors.register(latex_preprocessor, 'latex', 175)

        # Create postprocessor with equations from preprocessor
        latex_postprocessor = LaTeXPostprocessor(md, latex_preprocessor.equations)
        md.postprocessors.register(latex_postprocessor, 'latex', 175)

def makeExtension(**kwargs):
    """Create LaTeX extension."""
    return LaTeXExtension(**kwargs)
