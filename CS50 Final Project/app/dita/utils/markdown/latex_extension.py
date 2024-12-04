# app/dita/utils/markdown/latex_extension.py
import re
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor
from ..latex.latex_processor import DitaLaTeXProcessor

class LaTeXPreprocessor(Preprocessor):
    def __init__(self, md, latex_processor):
        super().__init__(md)
        self.latex_processor = latex_processor

    def run(self, lines):
        content = '\n'.join(lines)
        processed = self.latex_processor.process_content(content)
        return processed.split('\n')

class LaTeXExtension(Extension):
    def __init__(self, **kwargs):
        self.config = {
            'latex_processor': [None, 'LaTeX processor instance']
        }
        super().__init__(**kwargs)

    def extendMarkdown(self, md):
        latex_processor = self.getConfig('latex_processor')
        md.preprocessors.register(
            LaTeXPreprocessor(md, latex_processor),
            'latex_equations',
            25  # High priority to run early
        )
