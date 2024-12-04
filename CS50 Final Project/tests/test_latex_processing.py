import pytest
from dita.utils.latex.latex_processor import DitaLaTeXProcessor
from dita.utils.latex.latex_validator import LaTeXValidator

@pytest.fixture
def latex_processor():
   return DitaLaTeXProcessor()

@pytest.fixture
def validator():
   return LaTeXValidator()

def test_inline_equation(latex_processor):
   content = "This is an inline equation $E=mc^2$ in text."
   result = latex_processor.process_content(content)
   assert '<span class="math-tex">' in result
   assert 'E=mc^2' in result

def test_block_equation(latex_processor):
   content = "Here's a block equation:\n$$\\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$$\n"
   result = latex_processor.process_content(content)
   assert '<div class="math-block">' in result

def test_invalid_latex(validator):
   invalid_latex = "\\frac{1}{2"  # Missing brace
   assert not validator.validate_equation(invalid_latex)

def test_mixed_content(latex_processor):
   content = """
   # Test Heading
   Inline math $\\alpha + \\beta$ and block math:
   $$\\int_0^\\infty e^{-x} dx$$
   """
   result = latex_processor.process_content(content)
   assert '<span class="math-tex">' in result
   assert '<div class="math-block">' in result
