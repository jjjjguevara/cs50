import sys
from pathlib import Path
import pytest

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.dita.markdown.parser import MarkdownParser

# Setup fixtures
@pytest.fixture
def markdown_parser():
    """Create a fresh markdown parser instance for each test"""
    return MarkdownParser()

@pytest.fixture
def sample_markdown():
    """Sample markdown content with YAML front matter"""
    return """---
title: Test Document
type: concept
authors:
  - John Doe
  - Jane Smith
keywords:
  - test
  - markdown
  - dita
---

# Main Title

This is a test paragraph with some **bold** and *italic* text.

## Section 1

* List item 1
* List item 2

## Section 2

1. Numbered item 1
2. Numbered item 2
"""

def test_markdown_parser_initialization(markdown_parser):
    """Test that markdown parser initializes correctly"""
    assert markdown_parser is not None
    assert markdown_parser.md is not None
    assert markdown_parser.yaml_ext is not None

def test_parse_content_metadata(markdown_parser, sample_markdown):
    """Test that metadata is correctly extracted from markdown content"""
    metadata, _ = markdown_parser.parse_content(sample_markdown)

    assert metadata is not None
    assert metadata.get('title') == 'Test Document'
    assert metadata.get('type') == 'concept'
    assert 'authors' in metadata
    assert len(metadata['authors']) == 2
    assert 'keywords' in metadata
    assert len(metadata['keywords']) == 3

def test_parse_content_html(markdown_parser, sample_markdown):
    """Test that markdown content is correctly converted to HTML with DITA classes"""
    _, html = markdown_parser.parse_content(sample_markdown)

    assert html is not None
    # Check for DITA-specific classes
    assert '<h1 class="dita-title">' in html
    assert '<h2 class="dita-section-title">' in html
    assert '<ul class="dita-ul">' in html
    assert '<ol class="dita-ol">' in html
    # Check for basic HTML elements
    assert '<strong>' in html
    assert '<em>' in html
    assert '<li>' in html

    # Check actual content
    assert 'Main Title' in html
    assert 'Section 1' in html
    assert 'Section 2' in html
    assert 'List item 1' in html
    assert 'Numbered item 1' in html
    assert 'bold' in html
    assert 'italic' in html

def test_metadata_validation(markdown_parser):
    """Test metadata validation"""
    valid_metadata = {
        'title': 'Test',
        'type': 'concept'
    }
    assert markdown_parser.validate_metadata(valid_metadata) is True

    invalid_metadata = {
        'title': 'Test'
        # missing 'type'
    }
    assert markdown_parser.validate_metadata(invalid_metadata) is False

def test_error_handling(markdown_parser):
    """Test error handling for invalid content"""
    metadata, html = markdown_parser.parse_content("Invalid YAML\n---\nContent")
    assert metadata == {}
    assert html != ""  # Should still try to render the content

if __name__ == '__main__':
    pytest.main([__file__])
