import re
import json
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from bs4 import BeautifulSoup

@dataclass
class Author:
    family: str
    given: str

@dataclass
class Citation:
    type: str
    id: str
    number: str
    author: List[Author]
    issued: Dict[str, int]
    title: str
    container_title: str
    volume: str
    issue: Optional[str]
    page: str

class CitationParser:
    """Parser for extracting and processing citations from HTML content"""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def parse_html_content(self, html_content: str) -> Dict:
        """Parse HTML content and extract citations"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Find all footnote references in the content
            citations = []
            footnotes = self._extract_footnotes(soup)

            for number, text in footnotes.items():
                citation = self._parse_citation(number, text)
                if citation:
                    citations.append(citation)

            return {
                "references": citations,
                "schema": "https://resource.citationstyles.org/schema/v1/csl-data.json"
            }

        except Exception as e:
            self.logger.error(f"Error parsing HTML content: {e}")
            return {"references": [], "schema": "https://resource.citationstyles.org/schema/v1/csl-data.json"}

    def _extract_footnotes(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract footnotes from HTML content"""
        footnotes = {}

        # Look for markdown-style footnotes [^1]: citation text
        footnote_pattern = re.compile(r'\[\^(\d+)\]:\s*(.*?)(?=\[\^|$)', re.DOTALL)
        text = soup.get_text()
        matches = footnote_pattern.finditer(text)

        for match in matches:
            number = match.group(1)
            citation_text = match.group(2).strip()
            footnotes[number] = citation_text

        return footnotes

    def _parse_citation(self, number: str, text: str) -> Optional[Dict]:
        """Parse a citation text into structured data"""
        try:
            # Regular expressions for parsing different parts of the citation
            author_pattern = r'^((?:[A-Z][a-z]+(?:,\s[A-Z]\.(?:\s[A-Z]\.)?)?(?:\s&\s|,\s)?)+)'
            year_pattern = r'\((\d{4})\)'
            title_pattern = r'\.\s([^\.]+)\.'
            container_pattern = r'_([^_]+)_'
            volume_pattern = r',\s*(\d+)'
            issue_pattern = r'\((\d+(?:Supplement \d+)?)\)'
            pages_pattern = r',\s*(\d+(?:-\d+)?)'

            # Extract authors
            author_match = re.match(author_pattern, text)
            authors_text = author_match.group(1) if author_match else ""
            authors = self._parse_authors(authors_text)

            # Extract year
            year_match = re.search(year_pattern, text)
            year = int(year_match.group(1)) if year_match else None

            # Extract title
            title_match = re.search(title_pattern, text)
            title = title_match.group(1) if title_match else ""

            # Extract container title (journal/book)
            container_match = re.search(container_pattern, text)
            container_title = container_match.group(1) if container_match else ""

            # Extract volume
            volume_match = re.search(volume_pattern, text)
            volume = volume_match.group(1) if volume_match else ""

            # Extract issue
            issue_match = re.search(issue_pattern, text)
            issue = issue_match.group(1) if issue_match else None

            # Extract pages
            pages_match = re.search(pages_pattern, text)
            pages = pages_match.group(1) if pages_match else ""

            # Determine citation type based on container title properties
            citation_type = "article-journal"  # default
            if "Book" in container_title or "Research" in container_title:
                citation_type = "book-chapter"

            return {
                "type": citation_type,
                "id": f"cite-{number}",
                "number": number,
                "author": [{"family": a.family, "given": a.given} for a in authors],
                "issued": {"year": year},
                "title": title,
                "container-title": container_title,
                "volume": volume,
                "issue": issue,
                "page": pages
            }

        except Exception as e:
            self.logger.error(f"Error parsing citation {number}: {e}")
            return None

    def _parse_authors(self, authors_text: str) -> List[Author]:
        """Parse author strings into structured author objects"""
        authors = []
        # Split multiple authors
        author_list = re.split(r',\s&\s|\s&\s|,\s(?=[A-Z])', authors_text.strip())

        for author in author_list:
            if author:
                # Split family and given names
                parts = author.strip().split(', ')
                if len(parts) == 2:
                    family, given = parts
                    authors.append(Author(family=family, given=given))
                elif len(parts) == 1:
                    # Handle case where there's no comma
                    names = parts[0].split()
                    if len(names) > 1:
                        family = names[-1]
                        given = ' '.join(names[:-1])
                        authors.append(Author(family=family, given=given))

        return authors

    def save_citations(self, citations: Dict, output_path: Path) -> None:
        """Save citations to a JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(citations, f, indent=2)
            self.logger.info(f"Citations saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving citations: {e}")

def parse_citations(html_content: str, output_path: Optional[Path] = None) -> Dict:
    """
    Parse citations from HTML content and optionally save to file

    Args:
        html_content: HTML content containing citations
        output_path: Optional path to save citations JSON file

    Returns:
        Dict containing parsed citations
    """
    parser = CitationParser()
    citations = parser.parse_html_content(html_content)

    if output_path:
        parser.save_citations(citations, output_path)

    return citations
