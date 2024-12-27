# app/dita/processors/markdown_processor.py

from pathlib import Path
from typing import Dict, List, Optional, Any

# Base processor and strategy
from .base_processor import BaseProcessor

# Core managers
from ..event_manager import EventManager
from ..context_manager import ContextManager
from ..config_manager import ConfigManager
from ..metadata.metadata_manager import MetadataManager

# Utils
from ..utils.cache import ContentCache
from ..utils.id_handler import DITAIDHandler
from ..utils.html_helpers import HTMLHelper
from ..utils.heading import HeadingHandler
from ..utils.logger import DITALogger

# Types
from ..models.types import (
    TrackedElement,
    ProcessedContent,
    ProcessingMetadata,
    ProcessingContext,
    ElementType,
    MDElementType,
    MDElementInfo
)

class MarkdownProcessor(BaseProcessor):
    """Processor for Markdown content with support for custom syntax."""

    class MDMapStrategy(BaseProcessor.ProcessingStrategy):
        """Strategy for MDITA map processing."""

        def __init__(self, processor: 'MarkdownProcessor'):
            self.processor = processor

        def process(
            self,
            element: TrackedElement,
            context: ProcessingContext,
            metadata: ProcessingMetadata,
            rules: Dict[str, Any]
        ) -> ProcessedContent:
            """Process MDITA map elements."""
            # Handle MDITA map-specific metadata
            map_metadata = {
                "type": "mdita_map",
                "frontmatter": element.metadata.get("frontmatter", {}),
                "topics": element.topics,
                "context": context.to_dict(),
                "references": metadata.references
            }

            return ProcessedContent(
                element_id=element.id,
                html="",  # No transformation here
                metadata=map_metadata
            )

    class MarkdownTopicStrategy(BaseProcessor.ProcessingStrategy):
        """Strategy for Markdown topic processing."""

        def __init__(self, processor: 'MarkdownProcessor'):
            self.processor = processor

        def process(
            self,
            element: TrackedElement,
            context: ProcessingContext,
            metadata: ProcessingMetadata,
            rules: Dict[str, Any]
        ) -> ProcessedContent:
            """Process Markdown topic elements."""
            # Handle Markdown-specific metadata
            md_metadata = {
                "type": "markdown",
                "frontmatter": element.metadata.get("frontmatter", {}),
                "context": context.to_dict(),
                "references": metadata.references
            }

            return ProcessedContent(
                element_id=element.id,
                html="",  # No transformation here
                metadata=md_metadata
            )

    class MarkdownCalloutStrategy(BaseProcessor.ProcessingStrategy):
        """Strategy for Obsidian-style callouts."""

        def __init__(self, processor: 'MarkdownProcessor'):
            self.processor = processor

        def process(
            self,
            element: TrackedElement,
            context: ProcessingContext,
            metadata: ProcessingMetadata,
            rules: Dict[str, Any]
        ) -> ProcessedContent:
            """Process Markdown callout elements."""
            # Handle callout-specific metadata
            callout_metadata = {
                "type": "callout",
                "callout_type": element.metadata.get("callout_type", "note"),
                "context": context.to_dict(),
                "references": metadata.references
            }

            return ProcessedContent(
                element_id=element.id,
                html="",  # No transformation here
                metadata=callout_metadata
            )

    def _initialize_strategies(self) -> None:
        """Initialize Markdown-specific processing strategies."""
        self._strategies[ElementType.MARKDOWN] = self.MarkdownTopicStrategy(self)
        self._strategies[ElementType.NOTE] = self.MarkdownCalloutStrategy(self)
        self._strategies[ElementType.MAP] = self.MDMapStrategy(self)

    def process_file(self, file_path: Path) -> ProcessedContent:
        """Process a Markdown file."""
        try:
            # For Markdown, we need to determine if it's an MDITA map
            if self._is_mdita_map(file_path):
                return self.process_map(file_path)
            else:
                return self.process_topic(file_path)

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

    def _is_mdita_map(self, file_path: Path) -> bool:
        """Check if file is an MDITA map."""
        try:
            if not file_path.exists():
                return False

            if file_path.suffix.lower() == '.md':
                content = file_path.read_text()
                # Check for YAML frontmatter and list structure
                return (
                    content.startswith("---") and
                    "---" in content[3:] and
                    any(line.strip().startswith("- [") for line in content.splitlines())
                )

            return False

        except Exception:
            return False

    def process_topic(self, topic_path: Path) -> ProcessedContent:
        """Process a Markdown topic."""
        # Create tracked element for markdown topic
        element = TrackedElement.from_discovery(
            path=topic_path,
            element_type=ElementType.MARKDOWN,
            id_handler=self.id_handler
        )

        # Extract frontmatter if present
        frontmatter = self._extract_frontmatter(element)
        if frontmatter:
            element.metadata["frontmatter"] = frontmatter

        # Process using base processor's element processing
        return self.process_element(element)

    def process_map(self, map_path: Path) -> ProcessedContent:
        """Process an MDITA map."""
        try:
            # Create tracked element for markdown map
            element = TrackedElement.create_map(
                path=map_path,
                title="",  # Will be extracted from frontmatter
                id_handler=self.id_handler
            )

            # Extract YAML frontmatter
            frontmatter = self._extract_frontmatter(element)
            if frontmatter:
                element.metadata["frontmatter"] = frontmatter
                element.title = frontmatter.get("title", "")

            # Extract topics from markdown list structure
            self._extract_topics(element)

            # Process using base processor's element processing
            return self.process_element(element)

        except Exception as e:
            self.logger.error(f"Error processing MDITA map: {str(e)}")
            raise

    def _extract_topics(self, element: TrackedElement) -> None:
        """Extract topics from markdown list structure."""
        try:
            content = element.content.split("---", 2)[-1].strip()  # Skip frontmatter
            topics = []

            # Process markdown list items
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("- [") and "](" in line and ")" in line:
                    # Extract href from markdown link
                    href = line[line.index("](") + 2:line.index(")")]
                    topics.append(href)

            element.topics = topics

        except Exception as e:
            self.logger.error(f"Error extracting topics: {str(e)}")

    def _extract_frontmatter(self, element: TrackedElement) -> Dict[str, Any]:
        """Extract and parse YAML frontmatter."""
        try:
            import yaml

            content = element.content.strip()
            if content.startswith("---"):
                # Find end of frontmatter
                end_index = content.find("---", 3)
                if end_index != -1:
                    frontmatter = content[3:end_index].strip()
                    return yaml.safe_load(frontmatter)

            return {}

        except Exception as e:
            self.logger.error(f"Error extracting frontmatter: {str(e)}")
            return {}
