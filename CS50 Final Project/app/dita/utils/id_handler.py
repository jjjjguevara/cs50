from typing import Dict, List, Optional, Set, Union, Pattern, Any
from pathlib import Path
import re
import logging
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

# Import types
from ..models.types import (
    ProcessingPhase,
    ProcessingState,
    ElementType,
    ValidationResult,
    ValidationMessage,
    ValidationSeverity
)

# Global config
from app_config import DITAConfig

class IDType(Enum):
    """Types of IDs that require different patterns."""
    MAP = "map"
    TOPIC = "topic"
    HEADING = "heading"
    ARTIFACT = "artifact"
    FIGURE = "figure"
    TABLE = "table"
    FORMULA = "formula"
    EQUATION = "equation"
    CITATION = "citation"
    SECTION = "section"
    APPENDIX = "appendix"
    SUPPLEMENTAL = "supplemental"
    REFERENCE = "reference"
    HTML_ELEMENT = "element"
    CACHE_ENTRY = "cache"
    METADATA = "meta"
    EVENT = "event"
    STATE = "state"
    TRANSFORM = "transform"
    VALIDATION = "validation"

@dataclass
class IDMetadata:
    """Metadata for generated IDs."""
    id_type: IDType
    base_text: str
    context: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    validation_pattern: Optional[Pattern] = None

class DITAIDHandler:
    """Handler for DITA ID generation and validation."""

    def __init__(self):
        """Initialize ID handler."""
        # Core setup
        self.logger = logging.getLogger(__name__)

        # ID tracking
        self._generated_ids: Set[str] = set()
        self._id_metadata: Dict[str, IDMetadata] = {}

        # Pattern caching
        self._validation_patterns: Dict[IDType, Pattern] = {}

        # Statistics
        self._stats = {
            "ids_generated": 0,
            "validation_failures": 0,
            "pattern_cache_hits": 0,
            "id_collisions": 0
        }

    def generate_id(
        self,
        base: Union[str, Path],
        id_type: IDType,
        context: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Generate a unique ID based on type and context.

        Args:
            base: Base text or path for ID
            id_type: Type of ID to generate
            context: Optional context for ID generation
            kwargs: Additional parameters for specific ID types

        Returns:
            str: Generated unique ID
        """
        try:
            # Convert Path to string if needed
            base_text = base.stem if isinstance(base, Path) else base

            # Sanitize base text
            sanitized_base = self._sanitize_base(base_text)

            # Construct ID based on type
            if id_type == IDType.HEADING:
                level = kwargs.get('level', 1)
                candidate_id = f"heading-{sanitized_base}-h{level}"
            elif id_type == IDType.ARTIFACT:
                target = kwargs.get('target', '')
                candidate_id = f"artifact-{sanitized_base}-{self._sanitize_base(target)}"
            else:
                candidate_id = f"{id_type.value}-{sanitized_base}"

            # Ensure uniqueness
            unique_id = self._ensure_unique(candidate_id)

            # Store metadata
            self._id_metadata[unique_id] = IDMetadata(
                id_type=id_type,
                base_text=base_text,
                context=context,
                validation_pattern=self._get_validation_pattern(id_type)
            )

            # Update stats
            self._stats["ids_generated"] += 1

            return unique_id

        except Exception as e:
            self.logger.error(f"Error generating ID: {str(e)}")
            raise

    def _sanitize_base(self, base: str) -> str:
        """
        Sanitize base text for ID generation.

        Args:
            base: Text to sanitize

        Returns:
            str: Sanitized text
        """
        try:
            # Remove unwanted characters
            sanitized = re.sub(r"[^\w\-]", "-", base.lower())

            # Clean up multiple dashes
            sanitized = re.sub(r"-+", "-", sanitized)

            # Trim dashes from ends
            return sanitized.strip("-")

        except Exception as e:
            self.logger.error(f"Error sanitizing base text: {str(e)}")
            return "id"

    def _ensure_unique(self, candidate_id: str) -> str:
        """
        Ensure ID uniqueness by appending number if needed.

        Args:
            candidate_id: Proposed ID

        Returns:
            str: Unique ID
        """
        try:
            if candidate_id not in self._generated_ids:
                self._generated_ids.add(candidate_id)
                return candidate_id

            # Handle collision
            self._stats["id_collisions"] += 1
            counter = 1
            while f"{candidate_id}-{counter}" in self._generated_ids:
                counter += 1

            unique_id = f"{candidate_id}-{counter}"
            self._generated_ids.add(unique_id)
            return unique_id

        except Exception as e:
            self.logger.error(f"Error ensuring unique ID: {str(e)}")
            return f"{candidate_id}-{datetime.now().timestamp()}"

    def _get_validation_pattern(self, id_type: IDType) -> Pattern:
        """
        Get or compile validation pattern for ID type.

        Args:
            id_type: Type of ID

        Returns:
            Pattern: Compiled regex pattern
        """
        try:
            # Check cache
            if id_type in self._validation_patterns:
                self._stats["pattern_cache_hits"] += 1
                return self._validation_patterns[id_type]

            # Get pattern string based on type
            pattern_str = self._get_pattern_string(id_type)

            # Compile and cache pattern
            pattern = re.compile(pattern_str)
            self._validation_patterns[id_type] = pattern

            return pattern

        except Exception as e:
            self.logger.error(f"Error getting validation pattern: {str(e)}")
            return re.compile(r"^[a-z][a-z0-9_\-]*$")

    def _get_pattern_string(self, id_type: IDType) -> str:
        """Get validation pattern string for ID type."""
        patterns = {
            IDType.MAP: r"^map-[a-zA-Z0-9_\-]+$",
            IDType.TOPIC: r"^topic-[a-zA-Z0-9_\-]+$",
            IDType.HEADING: r"^heading-[a-zA-Z0-9_\-]+-h[1-6]$",
            IDType.ARTIFACT: r"^artifact-[a-zA-Z0-9_\-]+-[a-zA-Z0-9_\-]+$",
            IDType.FIGURE: r"^fig-[a-zA-Z0-9_\-]+$",
            IDType.TABLE: r"^tbl-[a-zA-Z0-9_\-]+$",
            IDType.FORMULA: r"^formula-[a-zA-Z0-9_\-]+$",
            IDType.EQUATION: r"^eq-[a-zA-Z0-9_\-]+$",
            IDType.CITATION: r"^cite-[a-zA-Z0-9_\-]+$",
            IDType.SECTION: r"^sec-[a-zA-Z0-9_\-]+$",
            IDType.APPENDIX: r"^app-[a-zA-Z0-9_\-]+$",
            IDType.SUPPLEMENTAL: r"^supp-[a-zA-Z0-9_\-]+$",
            IDType.REFERENCE: r"^ref-[a-zA-Z0-9_\-]+$"
        }
        return patterns.get(id_type, r"^[a-z][a-z0-9_\-]*$")

    def validate_id(self, id_str: str, id_type: Optional[IDType] = None) -> ValidationResult:
        """
        Validate an ID string.

        Args:
            id_str: ID to validate
            id_type: Optional type for specific validation

        Returns:
            ValidationResult with validation status and messages
        """
        try:
            messages: List[ValidationMessage] = []

            # Get metadata if available
            metadata = self._id_metadata.get(id_str)

            # Determine type and pattern
            if id_type is None and metadata:
                id_type = metadata.id_type

            if id_type:
                pattern = self._get_validation_pattern(id_type)
                if not pattern.match(id_str):
                    messages.append(
                        ValidationMessage(
                            path=id_str,
                            message=f"Invalid {id_type.value} ID format",
                            severity=ValidationSeverity.ERROR,
                            code="invalid_id_format"
                        )
                    )
            else:
                # Basic validation for unknown types
                if not re.match(r"^[a-z][a-z0-9_\-]*$", id_str):
                    messages.append(
                        ValidationMessage(
                            path=id_str,
                            message="Invalid ID format",
                            severity=ValidationSeverity.ERROR,
                            code="invalid_id_format"
                        )
                    )

            is_valid = len(messages) == 0
            if not is_valid:
                self._stats["validation_failures"] += 1

            return ValidationResult(
                is_valid=is_valid,
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating ID: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[
                    ValidationMessage(
                        path=id_str,
                        message=str(e),
                        severity=ValidationSeverity.ERROR,
                        code="validation_error"
                    )
                ]
            )

    def get_id_metadata(self, id_str: str) -> Optional[IDMetadata]:
        """Get metadata for an ID."""
        return self._id_metadata.get(id_str)

    def get_stats(self) -> Dict[str, int]:
        """Get ID handler statistics."""
        return self._stats.copy()

    def cleanup(self) -> None:
        """Clean up handler resources."""
        try:
            self._generated_ids.clear()
            self._id_metadata.clear()
            self._validation_patterns.clear()
            self._stats = {
                "ids_generated": 0,
                "validation_failures": 0,
                "pattern_cache_hits": 0,
                "id_collisions": 0
            }
            self.logger.debug("ID handler cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise
