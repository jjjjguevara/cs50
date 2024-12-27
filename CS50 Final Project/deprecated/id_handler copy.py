from pathlib import Path
from typing import Dict, List, Optional, Set, Union
import re
import logging

# Global config
from app_config import DITAConfig

class DITAIDHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.generated_ids = set()


    def configure(self, config: DITAConfig) -> None:
        """
        Configure the ID handler with provided settings.

        Args:
            config: A DITAConfig object containing configuration settings.
        """
        try:
            self.logger.debug("Configuring ID handler with provided settings")

            # Dynamically apply relevant configurations
            if hasattr(config, "validation_patterns"):
                self.validation_patterns = config.validation_patterns
                self.logger.debug(f"Updated validation patterns: {self.validation_patterns}")

            if hasattr(config, "reset_generated_ids") and config.reset_generated_ids:
                self.generated_ids.clear()
                self.logger.debug("Reset generated IDs")

            # Extend to handle additional configuration options as needed

            self.logger.debug("ID handler configuration completed successfully")

        except Exception as e:
            self.logger.error(f"ID handler configuration failed: {str(e)}")
            raise


    def generate_id(self, base: Union[str, Path], element_type: Optional[str] = None, **kwargs) -> str:
        """
        Generate a unique ID by prepending the element type and validating the result.

        Args:
            base: The base string or Path for generating the ID.
            element_type: The type of element (e.g., 'map', 'topic', 'heading', 'artifact').
            kwargs: Additional parameters (e.g., level for headings, target_heading for artifacts).

        Returns:
            str: A unique, sanitized ID string.
        """
        if isinstance(base, Path):
            base = base.stem

        # Prepend element type and sanitize
        sanitized_base = self._sanitize_base(base)
        candidate_id = f"{element_type}-{sanitized_base}" if element_type else sanitized_base

        # Add optional parameters like heading levels or target heading for artifacts
        if element_type == "heading" and "level" in kwargs:
            candidate_id = f"{candidate_id}-h{kwargs['level']}"
        elif element_type == "artifact" and "target_heading" in kwargs:
            target_heading = self._sanitize_base(kwargs["target_heading"])
            candidate_id = f"{candidate_id}-{target_heading}"

        # Ensure the ID is valid for the given element type
        if not self._validate_id(candidate_id, element_type):
            raise ValueError(f"Generated ID '{candidate_id}' is invalid for type '{element_type}'")

        # Ensure uniqueness and return
        unique_id = self._ensure_unique(candidate_id)
        self.logger.debug(f"Generated ID: {unique_id} for element_type={element_type}")
        return unique_id

    def _validate_id(self, candidate_id: str, element_type: Optional[str]) -> bool:
        """
        Validate the generated ID based on element type.

        Args:
            candidate_id: The ID to validate.
            element_type: The type of element (e.g., 'map', 'topic', 'heading', 'artifact').

        Returns:
            bool: True if the ID is valid, otherwise False.
        """
        # Define validation rules for each element type
        validation_patterns = {
            "map": r"^map-[a-zA-Z0-9_\-]+$",
            "topic": r"^topic-[a-zA-Z0-9_\-]+$",
            "heading": r"^heading-[a-zA-Z0-9_\-]+-h[1-6]$",
            "artifact": r"^artifact-[a-zA-Z0-9_\-]+-[a-zA-Z0-9_\-]+$",
            None: r"^[a-zA-Z0-9_\-]+$",  # Default validation
        }

        pattern = validation_patterns.get(element_type, validation_patterns[None])
        is_valid = bool(re.match(pattern, candidate_id))
        if not is_valid:
            self.logger.warning(f"Invalid ID '{candidate_id}' for type '{element_type}'")
        return is_valid

    def resolve_xref(self, source_id: str, target_ref: str) -> str:
            """
            Resolve cross-references between topics and headings.

            Args:
                source_id: The ID of the source topic.
                target_ref: The target reference, potentially including a topic and/or heading ID.

            Returns:
                str: A resolved URL or anchor link.
            """
            if "#" in target_ref:
                topic_ref, heading_ref = target_ref.split("#", 1)
                sanitized_topic_ref = self._sanitize_base(topic_ref) if topic_ref else ""
                sanitized_heading_ref = self._sanitize_base(heading_ref)
                if topic_ref:
                    return f"/entry/{sanitized_topic_ref}#{sanitized_heading_ref}"
                return f"#{sanitized_heading_ref}"
            sanitized_target_ref = self._sanitize_base(target_ref)
            return f"/entry/{sanitized_target_ref}"

    def _sanitize_base(self, base: str) -> str:
        """
        Sanitize the base string to create a web-safe ID.

        Args:
            base: The base string to sanitize.

        Returns:
            A sanitized ID string.
        """
        return re.sub(r"[^\w\-]+", "-", base).strip("-")

    def _ensure_unique(self, candidate_id: str) -> str:
        """
        Ensure an ID is unique by appending a counter if needed.

        Args:
            candidate_id: The candidate ID.

        Returns:
            A unique ID string.
        """
        original_id = candidate_id
        counter = 1
        while candidate_id in self.generated_ids:
            candidate_id = f"{original_id}-{counter}"
            counter += 1
        self.generated_ids.add(candidate_id)
        return candidate_id
