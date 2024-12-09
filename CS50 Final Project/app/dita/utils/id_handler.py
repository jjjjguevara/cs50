from pathlib import Path
from typing import Dict, List, Optional, Set
import re
import logging

# Global config
from app_config import DITAConfig

class DITAIDHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.used_ids: Dict[str, int] = {}

    def configure(self, config: DITAConfig) -> None:
        """Configure ID handler with provided settings."""
        try:
            self.logger.debug("Configuring ID handler")

            # Reset used IDs if needed
            self.used_ids.clear()

            # Add any configuration-specific settings here
            # Currently just preparing for future configuration needs

            self.logger.debug("ID handler configuration completed")

        except Exception as e:
            self.logger.error(f"ID handler configuration failed: {str(e)}")
            raise

    def generate_id(self, text: str) -> str:
            """Generate a unique ID for any text."""
            base_id = re.sub(r'[^\w\- ]', '', text.lower()).strip().replace(' ', '-')
            unique_id = base_id
            counter = 1
            while unique_id in self.used_ids:
                unique_id = f"{base_id}-{counter}"
                counter += 1
            self.used_ids[unique_id] = 1
            return unique_id



    def generate_content_id(self, path: Path) -> str:
        """Generate clean ID for content file"""
        base_id = path.stem
        return self._sanitize_id(base_id)

    def generate_map_id(self, map_path: Path) -> str:
        """Generate clean ID for ditamap"""
        base_id = map_path.stem.replace('.ditamap', '')
        return self._sanitize_id(base_id)

    def generate_topic_id(self, topic_path: Path, parent_map: Optional[Path] = None) -> str:
        """Generate unique ID for topic file"""
        base_id = topic_path.stem
        if parent_map:
            map_id = self.generate_map_id(parent_map)
            base_id = f"{map_id}-{base_id}"
        return self._sanitize_id(base_id)

    def generate_artifact_id(self, name: str, target_heading: str) -> str:
        """Generate unique ID for artifact"""
        base_id = f"artifact-{name}-{target_heading}"
        return self._sanitize_id(base_id)

    def _sanitize_id(self, raw_id: str) -> str:
            """Sanitize and ensure the ID is unique."""
            sanitized_id = re.sub(r'[^\w\-]', '', raw_id.lower().strip())
            if sanitized_id in self.used_ids:
                self.used_ids[sanitized_id] += 1
                sanitized_id = f"{sanitized_id}-{self.used_ids[sanitized_id]}"
            else:
                self.used_ids[sanitized_id] = 1
            return sanitized_id

    def resolve_xref(self, source_id: str, target_ref: str) -> str:
        """Resolve cross-references between topics"""
        # Handle both internal and cross-topic references
        if '#' in target_ref:
            topic_ref, heading_ref = target_ref.split('#', 1)
            if topic_ref:
                return f"/entry/{self._sanitize_id(topic_ref)}#{heading_ref}"
            return f"#{heading_ref}"
        return f"/entry/{self._sanitize_id(target_ref)}"

    def cleanup(self) -> None:
            """Clean up ID handler resources and state."""
            try:
                self.logger.debug("Starting ID handler cleanup")

                # Clear used IDs
                self.used_ids.clear()

                self.logger.debug("ID handler cleanup completed")

            except Exception as e:
                self.logger.error(f"ID handler cleanup failed: {str(e)}")
                raise
