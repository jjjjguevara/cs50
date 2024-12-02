from pathlib import Path
from typing import Dict, List, Optional
import re
import logging

class DITAIDHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.used_ids: Dict[str, int] = {}

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

    def _sanitize_id(self, id_str: str) -> str:
        """Sanitize and ensure uniqueness of IDs"""
        clean_id = re.sub(r'[^\w\-]', '-', id_str.lower())
        clean_id = re.sub(r'-+', '-', clean_id).strip('-')

        if clean_id in self.used_ids:
            self.used_ids[clean_id] += 1
            return f"{clean_id}-{self.used_ids[clean_id]}"

        self.used_ids[clean_id] = 1
        return clean_id

    def resolve_xref(self, source_id: str, target_ref: str) -> str:
        """Resolve cross-references between topics"""
        # Handle both internal and cross-topic references
        if '#' in target_ref:
            topic_ref, heading_ref = target_ref.split('#', 1)
            if topic_ref:
                return f"/entry/{self._sanitize_id(topic_ref)}#{heading_ref}"
            return f"#{heading_ref}"
        return f"/entry/{self._sanitize_id(target_ref)}"
