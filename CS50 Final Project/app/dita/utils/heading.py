import re
from typing import Set, Optional
import hashlib

class HeadingIDGenerator:
    def __init__(self):
        self.used_ids: Set[str] = set()

    def generate_id(self, heading_text: str, parent_id: Optional[str] = None) -> str:
        """Generate a unique ID for a heading"""
        # Basic sanitization
        base_id = re.sub(r'[^\w\s-]', '', heading_text.lower())
        base_id = re.sub(r'\s+', '-', base_id.strip())

        # Truncate if too long
        if len(base_id) > 50:
            base_id = base_id[:47] + '...'

        # Add parent context if available
        if parent_id is not None:  # More explicit None check
            base_id = f"{parent_id}--{base_id}"

        # Ensure uniqueness
        final_id = base_id
        counter = 1

        while final_id in self.used_ids:
            final_id = f"{base_id}-{counter}"
            counter += 1

        self.used_ids.add(final_id)
        return final_id
