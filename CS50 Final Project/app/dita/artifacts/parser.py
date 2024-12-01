from pathlib import Path
from typing import Dict, List, Optional
import logging
from lxml import etree

logger = logging.getLogger(__name__)

class ArtifactParser:
    def __init__(self, dita_root: Path):
        self.dita_root = dita_root

    def parse_artifact_references(self, ditamap_path: Path) -> List[Dict]:
        """Parse artifact references from a DITAMAP file"""
        artifacts = []
        try:
            tree = etree.parse(str(ditamap_path))
            for topicref in tree.xpath('//topicref'):
                if href_append := topicref.get('href_append'):
                    append_location = topicref.get('append_location', '')
                    artifacts.append({
                        'source': href_append,
                        'target_heading': append_location.split('#')[1] if '#' in append_location else None,
                        'topic_ref': topicref.get('href')
                    })
            return artifacts
        except Exception as e:
            logger.error(f"Error parsing artifacts: {e}")
            return []
