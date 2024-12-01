from pathlib import Path
from typing import Dict, Any, Optional
import logging
import json

logger = logging.getLogger(__name__)

class ArtifactRenderer:
    def __init__(self, artifacts_root: Path):
        self.artifacts_root = artifacts_root

    def render_artifact(self, artifact_path: Path, context: Optional[Dict[str, Any]] = None) -> str:
        """Render an artifact with optional context"""
        try:
            # Handle different artifact types (React, Vue, etc)
            if artifact_path.suffix == '.jsx':
                return self._render_react_component(artifact_path, context)
            # Add more renderers as needed
            return ''
        except Exception as e:
            logger.error(f"Error rendering artifact: {e}")
            return ''

    def _render_react_component(self, component_path: Path, context: Optional[Dict[str, Any]] = None) -> str:
        """Render a React component as a web component wrapper"""
        component_name = component_path.stem
        context_json = json.dumps(context or {})  # Handle None by defaulting to an empty dict

        return f"""
        <div class="artifact-wrapper" data-artifact-type="react">
            <web-component-wrapper
                component="{component_name}"
                context='{context_json}'
            ></web-component-wrapper>
        </div>
        """
