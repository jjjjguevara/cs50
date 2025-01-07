# app/dita/artifacts/renderer.py
from pathlib import Path
from typing import Dict, Any, Optional
import json
from lxml import etree
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

class ArtifactRenderer:
    def __init__(self, artifacts_root: Path):
        self.artifacts_root = artifacts_root
        self.logger = logging.getLogger(__name__)
        self.parser = etree.XMLParser(
            recover=True,
            remove_blank_text=True,
            resolve_entities=False,
            dtd_validation=False,
            load_dtd=False,
            no_network=True
        )

    def render_artifact(self, artifact_path: Path, target_id: str) -> str:
        """Render an artifact as web component with proper context"""
        try:
            self.logger.info(f"Rendering artifact: {artifact_path}")

            # Create base context
            context: Dict[str, Any] = {
                'targetHeading': target_id,
                'artifactPath': str(artifact_path)
            }

            # For JSX components
            if artifact_path.suffix == '.jsx':
                component_name = artifact_path.stem
                self.logger.info(f"Creating web component for: {component_name}")
                return self._render_react_component(artifact_path, context)

            # For DITA content
            elif artifact_path.suffix == '.dita':
                return self._render_dita_content(artifact_path, context)

            else:
                self.logger.warning(f"Unsupported artifact type: {artifact_path.suffix}")
                return ""

        except Exception as e:
            self.logger.error(f"Error rendering artifact: {e}")
            return ""

    def _render_react_component(self, component_path: Path, context: Dict[str, Any]) -> str:
        """Render a React component as a web component wrapper"""
        try:
            component_name = component_path.stem
            context_json = json.dumps(context)

            return f"""
            <div class="artifact-wrapper" data-artifact-type="react">
                <web-component-wrapper
                    component="{component_name}"
                    context='{context_json}'
                    target-heading="{context['targetHeading']}"
                ></web-component-wrapper>
            </div>
            """
        except Exception as e:
            self.logger.error(f"Error rendering React component: {e}")
            return ''

    def _render_dita_content(self, dita_path: Path, context: Dict[str, Any]) -> str:
        """Parse DITA content into sections for ScrollspyContent"""
        try:
            tree = etree.parse(str(dita_path), self.parser)
            sections = []

            for section in tree.xpath('//section'):
                section_id = section.get('id', '')
                title = section.find('title').text if section.find('title') is not None else ''

                # Convert section content to HTML, excluding the title
                for title_elem in section.findall('title'):
                    section.remove(title_elem)
                content = etree.tostring(section, encoding='unicode', method='html')

                sections.append({
                    'id': section_id,
                    'title': title,
                    'content': content
                })

            # Get the main title
            main_title = tree.find('//title')
            nav_title = main_title.text if main_title is not None else dita_path.stem

            # Create ScrollspyContent context
            scrollspy_context = {
                'sections': sections,
                'navTitle': nav_title,
                'navId': f"nav-{dita_path.stem}",
                'targetHeading': context.get('targetHeading', '')
            }

            return f"""
            <div class="artifact-wrapper" data-artifact-type="react">
                <web-component-wrapper
                    component="ScrollspyContent"
                    context='{json.dumps(scrollspy_context)}'
                    target-heading="{context['targetHeading']}"
                ></web-component-wrapper>
            </div>
            """
        except Exception as e:
            self.logger.error(f"Error rendering DITA content: {e}", exc_info=True)
            return ''
