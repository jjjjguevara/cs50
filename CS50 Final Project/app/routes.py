# Standard library imports
import os
import re
import json
import logging
import traceback
from lxml import etree
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Third-party imports
from flask import (
    Blueprint,
    current_app,
    jsonify,
    request,
    Response,
    current_app,
    send_from_directory,
    render_template,
    url_for,
    redirect,
)
from flask.typing import ResponseReturnValue
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString

# Local imports
from .dita.processor import DITAProcessor
from .dita.artifacts.parser import ArtifactParser
from app.dita.utils.latex.latex_processor import DitaLaTeXProcessor


# Type aliases
FlaskResponse = Union[Response, tuple[Response, int], tuple[str, int], Any]

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize blueprint
bp = Blueprint(
    'main',
    __name__,
    url_prefix='/',
    static_folder='static',
    template_folder='templates'
)

# Initialize DITA processor
dita_processor = DITAProcessor()

# Type aliases
FlaskResponse = Union[Response, tuple[Response, int], tuple[str, int], Any]

# Main routes
# Main routes (modify the current index route and add new test route)
@bp.route('/static/dist/<path:filename>')
def serve_dist(filename: str) -> ResponseReturnValue:
    """Serve built Vite assets"""
    try:
        dist_dir = os.path.join(current_app.root_path, 'static', 'dist')
        # Handle both root files and subdirectories
        file_path = os.path.join(dist_dir, filename)
        directory = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)

        if not os.path.exists(file_path):
            logger.error(f"Dist file not found: {filename}")
            return jsonify({'error': 'File not found'}), 404

        return send_from_directory(directory, base_name)
    except Exception as e:
        logger.error(f"Error serving dist file {filename}: {str(e)}")
        return jsonify({'error': str(e)}), 500


@bp.route('/')
def index() -> ResponseReturnValue:
    """Home route redirects to roadmap"""
    try:
        logger.info("Redirecting home to roadmap")
        return redirect(url_for('main.view_entry', topic_id='roadmap'))
    except Exception as e:
        logger.error(f"Error in home redirect: {str(e)}")
        return jsonify({'error': 'Failed to load roadmap'}), 500


@bp.route('/static/topics/<path:filename>')
def serve_topic_files(filename: str) -> ResponseReturnValue:
    """Serve files from the topics directory"""
    try:
        topics_dir = Path(current_app.root_path) / 'dita' / 'topics'
        logger.info(f"Request for topics file: {filename}")
        logger.info(f"Looking in topics dir: {topics_dir}")

        # Clean the filename and construct full path
        file_path = (topics_dir / filename).resolve()
        logger.info(f"Full file path: {file_path}")

        # Security check
        if not str(file_path).startswith(str(topics_dir)):
            logger.error(f"Attempted path traversal: {filename}")
            return jsonify({'error': 'Invalid path'}), 403

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404

        logger.info(f"Serving file: {file_path}")
        # Get directory and filename for send_from_directory
        directory = str(file_path.parent)
        basename = file_path.name

        return send_from_directory(directory, basename)
    except Exception as e:
        logger.error(f"Error serving topic file {filename}: {str(e)}")
        return jsonify({'error': str(e)}), 500


@bp.route('/media/<path:filename>')
def serve_map_media(filename: str) -> ResponseReturnValue:
    """Serve media files referenced in maps"""
    try:
        # Try multiple possible media locations
        possible_paths = [
            Path(current_app.root_path) / 'dita' / 'topics' / 'cs50' / 'media' / filename,
            Path(current_app.root_path) / 'dita' / 'maps' / 'media' / filename,
            Path(current_app.root_path) / 'dita' / 'media' / filename
        ]

        for file_path in possible_paths:
            logger.info(f"Checking for media file at: {file_path}")
            if file_path.exists():
                logger.info(f"Found media file at: {file_path}")
                return send_from_directory(str(file_path.parent), file_path.name)

        logger.error(f"Media file not found: {filename}")
        return jsonify({'error': 'Media file not found'}), 404
    except Exception as e:
        logger.error(f"Error serving media file {filename}: {str(e)}")
        return jsonify({'error': str(e)}), 500


@bp.route('/academic.html')
def academic_view():
    """Serve academic view template"""
    try:
        logger.info("Serving academic view template")
        return render_template('academic.html')
    except Exception as e:
        logger.error(f"Error serving academic view: {str(e)}")
        return jsonify({'error': 'Failed to load academic view'}), 500

@bp.route('/articles')
def articles() -> ResponseReturnValue:
    """Articles route that redirects to first ditamap"""
    try:
        logger.info("Processing articles redirect")
        maps_dir = Path(current_app.root_path) / 'dita' / 'maps'

        # Get first .ditamap file
        ditamaps = list(maps_dir.glob('*.ditamap'))
        if ditamaps:
            first_map = ditamaps[0]
            map_id = first_map.stem  # Just get the stem without .ditamap
            logger.info(f"Redirecting to first ditamap: {map_id}")
            return redirect(url_for('main.view_entry', topic_id=map_id))  # Don't add .ditamap
        else:
            logger.error("No ditamaps found")
            return render_template('academic.html', error="No articles found"), 404

    except Exception as e:
        logger.error(f"Error in articles redirect: {str(e)}")
        return jsonify({'error': 'Failed to load articles'}), 500


## Artifact-specific routes
@bp.route('/api/artifacts/<path:artifact_id>')
def get_artifact(artifact_id: str) -> ResponseReturnValue:
    """Get rendered artifact content"""
    try:
        artifact_path = dita_processor.dita_root / 'artifacts' / artifact_id
        if not artifact_path.exists():
            return jsonify({'error': 'Artifact not found'}), 404

        context_str = request.args.get('context', '{}')
        try:
            context = json.loads(context_str)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid context JSON: {e}")
            return jsonify({'error': 'Invalid context format'}), 400

        rendered_content = dita_processor.artifact_renderer.render_artifact(
            artifact_path,
            context
        )
        return Response(rendered_content, mimetype='text/html')
    except Exception as e:
        logger.error(f"Error serving artifact: {str(e)}")
        return jsonify({'error': str(e)}), 500

## Artifact debug routes
@bp.route('/api/debug/ditamap/<map_id>')
def debug_ditamap(map_id: str) -> ResponseReturnValue:
    """Debug endpoint to examine DITAMAP processing"""
    try:
        map_path = Path(current_app.root_path) / 'dita' / 'maps' / f"{map_id}.ditamap"
        if not map_path.exists():
            return jsonify({'error': 'Map not found'}), 404

        # Parse the map
        parser = ArtifactParser(Path(current_app.root_path) / 'dita')
        artifacts = parser.parse_artifact_references(map_path)

        # Read raw content
        with open(map_path, 'r') as f:
            content = f.read()

        return jsonify({
            'map_id': map_id,
            'artifacts': artifacts,
            'raw_content': content
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500



## Topic test environment
@bp.route('/test')
def test_interface() -> ResponseReturnValue:
    """Test interface"""
    try:
        logger.info("Serving development index.html")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error serving index page: {str(e)}")
        return jsonify({'error': 'Failed to load application'}), 500


@bp.route('/entry/<topic_id>')
def view_entry(topic_id: str):
    try:
        logger.info(f"Processing entry request for: {topic_id}")

        if not topic_id.endswith(('.md', '.dita', '.ditamap')):
            topic_id = f"{topic_id}.ditamap"

        map_path = dita_processor.get_topic(topic_id)
        if not map_path:
            logger.error(f"Map not found: {topic_id}")
            return render_template('academic.html', error="Entry not found"), 404

        # Get the actual content file path
        tree = etree.parse(str(map_path))
        topicref = tree.find(".//topicref")
        if topicref is not None and (href := topicref.get('href')):
            content_path = Path(map_path).parent / href
            logger.info(f"Processing content file: {content_path}")
        else:
            logger.error("No content reference found in map")
            return render_template('academic.html', error="Content not found"), 404

        # Process content
        content = dita_processor.transform(map_path)

        # Count LaTeX elements
        latex_count = len(re.findall(r'class="math-(?:tex|block)"', content))
        logger.info(f"Found {latex_count} LaTeX elements in transformed content")

        # Add debug info
        debug_info = {
            'topic_id': topic_id,
            'map_path': str(map_path),
            'content_path': str(content_path),
            'latex_count': latex_count,
            'has_latex_elements': latex_count > 0
        }

        return render_template(
            'academic.html',
            content=content,
            debug_info=debug_info
        )

    except Exception as e:
        logger.error(f"Error viewing entry: {str(e)}", exc_info=True)
        return render_template('academic.html', error=str(e)), 500

@bp.route('/api/debug/heading-state/<topic_id>')
def debug_heading_state(topic_id: str) -> ResponseReturnValue:
    """Debug endpoint to check heading counter state"""
    try:
        if not topic_id.endswith(('.md', '.dita', '.ditamap')):
            topic_id = f"{topic_id}.ditamap"

        topic_path = dita_processor.get_topic(topic_id)
        if not topic_path:
            return jsonify({
                'error': 'Topic not found',
                'topic_id': topic_id
            }), 404

        # Create new processor to test clean state
        test_processor = DITAProcessor()

        # Capture state at different points
        states = {
            'initial': dict(test_processor.transformer.heading_handler.counters),
        }

        # Process content
        content = test_processor.transform(topic_path)

        # Capture final state
        states['final'] = dict(test_processor.transformer.heading_handler.counters)

        # Get all headings from processed content
        soup = BeautifulSoup(content, 'html.parser')
        headings = []
        for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            if isinstance(h, Tag):
                headings.append({
                    'level': h.name,
                    'text': h.get_text(strip=True),
                    'id': h.get('id', '')
                })

        return jsonify({
            'topic_id': topic_id,
            'heading_states': states,
            'headings_found': headings,
            'path': str(topic_path)
        })

    except Exception as e:
        logger.error(f"Debug error: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@bp.route('/static/<path:filename>')
def serve_static(filename: str) -> ResponseReturnValue:
    """Serve static files"""
    try:
        return send_from_directory('static', filename)
    except Exception as e:
        logger.error(f"Error serving static file {filename}: {str(e)}")
        return jsonify({'error': 'File not found'}), 404

@bp.route('/api/ditamaps')
def get_ditamaps():
    """Get list of available DITA maps"""
    try:
        maps_dir = Path(current_app.root_path) / 'dita' / 'maps'
        maps = []

        # Initialize parser
        parser = etree.XMLParser(
            recover=True,
            remove_blank_text=True,
            resolve_entities=False,
            dtd_validation=False,
            load_dtd=False,
            no_network=True
        )

        for map_file in maps_dir.glob('*.ditamap'):
            try:
                # Only process non-empty files
                if map_file.stat().st_size > 0:
                    logger.info(f"Processing map file: {map_file}")
                    with open(map_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            tree = etree.fromstring(content.encode('utf-8'), parser)
                            title_elem = tree.find(".//title")
                            title = title_elem.text if title_elem is not None else map_file.stem

                            map_data = {
                                'id': map_file.stem,
                                'title': title,
                                'path': str(map_file.relative_to(maps_dir))
                            }
                            maps.append(map_data)
                            logger.info(f"Found DITA map: {map_data}")
            except Exception as e:
                logger.warning(f"Skipping invalid map file {map_file}: {str(e)}")
                continue

        return jsonify({
            'success': True,
            'maps': maps
        })
    except Exception as e:
        logger.error(f"Error listing DITA maps: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# API routes
@bp.route('/api/view/<topic_id>', methods=['GET'])
def view_topic(topic_id: str) -> ResponseReturnValue:
    """View a topic as HTML"""
    try:
        logger.info(f"Attempting to view topic: {topic_id}")
        topic_path = dita_processor.get_topic(topic_id)  # Updated method name
        if not topic_path:
            logger.error(f"Topic not found: {topic_id}")
            return Response("""
                <div class="error-container p-4 bg-red-50 border-l-4 border-red-500 text-red-700">
                    <h3 class="font-bold">Topic Not Found</h3>
                    <p>The requested topic could not be found (.dita or .md)</p>
                </div>
            """, mimetype='text/html')

        html_content = dita_processor.transform(topic_path)  # Updated to use transform
        if html_content:
            return Response(html_content, mimetype='text/html')
        else:
            return Response("""
                <div class="error-container p-4 bg-yellow-50 border-l-4 border-yellow-500 text-yellow-700">
                    <h3 class="font-bold">Empty Content</h3>
                    <p>The topic exists but contains no content.</p>
                </div>
            """, mimetype='text/html')

    except Exception as e:
        logger.error(f"Unexpected error viewing topic: {str(e)}")
        return Response(f"""
            <div class="error-container p-4 bg-red-50 border-l-4 border-red-500 text-red-700">
                <h3 class="font-bold">Error</h3>
                <p>{str(e)}</p>
            </div>
        """, mimetype='text/html')

@bp.route('/api/search', methods=['GET'])
def search() -> ResponseReturnValue:
    """Search through content"""
    try:
        query = request.args.get('q', '')
        if not query:
            return jsonify([])

        # Get all topics
        topics = dita_processor.list_topics()
        results = []

        for topic in topics:
            topic_path = dita_processor.get_topic(topic['id'])
            if not topic_path:
                continue

            try:
                # Read content
                with open(topic_path, 'r', encoding='utf-8') as f:
                    content = f.read().lower()

                # Simple text search
                if query.lower() in content:
                    # Get content preview
                    index = content.lower().find(query.lower())
                    start = max(0, index - 100)
                    end = min(len(content), index + 100)
                    preview = content[start:end]

                    results.append({
                        'id': topic['id'],
                        'title': topic.get('title', topic['id']),
                        'preview': f"...{preview}..." if start > 0 else preview,
                        'type': topic.get('type', 'topic')
                    })

            except Exception as e:
                logger.error(f"Error searching topic {topic['id']}: {str(e)}")
                continue

        return jsonify(results)

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/api/debug/topics')
def debug_topics() -> ResponseReturnValue:
    """Debug endpoint for topic listing and processing"""
    try:
        topics = dita_processor.list_topics()
        debug_info = {
            'topics_found': len(topics),
            'topics': [],
            'app_root': str(current_app.root_path),
            'dita_root': str(dita_processor.dita_root)
        }

        # Get detailed information for each topic
        for topic in topics:
            topic_path = dita_processor.get_topic(topic['id'])
            if not topic_path:
                continue

            try:
                with open(topic_path, 'r', encoding='utf-8') as f:
                    raw_content = f.read()

                # Use transformer instead of process_content
                processed_content = dita_processor.transform(topic_path)

                # Parse processed content for structure info
                soup = BeautifulSoup(processed_content, 'html.parser')
                headings = []
                for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    if isinstance(h, Tag):
                        headings.append({
                            'id': h.get('id', ''),
                            'text': h.get_text(strip=True),
                            'level': int(h.name[1])
                        })

                topic_info = {
                    **topic,
                    'exists': True,
                    'size': len(raw_content),
                    'headings': headings,
                    'has_content': bool(processed_content.strip())
                }
                debug_info['topics'].append(topic_info)

            except Exception as e:
                logger.error(f"Error processing topic {topic['id']}: {str(e)}")
                topic_info = {
                    **topic,
                    'exists': False,
                    'error': str(e)
                }
                debug_info['topics'].append(topic_info)

        return jsonify(debug_info)

    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


## Debug routes

## Topic content debug route

@bp.route('/api/debug/content/<topic_id>')
def debug_content(topic_id: str) -> ResponseReturnValue:
    """Debug endpoint to examine processed content and transformations"""
    try:
        logger.info(f"Debug request for topic: {topic_id}")

        # First get the map file
        if not topic_id.endswith(('.md', '.dita', '.ditamap')):
            topic_id = f"{topic_id}.ditamap"

        map_path = dita_processor.get_topic(topic_id)
        if not map_path:
            return jsonify({'error': 'Map not found'}), 404

        # Parse the DITA map to get the actual content file
        try:
            tree = etree.parse(str(map_path))
            # Find the first topicref's href
            topicref = tree.find(".//topicref")
            if topicref is None:
                return jsonify({'error': 'No topicref found in map'}), 404

            content_href = topicref.get('href')
            if not content_href:
                return jsonify({'error': 'No href attribute in topicref'}), 404

            logger.info(f"Found content reference: {content_href}")

            # Resolve the content path relative to the map using Path
            content_path = Path(map_path).parent.joinpath(content_href)
            content_path = content_path.resolve()  # Resolve any relative path components
            logger.info(f"Resolved content path: {content_path}")

        except Exception as e:
            logger.error(f"Error parsing map: {e}")
            return jsonify({'error': f'Map parsing error: {str(e)}'}), 500

        # Read the actual content file
        try:
            with open(content_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
        except Exception as e:
            logger.error(f"Error reading content file: {e}")
            return jsonify({'error': f'Content file error: {str(e)}'}), 500

        # Extract LaTeX equations before transformation
        latex_processor = DitaLaTeXProcessor()
        equations = latex_processor.extract_equations(raw_content)

        # Process content
        processed_content = dita_processor.transform(content_path)

        # Parse processed content
        soup = BeautifulSoup(processed_content, 'html.parser')

        # Get headings
        headings = []
        for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            if isinstance(h, Tag):
                headings.append({
                    'id': h.get('id', ''),
                    'text': h.get_text(strip=True),
                    'level': int(h.name[1])
                })

        # Find LaTeX elements
        math_elements = []
        for elem in soup.find_all(['span', 'div'], class_=['math-tex', 'math-block']):
            math_elements.append({
                'type': 'block' if 'math-block' in elem.get('class', []) else 'inline',
                'content': elem.get_text(strip=True),
                'html': str(elem),
                'attributes': {
                    attr: elem.get(attr) for attr in elem.attrs
                }
            })

        # Look for specific LaTeX patterns in raw content
        latex_patterns = {
            'block_dollars': len(re.findall(r'\$\$(.*?)\$\$', raw_content, re.DOTALL)),
            'inline_dollars': len(re.findall(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', raw_content)),
            'display_math': len(re.findall(r'\\\[(.*?)\\\]', raw_content, re.DOTALL)),
            'inline_math': len(re.findall(r'\\\((.*?)\\\)', raw_content))
        }

        debug_info = {
            'map_file': str(map_path),
            'content_file': str(content_path),
            'raw_content_length': len(raw_content),
            'processed_content_length': len(processed_content),
            'heading_count': len(headings),
            'latex_patterns_found': latex_patterns,
            'equations_found': {
                'raw': [
                    {
                        'type': eq['type'],
                        'latex': eq['latex'][:100],
                        'full': eq['full'][:100]
                    }
                    for eq in equations
                ],
                'processed': math_elements
            },
            'equation_counts': {
                'raw': len(equations),
                'processed': len(math_elements)
            }
        }

        return jsonify({
            'success': True,
            'topic_id': topic_id,
            'raw_content': raw_content,
            'processed_content': processed_content,
            'headings': headings,
            'debug_info': debug_info,
            'equations': {
                'raw': equations,
                'processed': math_elements
            }
        })

    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'topic_id': topic_id
        }), 500



## LaTeX debug
@bp.route('/api/debug/latex/<topic_id>')
def debug_latex(topic_id: str):
    try:
        logger.info(f"Debug request for topic: {topic_id}")

        # Add .ditamap extension if needed
        if not topic_id.endswith(('.md', '.dita', '.ditamap')):
            topic_id = f"{topic_id}.ditamap"

        topic_path = dita_processor.get_topic(topic_id)
        if not topic_path:
            logger.error(f"Topic not found: {topic_id}")
            return jsonify({
                'error': 'Topic not found',
                'topic_id': topic_id
            }), 404

        # Get raw content
        with open(topic_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()

        # Process content
        processor = DitaLaTeXProcessor()
        equations = processor.extract_equations(raw_content)
        processed = processor.process_content(raw_content)

        return jsonify({
            'success': True,
            'topic_id': topic_id,
            'raw_content': raw_content[:500],
            'equations_found': equations,
            'processed_content': processed[:500],
            'equation_count': len(equations)
        })

    except Exception as e:
        logger.error(f"Debug endpoint error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


# Latex transformation debug

@bp.route('/api/debug/latex-transform/<topic_id>')
def debug_latex_transform(topic_id: str):
    try:
        if not topic_id.endswith(('.md', '.dita', '.ditamap')):
            topic_id = f"{topic_id}.ditamap"

        topic_path = dita_processor.get_topic(topic_id)
        if not topic_path:
            return jsonify({'error': 'Topic not found'}), 404

        # Get content file path from map
        tree = etree.parse(str(topic_path))
        topicref = tree.find(".//topicref")
        if topicref is None or not (href := topicref.get('href')):
            return jsonify({'error': 'No content reference found'}), 404

        content_path = Path(topic_path).parent.joinpath(href).resolve()

        # Read content
        with open(content_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()

        # Process with LaTeX processor
        latex_processor = DitaLaTeXProcessor()
        processed = latex_processor.process_content(raw_content)

        # Find equations in processed content
        soup = BeautifulSoup(processed, 'html.parser')
        math_elements = soup.find_all(['span', 'div'], class_=['math-tex', 'math-block'])

        return jsonify({
            'success': True,
            'raw_content_sample': raw_content[:500],
            'processed_content_sample': processed[:500],
            'equations_found': len(math_elements),
            'equation_samples': [
                {
                    'html': str(elem),
                    'text': elem.get_text(strip=True),
                    'attributes': {k: v for k, v in elem.attrs.items()}
                }
                for elem in math_elements[:5]  # First 5 equations
            ]
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

## Component registration debug
@bp.route('/api/debug/components')
def debug_components():
    """Debug endpoint to check component registration"""
    try:
        # List all .jsx files in artifacts directory
        artifacts_dir = Path(current_app.root_path) / 'dita' / 'artifacts' / 'components'
        components = []

        for jsx_file in artifacts_dir.glob('*.jsx'):
            components.append({
                'name': jsx_file.stem,
                'path': str(jsx_file.relative_to(current_app.root_path)),
                'exists': jsx_file.exists()
            })

        return jsonify({
            'components': components,
            'artifacts_dir': str(artifacts_dir)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/debug/files')
def debug_files():
    """Debug endpoint to list all available files"""
    try:
        files = {
            'topics': [],
            'maps': []
        }

        # List topics
        for subdir in ['acoustics', 'articles', 'audio', 'abstracts', 'journals', 'reference']:
            topic_dir = Path(current_app.root_path) / 'dita' / 'topics' / subdir
            if topic_dir.exists():
                for file in topic_dir.glob('*.*'):
                    files['topics'].append({
                        'path': str(file.relative_to(current_app.root_path)),
                        'name': file.name,
                        'type': file.suffix
                    })

        # List maps
        maps_dir = Path(current_app.root_path) / 'dita' / 'maps'
        if maps_dir.exists():
            for file in maps_dir.glob('*.ditamap'):
                files['maps'].append({
                    'path': str(file.relative_to(current_app.root_path)),
                    'name': file.name
                })

        return jsonify(files)
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@bp.route('/api/map/<map_id>')
def get_map_content(map_id: str):
    """Get content of a specific DITA map"""
    try:
        map_path = Path(current_app.root_path) / 'dita' / 'maps' / f"{map_id}.ditamap"
        logger.info(f"Loading map from: {map_path}")

        if not map_path.exists():
            logger.error(f"Map not found: {map_id}")
            return jsonify({
                'success': False,
                'error': 'Map not found'
            }), 404

        # Initialize parser
        parser = etree.XMLParser(
            recover=True,
            remove_blank_text=True,
            resolve_entities=False,
            dtd_validation=False,
            load_dtd=False,
            no_network=True
        )

        with open(map_path, 'r', encoding='utf-8') as f:
            content = f.read()
            logger.info(f"Processing map content...")

        tree = etree.fromstring(content.encode('utf-8'), parser)

        # Extract map structure
        map_data = {
            'id': map_id,
            'title': '',
            'groups': []
        }

        # Get title
        title_elem = tree.find(".//title")
        if title_elem is not None:
            map_data['title'] = title_elem.text
            logger.info(f"Found map title: {map_data['title']}")

        # Get groups and topics
        for topicgroup in tree.findall(".//topicgroup"):
            group = {
                'navtitle': '',
                'topics': []
            }

            # Get group title from topicmeta/navtitle
            navtitle = topicgroup.find(".//navtitle")
            if navtitle is not None:
                group['navtitle'] = navtitle.text
                logger.info(f"Found group: {group['navtitle']}")

            # Get topics in group
            for topicref in topicgroup.findall(".//topicref"):
                href = topicref.get('href')
                if href:
                    topic_id = Path(href).stem
                    topic_title = topicref.get('navtitle', topic_id)
                    topic_data = {
                        'id': topic_id,
                        'title': topic_title,
                        'href': href
                    }
                    group['topics'].append(topic_data)
                    logger.info(f"Added topic: {topic_data}")

            map_data['groups'].append(group)

        logger.info(f"Processed map with {len(map_data['groups'])} groups")
        return jsonify({
            'success': True,
            'map': map_data
        })

    except Exception as e:
        logger.error(f"Error getting map content: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bp.route('/api/topics')
def get_topics():
    """Get all available topics"""
    try:
        logger.info("Fetching topics")
        topics = dita_processor.list_topics()
        logger.info(f"Found {len(topics)} topics")
        return jsonify({
            'success': True,
            'topics': topics
        })
    except Exception as e:
        logger.error(f"Error getting topics: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bp.route('/api/maps')
def get_maps() -> ResponseReturnValue:
    """Get all available DITA maps"""
    try:
        logger.info("Fetching maps")
        maps = dita_processor.list_maps()
        logger.info(f"Found {len(maps)} maps")
        return jsonify({
            'success': True,
            'maps': maps
        })
    except Exception as e:
        logger.error(f"Error getting maps: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Error handling
@bp.errorhandler(404)
def not_found_error(error) -> Tuple[Response, int]:
    """Handle 404 errors"""
    logger.warning(f"404 error: {error}")
    return jsonify({'error': 'Resource not found'}), 404

@bp.errorhandler(500)
def internal_error(error) -> Tuple[Response, int]:
    """Handle 500 errors"""
    logger.error(f"500 error: {error}")
    return jsonify({'error': 'Internal server error'}), 500
