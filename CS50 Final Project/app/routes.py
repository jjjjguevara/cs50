from flask import (
    Blueprint,
    jsonify,
    request,
    Response,
    current_app,
    send_from_directory,
    render_template,
    current_app,
    url_for,
    redirect,
)
from pathlib import Path
from lxml import etree
from typing import Union, Tuple, Any
import logging
from .dita.processor import DITAProcessor
import traceback
import os

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize blueprint and processor
bp = Blueprint(
    'main',
    __name__,
    url_prefix='/',
    static_folder='static',
    template_folder='templates'
)
dita_processor = DITAProcessor()

# Type alias for Flask responses
FlaskResponse = Union[Response, Tuple[Response, int], Tuple[str, int], Any]

# Main routes
# Main routes (modify the current index route and add new test route)
@bp.route('/static/dist/<path:filename>')
def serve_dist(filename: str) -> FlaskResponse:
    """Serve built Vite assets"""
    try:
        dist_dir = os.path.join(current_app.root_path, 'static', 'dist')
        if not os.path.exists(os.path.join(dist_dir, filename)):
            logger.error(f"Dist file not found: {filename}")
            return jsonify({'error': 'File not found'}), 404
        return send_from_directory(dist_dir, filename)
    except Exception as e:
        logger.error(f"Error serving dist file {filename}: {str(e)}")
        return jsonify({'error': 'File not found'}), 404


@bp.route('/')
def index() -> FlaskResponse:
    """Home route redirects to roadmap"""
    try:
        logger.info("Redirecting home to roadmap")
        return redirect(url_for('main.view_entry', topic_id='roadmap'))
    except Exception as e:
        logger.error(f"Error in home redirect: {str(e)}")
        return jsonify({'error': 'Failed to load roadmap'}), 500


@bp.route('/static/topics/<path:filename>')
def serve_topic_files(filename: str) -> FlaskResponse:
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
def serve_map_media(filename: str) -> FlaskResponse:
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
def articles() -> FlaskResponse:
    """Articles route that redirects to first ditamap"""
    try:
        logger.info("Processing articles redirect")
        maps_dir = Path(current_app.root_path) / 'dita' / 'maps'

        # Get first .ditamap file
        ditamaps = list(maps_dir.glob('*.ditamap'))
        if ditamaps:
            first_map = ditamaps[0]
            map_id = first_map.stem
            logger.info(f"Redirecting to first ditamap: {map_id}")
            return redirect(url_for('main.view_entry', topic_id=f"{map_id}.ditamap"))
        else:
            logger.error("No ditamaps found")
            return render_template('academic.html', error="No articles found"), 404

    except Exception as e:
        logger.error(f"Error in articles redirect: {str(e)}")
        return jsonify({'error': 'Failed to load articles'}), 500

@bp.route('/test')
def test_interface() -> FlaskResponse:
    """Test interface"""
    try:
        logger.info("Serving development index.html")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error serving index page: {str(e)}")
        return jsonify({'error': 'Failed to load application'}), 500


@bp.route('/entry/<topic_id>')
def view_entry(topic_id: str) -> FlaskResponse:
    """Academic article view"""
    try:
        # Add .ditamap extension if it's not a full path
        if not topic_id.endswith(('.md', '.dita', '.ditamap')):
            topic_id = f"{topic_id}.ditamap"

        logger.info(f"Attempting to view topic: {topic_id}")

        topic_path = dita_processor.get_topic_path(topic_id)
        if not topic_path:
            logger.error(f"Topic not found: {topic_id}")
            return render_template('academic.html',
                                error="Entry not found",
                                title="Not Found"), 404

        try:
            # Process the topic content and metadata
            content = dita_processor.transform_to_html(topic_path)
            toc = dita_processor.generate_toc(topic_path)
            metadata = dita_processor.get_topic_metadata(topic_path)

            # Get clean topic ID (without extension) for links
            clean_topic_id = Path(topic_id).stem

            return render_template('academic.html',
                               content=content,
                               toc=toc,
                               metadata=metadata,
                               topic_id=clean_topic_id,  # Use clean ID
                               title=metadata.get('title', 'Academic View'))
        except Exception as e:
            logger.error(f"Error processing topic {topic_id}: {str(e)}")
            return render_template('academic.html',
                                error=f"Error processing topic: {str(e)}",
                                title="Error"), 500

    except Exception as e:
        logger.error(f"Error viewing entry: {str(e)}")
        return render_template('academic.html',
                            error=str(e),
                            title="Error"), 500

@bp.route('/static/<path:filename>')
def serve_static(filename: str) -> FlaskResponse:
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
def view_topic(topic_id: str) -> FlaskResponse:
    """View a topic as HTML"""
    try:
        logger.info(f"Attempting to view topic: {topic_id}")
        topic_path = dita_processor.get_topic_path(topic_id)
        if not topic_path:
            logger.error(f"Topic not found: {topic_id}")
            return Response("""
                <div class="error-container p-4 bg-red-50 border-l-4 border-red-500 text-red-700">
                    <h3 class="font-bold">Topic Not Found</h3>
                    <p>The requested topic could not be found (.dita or .md)</p>
                </div>
            """, mimetype='text/html')

        html_content = dita_processor.transform_to_html(topic_path)
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
def search() -> FlaskResponse:
    """Search functionality"""
    query = request.args.get('q', '')
    try:
        results = dita_processor.search_topics(query)
        return jsonify(results)
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/api/debug/topics')
def debug_topics() -> FlaskResponse:
    """Debug endpoint for topic listing"""
    try:
        topics = dita_processor.list_topics()
        debug_info = {
            'topics_found': len(topics),
            'topics': topics,
            'app_root': str(current_app.root_path),
            'dita_root': str(dita_processor.dita_root)
        }
        return jsonify(debug_info)
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


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
def get_maps():
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

# Error handlers
@bp.errorhandler(404)
def not_found_error(error) -> Tuple[Response, int]:
    logger.warning(f"404 error: {error}")
    return jsonify({'error': 'Resource not found'}), 404

@bp.errorhandler(500)
def internal_error(error) -> Tuple[Response, int]:
    logger.error(f"500 error: {error}")
    return jsonify({'error': 'Internal server error'}), 500
