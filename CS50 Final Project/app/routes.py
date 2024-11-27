from flask import (
    Blueprint,
    jsonify,
    request,
    Response,
    current_app,
    send_from_directory,
    send_file,
    render_template
)
import os
from typing import Union, Tuple
from pathlib import Path
import logging
from .dita.processor import DITAProcessor
import traceback

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

bp = Blueprint('main', __name__,
               url_prefix='/',
               static_folder='static',
               template_folder='static')  # Changed template_folder to static
dita_processor = DITAProcessor()

@bp.route('/')
def index():
    if current_app.debug:
        return render_template('index.html')
    return send_from_directory('../dist', 'index.html')

@bp.route('/<path:path>')
def serve_static(path):
    if current_app.debug:
        return send_from_directory('static', path)
    return send_from_directory('../dist', path)


# API Routes
@bp.route('/api/test', methods=['GET'])
def test() -> Response:
    """Test endpoint"""
    return jsonify({'message': 'API is working!'})

@bp.route('/api/topics', methods=['GET', 'POST'])
def topics() -> Union[Response, Tuple[Response, int]]:
    """Handle topic operations"""
    if request.method == 'GET':
        topics = dita_processor.list_topics()
        return jsonify(topics)

    # POST method handling
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Empty request body'}), 400

    title = data.get('title')
    content = data.get('content')
    topic_type = data.get('type', 'concept')

    if not title or not content:
        return jsonify({'error': 'Title and content are required'}), 400

    topic_path = dita_processor.create_topic(
        title,
        content,
        topic_type
    )

    if topic_path:
        return jsonify({
            'message': 'Topic created successfully',
            'path': str(topic_path)
        })
    return jsonify({'error': 'Failed to create topic'}), 500

@bp.route('/api/maps', methods=['GET', 'POST'])
def maps() -> Union[Response, Tuple[Response, int]]:
    """Handle map operations"""
    if request.method == 'GET':
        maps = dita_processor.list_maps()
        return jsonify(maps)

    # POST method handling
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Empty request body'}), 400

    title = data.get('title')
    topics = data.get('topics')

    if not title or not topics:
        return jsonify({'error': 'Title and topics are required'}), 400

    if not isinstance(topics, list):
        return jsonify({'error': 'Topics must be a list'}), 400

    map_path = dita_processor.create_map(title, topics)

    if map_path:
        return jsonify({
            'message': 'Map created successfully',
            'path': str(map_path)
        })
    return jsonify({'error': 'Failed to create map'}), 500

@bp.route('/api/transform/<path:file_path>', methods=['POST'])
def transform_to_html(file_path: str) -> Union[Response, Tuple[Response, int]]:
    """Transform a DITA file to HTML"""
    if not file_path:
        return jsonify({'error': 'File path is required'}), 400

    input_path = Path(file_path)
    output_path = dita_processor.transform_to_html(input_path)

    if output_path:
        return jsonify({
            'message': 'Transformation successful',
            'output_path': str(output_path)
        })
    return jsonify({'error': 'Transformation failed'}), 500

@bp.route('/api/view/<topic_id>', methods=['GET'])
def view_topic(topic_id: str) -> Union[Response, Tuple[Response, int]]:
    """View a topic as HTML"""
    try:
        logger.info(f"Attempting to view topic: {topic_id}")

        # Get topic path
        topic_path = dita_processor.get_topic_path(topic_id)

        if not topic_path:
            logger.error(f"Topic not found: {topic_id}")
            return jsonify({
                'error': 'Topic not found',
                'topic_id': topic_id,
                'searched_directories': [
                    str(dita_processor.topics_dir / subdir)
                    for subdir in ['acoustics', 'articles', 'audio', 'abstracts', 'journals']
                ]
            }), 404

        logger.info(f"Found topic at: {topic_path}")

        # Transform to HTML directly
        try:
            html_content = dita_processor.transform_to_html(topic_path)
            if html_content:
                logger.info("Successfully transformed content to HTML")
                return Response(html_content, mimetype='text/html')
            else:
                logger.error("HTML transformation returned empty content")
                return jsonify({'error': 'Empty content after transformation'}), 500
        except Exception as e:
            logger.error(f"Error transforming content: {e}")
            # Return a basic HTML error message that can be displayed
            error_html = f"""
            <div class="error-container p-4 bg-red-50 border-l-4 border-red-500 text-red-700">
                <h3 class="font-bold">Error Loading Content</h3>
                <p>{str(e)}</p>
                <p class="text-sm mt-2">Topic ID: {topic_id}</p>
                <p class="text-sm">File: {topic_path}</p>
            </div>
            """
            return Response(error_html, mimetype='text/html')

    except Exception as e:
        logger.error(f"Unexpected error viewing topic: {str(e)}")
        error_html = f"""
        <div class="error-container p-4 bg-red-50 border-l-4 border-red-500 text-red-700">
            <h3 class="font-bold">Unexpected Error</h3>
            <p>{str(e)}</p>
        </div>
        """
        return Response(error_html, mimetype='text/html')

# Serve static files
@bp.route('/static/<path:filename>')
def static_files(filename: str) -> Response:
    """Serve static files"""
    return send_from_directory('static', filename)

# Catch-all route should be last
@bp.route('/<path:path>')
def catch_all(path: str) -> Tuple[Response, int]:
    """Catch-all route for undefined paths"""
    return jsonify({'error': f'Path {path} not found'}), 404

@bp.route('/api/test-topics')
def test_topics():
    """Test endpoint that returns some hardcoded topics"""
    return jsonify([
        {
            'id': 'test-topic-1',
            'title': 'Test Topic 1',
            'type': 'article'
        },
        {
            'id': 'test-topic-2',
            'title': 'Test Topic 2',
            'type': 'article'
        }
    ])

@bp.route('/api/debug/topics')
def debug_topics():
    """Debug endpoint for topic listing"""
    try:
        topics = dita_processor.list_topics()
        debug_info = {
            'topics_found': len(topics),
            'topics': topics,
            'directories_checked': [
                str(dita_processor.topics_dir / subdir)
                for subdir in ['abstracts', 'acoustics', 'articles', 'audio', 'journals']
            ],
            'app_root': str(current_app.root_path),
            'dita_root': str(dita_processor.dita_root)
        }
        return jsonify(debug_info)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500
