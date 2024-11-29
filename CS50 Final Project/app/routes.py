from flask import (
    Blueprint,
    jsonify,
    request,
    Response,
    current_app,
    send_from_directory,
    render_template,
)
from typing import Union, Tuple, Any
import logging
from .dita.processor import DITAProcessor
import traceback

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
    template_folder='static'
)
dita_processor = DITAProcessor()

# Type alias for Flask responses
FlaskResponse = Union[Response, Tuple[Response, int], Tuple[str, int], Any]

# Main routes
@bp.route('/')
def index() -> FlaskResponse:
    """Serve the main application page"""
    try:
        if current_app.debug:
            logger.info("Serving development index.html")
            return render_template('index.html')
        logger.info("Serving production index.html")
        return send_from_directory('../dist', 'index.html')
    except Exception as e:
        logger.error(f"Error serving index page: {str(e)}")
        return jsonify({'error': 'Failed to load application'}), 500

@bp.route('/static/<path:filename>')
def serve_static(filename: str) -> FlaskResponse:
    """Serve static files"""
    try:
        if current_app.debug:
            return send_from_directory('static', filename)
        return send_from_directory('../dist', filename)
    except Exception as e:
        logger.error(f"Error serving static file {filename}: {str(e)}")
        return jsonify({'error': 'File not found'}), 404

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

# Error handlers
@bp.errorhandler(404)
def not_found_error(error) -> Tuple[Response, int]:
    logger.warning(f"404 error: {error}")
    return jsonify({'error': 'Resource not found'}), 404

@bp.errorhandler(500)
def internal_error(error) -> Tuple[Response, int]:
    logger.error(f"500 error: {error}")
    return jsonify({'error': 'Internal server error'}), 500
