# app/routes.py
from pathlib import Path
import logging
from typing import Union, Optional
from flask import (
    Blueprint,
    current_app,
    jsonify,
    render_template,
    send_from_directory,
    redirect,
    url_for
)
from flask.typing import ResponseReturnValue

from .dita.processor import DITAProcessor
from .dita.config_manager import config_manager
from .models import Topic, Map, ProcessedContent, ContentType

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

@bp.route('/')
def index() -> ResponseReturnValue:
    """Redirect home to roadmap."""
    try:
        logger.info("Redirecting home to roadmap")
        return redirect(url_for('main.view_entry', topic_id='roadmap'))
    except Exception as e:
        logger.error(f"Error in home redirect: {str(e)}")
        return jsonify({'error': 'Failed to load roadmap'}), 500

@bp.route('/entry/<topic_id>')
def view_entry(topic_id: str) -> ResponseReturnValue:
    """View a topic or map entry."""
    try:
        # Get DITA configuration and initialize processor
        dita_config = config_manager.get_config()
        processor = DITAProcessor(dita_config)

        # Get map or topic path
        content_path = processor.get_topic(topic_id)
        if not content_path:
            logger.error(f"Content not found: {topic_id}")
            return render_template('academic.html', error="Entry not found"), 404

        # Transform content
        content = processor.transform(content_path)
        if not content:
            logger.error(f"No content generated for: {topic_id}")
            return render_template('academic.html', error="No content available"), 404

        return render_template(
            'academic.html',
            content=content,
            topic_id=topic_id
        )

    except Exception as e:
        logger.error(f"Error viewing entry: {str(e)}")
        return render_template('academic.html', error=str(e)), 500

@bp.route('/static/topics/<path:filename>')
def serve_topic_files(filename: str) -> ResponseReturnValue:
    """Serve files from topics directory."""
    try:
        topics_dir = Path(current_app.root_path) / 'dita' / 'topics'
        file_path = (topics_dir / filename).resolve()

        # Security check
        if not str(file_path).startswith(str(topics_dir)):
            logger.error(f"Attempted path traversal: {filename}")
            return jsonify({'error': 'Invalid path'}), 403

        # Handle image files without extension
        if not file_path.exists() and '.' not in filename:
            for ext in ['.svg', '.png', '.jpg', '.jpeg']:
                test_path = (topics_dir / f"{filename}{ext}").resolve()
                if test_path.exists():
                    file_path = test_path
                    break

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404

        return send_from_directory(str(file_path.parent), file_path.name)

    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/static/dist/<path:filename>')
def serve_dist(filename: str) -> ResponseReturnValue:
    """Serve built assets."""
    try:
        dist_dir = Path(current_app.root_path) / 'static' / 'dist'
        file_path = (dist_dir / filename).resolve()

        if not str(file_path).startswith(str(dist_dir)):
            logger.error(f"Attempted path traversal: {filename}")
            return jsonify({'error': 'Invalid path'}), 403

        if not file_path.exists():
            logger.error(f"Dist file not found: {filename}")
            return jsonify({'error': 'File not found'}), 404

        return send_from_directory(str(file_path.parent), file_path.name)

    except Exception as e:
        logger.error(f"Error serving dist file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/api/maps')
def get_maps() -> ResponseReturnValue:
    """Get all available DITA maps."""
    try:
        dita_config = config_manager.get_config()
        processor = DITAProcessor(dita_config)
        parser = processor.dita_parser  # Using the parser from processor

        maps_dir = Path(current_app.root_path) / 'dita' / 'maps'
        maps = []

        for map_file in maps_dir.glob('*.ditamap'):
            try:
                map_data = parser.parse_map(map_file)  # Using parser's parse_map method
                if map_data:
                    maps.append({
                        'id': map_file.stem,
                        'title': map_data.title or map_file.stem,
                        'path': str(map_file.relative_to(maps_dir)),
                        'topics': len(map_data.topics)
                    })
            except Exception as e:
                logger.warning(f"Error parsing map {map_file}: {str(e)}")
                continue

        return jsonify({
            'success': True,
            'maps': maps
        })

    except Exception as e:
        logger.error(f"Error listing maps: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Error handlers
@bp.errorhandler(404)
def not_found_error(error: Exception) -> ResponseReturnValue:
    """Handle 404 errors."""
    logger.warning(f"404 error: {error}")
    return jsonify({'error': 'Resource not found'}), 404

@bp.errorhandler(500)
def internal_error(error: Exception) -> ResponseReturnValue:
    """Handle 500 errors."""
    logger.error(f"500 error: {error}")
    return jsonify({'error': 'Internal server error'}), 500
