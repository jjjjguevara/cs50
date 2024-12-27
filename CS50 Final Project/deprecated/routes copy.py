# app/routes.py
from datetime import datetime
from pathlib import Path
import logging
import html
from typing import Union, Optional, List
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
from app_config import DITAConfig
from .dita.config_manager import ConfigManager, config_manager
from .models import Topic, Map, ProcessedContent, ContentType
from .dita.models.types import ProcessedContent, ParsedElement, ElementType, ParsedMap, ElementType
from .dita.processors.dita_parser import DITAParser
from app.dita.utils.html_helpers import consolidate_transformed_html


bp = Blueprint('debug', __name__, url_prefix='/debug')

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

def clean_topic_path(topic_path: str, base_dir: Path) -> Optional[Path]:
    """
    Sanitize and resolve the topic path.

    Args:
        topic_path (str): Relative path to the topic file.
        base_dir (Path): Base directory for topics.

    Returns:
        Path: Resolved and sanitized path.
    """
    try:
        resolved_path = (base_dir / topic_path).resolve()
        if not str(resolved_path).startswith(str(base_dir)):
            raise ValueError("Path traversal detected")
        return resolved_path
    except Exception as e:
        logger.error(f"Invalid topic path: {topic_path}, Error: {str(e)}")
        return None



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
    """
    View a topic or map entry.

    Args:
        topic_id (str): The ID of the topic or map to render.

    Returns:
        ResponseReturnValue: Rendered HTML page or error response.
    """
    processor = None  # Ensure `processor` is always defined
    try:
        # Load configuration
        dita_config = config_manager.get_config()

        # Initialize processor
        processor = DITAProcessor(config=dita_config)

        # Retrieve the topic or map path
        topic_path = processor.get_topic(topic_id)

        # Log resolved topic path
        if topic_path:
            processor.logger.debug(f"Resolved topic path for {topic_id}: {topic_path}")
        else:
            processor.logger.error(f"Failed to resolve topic path for {topic_id}")

        if not topic_path or not isinstance(topic_path, Path):
            processor.logger.error(f"Invalid topic path for: {topic_id}")
            return render_template('academic.html', error="Entry not found"), 404

        # Log path existence check
        if not topic_path.exists():
            processor.logger.error(f"Topic file does not exist: {topic_path}")
            return render_template('academic.html', error="Entry not found"), 404

        # Process DITA map or topic
        transformed_html = ""
        metadata = {}

        if topic_path.suffix == '.ditamap':
            processor.logger.debug(f"Processing DITA map: {topic_path}")
            transformed_contents, metadata = processor.process_ditamap(topic_path)
            transformed_html = "".join(
                content.html if isinstance(content, ProcessedContent) else str(content)
                for content in transformed_contents
            )
        elif topic_path.suffix in ['.dita', '.md']:
            processor.logger.debug(f"Processing topic: {topic_path}")
            topic_content = processor.process_topic(topic_path)
            transformed_html = topic_content.html
            metadata = topic_content.metadata or {}
        else:
            processor.logger.error(f"Unsupported file format: {topic_path.suffix}")
            raise ValueError("Unsupported file format")

        # Render the entry page
        processor.logger.debug(f"Rendering content for topic {topic_id}")
        return render_template(
            'academic.html',
            content=transformed_html,
            metadata=metadata
        )

    except Exception as e:
        logger = processor.logger if processor else current_app.logger
        logger.error(f"Error rendering topic/map {topic_id}: {str(e)}", exc_info=True)
        return render_template('academic.html', error="Failed to render entry"), 500


@bp.route('/dita/topics/<path:filename>')
def serve_dita_media(filename: str) -> ResponseReturnValue:
    """Serve media files from DITA topics directory."""
    try:
        # Base directory for DITA topics
        dita_dir = Path(current_app.root_path) / 'dita' / 'topics'

        # Resolve the full file path
        file_path = (dita_dir / filename).resolve()

        # Security check for path traversal
        if not str(file_path).startswith(str(dita_dir)):
            logger.error(f"Attempted path traversal: {filename}")
            return jsonify({'error': 'Invalid path'}), 403

        # Check if file exists
        if not file_path.exists():
            logger.error(f"Media file not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404

        # Get the directory and filename for send_from_directory
        directory = str(file_path.parent)
        filename = file_path.name

        logger.debug(f"Serving media file: {filename} from {directory}")
        return send_from_directory(directory, filename)

    except Exception as e:
        logger.error(f"Error serving media file {filename}: {str(e)}")
        return jsonify({'error': str(e)}), 500


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
    processor = None  # Ensure `processor` is always defined
    try:
        # Load configuration
        dita_config = config_manager.get_config()

        # Initialize processor
        processor = DITAProcessor(config=dita_config)

        # Access maps directory
        maps_dir = processor.maps_dir
        maps = []

        for map_file in maps_dir.glob('*.ditamap'):
            try:
                # Parse DITA map
                map_data = processor.dita_parser.parse_ditamap(map_file)
                if map_data:
                    maps.append({
                        'id': map_file.stem,
                        'title': map_data.title or map_file.stem,
                        'path': str(map_file.relative_to(maps_dir)),
                        'topics': len(map_data.topics),
                    })
            except Exception as e:
                processor.logger.warning(f"Error parsing map {map_file}: {str(e)}")
                continue

        return jsonify({
            'success': True,
            'maps': maps,
        })

    except Exception as e:
        logger = processor.logger if processor else current_app.logger
        logger.error(f"Error listing maps: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
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



# DEBUG ROUTES

@bp.route('/topic/<path:topic_path>')
def debug_topic(topic_path: str):
    """
    Debug route to process and render individual topics (Markdown or DITA).

    Args:
        topic_path (str): Relative path to the topic file.

    Returns:
        Rendered HTML content for debugging.
    """
    try:
        # Get the full topic path
        full_topic_path = Path(current_app.config['TOPICS_DIR']) / topic_path

        if not full_topic_path.exists():
            return render_template('debug.html', content="File not found", metadata={}), 404

        # Load configuration
        dita_config = config_manager.get_config()

        # Initialize the processor
        processor = DITAProcessor(config=dita_config)
        parsed_element = processor.dita_parser.parse_topic(full_topic_path)

        # Process based on element type
        if parsed_element.type == ElementType.MARKDOWN:
            transformed_content = processor.md_transformer.transform_topic(parsed_element)
        elif parsed_element.type == ElementType.DITA:
            transformed_content = processor.dita_transformer.transform_topic(parsed_element)
        else:
            return render_template('debug.html', content="Unsupported file type", metadata={}), 400

        # Render the transformed content
        return render_template(
            'debug.html',
            content=transformed_content.html,
            metadata=transformed_content.metadata,
        )

    except Exception as e:
        current_app.logger.error(f"Error debugging topic {topic_path}: {str(e)}", exc_info=True)
        return render_template(
            'debug.html',
            content=f"An error occurred: {str(e)}",
            metadata={},
        ), 500
