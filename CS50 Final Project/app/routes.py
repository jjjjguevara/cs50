"""Routes module for DITA processing application."""
from flask import (
    Blueprint, render_template, abort, send_file, current_app, request, jsonify, g
)
from pathlib import Path
from typing import Optional
import logging

# Core factory and managers
from .dita.content_factory import ContentFactory, AssemblyOptions
from .dita.processors.dita_parser import DITAParser
from .dita.event_manager import EventManager
from .dita.context_manager import ContextManager
from .dita.config.config_manager import ConfigManager
from .dita.key_manager import KeyManager
from .dita.metadata.metadata_manager import MetadataManager

# Types
from .dita.models.types import ProcessingPhase, IDType

# Utils
from .dita.utils.cache import ContentCache
from .dita.utils.html_helpers import HTMLHelper
from .dita.utils.heading import HeadingHandler
from .dita.utils.id_handler import DITAIDHandler
from .dita.utils.logger import DITALogger

# Create blueprints
main_bp = Blueprint('main', __name__)
dita_bp = Blueprint('dita', __name__)

def init_managers(app):
    """Initialize all managers within application context."""
    # Initialize utilities
    content_cache = ContentCache()
    logger = DITALogger(name="routes")
    id_handler = DITAIDHandler()

    # Initialize event manager
    event_manager = EventManager(cache=content_cache)

    # Initialize core managers
    config_manager = ConfigManager(
        metadata_manager=None,
        context_manager=None,
        content_cache=content_cache,
        event_manager=event_manager,
        id_handler=id_handler
    )

    metadata_manager = MetadataManager(
        db_path=app.config.get('METADATA_DB_PATH', Path(app.instance_path) / 'metadata.db'),
        cache=content_cache,
        event_manager=event_manager,
        context_manager=None,
        config_manager=config_manager,
        logger=logger
    )

    context_manager = ContextManager(
        event_manager=event_manager,
        content_cache=content_cache,
        metadata_manager=metadata_manager,
        config_manager=config_manager,
        logger=logger
    )

    # Resolve circular dependencies
    metadata_manager.context_manager = context_manager
    config_manager.metadata_manager = metadata_manager
    config_manager.context_manager = context_manager

    # Initialize additional utilities
    html_helper = HTMLHelper(dita_root=app.config.get('DITA_ROOT'))
    heading_handler = HeadingHandler(event_manager=event_manager)

    # Store initialized managers in app config
    app.config.update(
        CONTENT_CACHE=content_cache,
        LOGGER=logger,
        ID_HANDLER=id_handler,
        EVENT_MANAGER=event_manager,
        CONFIG_MANAGER=config_manager,
        METADATA_MANAGER=metadata_manager,
        CONTEXT_MANAGER=context_manager,
        HTML_HELPER=html_helper,
        HEADING_HANDLER=heading_handler
    )

    return app

def get_content_factory():
    """Get or create ContentFactory instance."""
    if 'content_factory' not in g:
        metadata_storage = current_app.config['METADATA_MANAGER'].storage
        g.content_factory = ContentFactory(
            event_manager=current_app.config['EVENT_MANAGER'],
            context_manager=current_app.config['CONTEXT_MANAGER'],
            config_manager=current_app.config['CONFIG_MANAGER'],
            key_manager=KeyManager(
                event_manager=current_app.config['EVENT_MANAGER'],
                cache=current_app.config['CONTENT_CACHE'],
                config_manager=current_app.config['CONFIG_MANAGER'],
                context_manager=current_app.config['CONTEXT_MANAGER'],
                metadata_storage=metadata_storage,  # Pass metadata_storage
                logger=current_app.config['LOGGER']
            ),
            metadata_manager=current_app.config['METADATA_MANAGER'],
            content_cache=current_app.config['CONTENT_CACHE'],
            html_helper=current_app.config['HTML_HELPER'],
            heading_handler=current_app.config['HEADING_HANDLER'],
            id_handler=current_app.config['ID_HANDLER'],
            logger=current_app.config['LOGGER']
        )
    return g.content_factory

def get_content_path(entry_name: str) -> Optional[Path]:
    """Get content file path from entry name."""
    try:
        content_root = current_app.config['CONTENT_ROOT']
        for folder in ['maps', 'topics']:
            for ext in ['.ditamap', '.dita', '.md']:
                path = content_root / folder / f"{entry_name}{ext}"
                current_app.logger.debug(f"Checking path: {path}")
                if path.exists():
                    current_app.logger.debug(f"Found file at: {path}")
                    return path

        current_app.logger.warning(f"No content found for {entry_name}")
        return None

    except Exception as e:
        current_app.logger.error(f"Error getting content path: {str(e)}")
        return None

def init_app(app):
    """Initialize routes and components."""
    app = init_managers(app)
    app.register_blueprint(main_bp)
    app.register_blueprint(dita_bp)

    @app.before_request
    def initialize_components():
        """Initialize components if not already initialized."""
        if not getattr(app, '_got_first_request', False):
            try:
                with app.app_context():
                    # Initialize config manager
                    current_app.config['CONFIG_MANAGER'].initialize()

                    # Initialize metadata storage
                    current_app.config['METADATA_MANAGER'].storage._init_db()

                    current_app.logger.info("Components initialized successfully")
                    app._got_first_request = True
            except Exception as e:
                current_app.logger.error(f"Error initializing components: {str(e)}")
                raise

    @app.teardown_appcontext
    def cleanup(exc):
        """Clean up resources safely."""
        try:
            # Get managers from config
            metadata_manager = current_app.config.get('METADATA_MANAGER')
            content_cache = current_app.config.get('CONTENT_CACHE')
            context_manager = current_app.config.get('CONTEXT_MANAGER')

            # Clean up in order
            if content_cache:
                content_cache.clear()

            if metadata_manager:
                try:
                    metadata_manager.cleanup()
                except Exception as e:
                    current_app.logger.error(f"Error cleaning metadata manager: {str(e)}")

            if context_manager:
                try:
                    context_manager.cleanup()
                except Exception as e:
                    current_app.logger.error(f"Error cleaning context manager: {str(e)}")

            current_app.logger.info("Cleanup completed successfully")

        except Exception as e:
            current_app.logger.error(f"Error during cleanup: {str(e)}")


@main_bp.route('/')
def index():
    """Render index page with list of available DITA maps."""
    try:
        # Get content root path from config
        content_root = current_app.config['CONTENT_ROOT']
        maps_dir = content_root / 'maps'

        # Get all .ditamap files
        available_maps = []
        if maps_dir.exists():
            available_maps = [
                {
                    'name': path.stem,
                    'path': path.name
                }
                for path in maps_dir.glob('*.ditamap')
            ]

        # Sort maps by name
        available_maps.sort(key=lambda x: x['name'])

        return render_template('index.html', maps=available_maps)
    except Exception as e:
        current_app.config['LOGGER'].error(f"Error listing maps: {str(e)}")
        abort(500)

@main_bp.route('/entry/<entry_name>')
def entry(entry_name: str):
    """Process and render content entry."""
    try:
        content_path = get_content_path(entry_name)
        if not content_path:
            abort(404)

        factory = get_content_factory()
        entry_id = current_app.config['ID_HANDLER'].generate_id(entry_name, IDType.TOPIC)

        with current_app.config['METADATA_MANAGER'].metadata_transaction(entry_id) as txn:
            processed_content = factory.process_entry(
                content_path,
                AssemblyOptions(
                    add_toc=True,
                    add_navigation=True,
                    add_metadata=True,
                    validate_output=True,
                    minify_html=False
                )
            )
            metadata = txn.updates
            current_app.config['METADATA_MANAGER'].store_metadata(entry_id, metadata)

        return render_template('entry.html', content=processed_content, title=entry_name, metadata=metadata)
    except Exception as e:
        current_app.config['LOGGER'].error(f"Error processing entry {entry_name}: {str(e)}")
        abort(500)

# Routes for dita_bp
@dita_bp.route('/api/process', methods=['POST'])
def process_content():
    """Process DITA content via API."""
    try:
        content = request.get_json()
        if not content or 'path' not in content:
            return jsonify({'error': 'Missing required fields'}), 400

        factory = get_content_factory()
        content_id = current_app.config['ID_HANDLER'].generate_id(Path(content['path']).stem, IDType.TOPIC)

        with current_app.config['METADATA_MANAGER'].metadata_transaction(content_id) as txn:
            result = factory.process_entry(
                entry_path=content['path'],
                options=AssemblyOptions(**content.get('options', {}))
            )
            metadata = txn.updates
            current_app.config['METADATA_MANAGER'].store_metadata(content_id, metadata)

        return jsonify({'html': result, 'metadata': metadata, 'content_id': content_id})
    except Exception as e:
        current_app.config['LOGGER'].error(f"Error processing content: {str(e)}")
        return jsonify({'error': str(e)}), 500

@dita_bp.route('/api/metadata/<content_id>', methods=['GET', 'POST'])
def metadata_endpoint(content_id: str):
    """Handle metadata retrieval and updates."""
    try:
        if request.method == 'GET':
            metadata = current_app.config['METADATA_MANAGER'].get_metadata(content_id)
            if not metadata:
                return jsonify({'error': 'Metadata not found'}), 404
            return jsonify({'metadata': metadata})

        updates = request.get_json()
        if not updates:
            return jsonify({'error': 'No updates provided'}), 400

        with current_app.config['METADATA_MANAGER'].metadata_transaction(content_id) as txn:
            existing = txn.updates
            existing.update(updates)
            current_app.config['METADATA_MANAGER'].store_metadata(content_id, existing)
        return jsonify({'status': 'success', 'metadata': existing})
    except Exception as e:
        current_app.config['LOGGER'].error(f"Error handling metadata for {content_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500
