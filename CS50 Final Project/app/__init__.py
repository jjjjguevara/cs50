# app/__init__.py
import os
from flask import Flask
from flask_cors import CORS
from app_config import config
import logging
from pathlib import Path

# Import managers and dependencies
from .dita.event_manager import EventManager
from .dita.context_manager import ContextManager
from .dita.config.config_manager import ConfigManager
from .dita.key_manager import KeyManager
from .dita.metadata.metadata_manager import MetadataManager
from .dita.utils.cache import ContentCache
from .dita.utils.html_helpers import HTMLHelper
from .dita.utils.heading import HeadingHandler
from .dita.utils.id_handler import DITAIDHandler
from .dita.utils.logger import DITALogger
from .dita.validation_manager import ValidationManager
from .dita.schema_manager import SchemaManager

# Setup logging
logger = DITALogger(name=__name__)

def create_app(config_name=None):
    try:
        # Initialize app
        if config_name is None:
            config_name = os.environ.get('FLASK_ENV', 'development')

        app = Flask(__name__)
        app.config.from_object(config[config_name])

        # Set up paths with explicit instance folder
        instance_path = Path(app.instance_path)
        instance_path.mkdir(parents=True, exist_ok=True)

        # Explicitly set config paths
        app.config.update({
            'CONTENT_ROOT': Path(app.root_path) / 'content',
            'DITA_ROOT': Path(app.root_path) / 'dita',
            'CONFIGS_PATH': Path(app.root_path) / 'dita' / 'configs',
            'METADATA_DB_PATH': instance_path / 'metadata.db'
        })

        # Ensure required directories exist
        for dir_path in [
            app.config['CONTENT_ROOT'],
            app.config['CONTENT_ROOT'] / 'maps',
            app.config['CONTENT_ROOT'] / 'topics',
            app.config['DITA_ROOT'],
            app.config['CONFIGS_PATH']
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize core components with app context
        with app.app_context():
            # Initialize shared components
            content_cache = ContentCache()
            id_handler = DITAIDHandler()
            event_manager = EventManager(cache=content_cache)

            # Initialize validation and schema managers first
            validation_manager = ValidationManager(
                cache=content_cache,
                event_manager=event_manager,
                config_manager=None,  # Initially None
                logger=logger
            )

            schema_manager = SchemaManager(
                config_path=app.config['CONFIGS_PATH'],
                cache=content_cache,
                logger=logger
            )

            # Initialize config manager
            config_manager = ConfigManager(
                config_path=app.config['CONFIGS_PATH'],
                event_manager=event_manager,
                content_cache=content_cache,
                id_handler=id_handler,
                validation_manager=validation_manager,
                schema_manager=schema_manager,
                logger=logger
            )

            # Update validation manager with config manager
            validation_manager.config_manager = config_manager

            # Ensure config manager is initialized
            config_manager.initialize()

            # Initialize metadata manager
            metadata_manager = MetadataManager(
                db_path=app.config['METADATA_DB_PATH'],
                cache=content_cache,
                event_manager=event_manager,
                context_manager=None,  # Will be updated after creation
                config_manager=config_manager
            )

            # Initialize context manager
            context_manager = ContextManager(
                event_manager=event_manager,
                content_cache=content_cache,
                metadata_manager=metadata_manager,
                config_manager=config_manager,
                logger=logger
            )

            # Update circular references
            metadata_manager.context_manager = context_manager

            # Initialize HTML helper with content root
            html_helper = HTMLHelper(dita_root=app.config['CONTENT_ROOT'])

            # Store instances in app config
            app.config.update({
                'DITA_CONFIG_MANAGER': config_manager,
                'DITA_CONTEXT_MANAGER': context_manager,
                'DITA_METADATA_MANAGER': metadata_manager,
                'DITA_EVENT_MANAGER': event_manager,
                'DITA_CONTENT_CACHE': content_cache,
                'DITA_HTML_HELPER': html_helper,
                'DITA_ID_HANDLER': id_handler
            })

        # Configure CORS
        CORS(app, resources={
            r"/api/*": {
                "origins": app.config.get('CORS_ORIGINS', '*'),
                "methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"],
                "supports_credentials": True
            }
        })

        # Register blueprint
        from .routes import init_app as init_routes
        init_routes(app)
        logger.info("Successfully registered routes blueprint")

        return app

    except Exception as e:
        logger.error(f"Application initialization failed: {str(e)}")
        raise
