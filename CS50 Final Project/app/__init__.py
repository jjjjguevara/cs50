# app/__init__.py
import os
from flask import Flask
from flask_cors import CORS
from app_config import config, DITAConfig
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
        app.config['CONTENT_ROOT'] = Path(app.root_path) / 'content'
        app.config['CONFIG_PATH'] = Path(app.root_path) / 'config'
        app.config['DTD_PATH'] = Path(app.root_path) / 'dita' / 'dtd'
        app.config['METADATA_DB_PATH'] = Path(app.instance_path) / 'metadata.db'

        # Load DITA configuration
        dita_config = DITAConfig.from_environment()

        # Update app config with DITA paths
        app.config.update({
            'CONTENT_ROOT': Path(app.root_path) / 'content',
            'DITA_ROOT': Path(app.root_path) / 'dita',
            'CONFIGS_PATH': Path(app.root_path) / 'dita' / 'configs',
            'METADATA_DB_PATH': instance_path / 'metadata.db',
            'DTD_PATH': dita_config.dtd_config.base_path,
            'DTD_SCHEMAS_PATH': dita_config.dtd_config.schemas_path,
            'DTD_CATALOG_PATH': dita_config.dtd_config.catalog_path
        })

        # Ensure required directories exist
        for dir_path in [
            app.config['CONTENT_ROOT'],
            app.config['CONTENT_ROOT'] / 'maps',
            app.config['CONTENT_ROOT'] / 'topics',
            app.config['DITA_ROOT'],
            app.config['CONFIGS_PATH'],
            app.config['DTD_PATH'],
            app.config['DTD_SCHEMAS_PATH']
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Create catalog.xml if it doesn't exist
        if not app.config['DTD_CATALOG_PATH'].exists():
            app.config['DTD_CATALOG_PATH'].parent.mkdir(parents=True, exist_ok=True)
            with open(app.config['DTD_CATALOG_PATH'], 'w') as f:
                f.write('''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE catalog PUBLIC "-//OASIS//DTD Entity Resolution XML Catalog V1.0//EN"
                        "http://www.oasis-open.org/committees/entity/release/1.0/catalog.dtd">
<catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
    <!-- Map DTD public identifiers to local files -->
    <public publicId="-//OASIS//DTD DITA Map//EN" uri="schemas/map.dtd"/>
    <public publicId="-//OASIS//DTD DITA Topic//EN" uri="schemas/topic.dtd"/>
    <public publicId="-//OASIS//DTD DITA Concept//EN" uri="schemas/concept.dtd"/>
    <public publicId="-//OASIS//DTD DITA Task//EN" uri="schemas/task.dtd"/>
    <public publicId="-//OASIS//DTD DITA Reference//EN" uri="schemas/reference.dtd"/>
    <public publicId="-//OASIS//DTD DITA Glossary//EN" uri="schemas/glossentry.dtd"/>
    <public publicId="-//OASIS//DTD DITA BookMap//EN" uri="schemas/bookmap.dtd"/>

    <!-- Map system identifiers to local files -->
    <system systemId="map.dtd" uri="schemas/map.dtd"/>
    <system systemId="topic.dtd" uri="schemas/topic.dtd"/>
    <system systemId="concept.dtd" uri="schemas/concept.dtd"/>
    <system systemId="task.dtd" uri="schemas/task.dtd"/>
    <system systemId="reference.dtd" uri="schemas/reference.dtd"/>
    <system systemId="glossentry.dtd" uri="schemas/glossentry.dtd"/>
    <system systemId="bookmap.dtd" uri="schemas/bookmap.dtd"/>
</catalog>''')

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

            # Initialize config manager with DTD settings
            config_manager = ConfigManager(
                config_path=app.config['CONFIGS_PATH'],
                event_manager=event_manager,
                content_cache=content_cache,
                id_handler=id_handler,
                validation_manager=validation_manager,
                schema_manager=schema_manager,
                logger=logger,
            )

            # After initialization, set DTD configuration
            config_manager.set_dtd_config({
                'base_path': str(app.config['DTD_PATH']),
                'schemas_path': str(app.config['DTD_SCHEMAS_PATH']),
                'catalog_path': str(app.config['DTD_CATALOG_PATH']),
                'validation_mode': dita_config.dtd_config.validation_mode,
                'specialization_handling': dita_config.dtd_config.specialization_handling,
                'attribute_inheritance': dita_config.dtd_config.attribute_inheritance,
                'enable_caching': dita_config.dtd_config.enable_caching
            })

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
            'CONTENT_ROOT': Path(app.root_path) / 'content',
            'DITA_ROOT': Path(app.root_path) / 'dita',
            'CONFIGS_PATH': Path(app.root_path) / 'dita' / 'configs',
            'METADATA_DB_PATH': instance_path / 'metadata.db',
            'DTD_PATH': Path(app.root_path) / 'dita' / 'dtd',  # Updated path
            'DTD_SCHEMAS_PATH': Path(app.root_path) / 'dita' / 'dtd' / 'schemas',  # Updated path
            'DTD_CATALOG_PATH': Path(app.root_path) / 'dita' / 'dtd' / 'catalog.xml'  # Updated path
        })

        app.config['DTD_CONFIG'] = dita_config.dtd_config

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
