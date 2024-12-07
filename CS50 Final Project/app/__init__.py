# app/__init__.py
import os
from flask import Flask
from flask_cors import CORS
from config import config
import logging
from .dita.config_manager import config_manager

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

def create_app(config_name=None):
    """Create and configure Flask application."""
    try:
        # Initialize app
        if config_name is None:
            config_name = os.environ.get('FLASK_ENV', 'development')

        app = Flask(__name__)
        app.config.from_object(config[config_name])

        # Initialize DITA configuration
        dita_config = config_manager.load_config()
        if not dita_config:
            logger.error("Failed to load DITA configuration")
            raise ValueError("DITA configuration failed")

        # Configure CORS
        CORS(app, resources={
            r"/api/*": {
                "origins": app.config['CORS_ORIGINS'],
                "methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"],
                "supports_credentials": True
            },
            r"/entry/*": {
                "origins": app.config['CORS_ORIGINS'],
                "methods": ["GET"],
                "supports_credentials": True
            },
            r"/static/*": {
                "origins": "*" if app.config['DEVELOPMENT'] else app.config['CORS_ORIGINS'],
                "methods": ["GET"]
            }
        })

        # Register blueprint
        from .routes import bp
        app.register_blueprint(bp)
        logger.info("Successfully registered routes blueprint")

        return app

    except Exception as e:
        logger.error(f"Application initialization failed: {str(e)}")
        raise
