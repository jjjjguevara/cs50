import os
from flask import Flask
from flask_cors import CORS
from app_config import config
import logging
from .dita.config_manager import config_manager

app = Flask(__name__, static_folder='dita')

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
        if config_name not in config:
            logger.error(f"Unknown configuration name: {config_name}")
            raise ValueError(f"Unknown configuration: {config_name}")

        app.config.from_object(config[config_name])

        # Initialize DITA configuration
        dita_config = config_manager.load_config()
        if not dita_config:
            logger.error("Failed to load DITA configuration")
            raise ValueError("DITA configuration failed")

        # Configure CORS
        CORS(app, resources={
            r"/api/*": {
                "origins": app.config.get('CORS_ORIGINS', '*'),
                "methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"],
                "supports_credentials": True
            },
            r"/entry/*": {
                "origins": app.config.get('CORS_ORIGINS', '*'),
                "methods": ["GET"],
                "supports_credentials": True
            },
            r"/static/*": {
                "origins": "*" if app.config.get('DEVELOPMENT', False) else app.config.get('CORS_ORIGINS', '*'),
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
