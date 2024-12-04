# app/__init__.py
from flask import Flask
from flask_cors import CORS
from config import config
import os
import logging

# Setup logging
logger = logging.getLogger(__name__)

def create_app(config_name=None):
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')

    app = Flask(__name__)

    # Load configuration
    app.config.from_object(config[config_name])

    # Configure CORS with specific settings
    if app.config['DEVELOPMENT']:
        logger.info("Configuring CORS for development environment")
        CORS(app, resources={
            r"/api/*": {
                "origins": ["http://localhost:5000", "http://127.0.0.1:5000",
                           "http://localhost:5001", "http://127.0.0.1:5001"],
                "methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"],
                "supports_credentials": True
            },
            r"/entry/*": {
                "origins": ["http://localhost:5000", "http://127.0.0.1:5000",
                           "http://localhost:5001", "http://127.0.0.1:5001"],
                "methods": ["GET"],
                "supports_credentials": True
            },
            r"/static/*": {
                "origins": "*",
                "methods": ["GET"]
            }
        })
    else:
        logger.info("Configuring CORS for production environment")
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
                "origins": app.config['CORS_ORIGINS'],
                "methods": ["GET"]
            }
        })

    # Enable CORS pre-flight responses
    def add_cors_headers(response):
        if app.config['DEVELOPMENT']:
            response.headers['Access-Control-Allow-Origin'] = '*'
        else:
            response.headers['Access-Control-Allow-Origin'] = ', '.join(app.config['CORS_ORIGINS'])
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

    app.after_request(add_cors_headers)

    # Register blueprint
    try:
        from .routes import bp
        app.register_blueprint(bp)
        logger.info("Successfully registered routes blueprint")
    except Exception as e:
        logger.error(f"Failed to register blueprint: {str(e)}")
        raise

    # Add debug route to test CORS
    @app.route('/api/test-cors')
    def test_cors():
        return {'message': 'CORS test successful'}, 200

    return app
