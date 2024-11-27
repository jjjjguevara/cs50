from flask import Flask
from flask_cors import CORS
from config import config
import os

def create_app(config_name=None):
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')

    app = Flask(__name__)

    # Load configuration
    app.config.from_object(config[config_name])

    # Configure CORS based on environment
    if app.config['DEVELOPMENT']:
        CORS(app, resources={r"/api/*": {"origins": "*"}})
    else:
        CORS(app, resources={r"/api/*": {"origins": app.config['CORS_ORIGINS']}})

    # Register blueprint
    from .routes import bp
    app.register_blueprint(bp)

    return app
