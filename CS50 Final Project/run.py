from app import create_app
from flask import Flask
from pathlib import Path

app = create_app()

if __name__ == '__main__':
    # Development configuration
    app_root = Path(__file__).parent
    app.config.update(
        TEMPLATES_AUTO_RELOAD=True,
        STATIC_FOLDER=str(app_root / 'app' / 'static'),
        TEMPLATE_FOLDER=str(app_root / 'app' / 'templates'),
        # Add the DITA directories to the config
        DITA_ROOT=str(app_root / 'app' / 'dita'),
        DITA_TOPICS_DIR=str(app_root / 'app' / 'dita' / 'topics')
    )
    app.run(debug=True, port=5001)
