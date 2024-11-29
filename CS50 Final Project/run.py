from app import create_app
from flask import Flask

app = create_app()

if __name__ == '__main__':
    # Development configuration
    app.config.update(
        TEMPLATES_AUTO_RELOAD=True,
        STATIC_FOLDER='app/static',
        TEMPLATE_FOLDER='app/templates'
    )
    app.run(debug=True, port=5001)
