import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, ".env"))


@dataclass
class DITAPathConfig:
    """DITA path configuration."""
    dita_root: Path = Path(basedir) / 'app' / 'dita'
    maps_dir: Path = Path(basedir) / 'app' / 'dita' / 'maps'
    topics_dir: Path = Path(basedir) / 'app' / 'dita' / 'topics'
    output_dir: Path = Path(basedir) / 'app' / 'dita' / 'output'
    artifacts_dir: Path = Path(basedir) / 'app' / 'dita' / 'artifacts'
    media_dir: Path = Path(basedir) / 'app' / 'dita' / 'media'

@dataclass
class DITAParserConfig:
    """DITA parser configuration."""
    validate_dtd: bool = False
    resolve_entities: bool = False
    load_dtd: bool = False
    remove_blank_text: bool = True
    no_network: bool = True
    dtd_validation: bool = False

@dataclass
class DITAProcessingConfig:
    """DITA processing configuration."""
    process_latex: bool = True
    number_headings: bool = True
    enable_cross_refs: bool = True
    process_artifacts: bool = True
    show_toc: bool = True
    enable_debug: bool = False


class DITAConfig:
    """DITA-specific configuration."""
    metadata: Dict[str, Any] = field(default_factory=dict)
    def __init__(self):
        self.paths = DITAPathConfig()
        self.parser = DITAParserConfig()
        self.processing = DITAProcessingConfig()

    @classmethod
    def validate_paths(cls) -> bool:
        """Validate all required paths exist."""
        try:
            paths = DITAPathConfig()
            required_paths = [
                paths.dita_root,
                paths.maps_dir,
                paths.topics_dir,
                paths.output_dir,
                paths.artifacts_dir,
                paths.media_dir
            ]

            return all(path.exists() for path in required_paths)
        except Exception:
            return False

    @classmethod
    def create_paths(cls) -> None:
        """Create required paths if they don't exist."""
        paths = DITAPathConfig()
        for path_attr in dir(paths):
            if not path_attr.startswith('_'):
                path = getattr(paths, path_attr)
                if isinstance(path, Path):
                    path.mkdir(parents=True, exist_ok=True)


class Config:
    # Basic configuration
    SECRET_KEY = os.environ.get("SECRET_KEY") or "you-will-never-guess"
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL"
    ) or "sqlite:///" + os.path.join(basedir, "app.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # CORS settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:5000,http://127.0.0.1:5000,http://localhost:5001,http://127.0.0.1:5001').split(',')
    CORS_METHODS = ['GET', 'POST', 'OPTIONS']
    CORS_ALLOWED_HEADERS = ['Content-Type', 'Authorization']
    CORS_SUPPORTS_CREDENTIALS = True

    # Static files
    STATIC_FOLDER = 'static'
    TEMPLATES_FOLDER = 'templates'

    # Security settings
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'False').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

    DITA = DITAConfig()

    @classmethod
    def validate_dita_config(cls) -> bool:
        """Validate DITA configuration."""
        return cls.DITA.validate_paths()



class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    DEVELOPMENT = True

    # Override CORS settings for development
    CORS_ORIGINS = [
        'http://localhost:5000',
        'http://127.0.0.1:5000',
        'http://localhost:5001',
        'http://127.0.0.1:5001'
    ]

    # Development-specific settings
    TEMPLATES_AUTO_RELOAD = True
    SEND_FILE_MAX_AGE_DEFAULT = 0

    # Logging
    LOG_LEVEL = 'DEBUG'

    # Development-specific DITA settings
    DITA = DITAConfig()
    DITA.processing.enable_debug = True
    DITA.parser.remove_blank_text = True

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    DEVELOPMENT = False

    # Override CORS settings for production
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '').split(',')

    # Production-specific settings
    TEMPLATES_AUTO_RELOAD = False
    SEND_FILE_MAX_AGE_DEFAULT = 31536000  # 1 year

    # Security in production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Strict'

    # Logging
    LOG_LEVEL = 'INFO'

    # Production-specific DITA settings
    DITA = DITAConfig()
    DITA.processing.enable_debug = False
    DITA.parser.no_network = True


# Environment-specific configurations
class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False
    CORS_ORIGINS = ['http://localhost:5000']

    # Testing-specific DITA settings
    DITA = DITAConfig()
    DITA.processing.enable_debug = True

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

# Helper functions to get current config

def get_environment() -> str:
    """Get current environment."""
    return os.environ.get('FLASK_ENV', 'development')

def get_config():
    env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])

def init_dita_config():
    """Initialize DITA configuration."""
    try:
        current_config = get_config()
        if not current_config.validate_dita_config():
            current_config.DITA.create_paths()
        return True
    except Exception:
        return False

def load_dita_config() -> DITAConfig:
    """Load DITA configuration based on environment."""
    env = get_environment()
    config_class = config.get(env, config['default'])
    return config_class.DITA

def validate_dita_config(config: DITAConfig) -> bool:
    """Validate DITA configuration."""
    return config.validate_paths()

def get_dita_config() -> DITAConfig:
    """Get current DITA configuration."""
    return get_config().DITA

def update_dita_config(updates: Dict[str, Any]) -> None:
    """Update DITA configuration."""
    current_config = get_dita_config()
    for key, value in updates.items():
        if hasattr(current_config, key):
            setattr(current_config, key, value)
