import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import json

class DevelopmentConfig:
    DEBUG = True
    CORS_ORIGINS = "*"

class ProductionConfig:
    DEBUG = False
    CORS_ORIGINS = "https://yourdomain.com"

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig
}

class DITAConfig:
    """
    Main configuration class for DITA processing.
    """
    def __init__(
        self,
        # Ensure topics_dir is either None or a Path object
        topics_dir: Optional[Path] = None,
        maps_dir: Optional[Path] = None,
        artifacts_dir: Optional[Path] = None,
        static_dir: Optional[Path] = None,
        metadata_db_path: Optional[Path] = None,
        number_headings: bool = False,
        enable_cross_refs: bool = True,
        show_toc: bool = True,
        process_latex: bool = False,
        latex_settings: Optional[dict] = None,
    ):


        # First, use os.getenv() to get the directory (returns a string or None)
        env_var = os.getenv("DITA_TOPICS_DIR", "app/dita/topics")

        # Convert the environment variable (or fallback) to a Path object
        if isinstance(env_var, str):
            topics_dir = Path(env_var)

        # Rest of the configuration setup
        self.topics_dir = topics_dir.resolve() if topics_dir else None
        self.maps_dir = maps_dir or self._default_dir("DITA_MAPS_DIR", "app/dita/maps")
        self.artifacts_dir = artifacts_dir or self._default_dir("DITA_ARTIFACTS_DIR", "app/dita/artifacts")
        self.static_dir = static_dir or self._default_dir("DITA_STATIC_DIR", "app/static")
        self.metadata_db_path = metadata_db_path or Path('metadata.db')
        self.number_headings = number_headings
        self.enable_cross_refs = enable_cross_refs
        self.show_toc = show_toc
        self.process_latex = process_latex


        # LaTeX configuration
        self.latex_settings = latex_settings or {
            'macros': {
                "\\N": "\\mathbb{N}",
                "\\R": "\\mathbb{R}"
            },
            'throw_on_error': False,
            'output_mode': 'html'
        }

    def _default_dir(self, env_var: str, default: str) -> Path:
        """
        Fetch the directory from the environment or use a default value.

        Args:
            env_var (str): The environment variable to check.
            default (str): Default directory to use if the environment variable is not set.

        Returns:
            Path: The resolved directory path.
        """
        dir_path = os.getenv(env_var, default)
        resolved_path = Path(dir_path).resolve()
        if not resolved_path.exists():
            raise ValueError(f"Configured directory '{env_var}' does not exist: {resolved_path}")
        return resolved_path

    def validate(self) -> None:
        """
        Validate the configuration to ensure all critical directories exist.

        Raises:
            ValueError: If required directories are invalid.
        """
        for attr in ["topics_dir", "maps_dir", "artifacts_dir", "static_dir"]:
            dir_path = getattr(self, attr, None)
            if not dir_path or not Path(dir_path).exists():
                raise ValueError(f"The specified directory for '{attr}' does not exist: {dir_path}")

    @classmethod
    def from_environment(cls) -> "DITAConfig":
        """
        Create a DITAConfig instance from environment variables.

        Returns:
            DITAConfig: Configured instance based on environment variables.
        """
        topics_dir = os.getenv("DITA_TOPICS_DIR")
        maps_dir = os.getenv("DITA_MAPS_DIR")
        artifacts_dir = os.getenv("DITA_ARTIFACTS_DIR")
        static_dir = os.getenv("DITA_STATIC_DIR")
        number_headings = os.getenv("DITA_NUMBER_HEADINGS", "False").lower() in ("true", "1", "yes")
        enable_cross_refs = os.getenv("DITA_ENABLE_CROSS_REFS", "True").lower() in ("true", "1", "yes")
        show_toc = os.getenv("DITA_SHOW_TOC", "True").lower() in ("true", "1", "yes")
        process_latex = os.getenv("DITA_PROCESS_LATEX", "False").lower() in ("true", "1", "yes")

        # Parse LaTeX settings from environment
        latex_settings = {}
        if process_latex:
            latex_settings = {
                'macros': json.loads(os.getenv("DITA_LATEX_MACROS", "{}")),
                'throw_on_error': os.getenv("DITA_LATEX_THROW_ON_ERROR", "False").lower() in ("true", "1", "yes"),
                'output_mode': os.getenv("DITA_LATEX_OUTPUT_MODE", "html")
            }

        return cls(
            topics_dir=Path(topics_dir) if topics_dir else None,
            maps_dir=Path(maps_dir) if maps_dir else None,
            artifacts_dir=Path(artifacts_dir) if artifacts_dir else None,
            static_dir=Path(static_dir) if static_dir else None,
            number_headings=number_headings,
            enable_cross_refs=enable_cross_refs,
            show_toc=show_toc,
            process_latex=process_latex,
            latex_settings=latex_settings
        )

class ConfigValidator:
    """
    Utility class to validate and debug configuration settings.
    """

    def __init__(self, config: DITAConfig):
        self.config = config

    def validate_directories(self) -> None:
        """
        Validate all directories in the configuration.

        Raises:
            ValueError: If any directory is invalid.
        """
        try:
            self.config.validate()
        except ValueError as e:
            raise ValueError(f"Configuration validation error: {str(e)}")

    def log_config(self, logger) -> None:
        """
        Log the configuration settings for debugging.

        Args:
            logger: Logger instance to output configuration.
        """
        logger.debug("DITA Configuration Settings:")
        for attr, value in vars(self.config).items():
            logger.debug(f"{attr}: {value}")


def get_environment() -> str:
    """
    Determine the current environment.

    Returns:
        str: The current environment ('development', 'production', etc.).
    """
    return os.getenv("FLASK_ENV", "development")


def load_config() -> DITAConfig:
    """
    Load and validate the configuration.

    Returns:
        DITAConfig: Validated configuration instance.
    """
    try:
        config = DITAConfig.from_environment()
        config.validate()
        return config
    except ValueError as e:
        raise ValueError(f"Failed to load DITA configuration: {str(e)}")
