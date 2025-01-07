import os
from pathlib import Path
from typing import Optional, TypedDict, Dict
from dataclasses import dataclass, field
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

@dataclass
class LaTeXConfig:
    """LaTeX processing configuration."""
    macros: Dict[str, str] = field(default_factory=lambda: {
        "\\N": "\\mathbb{N}",
        "\\R": "\\mathbb{R}"
    })
    throw_on_error: bool = False
    output_mode: str = "html"

@dataclass
class DTDConfig:
    """DTD processing configuration."""
    validation_mode: str = "strict"  # strict, lax, or none
    specialization_handling: str = "inherit"  # inherit, override, or merge
    attribute_inheritance: str = "merge"  # merge or override
    enable_caching: bool = True
    base_path: Optional[Path] = None
    schemas_path: Optional[Path] = None
    catalog_path: Optional[Path] = None

    def __post_init__(self):
        """Convert paths to Path objects and set defaults if needed."""
        # Set default paths relative to base_path if not provided
        if self.base_path is None:
            self.base_path = Path("app/dita/dtd")
        elif isinstance(self.base_path, str):
            self.base_path = Path(self.base_path)

        if self.schemas_path is None:
            self.schemas_path = self.base_path / "schemas"
        elif isinstance(self.schemas_path, str):
            self.schemas_path = Path(self.schemas_path)

        if self.catalog_path is None:
            self.catalog_path = self.base_path / "catalog.xml"
        elif isinstance(self.catalog_path, str):
            self.catalog_path = Path(self.catalog_path)

        # Create directories if they don't exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.schemas_path.mkdir(parents=True, exist_ok=True)



class DITAConfig:
    """
    Main configuration class for DITA processing.
    """
    def __init__(
        # Ensure topics_dir is either None or a Path object
        self,
        topics_dir: Optional[Path] = None,
        maps_dir: Optional[Path] = None,
        artifacts_dir: Optional[Path] = None,
        static_dir: Optional[Path] = None,
        metadata_db_path: Optional[Path] = None,
        features: Optional[Dict[str, bool]] = None,
        latex_config: Optional[LaTeXConfig] = None,
        dtd_config: Optional[DTDConfig] = None,
        number_headings: bool = False,
        enable_cross_refs: bool = True,
        show_toc: bool = True,
        process_latex: bool = False,
        latex_settings: Optional[dict] = None,
    ):
        # Directory configuration
        self.topics_dir = topics_dir.resolve() if topics_dir else None
        self.maps_dir = maps_dir or self._default_dir("DITA_MAPS_DIR", "app/dita/maps")
        self.artifacts_dir = artifacts_dir or self._default_dir("DITA_ARTIFACTS_DIR", "app/dita/artifacts")
        self.static_dir = static_dir or self._default_dir("DITA_STATIC_DIR", "app/static")
        self.metadata_db_path = metadata_db_path or Path('metadata.db')
        self.number_headings = number_headings
        self.enable_cross_refs = enable_cross_refs
        self.show_toc = show_toc
        self.process_latex = process_latex

        # Add DTD configuration
        self.dtd_config = dtd_config or DTDConfig(
            base_path=Path("app/dita/dtd"),
            schemas_path=Path("app/dita/dtd/schemas"),
            catalog_path=Path("app/dita/dtd/catalog.xml")
        )

        # Add DTD paths
        self.dtd_schemas_dir = self._default_dir("DITA_DTD_SCHEMAS", "app/dita/dtd/schemas")
        self.dtd_catalog_path = Path(os.getenv("DITA_DTD_CATALOG", "app/dita/dtd/catalog.xml"))

        # Ensure DTD directories exist
        if self.dtd_config.base_path:
            self.dtd_config.base_path.mkdir(parents=True, exist_ok=True)
        if self.dtd_config.schemas_path:
            self.dtd_config.schemas_path.mkdir(parents=True, exist_ok=True)

        # Features configuration
        self.features = features or {
            "process_latex": False,
            "number_headings": False,
            "enable_cross_refs": True,
            "process_artifacts": False,
            "show_toc": True
        }

        # ID validation patterns
        self.validation_patterns = {
                    "map": r"^map-[a-zA-Z0-9_\-]+$",
                    "topic": r"^topic-[a-zA-Z0-9_\-]+$",
                    "heading": r"^heading-[a-zA-Z0-9_\-]+-h[1-6]$",
                    "artifact": r"^artifact-[a-zA-Z0-9_\-]+-[a-zA-Z0-9_\-]+$",
                }
        self.reset_generated_ids = True


        # LaTeX configuration
        self.latex_config = latex_config or LaTeXConfig()

        # First, use os.getenv() to get the directory (returns a string or None)
        env_var = os.getenv("DITA_TOPICS_DIR", "app/dita/topics")

        # Convert the environment variable (or fallback) to a Path object
        if isinstance(env_var, str):
            topics_dir = Path(env_var)


    def _default_dir(self, env_var: str, default: str) -> Path:
        """
        Fetch the directory from the environment or use a default value.
        Creates the directory if it doesn't exist.

        Args:
            env_var (str): The environment variable to check.
            default (str): Default directory to use if the environment variable is not set.

        Returns:
            Path: The resolved directory path.
        """
        dir_path = os.getenv(env_var, default)
        resolved_path = Path(dir_path).resolve()

        # Create directory if it doesn't exist
        resolved_path.mkdir(parents=True, exist_ok=True)

        return resolved_path

    def validate(self) -> None:
        """
        Validate the configuration and create required directories if they don't exist.
        """
        for attr in ["topics_dir", "maps_dir", "artifacts_dir", "static_dir"]:
            dir_path = getattr(self, attr, None)
            if dir_path:
                Path(dir_path).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_environment(cls) -> "DITAConfig":
        """Create a DITAConfig instance from environment variables."""
        # Directory configuration
        topics_dir = os.getenv("DITA_TOPICS_DIR")
        maps_dir = os.getenv("DITA_MAPS_DIR")
        artifacts_dir = os.getenv("DITA_ARTIFACTS_DIR")
        static_dir = os.getenv("DITA_STATIC_DIR")

        # Features configuration
        features = {
            "process_latex": os.getenv("DITA_PROCESS_LATEX", "False").lower() in ("true", "1", "yes"),
            "number_headings": os.getenv("DITA_NUMBER_HEADINGS", "False").lower() in ("true", "1", "yes"),
            "enable_cross_refs": os.getenv("DITA_ENABLE_CROSS_REFS", "True").lower() in ("true", "1", "yes"),
            "process_artifacts": os.getenv("DITA_PROCESS_ARTIFACTS", "False").lower() in ("true", "1", "yes"),
            "show_toc": os.getenv("DITA_SHOW_TOC", "True").lower() in ("true", "1", "yes")
        }

        # LaTeX configuration
        latex_config = LaTeXConfig(
            macros=json.loads(os.getenv("DITA_LATEX_MACROS", "{}")),
            throw_on_error=os.getenv("DITA_LATEX_THROW_ON_ERROR", "False").lower() in ("true", "1", "yes"),
            output_mode=os.getenv("DITA_LATEX_OUTPUT_MODE", "html")
        ) if features["process_latex"] else None

        # DTD configuration
        dtd_config = DTDConfig(
            validation_mode=os.getenv("DITA_DTD_VALIDATION_MODE", "strict"),
            specialization_handling=os.getenv("DITA_DTD_SPECIALIZATION", "inherit"),
            attribute_inheritance=os.getenv("DITA_DTD_INHERITANCE", "merge"),
            enable_caching=os.getenv("DITA_DTD_CACHING", "True").lower() in ("true", "1", "yes"),
            base_path=Path(os.getenv("DITA_DTD_PATH", "app/dita/dtd")),
            schemas_path=Path(os.getenv("DITA_DTD_SCHEMAS_PATH", "app/dita/dtd/schemas")),
            catalog_path=Path(os.getenv("DITA_DTD_CATALOG_PATH", "app/dita/dtd/catalog.xml"))
        )


        return cls(
            topics_dir=Path(topics_dir) if topics_dir else None,
            maps_dir=Path(maps_dir) if maps_dir else None,
            artifacts_dir=Path(artifacts_dir) if artifacts_dir else None,
            static_dir=Path(static_dir) if static_dir else None,
            features=features,
            latex_config=latex_config,
            dtd_config=dtd_config
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
