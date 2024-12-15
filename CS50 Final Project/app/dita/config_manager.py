# app/dita/config_manager.py
from typing import Optional, TYPE_CHECKING
from pathlib import Path
import os
import logging
import sqlite3
from typing import Dict, Any, Optional
from dataclasses import dataclass

from app_config import DITAConfig, get_environment
from .models.types import (
    TrackedElement,
    DITAProcessingConfig,
    DITAParserConfig,
    ProcessingPhase,
    ProcessingState,

)

if TYPE_CHECKING:
    from .context_manager import ContextManager

class ConfigManager:
    """Manages DITA configuration and context management."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_flags = {
            "enable_toc": True,
            "enable_heading_numbering": True,
            "enable_cross_refs": True,
        }
        self.validation_patterns = {
            "map": r"^map-[a-zA-Z0-9_\-]+$",
            "topic": r"^topic-[a-zA-Z0-9_\-]+$",
            "heading": r"^heading-[a-zA-Z0-9_\-]+-h[1-6]$",
            "artifact": r"^artifact-[a-zA-Z0-9_\-]+-[a-zA-Z0-9_\-]+$",
        }
        self.default_metadata = {
            "language": "en",
            "display_flags": {"visible": True, "enabled": True, "expanded": True},
        }

    def initialize(self) -> None:
        """Initialize both configuration and context management."""
        try:
            # Load configuration first
            if not self._config:
                self._config = self.load_config()

            # Initialize context manager with metadata DB path
            if not self._context_manager:
                metadata_db_path = self._get_metadata_db_path()
                self._ensure_metadata_db(metadata_db_path)
                self._context_manager = ContextManager(str(metadata_db_path))  # Convert Path to str

            self.logger.info("Configuration and context management initialized successfully")

        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            raise


    def load_config(self) -> 'DITAConfig':
        """
        Load and validate the DITA configuration based on the current environment.

        Returns:
            DITAConfig instance
        """
        try:
            # Determine the environment
            env = get_environment()
            self.logger.debug(f"Loading DITA config for environment: {env}")

            # Load environment-specific configuration
            if env == 'development':
                config = self._load_development_config()
            elif env == 'production':
                config = self._load_production_config()
            else:
                config = self._load_default_config()

            # Additional settings from environment variables
            number_headings = os.getenv('DITA_NUMBER_HEADINGS', 'False').lower() in ('true', '1', 'yes')
            topics_dir = os.getenv('DITA_TOPICS_DIR')

            # Logging additional settings
            self.logger.debug(f"Number headings enabled: {number_headings}")
            self.logger.debug(f"Topics directory: {topics_dir}")

            # Update the configuration object
            config.number_headings = number_headings

            # Safely handle topics_dir Path creation
            if topics_dir is not None:
                config.topics_dir = Path(topics_dir)
            elif config.topics_dir is None:
                config.topics_dir = Path('topics')  # Default value
            # If config.topics_dir is already a Path, leave it as is

            # Validate configuration
            if not self.validate_config(config):
                raise ValueError(f"Invalid DITA configuration: {config}")

            # Cache and return the configuration
            self._config = config
            self.logger.info(f"DITA configuration loaded successfully for {env} environment.")
            return config

        except Exception as e:
            self.logger.error(f"Failed to load DITA configuration: {str(e)}", exc_info=True)
            raise

    def _load_development_config(self) -> 'DITAConfig':
        """
        Load development-specific configuration.

        Returns:
            DITAConfig: Development configuration instance.
        """
        try:
            config = DITAConfig(
                number_headings=True,
                enable_cross_refs=True,
                show_toc=True,
            )
            self.logger.debug("Development configuration loaded successfully.")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load development configuration: {str(e)}", exc_info=True)
            raise


    def _load_production_config(self) -> 'DITAConfig':
        """
        Load production-specific configuration.

        Returns:
            DITAConfig: Production configuration instance.
        """
        try:
            config = DITAConfig(
                number_headings=False,
                enable_cross_refs=True,
                show_toc=True,
            )
            self.logger.debug("Production configuration loaded successfully.")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load production configuration: {str(e)}", exc_info=True)
            raise


    def _load_default_config(self) -> 'DITAConfig':
        """
        Load default configuration.

        Returns:
            DITAConfig: Default configuration instance.
        """
        try:
            config = DITAConfig()
            self.logger.debug("Default configuration loaded successfully.")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load default configuration: {str(e)}", exc_info=True)
            raise


    def get_config(self) -> 'DITAConfig':
        """
        Get current DITA configuration.

        Returns:
            Current DITAConfig instance
        """
        if not self._config:
            self._config = self.load_config()
        return self._config


    def get_feature_flag(self, flag: str) -> bool:
            """Get the value of a feature flag."""
            return self.feature_flags.get(flag, False)

    def get_validation_pattern(self, element_type: str) -> str:
        """Get the validation pattern for a specific element type."""
        return self.validation_patterns.get(element_type, r"^[a-zA-Z0-9_\-]+$")

    def get_default_metadata(self) -> Dict[str, Any]:
        """Get the default metadata configuration."""
        return self.default_metadata

    def update_feature_flag(self, flag: str, value: bool) -> None:
        """Update a feature flag dynamically."""
        self.feature_flags[flag] = value
        self.logger.debug(f"Feature flag {flag} updated to {value}")

    def update_validation_pattern(self, element_type: str, pattern: str) -> None:
        """Update a validation pattern dynamically."""
        self.validation_patterns[element_type] = pattern
        self.logger.debug(f"Validation pattern for {element_type} updated to {pattern}")

    def _get_metadata_db_path(self) -> Path:
            """Get the metadata database path from config or environment."""
            if self._config:
                return self._config.metadata_db_path

            # Default to environment variable or fallback path
            db_path = os.getenv('DITA_METADATA_DB', 'metadata.db')
            return Path(db_path)

    def _ensure_metadata_db(self, db_path: Path) -> None:
        """Ensure metadata database exists and has correct schema."""
        try:
            # Create parent directories if needed
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # Initialize database connection
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()

            # Check if schema needs to be created
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = {row[0] for row in cur.fetchall()}

            if not existing_tables:
                self.logger.info("Creating metadata database schema")
                self._create_metadata_schema(conn)

            conn.close()
            self.logger.debug(f"Metadata database ready at {db_path}")

        except Exception as e:
            self.logger.error(f"Failed to initialize metadata database: {str(e)}")
            raise

    def _create_metadata_schema(self, conn: sqlite3.Connection) -> None:
        """Create metadata database schema."""
        try:
            with open('metadata.sql', 'r') as f:
                schema = f.read()
                conn.executescript(schema)
            conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to create schema: {str(e)}")
            raise

    def get_context_manager(self) -> 'ContextManager':
            """Get the context manager instance."""
            if not self._context_manager:
                self.initialize()
            if not self._context_manager:
                raise RuntimeError("Failed to initialize context manager")
            return self._context_manager

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update DITA configuration and reinitialize context if needed.

        Args:
            updates: Dictionary of updates to apply
        """
        try:
            config = self.get_config()

            # Track if metadata DB path is changing
            old_db_path = getattr(config, 'metadata_db_path', None)

            # Apply updates
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            # Validate updated config
            if not self.validate_config(config):
                raise ValueError("Invalid configuration after update")

            self._config = config

            # Reinitialize context manager if DB path changed
            new_db_path = getattr(config, 'metadata_db_path', None)
            if old_db_path != new_db_path:
                self.logger.info("Metadata DB path changed, reinitializing context manager")
                if self._context_manager:
                    self._context_manager.cleanup()
                self._context_manager = None
                self.initialize()

        except Exception as e:
            self.logger.error(f"Config update failed: {str(e)}")
            raise

    def validate_config(self, config: DITAConfig) -> bool:
            """
            Validate the provided DITAConfig instance.
            """
            try:
                # Check required fields
                required_fields = ['topics_dir', 'metadata_db_path']
                for field in required_fields:
                    if not hasattr(config, field) or getattr(config, field) is None:
                        self.logger.error(f"Missing required configuration field: {field}")
                        return False

                # Validate topics directory
                topics_dir = config.topics_dir
                if topics_dir is None:
                    self.logger.error("Topics directory is not set")
                    return False

                # Validate metadata DB path
                db_path = config.metadata_db_path
                if not db_path.parent.exists():
                    self.logger.warning(f"Metadata DB directory doesn't exist: {db_path.parent}")
                    try:
                        db_path.parent.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        self.logger.error(f"Could not create metadata DB directory: {e}")
                        return False

                return True

            except Exception as e:
                self.logger.error(f"Configuration validation failed: {str(e)}")
                return False


    def _validate_processing_features(self, config: DITAConfig) -> bool:
        """Validate feature configuration."""
        try:
            # Get features from config
            features = getattr(config, "features", {})
            if not isinstance(features, dict):
                self.logger.error("Invalid features configuration: must be a dictionary")
                return False

            # Required features - must be present and boolean
            required_features = {
                "process_latex": True,
                "number_headings": True,
                "enable_cross_refs": True,
                "process_artifacts": True,
                "show_toc": True,
            }

            # Validate required features exist and are proper type
            for feature, default in required_features.items():
                if feature not in features:
                    self.logger.debug(f"Setting default value for {feature}: {default}")
                    features[feature] = default
                elif not isinstance(features[feature], bool):
                    self.logger.error(f"Feature {feature} must be boolean")
                    return False

            # LaTeX-specific validation
            if features.get("process_latex"):
                latex_config = getattr(config, "latex_config", {})

                # Define LaTeX settings validation rules
                required_latex_settings = {
                    "macros": dict,
                    "throw_on_error": bool,
                    "output_mode": str,
                }

                if isinstance(latex_config, dict):
                    # Validate LaTeX config dictionary
                    for setting, expected_type in required_latex_settings.items():
                        if setting not in latex_config:
                            self.logger.error(f"Missing required LaTeX setting: {setting}")
                            return False
                        if not isinstance(latex_config[setting], expected_type):
                            self.logger.error(f"Invalid type for LaTeX setting {setting}: expected {expected_type}")
                            return False
                elif hasattr(latex_config, "__dict__"):
                    # Validate if LaTeXConfig is a custom object with attributes
                    for setting, expected_type in required_latex_settings.items():
                        if not hasattr(latex_config, setting):
                            self.logger.error(f"Missing required LaTeX setting: {setting}")
                            return False
                        if not isinstance(getattr(latex_config, setting), expected_type):
                            self.logger.error(f"Invalid type for LaTeX setting {setting}: expected {expected_type}")
                            return False
                else:
                    self.logger.error("Invalid LaTeX configuration: must be a dictionary or a valid LaTeXConfig object")
                    return False

            # Store validated features back in config
            config.features = features
            return True

        except Exception as e:
            self.logger.error(f"Feature validation failed: {str(e)}")
            return False


    def setup_paths(self, config: 'DITAConfig') -> None:
            """
            Setup required paths in the DITA configuration.

            Args:
                config (DITAConfig): The configuration with paths to set up.
            """
            try:
                paths_to_setup = [
                    config.topics_dir,
                    config.maps_dir,
                    config.artifacts_dir,
                    config.static_dir,
                ]

                for path in paths_to_setup:
                    if not path.exists():
                        self.logger.info(f"Creating missing directory: {path}")
                        path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.error(f"Path setup failed: {str(e)}", exc_info=True)
                raise

    def validate_paths(self, config: 'DITAConfig') -> bool:
        """
        Validate the paths in the DITA configuration.

        Args:
            config (DITAConfig): The configuration to validate.

        Returns:
            bool: True if all required paths are valid, False otherwise.
        """
        try:
            required_paths = [
                config.topics_dir,
                config.maps_dir,
                config.artifacts_dir,
                config.static_dir,
            ]

            # Check that all paths exist and are directories
            for path in required_paths:
                if not path.exists() or not path.is_dir():
                    self.logger.error(f"Invalid path: {path} does not exist or is not a directory.")
                    return False

            return True
        except Exception as e:
            self.logger.error(f"Path validation failed: {str(e)}", exc_info=True)
            return False


    def _validate_parser_config(self, parser_config: 'DITAParserConfig') -> bool:
        """
        Validate parser configuration.

        Args:
            parser_config (DITAParserConfig): Parser configuration to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            # Example validations for a hypothetical DITAParserConfig class
            if not isinstance(parser_config.enable_strict_mode, bool):
                raise ValueError("enable_strict_mode must be a boolean.")
            if not isinstance(parser_config.parse_external_references, bool):
                raise ValueError("parse_external_references must be a boolean.")
            if not isinstance(parser_config.support_legacy_dita, bool):
                raise ValueError("support_legacy_dita must be a boolean.")

            # Validate file extensions (if applicable)
            if not isinstance(parser_config.allowed_extensions, list) or not all(
                isinstance(ext, str) for ext in parser_config.allowed_extensions
            ):
                raise ValueError("allowed_extensions must be a list of strings.")

            return True
        except Exception as e:
            self.logger.error(f"Parser configuration validation failed: {str(e)}", exc_info=True)
            return False


    def _validate_processing_config(self, processing_config: 'DITAProcessingConfig') -> bool:
        """
        Validate processing configuration.

        Args:
            processing_config (DITAProcessingConfig): Processing configuration to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            # Validate individual attributes
            if not isinstance(processing_config.enable_debug, bool):
                raise ValueError("enable_debug must be a boolean.")
            if not isinstance(processing_config.show_toc, bool):
                raise ValueError("show_toc must be a boolean.")
            if not isinstance(processing_config.enable_cross_refs, bool):
                raise ValueError("enable_cross_refs must be a boolean.")
            if not isinstance(processing_config.process_latex, bool):
                raise ValueError("process_latex must be a boolean.")

            return True
        except Exception as e:
            self.logger.error(f"Processing configuration validation failed: {str(e)}", exc_info=True)
            return False

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self._context_manager:
                self._context_manager.cleanup()
                self._context_manager = None
            self._config = None
            self.logger.debug("Configuration manager cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")
            raise

# Create singleton instance
config_manager = ConfigManager()
