from pathlib import Path
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from app_config import (
    DITAConfig,
    get_environment
)
from .models.types import ProcessingOptions, DITAProcessingConfig, DITAParserConfig

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages DITA configuration loading and validation."""

    def __init__(self):
        self._config: Optional[DITAConfig] = None
        self.logger = logging.getLogger(__name__)

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

    def validate_config(self, config: 'DITAConfig') -> bool:
        """
        Validate the provided DITAConfig instance.

        Args:
            config: DITAConfig instance to validate.

        Returns:
            bool: True if the configuration is valid, False otherwise.
        """
        # Check required fields
        required_fields = ['topics_dir']
        for field in required_fields:
            if not hasattr(config, field) or getattr(config, field) is None:
                self.logger.error(f"Missing required configuration field: {field}")
                return False

        # Validate topics directory
        topics_dir = config.topics_dir
        if topics_dir is None:
            self.logger.error("Topics directory is not set")
            return False

        try:
            topics_path = Path(topics_dir)
            if not topics_path.exists():
                self.logger.error(f"Topics directory does not exist: {topics_dir}")
                return False
        except Exception as e:
            self.logger.error(f"Invalid topics directory path: {str(e)}")
            return False

        return True



    def get_config(self) -> 'DITAConfig':
        """
        Get current DITA configuration.

        Returns:
            Current DITAConfig instance
        """
        if not self._config:
            self._config = self.load_config()
        return self._config

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update DITA configuration.

        Args:
            updates: Dictionary of updates to apply
        """
        try:
            config = self.get_config()

            # Apply updates
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            # Validate updated config
            if not self.validate_config(config):
                raise ValueError("Invalid configuration after update")

            self._config = config

        except Exception as e:
            self.logger.error(f"Config update failed: {str(e)}")
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



# Create singleton instance
config_manager = ConfigManager()
