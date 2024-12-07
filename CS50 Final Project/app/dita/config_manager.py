from config import DITAPathConfig
from pathlib import Path
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from config import (
    DITAConfig,
    DITAPathConfig,
    DITAParserConfig,
    DITAPathConfig,
    DITAProcessingConfig,
    get_environment
)
from .utils.types import ProcessingOptions

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages DITA configuration loading and validation."""

    def __init__(self):
        self._config: Optional[DITAConfig] = None
        self.logger = logging.getLogger(__name__)

    def load_config(self) -> 'DITAConfig':
        """
        Load DITA configuration based on environment.

        Returns:
            DITAConfig instance
        """
        try:
            env = get_environment()
            self.logger.debug(f"Loading DITA config for environment: {env}")

            # Load config based on environment
            if env == 'development':
                config = self._load_development_config()
            elif env == 'production':
                config = self._load_production_config()
            else:
                config = self._load_default_config()

            # Validate config
            if self.validate_config(config):
                self._config = config
                return config
            else:
                raise ValueError("Invalid DITA configuration")

        except Exception as e:
            self.logger.error(f"Failed to load DITA config: {str(e)}")
            raise

    def validate_config(self, config: 'DITAConfig') -> bool:
        """
        Validate DITA configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if configuration is valid
        """
        try:
            # Validate paths
            if not self.validate_paths(config.paths):
                return False

            # Validate parser settings
            if not self._validate_parser_config(config.parser):
                return False

            # Validate processing settings
            if not self._validate_processing_config(config.processing):
                return False

            return True

        except Exception as e:
            self.logger.error(f"Config validation failed: {str(e)}")
            return False

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

    def validate_paths(self, paths: 'DITAPathConfig') -> bool:
        """
        Validate path configuration.

        Args:
            paths: Path configuration to validate

        Returns:
            True if paths are valid
        """
        try:
            required_paths = [
                paths.dita_root,
                paths.maps_dir,
                paths.topics_dir,
                paths.output_dir,
                paths.artifacts_dir
            ]

            # Check path existence
            return all(path.exists() for path in required_paths)

        except Exception as e:
            self.logger.error(f"Path validation failed: {str(e)}")
            return False

    def setup_paths(self, paths: 'DITAPathConfig') -> None:
        """
        Setup required paths.

        Args:
            paths: Path configuration
        """
        try:
            # Create directories if they don't exist
            for path_attr in dir(paths):
                if not path_attr.startswith('_'):
                    path = getattr(paths, path_attr)
                    if isinstance(path, Path):
                        path.mkdir(parents=True, exist_ok=True)

        except Exception as e:
            self.logger.error(f"Path setup failed: {str(e)}")
            raise

    def _load_development_config(self) -> 'DITAConfig':
        """Load development configuration."""
        config = DITAConfig()
        config.processing.enable_debug = True
        return config

    def _load_production_config(self) -> 'DITAConfig':
        """Load production configuration."""
        config = DITAConfig()
        config.processing.enable_debug = False
        return config

    def _load_default_config(self) -> 'DITAConfig':
        """Load default configuration."""
        return DITAConfig()

    def _validate_parser_config(self, parser_config: 'DITAParserConfig') -> bool:
        """Validate parser configuration."""
        # Add specific parser validation if needed
        return True

    def _validate_processing_config(self, processing_config: 'DITAProcessingConfig') -> bool:
        """Validate processing configuration."""
        # Add specific processing validation if needed
        return True

# Create singleton instance
config_manager = ConfigManager()
