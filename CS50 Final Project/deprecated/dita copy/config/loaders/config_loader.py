"""Configuration loading and validation for DITA processing."""
from typing import Dict, Optional, Any, Union, Type, List, Tuple
from pathlib import Path
import json
import yaml
from datetime import datetime
from dataclasses import dataclass, field
import logging
from dataclasses import dataclass
import re

from ...models.types import (
    ProcessingPhase,
    ValidationResult,
    ValidationMessage,
    ValidationSeverity,
    ElementType,
    ProcessingStatus,
    ProcessingState
)

from ...validation_manager import ValidationManager
from ...schema.schema_manager import SchemaManager, SchemaComposer, CompositionStrategy
from ...utils.cache import ContentCache, CacheEntryType
from ...utils.logger import DITALogger
from ...event_manager import EventManager, EventType

@dataclass
class ConfigState:
    """Track configuration state."""
    environment: str
    last_updated: datetime
    is_loaded: bool = False
    is_validated: bool = False
    error_messages: list[str] = field(default_factory=list)  # Use dataclass field

    def __post_init__(self):
        if self.error_messages is None:
            self.error_messages = []

class ConfigLoader:
    """
    Handles loading and validation of configuration files.
    Manages environment-specific configuration and inheritance.
    """

    def __init__(
        self,
        config_path: Union[str, Path],
        validation_manager: ValidationManager,
        schema_manager: SchemaManager,
        content_cache: ContentCache,
        event_manager: EventManager,
        logger: Optional[DITALogger] = None
    ):
        """Initialize configuration loader."""
        self.config_path = Path(config_path)
        self.validation_manager = validation_manager
        self.schema_manager = schema_manager
        self.cache = content_cache
        self.event_manager = event_manager
        self.logger = logger or DITALogger(name=__name__)

        self.schema_composer = SchemaComposer()

        # State tracking
        self._state = ConfigState(
            environment="development",
            last_updated=datetime.now()
        )

        # Configuration storage
        self._loaded_configs: Dict[str, Any] = {}
        self._env_config: Dict[str, Any] = {}
        self._base_config: Dict[str, Any] = {}

    def load_environment_config(self, environment: str) -> ValidationResult:
        """
        Load environment-specific configuration.

        Args:
            environment: Environment name (development, production, etc.)

        Returns:
            ValidationResult: Validation status and messages
        """
        try:
            # Get environment config file
            config_file = self.config_path / f"{environment}.yml"
            if not config_file.exists():
                return ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path="environment",
                        message=f"Config file not found: {config_file}",
                        severity=ValidationSeverity.ERROR,
                        code="missing_config_file"
                    )]
                )

            # Load base configuration first
            base_config = self._load_base_config()

            # Load environment specific config
            with open(config_file, 'r') as f:
                env_config = yaml.safe_load(f)

            # Use schema composer directly for merging
            merged_config = self.schema_composer.compose(
                base=base_config,
                extension=env_config,
                strategy=CompositionStrategy.MERGE
            )

            # Validate merged configuration
            validation_result = self.validation_manager.validate_configuration_set(merged_config)
            if validation_result.is_valid:
                self._env_config = merged_config
                self._state.environment = environment
                self._state.is_loaded = True
                self._state.last_updated = datetime.now()

                # Emit configuration loaded event
                self.event_manager.emit(
                    EventType.STATE_CHANGE,
                    element_id="env_config",
                    state_info=ProcessingStatus(
                        element_id="env_config",
                        phase=ProcessingPhase.DISCOVERY,
                        state=ProcessingState.COMPLETED
                    )
                )

            return validation_result

        except Exception as e:
            self.logger.error(f"Failed to load environment config: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="environment",
                    message=str(e),
                    severity=ValidationSeverity.ERROR,
                    code="config_load_error"
                )]
            )

    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration."""
        try:
            base_config_path = self.config_path / "base.yml"
            if not base_config_path.exists():
                return {}

            with base_config_path.open('r') as f:
                config = yaml.safe_load(f)
                self._base_config = config or {}
                return self._base_config

        except Exception as e:
            self.logger.error(f"Error loading base config: {str(e)}")
            return {}

    def load_config_file(
        self,
        filename: str,
        required: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Load a specific configuration file.

        Args:
            filename: Name of config file to load
            required: Whether file is required

        Returns:
            Optional[Dict[str, Any]]: Loaded configuration or None
        """
        try:
            # Check cache first
            cache_key = f"config_{filename}"
            if cached := self.cache.get(cache_key, CacheEntryType.CONFIG):
                return cached

            file_path = self.config_path / filename
            if not file_path.exists():
                if required:
                    raise FileNotFoundError(f"Required config file not found: {filename}")
                return None

            # Load based on file extension
            with open(file_path) as f:
                if filename.endswith('.json'):
                    config = json.load(f)
                elif filename.endswith(('.yml', '.yaml')):
                    config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file type: {filename}")

            # Cache result
            self.cache.set(
                key=cache_key,
                data=config,
                entry_type=CacheEntryType.CONFIG,
                element_type=ElementType.UNKNOWN,
                phase=ProcessingPhase.DISCOVERY
            )

            # Store in loaded configs
            self._loaded_configs[filename] = config
            return config

        except Exception as e:
            self.logger.error(f"Error loading config file {filename}: {str(e)}")
            if required:
                raise
            return None

    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration using ValidationManager."""
        return self.validation_manager.validate(
            content=config,
            validation_type="config",
            context={"environment": self._environment}
        )

    def store_bulk_metadata(self, metadata_entries: List[Tuple[str, Dict[str, Any]]]) -> None:
        """Store multiple config entries."""
        try:
            for config_name, config_data in metadata_entries:
                self._loaded_configs[config_name] = config_data

                # Cache the config
                self.cache.set(
                    key=f"config_{config_name}",
                    data=config_data,
                    entry_type=CacheEntryType.CONFIG,
                    element_type=ElementType.UNKNOWN,
                    phase=ProcessingPhase.DISCOVERY
                )

        except Exception as e:
            self.logger.error(f"Error storing bulk metadata: {str(e)}")
            raise

    def get_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get loaded configuration by name."""
        return self._loaded_configs.get(name)

    def get_env_config(self) -> Dict[str, Any]:
        """Get current environment configuration."""
        return self._env_config

    def get_state(self) -> ConfigState:
        """Get current configuration state."""
        return self._state

    def reload_all(self) -> ValidationResult:
        """Reload all configuration files."""
        try:
            # Clear caches
            self.cache.invalidate_by_pattern("config_*")
            self._loaded_configs.clear()

            # Reload base config
            self._load_base_config()

            # Reload environment config
            return self.load_environment_config(self._state.environment)

        except Exception as e:
            self.logger.error(f"Error reloading configurations: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="reload",
                    message=str(e),
                    severity=ValidationSeverity.ERROR,
                    code="reload_error"
                )]
            )

    def _load_validation_patterns(self) -> None:
        """Load validation patterns from configuration."""
        try:
            patterns_file = self.config_path / "validation_patterns.json"
            self.logger.debug(f"Attempting to load validation patterns from: {patterns_file}")

            if not patterns_file.exists():
                self.logger.warning(f"Validation patterns file not found: {patterns_file}")
                self._validation_patterns = {}
                return

            with open(patterns_file) as f:
                patterns_data = json.load(f)
                self.logger.debug(f"Loaded raw patterns data: {patterns_data.keys()}")

            # Validate structure
            if not isinstance(patterns_data, dict):
                self.logger.error("Validation patterns must be a dictionary")
                self._validation_patterns = {}
                return

            # Extract patterns with proper structure checking
            patterns = patterns_data.get("patterns", {})
            if not isinstance(patterns, dict):
                self.logger.error("'patterns' must be a dictionary")
                self._validation_patterns = {}
                return

            # Store patterns and log success
            self._validation_patterns = patterns
            self.logger.debug(
                f"Successfully loaded {len(self._validation_patterns)} validation patterns: "
                f"{list(self._validation_patterns.keys())}"
            )

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in validation patterns: {str(e)}")
            self._validation_patterns = {}
        except KeyError as e:
            self.logger.error(f"Missing key in validation patterns: {str(e)}")
            self._validation_patterns = {}
        except Exception as e:
            self.logger.error(f"Error loading validation patterns: {str(e)}")
            self._validation_patterns = {}

    def cleanup(self) -> None:
        """Clean up loader resources."""
        try:
            self._loaded_configs.clear()
            self._env_config.clear()
            self._base_config.clear()
            self.cache.invalidate_by_pattern("config_*")
            self._state = ConfigState(
                environment=self._state.environment,
                last_updated=datetime.now()
            )
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise
