from typing import Dict, List, Optional, Any, Union, TypeVar, Set, Type
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import os
import json
import logging
import uuid
from datetime import datetime
import yaml
from functools import lru_cache
from uuid import uuid4
import re
from contextlib import contextmanager

# Pydantic for schema validation

from pydantic import BaseModel, ValidationError, create_model, Field

# Project-specific imports
from .models.types import (
    ProcessingPhase,
    ProcessingState,
    ElementType,
    ContentType,
    DITAProcessingConfig,
    DITAParserConfig,
    DITAElementType,
    ProcessorConfig,
    ValidationResult,
    ValidationMessage,
    ValidationSeverity
)

# Import handlers we'll integrate with
from .context_manager import ContextManager
from app.dita.utils.id_handler import DITAIDHandler
from app.dita.utils.metadata import MetadataHandler
from .event_manager import EventManager, EventType
from .utils.logger import DITALogger
from .utils.cache import ContentCache


# Types and enums
class ConfigScope(Enum):
    """Configuration scope types."""
    GLOBAL = "global"      # Application-wide settings
    PIPELINE = "pipeline"  # Pipeline-specific settings
    COMPONENT = "component"  # Component-specific settings
    FEATURE = "feature"    # Feature flags and toggles
    RULE = "rule"         # Processing rules

class ConfigEnvironment(Enum):
    """Configuration environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

@dataclass
class ConfigState:
    """Track configuration state."""
    environment: ConfigEnvironment
    last_updated: datetime
    is_loaded: bool = False
    is_validated: bool = False
    error_messages: List[str] = field(default_factory=list)



class FeatureError(Exception):
    """Custom exception for feature-related errors."""
    pass

class FeatureScope(Enum):
    """Scope levels for features."""
    GLOBAL = "global"         # Application-wide features
    PIPELINE = "pipeline"     # Pipeline-specific features
    CONTENT = "content"       # Content-specific features
    COMPONENT = "component"   # Component-specific features
    UI = "ui"                 # User interface features

@dataclass
class Feature:
    """Feature definition with metadata."""
    name: str
    scope: FeatureScope
    default: bool
    description: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert feature to dictionary representation."""
        return {
            "name": self.name,
            "scope": self.scope.value,
            "default": self.default,
            "description": self.description,
            "dependencies": self.dependencies,
            "conflicts": self.conflicts,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat() if self.modified_at else None
        }


class ProcessingRuleType(Enum):
    """Types of processing rules."""
    ELEMENT = "element"           # Basic element processing
    TRANSFORMATION = "transform"  # Content transformation
    VALIDATION = "validation"     # Content validation
    SPECIALIZATION = "special"    # Content specialization
    ENRICHMENT = "enrichment"     # Content enrichment
    PUBLICATION = "publication"   # Publication-specific

@dataclass
class ProcessingRule:
    """Definition of a processing rule."""
    rule_id: str
    rule_type: ProcessingRuleType
    element_type: ElementType
    config: Dict[str, Any]
    conditions: Dict[str, Any]
    priority: int = 0
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: Optional[datetime] = None


class ValidationScope(Enum):
    """Validation scope types."""
    CONTENT = "content"       # Content structure validation
    METADATA = "metadata"     # Metadata validation
    REFERENCE = "reference"   # Reference/citation validation
    STRUCTURE = "structure"   # Document structure validation
    SEMANTIC = "semantic"     # Semantic validation
    PUBLICATION = "publication"  # Publication requirements

@dataclass
class ValidationSchema:
    """Schema definition with metadata."""
    name: str
    scope: ValidationScope
    model: Type[BaseModel]
    description: Optional[str] = None
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineType(Enum):
    """Types of processing pipelines."""
    DITA = "dita"             # DITA XML processing
    MARKDOWN = "markdown"      # Markdown processing
    LATEX = "latex"           # LaTeX processing
    MEDIA = "media"           # Media processing
    ARTIFACT = "artifact"     # Artifact processing
    PUBLICATION = "publication"  # Publication processing

@dataclass
class PipelineConfig:
    """Pipeline configuration with metadata."""
    pipeline_type: PipelineType
    config: Dict[str, Any]
    features: Dict[str, bool]
    processors: List[str]
    validators: List[str]
    transformers: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: Optional[datetime] = None


class ConfigManager:
    """
    Centralized configuration management for the DITA processing pipeline.
    Manages environment-specific configurations, feature flags, and processing rules.
    """
    _instance = None  # Singleton instance

    def __new__(cls, *args, **kwargs):
        """Ensure singleton pattern."""
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        event_manager: EventManager,
        context_manager: ContextManager,
        metadata_handler: MetadataHandler,
        id_handler: DITAIDHandler,
        content_cache: ContentCache,
        logger: Optional[DITALogger] = None,
    ):
        """
        Initialize configuration manager.

        Args:
            env: Optional environment name (development/testing/production)
            config_path: Optional path to configuration files
            event_manager: Optional event management system
            logger: Optional logger instance
        """
        # Skip initialization if already initialized
        if hasattr(self, '_initialized'):
            return

        # Core initialization
        self.logger = logger or logging.getLogger(__name__)
        self.event_manager = event_manager
        self.context_manager = context_manager
        self.metadata_handler = metadata_handler
        self.content_cache = content_cache
        self.id_handler = id_handler or DITAIDHandler()
        self.config_path = config_path or Path("config")
        self.cache = ContentCache()



        # Load schema
        self._attribute_schema = self._load_attribute_schema()

        # Cache for resolved configurations
        self._config_cache: Dict[str, Any] = {}

        # Register for configuration events
        self._register_event_handlers()


        # Initialize configuration stores
        self._env_config: Dict[str, Any] = {}
        self._component_config: Dict[str, Dict[str, Any]] = {}

        # Initialize registries and caches
        self._feature_registry: Dict[str, Feature] = {}
        self._rule_registry: Dict[str, Dict[str, ProcessingRule]] = {}
        self._schema_registry: Dict[str, ValidationSchema] = {}
        self._validation_results: Dict[str, Dict[int, ValidationResult]] = {}
        self._element_rule_index: Dict[ElementType, Dict[ProcessingRuleType, List[str]]] = {}
        self._dita_rules: Dict[str, Any] = {}
        self._dita_type_mapping: Dict[str, str] = {}
        self._keyref_config: Dict[str, Any] = {}
        self._processing_hierarchy: List[str] = []
        self._global_defaults: Dict[str, Any] = {}
        self._element_defaults: Dict[str, Any] = {}

        # Cache and state
        self._config_cache: Dict[str, Any] = {}
        self._runtime_config: Dict[str, Any] = {}
        self._dirty_configs: Set[str] = set()

        # Determine environment
        self.environment = ConfigEnvironment(env.lower()) if env else self._detect_environment()

        # Initialize state tracking
        self._state = ConfigState(
            environment=self.environment,
            last_updated=datetime.now()
        )

        # Track initialization
        self._initialized = True
        self.logger.info(f"ConfigManager initialized in {self.environment.value} environment")

    def _detect_environment(self) -> ConfigEnvironment:
        """Detect running environment from environment variables."""
        env = os.getenv("DITA_ENV", "development").lower()
        try:
            return ConfigEnvironment(env)
        except ValueError:
            self.logger.warning(f"Invalid environment '{env}', defaulting to development")
            return ConfigEnvironment.DEVELOPMENT

    @property
    def state(self) -> ConfigState:
        """Get current configuration state."""
        return self._state

    def initialize(self) -> None:
        """
        Initialize configuration system.
        Loads environment config, validates schemas, and prepares registries.
        """
        try:
            self.logger.info("Initializing configuration system")

            # Load environment configuration
            self.load_environment_config()

            # Initialize schema registry # Load configuration files
            self._load_config_files()

            # Track initialization
            self._initialized = True

            # Initialize schema registry
            self._initialize_schemas()

            # Load component configurations
            self._load_component_configs()

            # Initialize registries
            self._initialize_feature_registry()
            self._initialize_rule_registry()

            # Validate all configurations
            self._validate_all_configs()

            # Update state
            self._state.is_loaded = True
            self._state.last_updated = datetime.now()

            self.logger.info(f"ConfigManager initialized in {self.environment.value} environment")

            # Emit initialization event
            if self.event_manager:
                self.event_manager.emit(
                    EventType.STATE_CHANGE,
                    old_state=ProcessingState.PENDING,
                    new_state=ProcessingState.COMPLETED
                )

        except Exception as e:
            self.logger.error(f"Configuration initialization failed: {str(e)}")
            self._state.error_messages.append(str(e))
            raise


    ##########################
    # Initialization methods #
    ##########################

    def _initialize_schemas(self) -> None:
        """Initialize validation schemas."""
        try:
            # Load schema definitions from config files
            schema_dir = self.config_path / "schemas"
            if not schema_dir.exists():
                return

            for schema_file in schema_dir.glob("*.json"):
                with open(schema_file) as f:
                    schema_def = json.load(f)
                    schema_name = schema_file.stem

                    # Create Pydantic model
                    model = self._create_pydantic_model(schema_name, schema_def)

                    # Create ValidationSchema with the model
                    schema = ValidationSchema(
                        name=schema_name,
                        scope=ValidationScope(schema_def.get("scope", "default")),
                        model=model,
                        description=schema_def.get("description"),
                        version=schema_def.get("version", "1.0"),
                        metadata=schema_def.get("metadata", {})
                    )

                    # Store ValidationSchema in registry
                    self._schema_registry[schema_name] = schema

            self.logger.debug("Schema initialization completed")

        except Exception as e:
            self.logger.error(f"Schema initialization failed: {str(e)}")
            raise


    def _load_component_configs(self) -> None:
        """Load component-specific configurations."""
        try:
            component_dir = self.config_path / "components"
            if not component_dir.exists():
                return

            for config_file in component_dir.glob("*.json"):
                with open(config_file) as f:
                    component_config = json.load(f)
                    component_name = config_file.stem
                    self._component_config[component_name] = component_config

            self.logger.debug("Component configurations loaded")

        except Exception as e:
            self.logger.error(f"Component config loading failed: {str(e)}")
            raise

    def _initialize_feature_registry(self) -> None:
        """Initialize feature registry from loaded configuration."""
        try:
            # Store loaded JSON config temporarily
            raw_features = self._feature_registry

            # Reset feature registry to empty state
            self._feature_registry = {}

            # Process each feature from raw config
            for name, feature_data in raw_features.items():
                if isinstance(feature_data, dict):
                    self.register_feature(
                        name=name,
                        default=feature_data.get('enabled', False),
                        scope=FeatureScope(feature_data.get('scope', 'global')),
                        description=feature_data.get('description'),
                        dependencies=feature_data.get('dependencies', []),
                        conflicts=feature_data.get('conflicts', []),
                        metadata=feature_data.get('metadata', {})
                    )

            self.logger.debug("Feature registry initialized")

        except Exception as e:
            self.logger.error(f"Feature registry initialization failed: {str(e)}")
            raise

    def _initialize_rule_registry(self) -> None:
        """Initialize processing rules registry from loaded configuration."""
        try:
            # Store loaded JSON config temporarily
            raw_rules = self._rule_registry

            # Reset registries
            self._rule_registry = {}
            self._element_rule_index = {}

            # Process each rule from raw config
            for rule_type_str, type_rules in raw_rules.items():
                if isinstance(type_rules, dict):
                    for element_type_str, rule_data in type_rules.items():
                        if isinstance(rule_data, dict):
                            try:
                                rule_type = ProcessingRuleType(rule_type_str)
                                element_type = ElementType(element_type_str)

                                self.register_processing_rule(
                                    rule_type=rule_type,
                                    element_type=element_type,
                                    rule_config=rule_data.get('config', {}),
                                    conditions=rule_data.get('conditions', {}),
                                    priority=rule_data.get('priority', 0),
                                    metadata=rule_data.get('metadata', {})
                                )
                            except ValueError as e:
                                self.logger.warning(f"Invalid rule type or element type: {str(e)}")
                                continue

            self.logger.debug("Rule registry initialized")

        except Exception as e:
            self.logger.error(f"Rule registry initialization failed: {str(e)}")
            raise

    def _validate_feature_config(self, feature: Feature) -> bool:
        """
        Validate feature configuration.

        Args:
            feature: Feature to validate

        Returns:
            bool: True if configuration is valid
        """
        try:
            # Validate basic requirements
            if not feature.name:
                return False

            # Validate scope
            if not isinstance(feature.scope, FeatureScope):
                return False

            # Validate dependencies exist
            for dep in feature.dependencies:
                if dep not in self._feature_registry:
                    self.logger.error(f"Unknown dependency {dep} for feature {feature.name}")
                    return False

            # Validate conflicts exist
            for conflict in feature.conflicts:
                if conflict not in self._feature_registry:
                    self.logger.error(f"Unknown conflict {conflict} for feature {feature.name}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Feature validation error: {str(e)}")
            return False

    def _validate_all_configs(self) -> bool:
        """Validate all loaded configurations."""
        try:
            all_valid = True

            # Validate feature configurations
            for name, feature in self._feature_registry.items():
                if not self._validate_feature_config(feature):
                    all_valid = False
                    self.logger.error(f"Invalid feature configuration: {name}")

            # Validate processing rules
            for rule_type_str, rules in self._rule_registry.items():
                for rule_id, rule in rules.items():
                    if not self.validate_processing_rule(rule.config):
                        all_valid = False
                        self.logger.error(
                            f"Invalid rule configuration: {rule_id} "
                            f"(type: {rule_type_str})"
                        )

            # Validate keyref configuration if present
            if self._keyref_config:
                if not self._validate_keyref_config():
                    all_valid = False
                    self.logger.error("Invalid keyref configuration")

            # Update state
            self._state.is_validated = all_valid
            self._state.last_updated = datetime.now()

            return all_valid

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            return False

    def _validate_keyref_config(self) -> bool:
        """Validate keyref configuration."""
        try:
            required_sections = [
                "processing_hierarchy",
                "global_defaults",
                "element_defaults",
                "keyref_resolution"
            ]

            # Check required sections exist
            for section in required_sections:
                if section not in self._keyref_config:
                    self.logger.error(f"Missing required section in keyref config: {section}")
                    return False

            # Validate processing hierarchy
            if not isinstance(self._keyref_config["processing_hierarchy"].get("order"), list):
                self.logger.error("Invalid processing hierarchy format")
                return False

            # Validate global defaults
            if not isinstance(self._keyref_config["global_defaults"], dict):
                self.logger.error("Invalid global defaults format")
                return False

            # Validate element defaults
            if not isinstance(self._keyref_config["element_defaults"], dict):
                self.logger.error("Invalid element defaults format")
                return False

            # Validate keyref resolution rules
            resolution = self._keyref_config["keyref_resolution"]
            if not (isinstance(resolution.get("scopes"), list) and
                    isinstance(resolution.get("fallback_order"), list) and
                    isinstance(resolution.get("inheritance_rules"), dict)):
                self.logger.error("Invalid keyref resolution rules format")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Keyref config validation error: {str(e)}")
            return False


    #################################
    # Core configuration management #
    #################################

    def load_environment_config(self) -> None:
        """
        Load environment-specific configuration.
        Handles development, testing, and production environments.
        """
        try:
            # Get environment-specific config file
            config_file = self.config_path / f"{self.environment.value}.yml"
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")

            # Load base configuration first
            base_config = self._load_base_config()

            # Load environment specific config
            with open(config_file, 'r') as f:
                env_config = yaml.safe_load(f)

            # Deep merge with base config
            self._env_config = self._deep_merge(base_config, env_config)

            # Apply environment variables overrides
            self._apply_env_overrides()

            # Validate loaded configuration
            if not self.validate_config(self._env_config):
                raise ValueError("Invalid environment configuration")

            # Update state
            self._state.last_updated = datetime.now()
            self._state.is_validated = True

            self.logger.info(
                f"Loaded configuration for {self.environment.value} environment"
            )

            # Emit configuration loaded event
            if self.event_manager:
                self.event_manager.emit(
                    EventType.STATE_CHANGE,
                    element_id="env_config",
                    old_state=ProcessingState.PENDING,
                    new_state=ProcessingState.COMPLETED
                )

        except Exception as e:
            self.logger.error(f"Failed to load environment config: {str(e)}")
            self._state.error_messages.append(str(e))
            raise

    def _load_config_files(self) -> None:
        """Load configuration files from config directory."""
        try:
            config_dir = self.config_path / "configs"

            # Load feature flags
            with open(config_dir / "feature_flags.json") as f:
                self._feature_registry = json.load(f)["features"]

            # Load processing rules
            with open(config_dir / "processing_rules.json") as f:
                self._rule_registry = json.load(f)["rules"]

            # Load DITA processing rules
            with open(config_dir / "dita_processing_rules.json") as f:
                dita_config = json.load(f)
                self._dita_rules = dita_config["element_rules"]
                self._dita_type_mapping = dita_config["element_type_mapping"]

            # Load validation patterns
            with open(config_dir / "validation_patterns.json") as f:
                patterns_config = json.load(f)
                self.validation_patterns = {
                    k: v["pattern"]
                    for k, v in patterns_config["patterns"].items()
                }
                self.default_metadata = patterns_config["default_metadata"]

            # Load keyref configuration
            with open(config_dir / "keyref_config.json") as f:
                keyref_config = json.load(f)
                self._keyref_config = keyref_config
                self._processing_hierarchy = keyref_config["processing_hierarchy"]["order"]
                self._global_defaults = keyref_config["global_defaults"]
                self._element_defaults = keyref_config["element_defaults"]

            self.logger.info("Successfully loaded configuration files")

        except FileNotFoundError as e:
            self.logger.error(f"Configuration file not found: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in configuration file: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading configuration files: {str(e)}")
            raise

    def get_dita_element_rules(self, element_type: DITAElementType) -> Dict[str, Any]:
        """Get processing rules for a DITA element type."""
        try:
            # Get rule path from type mapping
            rule_path = self._dita_type_mapping.get(element_type.value)
            if not rule_path:
                return self._dita_rules["default"]["unknown"]

            # Navigate to rule in hierarchy
            current = self._dita_rules
            for part in rule_path.split('.'):
                current = current[part]

            return current

        except Exception as e:
            self.logger.error(f"Error getting DITA element rules: {str(e)}")
            return self._dita_rules["default"]["unknown"]

    def load_component_config(self, component: str) -> Dict[str, Any]:
        """
        Load configuration for a specific component.

        Args:
            component: Component identifier

        Returns:
            Dict containing component configuration
        """
        try:
            # Check cache first
            cache_key = f"component_{component}"
            if cache_key in self._config_cache:
                return self._config_cache[cache_key]

            # Get component config file
            config_file = self.config_path / "components" / f"{component}.yml"
            if not config_file.exists():
                raise FileNotFoundError(f"Component config not found: {config_file}")

            # Load component config
            with open(config_file, 'r') as f:
                component_config = yaml.safe_load(f)

            # Apply environment-specific overrides
            env_overrides = self._env_config.get('components', {}).get(component, {})
            if env_overrides:
                component_config = self._deep_merge(component_config, env_overrides)

            # Validate component config
            if not self.validate_config(component_config):
                raise ValueError(f"Invalid configuration for component: {component}")

            # Store in component registry
            self._component_config[component] = component_config

            # Cache the result
            self._config_cache[cache_key] = component_config

            self.logger.debug(f"Loaded configuration for component: {component}")

            return component_config

        except Exception as e:
            self.logger.error(f"Failed to load component config {component}: {str(e)}")
            raise


    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update configuration with runtime overrides.

        Args:
            updates: Dictionary of configuration updates

        Returns:
            bool: True if update was successful
        """
        try:
            # Validate updates
            if not self.validate_config(updates):
                raise ValueError("Invalid configuration updates")

            # Store previous state for rollback
            previous_config = self._runtime_config.copy()

            try:
                # Apply updates
                for key, value in updates.items():
                    if '.' in key:
                        # Handle nested updates
                        self._update_nested_config(key, value)
                    else:
                        self._runtime_config[key] = value

                # Mark configs as dirty
                self._dirty_configs.update(updates.keys())

                # Update state
                self._state.last_updated = datetime.now()
                self._state.is_validated = True

                # Emit config updated event
                if self.event_manager:
                    self.event_manager.emit(
                        EventType.STATE_CHANGE,
                        element_id="runtime_config",
                        old_state=ProcessingState.PROCESSING,
                        new_state=ProcessingState.COMPLETED
                    )

                return True

            except Exception as e:
                # Rollback on error
                self._runtime_config = previous_config
                raise

        except Exception as e:
            self.logger.error(f"Failed to update configuration: {str(e)}")
            return False

    # Helper methods


    def _merge_attributes(
        self,
        base: Dict[str, Any],
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge attributes according to inheritance rules.

        Args:
            base: Base attributes
            updates: Updates to apply

        Returns:
            Dict[str, Any]: Merged attributes
        """
        try:
            result = base.copy()
            inheritance_rules = self._keyref_config["keyref_resolution"]["inheritance_rules"]

            for key, value in updates.items():
                if (key == "props" and
                    inheritance_rules.get("props") == "merge" and
                    isinstance(value, dict)):
                    result["props"] = {**result.get("props", {}), **value}
                elif (key == "outputclass" and
                      inheritance_rules.get("outputclass") == "append" and
                      isinstance(value, str)):
                    existing = result.get("outputclass", "").split()
                    new = value.split()
                    result["outputclass"] = " ".join(list(dict.fromkeys(existing + new)))
                else:
                    result[key] = value

            return result

        except Exception as e:
            self.logger.error(f"Error merging attributes: {str(e)}")
            return base


    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration."""
        base_config_file = self.config_path / "base.yml"
        if not base_config_file.exists():
            return {}

        with open(base_config_file, 'r') as f:
            return yaml.safe_load(f)


    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        prefix = "DITA_CONFIG_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower().replace('__', '.')
                try:
                    # Convert environment variable to appropriate type
                    typed_value = self._convert_env_value(value)
                    self._update_nested_config(config_key, typed_value)
                except Exception as e:
                    self.logger.warning(f"Failed to apply env override {key}: {str(e)}")

    def _update_nested_config(self, key: str, value: Any) -> None:
        """Update nested configuration value."""
        current = self._runtime_config
        parts = key.split('.')
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value

    def _determine_schema(self, config: Dict[str, Any]) -> Optional[str]:
        """Determine appropriate schema for configuration."""
        # Logic to determine schema based on config content
        if 'pipeline' in config:
            return 'pipeline_config'
        elif 'component' in config:
            return 'component_config'
        elif 'feature' in config:
            return 'feature_config'
        return 'base_config'

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Handle booleans
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        # Handle numbers
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            # Handle lists
            if value.startswith('[') and value.endswith(']'):
                return [v.strip() for v in value[1:-1].split(',')]
            # Return as string
            return value

    ########################
    # Attribute Resolution #
    ########################

    def resolve_attributes(
        self,
        element_type: ElementType,
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Resolve attributes through hierarchy levels with validation.

        Args:
            element_type: Type of element needing configuration
            context_id: Optional context ID for context-aware resolution

        Returns:
            Dict containing resolved attributes
        """
        try:
            # Get hierarchy levels from schema
            hierarchy = self._attribute_schema["hierarchy"]["resolution_order"]

            # Start with empty config
            resolved_config = {}

            # Process each level in order
            for level in hierarchy:
                # Get level config
                level_config = self._get_level_config(level, element_type)

                if level_config:
                    # Validate level configuration
                    valid_config = self._validate_level(
                        config=level_config,
                        level=level,
                        element_type=element_type
                    )

                    if valid_config:
                        # Merge valid configuration
                        resolved_config = self._merge_configs(
                            base=resolved_config,
                            override=valid_config,
                            level=level
                        )

            return resolved_config

        except Exception as e:
            self.logger.error(f"Error resolving attributes: {str(e)}")
            return {}

    def _get_level_config(
        self,
        level: str,
        element_type: ElementType
    ) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific hierarchy level using schema.

        Args:
            level: Hierarchy level to resolve
            element_type: Type of element being processed

        Returns:
            Optional[Dict]: Level configuration if found
        """
        try:
            schema = self._attribute_schema

            # Get level configuration from schema
            if level_config := schema["hierarchy"]["levels"].get(level):
                # Try cache first
                cache_key = f"{level}_{element_type.value}"
                if cached := self.content_cache.get(cache_key):
                    return cached

                config = {}

                # Get sources for this level
                sources = level_config.get("sources", [])

                # Load and merge all source configurations
                for source in sources:
                    source_path = Path(source["path"])
                    if not source_path.exists():
                        continue

                    with open(source_path) as f:
                        source_config = json.load(f)

                    # Apply source-specific resolution rules
                    if rules := source.get("resolution_rules"):
                        resolved = self._apply_resolution_rules(
                            config=source_config,
                            rules=rules,
                            element_type=element_type
                        )
                        config.update(resolved)
                    else:
                        # Direct update if no special rules
                        config.update(source_config)

                # Cache if config was resolved
                if config:
                    self.content_cache.set(
                        cache_key,
                        config,
                        element_type,
                        ProcessingPhase.DISCOVERY
                    )

                return config

            return None

        except Exception as e:
            self.logger.error(f"Error getting level config: {str(e)}")
            return None

    def _apply_resolution_rules(
        self,
        config: Dict[str, Any],
        rules: Dict[str, Any],
        element_type: ElementType
    ) -> Dict[str, Any]:
        """
        Apply resolution rules to configuration.

        Args:
            config: Raw configuration
            rules: Resolution rules from schema
            element_type: Type of element being processed

        Returns:
            Dict: Resolved configuration
        """
        try:
            resolved = {}

            # Apply type-specific resolution
            if element_rules := rules.get("element_types", {}).get(element_type.value):
                # Get paths to resolve
                for path in element_rules.get("paths", []):
                    value = self._resolve_config_path(config, path)
                    if value is not None:
                        resolved[path["target"]] = value

            # Apply general resolution
            if general_rules := rules.get("general"):
                for path in general_rules.get("paths", []):
                    value = self._resolve_config_path(config, path)
                    if value is not None:
                        resolved[path["target"]] = value

            return resolved

        except Exception as e:
            self.logger.error(f"Error applying resolution rules: {str(e)}")
            return {}

    def _resolve_config_path(
        self,
        config: Dict[str, Any],
        path_info: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Resolve value from configuration using path information.

        Args:
            config: Configuration to resolve from
            path_info: Path and resolution information

        Returns:
            Optional[Any]: Resolved value if found
        """
        try:
            # Get path components
            path = path_info["source"].split(".")

            # Navigate config
            current = config
            for component in path:
                if component not in current:
                    if path_info.get("required"):
                        raise KeyError(f"Required path {path_info['source']} not found")
                    return None
                current = current[component]

            # # Apply any transforms
            # if transform := path_info.get("transform"):
            #     current = self._apply_transform(current, transform)

            return current

        except Exception as e:
            self.logger.error(f"Error resolving config path: {str(e)}")
            return None

    def _validate_level(
        self,
        config: Dict[str, Any],
        level: str,
        element_type: ElementType
    ) -> Optional[Dict[str, Any]]:
        """
        Validate configuration against schema rules for level.

        Args:
            config: Configuration to validate
            level: Hierarchy level being validated
            element_type: Type of element being processed

        Returns:
            Optional[Dict]: Validated configuration or None if invalid
        """
        try:
            schema = self._attribute_schema
            validation_rules = schema["validation_rules"]
            feature_defs = schema["feature_definitions"]

            # Validate required attributes
            if required := validation_rules["required_attributes"].get(element_type.value):
                missing = [attr for attr in required if attr not in config]
                if missing:
                    self.logger.warning(
                        f"Missing required attributes for {element_type}: {missing}"
                    )
                    return None

            # Validate attribute patterns
            for attr, pattern in validation_rules["attribute_patterns"].items():
                if attr in config:
                    value = str(config[attr])
                    if not re.match(pattern, value):
                        self.logger.warning(
                            f"Invalid {attr} value: {value} for {element_type}"
                        )
                        del config[attr]

            # Validate allowed values
            for attr, allowed in validation_rules["allowed_values"].items():
                if attr in config and config[attr] not in allowed:
                    self.logger.warning(
                        f"Invalid {attr} value: {config[attr]} for {element_type}"
                    )
                    del config[attr]

            # Validate feature flags
            if "features" in config:
                for feature, state in config["features"].items():
                    if feature_def := feature_defs.get(feature):
                        # Validate feature requirements
                        if feature_def.get("requires_libraries"):
                            # Feature validation logic here
                            pass
                    else:
                        self.logger.warning(f"Unknown feature flag: {feature}")
                        del config["features"][feature]

            return config

        except Exception as e:
            self.logger.error(f"Error validating level config: {str(e)}")
            return None

    def _merge_configs(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any],
        level: str
    ) -> Dict[str, Any]:
        """
        Merge configurations following schema rules.

        Args:
            base: Base configuration
            override: Override configuration
            level: Current hierarchy level

        Returns:
            Dict: Merged configuration
        """
        try:
            schema = self._attribute_schema
            inheritance_rules = schema["inheritance_rules"]
            result = base.copy()

            # Process each attribute type according to rules
            for key, value in override.items():
                if rule := inheritance_rules.get(key):
                    match rule["strategy"]:
                        case "append":
                            # Append with delimiter (e.g., classes)
                            if key in result:
                                existing = result[key].split(rule["delimiter"])
                                new = value.split(rule["delimiter"])
                                if rule.get("unique"):
                                    combined = list(dict.fromkeys(existing + new))
                                else:
                                    combined = existing + new
                                result[key] = rule["delimiter"].join(combined)
                            else:
                                result[key] = value

                        case "merge":
                            # Merge dictionaries (e.g., feature flags)
                            if key in result and isinstance(result[key], dict):
                                result[key] = {**result[key], **value}
                            else:
                                result[key] = value

                        case "deep_merge":
                            # Deep merge (e.g., metadata)
                            if key in result:
                                result[key] = self._deep_merge(
                                    result[key],
                                    value,
                                    preserve_arrays=rule.get("preserve_arrays", False)
                                )
                            else:
                                result[key] = value

                        case "override":
                            # Simple override
                            if not rule.get("preserve_existing") or key not in result:
                                result[key] = value
                else:
                    # Default to override if no rule specified
                    result[key] = value

            return result

        except Exception as e:
            self.logger.error(f"Error merging configs: {str(e)}")
            return base

    def _deep_merge(
        self,
        dict1: Dict[str, Any],
        dict2: Dict[str, Any],
        preserve_arrays: bool = False
    ) -> Dict[str, Any]:
        """
        Deep merge helper for nested structures.

        Args:
            dict1: First dictionary
            dict2: Second dictionary
            preserve_arrays: Whether to preserve array structures

        Returns:
            Dict: Deeply merged dictionary
        """
        try:
            result = dict1.copy()

            for key, value in dict2.items():
                if key in result:
                    if isinstance(result[key], dict) and isinstance(value, dict):
                        # Recurse for nested dicts
                        result[key] = self._deep_merge(
                            result[key],
                            value,
                            preserve_arrays
                        )
                    elif isinstance(result[key], list) and isinstance(value, list):
                        # Handle arrays based on preservation flag
                        if preserve_arrays:
                            result[key].extend(value)
                        else:
                            result[key] = value
                    else:
                        # Override for non-dict/non-list
                        result[key] = value
                else:
                    result[key] = value

            return result

        except Exception as e:
            self.logger.error(f"Error in deep merge: {str(e)}")
            return dict1

    #######################
    # Feature management #
    #######################

    def register_feature(
        self,
        name: str,
        default: bool = False,
        scope: FeatureScope = FeatureScope.GLOBAL,
        description: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        conflicts: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a new feature with the configuration system."""
        try:
            # Validate feature name
            if not re.match(r'^[a-z][a-z0-9_]*$', name):
                raise FeatureError(f"Invalid feature name: {name}")

            # Check if feature already exists
            if name in self._feature_registry:
                raise FeatureError(f"Feature already registered: {name}")

            # Create feature instance
            feature = Feature(
                name=name,
                scope=scope,
                default=default,
                description=description,
                dependencies=dependencies or [],
                conflicts=conflicts or [],
                metadata=metadata or {}
            )

            # Store in registry (as Feature object, not dict)
            self._feature_registry[name] = feature

            # Initialize feature state in runtime config
            self._runtime_config.setdefault('features', {})[name] = default

            self.logger.debug(f"Registered feature: {name} (default: {default})")

        except Exception as e:
            self.logger.error(f"Failed to register feature {name}: {str(e)}")
            raise

    def get_feature_state(self, name: str) -> bool:
        """Get current state of a feature."""
        try:
            # Validate feature exists
            feature = self._feature_registry.get(name)
            if not feature:
                raise FeatureError(f"Unknown feature: {name}")

            # Check runtime configuration first
            runtime_features = self._runtime_config.get('features', {})
            if name in runtime_features:
                return runtime_features[name]

            # Fall back to feature's default state
            return feature.default

        except Exception as e:
            self.logger.error(f"Error getting feature state for {name}: {str(e)}")
            return False

    def update_feature(self, name: str, state: bool) -> None:
        """Update feature state with dependency and conflict checking."""
        try:
            # Validate feature exists
            feature = self._feature_registry.get(name)
            if not feature:
                raise FeatureError(f"Unknown feature: {name}")

            # Check dependencies if enabling
            if state:
                for dep in feature.dependencies:
                    if not self.get_feature_state(dep):
                        raise FeatureError(f"Cannot enable {name}: requires {dep}")

            # Check conflicts
            for conflict in feature.conflicts:
                if self.get_feature_state(conflict) and state:
                    raise FeatureError(f"Cannot enable {name}: conflicts with {conflict}")

            # Update state
            self._runtime_config.setdefault('features', {})[name] = state

            # Update modification time
            feature.modified_at = datetime.now()

            # Emit event if available
            if self.event_manager:
                self.event_manager.emit(
                    EventType.STATE_CHANGE,
                    element_id=f"feature_{name}",
                    old_state=ProcessingState.PROCESSING,
                    new_state=ProcessingState.COMPLETED
                )

        except Exception as e:
            self.logger.error(f"Failed to update feature {name}: {str(e)}")
            raise

    def get_effective_features(self, context_id: str) -> Dict[str, bool]:
        """Get effective feature states for a given context."""
        try:
            if not self.context_manager:
                return {name: feature.default for name, feature in self._feature_registry.items()}

            # Get context
            context = self.context_manager.get_context(context_id)
            if not context:
                return {}

            # Start with global features
            effective_features: Dict[str, bool] = {}

            for name, feature in self._feature_registry.items():
                # Get base state
                state = self.get_feature_state(name)

                # Apply scope-based rules
                if feature.scope == FeatureScope.GLOBAL:
                    effective_features[name] = state
                elif feature.scope == FeatureScope.PIPELINE:
                    pipeline_features = context.get('pipeline_features', {})
                    effective_features[name] = pipeline_features.get(name, state)
                elif feature.scope == FeatureScope.CONTENT:
                    content_type = context.get('content_type')
                    if content_type:
                        content_features = self._get_content_type_features(content_type)
                        effective_features[name] = content_features.get(name, state)
                elif feature.scope == FeatureScope.COMPONENT:
                    component = context.get('component')
                    if component:
                        component_features = (
                            self._component_config
                            .get(component, {})
                            .get('features', {})
                        )
                        effective_features[name] = component_features.get(name, state)

            return effective_features

        except Exception as e:
            self.logger.error(f"Error getting effective features: {str(e)}")
            return {}

    def _get_content_type_features(self, content_type: str) -> Dict[str, bool]:
        """Get feature overrides for a content type."""
        try:
            return self._env_config.get('content_features', {}).get(content_type, {})
        except Exception as e:
            self.logger.error(f"Error getting content type features: {str(e)}")
            return {}

    @contextmanager
    def feature_override(self, overrides: Dict[str, bool]):
        """
        Temporarily override feature states.

        Args:
            overrides: Dict mapping feature names to temporary states
        """
        original_states: Dict[str, bool] = {}
        try:
            # Store original states
            original_states = {
                name: self.get_feature_state(name)
                for name in overrides
            }

            # Apply overrides
            for name, state in overrides.items():
                self.update_feature(name, state)

            yield

        finally:
            # Restore original states
            for name, state in original_states.items():
                self.update_feature(name, state)

    ####################
    # Processing rules #
    ####################


    def register_processing_rule(
        self,
        rule_type: ProcessingRuleType,
        rule_config: Dict[str, Any],
        element_type: ElementType,
        conditions: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register a new processing rule."""
        try:
            # Generate unique rule ID using id_handler
            rule_id = self.id_handler.generate_id(
                base=f"{rule_type.value}_{element_type.value}",
                element_type="rule"
            )

            # Create rule instance
            rule = ProcessingRule(
                rule_id=rule_id,
                rule_type=rule_type,
                element_type=element_type,
                config=rule_config,
                conditions=conditions or {},
                priority=priority,
                metadata=metadata or {}
            )

            # Initialize registries if needed
            rule_type_str = rule_type.value
            if rule_type_str not in self._rule_registry:
                self._rule_registry[rule_type_str] = {}

            # Store rule
            self._rule_registry[rule_type_str][rule_id] = rule

            # Update element type index
            if element_type not in self._element_rule_index:
                self._element_rule_index[element_type] = {}
            if rule_type not in self._element_rule_index[element_type]:
                self._element_rule_index[element_type][rule_type] = []

            self._element_rule_index[element_type][rule_type].append(rule_id)

            # Sort rules by priority
            self._element_rule_index[element_type][rule_type].sort(
                key=lambda x: self._rule_registry[rule_type_str][x].priority,
                reverse=True
            )

            return rule_id

        except Exception as e:
            self.logger.error(f"Failed to register processing rule: {str(e)}")
            raise

    def get_processing_rules(
        self,
        element_type: ElementType,
        rule_types: Optional[List[ProcessingRuleType]] = None,
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get applicable processing rules for an element type."""
        try:
            # Generate cache key
            rule_types_str = '_'.join(rt.value for rt in (rule_types or []))
            cache_key = f"rules_{element_type.value}_{rule_types_str}"

            # Check cache if no context needed
            if not context_id:
                if cached := self._config_cache.get(cache_key):
                    return cached

            # Get context if provided
            context = None
            if context_id and self.context_manager:
                context = self.context_manager.get_context(context_id)

            # Get all applicable rule types if none specified
            if not rule_types:
                rule_types = list(ProcessingRuleType)

            applicable_rules: Dict[str, Any] = {}

            # Collect rules from element type index
            for rule_type in rule_types:
                rule_type_str = rule_type.value
                if (element_type in self._element_rule_index and
                    rule_type in self._element_rule_index[element_type]):

                    for rule_id in self._element_rule_index[element_type][rule_type]:
                        rule = self._rule_registry[rule_type_str][rule_id]

                        # Skip inactive rules
                        if not rule.is_active:
                            continue

                        # Check conditions if context provided
                        if context and rule.conditions:
                            if not self._evaluate_rule_conditions(rule.conditions, context):
                                continue

                        # Add rule configuration
                        applicable_rules[rule_id] = rule.config

            # Cache results if no context-specific rules
            if not context_id:
                self._config_cache[cache_key] = applicable_rules

            return applicable_rules

        except Exception as e:
            self.logger.error(f"Error getting processing rules: {str(e)}")
            return {}

    def validate_processing_rule(self, rule: Dict[str, Any]) -> bool:
        """
        Validate a processing rule configuration.

        Args:
            rule: Rule configuration to validate

        Returns:
            bool: True if rule is valid
        """
        try:
            # Required rule components
            required_fields = {
                'operation': str,      # Rule operation type
                'target': str,         # Target element or attribute
                'action': dict,        # Action configuration
            }

            # Validate required fields and types
            for field, field_type in required_fields.items():
                if field not in rule:
                    self.logger.error(f"Missing required field: {field}")
                    return False
                if not isinstance(rule[field], field_type):
                    self.logger.error(f"Invalid type for {field}: expected {field_type}")
                    return False

            # Validate operation
            valid_operations = {
                'transform',    # Content transformation
                'validate',     # Content validation
                'enrich',      # Content enrichment
                'extract',     # Metadata extraction
                'inject',      # Content injection
                'specialize'   # Content specialization
            }
            if rule['operation'] not in valid_operations:
                self.logger.error(f"Invalid operation: {rule['operation']}")
                return False

            # Validate action configuration
            action_config = rule['action']
            if not self._validate_action_config(action_config, rule['operation']):
                return False

            # Validate optional fields
            if 'conditions' in rule and not isinstance(rule['conditions'], dict):
                self.logger.error("Invalid conditions format")
                return False

            if 'metadata' in rule and not isinstance(rule['metadata'], dict):
                self.logger.error("Invalid metadata format")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating processing rule: {str(e)}")
            return False

    def _validate_action_config(self, config: Dict[str, Any], operation: str) -> bool:
        """Validate operation-specific action configuration."""
        try:
            # Define validation rules for each operation type
            validation_rules = {
                'transform': {
                    'required': {'method', 'format'},
                    'optional': {'parameters', 'fallback'}
                },
                'validate': {
                    'required': {'constraints'},
                    'optional': {'severity', 'message'}
                },
                'enrich': {
                    'required': {'source', 'target'},
                    'optional': {'conditions', 'fallback'}
                },
                'extract': {
                    'required': {'pattern', 'target'},
                    'optional': {'format', 'multiple'}
                },
                'inject': {
                    'required': {'content', 'position'},
                    'optional': {'conditions', 'wrapper'}
                },
                'specialize': {
                    'required': {'type', 'rules'},
                    'optional': {'inheritance', 'overrides'}
                }
            }

            # Get validation rules for operation
            rules = validation_rules.get(operation)
            if not rules:
                self.logger.error(f"No validation rules for operation: {operation}")
                return False

            # Check required fields
            for field in rules['required']:
                if field not in config:
                    self.logger.error(f"Missing required field for {operation}: {field}")
                    return False

            # Check all fields are either required or optional
            all_fields = rules['required'].union(rules['optional'])
            for field in config:
                if field not in all_fields:
                    self.logger.error(f"Unknown field for {operation}: {field}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating action config: {str(e)}")
            return False

    def _evaluate_rule_conditions(
        self,
        conditions: Dict[str, Any],
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate rule conditions against context."""
        try:
            for key, value in conditions.items():
                # Handle nested conditions
                if isinstance(value, dict):
                    if not self._evaluate_rule_conditions(value, context.get(key, {})):
                        return False
                    continue

                # Handle list conditions (any match)
                if isinstance(value, list):
                    if key not in context or context[key] not in value:
                        return False
                    continue

                # Handle simple equality
                if context.get(key) != value:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error evaluating rule conditions: {str(e)}")
            return False



    ####################################
    # Validation and schema management #
    ####################################


    def _load_attribute_schema(self) -> Dict[str, Any]:
        """Load and validate attribute schema."""
        try:
            schema_path = Path(__file__).parent / "configs" / "attribute_schema.json"
            if not schema_path.exists():
                self.logger.error(f"Schema not found: {schema_path}")
                return {}  # Return empty dict instead of implicitly returning None

            with open(schema_path, 'r') as f:
                schema = json.load(f)

            # Validate schema structure
            required_sections = {
                "hierarchy", "attribute_types", "inheritance_rules",
                "validation_rules", "feature_definitions"
            }

            missing = required_sections - set(schema.keys())
            if missing:
                self.logger.error(f"Missing required schema sections: {missing}")
                return {}  # Return empty dict on validation failure

            return schema

        except Exception as e:
            self.logger.error(f"Error loading attribute schema: {str(e)}")
            return {}  # Return empty dict on any error

    def get_config(
        self,
        element_type: ElementType,
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get configuration using schema-based resolution.

        Args:
            element_type: Element type needing configuration
            context_id: Optional context ID for contextual resolution

        Returns:
            Dict containing resolved configuration
        """
        try:
            # Generate cache key
            cache_key = (f"config_{element_type.value}"
                        f"_{context_id if context_id else 'default'}")

            # Check cache
            if cached := self._config_cache.get(cache_key):
                return cached

            # Resolve attributes through hierarchy
            config = self.resolve_attributes(
                element_type=element_type,
                context_id=context_id
            )

            # Validate final configuration
            if not self.validate_config(config, element_type):
                self.logger.warning(
                    f"Invalid configuration for {element_type}. Using defaults."
                )
                return self._get_default_config(element_type)

            # Cache valid configuration
            self._config_cache[cache_key] = config

            return config

        except Exception as e:
            self.logger.error(f"Error getting config: {str(e)}")
            return self._get_default_config(element_type)

    def validate_config(
        self,
        config: Dict[str, Any],
        element_type: ElementType
    ) -> bool:
        """
        Validate complete configuration against schema.

        Args:
            config: Configuration to validate
            element_type: Element type for validation rules

        Returns:
            bool: True if configuration is valid
        """
        try:
            schema = self._attribute_schema
            validation_rules = schema["validation_rules"]

            # Check required attributes
            if required := validation_rules["required_attributes"].get(element_type.value):
                missing = [attr for attr in required if attr not in config]
                if missing:
                    self.logger.warning(
                        f"Missing required attributes for {element_type}: {missing}"
                    )
                    return False

            # Validate attribute types
            for attr, value in config.items():
                if attr_type := schema["attribute_types"].get(attr):
                    if not self._validate_attribute_type(value, attr_type):
                        self.logger.warning(
                            f"Invalid type for {attr}: {type(value)}"
                        )
                        return False

            # Validate feature combinations
            if "features" in config:
                if not self._validate_feature_combinations(
                    config["features"],
                    schema["feature_definitions"]
                ):
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating config: {str(e)}")
            return False

    def _validate_attribute_type(
        self,
        value: Any,
        type_def: Dict[str, Any]
    ) -> bool:
        """Validate attribute value against type definition."""
        try:
            expected_type = type_def["type"]

            match expected_type:
                case "boolean":
                    return isinstance(value, bool)
                case "string":
                    return isinstance(value, str)
                case "enum":
                    return (
                        isinstance(value, str) and
                        ("values" not in type_def or value in type_def["values"])
                    )
                case "object":
                    return isinstance(value, dict)
                case "array":
                    return isinstance(value, list)
                case _:
                    return False

        except Exception as e:
            self.logger.error(f"Error validating attribute type: {str(e)}")
            return False

    def _validate_feature_combinations(
        self,
        features: Dict[str, bool],
        feature_defs: Dict[str, Any]
    ) -> bool:
        """Validate feature flag combinations."""
        try:
            for feature, enabled in features.items():
                if not enabled:
                    continue

                if feature_def := feature_defs.get(feature):
                    # Check dependencies
                    if deps := feature_def.get("requires_features"):
                        missing = [dep for dep in deps if not features.get(dep)]
                        if missing:
                            self.logger.warning(
                                f"Feature {feature} requires: {missing}"
                            )
                            return False

                    # Check conflicts
                    if conflicts := feature_def.get("conflicts_with"):
                        active_conflicts = [
                            conf for conf in conflicts
                            if features.get(conf)
                        ]
                        if active_conflicts:
                            self.logger.warning(
                                f"Feature {feature} conflicts with: {active_conflicts}"
                            )
                            return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating feature combinations: {str(e)}")
            return False

    def register_validation_schema(
        self,
        name: str,
        schema: Dict[str, Any],
        scope: Union[str, ValidationScope],
        description: Optional[str] = None,
        version: str = "1.0",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a new validation schema.

        Args:
            name: Schema identifier
            schema: Schema definition
            scope: Validation scope
            description: Optional schema description
            version: Schema version
            metadata: Optional schema metadata
        """
        try:
            # Convert string scope to enum if needed
            if isinstance(scope, str):
                scope = ValidationScope(scope)

            # Validate schema name
            if not re.match(r'^[a-z][a-z0-9_]*$', name):
                raise ValueError(f"Invalid schema name: {name}")

            # Check if schema already exists
            if name in self._schema_registry:
                raise ValueError(f"Schema already registered: {name}")

            # Create Pydantic model from schema
            try:
                model = self._create_pydantic_model(name, schema)
            except Exception as e:
                raise ValueError(f"Invalid schema definition: {str(e)}")

            # Create Pydantic model and ValidationSchema
                model = self._create_pydantic_model(name, schema)
                validation_schema = ValidationSchema(
                    name=name,
                    scope=scope,
                    model=model,
                    description=description,
                    version=version,
                    metadata=metadata or {}
                )

                # Store schema
                self._schema_registry[name] = validation_schema
                self._validation_results[name] = {}

                self.logger.debug(f"Registered validation schema: {name}")

        except Exception as e:
            self.logger.error(f"Schema registration failed: {str(e)}")
            raise

    def validate_against_schema(
        self,
        data: Dict[str, Any],
        schema_name: str,
        context_id: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate data against a registered schema.

        Args:
            data: Data to validate
            schema_name: Schema identifier
            context_id: Optional context identifier

        Returns:
            ValidationResult with validation details
        """
        try:
            # Get schema
            schema = self._schema_registry.get(schema_name)
            if not schema:
                raise ValueError(f"Unknown schema: {schema_name}")

            result = ValidationResult(
                is_valid=False,
                messages=[]
            )

            try:
                # Validate using Pydantic model
                validated_data = schema.model.parse_obj(data)  # Use parse_obj instead
                result.is_valid = True

                # Get context for validation
                if context_id and self.context_manager:
                    context = self.context_manager.get_context(context_id)
                    if context:
                        # Perform scope-specific validation
                        scope_result = self._validate_scope(
                            validated_data,
                            schema.scope,
                            context
                        )
                        if not scope_result.is_valid:
                            result.messages.extend(scope_result.messages)
                            result.is_valid = False


            except ValidationError as e:
                # Process Pydantic validation errors
                for error in e.errors():
                    result.messages.append(
                        ValidationMessage(
                            path='.'.join(str(x) for x in error['loc']),
                            message=error['msg'],
                            severity=ValidationSeverity.ERROR,
                            code=error.get('type', 'validation_error')
                        )
                    )

            # Store validation result
            self._validation_results[schema_name][id(data)] = result
            return result

        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=str(e),
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def get_validation_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a registered validation schema."""
        try:
            schema = self._schema_registry.get(name)
            if not schema:
                return None

            return {
                'name': schema.name,
                'scope': schema.scope.value,
                'description': schema.description,
                'version': schema.version,
                'created_at': schema.created_at.isoformat(),
                'modified_at': schema.modified_at.isoformat() if schema.modified_at else None,
                'metadata': schema.metadata,
                'schema': schema.model.schema()  # Use schema() instead of schema_json()
            }

        except Exception as e:
            self.logger.error(f"Error getting validation schema: {str(e)}")
            return None

    def _create_pydantic_model(self, name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
        """Create a Pydantic model from schema definition."""
        try:
            # Convert schema fields to Pydantic field definitions
            fields = {}
            for field_name, field_def in schema.get('properties', {}).items():
                field_type = self._get_field_type(field_def)
                field_config = self._get_field_config(field_def)
                fields[field_name] = (field_type, Field(**field_config))

            # Create model dynamically
            model = create_model(
                name,
                **fields,
                __module__=__name__
            )

            return model

        except Exception as e:
            self.logger.error(f"Error creating Pydantic model: {str(e)}")
            raise

    def _get_field_type(self, field_def: Dict[str, Any]) -> Type:
        """Get Python type from schema field definition."""
        type_mapping = {
            'string': str,
            'integer': int,
            'number': float,
            'boolean': bool,
            'array': List,
            'object': Dict[str, Any],
            'null': None
        }
        return type_mapping.get(field_def.get('type', 'string'), str)

    def _get_field_config(self, field_def: Dict[str, Any]) -> Dict[str, Any]:
        """Get Pydantic field configuration from schema field definition."""
        config = {}

        # Handle required fields
        if field_def.get('required', False):
            config['required'] = True

        # Handle default values
        if 'default' in field_def:
            config['default'] = field_def['default']

        # Handle descriptions
        if 'description' in field_def:
            config['description'] = field_def['description']

        # Handle constraints
        for constraint in ['min_length', 'max_length', 'pattern', 'minimum', 'maximum']:
            if constraint in field_def:
                config[constraint] = field_def[constraint]

        return config




    #######################
    # Integration methods #
    #######################

    def _get_default_config(self, element_type: ElementType) -> Dict[str, Any]:
        """
        Get default configuration for element type from schema.
        Represents the base fallback layer in our hierarchy.

        Args:
            element_type: Element type needing defaults

        Returns:
            Dict containing default configuration
        """
        try:
            schema = self._attribute_schema

            # Start with global defaults
            defaults = schema.get("global_defaults", {}).copy()

            # Add element-specific defaults if they exist
            if element_defaults := schema.get("element_defaults", {}).get(element_type.value):
                defaults.update(element_defaults)

            # Add required attributes with their fallback values
            for attr, type_def in schema["attribute_types"].items():
                if attr not in defaults and type_def.get("required"):
                    defaults[attr] = type_def["fallback"]

            return defaults

        except Exception as e:
            self.logger.error(f"Error getting default config: {str(e)}")
            return {}

    def get_pipeline_config(
        self,
        pipeline_type: Union[str, PipelineType],
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get configuration for a specific pipeline type."""
        try:
            # Convert string pipeline type to enum if needed
            if isinstance(pipeline_type, str):
                pipeline_type = PipelineType(pipeline_type.lower())

            # Check cache first
            cache_key = f"pipeline_{pipeline_type.value}"
            if not context_id and cache_key in self._config_cache:
                return self._config_cache[cache_key]

            # Get base pipeline configuration
            base_config = self._env_config.get('pipelines', {}).get(pipeline_type.value, {})

            # Get pipeline features
            features = self._get_pipeline_features(pipeline_type)

            # Create pipeline config instance
            pipeline_config = PipelineConfig(
                pipeline_type=pipeline_type,
                config=base_config,
                features=features,
                processors=self._get_pipeline_processors(pipeline_type),
                validators=self._get_pipeline_validators(pipeline_type),
                transformers=self._get_pipeline_transformers(pipeline_type)
            )

            # Check for context manager and apply overrides if available
            if context_id is not None and self.context_manager is not None:
                try:
                    if context := self.context_manager.get_context(context_id):
                        pipeline_config = self._apply_context_overrides(
                            pipeline_config,
                            context
                        )
                except AttributeError:
                    self.logger.warning("Context manager get_context method not available")

            # Convert to dictionary for return
            config_dict = {
                'type': pipeline_type.value,
                'config': pipeline_config.config,
                'features': pipeline_config.features,
                'processors': pipeline_config.processors,
                'validators': pipeline_config.validators,
                'transformers': pipeline_config.transformers,
                'metadata': pipeline_config.metadata
            }

            # Cache if no context-specific configuration
            if not context_id:
                self._config_cache[cache_key] = config_dict

            return config_dict

        except Exception as e:
            self.logger.error(f"Error getting pipeline config: {str(e)}")
            return {}

    def get_component_features(
        self,
        component: str,
        context_id: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        Get feature states for a specific component.

        Args:
            component: Component identifier
            context_id: Optional context identifier

        Returns:
            Dict mapping feature names to their states
        """
        try:
            # Check cache first
            cache_key = f"features_{component}"
            if not context_id and cache_key in self._config_cache:
                return self._config_cache[cache_key]

            # Get component configuration
            component_config = self._component_config.get(component, {})

            # Get base features
            base_features = component_config.get('features', {}).copy()

            # Get feature overrides from environment config
            env_features = self._env_config.get('components', {}).get(
                component, {}
            ).get('features', {})
            base_features.update(env_features)

            # Check for context manager and apply overrides if available
            if context_id is not None and self.context_manager is not None:
                try:
                    if context := self.context_manager.get_context(context_id):
                        if context_features := context.get('features', {}).get(component, {}):
                            base_features.update(context_features)
                except AttributeError:
                    self.logger.warning(
                        f"Unable to get context features for component {component}: "
                        "Context manager get_context method not available"
                    )

            # Apply feature dependencies
            features = self._resolve_feature_dependencies(base_features)

            # Cache if no context-specific features
            if not context_id:
                self._config_cache[cache_key] = features

            return features

        except Exception as e:
            self.logger.error(f"Error getting component features: {str(e)}")
            return {}

    def update_runtime_config(
        self,
        updates: Dict[str, Any],
        validate: bool = True
    ) -> bool:
        """Update runtime configuration with validation."""
        try:
            # Store previous state for rollback
            previous_config = self._runtime_config.copy()

            if validate and not self.validate_config(updates):
                raise ValueError("Invalid configuration updates")

            try:
                # Apply updates
                for key, value in updates.items():
                    if isinstance(value, dict) and key in self._runtime_config:
                        self._runtime_config[key] = self._deep_merge(
                            self._runtime_config[key],
                            value
                        )
                    else:
                        self._runtime_config[key] = value

                # Invalidate affected caches
                self._invalidate_affected_caches(list(updates.keys()))  # Convert keys to list

                # Update state
                self._state.last_updated = datetime.now()

                return True

            except Exception:
                # Rollback on error
                self._runtime_config = previous_config
                raise

        except Exception as e:
            self.logger.error(f"Failed to update runtime config: {str(e)}")
            return False

    def _get_pipeline_features(self, pipeline_type: PipelineType) -> Dict[str, bool]:
        """Get features for a pipeline type."""
        features = {}

        # Get base features
        base_features = self._env_config.get('pipelines', {}).get(
            pipeline_type.value, {}
        ).get('features', {})
        features.update(base_features)

        # Get runtime overrides
        runtime_features = self._runtime_config.get('pipelines', {}).get(
            pipeline_type.value, {}
        ).get('features', {})
        features.update(runtime_features)

        return features

    def _get_pipeline_processors(self, pipeline_type: PipelineType) -> List[str]:
        """Get processors for a pipeline type."""
        return (
            self._env_config.get('pipelines', {})
            .get(pipeline_type.value, {})
            .get('processors', [])
        )

    def _get_pipeline_validators(self, pipeline_type: PipelineType) -> List[str]:
        """Get validators for a pipeline type."""
        return (
            self._env_config.get('pipelines', {})
            .get(pipeline_type.value, {})
            .get('validators', [])
        )

    def _get_pipeline_transformers(self, pipeline_type: PipelineType) -> List[str]:
        """Get transformers for a pipeline type."""
        return (
            self._env_config.get('pipelines', {})
            .get(pipeline_type.value, {})
            .get('transformers', [])
        )

    def _apply_context_overrides(
        self,
        pipeline_config: PipelineConfig,
        context: Dict[str, Any]
    ) -> PipelineConfig:
        """Apply context-specific overrides to pipeline configuration."""
        # Get context-specific pipeline overrides
        context_config = context.get('pipeline_config', {}).get(
            pipeline_config.pipeline_type.value,
            {}
        )

        # Update configuration
        if context_config:
            pipeline_config.config = self._deep_merge(
                pipeline_config.config,
                context_config
            )

        # Update features
        context_features = context.get('pipeline_features', {}).get(
            pipeline_config.pipeline_type.value,
            {}
        )
        pipeline_config.features.update(context_features)

        return pipeline_config

    def _invalidate_affected_caches(self, updated_keys: List[str]) -> None:
        """Invalidate caches affected by configuration updates."""
        try:
            for key in updated_keys:
                # Invalidate pipeline caches
                if key.startswith('pipeline_'):
                    pattern = f"pipeline_{key.split('_')[1]}"
                    self.cache.invalidate_pattern(pattern)

                # Invalidate component caches
                elif key.startswith('component_'):
                    pattern = f"features_{key.split('_')[1]}"
                    self.cache.invalidate_pattern(pattern)

                # Invalidate feature caches
                elif key == 'features':
                    self.cache.invalidate_pattern('features_')

        except Exception as e:
            self.logger.error(f"Error invalidating caches: {str(e)}")

    def _resolve_feature_dependencies(self, features: Dict[str, bool]) -> Dict[str, bool]:
        """Resolve feature dependencies and conflicts."""
        resolved = features.copy()

        # Get feature definitions
        for name, state in features.items():
            if not state:
                continue

            feature = self._feature_registry.get(name)
            if not feature:
                continue

            # Enable required features
            for dep in feature.dependencies:
                resolved[dep] = True

            # Disable conflicting features
            for conflict in feature.conflicts:
                resolved[conflict] = False

        return resolved
