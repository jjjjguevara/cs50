from typing import Dict, List, Optional, Any, Union, TypeVar, Set, Type, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import json
import jsonschema
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

# Import handlers we'll integrate with
from app.dita.utils.id_handler import DITAIDHandler, IDType
from .event_manager import EventManager, EventType
from .utils.logger import DITALogger
from .utils.cache import ContentCache, CacheEntryType


if TYPE_CHECKING:
    from .metadata.metadata_manager import MetadataManager
    from .context_manager import ContextManager

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
    ValidationSeverity,
    ProcessingContext,
    ContentScope,
    ProcessingRuleType,
    MDElementType
)



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
        content_cache: ContentCache,
        event_manager: EventManager,
        id_handler: DITAIDHandler,
        env: Optional[str] = None,
        metadata_manager: Optional['MetadataManager'] = None,
        context_manager: Optional['ContextManager'] = None,
        config_path: Optional[Union[str, Path]] = None,
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
        self.metadata_manager = metadata_manager
        self.content_cache = content_cache
        self.id_handler = id_handler or DITAIDHandler()
        self.config_path = Path(config_path) if config_path else Path(__file__).parent / "configs"
        self.cache = ContentCache()

        # Load schema
        self._attribute_schema = self._load_attribute_schema()

        self._invalidation_depth = 0
        self._MAX_INVALIDATION_DEPTH = 10  # Configurable maximum depth

        # Cache for resolved configurations
        self._config_cache: Dict[str, Any] = {}

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

        # Load metadata-specific configurations
        self._metadata_config = self._load_metadata_config()
        self._key_resolution_config = self._load_key_resolution_config()
        self._context_validation_config = self._load_context_validation_config()

        # Initialize state tracking
        self._state = ConfigState(
            environment=self.environment,
            last_updated=datetime.now()
        )

        # Register for events
        self.event_manager.subscribe(
            EventType.STATE_CHANGE,
            self._handle_config_change
        )

        # Setup config paths
        self.config_path = Path(__file__).parent / "configs"
        if not self.config_path.exists():
            self.config_path.mkdir(parents=True)

        try:
            self._load_config_files()
        except Exception as e:
            self.logger.error(f"Error loading config files: {str(e)}")

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
            self._rule_registry = {}
            self._element_rule_index = {}

            valid_rule_count = 0
            skipped_rule_count = 0

            for rule_type, type_rules in self._dita_rules.items():
                try:
                    rule_type_enum = ProcessingRuleType(rule_type)
                except ValueError:
                    self.logger.error(f"Invalid ProcessingRuleType: {rule_type}")
                    skipped_rule_count += 1
                    continue

                if not isinstance(type_rules, dict):
                    self.logger.warning(f"Invalid structure for rule_type: {rule_type}. Expected a dictionary.")
                    skipped_rule_count += 1
                    continue

                for rule_id, rule_data in type_rules.items():
                    if not isinstance(rule_data, dict):
                        self.logger.warning(f"Invalid rule format for {rule_id}. Skipping rule.")
                        skipped_rule_count += 1
                        continue

                    # Detect circular references
                    if self._has_circular_reference(rule_data):
                        self.logger.error(f"Circular reference detected in rule {rule_id}. Skipping rule.")
                        skipped_rule_count += 1
                        continue

                    # Preprocess and register the rule
                    try:
                        rule_data = self._preprocess_rule(rule_id, rule_data)
                        element_type_enum = ElementType(rule_data["element_type"])
                        self.register_processing_rule(
                            rule_type=rule_type_enum,
                            element_type=element_type_enum,
                            rule_config=rule_data["action"],
                            conditions=rule_data.get("conditions", {}),
                            priority=rule_data.get("priority", 0),
                            metadata=rule_data.get("metadata", {})
                        )
                        valid_rule_count += 1
                    except ValueError as e:
                        self.logger.error(f"Error processing rule {rule_id}: {str(e)}")
                        skipped_rule_count += 1
                        self.logger.error(
                            f"Error processing rule {rule_id}: '{rule_data['element_type']}' is not a valid ElementType. "
                            f"Available types: {[e.value for e in ElementType]}"
                        )
                        raise
                    except Exception as e:
                        self.logger.error(f"Unexpected error for rule {rule_id}: {str(e)}")
                        skipped_rule_count += 1

            self.logger.info(f"Rule registry initialization completed. "
                             f"Valid rules: {valid_rule_count}, Skipped rules: {skipped_rule_count}")
        except Exception as e:
            self.logger.error(f"Rule registry initialization failed: {str(e)}")
            raise

    def _has_circular_reference(self, obj, visited=None) -> bool:
        """Detect circular references in nested structures."""
        if visited is None:
            visited = set()
        if id(obj) in visited:
            return True
        visited.add(id(obj))
        if isinstance(obj, dict):
            return any(self._has_circular_reference(value, visited) for value in obj.values())
        elif isinstance(obj, list):
            return any(self._has_circular_reference(item, visited) for item in obj)
        visited.remove(id(obj))
        return False

    def _preprocess_rule(self, rule_id: str, rule_data: dict) -> dict:
        """Preprocess a single rule to ensure completeness."""
        try:
            rule_data.setdefault("operation", "transform")
            rule_data.setdefault("target", rule_data.get("html_tag", "div"))
            rule_data.setdefault("action", rule_data)
            if "element_type" not in rule_data:
                inferred_type = rule_id.split("_")[-1]  # Infer element_type from rule_id
                rule_data["element_type"] = inferred_type
            return rule_data
        except Exception as e:
            self.logger.error(f"Error preprocessing rule {rule_id}: {str(e)}")
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

    def _load_key_resolution_config(self) -> Dict[str, Any]:
        """Load key resolution configuration."""
        try:
            keyref_path = self.config_path / "keyref_config.json"
            key_resolution_path = self.config_path / "key_resolution.json"

            # Try loading from keyref_config.json first
            if keyref_path.exists():
                with open(keyref_path) as f:
                    config = json.load(f)
                    if "keyref_resolution" in config:
                        return {
                            "resolution_rules": config["keyref_resolution"],
                            "processing_hierarchy": config.get("processing_hierarchy", {}),
                            "global_defaults": config.get("global_defaults", {}),
                            "element_defaults": config.get("element_defaults", {})
                        }

            # Fall back to key_resolution.json
            if key_resolution_path.exists():
                with open(key_resolution_path) as f:
                    config = json.load(f)
                    if "resolution_rules" not in config:
                        raise ValueError("Missing resolution_rules in key resolution config")
                    return config

            raise FileNotFoundError("No valid key resolution config found")

        except Exception as e:
            self.logger.error(f"Error loading key resolution config: {str(e)}")
            return {
                "resolution_rules": {
                    "scopes": ["local", "peer", "external"],
                    "fallback_order": ["local", "peer", "external"]
                },
                "processing_hierarchy": {
                    "order": ["map", "topic", "element"]
                },
                "global_defaults": {},
                "element_defaults": {}
            }

    def _load_context_validation_config(self) -> Dict[str, Any]:
        """
        Load and validate context validation configuration with schema-based resolution.
        Implements fallback-override system for validation rules.
        """
        try:
            config_path = self.config_path / "context_validation.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Context validation config not found: {config_path}")

            with open(config_path) as f:
                config = json.load(f)

            # Load base schema for validation
            schema = self._attribute_schema.get("context_rules", {})
            validation_rules = schema.get("validation", {})

            # Resolve validation rules through hierarchy
            if "validation_rules" not in config:
                resolved_rules = self._resolve_validation_rules(
                    config.get("context", {}),
                    validation_rules
                )
                config["validation_rules"] = resolved_rules

            # Validate required sections and structure
            self._validate_context_config(config)

            # Apply scope inheritance rules
            config = self._apply_scope_inheritance(config)

            # Merge with attribute schema rules
            if schema_rules := schema.get("attribute_inheritance"):
                config["validation_rules"]["attribute_inheritance"] = self._deep_merge(
                    schema_rules,
                    config["validation_rules"].get("attribute_inheritance", {})
                )

            return config

        except Exception as e:
            self.logger.error(f"Error loading context validation config: {str(e)}")
            return self._get_default_context_config()

    def _resolve_validation_rules(
        self,
        context_config: Dict[str, Any],
        schema_rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve validation rules through schema hierarchy."""
        resolved = {
            "context": {
                "required_fields": schema_rules.get("required_fields", []),
                "scope_validation": {
                    "enabled": True,
                    "rules": {}
                }
            },
            "metadata": {
                "transient": schema_rules.get("transient", {}),
                "persistent": schema_rules.get("persistent", {})
            }
        }

        # Apply scope rules from schema
        if scope_rules := schema_rules.get("scopes"):
            for scope, rules in scope_rules.items():
                resolved["context"]["scope_validation"]["rules"][scope] = {
                    "allowed_references": rules.get("allows_external_refs", []),
                    "metadata_inheritance": rules.get("requires_validation", True)
                }

        # Override with context-specific rules
        if context_scopes := context_config.get("scopes"):
            for scope, rules in context_scopes.items():
                if scope in resolved["context"]["scope_validation"]["rules"]:
                    resolved["context"]["scope_validation"]["rules"][scope].update(rules)

        return resolved

    def _validate_context_config(self, config: Dict[str, Any]) -> None:
        """Validate context configuration structure."""
        required_sections = {
            "validation_rules": {
                "context": ["required_fields", "scope_validation"],
                "metadata": ["transient", "persistent"]
            }
        }

        def validate_section(data: Dict[str, Any], requirements: Dict[str, Any], path: str = "") -> None:
            for key, value in requirements.items():
                current_path = f"{path}.{key}" if path else key
                if key not in data:
                    raise ValueError(f"Missing required section: {current_path}")
                if isinstance(value, list):
                    missing = [field for field in value if field not in data[key]]
                    if missing:
                        raise ValueError(f"Missing required fields in {current_path}: {missing}")
                elif isinstance(value, dict):
                    validate_section(data[key], value, current_path)

        validate_section(config, required_sections)

    def _apply_scope_inheritance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply scope inheritance rules to configuration."""
        rules = config["validation_rules"]
        if "context" not in rules or "scope_validation" not in rules["context"]:
            return config

        scope_rules = rules["context"]["scope_validation"]["rules"]
        inheritance_order = ["local", "peer", "external", "global"]

        # Apply inheritance
        for i, scope in enumerate(inheritance_order):
            if scope not in scope_rules:
                continue
            # Inherit from previous scope if exists
            if i > 0 and inheritance_order[i-1] in scope_rules:
                parent_scope = scope_rules[inheritance_order[i-1]]
                scope_rules[scope] = self._deep_merge(
                    parent_scope.copy(),
                    scope_rules[scope]
                )

        return config

    def _get_default_context_config(self) -> Dict[str, Any]:
        """Get default context validation configuration."""
        return {
            "validation_rules": {
                "context": {
                    "required_fields": ["element_id", "element_type"],
                    "scope_validation": {
                        "enabled": True,
                        "rules": {
                            "local": {
                                "allowed_references": ["local", "peer"],
                                "metadata_inheritance": True
                            },
                            "peer": {
                                "allowed_references": ["local", "peer", "external"],
                                "metadata_inheritance": False
                            },
                            "external": {
                                "allowed_references": ["external"],
                                "metadata_inheritance": False
                            }
                        }
                    }
                },
                "metadata": {
                    "transient": {
                        "allowed_scopes": ["local", "phase"],
                        "max_lifetime": 3600
                    },
                    "persistent": {
                        "required_validation": True,
                        "schema_validation": True
                    }
                }
            }
        }

    def _validate_metadata_config(self, config: Dict[str, Any]) -> None:
        """Validate metadata configuration structure."""
        try:
            # Check top-level structure
            if "metadata_processing" not in config:
                raise ValueError("Missing metadata_processing section")

            processing = config["metadata_processing"]
            if "phases" not in processing:
                raise ValueError("Missing phases section in metadata_processing")

            # Validate phase configurations
            required_phases = {"discovery", "transformation"}
            phases = processing["phases"]
            if missing := required_phases - set(phases):
                raise ValueError(f"Missing required phases: {missing}")

            # Validate phase content
            for phase in required_phases:
                phase_config = phases[phase]
                if phase == "discovery":
                    if "extractors" not in phase_config:
                        raise ValueError("Missing extractors in discovery phase")
                    if "validation_rules" not in phase_config:
                        raise ValueError("Missing validation_rules in discovery phase")

                elif phase == "transformation":
                    if "rules" not in phase_config:
                        raise ValueError("Missing rules in transformation phase")

        except Exception as e:
            self.logger.error(f"Error validating metadata config: {str(e)}")
            raise ValueError(f"Invalid metadata configuration: {str(e)}")

    def _get_context_rules(
        self,
        context: ProcessingContext,
        phase: ProcessingPhase
    ) -> Dict[str, Any]:
        """Get context-specific rules."""
        try:
            # Get base context rules
            context_rules = self._context_validation_config.get("validation_rules", {})
            context_specific = context_rules.get("context", {})

            # Apply phase-specific rules
            phase_rules = context_specific.get("phases", {}).get(phase.value, {})

            # Apply scope-specific rules
            scope_rules = context_specific.get("scope_validation", {}).get(
                "rules", {}
            ).get(context.scope.value, {})

            # Merge rules
            return {
                **phase_rules,
                **scope_rules,
                "metadata_inheritance": scope_rules.get("metadata_inheritance", True)
            }

        except Exception as e:
            self.logger.error(f"Error getting context rules: {str(e)}")
            return {}

    def _get_context_key_rules(self, context: ProcessingContext) -> Dict[str, Any]:
        """Get context-specific key resolution rules."""
        try:
            # Get base key resolution rules
            key_rules = self._key_resolution_config.get("resolution_rules", {})

            # Get scope-specific rules
            scope_rules = key_rules.get("scope_rules", {}).get(
                context.scope.value, {}
            )

            # Apply context-specific overrides
            context_refs = context.metadata_state.metadata_refs
            if isinstance(context_refs, dict):  # Type check
                if key_resolution := context_refs.get("key_resolution"):
                    # Ensure key_resolution is a dictionary
                    if isinstance(key_resolution, dict):
                        scope_rules = self._merge_rules(scope_rules, key_resolution)
                    else:
                        self.logger.warning(
                            f"Invalid key_resolution format in context {context.context_id}: "
                            f"expected dict, got {type(key_resolution)}"
                        )

            return scope_rules

        except Exception as e:
            self.logger.error(f"Error getting context key rules: {str(e)}")
            return {}


    #################################
    # Core configuration management #
    #################################

    def _load_metadata_config(self) -> Dict[str, Any]:
            """Load metadata processing configuration."""
            try:
                config_path = self.config_path / "metadata_schema.json"
                if not config_path.exists():
                    raise FileNotFoundError(f"Metadata config not found: {config_path}")

                with open(config_path) as f:
                    config = json.load(f)

                # Validate configuration
                self._validate_metadata_config(config)
                return config

            except Exception as e:
                self.logger.error(f"Error loading metadata config: {str(e)}")
                return {}

    def get_metadata_rules(
        self,
        phase: ProcessingPhase,
        element_type: ElementType,
        context: Optional[ProcessingContext] = None
    ) -> Dict[str, Any]:
        """
        Get metadata processing rules for phase and element type.

        Args:
            phase: Current processing phase
            element_type: Element type
            context: Optional processing context

        Returns:
            Dict containing applicable rules
        """
        try:
            # Get base rules
            base_rules = self._metadata_config.get("metadata_processing", {})
            phase_rules = base_rules.get("phases", {}).get(phase.value, {})

            # Get element-specific rules
            element_rules = {}
            if element_type == ElementType.DITA:
                element_rules = phase_rules.get("extractors", {}).get("dita", {})
            elif element_type == ElementType.MARKDOWN:
                element_rules = phase_rules.get("extractors", {}).get("markdown", {})

            # Apply context overrides if provided
            if context:
                context_rules = self._get_context_rules(context, phase)
                element_rules = self._merge_rules(element_rules, context_rules)

            return element_rules

        except Exception as e:
            self.logger.error(f"Error getting metadata rules: {str(e)}")
            return {}

    def get_key_resolution_rules(
        self,
        scope: ContentScope,
        context: Optional[ProcessingContext] = None
    ) -> Dict[str, Any]:
        """Get key resolution rules for scope."""
        try:
            # Get base rules
            base_rules = self._key_resolution_config.get("resolution_rules", {})
            scope_rules = base_rules.get("scope_rules", {}).get(scope.value, {})

            # Apply context if provided
            if context:
                context_rules = self._get_context_key_rules(context)
                scope_rules = self._merge_rules(scope_rules, context_rules)

            return scope_rules

        except Exception as e:
            self.logger.error(f"Error getting key resolution rules: {str(e)}")
            return {}

    def validate_metadata_schema(
        self,
        metadata: Dict[str, Any],
        element_type: ElementType,
        phase: ProcessingPhase
    ) -> ValidationResult:
        """Validate metadata against schema."""
        try:
            # Get validation rules
            rules = self._metadata_config.get("metadata_processing", {})
            phase_rules = rules.get("phases", {}).get(phase.value, {})
            validation_rules = phase_rules.get("validation_rules", {})

            messages = []

            # Validate required fields
            if required_fields := validation_rules.get("required_fields", []):
                for field in required_fields:
                    if field not in metadata:
                        messages.append(
                            ValidationMessage(
                                path=field,
                                message=f"Missing required field: {field}",
                                severity=ValidationSeverity.ERROR,
                                code="missing_required_field"
                            )
                        )

            # Validate field types
            for field, value in metadata.items():
                if field_rules := validation_rules.get(field):
                    if not self._validate_field(value, field_rules):
                        messages.append(
                            ValidationMessage(
                                path=field,
                                message=f"Invalid value for field: {field}",
                                severity=ValidationSeverity.ERROR,
                                code="invalid_field_value"
                            )
                        )

            return ValidationResult(
                is_valid=len(messages) == 0,
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating metadata schema: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[
                    ValidationMessage(
                        path="",
                        message=str(e),
                        severity=ValidationSeverity.ERROR,
                        code="validation_error"
                    )
                ]
            )

    def _validate_field(self, value: Any, rules: Dict[str, Any]) -> bool:
        """Validate field value against rules."""
        try:
            field_type = rules.get("type")

            if field_type == "string":
                if not isinstance(value, str):
                    return False
                if min_length := rules.get("min_length"):
                    if len(value) < min_length:
                        return False
                if allowed := rules.get("allowed"):
                    if value not in allowed:
                        return False

            elif field_type == "list":
                if not isinstance(value, list):
                    return False
                if min_items := rules.get("min_items"):
                    if len(value) < min_items:
                        return False

            # Add more type validations as needed

            return True

        except Exception as e:
            self.logger.error(f"Error validating field: {str(e)}")
            return False

    def _merge_rules(
        self,
        base_rules: Dict[str, Any],
        override_rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge rule sets with proper inheritance."""
        try:
            merged = base_rules.copy()

            for key, value in override_rules.items():
                if key in merged and isinstance(merged[key], dict):
                    if isinstance(value, dict):
                        merged[key] = self._merge_rules(merged[key], value)
                    else:
                        merged[key] = value
                else:
                    merged[key] = value

            return merged

        except Exception as e:
            self.logger.error(f"Error merging rules: {str(e)}")
            return base_rules


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
        """Load configuration files from the config directory."""
        try:
            # Load feature flags
            with open(self.config_path / "feature_flags.json", "r") as f:
                self._feature_registry = json.load(f)["features"]

            # Load processing rules and validate structure
            with open(self.config_path / "processing_rules.json", "r") as f:
                processing_rules = json.load(f)

            # Define schema for validation
            schema = {
                "type": "object",
                "properties": {
                    "version": {"type": "string"},
                    "description": {"type": "string"},
                    "rules": {
                        "type": "object",
                        "patternProperties": {
                            ".*": {
                                "type": "object",
                                "patternProperties": {
                                    ".*": {
                                        "type": "object",
                                        "properties": {
                                            "element_type": {"type": "string"},
                                            "operation": {"type": "string"},
                                            "target": {"type": "string"},
                                            "action": {"type": "object"},
                                            "conditions": {"type": "object"},
                                            "priority": {"type": "integer"},
                                            "metadata": {"type": "object"},
                                        },
                                        "required": ["element_type", "operation", "target", "action"],
                                        "additionalProperties": False
                                    }
                                }
                            }
                        }
                    },
                    "defaults": {"type": "object"}
                },
                "required": ["version", "description", "rules"],
                "additionalProperties": False
            }

            jsonschema.validate(instance=processing_rules, schema=schema)
            self._rule_registry = processing_rules["rules"]

            # Load DITA processing rules
            with open(self.config_path / "dita_processing_rules.json", "r") as f:
                dita_config = json.load(f)
                self._dita_rules = dita_config["element_rules"]
                self._dita_type_mapping = dita_config["element_type_mapping"]

            # Validate all rules in _rule_registry
            for rule_type, rules in self._rule_registry.items():
                for rule_id, rule_data in rules.items():
                    # Ensure the rule is passed as a dictionary
                    if isinstance(rule_data, ProcessingRule):
                        rule_dict = {
                            "operation": rule_data.config.get("operation"),
                            "target": rule_data.config.get("target"),
                            "action": rule_data.config.get("action"),
                            "conditions": rule_data.conditions,
                            "metadata": rule_data.metadata,
                        }
                    elif isinstance(rule_data, dict):
                        rule_dict = rule_data
                    else:
                        self.logger.error(f"Invalid rule format for rule_id: {rule_id}")
                        continue

                    if not self.validate_processing_rule(rule_dict):
                        self.logger.error(f"Invalid rule detected: {rule_id}")

            # Cross-check DITA type mappings
            for element_type, rule_path in self._dita_type_mapping.items():
                resolved_rule = self.get_dita_element_rules(element_type)
                if resolved_rule.get("html_tag") is None:
                    self.logger.error(f"DITA element {element_type} mapping is invalid or unresolved: {rule_path}")

            self.logger.info("Successfully loaded and validated configuration files")
            self.logger.debug(f"Feature registry: {json.dumps(self._feature_registry, indent=2)}")
            self.logger.debug(f"Rule registry: {json.dumps(self._rule_registry, indent=2)}")
            self.logger.debug(f"DITA rules: {json.dumps(self._dita_rules, indent=2)}")
            self.logger.debug(f"DITA type mapping: {json.dumps(self._dita_type_mapping, indent=2)}")


        except jsonschema.ValidationError as ve:
            self.logger.error(f"Validation error in processing_rules.json: {ve.message}")
            raise ValueError(f"Invalid structure in processing_rules.json: {ve.message}")
        except FileNotFoundError as e:
            self.logger.error(f"Configuration file not found: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in configuration file: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading configuration files: {str(e)}")
            raise

    def get_dita_element_rules(self, element_type: str) -> Dict[str, Any]:
        """
        Get processing rules for a DITA element type.

        Args:
            element_type: The DITA element type as a string.

        Returns:
            Dict[str, Any]: Resolved processing rules for the element type.
        """
        try:
            # Get rule path from type mapping
            rule_path = self._dita_type_mapping.get(element_type)
            if not rule_path:
                self.logger.warning(f"DITA element type {element_type} has no rule path in type mapping.")
                return self._dita_rules["default"]["unknown"]

            # Navigate to rule in hierarchy
            current = self._dita_rules
            for part in rule_path.split("."):
                current = current.get(part)
                if current is None:
                    self.logger.warning(f"DITA element type {element_type} has an invalid rule path: {rule_path}")
                    return self._dita_rules["default"]["unknown"]

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

            component_config = {}
            for section in ["rules", "features", "metadata"]:
                section_path = self.config_path / f"{component}_{section}.json"
                if section_path.exists():
                    with open(section_path) as f:
                        component_config[section] = json.load(f)

            # Validate against schema
            if not self.validate_config(component_config, ElementType.UNKNOWN):
                raise ValueError(f"Invalid configuration for component: {component}")

            # Cache the result
            self._config_cache[cache_key] = component_config

            self.logger.debug(f"Loaded configuration for component: {component}")
            return component_config

        except Exception as e:
            self.logger.error(f"Failed to load component config {component}: {str(e)}")
            return {}

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
        try:
            base_config_path = self.config_path / "base.yml"
            if not base_config_path.exists():
                return {}

            with base_config_path.open('r') as f:
                return yaml.safe_load(f) or {}

        except Exception as e:
            self.logger.error(f"Error loading base config: {str(e)}")
            return {}


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
                if cached := self.content_cache.get(
                    cache_key,
                    entry_type=CacheEntryType.CONTENT
                ):
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
                        entry_type=CacheEntryType.CONTENT,
                        element_type=element_type,
                        phase=ProcessingPhase.DISCOVERY
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

    def get_effective_features(
        self,
        context: ProcessingContext
    ) -> Dict[str, bool]:
        """
        Get effective feature states for context.

        Args:
            context: Processing context

        Returns:
            Dict[str, bool]: Effective feature states
        """
        try:
            context_dict = context.to_dict()
            effective_features = {}

            for feature_name, feature_def in self._attribute_schema["feature_definitions"].items():
                # Get base state
                base_state = feature_def.get("default", False)

                # Apply context overrides
                if feature_overrides := context_dict.get("features", {}):
                    if feature_name in feature_overrides:
                        base_state = feature_overrides[feature_name]

                # Store effective state
                effective_features[feature_name] = base_state

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
                id_type=IDType.REFERENCE  # or we could add a RULE type to IDType
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
        context: Optional[ProcessingContext] = None
    ) -> Dict[str, Any]:
        """
        Get processing rules for element type and context.

        Args:
            element_type: Element type to get rules for
            context: Optional processing context

        Returns:
            Dict[str, Any]: Processing rules
        """
        try:
            # Get base rules
            base_rules = self._attribute_schema["processing"].get(element_type.value, {})

            # Apply context rules if provided
            if context:
                context_dict = context.to_dict()
                if context_rules := context_dict.get("processing_rules", {}):
                    base_rules = self._deep_merge(base_rules, context_rules)

            return base_rules

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
                if isinstance(rule, ProcessingRule):
                    rule = {
                        "operation": rule.config.get("operation"),
                        "target": rule.config.get("target"),
                        "action": rule.config.get("action"),
                        "conditions": rule.conditions,
                        "metadata": rule.metadata,
                    }

                # Required rule components
                required_fields = {
                    "operation": str,
                    "target": str,
                    "action": dict,
                }

                for field, expected_type in required_fields.items():
                    if field not in rule:
                        self.logger.error(f"Missing required field: {field}")
                        return False
                    if not isinstance(rule[field], expected_type):
                        self.logger.error(f"Invalid type for {field}: expected {expected_type}")
                        return False

                # Validate operation
                valid_operations = {"transform", "validate", "enrich", "extract", "inject", "specialize"}
                operation = rule.get("operation", None)
                if operation not in valid_operations:
                    self.logger.error(f"Invalid operation: {operation}")
                    return False

                # Validate action structure
                action = rule.get("action", {})
                if not isinstance(action, dict):
                    self.logger.error("Action must be a dictionary")
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
        context: ProcessingContext
    ) -> bool:
        """
        Evaluate rule conditions against context.

        Args:
            conditions: Conditions to evaluate
            context: Processing context to evaluate against

        Returns:
            bool: True if conditions are satisfied
        """
        try:
            context_dict = context.to_dict()

            for key, value in conditions.items():
                # Handle nested conditions
                if isinstance(value, dict):
                    nested_context = context_dict.get(key, {})
                    if not self._evaluate_rule_conditions(value, nested_context):
                        return False
                    continue

                # Handle list conditions (any match)
                if isinstance(value, list):
                    context_value = context_dict.get(key)
                    if context_value is None or context_value not in value:
                        return False
                    continue

                # Handle simple equality
                if context_dict.get(key) != value:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error evaluating rule conditions: {str(e)}")
            return False



    ####################################
    # Validation and schema management #
    ####################################

    def _handle_config_change(self, **event_data: Any) -> None:
        """
        Handle configuration change events.

        Args:
            event_data: Event information
        """
        try:
            if element_id := event_data.get("element_id"):
                # Clear cached configs for element
                cache_key = f"config_{element_id}"
                self._config_cache.pop(cache_key, None)

                # Clear configs that include this element
                invalidation_patterns = [
                    f".*{element_id}.*",  # Any config containing this element (regex match)
                    r"component_.*",      # Component configs might need update
                    r"pipeline_.*"        # Pipeline configs might need update
                ]

                for pattern in invalidation_patterns:
                    self.content_cache.invalidate_by_pattern(pattern)

        except Exception as e:
            self.logger.error(f"Error handling config change: {str(e)}")

    def _load_attribute_schema(self) -> Dict[str, Any]:
        """Load and validate attribute schema."""
        try:
            schema_path = self.config_path / "attribute_schema.json"
            if not schema_path.exists():
                self.logger.error(f"Schema not found: {schema_path}")
                return {}

            with open(schema_path, 'r') as f:
                schema = json.load(f)

            # Map legacy field names to new structure if necessary
            if 'resolution_rules' in schema:
                schema['inheritance_rules'] = schema.get('resolution_rules', {})

            if 'validation' in schema:
                schema['validation_rules'] = schema.get('validation', {})

            # Check feature definitions in both old and new locations
            if not schema.get('feature_definitions'):
                features_path = self.config_path / "feature_flags.json"
                if features_path.exists():
                    with open(features_path, 'r') as f:
                        feature_data = json.load(f)
                        schema['feature_definitions'] = feature_data.get('features', {})

            # Validate required sections
            required_sections = {
                "hierarchy",
                "inheritance_rules",
                "validation_rules",
                "feature_definitions"
            }

            missing = required_sections - set(schema.keys())
            if missing:
                self.logger.error(f"Missing required schema sections: {missing}")
                return {}

            return schema

        except Exception as e:
            self.logger.error(f"Error loading attribute schema: {str(e)}")
            return {}

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
        element_type: Optional[ElementType] = None
    ) -> bool:
        """
        Validate configuration against schema rules.

        Args:
            config: Configuration to validate
            element_type: Optional element type for specific validation

        Returns:
            bool: True if configuration is valid
        """
        try:
            validation_rules = self._attribute_schema["validation"]

            # Check required attributes if element type provided
            if element_type and (required := validation_rules["required_attributes"].get(element_type.value)):
                missing = [attr for attr in required if attr not in config]
                if missing:
                    self.logger.warning(f"Missing required attributes: {missing}")
                    return False

            # Validate patterns
            for attr, value in config.items():
                if pattern := validation_rules["attribute_patterns"].get(attr):
                    if not re.match(pattern, str(value)):
                        self.logger.warning(f"Invalid pattern for {attr}: {value}")
                        return False

            # Validate allowed values
            for attr, value in config.items():
                if allowed := validation_rules["allowed_values"].get(attr):
                    if value not in allowed:
                        self.logger.warning(f"Invalid value for {attr}: {value}")
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
                validated_data = schema.model.parse_obj(data)
                result.is_valid = True

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
        pipeline_type: PipelineType,
        context: Optional[ProcessingContext] = None
    ) -> PipelineConfig:
        """
        Get pipeline configuration with context overrides.

        Args:
            pipeline_type: Type of pipeline
            context: Optional processing context

        Returns:
            PipelineConfig: Pipeline configuration
        """
        try:
            # Get base config
            base_config = self._attribute_schema["pipelines"][pipeline_type.value]

            # Create pipeline config with all required fields
            pipeline_config = PipelineConfig(
                pipeline_type=pipeline_type,
                config=base_config.get("config", {}),
                features=base_config.get("features", {}),
                processors=base_config.get("processors", []),
                validators=base_config.get("validators", []),
                transformers=base_config.get("transformers", []),
                metadata=base_config.get("metadata", {})
            )

            # Apply context overrides if provided
            if context:
                pipeline_config = self._apply_context_overrides(pipeline_config, context)

            return pipeline_config

        except Exception as e:
            self.logger.error(f"Error getting pipeline config: {str(e)}")
            # Return a default config with empty collections
            return PipelineConfig(
                pipeline_type=pipeline_type,
                config={},
                features={},
                processors=[],
                validators=[],
                transformers=[]
            )

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



    def _apply_context_overrides(
        self,
        pipeline_config: PipelineConfig,
        context: ProcessingContext
    ) -> PipelineConfig:
        """
        Apply context-specific overrides to pipeline configuration.

        Args:
            pipeline_config: Base pipeline configuration
            context: Processing context with overrides

        Returns:
            PipelineConfig: Updated pipeline configuration
        """
        try:
            # Get context as dictionary
            context_dict = context.to_dict()

            # Get pipeline-specific overrides
            if pipeline_overrides := context_dict.get('pipeline_config', {}).get(
                pipeline_config.pipeline_type.value
            ):
                pipeline_config.config = self._deep_merge(
                    pipeline_config.config,
                    pipeline_overrides
                )

            # Get feature overrides
            if feature_overrides := context_dict.get('pipeline_features', {}).get(
                pipeline_config.pipeline_type.value
            ):
                pipeline_config.features.update(feature_overrides)

            return pipeline_config

        except Exception as e:
            self.logger.error(f"Error applying context overrides: {str(e)}")
            return pipeline_config

    def _invalidate_affected_caches(self, updated_keys: List[str]) -> None:
        """Invalidate caches affected by configuration updates."""
        try:
            if self._invalidation_depth >= self._MAX_INVALIDATION_DEPTH:
                self.logger.warning("Maximum invalidation depth reached, stopping cascade")
                return

            self._invalidation_depth += 1
            try:
                # Existing invalidation logic here
                for key in updated_keys:
                    if key.startswith('pipeline_'):
                        pattern = f"pipeline_{key.split('_')[1]}"
                        self.cache.invalidate_by_pattern(pattern)
                    # ... rest of the logic
            finally:
                self._invalidation_depth -= 1

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
