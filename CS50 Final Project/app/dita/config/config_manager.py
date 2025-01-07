"""Central facade for DITA configuration management."""
from typing import Dict, Optional, Any, Union, TypeVar, Set, Type
from dataclasses import dataclass, field
from pathlib import Path
import os
from datetime import datetime
import logging
from functools import lru_cache

from ..events.event_manager import EventManager, EventType
from ..utils.id_handler import DITAIDHandler, IDType
from ..cache.cache import ContentCache, CacheEntryType
from ..utils.logger import DITALogger
from ..validation.validation_manager import ValidationManager
from ..schema.schema_manager import SchemaManager, CompositionStrategy

# Import configuration components
from .loaders.config_loader import ConfigLoader
from .features.feature_manager import FeatureManager
from .rules.rule_resolver import RuleResolver

# DTD imports
from ..dtd.dtd_mapper import DTDSchemaMapper
from ..dtd.dtd_validator import DTDValidator
from ..dtd.dtd_resolver import DTDResolver

# Import core types
from ..models.types import (
    ProcessingPhase,
    ProcessingState,
    ElementType,
    ValidationResult,
    ValidationMessage,
    ValidationSeverity,
    ContentScope,
    ProcessingContext,
    ProcessingRuleType,
    FeatureScope,
    Feature
)

class ConfigManager:
    """
    Configuration management facade.
    Coordinates loaders, feature management, and rule resolution.
    """

    def __init__(
        self,
        config_path: Union[str, Path],
        event_manager: EventManager,
        content_cache: ContentCache,
        id_handler: DITAIDHandler,
        validation_manager: ValidationManager,
        schema_manager: SchemaManager,
        logger: Optional[DITALogger] = None
    ):
        """Initialize configuration manager facade."""
        # Core dependencies
        self.config_path = Path(config_path)
        self.event_manager = event_manager
        self.cache = content_cache
        self.id_handler = id_handler
        self.validation_manager = validation_manager
        self.logger = logger or DITALogger(name=__name__)

        # Initialize schema composer from schema manager
        self.schema_manager = schema_manager
        self.schema_composer = self.schema_manager.schema_composer

        # Initialize components
        self._init_components()

        # Register for events
        self._register_event_handlers()

        # State tracking
        self._initialized = False
        self._environment = "development"

        # Initialize DTD components with proper paths
        self._init_dtd_components()

    def _init_components(self) -> None:
        """Initialize configuration system components."""
        try:
            # Initialize core components
            self.config_loader = ConfigLoader(
                config_path=self.config_path,
                validation_manager=self.validation_manager,
                schema_manager=self.schema_manager,
                content_cache=self.cache,
                event_manager=self.event_manager,
                logger=self.logger
            )

            self.feature_manager = FeatureManager(
                event_manager=self.event_manager,
                cache=self.cache,
                logger=self.logger
            )

            self.rule_resolver = RuleResolver(
                event_manager=self.event_manager,
                cache=self.cache,
                logger=self.logger
            )

        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise

    def _register_event_handlers(self) -> None:
        """Register event handlers."""
        self.event_manager.subscribe(
            EventType.STATE_CHANGE,
            self._handle_state_change
        )

    def _init_dtd_components(self) -> None:
        """Initialize DTD-related components."""
        try:
            # Get absolute paths, using config_path as reference
            # config_path is typically /app/dita/configs
            dita_root = self.config_path.parent  # Points to /app/dita
            dtd_base_path = dita_root / "dtd"    # Points to /app/dita/dtd
            dtd_schemas_path = dtd_base_path / "schemas"
            dtd_catalog_path = dtd_base_path / "catalog.xml"

            # Debug log the paths
            self.logger.debug(
                f"Initializing DTD components with paths:\n"
                f"  DITA root: {dita_root}\n"
                f"  DTD base: {dtd_base_path}\n"
                f"  DTD schemas: {dtd_schemas_path}\n"
                f"  DTD catalog: {dtd_catalog_path}"
            )

            # Ensure directories exist
            dtd_base_path.mkdir(parents=True, exist_ok=True)
            dtd_schemas_path.mkdir(parents=True, exist_ok=True)

            # Create catalog if it doesn't exist
            if not dtd_catalog_path.exists():
                self._create_default_catalog(dtd_catalog_path)

            # Initialize DTD resolver with correct base path
            self.dtd_resolver = DTDResolver(
                base_path=dtd_base_path,
                logger=self.logger
            )

            # Initialize other components
            self.dtd_mapper = DTDSchemaMapper(logger=self.logger)
            self.dtd_validator = DTDValidator(
                resolver=self.dtd_resolver,
                dtd_mapper=self.dtd_mapper,
                validation_manager=self.validation_manager,
                logger=self.logger
            )

        except Exception as e:
            self.logger.error(f"Error initializing DTD components: {str(e)}")
            raise

    def initialize(self, environment: str = "development") -> ValidationResult:
        """
        Initialize configuration system.

        Args:
            environment: Target environment

        Returns:
            ValidationResult: Initialization status
        """
        try:
            self._environment = environment

            # Load environment configuration
            validation_result = self.config_loader.load_environment_config(environment)
            if not validation_result.is_valid:
                return validation_result

            # Load core configuration files
            self._load_core_configs()

            self._initialized = True
            return ValidationResult(is_valid=True, messages=[])

        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="initialization",
                    message=str(e),
                    severity=ValidationSeverity.ERROR,
                    code="init_error"
                )]
            )

    def _load_core_configs(self) -> None:
        """Load core configuration files."""
        required_configs = [
            "attribute_schema.json",
            "dita_processing_rules.json",
            "feature_flags.json",
            "validation_patterns.json"
        ]

        for config_file in required_configs:
            self.config_loader.load_config_file(config_file, required=True)

    # Feature Management Methods
    def register_feature(
        self,
        name: str,
        scope: FeatureScope,
        default: bool = False,
        **kwargs: Any
    ) -> ValidationResult:
        """Register feature through feature manager."""
        return self.feature_manager.register_feature(
            name=name,
            scope=scope,
            default=default,
            **kwargs
        )

    def update_feature(
        self,
        name: str,
        enabled: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Update feature state."""
        return self.feature_manager.update_feature(
            name=name,
            enabled=enabled,
            metadata=metadata
        )

    def get_feature_state(
        self,
        name: str,
        context: Optional[ProcessingContext] = None
    ) -> bool:
        """Get feature state."""
        return self.feature_manager.get_feature_state(
            name=name,
            context=context
        )

    # Rule Resolution Methods
    def resolve_rule(
        self,
        element_type: ElementType,
        rule_type: ProcessingRuleType,
        context: Optional[ProcessingContext] = None
    ) -> Optional[Dict[str, Any]]:
        """Resolve processing rule."""
        resolved = self.rule_resolver.resolve_rule(
            element_type=element_type,
            rule_type=rule_type,
            context=context
        )

        # Convert ResolvedRule to dict if present
        if resolved:
            return {
                "rule_id": resolved.rule_id,
                "element_type": resolved.element_type.value,
                "rule_type": resolved.rule_type.value,
                "config": resolved.config,
                "conditions": resolved.conditions,
                "metadata": resolved.metadata,
                "source_rules": resolved.source_rules,
                "created_at": resolved.created_at
            }
        return None

    # Configuration Access Methods
    def get_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get loaded configuration."""
        return self.config_loader.get_config(name)

    def get_env_config(self) -> Dict[str, Any]:
        """Get environment configuration."""
        return self.config_loader.get_env_config()

    def reload_configs(self) -> ValidationResult:
        """Reload all configurations."""
        try:
            # Reload through config loader
            result = self.config_loader.reload_all()
            if result.is_valid:
                # Clear caches
                self.cache.invalidate_by_pattern("config_*")
                self.cache.invalidate_by_pattern("feature_*")
                self.cache.invalidate_by_pattern("rule_*")

            return result

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

    def _handle_state_change(self, **event_data: Any) -> None:
        """Handle state change events."""
        pass  # Implement if needed

    def cleanup(self) -> None:
        """Clean up manager and component resources."""
        try:
            # Clean up components
            self.config_loader.cleanup()
            self.feature_manager.cleanup()
            self.rule_resolver.cleanup()

            # Clear initialization state
            self._initialized = False

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise

    def get_processing_rules(
            self,
            element_type: ElementType,
            context: Optional[ProcessingContext] = None
        ) -> Dict[str, Any]:
            """Get processing rules with DTD awareness."""
            try:
                # Get base rules
                if base_rules := self.config_loader.get_config("processing_rules.json"):
                    base_rules = base_rules.get("rules", {})
                else:
                    base_rules = {}

                # Get DTD-specific rules if available
                dtd_rules = {}
                if context and context.element_id:
                    if spec_info := self.dtd_mapper.get_specialization_info(
                        context.element_id
                    ):
                        dtd_rules = spec_info.constraints or {}

                        # Convert DTD rules to match our processing rules format
                        dtd_rules = self._convert_dtd_rules(dtd_rules)

                # Merge rules with DTD taking precedence for overlapping keys
                merged_rules = {
                    **base_rules,
                    **dtd_rules
                }

                return merged_rules

            except Exception as e:
                self.logger.error(f"Error getting processing rules: {str(e)}")
                return {}

    def _convert_dtd_rules(self, dtd_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Convert DTD rules to processing rules format."""
        converted = {}
        try:
            for rule_type, rules in dtd_rules.items():
                if isinstance(rules, dict):
                    converted[rule_type] = {
                        "type": rule_type,
                        "rules": rules,
                        "validation": {
                            "required": rules.get("required", []),
                            "allowed_values": rules.get("allowed_values", {}),
                            "patterns": rules.get("patterns", {})
                        }
                    }
        except Exception as e:
            self.logger.error(f"Error converting DTD rules: {str(e)}")

        return converted


    def get_metadata_rules(
        self,
        phase: ProcessingPhase,
        element_type: ElementType,
        context: Optional[ProcessingContext] = None
    ) -> Dict[str, Any]:
        """Get metadata processing rules."""
        try:
            # Get the metadata config
            metadata_config = self.config_loader.get_config("metadata_schema.json")
            if not metadata_config:
                return {}

            # Get phase-specific rules
            phase_rules = metadata_config.get("metadata_processing", {}).get("phases", {}).get(phase.value, {})

            # Get element-specific extractors
            extractors = phase_rules.get("extractors", {})
            if element_type == ElementType.DITA:
                rules = dict(extractors.get("dita", {}))  # Ensure we have a dict
            elif element_type == ElementType.MARKDOWN:
                rules = dict(extractors.get("markdown", {}))  # Ensure we have a dict
            else:
                rules = {}

            # Apply context overrides if available
            if context and context.metadata_refs:
                context_rules = context.metadata_refs.get("rules")
                if isinstance(context_rules, dict):
                    rules.update(context_rules)

            return rules

        except Exception as e:
            self.logger.error(f"Error getting metadata rules: {str(e)}")
            return {}

    @property
    def keyref_config(self) -> Dict[str, Any]:
        """Get keyref configuration."""
        return self.config_loader.get_config("keyref_config.json") or {}

    def load_key_resolution_config(self) -> Dict[str, Any]:
        """
        Load key resolution configuration.

        Returns:
            Dict[str, Any]: Key resolution configuration including resolution rules,
            processing hierarchy, and defaults
        """
        try:
            # Try loading from keyref_config.json first
            keyref_config = self.get_config("keyref_config.json") or {}
            if "keyref_resolution" in keyref_config:
                return {
                    "resolution_rules": {
                        "scopes": keyref_config["keyref_resolution"].get("scopes", ["local", "peer", "external"]),
                        "fallback_order": keyref_config["keyref_resolution"].get("fallback_order", []),
                        "inheritance_rules": keyref_config["keyref_resolution"].get("inheritance_rules", {})
                    },
                    "processing_hierarchy": keyref_config.get("processing_hierarchy", {}),
                    "global_defaults": keyref_config.get("global_defaults", {}),
                    "element_defaults": keyref_config.get("element_defaults", {})
                }

            # Fall back to key_resolution.json
            key_resolution = self.get_config("key_resolution.json") or {}
            if "resolution_rules" in key_resolution:
                return {
                    "resolution_rules": key_resolution["resolution_rules"],
                    "processing_hierarchy": key_resolution.get("processing_hierarchy", {}),
                    "global_defaults": key_resolution.get("global_defaults", {}),
                    "element_defaults": key_resolution.get("element_defaults", {})
                }

            # Return default configuration if no valid config found
            return {
                "resolution_rules": {
                    "scopes": ["local", "peer", "external"],
                    "fallback_order": ["local", "peer", "external"],
                    "inheritance_rules": {
                        "props": "merge",
                        "outputclass": "append",
                        "other": "override"
                    }
                },
                "processing_hierarchy": {
                    "order": ["map", "topic", "element"]
                },
                "global_defaults": {},
                "element_defaults": {}
            }

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

    def load_dtd_config(self) -> Dict[str, Any]:
        """Load DTD-specific configuration."""
        try:
            # Try loading from dtd_config.json first
            dtd_config = self.config_loader.load_config_file("dtd_config.json")
            if dtd_config:
                return dtd_config

            # Fall back to validation_patterns.json DTD section
            validation_patterns = self.config_loader.load_config_file("validation_patterns.json")
            if validation_patterns:
                return validation_patterns.get("dtd_validation", {})

            # Return default configuration
            return {
                "validation": {
                    "enabled": True,
                    "mode": "strict",
                    "phases": ["discovery", "validation"],
                    "specialization": {
                        "enabled": True,
                        "inheritance_mode": "strict"
                    }
                },
                "caching": {
                    "enabled": True,
                    "ttl": 3600
                },
                "error_handling": {
                    "fail_fast": True,
                    "strict_validation": True
                }
            }

        except Exception as e:
            self.logger.error(f"Error loading DTD configuration: {str(e)}")
            return {}

    def get_dtd_validation_config(self) -> Dict[str, Any]:
        """Get DTD validation configuration with inheritance."""
        try:
            # Get base configuration
            base_config = self.load_dtd_config()

            # Get environment overrides
            env_config = self.get_env_config()
            dtd_overrides = env_config.get("dtd_validation", {})

            # Merge configurations
            merged = self.schema_composer.compose(
                base=base_config,
                extension=dtd_overrides,
                strategy=CompositionStrategy.MERGE
            )

            # Apply feature flags
            if self.feature_manager:
                dtd_features = {
                    "validation_enabled": self.feature_manager.get_feature_state("dtd_validation"),
                    "strict_mode": self.feature_manager.get_feature_state("strict_dtd_validation"),
                    "caching_enabled": self.feature_manager.get_feature_state("dtd_caching")
                }
                merged["features"] = dtd_features

            return merged

        except Exception as e:
            self.logger.error(f"Error getting DTD validation config: {str(e)}")
            return {}

    def validate_dtd_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate DTD configuration structure."""
        try:
            messages = []

            # Validate required sections
            required_sections = {
                "validation": ["enabled", "mode", "phases"],
                "caching": ["enabled", "ttl"],
                "error_handling": ["fail_fast", "strict_validation"]
            }

            for section, fields in required_sections.items():
                if section not in config:
                    messages.append(ValidationMessage(
                        path=section,
                        message=f"Missing required section: {section}",
                        severity=ValidationSeverity.ERROR,
                        code="missing_section"
                    ))
                    continue

                section_config = config[section]
                for field in fields:
                    if field not in section_config:
                        messages.append(ValidationMessage(
                            path=f"{section}.{field}",
                            message=f"Missing required field: {field}",
                            severity=ValidationSeverity.ERROR,
                            code="missing_field"
                        ))

            # Validate value types and ranges
            if ttl := config.get("caching", {}).get("ttl"):
                if not isinstance(ttl, int) or ttl < 0:
                    messages.append(ValidationMessage(
                        path="caching.ttl",
                        message="TTL must be a positive integer",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_ttl"
                    ))

            if mode := config.get("validation", {}).get("mode"):
                if mode not in ["strict", "lax", "none"]:
                    messages.append(ValidationMessage(
                        path="validation.mode",
                        message="Invalid validation mode",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_mode"
                    ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating DTD config: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Configuration validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def set_dtd_config(self, dtd_config: Dict[str, Any]) -> None:
        """Set DTD configuration settings."""
        try:
            # Store DTD configuration
            self.config_loader.load_config_file('dtd_config.json', required=False)

            # Get loaded configs from config_loader
            if loaded_configs := self.config_loader.get_config('dtd_config.json'):
                # Update with new settings
                self.config_loader.store_bulk_metadata([
                    ('dtd_config.json', dtd_config)
                ])

            # Clear cache patterns for DTD
            self.cache.invalidate_by_pattern("dtd_*")

            # Emit configuration update event
            self.event_manager.emit(
                EventType.CONFIG_UPDATE,
                config_type="dtd",
                config=dtd_config
            )

        except Exception as e:
            self.logger.error(f"Error setting DTD configuration: {str(e)}")
            raise

    def _create_default_catalog(self, catalog_path: Path) -> None:
        """Create default catalog.xml file."""
        catalog_content = '''<?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE catalog PUBLIC "-//OASIS//DTD Entity Resolution XML Catalog V1.0//EN"
                            "http://www.oasis-open.org/committees/entity/release/1.0/catalog.dtd">
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
        <!-- Map DTD public identifiers to local files -->
        <public publicId="-//OASIS//DTD DITA Map//EN" uri="schemas/map.dtd"/>
        <public publicId="-//OASIS//DTD DITA Topic//EN" uri="schemas/topic.dtd"/>
        <public publicId="-//OASIS//DTD DITA Concept//EN" uri="schemas/concept.dtd"/>
        <public publicId="-//OASIS//DTD DITA Task//EN" uri="schemas/task.dtd"/>
        <public publicId="-//OASIS//DTD DITA Reference//EN" uri="schemas/reference.dtd"/>
        <public publicId="-//OASIS//DTD DITA Glossary//EN" uri="schemas/glossentry.dtd"/>
        <public publicId="-//OASIS//DTD DITA BookMap//EN" uri="schemas/bookmap.dtd"/>

        <!-- Map system identifiers to local files -->
        <system systemId="map.dtd" uri="schemas/map.dtd"/>
        <system systemId="topic.dtd" uri="schemas/topic.dtd"/>
        <system systemId="concept.dtd" uri="schemas/concept.dtd"/>
        <system systemId="task.dtd" uri="schemas/task.dtd"/>
        <system systemId="reference.dtd" uri="schemas/reference.dtd"/>
        <system systemId="glossentry.dtd" uri="schemas/glossentry.dtd"/>
        <system systemId="bookmap.dtd" uri="schemas/bookmap.dtd"/>
    </catalog>'''

        with open(catalog_path, 'w') as f:
            f.write(catalog_content)
