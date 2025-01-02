"""Central facade for DITA configuration management."""
from typing import Dict, Optional, Any, Union, TypeVar, Set, Type
from dataclasses import dataclass, field
from pathlib import Path
import os
from datetime import datetime
import logging
from functools import lru_cache

from ..event_manager import EventManager, EventType
from ..utils.id_handler import DITAIDHandler, IDType
from ..utils.cache import ContentCache, CacheEntryType
from ..utils.logger import DITALogger
from ..validation_manager import ValidationManager
from ..schema_manager import SchemaManager

# Import configuration components
from .loaders.config_loader import ConfigLoader
from .features.feature_manager import FeatureManager
from .rules.rule_resolver import RuleResolver


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
        validation_manager: ValidationManager,  # Add these parameters
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
        self.schema_manager = schema_manager
        self.logger = logger or DITALogger(name=__name__)

        # Initialize components
        self._init_components()

        # Register for events
        self._register_event_handlers()

        # State tracking
        self._initialized = False
        self._environment = "development"

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
        """Get processing rules for element type and context."""
        try:
            resolved = self.resolve_rule(
                element_type=element_type,
                rule_type=ProcessingRuleType.ELEMENT,  # Base processing rules
                context=context
            )
            return resolved or {}
        except Exception as e:
            self.logger.error(f"Error getting processing rules: {str(e)}")
            return {}

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
        """Load key resolution configuration."""
        try:
            # Try loading from keyref_config.json first
            if keyref_config := self.keyref_config:
                if "keyref_resolution" in keyref_config:
                    return {
                        "resolution_rules": keyref_config["keyref_resolution"],
                        "processing_hierarchy": keyref_config.get("processing_hierarchy", {}),
                        "global_defaults": keyref_config.get("global_defaults", {}),
                        "element_defaults": keyref_config.get("element_defaults", {})
                    }

            # Fall back to key_resolution.json
            key_resolution = self.config_loader.get_config("key_resolution.json")
            if key_resolution:
                if "resolution_rules" not in key_resolution:
                    raise ValueError("Missing resolution_rules in key resolution config")
                return key_resolution

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
