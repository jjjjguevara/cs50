from app.dita.validation_manager import SchemaValidationError
"""Configuration validation for DITA processing system."""
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set
from uuid import uuid4
from dataclasses import dataclass, field
from datetime import datetime
import json
import yaml
import re

# Core validators
from .base_validator import BaseValidator

# Cache system
from ..cache.cache import ContentCache
from ..cache.strategy.lru_cache import LRUCache  # To be implemented

# Event system
from ..events.event_manager import EventManager
from ..events.states.validation_state import ValidationState  # To be implemented

# Types and models
from ..types import (
    ValidationResult,
    ValidationMessage,
    ValidationSeverity,
    SchemaValidationPattern,
    CacheEntryType,
    EventType,
    ElementType,
    ProcessingState,
    ConfigurationScope,
    ValidationScope,
    ValidationPattern,
    ValidationType,
    ValidationContext

)

# Configuration components
from ..config.config_manager import ConfigManager
from ..config.builders.schema_builder import SchemaBuilder  # To be implemented
from ..config.validators.schema_validator import SchemaValidator  # To be implemented
from ..config.validators.dependency_validator import DependencyValidator  # To be implemented

# Context management
from ..main.context_manager import ContextManager

# Utilities
from ..utils.logger import DITALogger


class ConfigValidator(BaseValidator):
    """
    Configuration validator for DITA processing system.
    Handles validation of configuration files and settings.
    """

    def __init__(
        self,
        event_manager: EventManager,
        context_manager: ContextManager,
        config_manager: ConfigManager,
        cache: ContentCache,
        schema_builder: Optional[SchemaBuilder] = None,
        dependency_validator: Optional[DependencyValidator] = None,
        logger: Optional[DITALogger] = None
    ):
        """Initialize configuration validator.

        Args:
            event_manager: System event manager
            context_manager: Context management system
            config_manager: Configuration management system
            cache: Cache system
            schema_builder: Optional schema builder
            dependency_validator: Optional dependency validator
            logger: Optional logger instance
        """
        super().__init__(
            event_manager=event_manager,
            context_manager=context_manager,
            config_manager=config_manager,
            cache=cache,
            validator_type=ValidationType.CONFIG,
            logger=logger
        )

        # Configuration components
        self.schema_builder = schema_builder or SchemaBuilder()
        self.dependency_validator = dependency_validator or DependencyValidator()

        # Validation state tracking
        self._active_validations: Set[str] = set()
        self._validation_rules: Dict[str, Dict[str, Any]] = {}
        self._validation_patterns: Dict[str, ValidationPattern] = {}

        # Initialize configuration validation
        self._initialize_validation()

    def _initialize_validation(self) -> None:
        """Initialize validation rules and patterns."""
        try:
            # Load validation rules
            rules_path = self.config_manager.config_path / "validation_rules.json"
            if rules_path.exists():
                with open(rules_path) as f:
                    self._validation_rules = json.load(f)

            # Register validation patterns
            self._register_validation_patterns()

            # Initialize validation state
            self._register_validation_handlers()

        except Exception as e:
            self.logger.error(f"Error initializing config validation: {str(e)}")
            raise

    def _register_validation_patterns(self) -> None:
        """Register configuration validation patterns."""
        try:
            # Register structure validation pattern
            self._validation_patterns["config.structure"] = SchemaValidationPattern(
                pattern_id="config.structure",
                description="Validate configuration structure",
                severity=ValidationSeverity.ERROR,
                code="invalid_config_structure",
                schema={
                    "type": "object",
                    "required": ["version", "features", "validation"],
                    "properties": {
                        "version": {"type": "string"},
                        "features": {"type": "object"},
                        "validation": {"type": "object"}
                    }
                },
                metadata={
                    "validation_type": "structure",
                    "version": "1.0.0"
                },
                dependencies=[],
                conditions={}
            )

            # Register dependencies validation pattern
            self._validation_patterns["config.dependencies"] = SchemaValidationPattern(
                pattern_id="config.dependencies",
                description="Validate configuration dependencies",
                severity=ValidationSeverity.ERROR,
                code="invalid_config_dependencies",
                schema={
                    "type": "object",
                    "properties": {
                        "dependencies": {
                            "type": "object",
                            "patternProperties": {
                                ".*": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        }
                    }
                },
                metadata={
                    "validation_type": "dependencies",
                    "version": "1.0.0"
                },
                dependencies=["config.structure"],  # Depends on structure validation
                conditions={}
            )

            # Register inheritance validation pattern
            self._validation_patterns["config.inheritance"] = SchemaValidationPattern(
                pattern_id="config.inheritance",
                description="Validate configuration inheritance",
                severity=ValidationSeverity.ERROR,
                code="invalid_config_inheritance",
                schema={
                    "type": "object",
                    "properties": {
                        "extends": {"type": "string"},
                        "inheritance_type": {
                            "type": "string",
                            "enum": ["merge", "override", "extend"]
                        },
                        "inheritance_chain": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                },
                metadata={
                    "validation_type": "inheritance",
                    "version": "1.0.0"
                },
                dependencies=["config.structure", "config.dependencies"],  # Depends on both
                conditions={}
            )

        except Exception as e:
            self.logger.error(f"Error registering validation patterns: {str(e)}")
            raise

    def _register_validation_handlers(self) -> None:
        """Register validation event handlers."""
        try:
            self.event_manager.subscribe(
                EventType.STATE_CHANGE,
                self._handle_state_change
            )
            self.event_manager.subscribe(
                EventType.CONFIG_UPDATE,
                self._handle_config_update
            )

        except Exception as e:
            self.logger.error(f"Error registering validation handlers: {str(e)}")
            raise

    def validate(
        self,
        content: Any,
        context: Optional[ValidationContext] = None,
        **kwargs: Any
    ) -> ValidationResult:
        """
        Main validation entry point for configuration content.

        Args:
            content: Configuration content to validate
            context: Optional validation context
            **kwargs: Additional validation parameters

        Returns:
            ValidationResult: Validation results
        """
        try:
            if not isinstance(content, dict):
                return ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path="",
                        message="Configuration must be a dictionary",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_type"
                    )]
                )

            # Create or validate context
            validation_context: ValidationContext
            if context is None:
                # Create new validation context for configuration
                validation_context = ValidationContext.create_config_context(
                    validation_id=f"config_{uuid4().hex[:8]}",
                    schema_path=self.config_manager.config_path,
                    environment=self.config_manager.get_env_config().get("environment", "development"),
                    config_scope=ConfigurationScope.GLOBAL,
                    metadata=kwargs.get('metadata', {}),
                    strict_mode=kwargs.get('strict_mode', False)
                )
            else:
                validation_context = context
                # Ensure correct scope
                if validation_context.scope != ValidationScope.CONFIG:
                    validation_context = ValidationContext.create_config_context(
                        validation_id=validation_context.validation_id,
                        schema_path=self.config_manager.config_path,
                        environment=self.config_manager.get_env_config().get("environment", "development"),
                        config_scope=ConfigurationScope.GLOBAL,
                        metadata={**validation_context.metadata},
                        parent_context=validation_context
                    )

            # Check cache
            cache_key = f"config_validation_{hash(str(content))}_{validation_context.validation_id}"
            if cached := self.cache.get(cache_key, CacheEntryType.VALIDATION):
                return cached

            # Update validation state
            validation_context.update_state(ProcessingState.PROCESSING)

            # Check if validation can proceed
            if not validation_context.can_validate():
                return ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path="",
                        message="Maximum validation attempts exceeded",
                        severity=ValidationSeverity.ERROR,
                        code="max_attempts_exceeded"
                    )]
                )

            messages = []

            # Structure validation
            structure_result = self._validate_structure(content, validation_context)
            messages.extend(structure_result.messages)

            # Only proceed if structure is valid
            if structure_result.is_valid:
                # Schema validation
                schema_result = self._validate_schema(content, validation_context)
                messages.extend(schema_result.messages)

                # Dependencies validation
                dependency_result = self._validate_dependencies(content, validation_context)
                messages.extend(dependency_result.messages)

                # Inheritance validation if applicable
                if "extends" in content:
                    inheritance_result = self._validate_inheritance(content, validation_context)
                    messages.extend(inheritance_result.messages)

            # Create validation result
            result = ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

            # Update validation state based on result
            validation_context.update_state(
                ProcessingState.COMPLETED if result.is_valid else ProcessingState.ERROR
            )

            # Cache result
            self.cache.set(
                key=cache_key,
                data=result,
                entry_type=CacheEntryType.VALIDATION,
                element_type=ElementType.UNKNOWN,
                phase=validation_context.phase
            )

            return result

        except Exception as e:
            self.logger.error(f"Error validating configuration: {str(e)}")
            if context:
                context.update_state(ProcessingState.ERROR)
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Configuration validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )


    def _validate_structure(
        self,
        config: Dict[str, Any],
        context: ValidationContext
    ) -> ValidationResult:
        """Validate configuration structure."""
        messages = []

        try:
            # Get structure rules
            rules = self._validation_rules.get("structure", {})

            # Validate required fields
            if required := rules.get("required_fields"):
                for field in required:
                    if field not in config:
                        messages.append(ValidationMessage(
                            path=field,
                            message=f"Missing required field: {field}",
                            severity=ValidationSeverity.ERROR,
                            code="missing_required"
                        ))

            # Validate field types
            if field_types := rules.get("field_types"):
                for field, expected_type in field_types.items():
                    if field in config:
                        if not self._validate_field_type(config[field], expected_type):
                            messages.append(ValidationMessage(
                                path=field,
                                message=f"Invalid type for {field}",
                                severity=ValidationSeverity.ERROR,
                                code="invalid_type"
                            ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Structure validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="structure_validation_error"
                )]
            )

    def _validate_schema(
        self,
        config: Dict[str, Any],
        context: ValidationContext
    ) -> ValidationResult:
        """
        Validate configuration against schema.

        Args:
            config: Configuration to validate
            context: Validation context

        Returns:
            ValidationResult: Schema validation results
        """
        try:
            # Get schema for validation
            schema = self.schema_builder.build_schema(
                scope=context.config_scope
            )

            # Create schema validation context
            schema_context = ValidationContext.create_schema_context(
                validation_id=f"schema_{context.validation_id}",
                base_schema=schema,
                inheritance_chain=context.inheritance_chain,
                metadata=context.metadata,
                parent_context=context
            )

            # Validate against schema
            return self.config_manager.validation_manager.validate(
                content=config,
                context=schema_context
            )

        except Exception as e:
            self.logger.error(f"Error in schema validation: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Schema validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="schema_validation_error"
                )]
            )

    def _validate_dependencies(
        self,
        config: Dict[str, Any],
        context: ValidationContext
    ) -> ValidationResult:
        """Validate configuration dependencies."""
        try:
            return self.dependency_validator.validate_dependencies(
                config=config,
                scope=context.scope,
                context=context
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Dependency validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="dependency_validation_error"
                )]
            )

    def _validate_inheritance(
        self,
        config: Dict[str, Any],
        context: ValidationContext
    ) -> ValidationResult:
        """Validate configuration inheritance."""
        messages = []

        try:
            # Get base configuration
            base_name = config.get("extends")
            if not base_name:
                return ValidationResult(is_valid=True, messages=[])

            base_config = self.config_manager.get_config(base_name)
            if not base_config:
                return ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path="extends",
                        message=f"Base configuration not found: {base_name}",
                        severity=ValidationSeverity.ERROR,
                        code="base_not_found"
                    )]
                )

            # Validate inheritance rules
            if rules := self._validation_rules.get("inheritance"):
                # Check required field preservation
                if preserved := rules.get("preserved_fields"):
                    for field in preserved:
                        if field in base_config and field not in config:
                            messages.append(ValidationMessage(
                                path=field,
                                message=f"Required field must be preserved: {field}",
                                severity=ValidationSeverity.ERROR,
                                code="field_not_preserved"
                            ))

                # Check inheritance restrictions
                if restrictions := rules.get("restrictions"):
                    for field, restriction in restrictions.items():
                        if field in config:
                            if not self._validate_inheritance_restriction(
                                config[field],
                                base_config.get(field),
                                restriction
                            ):
                                messages.append(ValidationMessage(
                                    path=field,
                                    message=f"Invalid inheritance for {field}",
                                    severity=ValidationSeverity.ERROR,
                                    code="invalid_inheritance"
                                ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Inheritance validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="inheritance_validation_error"
                )]
            )

    def _validate_field_type(self, value: Any, expected_type: str) -> bool:
        """Validate field value against expected type."""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }

        expected = type_map.get(expected_type)
        if not expected:
            return True  # Skip validation for unknown types

        return isinstance(value, expected)

    def _validate_inheritance_restriction(
        self,
        value: Any,
        base_value: Any,
        restriction: str
    ) -> bool:
        """Validate value against inheritance restriction."""
        if restriction == "immutable":
            return value == base_value
        elif restriction == "extend_only":
            if isinstance(value, (list, set)):
                return all(item in value for item in base_value)
            elif isinstance(value, dict):
                return all(key in value for key in base_value)
        elif restriction == "reduce_only":
            if isinstance(value, (list, set)):
                return all(item in base_value for item in value)
            elif isinstance(value, dict):
                return all(key in base_value for key in value)
        return True

    def _handle_state_change(self, **event_data: Any) -> None:
        """Handle validation state changes."""
        try:
            element_id = event_data.get("element_id")
            state_info = event_data.get("state_info")

            if element_id and state_info:
                if element_id in self._active_validations:
                    if state_info.state == ProcessingState.ERROR:
                        # Handle validation error
                        self.cache.invalidate_by_pattern(
                            f"config_validation_{element_id}_*",
                            entry_type=CacheEntryType.VALIDATION
                        )
                    elif state_info.state == ProcessingState.COMPLETED:
                        # Clear from active validations
                        self._active_validations.discard(element_id)

        except Exception as e:
            self.logger.error(f"Error handling state change: {str(e)}")

    def _handle_config_update(self, **event_data: Any) -> None:
        """Handle configuration update events."""
        try:
            config_type = event_data.get("config_type")
            if config_type:
                # Invalidate relevant validation cache
                self.cache.invalidate_by_pattern(
                    f"config_validation_{config_type}_*",
                    entry_type=CacheEntryType.VALIDATION
                )

                # Emit validation event
                self.event_manager.emit(
                    EventType.VALIDATION_FAILED,
                    element_id=f"config_{config_type}",
                    validation_type="config",
                    message="Configuration updated, revalidation required"
                )

        except Exception as e:
            self.logger.error(f"Error handling config update: {str(e)}")

    def can_validate(self, content: Any) -> bool:
        """Check if validator can handle content."""
        return isinstance(content, dict)

    def get_supported_types(self) -> Set[ElementType]:
        """Get supported element types."""
        return {ElementType.UNKNOWN}  # Config validation is type-agnostic

    def cleanup(self) -> None:
        """Clean up validator resources."""
        try:
            self._active_validations.clear()
            self._validation_rules.clear()
            self._validation_patterns.clear()
            self.cache.invalidate_by_pattern("config_validation_*")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise
