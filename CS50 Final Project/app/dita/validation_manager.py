"""Centralized validation management for DITA processing pipeline."""

from typing import Callable, Dict, Optional, Any, List, Set, Union, Tuple, Type, TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
import re


# Type checking imports
if TYPE_CHECKING:
    from .config.config_manager import ConfigManager
    from .context_manager import ContextManager
    from .metadata.metadata_manager import MetadataManager
    from .schema_manager import CompositionStrategy

# Direct imports
from .event_manager import EventManager, EventType
from .utils.cache import ContentCache, CacheEntryType
from .utils.logger import DITALogger


# Type imports
from .models.types import (
    ElementType,
    ProcessingPhase,
    ProcessingState,
    ContentScope,
    ContentRelationship,
    ContentRelationType,
    ValidationResult,
    ValidationMessage,
    ValidationSeverity,
    ProcessingContext,
    TrackedElement,
    MetadataState,
    ProcessingRule,
    ProcessingStateInfo,
    ProcessingRuleType,
    Feature,
    FeatureScope,
    NavigationContext
)


from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set, Type
from datetime import datetime


@dataclass
class FieldValidationRule:
    """Field validation rule definition."""
    field_type: str
    required: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    min_items: Optional[int] = None
    max_items: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class ValidationPipeline:
    """Pipeline for sequential validation operations."""

    def __init__(self):
        self.validators: List[Tuple[str, Callable[..., ValidationResult]]] = []
        self._active = True
        self._fail_fast = False

    def add_validator(
        self,
        name: str,
        validator: Callable[..., ValidationResult],
        position: Optional[int] = None
    ) -> None:
        """
        Add validator to pipeline.

        Args:
            name: Validator name
            validator: Validation function
            position: Optional position in pipeline (default: append)
        """
        if position is not None:
            self.validators.insert(position, (name, validator))
        else:
            self.validators.append((name, validator))

    def set_fail_fast(self, fail_fast: bool) -> None:
        """Set whether to stop on first error."""
        self._fail_fast = fail_fast

    def is_active(self) -> bool:
        """Check if pipeline is active."""
        return self._active

    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Run validation pipeline.

        Args:
            value: Value to validate
            context: Optional validation context

        Returns:
            ValidationResult: Combined validation result
        """
        messages = []
        is_valid = True

        for name, validator in self.validators:
            try:
                if context:
                    result = validator(value, **context)
                else:
                    result = validator(value)

                messages.extend(result.messages)

                if not result.is_valid:
                    is_valid = False
                    if self._fail_fast:
                        break

            except Exception as e:
                messages.append(ValidationMessage(
                    path=name,
                    message=f"Validator failed: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validator_error"
                ))
                is_valid = False
                if self._fail_fast:
                    break

        return ValidationResult(
            is_valid=is_valid,
            messages=messages
        )

class ValidationPattern:
    """Base class for validation patterns."""

    def __init__(
        self,
        pattern_id: str,
        description: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ):
        self.pattern_id = pattern_id
        self.description = description
        self.severity = severity

    def validate(self, value: Any) -> ValidationResult:
        """Validate value against pattern."""
        raise NotImplementedError

class RegexValidationPattern(ValidationPattern):
    """Pattern for regex-based validation."""

    def __init__(
        self,
        pattern_id: str,
        pattern: str,
        description: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        flags: int = 0
    ):
        super().__init__(pattern_id, description, severity)
        self.regex = re.compile(pattern, flags)

    def validate(self, value: Any) -> ValidationResult:
        """Validate value against regex pattern."""
        if not isinstance(value, str):
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path=self.pattern_id,
                    message="Value must be a string",
                    severity=self.severity,
                    code="invalid_type"
                )]
            )

        if not self.regex.match(value):
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path=self.pattern_id,
                    message=f"Value does not match pattern: {self.description}",
                    severity=self.severity,
                    code="pattern_mismatch"
                )]
            )

        return ValidationResult(is_valid=True, messages=[])

class SchemaValidationPattern(ValidationPattern):
    """Pattern for schema-based validation."""

    def __init__(
        self,
        pattern_id: str,
        schema: Dict[str, Any],
        description: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ):
        super().__init__(pattern_id, description, severity)
        self.schema = schema

    def validate(self, value: Any) -> ValidationResult:
        """Validate value against schema pattern."""
        messages = []

        def validate_against_schema(
            val: Any,
            schema: Dict[str, Any],
            path: str
        ) -> None:
            schema_type = schema.get("type")

            # Type validation
            if schema_type:
                if not self._validate_type(val, schema_type):
                    messages.append(ValidationMessage(
                        path=path,
                        message=f"Expected type {schema_type}",
                        severity=self.severity,
                        code="type_mismatch"
                    ))
                    return

            # Required fields
            if required := schema.get("required", []):
                if isinstance(val, dict):
                    for field in required:
                        if field not in val:
                            messages.append(ValidationMessage(
                                path=f"{path}.{field}",
                                message=f"Missing required field: {field}",
                                severity=self.severity,
                                code="missing_required"
                            ))

            # Properties validation
            if properties := schema.get("properties"):
                if isinstance(val, dict):
                    for prop, prop_schema in properties.items():
                        if prop in val:
                            validate_against_schema(
                                val[prop],
                                prop_schema,
                                f"{path}.{prop}"
                            )

            # Array validation
            if items := schema.get("items"):
                if isinstance(val, list):
                    for i, item in enumerate(val):
                        validate_against_schema(
                            item,
                            items,
                            f"{path}[{i}]"
                        )

        validate_against_schema(value, self.schema, self.pattern_id)

        return ValidationResult(
            is_valid=not messages,
            messages=messages
        )

    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value type against schema type."""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }

        expected_types = type_map.get(expected_type)
        if expected_types:
            return isinstance(value, expected_types)
        return True

class ValidationManager:
    """Enhanced ValidationManager with pattern support and pipeline."""

    def __init__(
        self,
        cache: ContentCache,
        event_manager: EventManager,
        config_manager: Optional['ConfigManager'] = None,
        context_manager: Optional['ContextManager'] = None,
        metadata_manager: Optional['MetadataManager'] = None,
        logger: Optional[DITALogger] = None
    ):
        """
        Initialize validation manager with required dependencies.

        Args:
            cache: For caching validation results
            event_manager: For validation events
            config_manager: For validation rules and patterns
            context_manager: Optional, for context-aware validation
            metadata_manager: Optional, for metadata validation
            logger: Optional custom logger
        """
        # Core dependencies
        self.cache = cache
        self.event_manager = event_manager
        self._config_manager = None
        self.logger = logger or DITALogger(name=__name__)

        if config_manager:
            self.config_manager = config_manager  # This will trigger the setter

        # Optional dependencies
        self._context_manager = context_manager
        self._metadata_manager = metadata_manager

        # Validation tracking
        self._validation_results: Dict[str, ValidationResult] = {}
        self._active_validations: Set[str] = set()
        self._validation_cache: Dict[str, Dict[str, Any]] = {}

        self._validation_patterns = {}
        if self._config_manager:
            self._load_validation_patterns()

        # Register for events
        self._register_event_handlers()

        # Pattern registry
        self._patterns: Dict[str, ValidationPattern] = {}

        # Pipeline registry
        self._pipelines: Dict[str, ValidationPipeline] = {}

        # Initialize patterns
        self._initialize_patterns()

        # Initialize pipelines
        self._initialize_pipelines()

    @property
    def config_manager(self) -> Optional['ConfigManager']:
        """Get config manager."""
        return self._config_manager

    @config_manager.setter
    def config_manager(self, manager: 'ConfigManager') -> None:
        """Set config manager and initialize patterns."""
        self._config_manager = manager
        self._load_validation_patterns()

    def _initialize_patterns(self) -> None:
        """Initialize validation patterns from configuration."""
        # Load patterns from validation_patterns.json
        try:
            pattern_path = Path("validation_patterns.json")
            if pattern_path.exists():
                with open(pattern_path) as f:
                    patterns = json.load(f)

                # Register patterns
                for pattern_id, pattern_def in patterns.get("patterns", {}).items():
                    pattern_type = pattern_def.get("type", "regex")

                    if pattern_type == "regex":
                        self._patterns[pattern_id] = RegexValidationPattern(
                            pattern_id=pattern_id,
                            pattern=pattern_def["pattern"],
                            description=pattern_def.get("description", ""),
                            severity=ValidationSeverity(
                                pattern_def.get("severity", "error")
                            )
                        )
                    elif pattern_type == "schema":
                        self._patterns[pattern_id] = SchemaValidationPattern(
                            pattern_id=pattern_id,
                            schema=pattern_def["schema"],
                            description=pattern_def.get("description", ""),
                            severity=ValidationSeverity(
                                pattern_def.get("severity", "error")
                            )
                        )

        except Exception as e:
            self.logger.error(f"Error initializing patterns: {str(e)}")

    def _initialize_pipelines(self) -> None:
        """Initialize validation pipelines."""
        # Initialize standard pipelines
        self._pipelines["content"] = ValidationPipeline()
        self._pipelines["metadata"] = ValidationPipeline()
        self._pipelines["config"] = ValidationPipeline()
        self._pipelines["schema"] = ValidationPipeline()


    def validate_type_mappings(
        self,
        type_mappings: Dict[str, str],
        valid_types: Set[str],
        schema: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate type mappings against valid types and schema.

        Args:
            type_mappings: Dictionary of element type mappings
            valid_types: Set of valid element types
            schema: Processing schema to validate against

        Returns:
            ValidationResult: Validation result
        """
        messages = []

        try:
            # Validate each mapping
            for element_type, rule_path in type_mappings.items():
                # Validate rule path exists in schema
                if not self._validate_rule_path(rule_path, schema):
                    messages.append(ValidationMessage(
                        path=f"mappings.{element_type}",
                        message=f"Invalid rule path for element type: {rule_path}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_rule_path"
                    ))
                    continue

                # Resolve and validate element type
                resolved_type = self._resolve_element_type(element_type, rule_path, schema)
                if resolved_type not in valid_types:
                    messages.append(ValidationMessage(
                        path=f"mappings.{element_type}",
                        message=f"Invalid resolved type: {resolved_type}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_element_type"
                    ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating type mappings: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Type mapping validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def _validate_rule_path(
        self,
        rule_path: str,
        schema: Dict[str, Any]
    ) -> bool:
        """Validate rule path exists in schema."""
        try:
            current = schema
            for part in rule_path.split('.'):
                if not isinstance(current, dict) or part not in current:
                    return False
                current = current[part]
            return True
        except Exception:
            return False

    def _resolve_element_type(
        self,
        element_type: str,
        rule_path: str,
        schema: Dict[str, Any]
    ) -> str:
        """Resolve element type from rule path."""
        try:
            current = schema
            for part in rule_path.split('.'):
                if not isinstance(current, dict):
                    return element_type  # Fallback to original type
                current = current.get(part, {})

            # Look for type information in resolved path
            if isinstance(current, dict):
                return current.get('type', element_type)
            return element_type

        except Exception:
            return element_type

    def _validate_scope_rules(
        self,
        scope_validation: Dict[str, Any]
    ) -> List[ValidationMessage]:
        """Validate scope validation rules."""
        messages = []

        if not isinstance(scope_validation.get("rules", {}), dict):
            messages.append(ValidationMessage(
                path="scope_validation.rules",
                message="Scope validation rules must be a dictionary",
                severity=ValidationSeverity.ERROR,
                code="invalid_rules_type"
            ))
            return messages

        for scope, rules in scope_validation["rules"].items():
            # Validate allowed references
            if refs := rules.get("allowed_references"):
                if not isinstance(refs, list):
                    messages.append(ValidationMessage(
                        path=f"scope_validation.rules.{scope}.allowed_references",
                        message="Allowed references must be a list",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_references_type"
                    ))

            # Validate metadata inheritance flag
            if "metadata_inheritance" in rules and not isinstance(rules["metadata_inheritance"], bool):
                messages.append(ValidationMessage(
                    path=f"scope_validation.rules.{scope}.metadata_inheritance",
                    message="Metadata inheritance must be a boolean",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_inheritance_type"
                ))

        return messages






    def _load_validation_patterns(self) -> None:
        """Load validation patterns from configuration."""
        try:
            # Check for config manager
            if not self._config_manager:
                self.logger.warning("Config manager not initialized, skipping pattern loading")
                return

            # Get validation patterns file path
            validation_path = self._config_manager.config_path / "validation_patterns.json"
            if not validation_path.exists():
                self.logger.warning("Validation patterns file not found")
                self._validation_patterns = {}
                return

            try:
                with open(validation_path) as f:
                    config = json.load(f)
                    self._validation_patterns = config.get("patterns", {})

                    # Initialize patterns
                    for pattern_type, patterns in self._validation_patterns.items():
                        for pattern_id, pattern_def in patterns.items():
                            pattern_type = pattern_def.get("type", "regex")

                            if pattern_type == "regex":
                                self._patterns[pattern_id] = RegexValidationPattern(
                                    pattern_id=pattern_id,
                                    pattern=pattern_def["pattern"],
                                    description=pattern_def.get("description", ""),
                                    severity=ValidationSeverity(
                                        pattern_def.get("severity", "error")
                                    )
                                )
                            elif pattern_type == "schema":
                                self._patterns[pattern_id] = SchemaValidationPattern(
                                    pattern_id=pattern_id,
                                    schema=pattern_def["schema"],
                                    description=pattern_def.get("description", ""),
                                    severity=ValidationSeverity(
                                        pattern_def.get("severity", "error")
                                    )
                                )

                    self.logger.debug(f"Loaded {len(self._patterns)} validation patterns")

            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing validation patterns: {str(e)}")
                self._validation_patterns = {}
                self._patterns = {}

        except Exception as e:
            self.logger.error(f"Error loading validation patterns: {str(e)}")
            self._validation_patterns = {}
            self._patterns = {}

    def _register_event_handlers(self) -> None:
        """Register validation-related event handlers."""
        self.event_manager.subscribe(
            EventType.STATE_CHANGE,
            self._handle_state_change
        )

    def validate_content(
        self,
        element: TrackedElement,
        context: ProcessingContext
    ) -> ValidationResult:
        """
        Validate content element based on type and context.

        Args:
            element: Element to validate
            context: Current processing context

        Returns:
            ValidationResult with validation status and messages
        """
        messages = []

        try:
            # Check for config manager
            if not self._config_manager:
                self.logger.warning("Config manager not initialized, skipping rule validation")
                return ValidationResult(is_valid=True, messages=[])

            # Get validation rules
            rules = self._config_manager.get_processing_rules(
                element_type=element.type,
                context=context
            )

            # Validate against rules
            if rules:
                messages.extend(self._validate_against_rules(element, rules))

            # Validate patterns if applicable
            if pattern_messages := self._validate_patterns(element):
                messages.extend(pattern_messages)

            # Validate metadata if available
            if metadata_result := self._validate_metadata(element, context):
                messages.extend(metadata_result.messages)

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating content: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def validate_all_configurations(
        self,
        feature_registry: Dict[str, Feature],
        rule_registry: Dict[str, Dict[str, ProcessingRule]],
        keyref_config: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate all configuration components.

        Args:
            feature_registry: Feature configurations to validate
            rule_registry: Processing rules to validate
            keyref_config: Optional keyref configuration

        Returns:
            ValidationResult with validation status and messages
        """
        messages = []

        # Validate features
        for name, feature in feature_registry.items():
            feature_result = self.validate_feature_config(feature)
            if not feature_result.is_valid:
                messages.extend([
                    ValidationMessage(
                        path=f"features.{name}",
                        message=msg.message,
                        severity=msg.severity,
                        code=msg.code
                    ) for msg in feature_result.messages
                ])

        # Validate rules
        for rule_type, rules in rule_registry.items():
            for rule_id, rule in rules.items():
                rule_result = self.validate_processing_rule(rule)
                if not rule_result.is_valid:
                    messages.extend([
                        ValidationMessage(
                            path=f"rules.{rule_type}.{rule_id}",
                            message=msg.message,
                            severity=msg.severity,
                            code=msg.code
                        ) for msg in rule_result.messages
                    ])

        # Validate keyref config
        if keyref_config:
            keyref_result = self.validate_keyref_config(keyref_config)
            if not keyref_result.is_valid:
                messages.extend([
                    ValidationMessage(
                        path="keyref_config",
                        message=msg.message,
                        severity=msg.severity,
                        code=msg.code
                    ) for msg in keyref_result.messages
                ])

        return ValidationResult(
            is_valid=not any(
                msg.severity == ValidationSeverity.ERROR for msg in messages
            ),
            messages=messages
        )

    def validate_feature_config(
        self,
        feature: Feature
    ) -> ValidationResult:
        """
        Validate feature configuration.

        Args:
            feature: Feature to validate

        Returns:
            ValidationResult with validation status and messages
        """
        messages = []

        try:
            # Validate name format
            if not re.match(r'^[a-z][a-z0-9_]*$', feature.name):
                messages.append(ValidationMessage(
                    path="name",
                    message="Invalid feature name format. Must start with lowercase letter and contain only letters, numbers, and underscores",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_name_format"
                ))

            # Validate scope
            if not isinstance(feature.scope, FeatureScope):
                messages.append(ValidationMessage(
                    path="scope",
                    message=f"Invalid scope type: {type(feature.scope)}",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_scope_type"
                ))

            # Validate default value type
            if not isinstance(feature.default, bool):
                messages.append(ValidationMessage(
                    path="default",
                    message="Default value must be boolean",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_default_type"
                ))

            # Validate dependencies format
            if not isinstance(feature.dependencies, list):
                messages.append(ValidationMessage(
                    path="dependencies",
                    message="Dependencies must be a list",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_dependencies_type"
                ))
            else:
                for dep in feature.dependencies:
                    if not isinstance(dep, str):
                        messages.append(ValidationMessage(
                            path="dependencies",
                            message="All dependencies must be strings",
                            severity=ValidationSeverity.ERROR,
                            code="invalid_dependency_type"
                        ))

            # Validate conflicts format
            if not isinstance(feature.conflicts, list):
                messages.append(ValidationMessage(
                    path="conflicts",
                    message="Conflicts must be a list",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_conflicts_type"
                ))
            else:
                for conflict in feature.conflicts:
                    if not isinstance(conflict, str):
                        messages.append(ValidationMessage(
                            path="conflicts",
                            message="All conflicts must be strings",
                            severity=ValidationSeverity.ERROR,
                            code="invalid_conflict_type"
                        ))

            # Validate metadata format
            if not isinstance(feature.metadata, dict):
                messages.append(ValidationMessage(
                    path="metadata",
                    message="Metadata must be a dictionary",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_metadata_type"
                ))

            # Validate timestamps
            if feature.created_at and not isinstance(feature.created_at, datetime):
                messages.append(ValidationMessage(
                    path="created_at",
                    message="Created timestamp must be a datetime",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_created_timestamp"
                ))

            if feature.modified_at and not isinstance(feature.modified_at, datetime):
                messages.append(ValidationMessage(
                    path="modified_at",
                    message="Modified timestamp must be a datetime",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_modified_timestamp"
                ))

            # Validate description if present
            if feature.description is not None and not isinstance(feature.description, str):
                messages.append(ValidationMessage(
                    path="description",
                    message="Description must be a string",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_description_type"
                ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating feature config: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def validate_keyref_config(
        self,
        config: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate keyref configuration structure and values.

        Args:
            config: Keyref configuration to validate

        Returns:
            ValidationResult: Validation result with messages
        """
        messages = []

        try:
            # Required sections
            required_sections = {
                "processing_hierarchy": [
                    "order"
                ],
                "global_defaults": [],  # No specific required fields
                "element_defaults": [], # No specific required fields
                "keyref_resolution": [
                    "scopes",
                    "fallback_order",
                    "inheritance_rules"
                ]
            }

            # Validate required sections exist
            for section, required_fields in required_sections.items():
                if section not in config:
                    messages.append(ValidationMessage(
                        path=section,
                        message=f"Missing required section: {section}",
                        severity=ValidationSeverity.ERROR,
                        code="missing_section"
                    ))
                    continue

                # Validate required fields in section
                for field in required_fields:
                    if field not in config[section]:
                        messages.append(ValidationMessage(
                            path=f"{section}.{field}",
                            message=f"Missing required field in {section}: {field}",
                            severity=ValidationSeverity.ERROR,
                            code="missing_field"
                        ))

            # Validate processing hierarchy
            if processing_hierarchy := config.get("processing_hierarchy", {}).get("order"):
                if not isinstance(processing_hierarchy, list):
                    messages.append(ValidationMessage(
                        path="processing_hierarchy.order",
                        message="Processing hierarchy must be a list",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_type"
                    ))
                elif not all(isinstance(level, str) for level in processing_hierarchy):
                    messages.append(ValidationMessage(
                        path="processing_hierarchy.order",
                        message="All hierarchy levels must be strings",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_level_type"
                    ))

            # Validate keyref resolution
            if resolution := config.get("keyref_resolution"):
                # Validate scopes
                if scopes := resolution.get("scopes"):
                    if not isinstance(scopes, list):
                        messages.append(ValidationMessage(
                            path="keyref_resolution.scopes",
                            message="Scopes must be a list",
                            severity=ValidationSeverity.ERROR,
                            code="invalid_type"
                        ))
                    else:
                        valid_scopes = {"local", "peer", "external"}
                        invalid_scopes = [
                            scope for scope in scopes
                            if scope not in valid_scopes
                        ]
                        if invalid_scopes:
                            messages.append(ValidationMessage(
                                path="keyref_resolution.scopes",
                                message=f"Invalid scopes: {', '.join(invalid_scopes)}",
                                severity=ValidationSeverity.ERROR,
                                code="invalid_scope"
                            ))

                # Validate fallback order
                if fallback := resolution.get("fallback_order"):
                    if not isinstance(fallback, list):
                        messages.append(ValidationMessage(
                            path="keyref_resolution.fallback_order",
                            message="Fallback order must be a list",
                            severity=ValidationSeverity.ERROR,
                            code="invalid_type"
                        ))

                # Validate inheritance rules
                if inheritance := resolution.get("inheritance_rules"):
                    if not isinstance(inheritance, dict):
                        messages.append(ValidationMessage(
                            path="keyref_resolution.inheritance_rules",
                            message="Inheritance rules must be a dictionary",
                            severity=ValidationSeverity.ERROR,
                            code="invalid_type"
                        ))
                    else:
                        valid_strategies = {"merge", "append", "override"}
                        for attr, strategy in inheritance.items():
                            if strategy not in valid_strategies:
                                messages.append(ValidationMessage(
                                    path=f"keyref_resolution.inheritance_rules.{attr}",
                                    message=f"Invalid inheritance strategy: {strategy}",
                                    severity=ValidationSeverity.ERROR,
                                    code="invalid_strategy"
                                ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating keyref config: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )



    def validate_schema_composition(
        self,
        base_schema: Dict[str, Any],
        extension_schema: Dict[str, Any],
        composition_type: 'CompositionStrategy'
    ) -> ValidationResult:
        """
        Validate schema composition before applying it.

        Args:
            base_schema: Base schema
            extension_schema: Extension schema
            composition_type: Type of composition being performed

        Returns:
            ValidationResult: Validation result
        """
        messages = []

        try:
            # Basic type validation
            if not all(isinstance(s, dict) for s in [base_schema, extension_schema]):
                return ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path="",
                        message="Both base and extension must be dictionaries",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_schema_type"
                    )]
                )

            # Validate based on composition type
            if composition_type == CompositionStrategy.MERGE:
                messages.extend(
                    self._validate_merge_compatibility(base_schema, extension_schema)
                )
            elif composition_type == CompositionStrategy.SELECTIVE:
                messages.extend(
                    self._validate_selective_compatibility(base_schema, extension_schema)
                )
            elif composition_type == CompositionStrategy.ADDITIVE:
                messages.extend(
                    self._validate_additive_compatibility(base_schema, extension_schema)
                )

            # Check for required fields preservation
            messages.extend(
                self._validate_required_fields_preservation(
                    base_schema,
                    extension_schema
                )
            )

            # Validate reference integrity
            messages.extend(
                self._validate_reference_integrity(base_schema, extension_schema)
            )

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating schema composition: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Schema composition validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def _validate_additive_compatibility(
        self,
        base: Dict[str, Any],
        extension: Dict[str, Any]
    ) -> List[ValidationMessage]:
        """
        Validate compatibility for additive composition.
        Ensures new fields don't conflict with existing ones.
        """
        messages = []

        def validate_additive_fields(
            base_schema: Dict[str, Any],
            ext_schema: Dict[str, Any],
            path: str = ""
        ) -> None:
            for key, ext_value in ext_schema.items():
                current_path = f"{path}.{key}" if path else key

                if key in base_schema:
                    base_value = base_schema[key]
                    # If both are objects, recurse
                    if isinstance(base_value, dict) and isinstance(ext_value, dict):
                        validate_additive_fields(base_value, ext_value, current_path)
                    # If types don't match for existing fields
                    elif type(base_value) != type(ext_value):
                        messages.append(ValidationMessage(
                            path=current_path,
                            message=f"Type conflict for existing field: {type(base_value)} vs {type(ext_value)}",
                            severity=ValidationSeverity.ERROR,
                            code="type_conflict"
                        ))
                    # If values are different for non-dict types
                    elif base_value != ext_value:
                        messages.append(ValidationMessage(
                            path=current_path,
                            message="Value conflict for existing field",
                            severity=ValidationSeverity.WARNING,
                            code="value_conflict"
                        ))

        validate_additive_fields(base, extension)
        return messages

    def _validate_selective_compatibility(
        self,
        base: Dict[str, Any],
        extension: Dict[str, Any]
    ) -> List[ValidationMessage]:
        """
        Validate compatibility for selective composition.
        Ensures specified fields can be safely merged.

        Args:
            base: Base schema
            extension: Extension schema

        Returns:
            List[ValidationMessage]: Validation messages
        """
        messages = []

        def validate_selective_field(
            base_value: Any,
            ext_value: Any,
            path: str
        ) -> None:
            # Type compatibility check
            if type(base_value) != type(ext_value):
                messages.append(ValidationMessage(
                    path=path,
                    message=f"Type mismatch for selective field: {type(base_value)} vs {type(ext_value)}",
                    severity=ValidationSeverity.ERROR,
                    code="selective_type_mismatch"
                ))
                return

            # For dict types, recurse into structure
            if isinstance(base_value, dict) and isinstance(ext_value, dict):
                for key, value in ext_value.items():
                    if key in base_value:
                        validate_selective_field(
                            base_value[key],
                            value,
                            f"{path}.{key}"
                        )
            # For list types, check structure compatibility
            elif isinstance(base_value, list) and isinstance(ext_value, list):
                if base_value and ext_value:  # If both lists have items
                    # Check first items to determine structure
                    validate_selective_field(
                        base_value[0],
                        ext_value[0],
                        f"{path}[0]"
                    )
            # For primitive types, no additional checks needed
            # They've already passed the type check

        # Find common fields between base and extension
        for key, ext_value in extension.items():
            if key in base:
                validate_selective_field(base[key], ext_value, key)
            else:
                messages.append(ValidationMessage(
                    path=key,
                    message=f"Field not present in base schema: {key}",
                    severity=ValidationSeverity.WARNING,
                    code="selective_missing_field"
                ))

        return messages

    def _validate_merge_compatibility(
        self,
        base: Dict[str, Any],
        extension: Dict[str, Any]
    ) -> List[ValidationMessage]:
        """Validate type compatibility for merge operation."""
        messages = []

        for key, ext_value in extension.items():
            if key in base:
                base_value = base[key]
                if type(base_value) != type(ext_value):
                    messages.append(ValidationMessage(
                        path=key,
                        message=f"Type mismatch: {type(base_value)} vs {type(ext_value)}",
                        severity=ValidationSeverity.ERROR,
                        code="type_mismatch"
                    ))
                elif isinstance(base_value, dict):
                    messages.extend(
                        self._validate_merge_compatibility(base_value, ext_value)
                    )

        return messages

    def _validate_required_fields_preservation(
        self,
        base: Dict[str, Any],
        extension: Dict[str, Any]
    ) -> List[ValidationMessage]:
        """Validate required fields are preserved."""
        messages = []

        def get_required_fields(schema: Dict[str, Any]) -> Set[str]:
            fields = set()
            if required := schema.get("required"):
                fields.update(required)
            for value in schema.values():
                if isinstance(value, dict):
                    fields.update(get_required_fields(value))
            return fields

        # Get required fields from base
        base_required = get_required_fields(base)

        # Check if extension preserves them
        for field in base_required:
            if not self._field_exists(field, extension):
                messages.append(ValidationMessage(
                    path=field,
                    message=f"Required field would be lost: {field}",
                    severity=ValidationSeverity.ERROR,
                    code="required_field_lost"
                ))

        return messages

    def _validate_reference_integrity(
        self,
        base: Dict[str, Any],
        extension: Dict[str, Any]
    ) -> List[ValidationMessage]:
        """Validate reference integrity between schemas."""
        messages = []

        def get_references(schema: Dict[str, Any]) -> Set[str]:
            refs = set()
            for key, value in schema.items():
                if key == "$ref" and isinstance(value, str):
                    refs.add(value)
                elif isinstance(value, dict):
                    refs.update(get_references(value))
            return refs

        # Get all references
        base_refs = get_references(base)
        ext_refs = get_references(extension)

        # Check if extension maintains reference validity
        for ref in base_refs:
            if not self._reference_exists(ref, extension):
                messages.append(ValidationMessage(
                    path=ref,
                    message=f"Reference would be broken: {ref}",
                    severity=ValidationSeverity.ERROR,
                    code="broken_reference"
                ))

        return messages

    def _field_exists(self, field: str, schema: Dict[str, Any]) -> bool:
        """Check if field exists in schema."""
        parts = field.split('.')
        current = schema

        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return False
            current = current[part]

        return True

    def _reference_exists(self, ref: str, schema: Dict[str, Any]) -> bool:
        """Check if reference exists in schema."""
        # Remove reference prefix if exists
        ref = ref.replace('#/', '') if ref.startswith('#/') else ref
        return self._field_exists(ref, schema)

    def validate_schema_completeness(
        self,
        schema: Dict[str, Any],
        required_types: List[ElementType]
    ) -> ValidationResult:
        """
        Validate schema completeness for required element types.

        Args:
            schema: Schema to validate
            required_types: Required element types

        Returns:
            ValidationResult with validation status and messages
        """
        messages = []
        type_mapping = schema.get("element_type_mapping", {})

        for element_type in required_types:
            rule_path = type_mapping.get(element_type.value)
            if not rule_path:
                messages.append(ValidationMessage(
                    path=f"element_type_mapping.{element_type.value}",
                    message=f"Missing rule path for element type: {element_type.value}",
                    severity=ValidationSeverity.ERROR,
                    code="missing_rule_path"
                ))
                continue

            # Validate rule path exists
            current = schema
            for part in rule_path.split("."):
                current = current.get(part)
                if current is None:
                    messages.append(ValidationMessage(
                        path=f"rules.{rule_path}",
                        message=f"Invalid rule path for element type: {element_type.value}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_rule_path"
                    ))
                    break

        return ValidationResult(
            is_valid=not messages,
            messages=messages
        )



    def _validate_action_config(
        self,
        config: Dict[str, Any],
        operation: str
    ) -> List[ValidationMessage]:
        """
        Validate operation-specific action configuration.

        Args:
            config: Action configuration to validate
            operation: Operation type

        Returns:
            List[ValidationMessage]: Validation messages
        """
        messages = []

        # Define validation rules for each operation type
        operation_rules = {
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

        if rules := operation_rules.get(operation):
            # Check required fields
            for field in rules['required']:
                if field not in config:
                    messages.append(ValidationMessage(
                        path=f"action.{field}",
                        message=f"Missing required field for {operation}: {field}",
                        severity=ValidationSeverity.ERROR,
                        code=f"missing_{field}"
                    ))

            # Check for unknown fields
            all_fields = rules['required'].union(rules['optional'])
            for field in config:
                if field not in all_fields:
                    messages.append(ValidationMessage(
                        path=f"action.{field}",
                        message=f"Unknown field for {operation}: {field}",
                        severity=ValidationSeverity.WARNING,
                        code="unknown_field"
                    ))

        return messages




    def _validate_metadata(
        self,
        element: TrackedElement,
        context: ProcessingContext
    ) -> Optional[ValidationResult]:
        """
        Validate element metadata according to schema rules.

        Args:
            element: Element being validated
            context: Processing context

        Returns:
            Optional[ValidationResult]: Validation result if successful
        """
        try:
            # Check cache first
            cache_key = f"metadata_validation_{element.id}_{context.state_info.phase.value}"
            if cached := self.cache.get(cache_key, CacheEntryType.VALIDATION):
                return cached

            # Check for config manager
            if not self._config_manager:
                self.logger.warning("Config manager not initialized, skipping metadata validation")
                return None

            # Get validation rules
            rules = self._config_manager.get_metadata_rules(
                phase=context.state_info.phase,
                element_type=element.type,
                context=context
            )

            messages = []

            # Validate required fields
            if required := rules.get('required_fields', []):
                for field in required:
                    if field not in element.metadata:
                        messages.append(ValidationMessage(
                            path=f"metadata.{field}",
                            message=f"Missing required metadata field: {field}",
                            severity=ValidationSeverity.ERROR,
                            code="missing_metadata_field"
                        ))

            # Validate field values
            for field, value in element.metadata.items():
                if field_rules := rules.get(field):
                    field_validation = FieldValidationRule(
                        field_type=field_rules.get('type', 'string'),
                        required=field_rules.get('required', False),
                        min_length=field_rules.get('min_length'),
                        max_length=field_rules.get('max_length'),
                        pattern=field_rules.get('pattern'),
                        allowed_values=field_rules.get('allowed_values'),
                        min_items=field_rules.get('min_items'),
                        max_items=field_rules.get('max_items'),
                        min_value=field_rules.get('min_value'),
                        max_value=field_rules.get('max_value')
                    )
                    messages.extend(self._validate_field(field, value, field_validation))

            # Create result
            result = ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

            # Cache result
            self.cache.set(
                key=cache_key,
                data=result,
                entry_type=CacheEntryType.VALIDATION,
                element_type=element.type,
                phase=context.state_info.phase
            )

            return result

        except Exception as e:
            self.logger.error(f"Error validating metadata: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="metadata",
                    message=f"Metadata validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="metadata_validation_error"
                )]
            )



    def validate_context(
            self,
            context: ProcessingContext,
            validation_pipeline: Optional[ValidationPipeline] = None
        ) -> ValidationResult:
            """
            Validate processing context using configurable pipeline.

            Args:
                context: Context to validate
                validation_pipeline: Optional custom validation pipeline

            Returns:
                ValidationResult with validation status
            """
            try:
                messages = []
                pipeline = validation_pipeline or self._get_default_context_pipeline()

                # Core validations - get messages from ValidationResults
                core_result = self._validate_core_context(context)
                messages.extend(core_result.messages)

                # Navigation validation
                nav_result = self._validate_navigation(context.navigation)
                messages.extend(nav_result.messages)

                # Metadata state validation
                metadata_result = self._validate_metadata_state(
                    context.metadata_state,
                    context.key_refs
                )
                messages.extend(metadata_result.messages)

                # Relationship validation
                rel_result = self._validate_relationships(
                    context.relationships,
                    context.scope
                )
                messages.extend(rel_result.messages)

                # Run custom pipeline validations
                if pipeline and pipeline.is_active():
                    pipeline_result = pipeline.validate(context)
                    messages.extend(pipeline_result.messages)

                return ValidationResult(
                    is_valid=not any(
                        msg.severity == ValidationSeverity.ERROR for msg in messages
                    ),
                    messages=messages
                )

            except Exception as e:
                self.logger.error(f"Context validation error: {str(e)}")
                return self._create_error_result("context_validation_error", str(e))

    def validate_processing_rule(
        self,
        rule: ProcessingRule,
        context: Optional[ProcessingContext] = None,
        validation_patterns: Optional[Dict[str, ValidationPattern]] = None
    ) -> ValidationResult:
        """
        Validate processing rule using patterns and context.

        Args:
            rule: Rule to validate
            context: Optional processing context
            validation_patterns: Optional custom validation patterns

        Returns:
            ValidationResult with validation status
        """
        try:
            messages = []
            patterns = validation_patterns or self._patterns

            # Validate rule structure
            messages.extend(self._validate_rule_structure(rule))

            # Validate rule type and operation
            messages.extend(self._validate_rule_type_and_operation(rule))

            # Validate rule configuration
            if rule.config:
                messages.extend(self._validate_rule_config(
                    rule.config,
                    patterns
                ))

            # Validate conditions if present
            if rule.conditions:
                messages.extend(self._validate_conditions(
                    rule.conditions,
                    context
                ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Rule validation error: {str(e)}")
            return self._create_error_result("rule_validation_error", str(e))

    def _validate_conditions(
        self,
        conditions: Dict[str, Any],
        context: Optional[ProcessingContext]
    ) -> List[ValidationMessage]:
        """
        Validate rule conditions against context.

        Args:
            conditions: Conditions to validate
            context: Optional context for condition evaluation

        Returns:
            List of validation messages
        """
        messages = []

        for key, condition in conditions.items():
            if isinstance(condition, dict):
                messages.extend(self._validate_complex_condition(
                    key,
                    condition,
                    context
                ))
            else:
                messages.extend(self._validate_simple_condition(
                    key,
                    condition
                ))

        return messages

    def _validate_complex_condition(
        self,
        key: str,
        condition: Dict[str, Any],
        context: Optional[ProcessingContext]
    ) -> List[ValidationMessage]:
        """Validate complex (dictionary) condition."""
        messages = []

        # Validate context path if present
        if context_path := condition.get("context_path"):
            if not self._validate_context_path(context_path, context):
                messages.append(ValidationMessage(
                    path=f"conditions.{key}.context_path",
                    message=f"Invalid context path: {context_path}",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_context_path"
                ))

        # Validate operator if present
        if operator := condition.get("operator"):
            if not self._is_valid_operator(operator):
                messages.append(ValidationMessage(
                    path=f"conditions.{key}.operator",
                    message=f"Invalid operator: {operator}",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_operator"
                ))

        return messages

    def _validate_context_path(
        self,
        path: str,
        context: Optional[ProcessingContext]
    ) -> bool:
        """
        Validate context path exists in context.

        Args:
            path: Path to validate
            context: Context to validate against

        Returns:
            bool: True if path is valid
        """
        if not context:
            return True

        try:
            parts = path.split('.')
            current = context

            for part in parts:
                if not hasattr(current, part):
                    return False
                current = getattr(current, part)

            return True
        except Exception:
            return False

    def _validate_field(
        self,
        field: str,
        value: Any,
        rule: FieldValidationRule
    ) -> List[ValidationMessage]:
        """
        Validate field value against rule.

        Args:
            field: Field name
            value: Field value
            rule: Field validation rule

        Returns:
            List[ValidationMessage]: Validation messages
        """
        messages = []

        try:
            # Type validation
            if not self._validate_field_type(value, rule.field_type):
                messages.append(self._create_validation_error(
                    field=field,
                    message=f"Invalid type for {field}, expected {rule.field_type}",
                    code="invalid_field_type"
                ))
                return messages

            # Type-specific validations with proper error codes
            if rule.field_type == "string" and isinstance(value, str):
                if rule.min_length and len(value) < rule.min_length:
                    messages.append(self._create_validation_error(
                        field=field,
                        message=f"Length must be at least {rule.min_length}",
                        code="min_length_violation"
                    ))
                if rule.max_length and len(value) > rule.max_length:
                    messages.append(self._create_validation_error(
                        field=field,
                        message=f"Length must not exceed {rule.max_length}",
                        code="max_length_violation"
                    ))
                if rule.pattern and not re.match(rule.pattern, value):
                    messages.append(self._create_validation_error(
                        field=field,
                        message="Value does not match required pattern",
                        code="pattern_mismatch"
                    ))

            # Number validation
            if rule.field_type in ("number", "integer") and isinstance(value, (int, float)):
                if rule.min_value is not None and value < rule.min_value:
                    messages.append(self._create_validation_error(
                        field=field,
                        message=f"Value must be at least {rule.min_value}",
                        code="min_value_violation"
                    ))
                if rule.max_value is not None and value > rule.max_value:
                    messages.append(self._create_validation_error(
                        field=field,
                        message=f"Value must not exceed {rule.max_value}",
                        code="max_value_violation"
                    ))

            # Array validation
            if rule.field_type == "array" and isinstance(value, (list, tuple)):
                if rule.min_items and len(value) < rule.min_items:
                    messages.append(self._create_validation_error(
                        field=field,
                        message=f"Array must contain at least {rule.min_items} items",
                        code="min_items_violation"
                    ))
                if rule.max_items and len(value) > rule.max_items:
                    messages.append(self._create_validation_error(
                        field=field,
                        message=f"Array must not exceed {rule.max_items} items",
                        code="max_items_violation"
                    ))

            # Allowed values validation
            if rule.allowed_values and value not in rule.allowed_values:
                messages.append(self._create_validation_error(
                    field=field,
                    message=f"Value not in allowed values: {rule.allowed_values}",
                    code="invalid_allowed_value"
                ))

            return messages

        except Exception as e:
            self.logger.error(f"Field validation error: {str(e)}")
            return [self._create_validation_error(
                field=field,
                message=f"Validation error: {str(e)}",
                code="validation_error"
            )]

    def _create_error_result(self, code: str, message: str) -> ValidationResult:
        """Create error validation result."""
        return ValidationResult(
            is_valid=False,
            messages=[ValidationMessage(
                path="",
                message=message,
                severity=ValidationSeverity.ERROR,
                code=code
            )]
        )

    def _get_default_context_pipeline(self) -> ValidationPipeline:
        """Get default validation pipeline for context validation."""
        if "context" not in self._pipelines:
            pipeline = ValidationPipeline()
            pipeline.add_validator("core", self._validate_core_context)
            pipeline.add_validator("navigation", self._validate_navigation)
            pipeline.add_validator("metadata", self._validate_metadata_state)
            pipeline.add_validator("relationships", self._validate_relationships)
            self._pipelines["context"] = pipeline
        return self._pipelines["context"]

    def _validate_core_context(
        self,
        context: ProcessingContext
    ) -> ValidationResult:
        """Validate core context structure and required fields."""
        messages = []

        # Validate required attributes
        if not context.context_id:
            messages.append(self._create_validation_error(
                "context_id",
                "Missing context ID",
                code="missing_context_id"
            ))

        if not context.element_id:
            messages.append(self._create_validation_error(
                "element_id",
                "Missing element ID",
                code="missing_element_id"
            ))

        # Validate element type
        if not isinstance(context.element_type, ElementType):
            messages.append(self._create_validation_error(
                "element_type",
                "Invalid element type",
                code="invalid_element_type"
            ))

        return ValidationResult(
            is_valid=not any(msg.severity == ValidationSeverity.ERROR for msg in messages),
            messages=messages
        )

    def _validate_navigation(
        self,
        navigation: NavigationContext
    ) -> ValidationResult:
        """Validate navigation context structure."""
        messages = []

        # Validate path consistency
        if navigation.parent_id:
            if not navigation.path:
                messages.append(self._create_validation_error(
                    "navigation.path",
                    "Missing path for parent context",
                    code="missing_path"
                ))
            elif navigation.parent_id not in navigation.path:
                messages.append(self._create_validation_error(
                    "navigation.path",
                    "Parent ID not in path",
                    code="invalid_path"
                ))

        # Validate level consistency
        if navigation.parent_id and navigation.level < 1:
            messages.append(self._create_validation_error(
                "navigation.level",
                "Invalid level for child context",
                code="invalid_level"
            ))

        return ValidationResult(
            is_valid=not any(msg.severity == ValidationSeverity.ERROR for msg in messages),
            messages=messages
        )

    def _validate_metadata_state(
        self,
        metadata_state: MetadataState,
        key_refs: Set[str]
    ) -> ValidationResult:
        """Validate metadata state consistency."""
        messages = []

        # Validate key reference consistency
        if metadata_state.key_references:
            missing_refs = set(metadata_state.key_references) - key_refs
            if missing_refs:
                messages.append(self._create_validation_error(
                    "metadata_state.key_references",
                    f"Missing key references: {', '.join(missing_refs)}",
                    code="missing_key_refs"
                ))

        # Validate state consistency
        if metadata_state.cached and not metadata_state.metadata_refs:
            messages.append(self._create_validation_error(
                "metadata_state",
                "Cached state missing metadata references",
                code="inconsistent_cache_state",
                severity=ValidationSeverity.WARNING  # Note: Using WARNING for cache inconsistency
            ))

        return ValidationResult(
            is_valid=not any(msg.severity == ValidationSeverity.ERROR for msg in messages),
            messages=messages
        )

    def _validate_relationships(
        self,
        relationships: List[ContentRelationship],
        scope: ContentScope
    ) -> ValidationResult:
        """Validate content relationships."""
        messages = []

        for relationship in relationships:
            # Skip if no relationship provided
            if not relationship:
                continue

            # Validate scope compatibility
            if not self._validate_scope_compatibility(scope, relationship.scope):
                messages.append(self._create_validation_error(
                    "relationships",
                    f"Incompatible scope: {relationship.scope.value}",
                    code="incompatible_scope"
                ))

            # Validate relationship type consistency
            if relationship.relation_type == ContentRelationType.CHILD and scope == ContentScope.EXTERNAL:
                messages.append(self._create_validation_error(
                    "relationships",
                    "External scope cannot have child relationships",
                    code="invalid_relationship_type"
                ))

        return ValidationResult(
            is_valid=not any(msg.severity == ValidationSeverity.ERROR for msg in messages),
            messages=messages
        )

    def _validate_rule_structure(self, rule: ProcessingRule) -> List[ValidationMessage]:
        """Validate basic rule structure."""
        messages = []

        # Validate required fields
        required_fields = {
            "rule_id": str,
            "rule_type": ProcessingRuleType,
            "element_type": ElementType,
            "config": dict
        }

        for field, expected_type in required_fields.items():
            value = getattr(rule, field, None)
            if value is None:
                messages.append(ValidationMessage(
                    path=field,
                    message=f"Missing required field: {field}",
                    severity=ValidationSeverity.ERROR,
                    code="missing_required_field"
                ))
            elif not isinstance(value, expected_type):
                messages.append(ValidationMessage(
                    path=field,
                    message=f"Invalid type for {field}: expected {expected_type.__name__}",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_field_type"
                ))

        return messages

    def _validate_rule_type_and_operation(self, rule: ProcessingRule) -> List[ValidationMessage]:
        """Validate rule type and operation compatibility."""
        messages = []

        # Validate rule type
        if not ProcessingRuleType.validate_rule_type(rule.rule_type.value):
            messages.append(ValidationMessage(
                path="rule_type",
                message=f"Invalid rule type: {rule.rule_type.value}",
                severity=ValidationSeverity.ERROR,
                code="invalid_rule_type"
            ))
            return messages  # Early return if rule type is invalid

        # Validate operation if config exists
        if rule.config:
            operation = rule.config.get("operation")
            if not operation:
                messages.append(ValidationMessage(
                    path="config.operation",
                    message="Missing required operation field",
                    severity=ValidationSeverity.ERROR,
                    code="missing_operation"
                ))
            else:
                # Validate operation is allowed for rule type
                allowed_operations = ProcessingRuleType.get_allowed_operations(rule.rule_type.value)
                if operation not in allowed_operations:
                    messages.append(ValidationMessage(
                        path="config.operation",
                        message=f"Invalid operation {operation} for rule type {rule.rule_type.value}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_operation"
                    ))

        return messages

    def _validate_rule_config(
        self,
        config: Dict[str, Any],
        patterns: Dict[str, ValidationPattern]
    ) -> List[ValidationMessage]:
        """Validate rule configuration against patterns."""
        messages = []

        # Validate required target field
        if not config.get("target"):
            messages.append(ValidationMessage(
                path="config.target",
                message="Missing required target field",
                severity=ValidationSeverity.ERROR,
                code="missing_target"
            ))

        # Validate action configuration if present
        if action_config := config.get("action"):
            operation = config.get("operation")
            if isinstance(operation, str):  # Type check added
                messages.extend(
                    self._validate_action_config(action_config, operation)
                )

        # Apply validation patterns
        for pattern_id, pattern in patterns.items():
            if pattern.pattern_id in config:
                result = pattern.validate(config[pattern_id])
                messages.extend(result.messages)

        return messages

    def _validate_simple_condition(
        self,
        key: str,
        condition: Any
    ) -> List[ValidationMessage]:
        """Validate simple (non-dictionary) condition."""
        messages = []

        # Validate condition value type
        if not isinstance(condition, (str, bool, int, float)):
            messages.append(ValidationMessage(
                path=f"conditions.{key}",
                message="Invalid condition value type",
                severity=ValidationSeverity.ERROR,
                code="invalid_condition_type"
            ))

        return messages




    def _validate_metadata_content(
        self,
        metadata: Dict[str, Any],
        rules: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate metadata content against rules.

        Args:
            metadata: Metadata content to validate
            rules: Validation rules to apply

        Returns:
            ValidationResult: Validation result
        """
        messages = []

        try:
            # Validate required fields
            if required_fields := rules.get('required_fields', []):
                for field in required_fields:
                    if field not in metadata:
                        messages.append(ValidationMessage(
                            path=field,
                            message=f"Missing required field: {field}",
                            severity=ValidationSeverity.ERROR,
                            code="missing_required_field"
                        ))

            # Validate field types and values
            for field, value in metadata.items():
                if field_rules := rules.get(field):
                    # Type validation
                    expected_type = field_rules.get('type')
                    if expected_type and not self._validate_field_type(value, expected_type):
                        messages.append(ValidationMessage(
                            path=field,
                            message=f"Invalid type for field {field}: expected {expected_type}",
                            severity=ValidationSeverity.ERROR,
                            code="invalid_field_type"
                        ))
                        continue

                    # Value validation
                    if validation_rules := field_rules.get('validation', {}):
                        field_messages = self._validate_field_value(
                            field,
                            value,
                            validation_rules
                        )
                        messages.extend(field_messages)

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating metadata content: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Metadata validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def _validate_field_type(self, value: Any, expected_type: str) -> bool:
        """Validate field value type."""
        type_mapping = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'array': list,
            'object': dict
        }

        expected_types = type_mapping.get(expected_type)
        if not expected_types:
            return True  # Skip validation for unknown types

        return isinstance(value, expected_types)

    def _validate_field_value(
        self,
        field: str,
        value: Any,
        rules: Dict[str, Any]
    ) -> List[ValidationMessage]:
        """Validate field value against rules."""
        messages = []

        # String validations
        if isinstance(value, str):
            if min_length := rules.get('min_length'):
                if len(value) < min_length:
                    messages.append(ValidationMessage(
                        path=field,
                        message=f"Field {field} must be at least {min_length} characters",
                        severity=ValidationSeverity.ERROR,
                        code="min_length"
                    ))

            if pattern := rules.get('pattern'):
                if not re.match(pattern, value):
                    messages.append(ValidationMessage(
                        path=field,
                        message=f"Field {field} must match pattern: {pattern}",
                        severity=ValidationSeverity.ERROR,
                        code="pattern_mismatch"
                    ))

        # Array validations
        elif isinstance(value, list):
            if min_items := rules.get('min_items'):
                if len(value) < min_items:
                    messages.append(ValidationMessage(
                        path=field,
                        message=f"Field {field} must have at least {min_items} items",
                        severity=ValidationSeverity.ERROR,
                        code="min_items"
                    ))

        # Value constraints
        if allowed_values := rules.get('allowed_values'):
            if value not in allowed_values:
                messages.append(ValidationMessage(
                    path=field,
                    message=f"Invalid value for field {field}",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_value"
                ))

        return messages

    def _validate_required_fields(
        self,
        metadata: Dict[str, Any],
        required: List[str],
        messages: List[ValidationMessage]
    ) -> None:
        """Validate required fields are present."""
        missing = [field for field in required if field not in metadata]
        if missing:
            messages.append(ValidationMessage(
                path="fields",
                message=f"Missing required fields: {', '.join(missing)}",
                severity=ValidationSeverity.ERROR,
                code="missing_required_fields"
            ))


    def validate_configuration_set(
        self,
        configs: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate complete configuration set.

        Args:
            configs: Dictionary of all configurations to validate
                Expected keys: "processing", "metadata", "validation", "features"

        Returns:
            ValidationResult: Combined validation result
        """
        messages = []

        try:
            # Validate processing configuration
            if processing_config := configs.get("processing"):
                processing_result = self.validate_processing_config(processing_config)
                messages.extend(processing_result.messages)

            # Validate metadata configuration
            if metadata_config := configs.get("metadata"):
                metadata_result = self.validate_metadata_rules(metadata_config)
                messages.extend(metadata_result.messages)

            # Validate feature configuration
            if feature_config := configs.get("features"):
                feature_result = self.validate_feature_registry(feature_config.get("features", {}))
                messages.extend(feature_result.messages)

            # Validate validation patterns
            if validation_config := configs.get("validation"):
                validation_result = self.validate_validation_patterns(validation_config)
                messages.extend(validation_result.messages)

            # Cross-validate configuration dependencies
            dependency_result = self._validate_config_dependencies(configs)
            messages.extend(dependency_result.messages)

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating configuration set: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Configuration validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def _validate_config_dependencies(
        self,
        configs: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate dependencies between different configurations.

        Args:
            configs: Complete configuration set

        Returns:
            ValidationResult: Validation result
        """
        messages = []

        try:
            # Validate feature dependencies in processing rules
            if (processing_config := configs.get("processing")) and (
                feature_config := configs.get("features")
            ):
                features = feature_config.get("features", {})
                rules = processing_config.get("rules", {})

                for rule_type, type_rules in rules.items():
                    for rule_id, rule in type_rules.items():
                        if required_features := rule.get("required_features", []):
                            for feature in required_features:
                                if feature not in features:
                                    messages.append(ValidationMessage(
                                        path=f"rules.{rule_type}.{rule_id}",
                                        message=f"Rule requires undefined feature: {feature}",
                                        severity=ValidationSeverity.ERROR,
                                        code="missing_required_feature"
                                    ))

            # Validate metadata references in validation patterns
            if (validation_config := configs.get("validation")) and (
                metadata_config := configs.get("metadata")
            ):
                patterns = validation_config.get("patterns", {})
                metadata_fields = set(metadata_config.get("fields", {}).keys())

                for pattern_id, pattern in patterns.items():
                    if metadata_refs := pattern.get("metadata_refs", []):
                        for ref in metadata_refs:
                            if ref not in metadata_fields:
                                messages.append(ValidationMessage(
                                    path=f"patterns.{pattern_id}",
                                    message=f"Pattern references undefined metadata field: {ref}",
                                    severity=ValidationSeverity.ERROR,
                                    code="invalid_metadata_reference"
                                ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating config dependencies: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="dependencies",
                    message=f"Dependency validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="dependency_validation_error"
                )]
            )

    def validate_processing_config(
        self,
        config: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate processing configuration.

        Args:
            config: Processing configuration to validate

        Returns:
            ValidationResult: Validation result
        """
        messages = []

        try:
            # Validate rules structure
            if rules := config.get("rules"):
                if not isinstance(rules, dict):
                    messages.append(ValidationMessage(
                        path="rules",
                        message="Rules must be a dictionary",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_rules_type"
                    ))
                else:
                    # Validate each rule type
                    for rule_type, type_rules in rules.items():
                        if not ProcessingRuleType.validate_rule_type(rule_type):
                            messages.append(ValidationMessage(
                                path=f"rules.{rule_type}",
                                message=f"Invalid rule type: {rule_type}",
                                severity=ValidationSeverity.ERROR,
                                code="invalid_rule_type"
                            ))

                        # Validate individual rules
                        for rule_id, rule in type_rules.items():
                            rule_messages = self._validate_rule_definition(
                                rule_id,
                                rule,
                                rule_type
                            )
                            messages.extend(rule_messages)

            # Validate type mapping
            if type_mapping := config.get("type_mapping"):
                if not isinstance(type_mapping, dict):
                    messages.append(ValidationMessage(
                        path="type_mapping",
                        message="Type mapping must be a dictionary",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_mapping_type"
                    ))
                else:
                    for element_type, mapping in type_mapping.items():
                        if not isinstance(mapping, str):
                            messages.append(ValidationMessage(
                                path=f"type_mapping.{element_type}",
                                message="Mapping must be a string",
                                severity=ValidationSeverity.ERROR,
                                code="invalid_mapping_value"
                            ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating processing config: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Processing configuration validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def validate_validation_patterns(
        self,
        config: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate validation patterns configuration.

        Args:
            config: Validation patterns configuration to validate

        Returns:
            ValidationResult: Validation result
        """
        messages = []

        try:
            # Validate patterns structure
            if patterns := config.get("patterns"):
                if not isinstance(patterns, dict):
                    messages.append(ValidationMessage(
                        path="patterns",
                        message="Patterns must be a dictionary",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_patterns_type"
                    ))
                else:
                    for pattern_id, pattern in patterns.items():
                        pattern_messages = self._validate_pattern_definition(
                            pattern_id,
                            pattern
                        )
                        messages.extend(pattern_messages)

            # Validate pattern groups
            if groups := config.get("pattern_groups"):
                if not isinstance(groups, dict):
                    messages.append(ValidationMessage(
                        path="pattern_groups",
                        message="Pattern groups must be a dictionary",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_groups_type"
                    ))
                else:
                    # Validate group references
                    for group_id, patterns in groups.items():
                        if not isinstance(patterns, list):
                            messages.append(ValidationMessage(
                                path=f"pattern_groups.{group_id}",
                                message="Group patterns must be a list",
                                severity=ValidationSeverity.ERROR,
                                code="invalid_group_patterns"
                            ))
                        else:
                            # Validate pattern references
                            for pattern in patterns:
                                if not self._pattern_exists(pattern, config):
                                    messages.append(ValidationMessage(
                                        path=f"pattern_groups.{group_id}",
                                        message=f"Referenced pattern does not exist: {pattern}",
                                        severity=ValidationSeverity.ERROR,
                                        code="invalid_pattern_reference"
                                    ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating validation patterns: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Validation patterns validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def validate_feature_registry(
        self,
        features: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate feature registry configuration.

        Args:
            features: Feature registry to validate

        Returns:
            ValidationResult: Validation result
        """
        messages = []

        try:
            for feature_name, feature_def in features.items():
                # Validate feature name format
                if not re.match(r'^[a-z][a-z0-9_]*$', feature_name):
                    messages.append(ValidationMessage(
                        path=f"features.{feature_name}",
                        message="Invalid feature name format",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_feature_name"
                    ))

                # Validate feature definition
                if not isinstance(feature_def, dict):
                    messages.append(ValidationMessage(
                        path=f"features.{feature_name}",
                        message="Feature definition must be a dictionary",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_feature_definition"
                    ))
                    continue

                # Validate required fields
                required_fields = {
                    "enabled": bool,
                    "scope": str
                }

                for field, expected_type in required_fields.items():
                    if field not in feature_def:
                        messages.append(ValidationMessage(
                            path=f"features.{feature_name}",
                            message=f"Missing required field: {field}",
                            severity=ValidationSeverity.ERROR,
                            code="missing_required_field"
                        ))
                    elif not isinstance(feature_def[field], expected_type):
                        messages.append(ValidationMessage(
                            path=f"features.{feature_name}.{field}",
                            message=f"Invalid type for {field}",
                            severity=ValidationSeverity.ERROR,
                            code="invalid_field_type"
                        ))

                # Validate scope value
                if scope := feature_def.get("scope"):
                    if not FeatureScope.validate_scope(scope):
                        messages.append(ValidationMessage(
                            path=f"features.{feature_name}.scope",
                            message=f"Invalid feature scope: {scope}",
                            severity=ValidationSeverity.ERROR,
                            code="invalid_scope"
                        ))

                # Validate dependencies and conflicts
                for field in ["dependencies", "conflicts"]:
                    if field_value := feature_def.get(field):
                        if not isinstance(field_value, list):
                            messages.append(ValidationMessage(
                                path=f"features.{feature_name}.{field}",
                                message=f"{field} must be a list",
                                severity=ValidationSeverity.ERROR,
                                code=f"invalid_{field}_type"
                            ))
                        else:
                            # Validate referenced features exist
                            for ref in field_value:
                                if ref not in features:
                                    messages.append(ValidationMessage(
                                        path=f"features.{feature_name}.{field}",
                                        message=f"Referenced feature does not exist: {ref}",
                                        severity=ValidationSeverity.ERROR,
                                        code=f"invalid_{field}_reference"
                                    ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating feature registry: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Feature registry validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def _validate_rule_definition(
        self,
        rule_id: str,
        rule: Dict[str, Any],
        rule_type: str
    ) -> List[ValidationMessage]:
        """Validate individual rule definition."""
        messages = []

        # Validate required fields
        required_fields = {
            "element_type": str,
            "operation": str,
            "target": str,
            "action": dict
        }

        for field, expected_type in required_fields.items():
            if field not in rule:
                messages.append(ValidationMessage(
                    path=f"rules.{rule_type}.{rule_id}",
                    message=f"Missing required field: {field}",
                    severity=ValidationSeverity.ERROR,
                    code="missing_required_field"
                ))
            elif not isinstance(rule[field], expected_type):
                messages.append(ValidationMessage(
                    path=f"rules.{rule_type}.{rule_id}.{field}",
                    message=f"Invalid type for {field}",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_field_type"
                ))

        # Validate operation is allowed for rule type
        if operation := rule.get("operation"):
            allowed_operations = ProcessingRuleType.get_allowed_operations(rule_type)
            if operation not in allowed_operations:
                messages.append(ValidationMessage(
                    path=f"rules.{rule_type}.{rule_id}.operation",
                    message=f"Invalid operation for rule type: {operation}",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_operation"
                ))

        return messages

    def _validate_pattern_definition(
        self,
        pattern_id: str,
        pattern: Dict[str, Any]
    ) -> List[ValidationMessage]:
        """Validate pattern definition."""
        messages = []

        # Validate required fields
        if pattern_type := pattern.get("type", "regex"):
            if pattern_type == "regex":
                if "pattern" not in pattern:
                    messages.append(ValidationMessage(
                        path=f"patterns.{pattern_id}",
                        message="Missing pattern for regex type",
                        severity=ValidationSeverity.ERROR,
                        code="missing_pattern"
                    ))
            elif pattern_type == "schema":
                if "schema" not in pattern:
                    messages.append(ValidationMessage(
                        path=f"patterns.{pattern_id}",
                        message="Missing schema for schema type",
                        severity=ValidationSeverity.ERROR,
                        code="missing_schema"
                    ))

        # Validate metadata if present
        if metadata := pattern.get("metadata"):
            if not isinstance(metadata, dict):
                messages.append(ValidationMessage(
                    path=f"patterns.{pattern_id}.metadata",
                    message="Metadata must be a dictionary",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_metadata_type"
                ))

        return messages

    def _pattern_exists(
        self,
        pattern_ref: str,
        config: Dict[str, Any]
    ) -> bool:
        """Check if referenced pattern exists."""
        patterns = config.get("patterns", {})
        parts = pattern_ref.split('.')

        current = patterns
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return False
            current = current[part]

        return True

    def _validate_scope_compatibility(
        self,
        source_scope: ContentScope,
        target_scope: ContentScope
    ) -> bool:
        """Validate scope compatibility between source and target."""
        if source_scope == ContentScope.LOCAL:
            return target_scope in {ContentScope.LOCAL, ContentScope.PEER}
        if source_scope == ContentScope.PEER:
            return True
        if source_scope == ContentScope.EXTERNAL:
            return target_scope == ContentScope.EXTERNAL
        return False

    def _handle_state_change(self, **event_data: Any) -> None:
        """Handle element state changes for validation tracking."""
        try:
            element_id = event_data.get("element_id")
            state_info = event_data.get("state_info")

            if element_id and state_info:
                # Clear validation cache on state changes
                self.cache.invalidate_by_pattern(
                    f"validation_{element_id}*"
                )

                # Update validation tracking
                if state_info.state == ProcessingState.ERROR:
                    if result := self._validation_results.get(element_id):
                        result.messages.append(ValidationMessage(
                            path="state",
                            message=f"Processing error: {state_info.error_message}",
                            severity=ValidationSeverity.ERROR,
                            code="processing_error"
                        ))

        except Exception as e:
            self.logger.error(f"Error handling state change: {str(e)}")

    def get_validation_result(
        self,
        element_id: str,
        phase: Optional[ProcessingPhase] = None
    ) -> Optional[ValidationResult]:
        """Get validation result for element."""
        try:
            cache_key = f"validation_{element_id}"
            if phase:
                cache_key = f"{cache_key}_{phase.value}"

            return self.cache.get(cache_key, CacheEntryType.VALIDATION)

        except Exception as e:
            self.logger.error(f"Error getting validation result: {str(e)}")
            return None

    def clear_validation_cache(self) -> None:
        """Clear validation cache."""
        try:
            self.cache.invalidate_by_pattern("validation_*")
            self._validation_results.clear()
            self._active_validations.clear()
            self._validation_cache.clear()
        except Exception as e:
            self.logger.error(f"Error clearing validation cache: {str(e)}")

    ############################
    # Field validation methods #
    ############################

    def _validate_string_field(
        self,
        field: str,
        value: str,
        rule: FieldValidationRule
    ) -> List[ValidationMessage]:
        """Validate string field value."""
        messages = []

        if rule.min_length and len(value) < rule.min_length:
            messages.append(self._create_validation_error(
                field,
                f"Length must be at least {rule.min_length}",
                "min_length_violation"
            ))

        if rule.max_length and len(value) > rule.max_length:
            messages.append(self._create_validation_error(
                field,
                f"Length must not exceed {rule.max_length}",
                "max_length_violation"
            ))

        if rule.pattern and not re.match(rule.pattern, value):
            messages.append(self._create_validation_error(
                field,
                "Value does not match required pattern",
                "pattern_mismatch"
            ))

        return messages

    def _validate_number_field(
        self,
        field: str,
        value: Union[int, float],
        rule: FieldValidationRule
    ) -> List[ValidationMessage]:
        """Validate numeric field value."""
        messages = []

        if rule.min_value is not None and value < rule.min_value:
            messages.append(self._create_validation_error(
                field,
                f"Value must be at least {rule.min_value}",
                "min_value_violation"
            ))

        if rule.max_value is not None and value > rule.max_value:
            messages.append(self._create_validation_error(
                field,
                f"Value must not exceed {rule.max_value}",
                "max_value_violation"
            ))

        return messages

    def _validate_array_field(
        self,
        field: str,
        value: List[Any],
        rule: FieldValidationRule
    ) -> List[ValidationMessage]:
        """Validate array field value."""
        messages = []

        if rule.min_items and len(value) < rule.min_items:
            messages.append(self._create_validation_error(
                field,
                f"Array must contain at least {rule.min_items} items",
                "min_items_violation"
            ))

        if rule.max_items and len(value) > rule.max_items:
            messages.append(self._create_validation_error(
                field,
                f"Array must not exceed {rule.max_items} items",
                "max_items_violation"
            ))

        return messages


    ##########################
    # Error handling methods #
    ##########################

    def _create_type_error(self, field: str, expected_type: str) -> ValidationMessage:
        """Create type validation error message."""
        return ValidationMessage(
            path=field,
            message=f"Invalid type for {field}: expected {expected_type}",
            severity=ValidationSeverity.ERROR,
            code="invalid_type"
        )

    def _create_value_error(
        self,
        field: str,
        allowed_values: List[Any]
    ) -> ValidationMessage:
        """Create value validation error message."""
        return ValidationMessage(
            path=field,
            message=f"Invalid value for {field}. Allowed values: {allowed_values}",
            severity=ValidationSeverity.ERROR,
            code="invalid_value"
        )


    def _create_validation_error(
        self,
        field: str,
        message: str,
        code: str,  # Make code required
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ) -> ValidationMessage:
        """Create generic validation error message with consistent error tracking."""
        return ValidationMessage(
            path=field,
            message=message,
            severity=severity,
            code=code
        )

    ######################################################
    # Pattern, rule, and core context validation methods #
    ######################################################

    def _validate_patterns(
        self,
        element: TrackedElement
    ) -> List[ValidationMessage]:
        """
        Validate element against defined validation patterns.
        """
        messages = []

        try:
            # Get patterns for element type from our stored patterns
            if patterns := self._validation_patterns.get(element.type.value, {}):
                # Each validation pattern is accessed directly
                for pattern_name in patterns:
                    pattern = patterns[pattern_name]
                    if isinstance(pattern, ValidationPattern):
                        validation_result = pattern.validate(element)
                        if not validation_result.is_valid:
                            messages.extend(validation_result.messages)

        except Exception as e:
            messages.append(self._create_validation_error(
                "patterns",
                f"Pattern validation error: {str(e)}",
                code="pattern_validation_error"
            ))

        return messages

    def _validate_against_rules(
        self,
        element: TrackedElement,
        rules: Dict[str, Any]
    ) -> List[ValidationMessage]:
        """
        Validate element against processing rules.
        """
        messages = []

        try:
            # Check required attributes
            if required := rules.get("required_attributes", []):
                for attr in required:
                    if attr not in element.metadata:
                        messages.append(self._create_validation_error(
                            f"attributes.{attr}",
                            f"Missing required attribute: {attr}",
                            "missing_required_attribute"
                        ))

            # Validate attribute patterns
            if patterns := rules.get("attribute_patterns", {}):
                for attr, pattern in patterns.items():
                    if value := element.metadata.get(attr):
                        if not re.match(pattern, str(value)):
                            messages.append(self._create_validation_error(
                                f"attributes.{attr}",
                                f"Invalid format for {attr}",
                                "invalid_attribute_pattern"
                            ))

            # Validate allowed values
            if allowed_values := rules.get("allowed_values", {}):
                for attr, allowed in allowed_values.items():
                    if value := element.metadata.get(attr):
                        if value not in allowed:
                            messages.append(self._create_validation_error(
                                f"attributes.{attr}",
                                f"Invalid value for {attr}",
                                "invalid_attribute_value"
                            ))

        except Exception as e:
            messages.append(self._create_validation_error(
                "rules",
                f"Rule validation error: {str(e)}",
                "rule_validation_error"
            ))

        return messages

    def _is_valid_operator(self, operator: str) -> bool:
        """Check if operator is valid."""
        valid_operators = {
            "eq",      # Equal
            "ne",      # Not equal
            "gt",      # Greater than
            "lt",      # Less than
            "gte",     # Greater than or equal
            "lte",     # Less than or equal
            "in",      # In list
            "nin",     # Not in list
            "contains", # Contains
            "startswith", # Starts with
            "endswith",  # Ends with
            "matches"   # Regex match
        }
        return operator in valid_operators

    def validate_metadata_rules(
        self,
        rules: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate metadata rules configuration.
        """
        messages = []

        try:
            # Validate required sections
            required_sections = ["processing_hierarchy", "validation_rules", "inheritance_rules"]
            for section in required_sections:
                if section not in rules:
                    messages.append(self._create_validation_error(
                        section,
                        f"Missing required section: {section}",
                        "missing_section"
                    ))

            # Validate processing hierarchy
            if hierarchy := rules.get("processing_hierarchy"):
                if not isinstance(hierarchy.get("order", []), list):
                    messages.append(self._create_validation_error(
                        "processing_hierarchy.order",
                        "Processing order must be a list",
                        "invalid_hierarchy_order"
                    ))

            # Validate validation rules
            if validation_rules := rules.get("validation_rules"):
                messages.extend(self._validate_rule_section(validation_rules))

            # Validate inheritance rules
            if inheritance_rules := rules.get("inheritance_rules"):
                messages.extend(self._validate_inheritance_rules(inheritance_rules))

        except Exception as e:
            messages.append(self._create_validation_error(
                "metadata_rules",
                f"Metadata rules validation error: {str(e)}",
                "metadata_rules_validation_error"
            ))

        return ValidationResult(
            is_valid=not any(msg.severity == ValidationSeverity.ERROR for msg in messages),
            messages=messages
        )

    def _validate_rule_section(
        self,
        rules: Dict[str, Any]
    ) -> List[ValidationMessage]:
        """Validate rules section of metadata configuration."""
        messages = []

        # Validate required fields
        required_fields = ["required_fields", "field_types", "validation_patterns"]
        for field in required_fields:
            if field not in rules:
                messages.append(self._create_validation_error(
                    f"validation_rules.{field}",
                    f"Missing required field: {field}",
                    "missing_required_field"
                ))

        # Validate field types
        if field_types := rules.get("field_types"):
            if not isinstance(field_types, dict):
                messages.append(self._create_validation_error(
                    "validation_rules.field_types",
                    "Field types must be a dictionary",
                    "invalid_field_types"
                ))

        # Validate validation patterns
        if patterns := rules.get("validation_patterns"):
            if not isinstance(patterns, dict):
                messages.append(self._create_validation_error(
                    "validation_rules.validation_patterns",
                    "Validation patterns must be a dictionary",
                    "invalid_validation_patterns"
                ))

        return messages

    def _validate_inheritance_rules(
        self,
        rules: Dict[str, Any]
    ) -> List[ValidationMessage]:
        """Validate inheritance rules configuration."""
        messages = []

        valid_strategies = {"merge", "override", "append"}

        for field, strategy in rules.items():
            if strategy not in valid_strategies:
                messages.append(self._create_validation_error(
                    f"inheritance_rules.{field}",
                    f"Invalid inheritance strategy: {strategy}",
                    "invalid_inheritance_strategy"
                ))

        return messages
