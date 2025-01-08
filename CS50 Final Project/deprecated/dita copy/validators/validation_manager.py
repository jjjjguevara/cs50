"""Centralized validation management for DITA processing pipeline."""

from typing import (
    Callable,
    Dict,
    Optional,
    Any,
    List,
    Set,
    Tuple,
    TYPE_CHECKING
)
from dataclasses import dataclass
from pathlib import Path
import re

# Type checking imports
if TYPE_CHECKING:
    from ..config.config_manager import ConfigManager
    from ..main.context_manager import ContextManager
    from ..metadata.metadata_manager import MetadataManager

# Import managers and utilities
from ..events.event_manager import EventManager, EventType
from ..cache.cache import ContentCache
from ..utils.logger import DITALogger
from .dtd_validator import DTDValidator

# Import core types
from ..types import (
    ElementType,
    ProcessingPhase,
    ProcessingState,
    ValidationState,
    ValidationResult,
    ValidationMessage,
    ValidationSeverity,
    ProcessingContext,
    ContentElement,
    CacheEntryType,
    ValidationPattern,
    RegexValidationPattern,
    SchemaValidationPattern
)



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
        """Add validator to pipeline."""
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

    def validate(
        self,
        value: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Run validation pipeline."""
        messages = []
        is_valid = True

        for name, validator in self.validators:
            try:
                result = validator(value, **context) if context else validator(value)
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

class ValidationManager:
    """Centralized validation management."""

    def __init__(
        self,
        cache: ContentCache,
        event_manager: EventManager,
        logger: Optional[DITALogger] = None
    ):
        """Initialize validation manager."""
        # Core dependencies
        self.cache = cache
        self.event_manager = event_manager
        self.logger = logger or DITALogger(name=__name__)

        # Initialize registries
        self._pattern_registry: Dict[str, ValidationPattern] = {}
        self._pipelines: Dict[str, ValidationPipeline] = {}
        self._active_validations: Set[str] = set()

        # State tracking
        self._active_validations: Set[str] = set()

        # Register for events
        self._register_event_handlers()

        # Initialize validation components
        self._initialize_validation_components()

    def _initialize_validation_components(self) -> None:
        """Initialize all validation components."""
        try:
            # Register core validation pipelines
            self.create_pipeline("content", [])  # Will be populated by processors
            self.create_pipeline("config", [])   # Will be populated by config manager
            self.create_pipeline("schema", [])   # For schema validation
            self.create_pipeline("dtd", [])      # For DTD validation

            # Register schema validators
            self.register_schema_validators()

            # Register for events (existing code)
            self._register_event_handlers()

        except Exception as e:
            self.logger.error(f"Error initializing validation components: {str(e)}")
            raise

    def _register_event_handlers(self) -> None:
        """Register for validation-related events."""
        self.event_manager.subscribe(
            EventType.STATE_CHANGE,
            self._handle_state_change
        )

    def validate(
        self,
        content: Any,
        validation_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Main validation entry point.

        Args:
            content: Content to validate
            validation_type: Type of validation to perform
            context: Optional validation context

        Returns:
            ValidationResult with validation status and messages
        """
        try:
            # Check cache first
            cache_key = f"validation_{validation_type}_{hash(str(content))}"
            if cached := self.cache.get(cache_key, CacheEntryType.VALIDATION):
                return cached

            # Get validation pipeline
            if pipeline := self._pipelines.get(validation_type):
                result = pipeline.validate(content, context)

                # Cache result
                self.cache.set(
                    key=cache_key,
                    data=result,
                    entry_type=CacheEntryType.VALIDATION,
                    element_type=ElementType.UNKNOWN,  # Generic type for validation results
                    phase=ProcessingPhase.VALIDATION
                )

                return result

            # No pipeline found
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="validation",
                    message=f"Unknown validation type: {validation_type}",
                    severity=ValidationSeverity.ERROR,
                    code="unknown_validation_type"
                )]
            )

        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="validation",
                    message=str(e),
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def register_validator(
        self,
        validation_type: str,
        name: str,
        validator: Callable[..., ValidationResult],
        position: Optional[int] = None
    ) -> None:
        """
        Register validator in pipeline.

        Args:
            validation_type: Type of validation pipeline
            name: Validator name
            validator: Validation function
            position: Optional position in pipeline
        """
        if validation_type not in self._pipelines:
            self._pipelines[validation_type] = ValidationPipeline()

        self._pipelines[validation_type].add_validator(name, validator, position)

    def register_pattern(self, pattern: ValidationPattern) -> None:
        """
        Register validation pattern.

        Args:
            pattern: Validation pattern to register
        """
        self._pattern_registry[pattern.pattern_id] = pattern

    def load_patterns(self, config: Dict[str, Any]) -> None:
        """
        Load validation patterns from configuration.

        Args:
            config: Pattern configuration dictionary
        """
        try:
            patterns = config.get("patterns", {})
            for category, category_patterns in patterns.items():
                for pattern_id, pattern_def in category_patterns.items():
                    full_id = f"{category}.{pattern_id}"

                    if pattern_def.get("type") == "schema":
                        pattern = SchemaValidationPattern(
                            pattern_id=full_id,
                            schema=pattern_def["schema"],
                            description=pattern_def["description"],
                            severity=ValidationSeverity(pattern_def["severity"])
                        )
                    else:
                        pattern = RegexValidationPattern(
                            pattern_id=full_id,
                            pattern=pattern_def["pattern"],
                            description=pattern_def["description"],
                            severity=ValidationSeverity(pattern_def["severity"])
                        )

                    self.register_pattern(pattern)

        except Exception as e:
            self.logger.error(f"Error loading validation patterns: {str(e)}")
            raise

    def validate_content(
        self,
        element: ContentElement,
        context: ProcessingContext
    ) -> ValidationResult:
        """
        Validate content element.

        Args:
            element: Element to validate
            context: Processing context

        Returns:
            ValidationResult with validation status and messages
        """
        try:
            # Check cache
            cache_key = f"validation_{element.id}_{context.state_info.phase.value}"
            if cached := self.cache.get(cache_key, CacheEntryType.VALIDATION):
                return cached

            messages = []

            # Apply registered patterns
            for pattern in self._pattern_registry.values():
                result = pattern.validate(element)
                if not result.is_valid:
                    messages.extend(result.messages)

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
            self.logger.error(f"Error validating content: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Content validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def clear_validation_cache(self) -> None:
        """Clear all validation caches."""
        try:
            self.cache.invalidate_by_pattern("validation_*")
            self._active_validations.clear()
        except Exception as e:
            self.logger.error(f"Error clearing validation cache: {str(e)}")

    def validate_context(
        self,
        context: ProcessingContext,
        validation_pipeline: Optional[ValidationPipeline] = None
    ) -> ValidationResult:
        """
        Validate processing context.

        Args:
            context: Context to validate
            validation_pipeline: Optional custom validation pipeline

        Returns:
            ValidationResult with validation status
        """
        try:
            # Check cache
            cache_key = f"validation_context_{context.context_id}"
            if cached := self.cache.get(cache_key, CacheEntryType.VALIDATION):
                return cached

            messages = []
            pipeline = validation_pipeline or self._pipelines.get("context")

            if not pipeline:
                return ValidationResult(
                    is_valid=False,
                    messages=[self._create_validation_error(
                        "context",
                        "No validation pipeline available for context",
                        "missing_pipeline"
                    )]
                )

            # Validate context
            result = pipeline.validate(context)

            # Cache result
            self.cache.set(
                key=cache_key,
                data=result,
                entry_type=CacheEntryType.VALIDATION,
                element_type=ElementType.UNKNOWN,
                phase=context.state_info.phase
            )

            return result

        except Exception as e:
            self.logger.error(f"Context validation error: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[self._create_validation_error(
                    "context",
                    f"Context validation error: {str(e)}",
                    "context_validation_error"
                )]
            )

    def validate_configuration_set(
        self,
        configs: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate complete configuration set.

        Args:
            configs: Configuration set to validate

        Returns:
            ValidationResult with validation status
        """
        try:
            # Check cache
            cache_key = f"validation_config_{hash(str(configs))}"
            if cached := self.cache.get(cache_key, CacheEntryType.VALIDATION):
                return cached

            messages = []
            pipeline = self._pipelines.get("config")

            if not pipeline:
                return ValidationResult(
                    is_valid=False,
                    messages=[self._create_validation_error(
                        "config",
                        "No validation pipeline available for configuration",
                        "missing_pipeline"
                    )]
                )

            # Validate configuration
            result = pipeline.validate(configs)

            # Cache result
            self.cache.set(
                key=cache_key,
                data=result,
                entry_type=CacheEntryType.VALIDATION,
                element_type=ElementType.UNKNOWN,
                phase=ProcessingPhase.VALIDATION
            )

            return result

        except Exception as e:
            self.logger.error(f"Configuration validation error: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[self._create_validation_error(
                    "config",
                    f"Configuration validation error: {str(e)}",
                    "config_validation_error"
                )]
            )

    def create_pipeline(
        self,
        validation_type: str,
        validators: List[Tuple[str, Callable[..., ValidationResult]]]
    ) -> ValidationPipeline:
        """
        Create validation pipeline.

        Args:
            validation_type: Type of validation pipeline
            validators: List of (name, validator) tuples

        Returns:
            Created ValidationPipeline
        """
        pipeline = ValidationPipeline()

        for name, validator in validators:
            pipeline.add_validator(name, validator)

        self._pipelines[validation_type] = pipeline
        return pipeline

    def register_dtd_pipeline(
        self,
        dtd_validator: 'DTDValidator'
    ) -> None:
        """
        Register DTD validation pipeline.

        Args:
            dtd_validator: DTD validator instance
        """
        try:
            def structure_validator(
                content: Any,
                context: Dict[str, Any]
            ) -> ValidationResult:
                messages = dtd_validator._validate_structure(
                    content,
                    ValidationState(context=context["validation_context"])
                )
                return ValidationResult(
                    is_valid=not any(
                        msg.severity == ValidationSeverity.ERROR for msg in messages
                    ),
                    messages=messages
                )

            def attributes_validator(
                content: Any,
                context: Dict[str, Any]
            ) -> ValidationResult:
                messages = dtd_validator._validate_attributes(
                    content,
                    ValidationState(context=context["validation_context"])
                )
                return ValidationResult(
                    is_valid=not any(
                        msg.severity == ValidationSeverity.ERROR for msg in messages
                    ),
                    messages=messages
                )

            # Create DTD pipeline
            dtd_pipeline = self.create_pipeline(
                "dtd",
                [
                    ("structure", structure_validator),
                    ("attributes", attributes_validator)
                ]
            )

            # Set fail-fast behavior
            dtd_pipeline.set_fail_fast(True)

        except Exception as e:
            self.logger.error(f"Error registering DTD pipeline: {str(e)}")
            raise

    def _create_validation_error(
        self,
        path: str,
        message: str,
        code: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ) -> ValidationMessage:
        """
        Create validation error message.

        Args:
            path: Error location path
            message: Error message
            code: Error code
            severity: Error severity

        Returns:
            ValidationMessage
        """
        return ValidationMessage(
            path=path,
            message=message,
            severity=severity,
            code=code
        )

    def _handle_state_change(self, **event_data: Any) -> None:
        """
        Handle state change events with validation tracking.

        Args:
            **event_data: Event data including:
                - element_id: ID of affected element
                - state_info: Processing state information
                - validation_context: Optional validation context
                - metadata: Optional validation metadata
        """
        try:
            element_id = event_data.get("element_id")
            state_info = event_data.get("state_info")
            validation_context = event_data.get("validation_context")
            metadata = event_data.get("metadata", {})

            if not (element_id and state_info):
                return

            # Track validation state
            if element_id in self._active_validations:
                # Update validation state
                validation_key = f"validation_{element_id}_{state_info.phase.value}"

                if state_info.state == ProcessingState.ERROR:
                    # Handle validation errors
                    if error_msg := state_info.error_message:
                        self.cache.set(
                            key=validation_key,
                            data=ValidationResult(
                                is_valid=False,
                                messages=[ValidationMessage(
                                    path=element_id,
                                    message=error_msg,
                                    severity=ValidationSeverity.ERROR,
                                    code="validation_error"
                                )]
                            ),
                            entry_type=CacheEntryType.VALIDATION,
                            element_type=ElementType.UNKNOWN,
                            phase=state_info.phase,
                            metadata=metadata
                        )

                elif state_info.state == ProcessingState.COMPLETED:
                    # Clear validation errors
                    self.cache.set(
                        key=validation_key,
                        data=ValidationResult(
                            is_valid=True,
                            messages=[]
                        ),
                        entry_type=CacheEntryType.VALIDATION,
                        element_type=ElementType.UNKNOWN,
                        phase=state_info.phase,
                        metadata=metadata
                    )

                # Clean up completed validations
                if state_info.state in {ProcessingState.COMPLETED, ProcessingState.ERROR}:
                    self._active_validations.discard(element_id)

            # Handle validation phase specifically
            if state_info.phase == ProcessingPhase.VALIDATION:
                # Ensure validation context exists
                if not validation_context:
                    self.logger.warning(
                        f"Missing validation context for {element_id} "
                        f"during {state_info.phase.value}"
                    )
                    return

                # Get appropriate pipeline
                if pipeline := self._pipelines.get(validation_context.get("type", "default")):
                    # Run validation
                    result = pipeline.validate(
                        element_id,
                        context=validation_context
                    )

                    # Cache result
                    self.cache.set(
                        key=f"validation_{element_id}_{state_info.phase.value}",
                        data=result,
                        entry_type=CacheEntryType.VALIDATION,
                        element_type=ElementType.UNKNOWN,
                        phase=state_info.phase,
                        metadata=metadata
                    )

                    # Emit validation results
                    self.event_manager.emit(
                        EventType.VALIDATION_COMPLETE,
                        element_id=element_id,
                        validation_result=result,
                        context=validation_context
                    )

        except Exception as e:
            self.logger.error(f"Error handling state change: {str(e)}")

    def _handle_cache_invalidation(self, **event_data: Any) -> None:
        """
        Handle cache invalidation events for validation results.

        Args:
            **event_data: Event data including:
                - pattern: Cache invalidation pattern
                - element_id: Optional specific element ID
                - phase: Optional processing phase
        """
        try:
            pattern = event_data.get("pattern")
            element_id = event_data.get("element_id")
            phase = event_data.get("phase")

            # Handle pattern-based invalidation
            if pattern:
                # Invalidate validation results matching pattern
                self.cache.invalidate_by_pattern(
                    f"validation_{pattern}",
                    entry_type=CacheEntryType.VALIDATION
                )

                # Remove from active validations if matching
                matching_validations = {
                    vid for vid in self._active_validations
                    if pattern in vid
                }
                self._active_validations.difference_update(matching_validations)

            # Handle specific element invalidation
            elif element_id:
                # Build cache key based on phase
                if phase:
                    cache_key = f"validation_{element_id}_{phase.value}"
                    self.cache.invalidate(
                        key=cache_key,
                        entry_type=CacheEntryType.VALIDATION
                    )
                else:
                    # Invalidate all phase results for element
                    self.cache.invalidate_by_pattern(
                        f"validation_{element_id}_*",
                        entry_type=CacheEntryType.VALIDATION
                    )

                # Remove from active validations
                self._active_validations.discard(element_id)

                # Emit validation invalidated event
                self.event_manager.emit(
                    EventType.CACHE_INVALIDATED,
                    element_id=element_id,
                    cache_type="validation",
                    phase=phase
                )

            # Handle full validation cache clear
            else:
                self.cache.invalidate_by_pattern(
                    "validation_*",
                    entry_type=CacheEntryType.VALIDATION
                )
                self._active_validations.clear()

        except Exception as e:
            self.logger.error(f"Error handling cache invalidation: {str(e)}")


    def _validate_element_definition(
        self,
        element_name: str,
        element_def: Dict[str, Any]
    ) -> bool:
        """
        Validate DTD element definition.

        Args:
            element_name: Name of element being validated
            element_def: Element definition dictionary

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check required fields
            required_fields = {"content_model", "attributes"}
            if not all(field in element_def for field in required_fields):
                return False

            # Validate content model
            content_model = element_def["content_model"]
            if not isinstance(content_model, dict):
                return False

            required_model_fields = {"type", "elements", "ordering", "occurrence"}
            if not all(field in content_model for field in required_model_fields):
                return False

            # Validate occurrence constraints
            occurrence = content_model["occurrence"]
            if not isinstance(occurrence, dict):
                return False

            if not all(field in occurrence for field in ["min", "max"]):
                return False

            # Validate ordering
            if content_model["ordering"] not in ["sequence", "choice"]:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating element definition: {str(e)}")
            return False

    def _validate_attribute_definition(
        self,
        attr_name: str,
        attr_def: Dict[str, Any],
        element_name: str
    ) -> bool:
        """
        Validate DTD attribute definition.

        Args:
            attr_name: Name of attribute being validated
            attr_def: Attribute definition dictionary
            element_name: Name of element this attribute belongs to

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check required fields
            required_fields = {"type", "required"}
            if not all(field in attr_def for field in required_fields):
                return False

            # Validate attribute type
            valid_types = {
                "string", "id", "reference", "references",
                "token", "tokens", "entity", "entities",
                "enum", "notation"
            }
            if attr_def["type"] not in valid_types:
                return False

            # Validate enum values if type is enum
            if attr_def["type"] == "enum":
                if "allowed_values" not in attr_def or not isinstance(attr_def["allowed_values"], list):
                    return False

            # Validate default value if present
            if "default_value" in attr_def:
                if attr_def["type"] == "enum":
                    if attr_def["default_value"] not in attr_def.get("allowed_values", []):
                        return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating attribute definition: {str(e)}")
            return False

    def _validate_inheritance_path(
        self,
        path: List[str],
        context: Dict[str, Any]
    ) -> bool:
        """
        Validate DTD inheritance path.

        Args:
            path: List of element types in inheritance chain
            context: Validation context with element definitions

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            if not path or len(path) < 2:
                return False

            elements = context.get("elements", {})

            # Check each element in path exists
            for element_type in path:
                if element_type not in elements:
                    return False

            # Validate inheritance relationships
            for i in range(len(path) - 1):
                base_type = path[i]
                derived_type = path[i + 1]

                # Get base element definition
                base_def = elements.get(base_type)
                if not base_def:
                    return False

                # Check if derivation is allowed
                allowed_specializations = base_def.get("allowed_specializations", [])
                if derived_type not in allowed_specializations:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating inheritance path: {str(e)}")
            return False


    ###########################
    # DTD Validation methods #
    ##########################

    def register_dtd_validators(self, dtd_validator: 'DTDValidator') -> None:
        """Register DTD validation components."""
        try:
            # Update structure validation
            def validate_structure(content: Any, **ctx: Any) -> ValidationResult:
                context = ctx.get("validation_context", {})
                if isinstance(content, dict):
                    return self._validate_dtd_structure(content, context)
                return ValidationResult(
                    is_valid=not any(
                        msg.severity == ValidationSeverity.ERROR
                        for msg in dtd_validator._validate_structure(
                            content,
                            ValidationState(context=context)
                        )
                    ),
                    messages=dtd_validator._validate_structure(
                        content,
                        ValidationState(context=context)
                    )
                )

            # Update attribute validation
            def validate_attributes(content: Any, **ctx: Any) -> ValidationResult:
                context = ctx.get("validation_context", {})
                if isinstance(content, dict):
                    return self._validate_dtd_attributes(content, context)
                return ValidationResult(
                    is_valid=not any(
                        msg.severity == ValidationSeverity.ERROR
                        for msg in dtd_validator._validate_attributes(
                            content,
                            ValidationState(context=context)
                        )
                    ),
                    messages=dtd_validator._validate_attributes(
                        content,
                        ValidationState(context=context)
                    )
                )

            # Update specialization validation
            def validate_specialization(content: Any, **ctx: Any) -> ValidationResult:
                context = ctx.get("validation_context", {})
                if isinstance(content, dict):
                    return self._validate_dtd_specialization(content, context)

                # Get specialization info from context
                spec_info = context.get("specialization_info")
                if not spec_info:
                    return ValidationResult(
                        is_valid=False,
                        messages=[ValidationMessage(
                            path="specialization",
                            message="Missing specialization info in context",
                            severity=ValidationSeverity.ERROR,
                            code="missing_specialization_info"
                        )]
                    )

                return ValidationResult(
                    is_valid=not any(
                        msg.severity == ValidationSeverity.ERROR
                        for msg in dtd_validator._validate_specialization_constraints(
                            content,
                            spec_info,
                            ValidationState(context=context)
                        )
                    ),
                    messages=dtd_validator._validate_specialization_constraints(
                        content,
                        spec_info,
                        ValidationState(context=context)
                    )
                )

            # Register validators
            self.register_validator(
                validation_type="dtd",
                name="structure",
                validator=validate_structure
            )

            self.register_validator(
                validation_type="dtd",
                name="attributes",
                validator=validate_attributes
            )

            self.register_validator(
                validation_type="dtd",
                name="specialization",
                validator=validate_specialization
            )

            self.logger.debug("Registered DTD validators")

        except Exception as e:
            self.logger.error(f"Error registering DTD validators: {str(e)}")
            raise


    def validate_dtd_schema(
            self,
            schema: Dict[str, Any],
            dtd_path: Path
        ) -> ValidationResult:
            """
            Validate DTD-derived schema.

            Args:
                schema: Schema converted from DTD
                dtd_path: Original DTD file path

            Returns:
                ValidationResult indicating validation status
            """
            try:
                messages = []

                # Get DTD validation pipeline
                if pipeline := self._pipelines.get("dtd"):
                    # Create validation context
                    validation_context = {
                        "dtd_path": str(dtd_path),
                        "phase": ProcessingPhase.VALIDATION,
                        "validation_type": "dtd_schema"
                    }

                    # Validate schema structure
                    structure_result = self._validate_dtd_structure(
                        schema,
                        validation_context
                    )
                    messages.extend(structure_result.messages)

                    # Validate attributes if structure is valid
                    if structure_result.is_valid:
                        attr_result = self._validate_dtd_attributes(
                            schema,
                            validation_context
                        )
                        messages.extend(attr_result.messages)

                    # Validate specializations if present
                    if specializations := schema.get("specializations"):
                        spec_result = self._validate_dtd_specialization(
                            specializations,
                            validation_context
                        )
                        messages.extend(spec_result.messages)

                else:
                    messages.append(ValidationMessage(
                        path="validation",
                        message="DTD validation pipeline not available",
                        severity=ValidationSeverity.ERROR,
                        code="missing_pipeline"
                    ))

                return ValidationResult(
                    is_valid=not any(
                        msg.severity == ValidationSeverity.ERROR for msg in messages
                    ),
                    messages=messages
                )

            except Exception as e:
                self.logger.error(f"Error validating DTD schema: {str(e)}")
                return ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path=str(dtd_path),
                        message=f"Schema validation error: {str(e)}",
                        severity=ValidationSeverity.ERROR,
                        code="dtd_validation_error"
                    )]
                )

    def _validate_dtd_structure(
        self,
        schema: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ValidationResult:
        """Validate DTD schema structure."""
        messages = []

        try:
            # Check required sections
            required_sections = {
                "elements",
                "attributes",
                "inheritance"
            }

            missing_sections = required_sections - set(schema.keys())
            if missing_sections:
                messages.append(ValidationMessage(
                    path="schema",
                    message=f"Missing required sections: {missing_sections}",
                    severity=ValidationSeverity.ERROR,
                    code="missing_sections"
                ))

            # Validate element definitions
            if elements := schema.get("elements"):
                for element_name, element_def in elements.items():
                    if not self._validate_element_definition(
                        element_name,
                        element_def
                    ):
                        messages.append(ValidationMessage(
                            path=f"elements.{element_name}",
                            message="Invalid element definition",
                            severity=ValidationSeverity.ERROR,
                            code="invalid_element"
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
                    path="structure",
                    message=f"Structure validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="structure_validation_error"
                )]
            )

    def _validate_dtd_attributes(
        self,
        schema: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ValidationResult:
        """Validate DTD schema attributes."""
        messages = []

        try:
            if attributes := schema.get("attributes"):
                for element_name, attrs in attributes.items():
                    # Check element exists
                    if element_name not in schema.get("elements", {}):
                        messages.append(ValidationMessage(
                            path=f"attributes.{element_name}",
                            message=f"Attributes defined for unknown element: {element_name}",
                            severity=ValidationSeverity.ERROR,
                            code="unknown_element"
                        ))
                        continue

                    # Validate each attribute
                    for attr_name, attr_def in attrs.items():
                        if not self._validate_attribute_definition(
                            attr_name,
                            attr_def,
                            element_name
                        ):
                            messages.append(ValidationMessage(
                                path=f"attributes.{element_name}.{attr_name}",
                                message="Invalid attribute definition",
                                severity=ValidationSeverity.ERROR,
                                code="invalid_attribute"
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
                    path="attributes",
                    message=f"Attribute validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="attribute_validation_error"
                )]
            )

    def _validate_dtd_specialization(
        self,
        specializations: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ValidationResult:
        """Validate DTD schema specializations."""
        messages = []

        try:
            for spec_name, spec_def in specializations.items():
                # Validate base type exists
                if base_type := spec_def.get("base_type"):
                    if base_type not in context.get("elements", {}):
                        messages.append(ValidationMessage(
                            path=f"specializations.{spec_name}",
                            message=f"Unknown base type: {base_type}",
                            severity=ValidationSeverity.ERROR,
                            code="unknown_base_type"
                        ))

                # Validate inheritance path
                if path := spec_def.get("inheritance_path"):
                    if not self._validate_inheritance_path(path, context):
                        messages.append(ValidationMessage(
                            path=f"specializations.{spec_name}",
                            message="Invalid inheritance path",
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
                    path="specializations",
                    message=f"Specialization validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="specialization_validation_error"
                )]
            )

    def register_schema_validators(self) -> None:
        """Register schema validation components."""
        try:
            # Register schema composition validation
            self.register_validator(
                validation_type="schema",
                name="composition",
                validator=self._validate_schema_composition
            )

            # Register schema completeness validation
            self.register_validator(
                validation_type="schema",
                name="completeness",
                validator=self._validate_schema_completeness
            )

            # Register reference integrity validation
            self.register_validator(
                validation_type="schema",
                name="references",
                validator=self._validate_reference_integrity
            )

            self.logger.debug("Registered schema validators")

        except Exception as e:
            self.logger.error(f"Error registering schema validators: {str(e)}")
            raise

    def _validate_schema_composition(
        self,
        schema: Dict[str, Any],
        **context: Any
    ) -> ValidationResult:
        """Validate schema composition."""
        messages = []

        try:
            # Validate required sections
            required_sections = {
                "elements", "attributes", "validation"
            }

            missing_sections = required_sections - set(schema.keys())
            if missing_sections:
                messages.append(ValidationMessage(
                    path="schema",
                    message=f"Missing required sections: {missing_sections}",
                    severity=ValidationSeverity.ERROR,
                    code="missing_sections"
                ))

            # Validate section types
            if elements := schema.get("elements"):
                if not isinstance(elements, dict):
                    messages.append(ValidationMessage(
                        path="schema.elements",
                        message="Elements must be an object",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_elements"
                    ))

            if attributes := schema.get("attributes"):
                if not isinstance(attributes, dict):
                    messages.append(ValidationMessage(
                        path="schema.attributes",
                        message="Attributes must be an object",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_attributes"
                    ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR
                    for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="schema",
                    message=f"Schema composition error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="composition_error"
                )]
            )

    def _validate_schema_completeness(
        self,
        schema: Dict[str, Any],
        **context: Any
    ) -> ValidationResult:
        """Validate schema completeness."""
        messages = []

        try:
            # Check attribute definitions completeness
            if attributes := schema.get("attributes"):
                for attr_name, attr_def in attributes.items():
                    if "type" not in attr_def:
                        messages.append(ValidationMessage(
                            path=f"schema.attributes.{attr_name}",
                            message="Missing attribute type",
                            severity=ValidationSeverity.ERROR,
                            code="missing_type"
                        ))

            # Check element definitions completeness
            if elements := schema.get("elements"):
                for elem_name, elem_def in elements.items():
                    if "content_model" not in elem_def:
                        messages.append(ValidationMessage(
                            path=f"schema.elements.{elem_name}",
                            message="Missing content model",
                            severity=ValidationSeverity.ERROR,
                            code="missing_content_model"
                        ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR
                    for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="schema",
                    message=f"Schema completeness error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="completeness_error"
                )]
            )

    def _validate_reference_integrity(
        self,
        schema: Dict[str, Any],
        **context: Any
    ) -> ValidationResult:
        """Validate schema reference integrity."""
        messages = []

        try:
            elements = schema.get("elements", {})

            # Check element references in content models
            for elem_name, elem_def in elements.items():
                if content_model := elem_def.get("content_model"):
                    if referenced := content_model.get("elements", []):
                        for ref in referenced:
                            if ref not in elements:
                                messages.append(ValidationMessage(
                                    path=f"schema.elements.{elem_name}",
                                    message=f"Referenced element not found: {ref}",
                                    severity=ValidationSeverity.ERROR,
                                    code="invalid_reference"
                                ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR
                    for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="schema",
                    message=f"Reference integrity error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="reference_error"
                )]
            )
