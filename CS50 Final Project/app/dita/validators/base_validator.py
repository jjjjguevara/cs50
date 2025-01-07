"""Abstract base class for all validators in the DITA processing pipeline."""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List, Set, Generator
from pathlib import Path
import re
from datetime import datetime
from contextlib import contextmanager

# Core system types
from ..types import (
    ValidationResult,
    ValidationMessage,
    ValidationSeverity,
    ValidationMode,
    ValidationType,
    ValidationContext,
    ValidationPattern,
    ValidationState,
    AttributeType,
    ElementType,
    ProcessingPhase,
    ProcessingState,
    CacheEntryType,
    EventType,
    DTDReference
)

# Core managers
from ..events.event_manager import EventManager
from ..config.config_manager import ConfigManager
from ..main.context_manager import ContextManager

# Cache system
from ..cache.cache import ContentCache

# Utilities
from ..utils.logger import DITALogger


class BaseValidator(ABC):
    """Abstract base validator for all validation operations."""

    def __init__(
        self,
        event_manager: EventManager,
        context_manager: ContextManager,
        config_manager: ConfigManager,
        cache: ContentCache,
        validator_type: ValidationType,
        logger: Optional[DITALogger] = None,
        *args, **kwargs,
    ):
        """Initialize base validator."""
        # Core dependencies initialization from previous code
        super().__init__(*args, **kwargs)
        self.event_manager = event_manager
        self.context_manager = context_manager
        self.config_manager = config_manager
        self.cache = cache
        self.logger = logger or DITALogger(name=__name__)

        # Validator configuration
        self.validator_type = validator_type
        self.validation_mode = ValidationMode.STRICT
        self._initialized = False
        self._active = False

        # State tracking
        self._validation_stack: List[str] = []
        self._active_validations: Dict[str, ValidationState] = {}
        self._validation_patterns: Dict[str, ValidationPattern] = {}
        self._validation_depth = 0
        self._max_validation_depth = 10

        # Validation results cache
        self._results_cache: Dict[str, ValidationResult] = {}
        self._pending_validations: Set[str] = set()

        # Load configuration and register handlers
        self._load_validator_config()
        self._register_event_handlers()

    def _register_validation_patterns(self) -> None:
            """Register common validation patterns."""
            self.register_pattern(ValidationPattern(
                pattern_id="attribute.required",
                description="Validate required attributes",
                severity=ValidationSeverity.ERROR
            ))
            self.register_pattern(ValidationPattern(
                pattern_id="content.structure",
                description="Validate content structure",
                severity=ValidationSeverity.ERROR
            ))
            # Add more common patterns

    def _validate_constraint(
        self,
        content: Any,
        constraint: Dict[str, Any],
        constraint_type: str,
        state: ValidationState
    ) -> bool:
        """Generic constraint validation."""
        constraint_validators = {
            'attribute': self._validate_attribute_constraint,
            'content': self._validate_content_constraint,
            'structural': self._validate_structural_constraint
        }
        validator = constraint_validators.get(constraint_type)
        if validator:
            return validator(content, constraint, state)
        return True

    @abstractmethod
    def _validate_attribute_constraint(
        self,
        content: Any,
        constraint: Dict[str, Any],
        state: ValidationState
    ) -> bool:
        """Abstract method for attribute validation."""
        pass

    @abstractmethod
    def _validate_content_constraint(
        self,
        content: Any,
        constraint: Dict[str, Any],
        state: ValidationState
    ) -> bool:
        """Abstract method for content validation."""
        pass

    @abstractmethod
    def _validate_structural_constraint(
        self,
        content: Any,
        constraint: Dict[str, Any],
        state: ValidationState
    ) -> bool:
        """Abstract method for structural validation."""
        pass

    def _validate_attribute_value(
        self,
        value: Any,
        attr_def: Any,
        state: ValidationState
    ) -> bool:
        """Common attribute value validation."""
        try:
            if attr_def.type == AttributeType.ENUM:
                return value in attr_def.allowed_values
            elif attr_def.type == AttributeType.ID:
                return bool(re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', str(value)))
            # Add more common validations
            return True
        except Exception as e:
            self.logger.error(f"Error validating attribute value: {str(e)}")
            return False


    @abstractmethod
    def validate(
        self,
        content: Any,
        context: Optional[ValidationContext] = None,
        **kwargs: Any
    ) -> ValidationResult:
        """
        Main validation entry point.

        Args:
            content: Content to validate
            context: Optional validation context
            **kwargs: Additional validation parameters

        Returns:
            ValidationResult: Validation results
        """
        pass

    @abstractmethod
    def can_validate(self, content: Any) -> bool:
        """
        Check if validator can handle content.

        Args:
            content: Content to check

        Returns:
            bool: True if validator can handle content
        """
        pass

    @abstractmethod
    def get_supported_types(self) -> Set[ElementType]:
        """
        Get supported element types.

        Returns:
            Set[ElementType]: Set of supported element types
        """
        pass

    def _load_validator_config(self) -> None:
        """Load validator configuration."""
        try:
            # Get validator-specific config
            config = self.config_manager.get_config(f"{self.validator_type.value}_validator")
            if config:
                self.validation_mode = ValidationMode(
                    config.get("validation_mode", ValidationMode.STRICT.value)
                )
                self._max_validation_depth = config.get("max_validation_depth", 10)

            self._initialized = True

        except Exception as e:
            self.logger.error(f"Error loading validator config: {str(e)}")
            raise

    def _register_event_handlers(self) -> None:
        """Register validation event handlers."""
        self.event_manager.subscribe(
            EventType.STATE_CHANGE,
            self._handle_state_change
        )
        self.event_manager.subscribe(
            EventType.CACHE_INVALIDATE,
            self._handle_cache_invalidation
        )

    @contextmanager
    def validation_context(
        self,
        element_id: str
    ) -> Generator[ValidationState, None, None]:
        """
        Context manager for validation operations.

        Args:
            element_id: Element being validated

        Yields:
            ValidationState: Current validation state
        """
        try:
            # Check validation depth
            if self._validation_depth >= self._max_validation_depth:
                raise ValueError(f"Maximum validation depth exceeded for {element_id}")

            # Check for circular validation
            if element_id in self._validation_stack:
                raise ValueError(f"Circular validation detected for {element_id}")

            # Initialize validation state
            self._validation_stack.append(element_id)
            self._validation_depth += 1
            self._active_validations.add(element_id)

            # Create a proper DTDReference
            dtd_ref = DTDReference(
                path=Path(""),  # Use appropriate path
                type="system",  # or "public" based on your needs
                public_id=None,
                system_id=None,
                resolved_path=None,
                last_modified=datetime.now()
            )

            # Create validation state with proper context
            state = ValidationState(
                context=ValidationContext(
                    dtd_ref=dtd_ref,
                    element_hierarchy=[],
                    in_mixed_content=False,
                    allow_unknown_elements=False,
                    metadata={}
                )
            )

            yield state

        finally:
            # Cleanup validation state
            self._validation_stack.pop()
            self._validation_depth -= 1
            self._active_validations.discard(element_id)

    def create_validation_message(
        self,
        path: str,
        message: str,
        severity: ValidationSeverity,
        code: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationMessage:
        """
        Create validation message with context.

        Args:
            path: Error location path
            message: Error message
            severity: Error severity
            code: Error code
            context: Optional message context

        Returns:
            ValidationMessage: Created message
        """
        return ValidationMessage(
            path=path,
            message=message,
            severity=severity,
            code=code,
        )

    def _handle_state_change(self, **event_data: Any) -> None:
        """Handle validation state changes."""
        try:
            element_id = event_data.get("element_id")
            state_info = event_data.get("state_info")

            if element_id and state_info:
                if element_id in self._active_validations:
                    if state_info.state == ProcessingState.ERROR:
                        self._handle_validation_error(element_id, state_info)
                    elif state_info.state == ProcessingState.COMPLETED:
                        self._handle_validation_complete(element_id, state_info)

        except Exception as e:
            self.logger.error(f"Error handling state change: {str(e)}")

    def _handle_validation_error(
        self,
        element_id: str,
        state_info: ProcessingState
    ) -> None:
        """Handle validation error state."""
        try:
            # Get error message from state
            error_msg = getattr(state_info, 'message', None) or "Validation failed"

            # Cache error result
            self._results_cache[element_id] = ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path=element_id,
                    message=error_msg,
                    severity=ValidationSeverity.ERROR,
                    code="validation_failed"
                )]
            )

            # Emit validation event with proper EventType
            self.event_manager.emit(
                EventType.VALIDATION_FAILED,
                element_id=element_id,
                validation_type=self.validator_type.value,
                error=error_msg
            )

        except Exception as e:
            self.logger.error(f"Error handling validation error: {str(e)}")

    def _handle_validation_complete(
        self,
        element_id: str,
        state_info: ProcessingState
    ) -> None:
        """Handle validation completion."""
        try:
            # Clear from pending validations
            self._pending_validations.discard(element_id)

            # Cache validation result if successful
            if result := self._results_cache.get(element_id):
                if result.is_valid:
                    self.cache.set(
                        key=f"validation_{element_id}_{self.validator_type.value}",
                        data=result,
                        entry_type=CacheEntryType.VALIDATION,
                        element_type=ElementType.UNKNOWN,
                        phase=ProcessingPhase.VALIDATION
                    )

        except Exception as e:
            self.logger.error(f"Error handling validation completion: {str(e)}")

    def _handle_cache_invalidation(self, **event_data: Any) -> None:
        """Handle cache invalidation events."""
        try:
            pattern = event_data.get("pattern")
            if pattern:
                # Clear cached results
                self._results_cache = {
                    k: v for k, v in self._results_cache.items()
                    if not k.startswith(pattern)
                }

        except Exception as e:
            self.logger.error(f"Error handling cache invalidation: {str(e)}")

    def cleanup(self) -> None:
        """Clean up validator resources."""
        try:
            self._active_validations.clear()
            self._validation_stack.clear()
            self._results_cache.clear()
            self._pending_validations.clear()
            self._validation_depth = 0
            self._initialized = False
            self._active = False

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise

    @property
    def is_active(self) -> bool:
        """Check if validator is active."""
        return self._active and self._initialized

    @property
    def current_validation_depth(self) -> int:
        """Get current validation depth."""
        return self._validation_depth

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.validator_type.value})"
