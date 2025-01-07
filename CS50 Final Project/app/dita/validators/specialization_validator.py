"""Specialization validation for DITA content."""
from typing import Dict, List, Optional, Any, Set, TYPE_CHECKING
import re

# Event and Cache
from ..events.event_manager import EventManager
from ..cache.cache import ContentCache

# Core validators and components
from .base_validator import BaseValidator

# DTD specific
from ..dtd.dtd_mapper import DTDSchemaMapper
from ..dtd.dtd_resolver import DTDResolver

# Types
from ..types import (
    ValidationType,
    ValidationResult,
    ValidationMessage,
    ValidationSeverity,
    ValidationContext,
    ValidationState,
    ValidationPattern,
    RegexValidationPattern,
    SchemaValidationPattern,
    ElementType,
    ProcessingPhase,
    ProcessingState,
    ProcessingContext,
    SpecializationInfo,
    DTDElement,
    ContentModel,
    ContentScope,
    CacheEntryType
)

# Forward references for type checking
if TYPE_CHECKING:
    from ..config.config_manager import ConfigManager
    from .validation_manager import ValidationManager
    from ..main.context_manager import ContextManager

from ..utils.logger import DITALogger

class SpecializationValidator(BaseValidator):
    """
    Handles validation of DITA specialization relationships.
    Extends BaseValidator for specialization-specific validation.
    """

    def __init__(
        self,
        validation_manager: 'ValidationManager',
        event_manager: EventManager,
        config_manager: 'ConfigManager',
        context_manager: 'ContextManager',
        content_cache: ContentCache,
        dtd_mapper: DTDSchemaMapper,
        dtd_resolver: DTDResolver,
        logger: Optional[DITALogger] = None
    ):
        """Initialize specialization validator.

        Args:
            validation_manager: System validation manager
            event_manager: Event management system
            config_manager: Configuration management system
            context_manager: Context management system
            content_cache: Cache system
            dtd_mapper: DTD to schema mapper
            dtd_resolver: DTD resolution system
            logger: Optional logger instance
        """
        super().__init__(
            event_manager=event_manager,
            context_manager=context_manager,
            config_manager=config_manager,
            cache=content_cache,
            validator_type=ValidationType.SPECIALIZATION,
            logger=logger
        )

        # Store dependencies
        self.validation_manager = validation_manager
        self.dtd_mapper = dtd_mapper
        self.dtd_resolver = dtd_resolver

        # Specialization tracking
        self._specialization_cache: Dict[str, SpecializationInfo] = {}
        self._inheritance_chains: Dict[str, List[str]] = {}
        self._base_elements: Dict[str, DTDElement] = {}

        # Validation state
        self._active_validations: Set[str] = set()
        self._validation_depth: int = 0
        self._max_validation_depth: int = 10  # Configurable

        # Register validation patterns
        self._register_validation_patterns()

        # Initialize from configuration
        self._initialize_from_config()

    def _register_validation_patterns(self) -> None:
        """Register specialization-specific validation patterns."""
        try:
            # Get validation patterns from config
            validation_config = self.config_manager.get_config("validation_patterns.json")
            if not validation_config:
                raise ValueError("Validation patterns configuration not found")

            # Get DTD specialization patterns
            dtd_patterns = validation_config.get("patterns", {}).get("dtd_validation", {})
            specialization_patterns = dtd_patterns.get("specialization_patterns", {})
            constraint_patterns = validation_config.get("patterns", {}).get("dtd_constraints", {})

            # Register inheritance pattern
            if inheritance_pattern := specialization_patterns.get("inheritance"):
                self.validation_manager.register_pattern(RegexValidationPattern(
                    pattern_id="specialization.inheritance",
                    pattern=inheritance_pattern["pattern"],
                    description=inheritance_pattern["description"],
                    severity=ValidationSeverity(inheritance_pattern["severity"]),
                    code=inheritance_pattern["code"],
                    metadata=inheritance_pattern.get("metadata", {}),
                    dependencies=validation_config.get("dependencies", {}).get(
                        "dtd_validation.specialization_patterns.inheritance",
                        []
                    )
                ))

            # Register schema-based patterns
            if schema_patterns := validation_config.get("patterns", {}).get("metadata", {}):
                for pattern_id, pattern_def in schema_patterns.items():
                    if pattern_def.get("type") == "schema":
                        self.validation_manager.register_pattern(SchemaValidationPattern(
                            pattern_id=f"specialization.schema.{pattern_id}",
                            schema=pattern_def["schema"],
                            description=pattern_def["description"],
                            severity=ValidationSeverity(pattern_def["severity"]),
                            code=pattern_def.get("code", "schema_validation"),
                            metadata=pattern_def.get("metadata", {})
                        ))

            # Register attribute patterns
            attribute_patterns = dtd_patterns.get("attribute_patterns", {})
            for attr_id, attr_pattern in attribute_patterns.items():
                self.validation_manager.register_pattern(RegexValidationPattern(
                    pattern_id=f"specialization.attribute.{attr_id}",
                    pattern=attr_pattern["pattern"],
                    description=attr_pattern["description"],
                    severity=ValidationSeverity(attr_pattern["severity"]),
                    code=attr_pattern["code"],
                    metadata=attr_pattern.get("metadata", {})
                ))

        except Exception as e:
            self.logger.error(f"Error registering validation patterns: {str(e)}")
            raise

    def _get_pattern_by_path(
        self,
        patterns: Dict[str, Any],
        path: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Get pattern definition by path reference."""
        current = patterns
        for part in path:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current if isinstance(current, dict) else None

    def _initialize_from_config(self) -> None:
        """Initialize validator from configuration."""
        try:
            # Get specialization configuration
            config = self.config_manager.get_config("specialization_validation")
            if config:
                self._max_validation_depth = config.get(
                    "max_validation_depth",
                    self._max_validation_depth
                )

        except Exception as e:
            self.logger.error(f"Error initializing from config: {str(e)}")
            raise

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
        try:
            # Check validation depth
            if self._validation_depth >= self._max_validation_depth:
                return ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path="",
                        message="Maximum validation depth exceeded",
                        severity=ValidationSeverity.ERROR,
                        code="max_depth_exceeded"
                    )]
                )

            # Get specialization info
            spec_info = self._get_specialization_info(content, context)
            if not spec_info:
                return ValidationResult(is_valid=True, messages=[])

            messages = []

            # Validate inheritance path
            inheritance_result = self._validate_inheritance_path(
                spec_info,
                context
            )
            messages.extend(inheritance_result.messages)

            if inheritance_result.is_valid:
                # Validate constraints
                constraint_result = self._validate_specialization_constraints(
                    content,
                    spec_info,
                    context
                )
                messages.extend(constraint_result.messages)

                # Validate attributes
                attribute_result = self._validate_specialized_attributes(
                    content,
                    spec_info,
                    context
                )
                messages.extend(attribute_result.messages)

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error in specialization validation: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Specialization validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def _get_specialization_info(
        self,
        content: Any,
        context: Optional[ValidationContext]
    ) -> Optional[SpecializationInfo]:
        """Get specialization information for content."""
        try:
            # Check cache first
            cache_key = f"spec_info_{hash(str(content))}"
            if cached := self.cache.get(cache_key, CacheEntryType.VALIDATION):
                return cached

            # Get info from DTD mapper
            if hasattr(content, 'tag'):
                spec_info = self.dtd_mapper.get_specialization_info(content.tag)
                if spec_info:
                    # Cache result
                    self.cache.set(
                        key=cache_key,
                        data=spec_info,
                        entry_type=CacheEntryType.VALIDATION,
                        element_type=ElementType.UNKNOWN,
                        phase=ProcessingPhase.VALIDATION
                    )
                return spec_info

            return None

        except Exception as e:
            self.logger.error(f"Error getting specialization info: {str(e)}")
            return None

    def _validate_inheritance_path(
        self,
        spec_info: SpecializationInfo,
        context: Optional[ValidationContext]
    ) -> ValidationResult:
        """Validate specialization inheritance path."""
        messages = []

        try:
            # Validate base type exists
            if not self._base_elements.get(spec_info.base_type):
                messages.append(ValidationMessage(
                    path="inheritance",
                    message=f"Base type not found: {spec_info.base_type}",
                    severity=ValidationSeverity.ERROR,
                    code="base_type_not_found"
                ))
                return ValidationResult(is_valid=False, messages=messages)

            # Validate inheritance path
            current = spec_info.base_type
            for ancestor in spec_info.inheritance_path[1:]:
                if ancestor not in self._base_elements:
                    messages.append(ValidationMessage(
                        path="inheritance",
                        message=f"Invalid inheritance path at {ancestor}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_inheritance_path"
                    ))
                current = ancestor

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating inheritance path: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="inheritance",
                    message=f"Inheritance validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="inheritance_validation_error"
                )]
            )

    def _validate_specialization_constraints(
        self,
        content: Any,
        spec_info: SpecializationInfo,
        context: Optional[ValidationContext]
    ) -> ValidationResult:
        """Validate specialization constraints."""
        messages = []

        try:
            # Validate against constraints
            for constraint_name, constraint in spec_info.constraints.items():
                if not self._validate_constraint(content, constraint, context):
                    messages.append(ValidationMessage(
                        path=f"constraints.{constraint_name}",
                        message=f"Failed constraint: {constraint_name}",
                        severity=ValidationSeverity.ERROR,
                        code="constraint_violation"
                    ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating specialization constraints: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="constraints",
                    message=f"Constraint validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="constraint_validation_error"
                )]
            )

    def _validate_specialized_attributes(
        self,
        content: Any,
        spec_info: SpecializationInfo,
        context: Optional[ValidationContext]
    ) -> ValidationResult:
        """Validate specialized attributes."""
        messages = []

        try:
            # Get base attributes
            base_element = self._base_elements.get(spec_info.base_type)
            if not base_element:
                return ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path="attributes",
                        message=f"Base element not found: {spec_info.base_type}",
                        severity=ValidationSeverity.ERROR,
                        code="base_element_not_found"
                    )]
                )

            # Validate required attributes
            for attr_name, attr in base_element.attributes.items():
                if attr.is_required and not hasattr(content, attr_name):
                    messages.append(ValidationMessage(
                        path=f"attributes.{attr_name}",
                        message=f"Missing required attribute: {attr_name}",
                        severity=ValidationSeverity.ERROR,
                        code="missing_required_attribute"
                    ))

            # Validate specialized attributes
            for attr_name, attr in spec_info.attributes.items():
                if not self._validate_attribute(content, attr_name, attr, context):
                    messages.append(ValidationMessage(
                        path=f"attributes.{attr_name}",
                        message=f"Invalid specialized attribute: {attr_name}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_specialized_attribute"
                    ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating specialized attributes: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="attributes",
                    message=f"Attribute validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="attribute_validation_error"
                )]
            )

    def _validate_constraint(
        self,
        content: Any,
        constraint: Dict[str, Any],
        context: Optional[ValidationContext]
    ) -> bool:
        """Validate individual constraint."""
        try:
            constraint_type = constraint.get('type')
            if constraint_type == 'attribute':
                return self._validate_attribute_constraint(
                    content,
                    constraint,
                    context
                )
            elif constraint_type == 'content':
                return self._validate_content_constraint(
                    content,
                    constraint,
                    context
                )
            elif constraint_type == 'structural':
                return self._validate_structural_constraint(
                    content,
                    constraint,
                    context
                )
            return True  # Unknown constraint types pass

        except Exception as e:
            self.logger.error(f"Error validating constraint: {str(e)}")
            return False

    def _validate_attribute(
        self,
        content: Any,
        attr_name: str,
        attr_def: Any,
        context: Optional[ValidationContext]
    ) -> bool:
        """Validate specialized attribute."""
        try:
            if not hasattr(content, attr_name):
                return not attr_def.is_required

            value = getattr(content, attr_name)
            if attr_def.allowed_values:
                return value in attr_def.allowed_values

            return True

        except Exception as e:
            self.logger.error(f"Error validating attribute {attr_name}: {str(e)}")
            return False

    def _validate_attribute_constraint(
            self,
            content: Any,
            constraint: Dict[str, Any],
            context: Optional[ValidationContext]
        ) -> bool:
            """Validate attribute-specific constraints."""
            try:
                attr_name = constraint.get('attribute')
                if not attr_name:
                    return False

                # Get attribute value
                if not hasattr(content, attr_name):
                    return False

                value = getattr(content, attr_name)

                # Check value constraints
                if allowed_values := constraint.get('allowed_values'):
                    return value in allowed_values

                # Check pattern constraint
                if pattern := constraint.get('pattern'):
                    return bool(re.match(pattern, str(value)))

                # Check type constraint
                if expected_type := constraint.get('type'):
                    if expected_type == 'string':
                        return isinstance(value, str)
                    elif expected_type == 'number':
                        return isinstance(value, (int, float))
                    elif expected_type == 'boolean':
                        return isinstance(value, bool)

                return True

            except Exception as e:
                self.logger.error(f"Error validating attribute constraint: {str(e)}")
                return False

    def _validate_content_constraint(
        self,
        content: Any,
        constraint: Dict[str, Any],
        context: Optional[ValidationContext]
    ) -> bool:
        """Validate content model constraints."""
        try:
            # Check allowed elements
            if allowed_elements := constraint.get('allowed_elements'):
                for child in getattr(content, 'children', []):
                    if child.tag not in allowed_elements:
                        return False

            # Check required elements
            if required_elements := constraint.get('required_elements'):
                existing_elements = {
                    child.tag for child in getattr(content, 'children', [])
                }
                if not all(elem in existing_elements for elem in required_elements):
                    return False

            # Check content pattern
            if pattern := constraint.get('content_pattern'):
                content_str = getattr(content, 'text', '')
                if not re.match(pattern, content_str):
                    return False

            # Check mixed content constraints
            if constraint.get('allow_mixed') is False:
                if getattr(content, 'text', '').strip():
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating content constraint: {str(e)}")
            return False

    def _validate_structural_constraint(
        self,
        content: Any,
        constraint: Dict[str, Any],
        context: Optional[ValidationContext]
    ) -> bool:
        """Validate structural constraints."""
        try:
            # Check parent constraints
            if allowed_parents := constraint.get('allowed_parents'):
                parent = getattr(content, 'parent', None)
                if parent and parent.tag not in allowed_parents:
                    return False

            # Check sibling constraints
            if allowed_siblings := constraint.get('allowed_siblings'):
                parent = getattr(content, 'parent', None)
                if parent:
                    siblings = [
                        child.tag for child in parent.children
                        if child != content
                    ]
                    if not all(sib in allowed_siblings for sib in siblings):
                        return False

            # Check nesting level
            if max_nesting := constraint.get('max_nesting'):
                current = content
                depth = 0
                while getattr(current, 'parent', None):
                    depth += 1
                    current = current.parent
                    if depth > max_nesting:
                        return False

            # Check structure pattern
            if pattern := constraint.get('structure_pattern'):
                structure = self._get_element_structure(content)
                if not re.match(pattern, structure):
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating structural constraint: {str(e)}")
            return False

    def _get_element_structure(self, element: Any) -> str:
        """Get string representation of element structure."""
        try:
            parts = [element.tag]
            for child in getattr(element, 'children', []):
                parts.append(self._get_element_structure(child))
            return f"({' '.join(parts)})"
        except Exception as e:
            self.logger.error(f"Error getting element structure: {str(e)}")
            return element.tag


    def cleanup(self) -> None:
        """Clean up validator resources."""
        try:
            self._specialization_cache.clear()
            self._inheritance_chains.clear()
            self._base_elements.clear()
            self._active_validations.clear()
            self._validation_depth = 0
            super().cleanup()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise
