from app.dita.types import ValidationContext
"""Inheritance validation for DITA content with specialization awareness."""
from typing import Dict, List, Optional, Any, Set, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Core validators and base classes
from .base_validator import BaseValidator

# Event and Cache
from ..main.context_manager import ContextManager
from ..events.event_manager import EventManager
from ..cache.cache import ContentCache
from ..utils.logger import DITALogger

# Core types
from ..types import (
    ElementType,
    ProcessingPhase,
    ProcessingState,
    ProcessingContext,
    ValidationResult,
    ValidationMessage,
    ValidationSeverity,
    ValidationContext,
    ValidationState,
    ValidationType,
    ValidationContext,
    ContentScope,
    DTDElement,
    ContentModel,
    SpecializationInfo,
    CacheEntryType
)

# Forward references for type checking
if TYPE_CHECKING:
    from ..config.config_manager import ConfigManager
    from ..metadata.metadata_manager import MetadataManager
    from ..dtd.dtd_mapper import DTDSchemaMapper
    from ..dtd.dtd_resolver import DTDResolver
    from .validation_manager import ValidationManager


class InheritanceValidator(BaseValidator):
    """Handles validation of inheritance relationships in DITA content."""

    def __init__(
        self,
        validation_manager: 'ValidationManager',
        event_manager: EventManager,
        config_manager: 'ConfigManager',
        metadata_manager: 'MetadataManager',
        content_cache: ContentCache,
        context_manager: 'ContextManager',
        dtd_mapper: 'DTDSchemaMapper',
        dtd_resolver: 'DTDResolver',
        logger: Optional[DITALogger] = None
    ):
        super().__init__(
            event_manager=event_manager,
            context_manager=context_manager,
            config_manager=config_manager,
            cache=content_cache,
            validator_type=ValidationType.INHERITANCE,
            logger=logger
        )

        # Store dependencies
        self.validation_manager = validation_manager
        self.metadata_manager = metadata_manager
        self.dtd_mapper = dtd_mapper
        self.dtd_resolver = dtd_resolver

        # Inheritance tracking
        self._inheritance_chains: Dict[str, List[str]] = {}
        self._base_elements: Dict[str, DTDElement] = {}
        self._specialization_cache: Dict[str, SpecializationInfo] = {}

        # Validation tracking
        self._active_validations: Dict[str, ValidationState] = {}
        self._max_validation_depth = 10  # Configurable

        # Initialize
        self._load_validator_config()
        self._register_validation_patterns()

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
                # Initialize validation state
                validation_state = ValidationState(
                    context=context or ValidationContext(dtd_ref=None),
                    validation_type=ValidationType.INHERITANCE
                )

                element_id = getattr(content, 'id', str(hash(content)))
                self._active_validations[element_id] = validation_state

                try:
                    # Get specialization info
                    spec_info = self.dtd_mapper.get_specialization_info(
                        getattr(content, 'tag', None)
                    )

                    if not spec_info:
                        return ValidationResult(is_valid=True, messages=[])

                    # Validate inheritance chain
                    validation_state.add_to_chain(element_id)

                    # Perform validations
                    self._validate_inheritance_chain(content, spec_info, validation_state)
                    self._validate_attribute_inheritance(content, spec_info, validation_state)
                    self._validate_content_model_inheritance(content, spec_info, validation_state)
                    self._validate_specialization_constraints(content, spec_info, validation_state)

                    return ValidationResult(
                        is_valid=not validation_state.has_errors,
                        messages=validation_state.errors + validation_state.warnings
                    )

                finally:
                    validation_state.remove_from_chain(element_id)
                    self._active_validations.pop(element_id, None)

            except Exception as e:
                self.logger.error(f"Error in inheritance validation: {str(e)}")
                return ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path="",
                        message=f"Inheritance validation error: {str(e)}",
                        severity=ValidationSeverity.ERROR,
                        code="validation_error"
                    )]
                )

    def _validate_inheritance_chain(
        self,
        content: Any,
        spec_info: SpecializationInfo,
        state: ValidationState
    ) -> None:
        """Validate inheritance chain integrity."""
        try:
            # Validate base type exists
            if not self._get_base_element(spec_info.base_type):
                state.errors.append(ValidationMessage(
                    path=f"inheritance.{spec_info.base_type}",
                    message=f"Base type not found: {spec_info.base_type}",
                    severity=ValidationSeverity.ERROR,
                    code="base_type_not_found"
                ))
                return

            # Validate inheritance path
            current = spec_info.base_type
            for ancestor in spec_info.inheritance_path[1:]:
                if ancestor not in self._base_elements:
                    state.errors.append(ValidationMessage(
                        path=f"inheritance.{ancestor}",
                        message=f"Invalid inheritance path at {ancestor}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_inheritance_path"
                    ))
                current = ancestor

        except Exception as e:
            self.logger.error(f"Error validating inheritance chain: {str(e)}")
            state.errors.append(ValidationMessage(
                path="inheritance",
                message=f"Chain validation error: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="chain_validation_error"
            ))

    def _validate_attribute_inheritance(
        self,
        content: Any,
        spec_info: SpecializationInfo,
        state: ValidationState
    ) -> None:
        """Validate attribute inheritance rules."""
        try:
            base_element = self._get_base_element(spec_info.base_type)
            if not base_element:
                return

            # Check required attributes
            for attr_name, attr_def in base_element.attributes.items():
                if attr_def.is_required and not hasattr(content, attr_name):
                    state.errors.append(ValidationMessage(
                        path=f"attributes.{attr_name}",
                        message=f"Missing required base attribute: {attr_name}",
                        severity=ValidationSeverity.ERROR,
                        code="missing_base_attribute"
                    ))

            # Validate specialized attributes
            for attr_name, attr_def in spec_info.attributes.items():
                if not self._validate_attribute_specialization(
                    content, attr_name, attr_def, base_element, state
                ):
                    state.errors.append(ValidationMessage(
                        path=f"attributes.{attr_name}",
                        message=f"Invalid attribute specialization: {attr_name}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_attribute_specialization"
                    ))

        except Exception as e:
            self.logger.error(f"Error validating attribute inheritance: {str(e)}")
            state.errors.append(ValidationMessage(
                path="attributes",
                message=f"Attribute inheritance error: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="attribute_inheritance_error"
            ))

    def _validate_content_model_inheritance(
        self,
        content: Any,
        spec_info: SpecializationInfo,
        state: ValidationState
    ) -> None:
        """Validate content model inheritance rules."""
        try:
            base_element = self._get_base_element(spec_info.base_type)
            if not base_element:
                return

            # Get content models
            base_model = base_element.content_model
            spec_model = self._get_content_model(content)

            if not spec_model:
                state.errors.append(ValidationMessage(
                    path="content_model",
                    message="Missing specialized content model",
                    severity=ValidationSeverity.ERROR,
                    code="missing_content_model"
                ))
                return

            # Validate model compatibility
            if not self._validate_content_model_compatibility(
                base_model, spec_model, state
            ):
                state.errors.append(ValidationMessage(
                    path="content_model",
                    message="Incompatible content model specialization",
                    severity=ValidationSeverity.ERROR,
                    code="incompatible_content_model"
                ))

        except Exception as e:
            self.logger.error(f"Error validating content model inheritance: {str(e)}")
            state.errors.append(ValidationMessage(
                path="content_model",
                message=f"Content model inheritance error: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="content_model_inheritance_error"
            ))

    def _validate_specialization_constraints(
        self,
        content: Any,
        spec_info: SpecializationInfo,
        state: ValidationState
    ) -> None:
        """Validate specialization-specific constraints."""
        try:
            for constraint_name, constraint in spec_info.constraints.items():
                if not self._validate_constraint(content, constraint, state):
                    state.errors.append(ValidationMessage(
                        path=f"constraints.{constraint_name}",
                        message=f"Failed constraint: {constraint_name}",
                        severity=ValidationSeverity.ERROR,
                        code="constraint_violation"
                    ))

        except Exception as e:
            self.logger.error(f"Error validating specialization constraints: {str(e)}")
            state.errors.append(ValidationMessage(
                path="constraints",
                message=f"Constraint validation error: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="constraint_validation_error"
            ))

    def _validate_attribute_specialization(
        self,
        content: Any,
        attr_name: str,
        attr_def: Any,
        base_element: DTDElement,
        state: ValidationState
    ) -> bool:
        """Validate single attribute specialization."""
        try:
            # Check if attribute is allowed to be specialized
            if attr_name in base_element.attributes:
                base_attr = base_element.attributes[attr_name]
                if base_attr.is_required and not hasattr(content, attr_name):
                    return False

            # Validate attribute type
            if hasattr(content, attr_name):
                value = getattr(content, attr_name)
                if not self._validate_attribute_value(value, attr_def):
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating attribute specialization: {str(e)}")
            return False

    def _validate_content_model_compatibility(
        self,
        base_model: ContentModel,
        spec_model: ContentModel,
        state: ValidationState
    ) -> bool:
        """Validate content model compatibility."""
        try:
            # Type compatibility
            if base_model.type != spec_model.type:
                return False

            # Element compatibility
            base_elements = set(base_model.elements)
            spec_elements = set(spec_model.elements)

            # Specialized model must include all base elements
            if not base_elements.issubset(spec_elements):
                return False

            # Ordering compatibility
            if base_model.ordering == "sequence" and spec_model.ordering != "sequence":
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating content model compatibility: {str(e)}")
            return False

    def _validate_constraint(
        self,
        content: Any,
        constraint: Dict[str, Any],
        state: ValidationState
    ) -> bool:
        """Validate single specialization constraint."""
        try:
            constraint_type = constraint.get('type')
            if constraint_type == 'attribute':
                return self._validate_attribute_constraint(content, constraint)
            elif constraint_type == 'content':
                return self._validate_content_constraint(content, constraint)
            elif constraint_type == 'structural':
                return self._validate_structural_constraint(content, constraint)
            return True  # Unknown constraint types pass

        except Exception as e:
            self.logger.error(f"Error validating constraint: {str(e)}")
            return False

    def _get_base_element(self, element_type: str) -> Optional[DTDElement]:
        """Get base element definition from cache."""
        return self._base_elements.get(element_type)

    def _get_content_model(self, content: Any) -> Optional[ContentModel]:
        """Extract content model from content."""
        try:
            if hasattr(content, 'tag'):
                if spec_info := self.dtd_mapper.get_specialization_info(content.tag):
                    return spec_info.content_model
            return None
        except Exception as e:
            self.logger.error(f"Error getting content model: {str(e)}")
            return None

    def cleanup(self) -> None:
        """Clean up validator resources."""
        try:
            self._inheritance_chains.clear()
            self._base_elements.clear()
            self._specialization_cache.clear()
            self._active_validations.clear()
            super().cleanup()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise
