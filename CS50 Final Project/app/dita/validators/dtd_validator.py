"""DTD validation for DITA content."""
from typing import Dict, List, Optional, Any, Union, Sequence
from xml.etree import ElementTree
from datetime import datetime
from pathlib import Path
import sys
import re

# Managers
from ..validators.validation_manager import ValidationManager, ValidationPattern
from ..utils.logger import DITALogger
from ..dtd.dtd_mapper import DTDSchemaMapper
from ..dtd.dtd_resolver import DTDResolver

# Types
from ..types import (
    ProcessingContext,
    ValidationContext,
    ContentModelType,
    OccurrenceIndicator,
    ContentModel,
    ContentModelParticle,
    DTDParsingResult,
    DTDElement,
    AttributeType,
    DTDAttribute,
    SpecializationInfo,
    ValidationResult,
    ValidationState,
    ValidationMessage,
    ValidationSeverity
)



class DTDValidator:
    """DTD validation for DITA content."""

    def __init__(
        self,
        resolver: DTDResolver,
        dtd_mapper: DTDSchemaMapper,
        validation_manager: ValidationManager,
        logger: Optional[DITALogger] = None
    ):
        """Initialize DTD validator.

        Args:
            resolver: DTD resolver for loading DTDs
            specialization_handler: Handler for DTD specializations
            validation_manager: System validation manager
            logger: Optional logger
        """
        # Core dependencies
        self.resolver = resolver
        self.dtd_mapper = dtd_mapper
        self.validation_manager = validation_manager
        self.logger = logger or DITALogger(__name__)
        self._register_validation_patterns()

        # Initialize handlers
        self._initialize_handlers()


        # State tracking
        self._active_validations: Dict[str, ValidationState] = {}
        self._parsed_dtds: Dict[Path, DTDParsingResult] = {}
        self._element_cache: Dict[str, DTDElement] = {}

        # Managers
        self.validation_manager.register_dtd_validators(self)


        # Configure validation settings
        self._validation_config = {
            'allow_unknown_elements': False,
            'strict_attribute_checking': True,
            'validate_specializations': True,
            'max_recursion_depth': 100,
            'cache_validation_results': True,
            'enable_dtd_caching': True
        }

    def _register_validation_patterns(self) -> None:
            """Register DTD-specific validation patterns."""
            try:
                # Structure validation patterns
                self.validation_manager.register_pattern(ValidationPattern(
                    pattern_id="dtd.structure.element_definition",
                    description="Validate element structure against DTD",
                    severity=ValidationSeverity.ERROR
                ))

                # Attribute validation patterns
                self.validation_manager.register_pattern(ValidationPattern(
                    pattern_id="dtd.attributes.required",
                    description="Validate required attributes",
                    severity=ValidationSeverity.ERROR
                ))

                # Specialization validation patterns
                self.validation_manager.register_pattern(ValidationPattern(
                    pattern_id="dtd.specialization.inheritance",
                    description="Validate specialization inheritance",
                    severity=ValidationSeverity.ERROR
                ))

            except Exception as e:
                self.logger.error(f"Error registering validation patterns: {str(e)}")
                raise

    def _initialize_handlers(self) -> None:
            """Initialize validation handlers."""
            try:
                # Register validators with validation manager
                if self.validation_manager:
                    # Register DTD content validation
                    self.validation_manager.register_validator(
                        validation_type="dtd",
                        name="content",
                        validator=self.validate_content,
                        position=0  # DTD validation runs first
                    )

                    # Register specialized validators
                    self.validation_manager.register_dtd_validators(self)

                self.logger.debug("DTD validator handlers initialized")

            except Exception as e:
                self.logger.error(f"Error initializing DTD validator handlers: {str(e)}")
                raise

    def _init_dtd_cache(self) -> None:
        """Initialize DTD caching."""
        try:
            # Clear existing caches
            self._parsed_dtds.clear()
            self._element_cache.clear()

            # Pre-load common DTDs if needed
            self._preload_common_dtds()

        except Exception as e:
            self.logger.error(f"Error initializing DTD cache: {str(e)}")

    def _preload_common_dtds(self) -> None:
        """Pre-load commonly used DTDs."""
        common_dtds = [
            "topic.dtd",
            "concept.dtd",
            "task.dtd",
            "reference.dtd",
            "map.dtd"
        ]

        for dtd_name in common_dtds:
            try:
                dtd_path = self.resolver.base_path / dtd_name
                if dtd_path.exists():
                    self._parsed_dtds[dtd_path] = self._parse_dtd(dtd_path.read_text())
            except Exception as e:
                self.logger.warning(f"Error pre-loading DTD {dtd_name}: {str(e)}")

    def validate_content(
            self,
            content: Union[str, ElementTree.Element],
            dtd_path: Path,
            context: Optional[ProcessingContext] = None
        ) -> ValidationResult:
            """Validate content against DTD using ValidationManager."""
            try:
                # Parse if string content
                root = (
                    ElementTree.fromstring(content)
                    if isinstance(content, str)
                    else content
                )

                # Get DTD information
                dtd_ref = self.resolver.resolve_dtd(dtd_path)
                if not dtd_ref.resolved_path:
                    return ValidationResult(
                        is_valid=False,
                        messages=[ValidationMessage(
                            path="dtd",
                            message=f"Could not resolve DTD: {dtd_path}",
                            severity=ValidationSeverity.ERROR,
                            code="dtd_not_found"
                        )]
                    )

                # Create validation context
                validation_ctx = ValidationContext(dtd_ref=dtd_ref)

                # Prepare validation context for manager
                validation_context = {
                    "dtd_reference": dtd_ref,
                    "validation_context": validation_ctx,
                    "element_type": root.tag,
                    "processing_context": context,
                    "validation_config": self._validation_config
                }

                # Register validators if not already registered
                self._ensure_validators_registered()

                # Use validation manager
                return self.validation_manager.validate(
                    content=root,
                    validation_type="dtd",
                    context=validation_context
                )

            except Exception as e:
                self.logger.error(f"Error validating content: {str(e)}")
                return ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path="",
                        message=f"DTD validation error: {str(e)}",
                        severity=ValidationSeverity.ERROR,
                        code="validation_error"
                    )]
                )

    def _ensure_validators_registered(self) -> None:
        """Ensure DTD validators are registered with ValidationManager."""
        if not hasattr(self, '_validators_registered'):
            self.validation_manager.register_validator(
                validation_type="dtd",
                name="structure",
                validator=lambda content, **ctx: ValidationResult(
                    is_valid=not any(msg.severity == ValidationSeverity.ERROR
                                   for msg in self._validate_structure(content, ValidationState(context=ctx.get("validation_context", {})))),
                    messages=self._validate_structure(content, ValidationState(context=ctx.get("validation_context", {})))
                )
            )
            self.validation_manager.register_validator(
                validation_type="dtd",
                name="attributes",
                validator=lambda content, **ctx: ValidationResult(
                    is_valid=not any(msg.severity == ValidationSeverity.ERROR
                                   for msg in self._validate_attributes(content, ValidationState(context=ctx.get("validation_context", {})))),
                    messages=self._validate_attributes(content, ValidationState(context=ctx.get("validation_context", {})))
                )
            )
            self.validation_manager.register_validator(
                validation_type="dtd",
                name="specialization",
                validator=lambda content, **ctx: ValidationResult(
                    is_valid=not any(msg.severity == ValidationSeverity.ERROR
                                   for msg in self._validate_specializations(content, ValidationState(context=ctx.get("validation_context", {})))),
                    messages=self._validate_specializations(content, ValidationState(context=ctx.get("validation_context", {})))
                )
            )
            self._validators_registered = True

    def validate_element(
        self,
        element: ElementTree.Element,
        dtd_path: Path,
        context: Optional[ProcessingContext] = None,
        validation_type: str = "all"  # all, structure, attributes, or specialization
    ) -> ValidationResult:
        """
        Validate single element against DTD.

        Args:
            element: Element to validate
            dtd_path: Path to DTD file
            context: Optional processing context
            validation_type: Type of validation to perform

        Returns:
            ValidationResult with validation status and messages
        """
        try:
            # Resolve DTD
            dtd_ref = self.resolver.resolve_dtd(dtd_path)
            if not dtd_ref.resolved_path:
                return ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path=element.tag,
                        message=f"Could not resolve DTD: {dtd_path}",
                        severity=ValidationSeverity.ERROR,
                        code="dtd_not_found"
                    )]
                )

            # Create validation context
            validation_ctx = ValidationContext(dtd_ref=dtd_ref)

            # Initialize validation state
            state = ValidationState(context=validation_ctx)
            self._active_validations[element.tag] = state

            try:
                messages = []

                # Perform requested validations
                if validation_type in ("all", "structure"):
                    structure_result = self._validate_structure(element, state)
                    messages.extend(structure_result)

                if validation_type in ("all", "attributes"):
                    attribute_result = self._validate_attributes(element, state)
                    messages.extend(attribute_result)

                if validation_type in ("all", "specialization"):
                    if self._validation_config['validate_specializations']:
                        spec_result = self._validate_specializations(element, state)
                        messages.extend(spec_result)

                return ValidationResult(
                    is_valid=not any(
                        msg.severity == ValidationSeverity.ERROR
                        for msg in messages
                    ),
                    messages=messages
                )

            finally:
                # Clean up validation state
                self._active_validations.pop(element.tag, None)

        except Exception as e:
            self.logger.error(f"Error validating element: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path=element.tag,
                    message=f"Validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def _validate_specializations(
        self,
        element: ElementTree.Element,
        state: ValidationState
    ) -> List[ValidationMessage]:
        """Validate element specializations."""
        messages = []

        # Get specialization info - already returns SpecializationInfo
        spec_info = self.dtd_mapper.get_specialization_info(element.tag)
        if spec_info:
            messages.extend(self._validate_against_base(element, spec_info, state))
            messages.extend(self._validate_specialization_constraints(
                element, spec_info, state
            ))

        return messages

    def _parse_dtd(self, content: str) -> DTDParsingResult:
        """Parse DTD content into internal model using DTDMapper."""
        try:
            # Use mapper to parse content
            parsing_result = self.dtd_mapper.parse_dtd(Path('inline.dtd'))  # Virtual path for inline content

            # Return parsing result directly since DTDMapper already returns DTDParsingResult
            return parsing_result

        except Exception as e:
            self.logger.error(f"Error parsing DTD: {str(e)}")
            return DTDParsingResult(
                elements={},
                entities={},
                specializations={},
                errors=[ValidationMessage(
                    path="dtd",
                    message=f"Error parsing DTD: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="parse_error"
                )],
                warnings=[],
                metadata={
                    'parsed_at': datetime.now().isoformat(),
                    'error': str(e)
                }
            )

    def validate_attribute(
        self,
        element_tag: str,
        attr_name: str,
        attr_value: str,
        dtd_path: Path,
        context: Optional[ProcessingContext] = None
    ) -> ValidationResult:
        """
        Validate single attribute against DTD definition.

        Args:
            element_tag: Element tag name
            attr_name: Attribute name
            attr_value: Attribute value
            dtd_path: Path to DTD file
            context: Optional processing context

        Returns:
            ValidationResult with validation status and messages
        """
        try:
            # Get DTD definition
            element_def = self._get_element_definition(element_tag, dtd_path)
            if not element_def:
                return ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path=f"{element_tag}@{attr_name}",
                        message=f"Unknown element: {element_tag}",
                        severity=ValidationSeverity.ERROR,
                        code="unknown_element"
                    )]
                )

            # Get attribute definition
            attr_def = element_def.attributes.get(attr_name)
            if not attr_def:
                if self._validation_config['strict_attribute_checking']:
                    return ValidationResult(
                        is_valid=False,
                        messages=[ValidationMessage(
                            path=f"{element_tag}@{attr_name}",
                            message=f"Unknown attribute: {attr_name}",
                            severity=ValidationSeverity.ERROR,
                            code="unknown_attribute"
                        )]
                    )
                return ValidationResult(is_valid=True, messages=[])

            # Validate against definition
            messages = []
            self._validate_attribute_value(
                element_tag,
                attr_name,
                attr_value,
                attr_def,
                messages
            )

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR
                    for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating attribute: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path=f"{element_tag}@{attr_name}",
                    message=f"Validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def validate_specialization(
        self,
        element: ElementTree.Element,
        base_dtd: Path,
        context: Optional[ProcessingContext] = None
    ) -> ValidationResult:
        """Validate element specialization."""
        try:
            # Use DTDMapper directly for specialization validation
            spec_info = self.dtd_mapper.get_specialization_info(
                element.tag
            )
            if not spec_info:
                return ValidationResult(is_valid=True, messages=[])

            # Resolve base DTD
            dtd_ref = self.resolver.resolve_dtd(base_dtd)
            if not dtd_ref.resolved_path:
                return ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path=element.tag,
                        message=f"Could not resolve base DTD: {base_dtd}",
                        severity=ValidationSeverity.ERROR,
                        code="dtd_not_found"
                    )]
                )

            # Create validation context
            validation_ctx = ValidationContext(dtd_ref=dtd_ref)

            # Initialize validation state
            state = ValidationState(context=validation_ctx)
            self._active_validations[element.tag] = state

            try:
                # Validate against base type
                messages = self._validate_against_base(element, spec_info, state)

                # Validate specialization constraints
                constraint_messages = self._validate_specialization_constraints(
                    element, spec_info, state
                )
                messages.extend(constraint_messages)

                return ValidationResult(
                    is_valid=not any(
                        msg.severity == ValidationSeverity.ERROR
                        for msg in messages
                    ),
                    messages=messages
                )

            finally:
                # Clean up validation state
                self._active_validations.pop(element.tag, None)

        except Exception as e:
            self.logger.error(f"Error validating specialization: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path=element.tag,
                    message=f"Specialization validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def _get_element_definition(
            self,
            element_tag: str,
            dtd_path: Path
        ) -> Optional[DTDElement]:
            """
            Get element definition from DTD.

            Args:
                element_tag: Element tag name
                dtd_path: Path to DTD file

            Returns:
                Optional[DTDElement]: Element definition if found
            """
            # Check cache first
            cache_key = f"{dtd_path}#{element_tag}"
            if element_def := self._element_cache.get(cache_key):
                return element_def

            # Load DTD if not already parsed
            if dtd_path not in self._parsed_dtds:
                try:
                    content = self.resolver.load_dtd_content(dtd_path)
                    self._parsed_dtds[dtd_path] = self._parse_dtd(content)
                except Exception as e:
                    self.logger.error(f"Error loading DTD {dtd_path}: {str(e)}")
                    return None

            # Get element from parsed DTD
            if parsed_dtd := self._parsed_dtds.get(dtd_path):
                if element_def := parsed_dtd.elements.get(element_tag):
                    # Cache and return
                    self._element_cache[cache_key] = element_def
                    return element_def

            return None

    def _validate_structure(
        self,
        element: ElementTree.Element,
        state: ValidationState
    ) -> List[ValidationMessage]:
        """
        Validate element structure recursively.

        Args:
            element: Element to validate
            state: Current validation state

        Returns:
            List[ValidationMessage]: Validation messages
        """
        messages: List[ValidationMessage] = []

        try:
            # Get element definition
            element_def = self._get_element_definition(
                element.tag,
                state.context.dtd_ref.path
            )
            if not element_def:
                if not self._validation_config['allow_unknown_elements']:
                    messages.append(ValidationMessage(
                        path=element.tag,
                        message=f"Unknown element: {element.tag}",
                        severity=ValidationSeverity.ERROR,
                        code="unknown_element"
                    ))
                return messages

            # Update validation state
            state.current_element = element_def
            if state.current_element:
                state.parent_elements.append(state.current_element)

            try:
                # Validate content model
                content_messages = self._validate_content_model(
                    element,
                    element_def.content_model,
                    state
                )
                messages.extend(content_messages)

                # Validate children recursively
                for child in element:
                    child_messages = self._validate_structure(child, state)
                    messages.extend(child_messages)

            finally:
                # Restore state
                if state.current_element:
                    state.parent_elements.pop()

        except Exception as e:
            messages.append(ValidationMessage(
                path=element.tag,
                message=f"Structure validation error: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="validation_error"
            ))

        return messages

    def _validate_content_model(
        self,
        element: ElementTree.Element,
        content_model: ContentModel,
        state: ValidationState
    ) -> List[ValidationMessage]:
        """
        Validate element against content model.

        Args:
            element: Element to validate
            content_model: Content model to validate against
            state: Current validation state

        Returns:
            List[ValidationMessage]: Validation messages
        """
        messages: List[ValidationMessage] = []

        try:
            if content_model.type == ContentModelType.EMPTY:
                if len(element) > 0 or element.text:
                    messages.append(ValidationMessage(
                        path=element.tag,
                        message="Empty element cannot have content",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_content"
                    ))

            elif content_model.type == ContentModelType.TEXT:
                if len(element) > 0:
                    messages.append(ValidationMessage(
                        path=element.tag,
                        message="Text-only element cannot have child elements",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_content"
                    ))

            else:  # Complex content models
                messages.extend(self._validate_complex_content(
                    element, content_model, state
                ))

        except Exception as e:
            messages.append(ValidationMessage(
                path=element.tag,
                message=f"Content model validation error: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="validation_error"
            ))

        return messages

    def _validate_complex_content(
        self,
        element: ElementTree.Element,
        content_model: ContentModel,
        state: ValidationState
    ) -> List[ValidationMessage]:
        """Validate complex content model."""
        messages: List[ValidationMessage] = []

        try:
            child_elements = list(element)
            total_children = len(child_elements)

            # Validate against occurrence constraints
            min_occur = content_model.occurrence.min
            max_occur = content_model.occurrence.max

            if total_children < min_occur:
                messages.append(ValidationMessage(
                    path=element.tag,
                    message=f"Element must occur at least {min_occur} times",
                    severity=ValidationSeverity.ERROR,
                    code="min_occurrence_violation"
                ))

            if max_occur != sys.maxsize and total_children > max_occur:
                messages.append(ValidationMessage(
                    path=element.tag,
                    message=f"Element cannot occur more than {max_occur} times",
                    severity=ValidationSeverity.ERROR,
                    code="max_occurrence_violation"
                ))

            # Convert elements to ContentModelParticle sequence
            particles: Sequence[ContentModelParticle] = [
                ContentModelParticle(
                    content=element_name,
                    type=ContentModelType.ELEMENT,
                    occurrence=OccurrenceIndicator.NONE,
                    is_group=False
                )
                for element_name in content_model.elements
            ]

            # Validate content model structure
            if content_model.type == ContentModelType.SEQUENCE.value:
                messages.extend(self._validate_sequence(
                    child_elements,
                    particles,
                    element.tag,
                    state
                ))

            elif content_model.type == ContentModelType.CHOICE.value:
                messages.extend(self._validate_choice(
                    child_elements,
                    particles,
                    element.tag,
                    state
                ))

            elif content_model.mixed:
                messages.extend(self._validate_mixed_content(
                    child_elements,
                    particles,
                    element.tag,
                    state
                ))

            return messages

        except Exception as e:
            messages.append(ValidationMessage(
                path=element.tag,
                message=f"Complex content validation error: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="validation_error"
            ))
            return messages

    def _validate_sequence(
        self,
        children: List[ElementTree.Element],
        particles: Sequence[ContentModelParticle],
        parent_tag: str,
        state: ValidationState
    ) -> List[ValidationMessage]:
        """Validate sequence content model."""
        messages: List[ValidationMessage] = []

        try:
            expected_sequence = [p.content for p in particles if isinstance(p.content, str)]
            child_sequence = [child.tag for child in children]

            # Check sequence matches
            i = j = 0
            while i < len(child_sequence) and j < len(expected_sequence):
                if child_sequence[i] == expected_sequence[j]:
                    i += 1
                    j += 1
                else:
                    messages.append(ValidationMessage(
                        path=f"{parent_tag}/{child_sequence[i]}",
                        message=f"Invalid sequence. Expected {expected_sequence[j]}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_sequence"
                    ))
                    i += 1

            # Check for extra children
            if i < len(child_sequence):
                messages.append(ValidationMessage(
                    path=parent_tag,
                    message="Extra children not allowed in sequence",
                    severity=ValidationSeverity.ERROR,
                    code="extra_children"
                ))

            # Check for missing required elements
            if j < len(expected_sequence):
                messages.append(ValidationMessage(
                    path=parent_tag,
                    message=f"Missing required element: {expected_sequence[j]}",
                    severity=ValidationSeverity.ERROR,
                    code="missing_element"
                ))

        except Exception as e:
            messages.append(ValidationMessage(
                path=parent_tag,
                message=f"Sequence validation error: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="validation_error"
            ))

        return messages

    def _validate_choice(
        self,
        children: List[ElementTree.Element],
        particles: Sequence[ContentModelParticle],
        parent_tag: str,
        state: ValidationState
    ) -> List[ValidationMessage]:
        """Validate choice content model."""
        messages: List[ValidationMessage] = []

        try:
            allowed_choices = set(
                p.content for p in particles
                if isinstance(p.content, str)
            )

            for child in children:
                if child.tag not in allowed_choices:
                    messages.append(ValidationMessage(
                        path=f"{parent_tag}/{child.tag}",
                        message=f"Invalid choice. Allowed: {allowed_choices}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_choice"
                    ))

        except Exception as e:
            messages.append(ValidationMessage(
                path=parent_tag,
                message=f"Choice validation error: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="validation_error"
            ))

        return messages

    def _validate_mixed_content(
        self,
        children: List[ElementTree.Element],
        particles: Sequence[ContentModelParticle],
        parent_tag: str,
        state: ValidationState
    ) -> List[ValidationMessage]:
        """Validate mixed content model."""
        messages: List[ValidationMessage] = []

        try:
            # Use particles parameter instead of undefined allowed_elements
            allowed = set(
                p.content for p in particles
                if isinstance(p.content, str)
            )

            for child in children:
                if child.tag not in allowed:
                    messages.append(ValidationMessage(
                        path=f"{parent_tag}/{child.tag}",
                        message=f"Element not allowed in mixed content",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_mixed_content"
                    ))

        except Exception as e:
            messages.append(ValidationMessage(
                path=parent_tag,
                message=f"Mixed content validation error: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="validation_error"
            ))

        return messages

    def _validate_attributes(
        self,
        element: ElementTree.Element,
        state: ValidationState
    ) -> List[ValidationMessage]:
        """Validate element attributes."""
        messages: List[ValidationMessage] = []

        try:
            # Get element definition
            if not state.current_element:
                return messages

            # Check required attributes
            for attr_name, attr_def in state.current_element.attributes.items():
                if attr_def.is_required and attr_name not in element.attrib:
                    messages.append(ValidationMessage(
                        path=f"{element.tag}@{attr_name}",
                        message="Missing required attribute",
                        severity=ValidationSeverity.ERROR,
                        code="missing_attribute"
                    ))

            # Validate each attribute
            for attr_name, attr_value in element.attrib.items():
                if attr_def := state.current_element.attributes.get(attr_name):
                    messages.extend(self._validate_attribute_value(
                        element.tag,
                        attr_name,
                        attr_value,
                        attr_def,
                        messages
                    ))
                elif self._validation_config['strict_attribute_checking']:
                    messages.append(ValidationMessage(
                        path=f"{element.tag}@{attr_name}",
                        message="Unknown attribute",
                        severity=ValidationSeverity.ERROR,
                        code="unknown_attribute"
                    ))

        except Exception as e:
            messages.append(ValidationMessage(
                path=element.tag,
                message=f"Attribute validation error: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="validation_error"
            ))

        return messages

    def _validate_attribute_value(
        self,
        element_tag: str,
        attr_name: str,
        attr_value: str,
        attr_def: DTDAttribute,
        messages: List[ValidationMessage]
    ) -> List[ValidationMessage]:
        """Validate attribute value against its definition."""
        validation_messages = []
        try:
            # Type validation
            if not self._validate_attribute_type(attr_value, attr_def.type):
                validation_messages.append(ValidationMessage(
                    path=f"{element_tag}@{attr_name}",
                    message=f"Invalid type for attribute {attr_name}",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_attribute_type"
                ))
                return validation_messages

            # Value validation
            if attr_def.allowed_values and attr_value not in attr_def.allowed_values:
                validation_messages.append(ValidationMessage(
                    path=f"{element_tag}@{attr_name}",
                    message=f"Invalid value for {attr_name}. Allowed: {attr_def.allowed_values}",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_attribute_value"
                ))

            return validation_messages

        except Exception as e:
            validation_messages.append(ValidationMessage(
                path=f"{element_tag}@{attr_name}",
                message=f"Error validating attribute: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="validation_error"
            ))
            return validation_messages

    def _validate_attribute_type(
        self,
        value: str,
        attr_type: AttributeType
    ) -> bool:
        """Validate attribute value against its type."""
        try:
            if attr_type == AttributeType.CDATA:
                return True  # CDATA accepts any string
            elif attr_type == AttributeType.ID:
                return bool(re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', value))
            elif attr_type == AttributeType.IDREF:
                # Should check if ID exists but need element context
                return bool(re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', value))
            elif attr_type == AttributeType.IDREFS:
                return all(bool(re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', ref))
                          for ref in value.split())
            elif attr_type == AttributeType.NMTOKEN:
                return bool(re.match(r'^[a-zA-Z0-9._-]+$', value))
            elif attr_type == AttributeType.NMTOKENS:
                return all(bool(re.match(r'^[a-zA-Z0-9._-]+$', token))
                          for token in value.split())
            return True
        except Exception:
            return False

    def _validate_against_base(
        self,
        element: ElementTree.Element,
        spec_info: SpecializationInfo,
        state: ValidationState
    ) -> List[ValidationMessage]:
        """Validate specialized element against its base type."""
        messages = []
        try:
            # Get base element definition
            base_def = self._get_element_definition(
                spec_info.base_type,
                state.context.dtd_ref.path
            )
            if not base_def:
                messages.append(ValidationMessage(
                    path=element.tag,
                    message=f"Base type not found: {spec_info.base_type}",
                    severity=ValidationSeverity.ERROR,
                    code="base_type_not_found"
                ))
                return messages

            # Validate content model
            content_messages = self._validate_content_model(
                element,
                base_def.content_model,
                state
            )
            messages.extend(content_messages)

            # Validate required attributes from base
            for attr_name, attr_def in base_def.attributes.items():
                if attr_def.is_required and attr_name not in element.attrib:
                    messages.append(ValidationMessage(
                        path=f"{element.tag}@{attr_name}",
                        message=f"Missing required attribute from base: {attr_name}",
                        severity=ValidationSeverity.ERROR,
                        code="missing_base_attribute"
                    ))

            return messages

        except Exception as e:
            messages.append(ValidationMessage(
                path=element.tag,
                message=f"Error validating against base: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="base_validation_error"
            ))
            return messages

    def _validate_specialization_constraints(
        self,
        element: ElementTree.Element,
        spec_info: SpecializationInfo,
        state: ValidationState
    ) -> List[ValidationMessage]:
        """Validate specialized element against its constraints."""
        messages = []
        try:
            # Validate structural constraints
            if constraints := spec_info.constraints.get('structural'):
                messages.extend(self._validate_structural_constraints(
                    element, constraints, state
                ))

            # Validate attribute constraints
            if constraints := spec_info.constraints.get('attributes'):
                messages.extend(self._validate_attribute_constraints(
                    element, constraints, state
                ))

            return messages

        except Exception as e:
            messages.append(ValidationMessage(
                path=element.tag,
                message=f"Error validating specialization constraints: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="constraint_validation_error"
            ))
            return messages

    def _validate_structural_constraints(
        self,
        element: ElementTree.Element,
        constraints: Dict[str, Any],
        state: ValidationState
    ) -> List[ValidationMessage]:
        """Validate structural constraints."""
        messages = []
        try:
            # Check element nesting constraints
            if allowed_parents := constraints.get('allowed_parents'):
                # Use find to get parent
                parent = element.find('..')
                parent_tag = parent.tag if parent is not None else None
                if parent_tag and parent_tag not in allowed_parents:
                    messages.append(ValidationMessage(
                        path=element.tag,
                        message=f"Invalid parent element: {parent_tag}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_parent"
                    ))

            # Check child element constraints
            if allowed_children := constraints.get('allowed_children'):
                for child in list(element):  # Use list() for proper iteration
                    if child.tag not in allowed_children:
                        messages.append(ValidationMessage(
                            path=f"{element.tag}/{child.tag}",
                            message=f"Invalid child element: {child.tag}",
                            severity=ValidationSeverity.ERROR,
                            code="invalid_child"
                        ))

            return messages

        except Exception as e:
            messages.append(ValidationMessage(
                path=element.tag,
                message=f"Error validating structural constraints: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="structural_validation_error"
            ))
            return messages

    def _validate_attribute_constraints(
        self,
        element: ElementTree.Element,
        constraints: Dict[str, Any],
        state: ValidationState
    ) -> List[ValidationMessage]:
        """Validate attribute constraints."""
        messages = []
        try:
            # Check required specialized attributes
            if required_attrs := constraints.get('required'):
                for attr in required_attrs:
                    if attr not in element.attrib:
                        messages.append(ValidationMessage(
                            path=f"{element.tag}@{attr}",
                            message=f"Missing required specialized attribute: {attr}",
                            severity=ValidationSeverity.ERROR,
                            code="missing_specialized_attribute"
                        ))

            # Check attribute value constraints
            if value_constraints := constraints.get('values', {}):
                for attr, allowed in value_constraints.items():
                    if value := element.get(attr):
                        if value not in allowed:
                            messages.append(ValidationMessage(
                                path=f"{element.tag}@{attr}",
                                message=f"Invalid specialized attribute value: {value}",
                                severity=ValidationSeverity.ERROR,
                                code="invalid_specialized_value"
                            ))

            return messages

        except Exception as e:
            messages.append(ValidationMessage(
                path=element.tag,
                message=f"Error validating attribute constraints: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="attribute_constraint_error"
            ))
            return messages
