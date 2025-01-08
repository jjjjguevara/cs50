from typing import Dict, List, Optional, Any, Set, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import sys

# Validation and models
from ..models.types import (
    ValidationResult,
    ValidationMessage,
    ValidationSeverity,
    ElementType,
    ProcessingPhase,
    ProcessingContext,
)

# Core managers
from ..validation_manager import ValidationManager
from ..event_manager import EventManager
from ..config.config_manager import ConfigManager
from ..schema.schema_manager import SchemaManager, CompositionStrategy, SchemaVersion

# DTD specific
from ..dtd.dtd_models import (
    DTDElement,
    DTDAttribute,
    DTDParsingResult,
    SpecializationInfo
)

# Utils
from ..utils.cache import ContentCache, CacheEntryType
from ..utils.logger import DITALogger
from .schema_validator import SchemaValidator


class SchemaCompatibilityChecker:
    """Handles schema compatibility checking with DTD awareness."""

    def __init__(
        self,
        validation_manager: ValidationManager,
        event_manager: EventManager,
        config_manager: ConfigManager,
        schema_validator: SchemaValidator,
        cache: ContentCache,
        logger: Optional[DITALogger] = None
    ):
        """Initialize compatibility checker.

        Args:
            validation_manager: System validation manager
            event_manager: System event manager
            config_manager: System configuration manager
            cache: Cache system
            logger: Optional logger
        """
        self.validation_manager = validation_manager
        self.event_manager = event_manager
        self.config_manager = config_manager
        self.schema_validator = schema_validator
        self.cache = cache
        self.logger = logger or DITALogger(name=__name__)

        # Initialize schema registry
        self.schema_registry: Dict[str, Dict[str, Any]] = {}

    def check_compatibility(
        self,
        base_schema: Dict[str, Any],
        extension_schema: Dict[str, Any],
        strategy: CompositionStrategy
    ) -> ValidationResult:
        """
        Check overall schema compatibility.

        Args:
            base_schema: Base schema
            extension_schema: Extension schema to compare
            strategy: Composition strategy

        Returns:
            ValidationResult: Compatibility status and issues
        """
        try:
            messages = []

            # Check if either schema is DTD-derived
            base_is_dtd = 'source_dtd' in base_schema
            ext_is_dtd = 'source_dtd' in extension_schema

            # Use appropriate validation based on schema types
            if base_is_dtd or ext_is_dtd:
                messages.extend(self._check_dtd_compatibility(
                    base_schema, extension_schema, strategy
                ))
            else:
                messages.extend(self._check_schema_compatibility(
                    base_schema, extension_schema, strategy
                ))

            # Cache compatibility result
            cache_key = f"compatibility_{hash(str(base_schema))}_{hash(str(extension_schema))}"
            result = ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

            self.cache.set(
                key=cache_key,
                data=result,
                entry_type=CacheEntryType.VALIDATION,
                element_type=ElementType.UNKNOWN,
                phase=ProcessingPhase.VALIDATION
            )

            return result

        except Exception as e:
            self.logger.error(f"Error checking schema compatibility: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Compatibility check error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="compatibility_error"
                )]
            )

    def _check_dtd_compatibility(
        self,
        base_schema: Dict[str, Any],
        extension_schema: Dict[str, Any],
        strategy: CompositionStrategy
    ) -> List[ValidationMessage]:
        """Check DTD schema compatibility."""
        messages = []

        # Version compatibility
        if not self._are_dtd_versions_compatible(
            base_schema.get('metadata', {}).get('dtd_version'),
            extension_schema.get('metadata', {}).get('dtd_version')
        ):
            messages.append(ValidationMessage(
                path="metadata.dtd_version",
                message="Incompatible DTD versions",
                severity=ValidationSeverity.ERROR,
                code="incompatible_dtd_version"
            ))

        # Domain compatibility
        if not self._are_element_domains_compatible(
            base_schema.get('elements', {}),
            extension_schema.get('elements', {})
        ):
            messages.append(ValidationMessage(
                path="elements",
                message="Incompatible element domains",
                severity=ValidationSeverity.ERROR,
                code="incompatible_domains"
            ))

        # Check specialization hierarchies
        if not self._are_specialization_hierarchies_compatible(
            base_schema.get('specializations', {}),
            extension_schema.get('specializations', {})
        ):
            messages.append(ValidationMessage(
                path="specializations",
                message="Incompatible specialization hierarchies",
                severity=ValidationSeverity.ERROR,
                code="incompatible_specializations"
            ))

        return messages

    def _check_schema_compatibility(
        self,
        base_schema: Dict[str, Any],
        extension_schema: Dict[str, Any],
        strategy: CompositionStrategy
    ) -> List[ValidationMessage]:
        """Check regular schema compatibility."""
        messages = []

        # Type compatibility
        messages.extend(self._check_type_compatibility(
            base_schema, extension_schema
        ))

        # Structural compatibility
        messages.extend(self._check_structural_compatibility(
            base_schema, extension_schema, strategy
        ))

        # Reference compatibility
        messages.extend(self._check_reference_compatibility(
            base_schema, extension_schema
        ))

        return messages

    def _are_dtd_versions_compatible(
        self,
        base_version: Optional[str],
        ext_version: Optional[str]
    ) -> bool:
        """Check if DTD versions are compatible."""
        if not base_version or not ext_version:
            return True  # Consider compatible if versions not specified

        try:
            base_parts = [int(p) for p in base_version.split('.')]
            ext_parts = [int(p) for p in ext_version.split('.')]

            # Major version must match
            return base_parts[0] == ext_parts[0]
        except (ValueError, IndexError):
            return False

    def _are_element_domains_compatible(
        self,
        base_elements: Dict[str, Any],
        ext_elements: Dict[str, Any]
    ) -> bool:
        """Check if element domains are compatible."""
        try:
            # Get domain prefixes
            base_domains = self._extract_domain_prefixes(base_elements)
            ext_domains = self._extract_domain_prefixes(ext_elements)

            # All extension domains must be compatible with base domains
            return all(
                self._is_domain_compatible(ext_domain, base_domains)
                for ext_domain in ext_domains
            )
        except Exception as e:
            self.logger.error(f"Error checking domain compatibility: {str(e)}")
            return False

    def _extract_domain_prefixes(
        self,
        elements: Dict[str, Any]
    ) -> Set[str]:
        """Extract domain prefixes from element names."""
        domains = set()
        for element_name in elements:
            if '+' in element_name:
                prefix = element_name.split('+')[0]
                domains.add(prefix)
        return domains

    def _is_domain_compatible(
        self,
        domain: str,
        base_domains: Set[str]
    ) -> bool:
        """Check if a domain is compatible with base domains."""
        # Domain must either exist in base or be a valid extension
        return domain in base_domains or any(
            domain.startswith(base) for base in base_domains
        )

    def _are_specialization_hierarchies_compatible(
        self,
        base_specs: Dict[str, Any],
        ext_specs: Dict[str, Any]
    ) -> bool:
        """Check if specialization hierarchies are compatible."""
        try:
            for spec_name, ext_spec in ext_specs.items():
                if spec_name in base_specs:
                    base_spec = base_specs[spec_name]

                    # Check inheritance paths
                    if not self._are_inheritance_paths_compatible(
                        base_spec.get('inheritance_path', []),
                        ext_spec.get('inheritance_path', [])
                    ):
                        return False

            return True
        except Exception as e:
            self.logger.error(f"Error checking specialization compatibility: {str(e)}")
            return False

    def _are_inheritance_paths_compatible(
        self,
        base_path: List[str],
        ext_path: List[str]
    ) -> bool:
        """Check if inheritance paths are compatible."""
        # Extension path must start with base path
        return len(ext_path) >= len(base_path) and ext_path[:len(base_path)] == base_path

    def _check_type_compatibility(
        self,
        base_schema: Dict[str, Any],
        extension_schema: Dict[str, Any]
    ) -> List[ValidationMessage]:
        """Check type compatibility between schemas."""
        messages = []

        def check_types(
            base_value: Any,
            ext_value: Any,
            path: str
        ) -> None:
            if type(base_value) != type(ext_value):
                messages.append(ValidationMessage(
                    path=path,
                    message=f"Type mismatch: {type(base_value)} vs {type(ext_value)}",
                    severity=ValidationSeverity.ERROR,
                    code="type_mismatch"
                ))

        self._traverse_schemas(base_schema, extension_schema, check_types)
        return messages

    def _check_structural_compatibility(
        self,
        base_schema: Dict[str, Any],
        extension_schema: Dict[str, Any],
        strategy: CompositionStrategy
    ) -> List[ValidationMessage]:
        """Check structural compatibility between schemas."""
        messages = []

        # Check required fields based on strategy
        if strategy == CompositionStrategy.MERGE:
            base_required = self._get_required_fields(base_schema)
            for field in base_required:
                if not self._field_exists(extension_schema, field):
                    messages.append(ValidationMessage(
                        path=field,
                        message=f"Required field missing in extension: {field}",
                        severity=ValidationSeverity.ERROR,
                        code="missing_required_field"
                    ))

        return messages

    def _check_reference_compatibility(
        self,
        base_schema: Dict[str, Any],
        extension_schema: Dict[str, Any]
    ) -> List[ValidationMessage]:
        """Check reference compatibility between schemas."""
        messages = []

        # Collect all references
        base_refs = self._collect_references(base_schema)
        ext_refs = self._collect_references(extension_schema)

        # Check that all extension references exist in base
        for ref in ext_refs:
            if ref not in base_refs:
                messages.append(ValidationMessage(
                    path=f"reference.{ref}",
                    message=f"Reference not found in base schema: {ref}",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_reference"
                ))

        return messages

    def _traverse_schemas(
        self,
        base: Dict[str, Any],
        extension: Dict[str, Any],
        callback: Callable[[Any, Any, str], None]
    ) -> None:
        """Traverse schemas and apply callback to matching fields."""
        def traverse(base_value: Any, ext_value: Any, path: str = "") -> None:
            callback(base_value, ext_value, path)

            if isinstance(base_value, dict) and isinstance(ext_value, dict):
                for key in set(base_value.keys()) & set(ext_value.keys()):
                    new_path = f"{path}.{key}" if path else key
                    traverse(base_value[key], ext_value[key], new_path)
            elif isinstance(base_value, list) and isinstance(ext_value, list):
                for i, (base_item, ext_item) in enumerate(
                    zip(base_value, ext_value)
                ):
                    traverse(base_item, ext_item, f"{path}[{i}]")

        traverse(base, extension)

    def _get_required_fields(self, schema: Dict[str, Any]) -> Set[str]:
        """Get all required fields from schema."""
        required = set()

        def collect(obj: Any, path: str = "") -> None:
            if isinstance(obj, dict):
                if obj.get('required', False):
                    required.add(path)
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    collect(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    collect(item, f"{path}[{i}]")

        collect(schema)
        return required

    def _field_exists(self, schema: Dict[str, Any], path: str) -> bool:
        """Check if field exists in schema."""
        try:
            parts = path.split('.')
            current = schema
            for part in parts:
                if part.endswith(']'):
                    array_part = part.split('[')
                    current = current[array_part[0]]
                    index = int(array_part[1][:-1])
                    current = current[index]
                else:
                    current = current[part]
            return True
        except (KeyError, IndexError, TypeError):
            return False

    def _collect_references(self, schema: Dict[str, Any]) -> Set[str]:
        """Collect all references in schema."""
        refs = set()

        def traverse(value: Any) -> None:
            if isinstance(value, dict):
                if "$ref" in value:
                    refs.add(value["$ref"])
                for v in value.values():
                    traverse(v)
            elif isinstance(value, list):
                for item in value:
                    traverse(item)

        traverse(schema)
        return refs

    def check_version_compatibility(
            self,
            version1: str,
            version2: str
        ) -> bool:
            """
            Check if two schema versions are compatible.

            Args:
                version1: First version string
                version2: Second version string

            Returns:
                bool: True if versions are compatible
            """
            try:
                v1 = SchemaVersion.from_string(version1)
                v2 = SchemaVersion.from_string(version2)

                # Major version must match for compatibility
                if v1.major != v2.major:
                    return False

                # If minor versions differ, older version must be first
                if v1.minor > v2.minor:
                    return False

                return True

            except Exception as e:
                self.logger.error(f"Error checking version compatibility: {str(e)}")
                return False

    def check_specialization_compatibility(
        self,
        base_type: str,
        specialized_type: str
    ) -> ValidationResult:
        """
        Check if specialization is compatible with base type.
        Now uses SchemaValidator for constraint validation.

        Args:
            base_type: Base type name
            specialized_type: Specialized type name

        Returns:
            ValidationResult: Compatibility status
        """
        try:
            messages = []

            # Get schemas
            base_schema = self.schema_registry.get(base_type)
            spec_schema = self.schema_registry.get(specialized_type)

            if not base_schema or not spec_schema:
                return ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path="specialization",
                        message="Missing base or specialized schema",
                        severity=ValidationSeverity.ERROR,
                        code="missing_schema"
                    )]
                )

            # Validate specialization constraints using SchemaValidator
            if constraints := spec_schema.get("specialization", {}).get("constraints"):
                constraint_result = self.schema_validator.validate_specialization_constraints(
                    constraints,
                    base_schema
                )
                if not constraint_result.is_valid:
                    return constraint_result

            # Check content model compatibility
            content_result = self._check_content_model_inheritance_compatibility(
                base_schema.get("content_model", {}),
                spec_schema.get("content_model", {}),
                spec_schema.get("specialization", {})
            )
            messages.extend(content_result.messages)

            # Check attribute compatibility
            attr_result = self._check_attribute_inheritance_compatibility(
                base_schema.get("attributes", {}),
                spec_schema.get("attributes", {}),
                spec_schema.get("specialization", {})
            )
            messages.extend(attr_result.messages)

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error checking specialization compatibility: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Specialization compatibility error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="compatibility_error"
                )]
            )

    def check_attribute_compatibility(
        self,
        base_attrs: Dict[str, Any],
        ext_attrs: Dict[str, Any]
    ) -> ValidationResult:
        """
        Check attribute compatibility between schemas.

        Args:
            base_attrs: Base attributes
            ext_attrs: Extension attributes

        Returns:
            ValidationResult: Compatibility status
        """
        try:
            messages = []

            # Check required attributes preservation
            for attr_name, base_attr in base_attrs.items():
                if base_attr.get("required", False):
                    if attr_name not in ext_attrs:
                        messages.append(ValidationMessage(
                            path=f"attributes.{attr_name}",
                            message="Missing required base attribute",
                            severity=ValidationSeverity.ERROR,
                            code="missing_required_attribute"
                        ))
                    else:
                        # Check type compatibility
                        ext_attr = ext_attrs[attr_name]
                        if not self._are_types_compatible(
                            base_attr.get("type"),
                            ext_attr.get("type")
                        ):
                            messages.append(ValidationMessage(
                                path=f"attributes.{attr_name}",
                                message="Incompatible attribute types",
                                severity=ValidationSeverity.ERROR,
                                code="incompatible_types"
                            ))

                        # Check enum values compatibility
                        if base_attr.get("type") == "enum":
                            base_values = set(base_attr.get("allowed_values", []))
                            ext_values = set(ext_attr.get("allowed_values", []))
                            if not base_values.issubset(ext_values):
                                messages.append(ValidationMessage(
                                    path=f"attributes.{attr_name}",
                                    message="Incompatible enum values",
                                    severity=ValidationSeverity.ERROR,
                                    code="incompatible_enum_values"
                                ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error checking attribute compatibility: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Attribute compatibility error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="compatibility_error"
                )]
            )

    def check_content_model_compatibility(
        self,
        base_model: Dict[str, Any],
        ext_model: Dict[str, Any]
    ) -> ValidationResult:
        """
        Check content model compatibility between schemas.

        Args:
            base_model: Base content model
            ext_model: Extension content model

        Returns:
            ValidationResult: Compatibility status
        """
        try:
            messages = []

            # Check content model type compatibility
            if base_model.get("type") != ext_model.get("type"):
                messages.append(ValidationMessage(
                    path="content_model.type",
                    message="Incompatible content model types",
                    severity=ValidationSeverity.ERROR,
                    code="incompatible_model_type"
                ))

            # Check element compatibility
            base_elements = set(base_model.get("elements", []))
            ext_elements = set(ext_model.get("elements", []))

            # Extension must include all base elements
            missing_elements = base_elements - ext_elements
            if missing_elements:
                messages.append(ValidationMessage(
                    path="content_model.elements",
                    message=f"Missing required elements: {missing_elements}",
                    severity=ValidationSeverity.ERROR,
                    code="missing_elements"
                ))

            # Check ordering compatibility
            if base_model.get("ordering") == "sequence" and ext_model.get("ordering") != "sequence":
                messages.append(ValidationMessage(
                    path="content_model.ordering",
                    message="Cannot relax sequence ordering",
                    severity=ValidationSeverity.ERROR,
                    code="incompatible_ordering"
                ))

            # Check occurrence compatibility
            base_occur = base_model.get("occurrence", {})
            ext_occur = ext_model.get("occurrence", {})

            if not self._are_occurrences_compatible(base_occur, ext_occur):
                messages.append(ValidationMessage(
                    path="content_model.occurrence",
                    message="Incompatible occurrence constraints",
                    severity=ValidationSeverity.ERROR,
                    code="incompatible_occurrence"
                ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error checking content model compatibility: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Content model compatibility error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="compatibility_error"
                )]
            )

    def _are_types_compatible(self, base_type: str, ext_type: str) -> bool:
        """Check if attribute types are compatible."""
        if base_type == ext_type:
            return True

        # Define type compatibility rules
        compatible_types = {
            "string": {"token"},
            "tokens": {"string"},
            "id": {"string"},
            "reference": {"string"}
        }

        return ext_type in compatible_types.get(base_type, set())

    def _are_occurrences_compatible(
        self,
        base_occur: Dict[str, int],
        ext_occur: Dict[str, int]
    ) -> bool:
        """Check if occurrence constraints are compatible."""
        try:
            # Extension cannot be more permissive than base
            if ext_occur.get("min", 0) < base_occur.get("min", 0):
                return False
            if ext_occur.get("max", sys.maxsize) > base_occur.get("max", sys.maxsize):
                return False
            return True
        except Exception:
            return False


    def check_inheritance_compatibility(
        self,
        parent_schema: Dict[str, Any],
        child_schema: Dict[str, Any]
    ) -> ValidationResult:
        """
        Check inheritance compatibility between parent and child schemas.
        Focuses on semantic compatibility of inheritance relationships.

        Args:
            parent_schema: Parent schema
            child_schema: Child schema

        Returns:
            ValidationResult: Compatibility status and messages
        """
        try:
            messages = []

            # Get inheritance definition
            inheritance = child_schema.get("inheritance", {})

            # Check each inherited aspect
            for aspect, inheritance_info in inheritance.items():
                if aspect == "attributes":
                    # Check attribute inheritance compatibility
                    attr_result = self._check_attribute_inheritance_compatibility(
                        parent_schema.get("attributes", {}),
                        child_schema.get("attributes", {}),
                        inheritance_info
                    )
                    messages.extend(attr_result.messages)

                elif aspect == "content_model":
                    # Check content model inheritance compatibility
                    model_result = self._check_content_model_inheritance_compatibility(
                        parent_schema.get("content_model", {}),
                        child_schema.get("content_model", {}),
                        inheritance_info
                    )
                    messages.extend(model_result.messages)

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error checking inheritance compatibility: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Inheritance compatibility error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="compatibility_error"
                )]
            )

    def _check_attribute_inheritance_compatibility(
        self,
        parent_attrs: Dict[str, Any],
        child_attrs: Dict[str, Any],
        inheritance_info: Dict[str, Any]
    ) -> ValidationResult:
        """Check attribute inheritance compatibility."""
        messages = []

        try:
            # Check each inherited attribute
            for attr_name, child_attr in child_attrs.items():
                if "inherited" in child_attr:
                    parent_attr = parent_attrs.get(attr_name)
                    if not parent_attr:
                        messages.append(ValidationMessage(
                            path=f"attributes.{attr_name}",
                            message="Inherited attribute not found in parent",
                            severity=ValidationSeverity.ERROR,
                            code="missing_parent_attribute"
                        ))
                        continue

                    # Check override compatibility
                    override_type = child_attr["inherited"].get("override_type")
                    if override_type == "merge":
                        if not self._can_merge_attributes(parent_attr, child_attr):
                            messages.append(ValidationMessage(
                                path=f"attributes.{attr_name}",
                                message="Incompatible attribute merge",
                                severity=ValidationSeverity.ERROR,
                                code="incompatible_merge"
                            ))
                    elif override_type == "extend":
                        if not self._can_extend_attribute(parent_attr, child_attr):
                            messages.append(ValidationMessage(
                                path=f"attributes.{attr_name}",
                                message="Incompatible attribute extension",
                                severity=ValidationSeverity.ERROR,
                                code="incompatible_extension"
                            ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error checking attribute inheritance compatibility: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Attribute inheritance compatibility error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="compatibility_error"
                )]
            )

    def _check_content_model_inheritance_compatibility(
        self,
        parent_model: Dict[str, Any],
        child_model: Dict[str, Any],
        inheritance_info: Dict[str, Any]
    ) -> ValidationResult:
        """
        Check content model inheritance compatibility.

        Args:
            parent_model: Parent content model
            child_model: Child content model
            inheritance_info: Inheritance information

        Returns:
            ValidationResult: Compatibility status and messages
        """
        try:
            messages = []

            # Check inheritance type
            inheritance_type = inheritance_info.get("inheritance_type")
            if inheritance_type not in {"extension", "restriction", "specialization"}:
                return ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path="content_model.inheritance_type",
                        message=f"Invalid inheritance type: {inheritance_type}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_inheritance_type"
                    )]
                )

            # Check type compatibility
            if parent_model.get("type") != child_model.get("type"):
                messages.append(ValidationMessage(
                    path="content_model.type",
                    message="Content model type mismatch",
                    severity=ValidationSeverity.ERROR,
                    code="type_mismatch"
                ))

            # Check based on inheritance type
            if inheritance_type == "extension":
                # Child must include all parent elements
                parent_elements = set(parent_model.get("elements", []))
                child_elements = set(child_model.get("elements", []))
                if not parent_elements.issubset(child_elements):
                    messages.append(ValidationMessage(
                        path="content_model.elements",
                        message="Extension must include all parent elements",
                        severity=ValidationSeverity.ERROR,
                        code="missing_parent_elements"
                    ))

            elif inheritance_type == "restriction":
                # Child elements must be subset of parent elements
                parent_elements = set(parent_model.get("elements", []))
                child_elements = set(child_model.get("elements", []))
                if not child_elements.issubset(parent_elements):
                    messages.append(ValidationMessage(
                        path="content_model.elements",
                        message="Restriction must use subset of parent elements",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_restriction_elements"
                    ))

                # Check occurrence constraints
                parent_occur = parent_model.get("occurrence", {})
                child_occur = child_model.get("occurrence", {})
                if not self._are_restriction_occurrences_compatible(parent_occur, child_occur):
                    messages.append(ValidationMessage(
                        path="content_model.occurrence",
                        message="Invalid occurrence restriction",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_occurrence_restriction"
                    ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error checking content model inheritance compatibility: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Content model inheritance compatibility error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="compatibility_error"
                )]
            )

    def _are_restriction_occurrences_compatible(
        self,
        parent_occur: Dict[str, int],
        child_occur: Dict[str, int]
    ) -> bool:
        """Check if occurrence restrictions are compatible."""
        try:
            parent_min = parent_occur.get("min", 0)
            parent_max = parent_occur.get("max", sys.maxsize)
            child_min = child_occur.get("min", 0)
            child_max = child_occur.get("max", sys.maxsize)

            # Child range must be within parent range
            return (
                child_min >= parent_min and
                child_max <= parent_max
            )
        except Exception:
            return False

    def _can_merge_attributes(
        self,
        parent_attr: Dict[str, Any],
        child_attr: Dict[str, Any]
    ) -> bool:
        """Check if attributes can be merged."""
        # Base type must match
        if parent_attr.get("type") != child_attr.get("type"):
            return False

        # Child can't relax required constraint
        if parent_attr.get("required") and not child_attr.get("required"):
            return False

        return True

    def _can_extend_attribute(
        self,
        parent_attr: Dict[str, Any],
        child_attr: Dict[str, Any]
    ) -> bool:
        """Check if attribute can be extended."""
        # Only certain types can be extended
        extensible_types = {"string", "token", "enum"}
        if parent_attr.get("type") not in extensible_types:
            return False

        # For enums, child values must include parent values
        if parent_attr.get("type") == "enum":
            parent_values = set(parent_attr.get("allowed_values", []))
            child_values = set(child_attr.get("allowed_values", []))
            if not parent_values.issubset(child_values):
                return False

        return True
