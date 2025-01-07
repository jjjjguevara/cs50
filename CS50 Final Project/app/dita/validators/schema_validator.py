from typing import Dict, List, Optional, Any, Set
from pathlib import Path

# Core managers
from .validation_manager import ValidationManager
from ..events.event_manager import EventManager
from ..config.config_manager import ConfigManager

# Utils
from ..cache.cache import ContentCache
from ..utils.logger import DITALogger

# Types
from ..types import (
    ValidationResult,
    ValidationMessage,
    ValidationSeverity,
    SchemaVersion
)




class SchemaValidator:
    """Handles schema validation with DTD awareness."""

    def __init__(
        self,
        validation_manager: ValidationManager,
        event_manager: EventManager,
        config_manager: ConfigManager,
        cache: ContentCache,
        logger: Optional[DITALogger] = None
    ):
        """Initialize schema validator.

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
        self.cache = cache
        self.logger = logger or DITALogger(name=__name__)

        # Initialize schema registry
        self.schema_registry: Dict[str, Dict[str, Any]] = {}

        self._component_versions = {
            "elements": "1.0.0",
            "attributes": "1.0.0",
            "validation": "1.1.0",
            "specialization": "2.0.0",
            "inheritance": "2.0.0",
            "dtd": "2.0.0"
        }

    def validate_schema_structure(
        self,
        schema: Dict[str, Any],
        required_types: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate schema structure and completeness.

        Args:
            schema: Schema to validate
            required_types: Optional list of required type definitions

        Returns:
            ValidationResult indicating completeness status
        """
        try:
            messages = []

            # Check required sections
            required_sections = {
                "version": str,
                "types": dict,
                "properties": dict
            }

            for section, expected_type in required_sections.items():
                if section not in schema:
                    messages.append(ValidationMessage(
                        path=section,
                        message=f"Missing required section: {section}",
                        severity=ValidationSeverity.ERROR,
                        code="missing_section"
                    ))
                elif not isinstance(schema[section], expected_type):
                    messages.append(ValidationMessage(
                        path=section,
                        message=f"Invalid type for {section}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_section_type"
                    ))

            # Check required type definitions
            if required_types and "types" in schema:
                for type_name in required_types:
                    if type_name not in schema["types"]:
                        messages.append(ValidationMessage(
                            path=f"types.{type_name}",
                            message=f"Missing required type definition: {type_name}",
                            severity=ValidationSeverity.ERROR,
                            code="missing_type"
                        ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating schema structure: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Schema structure validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def validate_schema_composition(
        self,
        base_schema: Dict[str, Any],
        extension_schema: Dict[str, Any],
        strategy: str
    ) -> ValidationResult:
        """
        Validate schema composition.

        Args:
            base_schema: Base schema
            extension_schema: Extension schema
            strategy: Composition strategy

        Returns:
            ValidationResult indicating validation status
        """
        try:
            messages = []

            # Basic schema validation
            if not isinstance(base_schema, dict) or not isinstance(extension_schema, dict):
                messages.append(ValidationMessage(
                    path="",
                    message="Both schemas must be dictionaries",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_schema_type"
                ))
                return ValidationResult(is_valid=False, messages=messages)

            # Validate required fields preservation
            messages.extend(self._validate_required_fields_preservation(
                base_schema,
                extension_schema,
                strategy
            ))

            # Validate reference integrity
            messages.extend(self._validate_reference_integrity(
                base_schema,
                extension_schema
            ))

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

            # Validate schema structure
            structure_result = self.validate_schema_structure(schema)
            messages.extend(structure_result.messages)

            if structure_result.is_valid:
                # Validate DTD-specific aspects
                messages.extend(self._validate_dtd_elements(schema.get('elements', {})))
                messages.extend(self._validate_dtd_attributes(schema.get('attributes', {})))
                messages.extend(self._validate_dtd_specializations(schema.get('specializations', {})))

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
                    message=f"DTD schema validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="dtd_validation_error"
                )]
            )

    def _validate_required_fields_preservation(
        self,
        base_schema: Dict[str, Any],
        extension_schema: Dict[str, Any],
        strategy: str
    ) -> List[ValidationMessage]:
        """Validate preservation of required fields."""
        messages = []

        try:
            required_fields = self._get_required_fields(base_schema)

            # Validate based on strategy
            if strategy == "merge":
                # All required fields must be preserved
                for field in required_fields:
                    if not self._field_exists(extension_schema, field):
                        messages.append(ValidationMessage(
                            path=field,
                            message=f"Required field not preserved: {field}",
                            severity=ValidationSeverity.ERROR,
                            code="missing_required_field"
                        ))

            elif strategy == "selective":
                # Only selected fields must preserve required fields
                selected_fields = extension_schema.keys()
                for field in required_fields:
                    field_base = field.split('.')[0]
                    if field_base in selected_fields and not self._field_exists(extension_schema, field):
                        messages.append(ValidationMessage(
                            path=field,
                            message=f"Required field not preserved in selective merge: {field}",
                            severity=ValidationSeverity.ERROR,
                            code="missing_required_field"
                        ))

            return messages

        except Exception as e:
            self.logger.error(f"Error validating required fields: {str(e)}")
            return [ValidationMessage(
                path="",
                message=f"Required fields validation error: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="required_fields_error"
            )]

    def _validate_reference_integrity(
        self,
        base_schema: Dict[str, Any],
        extension_schema: Dict[str, Any]
    ) -> List[ValidationMessage]:
        """Validate schema reference integrity."""
        messages = []

        try:
            # Collect references
            base_refs = self._collect_references(base_schema)
            extension_refs = self._collect_references(extension_schema)

            # Validate extension references exist in base
            for ref in extension_refs:
                if ref not in base_refs:
                    messages.append(ValidationMessage(
                        path=ref,
                        message=f"Unresolved reference: {ref}",
                        severity=ValidationSeverity.ERROR,
                        code="unresolved_reference"
                    ))

            return messages

        except Exception as e:
            self.logger.error(f"Error validating reference integrity: {str(e)}")
            return [ValidationMessage(
                path="",
                message=f"Reference integrity validation error: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="reference_integrity_error"
            )]

    def _validate_dtd_elements(
        self,
        elements: Dict[str, Any]
    ) -> List[ValidationMessage]:
        """Validate DTD element definitions."""
        messages = []

        try:
            for element_name, element in elements.items():
                # Validate content model
                if content_model := element.get('content_model'):
                    if not self._is_valid_content_model(content_model):
                        messages.append(ValidationMessage(
                            path=f"elements.{element_name}.content_model",
                            message="Invalid content model definition",
                            severity=ValidationSeverity.ERROR,
                            code="invalid_content_model"
                        ))

                # Validate attributes
                if attributes := element.get('attributes'):
                    if not isinstance(attributes, dict):
                        messages.append(ValidationMessage(
                            path=f"elements.{element_name}.attributes",
                            message="Attributes must be a dictionary",
                            severity=ValidationSeverity.ERROR,
                            code="invalid_attributes"
                        ))

            return messages

        except Exception as e:
            self.logger.error(f"Error validating DTD elements: {str(e)}")
            return [ValidationMessage(
                path="elements",
                message=f"DTD elements validation error: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="dtd_elements_error"
            )]

    def _validate_dtd_attributes(
        self,
        attributes: Dict[str, Any]
    ) -> List[ValidationMessage]:
        """Validate DTD attribute definitions."""
        messages = []

        try:
            for attr_name, attr_def in attributes.items():
                # Validate attribute type
                if not self._is_valid_attribute_type(attr_def.get('type')):
                    messages.append(ValidationMessage(
                        path=f"attributes.{attr_name}",
                        message="Invalid attribute type",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_attribute_type"
                    ))

                # Validate enum values
                if attr_def.get('type') == 'enum':
                    if not attr_def.get('allowed_values'):
                        messages.append(ValidationMessage(
                            path=f"attributes.{attr_name}",
                            message="Missing allowed values for enum",
                            severity=ValidationSeverity.ERROR,
                            code="missing_enum_values"
                        ))

            return messages

        except Exception as e:
            self.logger.error(f"Error validating DTD attributes: {str(e)}")
            return [ValidationMessage(
                path="attributes",
                message=f"DTD attributes validation error: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="dtd_attributes_error"
            )]

    def _validate_dtd_specializations(
        self,
        specializations: Dict[str, Any]
    ) -> List[ValidationMessage]:
        """Validate DTD specialization definitions."""
        messages = []

        try:
            for spec_name, spec_info in specializations.items():
                # Validate base type
                if not spec_info.get('base_type'):
                    messages.append(ValidationMessage(
                        path=f"specializations.{spec_name}",
                        message="Missing base type",
                        severity=ValidationSeverity.ERROR,
                        code="missing_base_type"
                    ))

                # Validate inheritance path
                if not spec_info.get('inheritance_path'):
                    messages.append(ValidationMessage(
                        path=f"specializations.{spec_name}",
                        message="Missing inheritance path",
                        severity=ValidationSeverity.ERROR,
                        code="missing_inheritance_path"
                    ))

            return messages

        except Exception as e:
            self.logger.error(f"Error validating DTD specializations: {str(e)}")
            return [ValidationMessage(
                path="specializations",
                message=f"DTD specializations validation error: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="dtd_specializations_error"
            )]

    def _get_required_fields(self, schema: Dict[str, Any]) -> Set[str]:
        """Get all required fields from schema."""
        required = set()

        def collect_required(obj: Any, path: str = '') -> None:
            if isinstance(obj, dict):
                if obj.get('required', False):
                    required.add(path)
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    collect_required(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    collect_required(item, f"{path}[{i}]")

        collect_required(schema)
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

    def _is_valid_content_model(self, content_model: Dict[str, Any]) -> bool:
        """Validate content model structure."""
        required_fields = {'type', 'elements', 'ordering', 'occurrence'}
        return all(field in content_model for field in required_fields)

    def _is_valid_attribute_type(self, attr_type: str) -> bool:
        """Validate attribute type."""
        valid_types = {
            'CDATA', 'ID', 'IDREF', 'IDREFS', 'NMTOKEN', 'NMTOKENS',
            'ENTITY', 'ENTITIES', 'NOTATION', 'enum'
        }
        return attr_type in valid_types

    def validate_schema_version(
        self,
        schema: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate schema version information.

        Args:
            schema: Schema to validate

        Returns:
            ValidationResult: Validation status and messages
        """
        try:
            messages = []

            # Check version exists
            if "version" not in schema:
                return ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path="version",
                        message="Missing version information",
                        severity=ValidationSeverity.ERROR,
                        code="missing_version"
                    )]
                )

            # Parse version
            try:
                version = SchemaVersion.from_string(schema["version"])
            except ValueError as e:
                return ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path="version",
                        message=f"Invalid version format: {e}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_version"
                    )]
                )

            # Validate schema components match version
            required_components = {
                "1.0.0": ["elements", "attributes"],
                "1.1.0": ["elements", "attributes", "validation"],
                "2.0.0": ["elements", "attributes", "validation", "specialization"]
            }

            # Get minimum required version for schema components
            min_version = self._get_minimum_version_for_components(schema)
            if version < min_version:
                messages.append(ValidationMessage(
                    path="version",
                    message=f"Schema requires minimum version {min_version}",
                    severity=ValidationSeverity.ERROR,
                    code="version_too_low"
                ))

            # Check schema has required components for version
            if version_reqs := required_components.get(str(version)):
                for component in version_reqs:
                    if component not in schema:
                        messages.append(ValidationMessage(
                            path=component,
                            message=f"Missing required component for version {version}: {component}",
                            severity=ValidationSeverity.ERROR,
                            code="missing_component"
                        ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating schema version: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Version validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def _get_minimum_version_for_components(self, schema: Dict[str, Any]) -> SchemaVersion:
            """
            Determine minimum required version based on schema components.

            Args:
                schema: Schema to analyze

            Returns:
                SchemaVersion: Minimum required version
            """
            try:
                required_version = SchemaVersion(1, 0, 0)  # Base version

                # Check each component's minimum version requirement
                for component, min_version_str in self._component_versions.items():
                    if component in schema:
                        component_version = SchemaVersion.from_string(min_version_str)
                        if component_version > required_version:
                            required_version = component_version

                return required_version

            except Exception as e:
                self.logger.error(f"Error determining minimum version: {str(e)}")
                return SchemaVersion(1, 0, 0)  # Return base version on error

    def validate_specialization_structure(
        self,
        base_type: str,
        specialized_type: str
    ) -> ValidationResult:
        """
        Validate specialization structure and constraints.

        Args:
            base_type: Base type name
            specialized_type: Specialized type name

        Returns:
            ValidationResult: Validation status and messages
        """
        try:
            messages = []

            # Get schemas
            base_schema = self.schema_registry.get(base_type)
            specialized_schema = self.schema_registry.get(specialized_type)

            if not base_schema or not specialized_schema:
                return ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path=specialized_type,
                        message="Missing base or specialized schema",
                        severity=ValidationSeverity.ERROR,
                        code="missing_schema"
                    )]
                )

            # Validate inheritance declaration
            if "specialization" not in specialized_schema:
                messages.append(ValidationMessage(
                    path=specialized_type,
                    message="Missing specialization declaration",
                    severity=ValidationSeverity.ERROR,
                    code="missing_specialization"
                ))
            else:
                spec_info = specialized_schema["specialization"]
                # Check base type reference
                if not spec_info.get("base_type") == base_type:
                    messages.append(ValidationMessage(
                        path=f"{specialized_type}.specialization.base_type",
                        message=f"Invalid base type reference: {spec_info.get('base_type')}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_base_type"
                    ))

            # Validate content model structure
            if content_model_result := self.validate_content_model_structure(
                specialized_schema.get("content_model", {})
            ):
                messages.extend(content_model_result.messages)

            # Validate attribute structure
            if attribute_result := self.validate_attribute_structure(
                specialized_schema.get("attributes", {})
            ):
                messages.extend(attribute_result.messages)

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating specialization structure: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Specialization validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def validate_attribute_structure(
        self,
        attributes: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate attribute structure.

        Args:
            attributes: Attribute definitions to validate

        Returns:
            ValidationResult: Validation status and messages
        """
        try:
            messages = []

            for attr_name, attr_def in attributes.items():
                # Validate required fields
                required_fields = {"type", "required"}
                missing_fields = required_fields - set(attr_def.keys())
                if missing_fields:
                    messages.append(ValidationMessage(
                        path=f"attributes.{attr_name}",
                        message=f"Missing required fields: {missing_fields}",
                        severity=ValidationSeverity.ERROR,
                        code="missing_required_fields"
                    ))

                # Validate attribute type
                if attr_type := attr_def.get("type"):
                    valid_types = {
                        "string", "id", "reference", "references",
                        "token", "tokens", "entity", "entities",
                        "enum", "notation"
                    }
                    if attr_type not in valid_types:
                        messages.append(ValidationMessage(
                            path=f"attributes.{attr_name}.type",
                            message=f"Invalid attribute type: {attr_type}",
                            severity=ValidationSeverity.ERROR,
                            code="invalid_attribute_type"
                        ))

                    # Validate enum values if type is enum
                    if attr_type == "enum":
                        if not attr_def.get("allowed_values"):
                            messages.append(ValidationMessage(
                                path=f"attributes.{attr_name}",
                                message="Missing allowed values for enum type",
                                severity=ValidationSeverity.ERROR,
                                code="missing_enum_values"
                            ))
                        elif not isinstance(attr_def["allowed_values"], list):
                            messages.append(ValidationMessage(
                                path=f"attributes.{attr_name}.allowed_values",
                                message="Allowed values must be a list",
                                severity=ValidationSeverity.ERROR,
                                code="invalid_enum_values"
                            ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating attribute structure: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Attribute validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def validate_content_model_structure(
        self,
        content_model: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate content model structure.

        Args:
            content_model: Content model to validate

        Returns:
            ValidationResult: Validation status and messages
        """
        try:
            messages = []

            # Check required fields
            required_fields = {
                "type": str,
                "elements": list,
                "ordering": str,
                "occurrence": dict
            }

            for field, expected_type in required_fields.items():
                if field not in content_model:
                    messages.append(ValidationMessage(
                        path=f"content_model.{field}",
                        message=f"Missing required field: {field}",
                        severity=ValidationSeverity.ERROR,
                        code="missing_required_field"
                    ))
                elif not isinstance(content_model[field], expected_type):
                    messages.append(ValidationMessage(
                        path=f"content_model.{field}",
                        message=f"Invalid type for {field}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_field_type"
                    ))

            # Validate specific fields
            if content_model.get("type") not in {"empty", "text", "element", "complex"}:
                messages.append(ValidationMessage(
                    path="content_model.type",
                    message=f"Invalid content model type: {content_model.get('type')}",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_content_type"
                ))

            if content_model.get("ordering") not in {"sequence", "choice"}:
                messages.append(ValidationMessage(
                    path="content_model.ordering",
                    message=f"Invalid ordering type: {content_model.get('ordering')}",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_ordering"
                ))

            # Validate occurrence constraints
            if occurrence := content_model.get("occurrence"):
                if "min" not in occurrence or "max" not in occurrence:
                    messages.append(ValidationMessage(
                        path="content_model.occurrence",
                        message="Missing min/max occurrence constraints",
                        severity=ValidationSeverity.ERROR,
                        code="missing_occurrence"
                    ))
                else:
                    try:
                        min_val = int(occurrence["min"])
                        max_val = int(occurrence["max"])
                        if min_val < 0 or max_val < min_val:
                            messages.append(ValidationMessage(
                                path="content_model.occurrence",
                                message="Invalid occurrence constraints",
                                severity=ValidationSeverity.ERROR,
                                code="invalid_occurrence"
                            ))
                    except (ValueError, TypeError):
                        messages.append(ValidationMessage(
                            path="content_model.occurrence",
                            message="Occurrence values must be integers",
                            severity=ValidationSeverity.ERROR,
                            code="invalid_occurrence_type"
                        ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating content model structure: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Content model validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def validate_inheritance_structure(
        self,
        schema: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate inheritance structure within a schema.
        Focuses purely on structural validation, not compatibility.

        Args:
            schema: Schema to validate

        Returns:
            ValidationResult: Validation status and messages
        """
        try:
            messages = []

            if "inheritance" not in schema:
                return ValidationResult(is_valid=True, messages=[])

            inheritance_def = schema["inheritance"]

            # Validate inheritance declaration structure
            if not isinstance(inheritance_def, dict):
                return ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path="inheritance",
                        message="Invalid inheritance definition type",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_inheritance_type"
                    )]
                )

            # Validate each inheritance relationship
            for child_type, parent_info in inheritance_def.items():
                # Validate parent reference
                if not isinstance(parent_info, dict):
                    messages.append(ValidationMessage(
                        path=f"inheritance.{child_type}",
                        message="Invalid parent definition structure",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_parent_structure"
                    ))
                    continue

                # Check required parent fields
                required_fields = {"parent_type", "inheritance_type"}
                missing_fields = required_fields - set(parent_info.keys())
                if missing_fields:
                    messages.append(ValidationMessage(
                        path=f"inheritance.{child_type}",
                        message=f"Missing required inheritance fields: {missing_fields}",
                        severity=ValidationSeverity.ERROR,
                        code="missing_inheritance_fields"
                    ))

                # Validate inheritance type
                if inheritance_type := parent_info.get("inheritance_type"):
                    valid_types = {"extension", "restriction", "specialization"}
                    if inheritance_type not in valid_types:
                        messages.append(ValidationMessage(
                            path=f"inheritance.{child_type}.inheritance_type",
                            message=f"Invalid inheritance type: {inheritance_type}",
                            severity=ValidationSeverity.ERROR,
                            code="invalid_inheritance_type"
                        ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating inheritance structure: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Inheritance validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def validate_inherited_attributes(
        self,
        schema: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate attribute declarations in inheritance context.
        Focuses on structural validation of inherited attributes.

        Args:
            schema: Schema to validate

        Returns:
            ValidationResult: Validation status and messages
        """
        try:
            messages = []

            if "attributes" not in schema:
                return ValidationResult(is_valid=True, messages=[])

            # Validate inherited attribute declarations
            for attr_name, attr_def in schema["attributes"].items():
                if "inherited" in attr_def:
                    # Validate inheritance declaration structure
                    if not isinstance(attr_def["inherited"], dict):
                        messages.append(ValidationMessage(
                            path=f"attributes.{attr_name}.inherited",
                            message="Invalid inheritance declaration",
                            severity=ValidationSeverity.ERROR,
                            code="invalid_inheritance_declaration"
                        ))
                        continue

                    # Validate required inheritance fields
                    required_fields = {"from_type", "override_type"}
                    missing_fields = required_fields - set(attr_def["inherited"].keys())
                    if missing_fields:
                        messages.append(ValidationMessage(
                            path=f"attributes.{attr_name}.inherited",
                            message=f"Missing required inheritance fields: {missing_fields}",
                            severity=ValidationSeverity.ERROR,
                            code="missing_inheritance_fields"
                        ))

                    # Validate override type if present
                    if override_type := attr_def["inherited"].get("override_type"):
                        valid_overrides = {"merge", "replace", "extend"}
                        if override_type not in valid_overrides:
                            messages.append(ValidationMessage(
                                path=f"attributes.{attr_name}.inherited.override_type",
                                message=f"Invalid override type: {override_type}",
                                severity=ValidationSeverity.ERROR,
                                code="invalid_override_type"
                            ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating inherited attributes: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Inherited attributes validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def validate_specialization_constraints(
        self,
        constraints: Dict[str, Any],
        base_schema: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate specialization constraints against schema rules.

        Args:
            constraints: Specialization constraints to validate
            base_schema: Base schema to validate against

        Returns:
            ValidationResult: Validation status and messages
        """
        try:
            messages = []

            # Validate constraint structure
            if not isinstance(constraints, dict):
                return ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path="constraints",
                        message="Invalid constraints structure",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_constraints"
                    )]
                )

            # Validate each constraint type
            for constraint_type, constraint_value in constraints.items():
                if constraint_type == "allowed_elements":
                    if not isinstance(constraint_value, list):
                        messages.append(ValidationMessage(
                            path=f"constraints.{constraint_type}",
                            message="Allowed elements must be a list",
                            severity=ValidationSeverity.ERROR,
                            code="invalid_allowed_elements"
                        ))
                    else:
                        # Validate each element exists in base schema
                        base_elements = set(base_schema.get("elements", []))
                        for element in constraint_value:
                            if element not in base_elements:
                                messages.append(ValidationMessage(
                                    path=f"constraints.{constraint_type}",
                                    message=f"Unknown element in constraints: {element}",
                                    severity=ValidationSeverity.ERROR,
                                    code="unknown_element"
                                ))

                elif constraint_type == "required_attributes":
                    if not isinstance(constraint_value, list):
                        messages.append(ValidationMessage(
                            path=f"constraints.{constraint_type}",
                            message="Required attributes must be a list",
                            severity=ValidationSeverity.ERROR,
                            code="invalid_required_attributes"
                        ))
                    else:
                        # Validate each attribute exists in base schema
                        base_attrs = base_schema.get("attributes", {})
                        for attr in constraint_value:
                            if attr not in base_attrs:
                                messages.append(ValidationMessage(
                                    path=f"constraints.{constraint_type}",
                                    message=f"Unknown attribute in constraints: {attr}",
                                    severity=ValidationSeverity.ERROR,
                                    code="unknown_attribute"
                                ))

                elif constraint_type == "content_model":
                    if not isinstance(constraint_value, dict):
                        messages.append(ValidationMessage(
                            path=f"constraints.{constraint_type}",
                            message="Content model constraints must be an object",
                            severity=ValidationSeverity.ERROR,
                            code="invalid_content_model_constraints"
                        ))
                    else:
                        # Validate content model constraints
                        content_result = self.validate_content_model_structure(constraint_value)
                        messages.extend(content_result.messages)

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
                    path="",
                    message=f"Constraint validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )
