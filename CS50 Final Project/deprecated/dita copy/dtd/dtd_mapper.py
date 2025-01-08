"""DTD to Schema mapping and conversion."""
from typing import Dict, List, Optional, Any, Set, NamedTuple, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import sys
from app.dita.utils.logger import DITALogger
if TYPE_CHECKING:
    from .dtd_models import (
        OccurrenceConstraint,
        ContentModel,
        DTDAttribute,
        DTDElement,
        SpecializationInfo,
        ValidationMessage,
        ValidationSeverity
    )
from .dtd_models import (
    OccurrenceConstraint,
    ContentModel,
    DTDAttribute,
    DTDElement,
    DTDEntity,
    DTDParsingResult,
    SpecializationInfo,
    AttributeType,
    AttributeDefault,
    ContentModelParticle
)


class DTDSchemaMapper:
    """Maps DTD definitions to internal schema format."""

    def __init__(self, logger: Optional[DITALogger] = None):
        """Initialize mapper with logger."""
        self.logger = logger or DITALogger(name=__name__)
        self.element_cache: Dict[str, DTDElement] = {}
        # Changed from inheritance_map to _inheritance_map since it's internal
        self._inheritance_map: Dict[str, List[Tuple[str, str]]] = {}
        self._processed_files: Set[str] = set()
        self._specializations: Dict[str, SpecializationInfo] = {}


    def _parse_element_defs(self, content: str) -> Dict[str, DTDElement]:
        """Parse element definitions from DTD content."""
        elements = {}

        # Match element definitions
        pattern = r'<!ELEMENT\s+(\w+)\s+([^>]+)>'
        for match in re.finditer(pattern, content):
            name = match.group(1)
            content_model_str = match.group(2).strip()

            # Create content model
            if content_model_str == 'EMPTY':
                content_model = ContentModel.empty()
            elif content_model_str == '(#PCDATA)':
                content_model = ContentModel.text()
            else:
                content_model = self._parse_content_model(content_model_str)

            elements[name] = DTDElement(
                name=name,
                content_model=content_model,
                attributes={},
                base_type=None,
                is_abstract=False
            )

        return elements

    def get_specialization_info(
            self,
            element_tag: str
        ) -> Optional[SpecializationInfo]:
            """
            Get specialization information for an element.

            Args:
                element_tag: Element tag name to look up

            Returns:
                Optional[SpecializationInfo]: Specialization info if element is specialized
            """
            try:
                return self._specializations.get(element_tag)
            except Exception as e:
                self.logger.error(f"Error getting specialization info: {str(e)}")
                return None

    def register_specialization(
            self,
            base_type: str,
            specialized_type: str,
            attributes: Dict[str, DTDAttribute],
            constraints: Optional[Dict[str, Any]] = None,
            metadata: Optional[Dict[str, Any]] = None
        ) -> None:
            """
            Register a DTD specialization.

            Args:
                base_type: Base DTD type
                specialized_type: Specialized type name
                attributes: Specialized attributes
                constraints: Optional constraints
                metadata: Optional metadata
            """
            try:
                # Build inheritance path
                inheritance_path = [base_type, specialized_type]

                # Create specialization info
                spec_info = SpecializationInfo(
                    base_type=base_type,
                    specialized_type=specialized_type,
                    inheritance_path=inheritance_path,
                    attributes=attributes,
                    constraints=constraints or {},
                    metadata=metadata or {}
                )

                # Store specialization info
                self._specializations[specialized_type] = spec_info

                # Update inheritance map
                if specialized_type not in self._inheritance_map:
                    self._inheritance_map[specialized_type] = []
                self._inheritance_map[specialized_type].append((specialized_type, base_type))

                self.logger.debug(
                    f"Registered specialization {specialized_type} "
                    f"(base: {base_type})"
                )

            except Exception as e:
                self.logger.error(
                    f"Error registering specialization {specialized_type}: {str(e)}"
                )
                raise


    def convert_dtd_to_schema(
            self,
            dtd_path: Path
        ) -> Dict[str, Any]:
            """
            Convert DTD to schema format.

            Args:
                dtd_path: Path to DTD file

            Returns:
                Dict[str, Any]: Converted schema
            """
            try:
                # Parse DTD
                parsed_result = self.parse_dtd(dtd_path)

                # Generate schema components
                elements_schema = self.map_elements_to_schema(parsed_result.elements)
                attributes_schema = self.map_attributes_to_schema(parsed_result.entities)

                # Generate validation patterns
                validation_patterns = self.generate_validation_patterns(parsed_result)

                # Build final schema
                schema = {
                    "name": dtd_path.stem,
                    "source_dtd": str(dtd_path),
                    "elements": elements_schema,
                    "attributes": attributes_schema,
                    "validation_patterns": validation_patterns,
                    "inheritance": self._build_inheritance_map(parsed_result.elements),
                    "specializations": {
                        name: self.map_specialization_to_schema(spec)
                        for name, spec in parsed_result.specializations.items()
                    }
                }

                return schema

            except Exception as e:
                self.logger.error(f"Error converting DTD to schema: {str(e)}")
                raise

    def map_elements_to_schema(
            self,
            elements: Dict[str, DTDElement]
        ) -> Dict[str, Any]:
            """
            Map DTD elements to schema format.

            Args:
                elements: Dictionary of DTD elements

            Returns:
                Dict[str, Any]: Schema element definitions
            """
            try:
                schema_elements = {}

                for name, element in elements.items():
                    schema_elements[name] = {
                        "content_model": self.map_content_model_to_schema(
                            element.content_model
                        ),
                        "attributes": {
                            attr_name: self._build_attribute_schema(attr)
                            for attr_name, attr in element.attributes.items()
                        },
                        "is_abstract": element.is_abstract,
                        "base_type": element.base_type,
                        "metadata": element.metadata.copy()
                    }

                return schema_elements

            except Exception as e:
                self.logger.error(f"Error mapping elements to schema: {str(e)}")
                raise

    def map_attributes_to_schema(
            self,
            entities: Dict[str, DTDEntity]
        ) -> Dict[str, Any]:
            """
            Map DTD attributes and entities to schema format.

            Args:
                entities: Dictionary of DTD entities

            Returns:
                Dict[str, Any]: Schema attribute definitions
            """
            try:
                schema_attributes = {}

                # Process parameter entities first
                for name, entity in entities.items():
                    if entity.is_parameter:
                        schema_attributes[name] = {
                            "type": "parameter_entity",
                            "value": entity.value,
                            "is_external": entity.is_external,
                            "system_id": entity.system_id,
                            "public_id": entity.public_id
                        }

                return schema_attributes

            except Exception as e:
                self.logger.error(f"Error mapping attributes to schema: {str(e)}")
                raise

    def map_content_model_to_schema(
            self,
            content_model: ContentModel
        ) -> Dict[str, Any]:
            """
            Map DTD content model to schema format.

            Args:
                content_model: DTD content model

            Returns:
                Dict[str, Any]: Schema content model definition
            """
            try:
                return {
                    "type": content_model.type,
                    "elements": content_model.elements,
                    "ordering": content_model.ordering,
                    "occurrence": {
                        "min": content_model.occurrence.min,
                        "max": content_model.occurrence.max
                    },
                    "mixed": content_model.mixed,
                    "particles": [
                        self._map_particle_to_schema(particle)
                        for particle in content_model.particles
                    ]
                }

            except Exception as e:
                self.logger.error(f"Error mapping content model to schema: {str(e)}")
                raise

    def map_specialization_to_schema(
            self,
            spec_info: SpecializationInfo
        ) -> Dict[str, Any]:
            """
            Map DTD specialization to schema format.

            Args:
                spec_info: Specialization information

            Returns:
                Dict[str, Any]: Schema specialization definition
            """
            try:
                return {
                    "base_type": spec_info.base_type,
                    "specialized_type": spec_info.specialized_type,
                    "inheritance_path": spec_info.inheritance_path,
                    "attributes": {
                        name: self._build_attribute_schema(attr)
                        for name, attr in spec_info.attributes.items()
                    },
                    "constraints": spec_info.constraints.copy(),
                    "metadata": spec_info.metadata.copy()
                }

            except Exception as e:
                self.logger.error(f"Error mapping specialization to schema: {str(e)}")
                raise

    def generate_validation_patterns(
            self,
            parsing_result: DTDParsingResult
        ) -> Dict[str, Any]:
            """
            Generate validation patterns from DTD parsing result.

            Args:
                parsing_result: DTD parsing result

            Returns:
                Dict[str, Any]: Schema validation patterns
            """
            try:
                patterns: Dict[str, Any] = {
                    "elements": self._generate_element_patterns(parsing_result.elements),
                    "attributes": self._generate_attribute_patterns(parsing_result.entities),
                    "specializations": {
                        name: self._generate_specialization_patterns(spec_info)
                        for name, spec_info in parsing_result.specializations.items()
                    }
                }

                if parsing_result.errors:
                    patterns["validation_errors"] = {
                        "messages": [
                            {
                                "message": msg.message,
                                "severity": msg.severity.value,
                                "code": msg.code
                            }
                            for msg in parsing_result.errors
                        ]
                    }

                if parsing_result.warnings:
                    patterns["validation_warnings"] = {
                        "messages": [
                            {
                                "message": msg.message,
                                "severity": msg.severity.value,
                                "code": msg.code
                            }
                            for msg in parsing_result.warnings
                        ]
                    }

                return patterns

            except Exception as e:
                self.logger.error(f"Error generating validation patterns: {str(e)}")
                raise

    def _generate_element_patterns(
            self,
            elements: Dict[str, DTDElement]
        ) -> Dict[str, Any]:
            """Generate validation patterns for elements."""
            patterns = {}
            for name, element in elements.items():
                patterns[name] = {
                    "content_model": {
                        "allowed_elements": element.content_model.elements,
                        "ordering": element.content_model.ordering,
                        "occurrence": {
                            "min": element.content_model.occurrence.min,
                            "max": element.content_model.occurrence.max
                        }
                    },
                    "required_attributes": [
                        attr_name for attr_name, attr in element.attributes.items()
                        if attr.is_required
                    ]
                }
            return patterns

    def _generate_attribute_patterns(
            self,
            entities: Dict[str, DTDEntity]
        ) -> Dict[str, Any]:
            """Generate validation patterns for attributes."""
            patterns = {}
            for name, entity in entities.items():
                if entity.is_parameter:
                    patterns[name] = {
                        "type": "parameter",
                        "validation": {
                            "pattern": self._create_entity_pattern(entity)
                        }
                    }
            return patterns

    def _generate_specialization_patterns(
        self,
        spec_info: SpecializationInfo
    ) -> Dict[str, Any]:
        """Generate validation patterns for specialization."""
        patterns = {
            "inheritance": {
                "base_type": {
                    "pattern": f"^{spec_info.base_type}$",
                    "message": f"Must extend {spec_info.base_type}"
                },
                "path": {
                    "pattern": "^" + "->".join(spec_info.inheritance_path) + "$",
                    "message": "Must follow valid inheritance path"
                }
            },
            "attributes": {}
        }

        # Add attribute patterns
        for name, attr in spec_info.attributes.items():
            if attr.type == AttributeType.ENUM and attr.allowed_values:
                patterns["attributes"][name] = {
                    "pattern": f"^({'|'.join(attr.allowed_values)})$",
                    "message": f"Must be one of: {', '.join(attr.allowed_values)}"
                }

        return patterns

    def _create_entity_pattern(self, entity: DTDEntity) -> str:
        """Create validation pattern for entity."""
        if entity.is_external:
            return f"^{entity.system_id}$"
        return f"^{re.escape(entity.value)}$"

    def _detect_specialization(
        self,
        element: DTDElement,
        dtd_content: str
    ) -> Optional[SpecializationInfo]:
        """Detect if element is a specialization."""
        try:
            # Look for class attribute that indicates specialization
            for attr_name, attr in element.attributes.items():
                if attr_name == 'class' and attr.default_value:
                    if base_match := re.search(r'- topic/(\w+)', attr.default_value):
                        base_type = base_match.group(1)
                        return SpecializationInfo(
                            base_type=base_type,
                            specialized_type=element.name,
                            inheritance_path=[base_type, element.name],
                            attributes=element.attributes,
                            constraints={},
                            metadata={
                                "source_dtd": dtd_content,
                                "detected": True
                            }
                        )
            return None
        except Exception as e:
            self.logger.error(f"Error detecting specialization: {str(e)}")
            return None

    def parse_dtd(self, dtd_path: Path) -> DTDParsingResult:
        """
        Parse DTD file into schema structure.
        Now includes specialization detection.
        """
        try:
            with open(dtd_path, 'r') as f:
                content = f.read()

            # Parse elements
            elements = self._parse_element_defs(content)

            # Parse attributes and convert to entities
            attributes = self._parse_attribute_defs(content)
            entities = self._convert_attributes_to_entities(attributes)

            # Initialize error tracking
            errors: List[ValidationMessage] = []
            warnings: List[ValidationMessage] = []

            # Detect specializations
            specializations: Dict[str, SpecializationInfo] = {}
            for name, element in elements.items():
                if spec_info := self._detect_specialization(element, content):
                    specializations[name] = spec_info

            # Track parsing metadata
            metadata = {
                "source_dtd": str(dtd_path),
                "parsed_at": datetime.now().isoformat(),
                "element_count": len(elements),
                "entity_count": len(entities),
                "specialization_count": len(specializations)
            }

            # Return structured result
            return DTDParsingResult(
                elements=elements,
                entities=entities,  # Now properly typed
                specializations=specializations,
                errors=errors,
                warnings=warnings,
                metadata=metadata
            )

        except Exception as e:
            error_msg = f"Error parsing DTD {dtd_path}: {str(e)}"
            self.logger.error(error_msg)
            # Return error result
            return DTDParsingResult(
                elements={},
                entities={},
                specializations={},
                errors=[ValidationMessage(
                    path=str(dtd_path),
                    message=error_msg,
                    severity=ValidationSeverity.ERROR,
                    code="dtd_parse_error"
                )],
                warnings=[],
                metadata={
                    "source_dtd": str(dtd_path),
                    "parsed_at": datetime.now().isoformat(),
                    "error": str(e)
                }
            )

    def _convert_attributes_to_entities(
        self,
        attributes: Dict[str, List[DTDAttribute]]
    ) -> Dict[str, DTDEntity]:
        """
        Convert attribute definitions to entity format.

        Args:
            attributes: Dictionary of attribute lists by element

        Returns:
            Dict[str, DTDEntity]: Converted attribute entities
        """
        entities: Dict[str, DTDEntity] = {}

        for element_name, attr_list in attributes.items():
            # Create entity for each attribute list
            entity_value = self._create_attribute_entity_value(attr_list)
            entities[element_name] = DTDEntity(
                name=f"{element_name}_attrs",
                value=entity_value,
                is_parameter=True,  # Attribute lists are parameter entities
                is_external=False,
                system_id=None,
                public_id=None
            )

        return entities

    def _create_attribute_entity_value(self, attributes: List[DTDAttribute]) -> str:
        """Create entity value string from attribute list."""
        attr_strings = []
        for attr in attributes:
            # Convert attribute type
            type_str = attr.type.value
            if attr.type == AttributeType.ENUM and attr.allowed_values:
                type_str = f"({' | '.join(attr.allowed_values)})"

            # Convert default value
            if attr.default_type == AttributeDefault.REQUIRED:
                default_str = "#REQUIRED"
            elif attr.default_type == AttributeDefault.IMPLIED:
                default_str = "#IMPLIED"
            elif attr.default_type == AttributeDefault.FIXED:
                default_str = f'#FIXED "{attr.default_value}"'
            else:
                default_str = f'"{attr.default_value}"' if attr.default_value else "#IMPLIED"

            attr_strings.append(f"{attr.name} {type_str} {default_str}")

        return "\n".join(attr_strings)

    def _map_particle_to_schema(
        self,
        particle: ContentModelParticle
    ) -> Dict[str, Any]:
        """
        Map content model particle to schema format.

        Args:
            particle: Content model particle

        Returns:
            Dict[str, Any]: Schema particle definition
        """
        try:
            return {
                "content": (
                    particle.content if isinstance(particle.content, str)
                    else [self._map_particle_to_schema(p) for p in particle.content]
                ),
                "type": particle.type.value,
                "occurrence": particle.occurrence.value,
                "is_group": particle.is_group
            }

        except Exception as e:
            self.logger.error(f"Error mapping particle to schema: {str(e)}")
            raise

    def _build_specialization_schema(
        self,
        spec_info: SpecializationInfo
    ) -> Dict[str, Any]:
        """
        Convert specialization info to schema format with validation support.

        Args:
            spec_info: Specialization information

        Returns:
            Dict[str, Any]: Enhanced schema definition for specialization
        """
        try:
            schema = {
                "base_type": spec_info.base_type,
                "inheritance_path": spec_info.inheritance_path,
                "attributes": {
                    name: self._build_attribute_schema(attr)
                    for name, attr in spec_info.attributes.items()
                },
                "constraints": spec_info.constraints.copy(),
                "validation": {
                    "required_attributes": [
                        name for name, attr in spec_info.attributes.items()
                        if attr.is_required
                    ],
                    "inheritance_rules": {
                        "must_preserve": [
                            attr_name for attr_name, attr in spec_info.attributes.items()
                            if attr.default_type == AttributeDefault.REQUIRED
                        ],
                        "allow_override": [
                            attr_name for attr_name, attr in spec_info.attributes.items()
                            if attr.default_type != AttributeDefault.FIXED
                        ]
                    },
                    "content_restrictions": spec_info.constraints.get("content", {}),
                    "attribute_restrictions": spec_info.constraints.get("attributes", {})
                },
                "metadata": spec_info.metadata.copy()
            }

            # Add validation patterns
            if spec_info.constraints:
                schema["validation_patterns"] = self._generate_specialization_patterns(spec_info)

            return schema

        except Exception as e:
            self.logger.error(f"Error building specialization schema: {str(e)}")
            raise

    def _parse_attribute_defs(self, content: str) -> Dict[str, List[DTDAttribute]]:
        """Parse attribute definitions from DTD content."""
        attributes = {}

        pattern = r'<!ATTLIST\s+(\w+)\s+([^>]+)>'
        for match in re.finditer(pattern, content):
            element_name = match.group(1)
            attr_defs = match.group(2).strip()

            attr_pattern = r'(\w+)\s+([\w\(\)|]+)\s+(#REQUIRED|#IMPLIED|#FIXED\s+"[^"]*"|"[^"]*")'
            element_attrs = []

            for attr_match in re.finditer(attr_pattern, attr_defs):
                attr_name = attr_match.group(1)
                attr_type_str = attr_match.group(2)
                default_str = attr_match.group(3)

                # Initialize allowed_values
                allowed_values: Optional[List[str]] = None

                # Parse type
                if '(' in attr_type_str:
                    attr_type = AttributeType.ENUM
                    values_match = re.findall(r'\((.+?)\)', attr_type_str)
                    if values_match:
                        allowed_values = [v.strip() for v in values_match[0].split('|')]
                else:
                    try:
                        attr_type = AttributeType[attr_type_str]
                    except KeyError:
                        attr_type = AttributeType.CDATA

                # Parse default
                if default_str == '#REQUIRED':
                    default_type = AttributeDefault.REQUIRED
                    default_value = None
                elif default_str == '#IMPLIED':
                    default_type = AttributeDefault.IMPLIED
                    default_value = None
                elif default_str.startswith('#FIXED'):
                    default_type = AttributeDefault.FIXED
                    default_value = default_str.split('"')[1]
                else:
                    default_type = AttributeDefault.DEFAULT
                    default_value = default_str.strip('"')

                attr = DTDAttribute(
                    name=attr_name,
                    type=attr_type,
                    default_type=default_type,
                    default_value=default_value,
                    allowed_values=allowed_values,  # Now it's properly initialized
                    is_required=default_type == AttributeDefault.REQUIRED
                )
                element_attrs.append(attr)

            attributes[element_name] = element_attrs

        return attributes

    def _build_element_schemas(self, elements: Dict[str, DTDElement]) -> Dict[str, Any]:
        """Build schema definitions for elements."""
        schemas = {}

        for name, element in elements.items():
            # Don't parse content model again since it's already a ContentModel
            schema = {
                "content_model": {
                    "type": element.content_model.type,
                    "elements": element.content_model.elements,
                    "ordering": element.content_model.ordering,
                    "occurrence": {
                        "min": element.content_model.occurrence.min,
                        "max": element.content_model.occurrence.max
                    }
                },
                "attributes": element.attributes,
                # Use is_abstract instead of is_empty since that's what our model has
                "is_abstract": element.is_abstract
            }

            if parent := self._inheritance_map.get(name):
                schema["extends"] = parent

            schemas[name] = schema

        return schemas

    def _build_attribute_schema(
        self,
        attribute: DTDAttribute
    ) -> Dict[str, Any]:
        """Convert attribute to schema format."""
        schema = {
            "type": self._map_attribute_type(attribute.type),
            "required": attribute.is_required,  # Use is_required from model
            "default_type": attribute.default_type.value,
            "default_value": attribute.default_value
        }

        if attribute.allowed_values:
            schema["allowed_values"] = attribute.allowed_values

        return schema
    def _build_attribute_schemas(
        self,
        attributes: Dict[str, List[DTDAttribute]]
    ) -> Dict[str, Any]:
        """Build schema definitions for attributes."""
        schemas = {}

        for element_name, attrs in attributes.items():
            element_attrs = {}

            for attr in attrs:
                attr_schema = {
                    "type": self._map_attribute_type(attr.type),
                    "required": attr.is_required,
                    "default_type": attr.default_type.value,
                    "default_value": attr.default_value
                }

                if attr.allowed_values:
                    attr_schema["allowed_values"] = attr.allowed_values

                element_attrs[attr.name] = attr_schema

            schemas[element_name] = element_attrs

        return schemas

    def _build_inheritance_map(
        self,
        elements: Dict[str, DTDElement]
    ) -> Dict[str, Any]:
        """
        Build element inheritance relationships with validation.

        Args:
            elements: Dictionary of DTD elements

        Returns:
            Dict[str, Any]: Inheritance map with validation info
        """
        try:
            inheritance: Dict[str, Any] = {}

            # Build basic inheritance relationships
            for name, element in elements.items():
                if base_type := self._detect_base_element(element.content_model):
                    if name not in inheritance:
                        inheritance[name] = {
                            "extends": base_type,
                            "validation": {
                                "rules": [],
                                "constraints": {}
                            }
                        }

            # Add validation rules for inheritance
            for name, info in inheritance.items():
                base_element = elements.get(info["extends"])
                if base_element:
                    # Add content model validation
                    info["validation"]["rules"].extend([
                        {
                            "type": "content_model",
                            "rule": "must_preserve_required",
                            "elements": base_element.content_model.elements
                        }
                    ])

                    # Add attribute validation
                    info["validation"]["rules"].extend([
                        {
                            "type": "attributes",
                            "rule": "must_preserve_required",
                            "attributes": [
                                attr_name for attr_name, attr in base_element.attributes.items()
                                if attr.is_required
                            ]
                        }
                    ])

                    # Add constraints
                    info["validation"]["constraints"].update({
                        "content_model": {
                            "allowed_elements": base_element.content_model.elements,
                            "required_elements": [
                                elem for elem in base_element.content_model.elements
                                if base_element.content_model.occurrence.min > 0
                            ]
                        },
                        "attributes": {
                            "required": [
                                attr_name for attr_name, attr in base_element.attributes.items()
                                if attr.is_required
                            ],
                            "fixed": [
                                attr_name for attr_name, attr in base_element.attributes.items()
                                if attr.default_type == AttributeDefault.FIXED
                            ]
                        }
                    })

            return inheritance

        except Exception as e:
            self.logger.error(f"Error building inheritance map: {str(e)}")
            raise

    def _parse_content_model(self, content_model: str) -> ContentModel:
        """Parse DTD content model into schema structure.

        Args:
            content_model: DTD content model string

        Returns:
            ContentModel object representing the content structure
        """
        if content_model == 'EMPTY':
            return ContentModel.empty()

        if content_model == '(#PCDATA)':
            return ContentModel.text()

        # Parse complex content models
        return ContentModel(
            type='complex',
            elements=self._parse_element_refs(content_model),
            ordering=self._detect_ordering(content_model),
            occurrence=self._parse_occurrence(content_model)
        )

    def _parse_element_refs(self, content_model: str) -> List[str]:
        """Extract element references from content model."""
        # Remove parentheses and operators
        cleaned = re.sub(r'[()?,+*|]', ' ', content_model)
        # Split and filter element names
        return [ref.strip() for ref in cleaned.split() if ref.strip()]

    def _detect_ordering(self, content_model: str) -> str:
        """Detect content model ordering requirements."""
        if '|' in content_model:
            return "choice"
        return "sequence"

    def _parse_occurrence(self, content_model: str) -> OccurrenceConstraint:
        """Parse element occurrence constraints.

        Args:
            content_model: DTD content model string

        Returns:
            OccurrenceConstraint with min/max values. For unbounded max,
            uses sys.maxsize to represent infinity.
        """
        if '+' in content_model:
            return OccurrenceConstraint(min=1, max=sys.maxsize)
        elif '*' in content_model:
            return OccurrenceConstraint(min=0, max=sys.maxsize)
        elif '?' in content_model:
            return OccurrenceConstraint(min=0, max=1)
        else:
            return OccurrenceConstraint(min=1, max=1)

    def _map_attribute_type(self, attr_type: AttributeType) -> str:
        """Map DTD attribute types to schema types."""
        type_mapping = {
            AttributeType.CDATA: 'string',
            AttributeType.ID: 'id',
            AttributeType.IDREF: 'reference',
            AttributeType.IDREFS: 'references',
            AttributeType.NMTOKEN: 'token',
            AttributeType.NMTOKENS: 'tokens',
            AttributeType.ENTITY: 'entity',
            AttributeType.ENTITIES: 'entities',
            AttributeType.ENUM: 'enum',
            AttributeType.NOTATION: 'notation'
        }
        return type_mapping.get(attr_type, 'string')

    def _parse_default_value(self, default: str) -> Any:
        """Parse attribute default value."""
        if default == '#REQUIRED':
            return None
        elif default == '#IMPLIED':
            return None
        elif default.startswith('#FIXED'):
            return default.split('"')[1]
        else:
            return default.strip('"')

    def _detect_base_element(self, content_model: ContentModel) -> Optional[str]:
        """Detect potential base element from content model."""
        try:
            if content_model.type == 'complex':
                # Look for specialization patterns in elements
                for element in content_model.elements:
                    if element.endswith('Base'):
                        return element
            return None
        except Exception as e:
            self.logger.error(f"Error detecting base element: {str(e)}")
            return None

    def map_to_validation_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map parsed DTD schema to validation schema format.

        Args:
            schema: Parsed DTD schema

        Returns:
            Dict[str, Any]: Validation schema structure
        """
        validation_schema = {
            "elements": {},
            "attributes": {},
            "patterns": []
        }

        # Map elements
        for name, element in schema["elements"].items():
            validation_schema["elements"][name] = {
                "validation": {
                    "content_model": self._map_content_validation(element["content_model"]),
                    "attributes": self._map_attribute_validation(
                        schema["attributes"].get(name, {})
                    )
                }
            }

        return validation_schema

    def _map_content_validation(self, content_model: Dict[str, Any]) -> Dict[str, Any]:
        """Map content model to validation rules."""
        validation = {
            "type": content_model["type"]
        }

        if content_model["type"] == "complex":
            validation.update({
                "allowed_elements": content_model["elements"],
                "ordering": content_model["ordering"],
                "occurrence": content_model["occurrence"]
            })

        return validation

    def _map_attribute_validation(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Map attribute definitions to validation rules."""
        validation = {}

        for name, attr in attributes.items():
            rules = {
                "type": attr["type"],
                "required": attr["required"]
            }

            if "allowed_values" in attr:
                rules["pattern"] = f"^({'|'.join(attr['allowed_values'])})$"

            validation[name] = rules

        return validation

    def get_inheritance_chain(
        self,
        schema_name: str,
        element_name: str
    ) -> List[str]:
        """
        Get inheritance chain for element.

        Args:
            schema_name: Name of schema containing element
            element_name: Name of element

        Returns:
            List[str]: Inheritance chain from root to element
        """
        try:
            chain = []
            current = element_name

            while current:
                chain.append(current)
                # Look for parent in inheritance map
                found_parent = False
                if schema_rules := self._inheritance_map.get(schema_name, []):
                    for child, parent in schema_rules:
                        if child == current:
                            current = parent
                            found_parent = True
                            break
                if not found_parent:
                    break

            return list(reversed(chain))

        except Exception as e:
            self.logger.error(f"Error getting inheritance chain: {str(e)}")
            return [element_name]
