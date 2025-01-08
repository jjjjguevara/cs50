# app/dita/schema/schema_manager.py
"""Schema management and inheritance for DITA processing."""

from typing import Dict, List, Optional, Any, Set, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import json
import re

# Core managers
from ..validation_manager import ValidationManager
from ..event_manager import EventManager, EventType
from ..config.config_manager import ConfigManager
from ..key_manager import KeyManager
from ..metadata.metadata_manager import MetadataManager

# Schema components
from .schema_compatibility import SchemaCompatibilityChecker
from .schema_validator import SchemaValidator
from .schema_resolver import SchemaResolver
from .schema_migrator import SchemaMigrator

# DTD specific
from ..dtd.dtd_models import (
    DTDElement,
    DTDEntity,
    DTDAttribute,
    SpecializationInfo,
    DTDParsingResult,
    ValidationState,
    ValidationMessage,
    ValidationSeverity
)

from ..dtd.dtd_validator import DTDValidator
from ..dtd.dtd_mapper import DTDSchemaMapper

# Utils
from ..utils.cache import ContentCache, CacheEntryType
from ..utils.logger import DITALogger

# Types
from ..models.types import (
    ElementType,
    ProcessingPhase,
    ProcessingContext,
    ValidationResult,
    ProcessingState,
    ProcessingStatus
)

class CompositionStrategy(Enum):
    """Strategies for schema composition."""
    MERGE = "merge"           # Deep recursive merge
    OVERRIDE = "override"     # Complete override
    ADDITIVE = "additive"     # Only add new fields
    SELECTIVE = "selective"   # Merge specific fields only

@dataclass
class SchemaVersion:
    """Schema version information with proper comparison support."""
    major: int
    minor: int
    patch: int
    metadata: Optional[str] = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SchemaVersion):
            return NotImplemented
        return (
            self.major == other.major and
            self.minor == other.minor and
            self.patch == other.patch
        )

    def __lt__(self, other: 'SchemaVersion') -> bool:
        if not isinstance(other, SchemaVersion):
            return NotImplemented
        return (
            (self.major, self.minor, self.patch) <
            (other.major, other.minor, other.patch)
        )

    def __le__(self, other: 'SchemaVersion') -> bool:
        return self < other or self == other

    def __gt__(self, other: 'SchemaVersion') -> bool:
        if not isinstance(other, SchemaVersion):
            return NotImplemented
        return (
            (self.major, self.minor, self.patch) >
            (other.major, other.minor, other.patch)
        )

    def __ge__(self, other: 'SchemaVersion') -> bool:
        return self > other or self == other

    @classmethod
    def from_string(cls, version_str: str) -> 'SchemaVersion':
        """Create version from string (e.g., '1.0.0')."""
        try:
            # Handle metadata suffix
            version_parts = version_str.split("-", 1)
            version = version_parts[0]
            metadata = version_parts[1] if len(version_parts) > 1 else None

            # Parse version numbers
            major, minor, patch = map(int, version.split("."))
            return cls(major, minor, patch, metadata)
        except Exception as e:
            raise ValueError(f"Invalid version string: {version_str}") from e

    def __str__(self) -> str:
        """Convert version to string."""
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.metadata:
            return f"{base}-{self.metadata}"
        return base

    def is_compatible_with(self, other: 'SchemaVersion') -> bool:
        """Check version compatibility (same major version)."""
        return self.major == other.major

@dataclass
class SchemaInfo:
    """Schema information with versioning and metadata."""
    name: str
    version: SchemaVersion
    schema: Dict[str, Any]
    base_schemas: List[str]
    overrides: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

class SchemaComposer:
    """Handles schema composition with different strategies."""

    def __init__(self):
        self.strategies: Dict[CompositionStrategy, Callable] = {
            CompositionStrategy.MERGE: self._merge_strategy,
            CompositionStrategy.OVERRIDE: self._override_strategy,
            CompositionStrategy.ADDITIVE: self._additive_strategy,
            CompositionStrategy.SELECTIVE: self._selective_strategy
        }

    def compose(
        self,
        base: Dict[str, Any],
        extension: Dict[str, Any],
        strategy: CompositionStrategy,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compose schemas using specified strategy."""
        composer = self.strategies.get(strategy, self._merge_strategy)
        return composer(base, extension, options or {})

    def _merge_strategy(
        self,
        base: Dict[str, Any],
        extension: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep recursive merge strategy."""
        result = base.copy()

        for key, value in extension.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_strategy(result[key], value, options)
            else:
                result[key] = value

        return result

    def _override_strategy(
        self,
        base: Dict[str, Any],
        extension: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Complete override strategy."""
        return extension.copy()

    def _additive_strategy(
        self,
        base: Dict[str, Any],
        extension: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add only new fields strategy."""
        result = base.copy()

        for key, value in extension.items():
            if key not in result:
                result[key] = value
            elif isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._additive_strategy(result[key], value, options)

        return result

    def _selective_strategy(
        self,
        base: Dict[str, Any],
        extension: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge only specified fields strategy."""
        result = base.copy()
        fields = options.get("fields", [])

        for key in fields:
            if key in extension:
                if key in result and isinstance(result[key], dict) and isinstance(extension[key], dict):
                    result[key] = self._merge_strategy(result[key], extension[key], options)
                else:
                    result[key] = extension[key]

        return result

class OverrideType(Enum):
    """Types of schema overrides."""
    VALUE = "value"           # Direct value override
    PATTERN = "pattern"       # Pattern-based override
    CONDITIONAL = "conditional"  # Condition-based override
    PRIORITY = "priority"     # Priority-based override

@dataclass
class Override:
    """Schema override definition."""
    type: OverrideType
    path: str
    value: Any
    pattern: Optional[str] = None
    condition: Optional[Dict[str, Any]] = None
    priority: int = 0

class OverrideManager:
    """Manages schema overrides with advanced features."""

    def __init__(self):
        self._override_handlers: Dict[OverrideType, Callable] = {
            OverrideType.VALUE: self._handle_value_override,
            OverrideType.PATTERN: self._handle_pattern_override,
            OverrideType.CONDITIONAL: self._handle_conditional_override,
            OverrideType.PRIORITY: self._handle_priority_override
        }

    def apply_overrides(
        self,
        schema: Dict[str, Any],
        overrides: List[Override],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Apply overrides to schema."""
        result = schema.copy()

        # Sort overrides by priority
        sorted_overrides = sorted(overrides, key=lambda x: x.priority, reverse=True)

        for override in sorted_overrides:
            handler = self._override_handlers.get(override.type)
            if handler:
                result = handler(result, override, context)

        return result

    def _handle_value_override(
        self,
        schema: Dict[str, Any],
        override: Override,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Handle direct value override."""
        result = schema.copy()
        parts = override.path.split('.')
        current = result

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = override.value
        return result

    def _handle_pattern_override(
        self,
        schema: Dict[str, Any],
        override: Override,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Handle pattern-based override."""
        result = schema.copy()
        if not override.pattern:
            return result

        pattern = re.compile(override.pattern)

        def apply_pattern_override(d: Dict[str, Any], path: str = "") -> None:
            for key, value in list(d.items()):
                current_path = f"{path}.{key}" if path else key
                if pattern.match(current_path):
                    d[key] = override.value
                elif isinstance(value, dict):
                    apply_pattern_override(value, current_path)

        apply_pattern_override(result)
        return result

    def _handle_conditional_override(
        self,
        schema: Dict[str, Any],
        override: Override,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Handle condition-based override."""
        if not context or not override.condition:
            return schema

        # Evaluate condition
        if self._evaluate_condition(override.condition, context):
            return self._handle_value_override(schema, override, context)

        return schema

    def _handle_priority_override(
        self,
        schema: Dict[str, Any],
        override: Override,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Handle priority-based override."""
        return self._handle_value_override(schema, override, context)

    def _evaluate_condition(
        self,
        condition: Dict[str, Any],
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate override condition."""
        for key, expected in condition.items():
            actual = context.get(key)

            if isinstance(expected, dict):
                if "$lt" in expected and not (actual < expected["$lt"]):
                    return False
                elif "$gt" in expected and not (actual > expected["$gt"]):
                    return False
                elif "$in" in expected and actual not in expected["$in"]:
                    return False
                elif "$regex" in expected and not re.match(expected["$regex"], str(actual)):
                    return False
            elif actual != expected:
                return False

        return True

class SchemaManager:
    """Enhanced schema management with versioning and inheritance."""

    def __init__(
        self,
        config_path: Path,
        cache: ContentCache,
        dtd_validator: DTDValidator,
        dtd_mapper: DTDSchemaMapper,
        config_manager: ConfigManager,
        validation_manager: ValidationManager,
        event_manager: EventManager,
        logger: Optional[DITALogger] = None
    ):
        """Initialize schema manager."""
        self.dtd_mapper = dtd_mapper
        self.config_path = Path(config_path)
        self.cache = cache
        self.logger = logger or DITALogger(name=__name__)

        # Schema storage
        self._schemas: Dict[str, SchemaInfo] = {}
        self._inheritance_graph: Dict[str, Set[str]] = {}
        self._override_maps: Dict[str, Dict[str, Any]] = {}
        self._schema_registry: Dict[str, Dict[str, Any]] = {}
        self._inheritance_map: Dict[str, List[Tuple[str, str]]] = {}
        self._schema_versions: Dict[str, SchemaVersion] = {}
        self._loaded_schemas: Set[str] = set()

        # Manager dependencies
        self.config_manager = config_manager
        self.validation_manager = validation_manager
        self.event_manager = event_manager

        # Schema components
        self.schema_composer = SchemaComposer()
        self.override_manager = OverrideManager()

        # Resolution tracking
        self._resolution_stack: List[str] = []
        self._resolution_cache: Dict[str, Dict[str, Any]] = {}

        # Track inheritance depth to prevent circular references
        self._inheritance_depth = 0
        self._max_inheritance_depth = 10

        # Add DTD-specific tracking
        self._dtd_validator: DTDValidator = dtd_validator
        self._dtd_schemas: Dict[str, Dict[str, Any]] = {}
        self._dtd_inheritance: Dict[str, List[str]] = {}

    def initialize(self) -> None:
            """Initialize schema system."""
            try:
                # Load core schemas
                self._load_attribute_schema()
                self._load_processing_schema()
                self._load_validation_schema()

                # Build inheritance map
                self._build_inheritance_map()

                self.logger.info("Schema manager initialized successfully")

            except Exception as e:
                self.logger.error(f"Failed to initialize schema manager: {str(e)}")
                raise

    def _load_attribute_schema(self) -> None:
        """Load attribute schema with inheritance support."""
        try:
            schema_path = self.config_path / "attribute_schema.json"
            if not schema_path.exists():
                raise FileNotFoundError(f"Attribute schema not found: {schema_path}")

            with open(schema_path) as f:
                schema = json.load(f)

            # Store schema version
            if version_str := schema.get("version"):
                self._schema_versions["attribute"] = SchemaVersion.from_string(version_str)

            # Store schema
            self._schema_registry["attribute"] = schema
            self._loaded_schemas.add("attribute")

        except Exception as e:
            self.logger.error(f"Error loading attribute schema: {str(e)}")
            raise

    def _load_processing_schema(self) -> None:
        """Load processing schema with inheritance support."""
        try:
            schema_path = self.config_path / "processing_rules.json"
            if not schema_path.exists():
                raise FileNotFoundError(f"Processing schema not found: {schema_path}")

            with open(schema_path) as f:
                schema = json.load(f)

            # Store schema version
            if version_str := schema.get("version"):
                self._schema_versions["processing"] = SchemaVersion.from_string(version_str)

            # Extract inheritance information
            for rule_type, rules in schema.get("rules", {}).items():
                for rule_id, rule_data in rules.items():
                    if inherits_from := rule_data.get("inherits_from"):
                        if rule_type not in self._inheritance_map:
                            self._inheritance_map[rule_type] = []
                        self._inheritance_map[rule_type].append((str(rule_id), str(inherits_from)))

            # Store schema
            self._schema_registry["processing"] = schema
            self._loaded_schemas.add("processing")

        except Exception as e:
            self.logger.error(f"Error loading processing schema: {str(e)}")
            raise

    def _load_validation_schema(self) -> None:
        """Load validation patterns schema."""
        try:
            schema_path = self.config_path / "validation_patterns.json"
            if not schema_path.exists():
                raise FileNotFoundError(f"Validation schema not found: {schema_path}")

            with open(schema_path) as f:
                schema = json.load(f)

            # Store schema version
            if version_str := schema.get("version"):
                self._schema_versions["validation"] = SchemaVersion.from_string(version_str)

            # Store schema
            self._schema_registry["validation"] = schema
            self._loaded_schemas.add("validation")

        except Exception as e:
            self.logger.error(f"Error loading validation schema: {str(e)}")
            raise

    def _build_inheritance_map(self) -> None:
        """Build complete inheritance map from all schemas."""
        try:
            # Reset existing map
            self._inheritance_map.clear()

            # Process each schema
            for schema_name, schema in self._schema_registry.items():
                if inheritance := schema.get("inheritance"):
                    for child, parent in inheritance.items():
                        if schema_name not in self._inheritance_map:
                            self._inheritance_map[schema_name] = []
                        # Ensure both child and parent are strings
                        self._inheritance_map[schema_name].append((str(child), str(parent)))

        except Exception as e:
            self.logger.error(f"Error building inheritance map: {str(e)}")
            raise

    def register_schema(
        self,
        name: str,
        schema: Dict[str, Any],
        version: str,
        base_schemas: Optional[List[str]] = None,
        overrides: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a new schema with inheritance information."""
        try:
            # Parse version
            schema_version = SchemaVersion.from_string(version)

            # Create schema info
            schema_info = SchemaInfo(
                name=name,
                version=schema_version,
                schema=schema,
                base_schemas=base_schemas or [],
                overrides=overrides or {},
                metadata=metadata or {},
                created_at=datetime.now()
            )

            # Update inheritance graph
            if base_schemas:
                self._inheritance_graph[name] = set(base_schemas)

            # Store schema
            self._schemas[name] = schema_info

            # Clear resolution cache
            self._resolution_cache.clear()

        except Exception as e:
            self.logger.error(f"Error registering schema {name}: {str(e)}")
            raise

    def register_dtd_schema(
        self,
        dtd_path: Path,
        dtd_mapper: DTDSchemaMapper
    ) -> ValidationResult:
        """Register schema from DTD source."""
        try:
            # Convert DTD to schema
            parsing_result = dtd_mapper.parse_dtd(dtd_path)

            # Check for parsing errors
            if parsing_result.errors:
                return ValidationResult(
                    is_valid=False,
                    messages=parsing_result.errors
                )

            # Convert parsing result to schema dictionary
            schema = {
                "elements": self._build_element_schemas(parsing_result.elements),
                "attributes": self._build_attribute_schemas(parsing_result.entities),
                "inheritance": parsing_result.metadata.get("inheritance", {}),
                "specializations": {
                    name: spec.__dict__ for name, spec in
                    parsing_result.specializations.items()
                },
                "metadata": parsing_result.metadata
            }

            # Get DTD configuration
            dtd_config = self.config_manager.get_dtd_validation_config()

            # Validate before storing if strict mode enabled
            if dtd_config.get("validation_mode") == "strict":
                validation_result = self.validation_manager.validate_dtd_schema(
                    schema=schema,
                    dtd_path=dtd_path
                )
                if not validation_result.is_valid:
                    return validation_result

            # Store schema with DTD source information
            schema_name = dtd_path.stem
            self._schema_registry[schema_name] = schema
            self._dtd_schemas[schema_name] = {
                'path': str(dtd_path),
                'schema': schema,
                'inheritance': dtd_mapper.get_inheritance_chain(schema_name, '')
            }

            # Track DTD-based inheritance
            if inheritance := schema.get('inheritance', {}):
                self._dtd_inheritance.update(inheritance)

            # Cache schema
            self.cache.set(
                key=f"schema_dtd_{schema_name}",
                data=schema,
                entry_type=CacheEntryType.CONTENT,
                element_type=ElementType.UNKNOWN,
                phase=ProcessingPhase.DISCOVERY
            )

            return ValidationResult(is_valid=True, messages=[])

        except Exception as e:
            self.logger.error(f"Error registering DTD schema: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path=str(dtd_path),
                    message=f"Schema registration failed: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="dtd_registration_error"
                )]
            )

    def _build_element_schemas(
        self,
        elements: Dict[str, DTDElement]
    ) -> Dict[str, Any]:
        """Build schema definitions for elements."""
        return {
            name: {
                "content_model": element.content_model.__dict__,
                "attributes": {
                    attr_name: attr.__dict__
                    for attr_name, attr in element.attributes.items()
                },
                "is_abstract": element.is_abstract,
                "base_type": element.base_type,
                "metadata": element.metadata
            }
            for name, element in elements.items()
        }

    def _build_attribute_schemas(
        self,
        entities: Dict[str, DTDEntity]
    ) -> Dict[str, Any]:
        """Build schema definitions for attributes."""
        return {
            name: {
                "type": entity.is_parameter and "parameter" or "general",
                "value": entity.value,
                "is_external": entity.is_external,
                "system_id": entity.system_id,
                "public_id": entity.public_id
            }
            for name, entity in entities.items()
        }

    def compose_dtd_schema(
        self,
        base_dtd: Dict[str, Any],
        extension_dtd: Dict[str, Any],
        strategy: CompositionStrategy = CompositionStrategy.MERGE,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compose DTD schemas using composition strategy."""
        try:
            # Initialize result with base DTD copy
            result = base_dtd.copy()

            # Apply composition based on strategy
            if strategy == CompositionStrategy.MERGE:
                # Merge elements
                if 'elements' in extension_dtd:
                    result['elements'] = self._merge_dtd_elements(
                        result.get('elements', {}),
                        extension_dtd['elements']
                    )

                # Merge attributes
                if 'attributes' in extension_dtd:
                    result['attributes'] = self._merge_dtd_attributes(
                        result.get('attributes', {}),
                        extension_dtd['attributes']
                    )

                # Handle specializations
                if 'specializations' in extension_dtd:
                    result['specializations'] = self._merge_dtd_specializations(
                        result.get('specializations', {}),
                        extension_dtd['specializations']
                    )

            elif strategy == CompositionStrategy.OVERRIDE:
                # Complete override
                result = extension_dtd.copy()

            elif strategy == CompositionStrategy.SELECTIVE:
                # Selective merge based on options
                if options and "fields" in options:
                    for field in options["fields"]:
                        if field in extension_dtd:
                            result[field] = extension_dtd[field]

            # Update metadata
            if "metadata" in extension_dtd:
                result["metadata"] = {
                    **result.get("metadata", {}),
                    **extension_dtd["metadata"]
                }

            return result

        except Exception as e:
            self.logger.error(f"Error composing DTD schemas: {str(e)}")
            raise

    def _merge_dtd_elements(
        self,
        base_elements: Dict[str, Any],
        extension_elements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge DTD element definitions."""
        result = base_elements.copy()

        for name, element in extension_elements.items():
            if name in result:
                # Merge existing element
                existing = result[name]
                result[name] = {
                    "content_model": self._merge_content_models(
                        existing.get('content_model', {}),
                        element.get('content_model', {})
                    ),
                    "attributes": self._merge_dtd_attributes(
                        existing.get('attributes', {}),
                        element.get('attributes', {})
                    ),
                    "is_abstract": element.get('is_abstract', existing.get('is_abstract')),
                    "base_type": element.get('base_type', existing.get('base_type')),
                    "metadata": {
                        **existing.get('metadata', {}),
                        **element.get('metadata', {})
                    }
                }
            else:
                # Add new element
                result[name] = element

        return result

    def _merge_content_models(
        self,
        base_model: Dict[str, Any],
        extension_model: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge content model definitions."""
        if base_model.get('type') != extension_model.get('type'):
            # If types don't match, use extension
            return extension_model.copy()

        return {
            "type": extension_model.get('type', base_model.get('type')),
            "elements": list(set(
                base_model.get('elements', []) +
                extension_model.get('elements', [])
            )),
            "ordering": extension_model.get('ordering', base_model.get('ordering')),
            "occurrence": extension_model.get('occurrence', base_model.get('occurrence')),
            "mixed": extension_model.get('mixed', base_model.get('mixed')),
            "particles": base_model.get('particles', []) + extension_model.get('particles', [])
        }

    def _merge_dtd_specializations(
        self,
        base_specs: Dict[str, Any],
        extension_specs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge DTD specialization definitions."""
        result = base_specs.copy()

        for name, spec in extension_specs.items():
            if name in result:
                # Merge existing specialization
                existing = result[name]
                result[name] = {
                    "base_type": spec.get('base_type', existing.get('base_type')),
                    "specialized_type": spec.get('specialized_type', existing.get('specialized_type')),
                    "inheritance_path": spec.get('inheritance_path', existing.get('inheritance_path')),
                    "attributes": self._merge_dtd_attributes(
                        existing.get('attributes', {}),
                        spec.get('attributes', {})
                    ),
                    "constraints": {
                        **existing.get('constraints', {}),
                        **spec.get('constraints', {})
                    },
                    "metadata": {
                        **existing.get('metadata', {}),
                        **spec.get('metadata', {})
                    }
                }
            else:
                # Add new specialization
                result[name] = spec

        return result

    def _merge_dtd_attributes(
        self,
        base_attrs: Dict[str, Any],
        extension_attrs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge DTD attribute definitions."""
        result = base_attrs.copy()

        for attr_name, ext_attr in extension_attrs.items():
            if attr_name in result:
                base_attr = result[attr_name]

                # Determine attribute type
                attr_type = ext_attr.get('type', base_attr.get('type'))

                # Merge allowed values for enums
                allowed_values = base_attr.get('allowed_values', [])
                if ext_attr.get('allowed_values'):
                    allowed_values = list(set(
                        allowed_values + ext_attr['allowed_values']
                    ))

                # Create merged attribute
                result[attr_name] = {
                    "type": attr_type,
                    "required": ext_attr.get('required', base_attr.get('required', False)),
                    "default_type": ext_attr.get('default_type', base_attr.get('default_type')),
                    "default_value": ext_attr.get('default_value', base_attr.get('default_value')),
                    "allowed_values": allowed_values if attr_type == "enum" else None,
                    "metadata": {
                        **base_attr.get('metadata', {}),
                        **ext_attr.get('metadata', {})
                    }
                }
            else:
                # Add new attribute
                result[attr_name] = ext_attr

        return result

    def cleanup(self) -> None:
        """Clean up manager resources."""
        try:
            self._schemas.clear()
            self._dtd_schemas.clear()
            self._inheritance_graph.clear()
            self._override_maps.clear()
            self._schema_registry.clear()
            self._inheritance_map.clear()
            self._schema_versions.clear()
            self._loaded_schemas.clear()
            self._resolution_cache.clear()
            self.cache.invalidate_by_pattern("schema_*")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise
