"""Schema management and inheritance for DITA processing."""

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..validation_manager import ValidationManager
from typing import (
    Dict,
    Any,
    Optional,
    Any,
    List,
    Union,
    Set,
    Tuple,
    Callable,
    Protocol
)
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
import logging
import json
import re

from ..dtd.dtd_validator import DTDValidator
from ..dtd.dtd_mapper import DTDSchemaMapper, SpecializationInfo
from ..utils.cache import ContentCache, CacheEntryType
from ..event_manager import EventManager, EventType
from ..config.config_manager import ConfigManager
from .schema_resolver import SchemaResolver
from ..utils.logger import DITALogger
from ..models.types import (
    ElementType,
    ProcessingPhase,
    ProcessingRuleType,
    ContentScope,
    ValidationResult,
    ValidationMessage,
    ValidationSeverity,
    ProcessingState,
    ProcessingPhase,
    ProcessingStatus
)
from ..dtd.dtd_models import (
    DTDCompositionOptions,
    DTDElement,
    DTDEntity,
    DTDParsingResult,
    SpecializationInfo
)




class CompositionStrategy(Enum):
    """Strategies for schema composition."""
    MERGE = "merge"           # Deep recursive merge
    OVERRIDE = "override"     # Complete override
    ADDITIVE = "additive"     # Only add new fields
    SELECTIVE = "selective"   # Merge specific fields only

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
        """
        Compose schemas using specified strategy.

        Args:
            base: Base schema
            extension: Extension schema
            strategy: Composition strategy
            options: Strategy-specific options

        Returns:
            Dict[str, Any]: Composed schema
        """
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
        """
        Apply overrides to schema.

        Args:
            schema: Schema to override
            overrides: List of overrides to apply
            context: Optional context for conditional overrides

        Returns:
            Dict[str, Any]: Schema with overrides applied
        """
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
    """
    Schema information with versioning and metadata.
    """
    name: str
    version: SchemaVersion
    schema: Dict[str, Any]
    base_schemas: List[str]
    overrides: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


@dataclass
class SchemaMigration:
    """Definition of schema migration."""
    from_version: SchemaVersion
    to_version: SchemaVersion
    changes: List[Dict[str, Any]]
    transforms: Dict[str, Callable]
    required_types: List[str]
    backwards_compatible: bool = True

class SchemaMigrationManager:
    """Manages schema migrations and versioning."""

    def __init__(
        self,
        validation_manager: 'ValidationManager', # String literal type annotation
        schema_manager: 'SchemaManager', # String literal type annotation
        logger: Optional[DITALogger] = None
    ):
        self.validation_manager = validation_manager
        self.schema_manager = schema_manager
        self.logger = logger or DITALogger(name=__name__)
        self._migrations: Dict[str, List[SchemaMigration]] = {}
        self._schema_versions: Dict[str, SchemaVersion] = {}

        # Register schema validation
        self.validation_manager.register_validator(
            validation_type="schema",
            name="schema_completeness",
            validator=self.schema_manager.validate_schema_completeness
        )

    def register_migration(
        self,
        schema_name: str,
        migration: SchemaMigration
    ) -> None:
        """Register a new schema migration."""
        if schema_name not in self._migrations:
            self._migrations[schema_name] = []
        self._migrations[schema_name].append(migration)

        # Sort migrations by version
        self._migrations[schema_name].sort(
            key=lambda m: (m.from_version.major, m.from_version.minor)
        )

    def migrate_schema(
        self,
        schema: Dict[str, Any],
        schema_name: str,
        from_version: SchemaVersion,
        to_version: SchemaVersion
    ) -> Dict[str, Any]:
        """
        Migrate schema from one version to another.

        Args:
            schema: Schema to migrate
            schema_name: Name of schema
            from_version: Starting version
            to_version: Target version

        Returns:
            Dict[str, Any]: Migrated schema

        Raises:
            ValueError: If migration path not found or validation fails
        """
        try:
            # Get applicable migrations
            migrations = self._get_migration_path(
                schema_name,
                from_version,
                to_version
            )

            if not migrations:
                raise ValueError(
                    f"No migration path found from {from_version} to {to_version}"
                )

            # Apply migrations sequentially
            current = schema.copy()
            for migration in migrations:
                current = self._apply_migration(current, migration)

                # Validate migration result
                result = self.validation_manager.validate(
                    content=current,
                    validation_type="schema",
                    context={
                        "schema_name": schema_name,
                        "phase": "migration",
                        "from_version": str(from_version),
                        "to_version": str(to_version),
                        "required_types": migration.required_types
                    }
                )

                if not result.is_valid:
                    error_messages = "\n".join(
                        f"- {msg.message} ({msg.path})"
                        for msg in result.messages
                        if msg.severity == ValidationSeverity.ERROR
                    )
                    raise ValueError(
                        f"Schema migration validation failed:\n{error_messages}"
                    )

            return current

        except Exception as e:
            self.logger.error(f"Error migrating schema: {str(e)}")
            raise

    def _get_migration_path(
        self,
        schema_name: str,
        from_version: SchemaVersion,
        to_version: SchemaVersion
    ) -> List[SchemaMigration]:
        """Get sequence of migrations to reach target version."""
        migrations = self._migrations.get(schema_name, [])
        path = []

        current = from_version
        while current != to_version:
            next_migration = None
            for migration in migrations:
                if (migration.from_version == current and
                    self._is_closer_to_target(migration.to_version, current, to_version)):
                    next_migration = migration
            if not next_migration:
                break
            path.append(next_migration)
            current = next_migration.to_version
        return path

    def _is_closer_to_target(
        self,
        version: SchemaVersion,
        current: SchemaVersion,
        target: SchemaVersion
    ) -> bool:
        """Check if version is closer to target than current."""
        return (
            abs(version.major - target.major) < abs(current.major - target.major) or
            (version.major == target.major and
             abs(version.minor - target.minor) < abs(current.minor - target.minor))
        )

    def _apply_migration(
        self,
        schema: Dict[str, Any],
        migration: SchemaMigration
    ) -> Dict[str, Any]:
        """Apply single migration to schema."""
        result = schema.copy()

        for change in migration.changes:
            change_type = change.get("type")
            if not change_type:
                self.logger.warning(f"Missing change type in migration: {change}")
                continue

            if not isinstance(change_type, str):
                self.logger.warning(f"Invalid change type in migration: {change_type}")
                continue

            transform = migration.transforms.get(change_type)
            if transform:
                result = transform(result, change)

        return result

    def check_schema_compatibility(
        self,
        schema_name: str,
        version: str
    ) -> bool:
        """
        Check if schema version is compatible.

        Args:
            schema_name: Name of schema to check
            version: Version string to check against

        Returns:
            bool: True if versions are compatible
        """
        try:
            if schema_name not in self._schema_versions:
                return False

            current_version = self._schema_versions[schema_name]
            check_version = SchemaVersion.from_string(version)

            return current_version.is_compatible_with(check_version)

        except Exception as e:
            self.logger.error(f"Error checking schema compatibility: {str(e)}")
            return False

    def validate_migration(
        self,
        schema: Dict[str, Any],
        from_version: str,
        to_version: str
    ) -> ValidationResult:
        """
        Validate schema migration compatibility.
        This is better placed here since SchemaMigrationManager already has:
        - Access to migration paths
        - Version compatibility checks
        - Migration rules and transforms
        """
        try:
            messages = []

            # Parse versions
            source_version = SchemaVersion.from_string(from_version)
            target_version = SchemaVersion.from_string(to_version)

            # Get schema name from schema dict
            schema_name = schema.get('name')
            if not schema_name:
                return ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path="schema",
                        message="Schema name not found",
                        severity=ValidationSeverity.ERROR,
                        code="missing_schema_name"
                    )]
                )

            # Get migration path using existing method with all required parameters
            migrations = self._get_migration_path(
                schema_name=schema_name,
                from_version=source_version,
                to_version=target_version
            )

            # Validate each migration step
            for migration in migrations:
                # Use existing validation methods
                result = self._validate_migration_step(schema, migration)
                messages.extend(result.messages)

                if not result.is_valid:
                    break

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating schema migration: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Migration validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="migration_error"
                )]
            )

    def _validate_migration_step(
        self,
        schema: Dict[str, Any],
        migration: SchemaMigration
    ) -> ValidationResult:
        """Validate individual migration step."""
        messages = []

        # Validate backwards compatibility if required
        if migration.backwards_compatible:
            result = self._validate_backwards_compatibility(schema, migration)
            messages.extend(result.messages)
            if not result.is_valid:
                return ValidationResult(is_valid=False, messages=messages)

        # Validate required fields preservation
        result = self._validate_required_fields_preservation(schema, migration)
        messages.extend(result.messages)

        # Validate transforms
        result = self._validate_transforms(schema, migration)
        messages.extend(result.messages)

        return ValidationResult(
            is_valid=not any(
                msg.severity == ValidationSeverity.ERROR for msg in messages
            ),
            messages=messages
        )

    def _validate_backwards_compatibility(
            self,
            schema: Dict[str, Any],
            migration: SchemaMigration
        ) -> ValidationResult:
            """Validate backwards compatibility of migration."""
            messages = []

            # Check if all required fields are preserved
            required_fields = self._get_required_fields(schema)
            for field in required_fields:
                if not self._field_preserved(field, migration.changes):
                    messages.append(ValidationMessage(
                        path=field,
                        message="Required field not preserved in migration",
                        severity=ValidationSeverity.ERROR,
                        code="field_not_preserved"
                    ))

            return ValidationResult(
                is_valid=not any(msg.severity == ValidationSeverity.ERROR for msg in messages),
                messages=messages
            )

    def _validate_required_fields_preservation(
        self,
        schema: Dict[str, Any],
        migration: SchemaMigration
    ) -> ValidationResult:
        """Validate preservation of required fields."""
        messages = []

        for field in migration.required_types:
            if not self._field_exists(schema, field):
                messages.append(ValidationMessage(
                    path=field,
                    message=f"Required field missing: {field}",
                    severity=ValidationSeverity.ERROR,
                    code="missing_required_field"
                ))

        return ValidationResult(
            is_valid=not any(msg.severity == ValidationSeverity.ERROR for msg in messages),
            messages=messages
        )

    def _validate_transforms(
        self,
        schema: Dict[str, Any],
        migration: SchemaMigration
    ) -> ValidationResult:
        """Validate migration transforms."""
        messages = []

        for transform_name, transform in migration.transforms.items():
            try:
                # Validate transform can be applied
                if not self._can_apply_transform(transform, schema):
                    messages.append(ValidationMessage(
                        path=transform_name,
                        message=f"Transform cannot be applied: {transform_name}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_transform"
                    ))
            except Exception as e:
                messages.append(ValidationMessage(
                    path=transform_name,
                    message=f"Transform validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="transform_error"
                ))

        return ValidationResult(
            is_valid=not any(msg.severity == ValidationSeverity.ERROR for msg in messages),
            messages=messages
        )

    def _can_apply_transform(
        self,
        transform: Callable,
        schema: Dict[str, Any]
    ) -> bool:
        """Check if transform can be applied to schema."""
        # Basic validation - could be enhanced based on transform requirements
        return callable(transform)

    def _field_exists(self, schema: Dict[str, Any], field_path: str) -> bool:
        """Check if field exists in schema."""
        try:
            parts = field_path.split('.')
            current = schema
            for part in parts:
                current = current[part]
            return True
        except (KeyError, TypeError):
            return False

    def _field_preserved(
        self,
        field: str,
        changes: List[Dict[str, Any]]
    ) -> bool:
        """Check if field is preserved through changes."""
        for change in changes:
            if change.get('type') == 'remove' and change.get('field') == field:
                return False
        return True



class SchemaManager:
    """
    Enhanced schema management with versioning and inheritance.
    """
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
        self.dtd_mapper = dtd_mapper
        self.config_path = Path(config_path)
        self.cache = cache
        self.logger = logger or DITALogger(name=__name__)

        # Schema storage
        self._schemas: Dict[str, SchemaInfo] = {}
        self._inheritance_graph: Dict[str, Set[str]] = {}
        self._override_maps: Dict[str, Dict[str, Any]] = {}
        self.schema_composer = SchemaComposer()
        self.override_manager = OverrideManager()
        self._schema_registry: Dict[str, Dict[str, Any]] = {}
        self._inheritance_map: Dict[str, List[Tuple[str, str]]] = {}
        self._schema_versions: Dict[str, SchemaVersion] = {}
        self._loaded_schemas: Set[str] = set()

        # Manager dependencies
        self.config_manager = config_manager
        self.validation_manager = validation_manager
        self.event_manager = event_manager


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
        self.dtd_composition_config = DTDCompositionConfig(config_manager)


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

    def compose_dtd_schema(
        self,
        base_dtd: Dict[str, Any],
        extension_dtd: Dict[str, Any],
        strategy: CompositionStrategy = CompositionStrategy.MERGE,
        options: Optional[DTDCompositionOptions] = None
    ) -> Dict[str, Any]:
        """
        Compose DTD schemas using composition strategy.

        Args:
            base_dtd: Base DTD schema
            extension_dtd: Extension DTD schema
            strategy: Composition strategy
            options: Optional composition options

        Returns:
            Dict[str, Any]: Composed schema
        """
        try:
            # Use default options if none provided
            options = options or DTDCompositionOptions()

            # Initialize result with base DTD copy
            result = base_dtd.copy()

            # Apply composition based on strategy
            if strategy == CompositionStrategy.MERGE:
                # Merge elements
                if 'elements' in extension_dtd:
                    if options.merge_content_models:
                        result['elements'] = self._merge_dtd_elements(
                            result.get('elements', {}),
                            extension_dtd['elements']
                        )
                    else:
                        # Only merge non-content model properties
                        result['elements'].update(extension_dtd['elements'])

                # Merge attributes if enabled
                if options.merge_attributes and 'attributes' in extension_dtd:
                    result['attributes'] = self._merge_dtd_attributes(
                        result.get('attributes', {}),
                        extension_dtd['attributes']
                    )

                # Handle specializations
                if options.preserve_specializations and 'specializations' in extension_dtd:
                    result['specializations'] = self._merge_dtd_specializations(
                        result.get('specializations', {}),
                        extension_dtd['specializations'],
                        options.specialization_strategy
                    )

            elif strategy == CompositionStrategy.OVERRIDE:
                # Complete override - just validate and use extension
                result = extension_dtd.copy()

            elif strategy == CompositionStrategy.SELECTIVE:
                # Selective merge based on specified fields
                if options and hasattr(options, 'fields'):
                    for field in options.fields:
                        if field in extension_dtd:
                            result[field] = extension_dtd[field]

            # Update metadata
            self._update_dtd_metadata(result, extension_dtd)

            # Validate the composed schema
            validation_result = self.validate_dtd_schema_mapping(
                dtd_path=Path(result.get('source_dtd', '')),
                schema_name=result.get('name', '')
            )

            if not validation_result.is_valid:
                raise ValueError(
                    "DTD composition validation failed: " +
                    "; ".join(msg.message for msg in validation_result.messages)
                )

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
        extension_specs: Dict[str, Any],
        strategy: str
    ) -> Dict[str, Any]:
        """Merge DTD specialization definitions."""
        result = base_specs.copy()

        for name, spec in extension_specs.items():
            if name in result:
                if strategy == "inherit":
                    # Keep base inheritance, add new constraints
                    result[name]['constraints'].update(spec.get('constraints', {}))
                elif strategy == "override":
                    # Use extension specialization
                    result[name] = spec
                elif strategy == "merge":
                    # Merge all aspects
                    existing = result[name]
                    result[name] = {
                        "base_type": spec.get('base_type', existing['base_type']),
                        "inheritance_path": spec.get('inheritance_path', existing['inheritance_path']),
                        "attributes": self._merge_dtd_attributes(
                            existing['attributes'],
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
                result[name] = spec

        return result

    def register_schema(
        self,
        name: str,
        schema: Dict[str, Any],
        version: str,
        base_schemas: Optional[List[str]] = None,
        overrides: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a new schema with inheritance information.

        Args:
            name: Schema name
            schema: Schema definition
            version: Schema version string
            base_schemas: Optional list of schemas to inherit from
            overrides: Optional override rules
            metadata: Optional schema metadata
        """
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

    def get_schema(
        self,
        name: str,
        include_inherited: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get schema with inheritance resolution and caching.

        Args:
            name: Schema name
            include_inherited: Whether to include inherited schemas

        Returns:
            Optional[Dict[str, Any]]: Resolved schema if found
        """
        try:
            # Check cache first
            cache_key = f"schema_{name}_{include_inherited}"
            if cached := self.cache.get(
                key=cache_key,
                entry_type=CacheEntryType.CONTENT
            ):
                return cached

            # Get base schema
            schema_info = self._schemas.get(name)
            if not schema_info:
                return None

            if not include_inherited:
                return schema_info.schema

            # Resolve inherited schema
            resolved = self._resolve_schema_inheritance(name)

            # Cache result
            if resolved:
                self.cache.set(
                    key=cache_key,
                    data=resolved,
                    entry_type=CacheEntryType.CONTENT,
                    element_type=ElementType.UNKNOWN,
                    phase=ProcessingPhase.DISCOVERY
                )

            return resolved

        except Exception as e:
            self.logger.error(f"Error getting schema {name}: {str(e)}")
            return None

    def _resolve_schema_inheritance(
        self,
        schema_name: str,
        visited: Optional[Set[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve schema inheritance with DTD awareness.

        Args:
            schema_name: Schema to resolve
            visited: Set of visited schemas

        Returns:
            Optional[Dict[str, Any]]: Resolved schema
        """
        try:
            if visited is None:
                visited = set()

            # Check for cycles
            if schema_name in visited:
                raise ValueError(f"Circular inheritance detected: {schema_name}")

            visited.add(schema_name)

            # Check resolution depth
            if len(visited) > self._max_inheritance_depth:
                raise ValueError(f"Maximum inheritance depth exceeded for {schema_name}")

            # Get schema info
            schema_info = self._schemas.get(schema_name)
            if not schema_info:
                # Try DTD schemas if not in regular schemas
                if dtd_schema := self._dtd_schemas.get(schema_name):
                    return self._resolve_dtd_inheritance(schema_name, visited)
                return None

            # Start with base schema
            resolved = schema_info.schema.copy()

            # Process base schemas
            for base_name in schema_info.base_schemas:
                base_schema = self._resolve_schema_inheritance(base_name, visited)
                if base_schema:
                    # Check if base is DTD-derived
                    if base_name in self._dtd_schemas:
                        # Use DTD-aware composition
                        resolved = self.compose_dtd_schema(
                            base_schema,
                            resolved,
                            strategy=CompositionStrategy.MERGE
                        )
                    else:
                        # Use regular composition
                        resolved = self.schema_composer.compose(
                            base=base_schema,
                            extension=resolved,
                            strategy=CompositionStrategy.MERGE
                        )

            # Apply overrides
            if schema_info.overrides:
                override_objects = [
                    Override(
                        type=OverrideType.VALUE,
                        path=path,
                        value=value
                    )
                    for path, value in schema_info.overrides.items()
                ]
                resolved = self.override_manager.apply_overrides(
                    schema=resolved,
                    overrides=override_objects
                )

            return resolved

        except Exception as e:
            self.logger.error(f"Error resolving schema inheritance for {schema_name}: {str(e)}")
            return None



    def get_element_schema(
        self,
        element_type: ElementType,
        processing_type: ProcessingRuleType
    ) -> Optional[Dict[str, Any]]:
        """
        Get element-specific schema with inheritance.

        Args:
            element_type: Type of element
            processing_type: Type of processing

        Returns:
            Optional[Dict[str, Any]]: Element schema if found
        """
        try:
            # Get processing schema
            if processing_schema := self.get_schema("processing"):
                # Get rule type section
                if rule_section := processing_schema.get("rules", {}).get(processing_type.value):
                    # Look for element-specific rules
                    return rule_section.get(element_type.value)

            return None

        except Exception as e:
            self.logger.error(f"Error getting element schema: {str(e)}")
            return None



    def get_dtd_schema(
        self,
        dtd_name: str,
        include_inherited: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Get schema derived from DTD."""
        try:
            if dtd_schema := self._dtd_schemas.get(dtd_name):
                if not include_inherited:
                    return dtd_schema['schema']

                # Include inherited schemas
                return self._resolve_dtd_inheritance(dtd_name)
            return None

        except Exception as e:
            self.logger.error(f"Error getting DTD schema: {str(e)}")
            return None

    def _resolve_dtd_inheritance(
        self,
        dtd_name: str,
        visited: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """Resolve DTD inheritance chain."""
        if visited is None:
            visited = set()

        if dtd_name in visited:
            raise ValueError(f"Circular DTD inheritance detected: {dtd_name}")

        visited.add(dtd_name)

        # Get base schema
        base_schema = self._dtd_schemas[dtd_name]['schema'].copy()

        # Process inheritance chain
        for parent in self._dtd_inheritance.get(dtd_name, []):
            if parent_schema := self._resolve_dtd_inheritance(parent, visited):
                base_schema = self.schema_composer.compose(
                    base=parent_schema,
                    extension=base_schema,
                    strategy=CompositionStrategy.MERGE
                )

        return base_schema




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



    def _merge_dtd_attributes(
        self,
        base_attrs: Dict[str, Any],
        extension_attrs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge DTD attribute definitions.

        Args:
            base_attrs: Base attributes
            extension_attrs: Extension attributes

        Returns:
            Dict[str, Any]: Merged attributes
        """
        result = base_attrs.copy()

        try:
            for attr_name, ext_attr in extension_attrs.items():
                if attr_name in result:
                    # Merge existing attribute
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

        except Exception as e:
            self.logger.error(f"Error merging DTD attributes: {str(e)}")
            raise


    def validate_specialization_inheritance(
        self,
        base_type: str,
        specialized_type: str
    ) -> ValidationResult:
        """
        Validate specialization inheritance rules.

        Args:
            base_type: Base type name
            specialized_type: Specialized type name

        Returns:
            ValidationResult: Validation status and messages
        """
        try:
            messages = []

            # Get inheritance chain
            inheritance_chain = self._get_inheritance_chain(specialized_type)
            if not inheritance_chain or base_type not in inheritance_chain:
                return ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path=specialized_type,
                        message=f"Invalid specialization chain for {specialized_type}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_specialization"
                    )]
                )

            # Get schemas
            base_schema = self.get_schema(base_type)
            specialized_schema = self.get_schema(specialized_type)

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

            # Validate attribute inheritance
            messages.extend(self._validate_attribute_inheritance(
                base_schema,
                specialized_schema
            ))

            # Validate content model compatibility
            messages.extend(self._validate_content_model_inheritance(
                base_schema,
                specialized_schema
            ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating specialization inheritance: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Specialization validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="specialization_error"
                )]
            )


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

            # Validate version compatibility
            if not self._validate_version_compatibility(version, schema):
                messages.append(ValidationMessage(
                    path="version",
                    message="Version incompatible with schema content",
                    severity=ValidationSeverity.ERROR,
                    code="version_mismatch"
                ))

            # Check minimum version requirements
            if not self._meets_minimum_version(version):
                messages.append(ValidationMessage(
                    path="version",
                    message="Schema version below minimum required version",
                    severity=ValidationSeverity.ERROR,
                    code="version_too_low"
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
                    code="version_error"
                )]
            )



    def _get_inheritance_chain(
        self,
        element_name: str
    ) -> List[str]:
        """Get inheritance chain for element."""
        try:
            # Use DTDMapper for inheritance resolution
            if hasattr(self, 'dtd_mapper'):
                spec_info = self.dtd_mapper.get_specialization_info(element_name)
                if spec_info:
                    return spec_info.inheritance_path
            return []
        except Exception as e:
            self.logger.error(f"Error getting inheritance chain: {str(e)}")
            return []

    def _validate_attribute_inheritance(
        self,
        base_schema: Dict[str, Any],
        specialized_schema: Dict[str, Any]
    ) -> List[ValidationMessage]:
        """Validate attribute inheritance in specialization."""
        messages = []

        try:
            base_attrs = base_schema.get('attributes', {})
            spec_attrs = specialized_schema.get('attributes', {})

            # Check all required base attributes are preserved
            for attr_name, base_attr in base_attrs.items():
                if base_attr.get('required', False):
                    if attr_name not in spec_attrs:
                        messages.append(ValidationMessage(
                            path=f"attributes.{attr_name}",
                            message=f"Required base attribute not preserved: {attr_name}",
                            severity=ValidationSeverity.ERROR,
                            code="missing_base_attribute"
                        ))
                    else:
                        # Validate attribute type compatibility
                        spec_attr = spec_attrs[attr_name]
                        if not self._are_types_compatible(
                            base_attr.get('type', ''),
                            spec_attr.get('type', '')
                        ):
                            messages.append(ValidationMessage(
                                path=f"attributes.{attr_name}",
                                message="Incompatible attribute type in specialization",
                                severity=ValidationSeverity.ERROR,
                                code="incompatible_attribute_type"
                            ))

            return messages

        except Exception as e:
            self.logger.error(f"Error validating attribute inheritance: {str(e)}")
            return [ValidationMessage(
                path="attributes",
                message=f"Attribute inheritance error: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="inheritance_error"
            )]

    def _validate_content_model_inheritance(
        self,
        base_schema: Dict[str, Any],
        specialized_schema: Dict[str, Any]
    ) -> List[ValidationMessage]:
        """Validate content model inheritance in specialization."""
        messages = []

        try:
            base_content = base_schema.get('content_model', {})
            spec_content = specialized_schema.get('content_model', {})

            # Check content model type compatibility
            if base_content.get('type') != spec_content.get('type'):
                messages.append(ValidationMessage(
                    path="content_model",
                    message="Content model type mismatch in specialization",
                    severity=ValidationSeverity.ERROR,
                    code="content_model_mismatch"
                ))

            # Check element inheritance
            base_elements = set(base_content.get('elements', []))
            spec_elements = set(spec_content.get('elements', []))

            # All base elements must be present in specialization
            missing_elements = base_elements - spec_elements
            if missing_elements:
                messages.append(ValidationMessage(
                    path="content_model.elements",
                    message=f"Missing base elements in specialization: {missing_elements}",
                    severity=ValidationSeverity.ERROR,
                    code="missing_base_elements"
                ))

            return messages

        except Exception as e:
            self.logger.error(f"Error validating content model inheritance: {str(e)}")
            return [ValidationMessage(
                path="content_model",
                message=f"Content model inheritance error: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="inheritance_error"
            )]


    def _validate_version_compatibility(
        self,
        version: SchemaVersion,
        schema: Dict[str, Any]
    ) -> bool:
        """Validate version compatibility with schema content."""
        min_version = self._get_minimum_version_for_features(schema)
        return version >= min_version

    def _meets_minimum_version(self, version: SchemaVersion) -> bool:
        """Check if version meets minimum requirements."""
        min_version = SchemaVersion(1, 0, 0)  # Define minimum supported version
        return version >= min_version

    def _get_minimum_version_for_features(
        self,
        schema: Dict[str, Any]
    ) -> SchemaVersion:
        """Determine minimum version required for schema features."""
        # This could be enhanced based on feature requirements
        return SchemaVersion(1, 0, 0)
