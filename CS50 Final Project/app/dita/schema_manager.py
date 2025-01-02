"""Schema management and inheritance for DITA processing."""

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .validation_manager import ValidationManager

from typing import Dict, Any, Optional, Any, List, Union, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import re
from datetime import datetime
import logging


from .utils.cache import ContentCache, CacheEntryType
from .utils.logger import DITALogger
from .models.types import (
    ElementType,
    ProcessingPhase,
    ProcessingRuleType,
    ContentScope
)

from enum import Enum, auto
from typing import Protocol, Dict, Any, Optional, List, Set, Callable

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
    """
    Schema version information.
    Handles semantic versioning for schemas.
    """
    major: int
    minor: int
    patch: int
    metadata: Optional[str] = None

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
    backwards_compatible: bool = True

class SchemaMigrationManager:
    """Manages schema migrations and versioning."""

    def __init__(
        self,
        validation_manager: 'ValidationManager', # String literal type annotation
        logger: Optional[DITALogger] = None
    ):
        self.validation_manager = validation_manager
        self.logger = logger or DITALogger(name=__name__)
        self._migrations: Dict[str, List[SchemaMigration]] = {}

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

                # Validate after each migration
                result = self.validation_manager.validate_schema_completeness(
                    current,
                    []  # Add required types if needed
                )
                if not result.is_valid:
                    raise ValueError(
                        f"Migration validation failed: {result.messages}"
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
                    self._is_closer_to_target(
                        migration.to_version,
                        current,
                        to_version
                    )):
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



class SchemaManager:
    """
    Enhanced schema management with versioning and inheritance.
    """
    def __init__(
        self,
        config_path: Path,
        cache: ContentCache,
        logger: Optional[DITALogger] = None
    ):
        self.config_path = Path(config_path)
        self.cache = cache
        self.logger = logger or DITALogger(name=__name__)

        # Schema storage
        self._schemas: Dict[str, SchemaInfo] = {}
        self._inheritance_graph: Dict[str, Set[str]] = {}
        self._override_maps: Dict[str, Dict[str, Any]] = {}

        # Resolution tracking
        self._resolution_stack: List[str] = []
        self._resolution_cache: Dict[str, Dict[str, Any]] = {}

        # Schema storage
        self._schema_registry: Dict[str, Dict[str, Any]] = {}
        self._inheritance_map: Dict[str, List[Tuple[str, str]]] = {}
        self._schema_versions: Dict[str, SchemaVersion] = {}
        self._loaded_schemas: Set[str] = set()

        # Track inheritance depth to prevent circular references
        self._inheritance_depth = 0
        self._max_inheritance_depth = 10


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
        Resolve schema inheritance with cycle detection.

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
                return None

            # Start with base schema
            resolved = schema_info.schema.copy()

            # Process base schemas
            for base_name in schema_info.base_schemas:
                base_schema = self._resolve_schema_inheritance(base_name, visited)
                if base_schema:
                    resolved = self._compose_schemas(base_schema, resolved)

            # Apply overrides
            if schema_info.overrides:
                resolved = self._apply_overrides(resolved, schema_info.overrides)

            return resolved

        except Exception as e:
            self.logger.error(f"Error resolving schema inheritance for {schema_name}: {str(e)}")
            return None

    def _compose_schemas(
        self,
        base: Dict[str, Any],
        extension: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compose two schemas with proper override behavior.

        Args:
            base: Base schema
            extension: Extension schema

        Returns:
            Dict[str, Any]: Composed schema
        """
        result = base.copy()

        for key, value in extension.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._compose_schemas(result[key], value)
            else:
                result[key] = value

        return result

    def _apply_overrides(
        self,
        schema: Dict[str, Any],
        overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply overrides to schema.

        Args:
            schema: Schema to override
            overrides: Override rules

        Returns:
            Dict[str, Any]: Schema with overrides applied
        """
        result = schema.copy()

        for path, value in overrides.items():
            parts = path.split('.')
            current = result

            # Navigate to the override location
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Apply the override
            current[parts[-1]] = value

        return result

    def _merge_schemas(
        self,
        parent: Dict[str, Any],
        child: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge parent and child schemas.

        Args:
            parent: Parent schema
            child: Child schema

        Returns:
            Dict[str, Any]: Merged schema
        """
        try:
            merged = parent.copy()

            for key, value in child.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = self._merge_schemas(merged[key], value)
                else:
                    merged[key] = value

            return merged

        except Exception as e:
            self.logger.error(f"Error merging schemas: {str(e)}")
            return child

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
