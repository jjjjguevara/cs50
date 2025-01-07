from typing import Dict, List, Optional, Any, Set, Union, Type, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import json

# Core managers
from ..validation_manager import ValidationManager
from ..event_manager import EventManager, EventType
from ..config.config_manager import ConfigManager
from ..metadata.metadata_manager import MetadataManager
from ..key_manager import KeyManager

# Schema components
from .schema_manager import SchemaVersion, SchemaManager
from .schema_validator import SchemaValidator
from .schema_resolver import SchemaResolver
from .schema_compatibility import SchemaCompatibilityChecker

# Utils
from ..utils.cache import ContentCache, CacheEntryType
from ..utils.logger import DITALogger

# Models and types
from ..models.types import (
    ValidationResult,
    ValidationMessage,
    ValidationSeverity,
    ElementType,
    ProcessingPhase,
    ProcessingContext,
    ProcessingState
)

@dataclass
class MigrationStep:
    """Individual migration step definition."""
    from_version: SchemaVersion
    to_version: SchemaVersion
    description: str
    changes: List[Dict[str, Any]]
    transform: Callable[[Dict[str, Any]], Dict[str, Any]]
    required_types: List[str] = field(default_factory=list)
    backwards_compatible: bool = True
    validation_rules: Optional[Dict[str, Any]] = None

@dataclass
class MigrationPath:
    """Complete migration path between versions."""
    start_version: SchemaVersion
    target_version: SchemaVersion
    steps: List[MigrationStep]
    metadata: Dict[str, Any] = field(default_factory=dict)

class MigrationType(Enum):
    """Types of schema migrations."""
    UPGRADE = "upgrade"
    DOWNGRADE = "downgrade"
    ROLLBACK = "rollback"
    FEATURE = "feature"

@dataclass
class MigrationResult:
    """Result of schema migration."""
    success: bool
    version: SchemaVersion
    changes: List[Dict[str, Any]]
    validation_result: ValidationResult
    metadata: Dict[str, Any] = field(default_factory=dict)


class SchemaMigrator:
    """Handles schema versioning and migration."""

    def __init__(
        self,
        validation_manager: ValidationManager,
        event_manager: EventManager,
        config_manager: ConfigManager,
        cache: ContentCache,
        schema_validator: SchemaValidator,
        schema_resolver: SchemaResolver,
        compatibility_checker: SchemaCompatibilityChecker,
        logger: Optional[DITALogger] = None
    ):
        """Initialize schema migrator.

        Args:
            validation_manager: System validation manager
            event_manager: System event manager
            config_manager: System configuration manager
            cache: Cache system
            schema_validator: Schema validation component
            schema_resolver: Schema resolution component
            compatibility_checker: Schema compatibility component
            logger: Optional logger instance
        """
        # Core dependencies
        self.validation_manager = validation_manager
        self.event_manager = event_manager
        self.config_manager = config_manager
        self.cache = cache
        self.logger = logger or DITALogger(name=__name__)

        # Schema components
        self.schema_validator = schema_validator
        self.schema_resolver = schema_resolver
        self.compatibility_checker = compatibility_checker

        # Migration tracking
        self._migrations: Dict[str, List[MigrationStep]] = {}
        self._pending_migrations: Dict[str, MigrationPath] = {}

        # Register validation handlers
        self._register_validators()

    def _register_validators(self) -> None:
        """Register migration validators with validation manager."""
        try:
            # Register migration structure validator
            self.validation_manager.register_validator(
                validation_type="migration",
                name="structure",
                validator=self.validate_migration_structure
            )

            # Register migration path validator
            self.validation_manager.register_validator(
                validation_type="migration",
                name="path",
                validator=self.validate_migration_path
            )

        except Exception as e:
            self.logger.error(f"Error registering migration validators: {str(e)}")
            raise

    def register_migration(
            self,
            schema_name: str,
            migration: MigrationStep
        ) -> ValidationResult:
            """
            Register a new schema migration step.

            Args:
                schema_name: Name of schema to migrate
                migration: Migration step definition

            Returns:
                ValidationResult: Migration registration validation
            """
            try:
                # Validate migration structure
                validation_result = self.validate_migration_structure(migration)
                if not validation_result.is_valid:
                    return validation_result

                # Store migration
                if schema_name not in self._migrations:
                    self._migrations[schema_name] = []
                self._migrations[schema_name].append(migration)

                # Sort migrations by version
                self._migrations[schema_name].sort(
                    key=lambda m: (m.from_version.major, m.from_version.minor)
                )

                # Clear migration cache
                self.cache.invalidate_by_pattern(f"migration_{schema_name}_*")

                # Emit migration registered event
                self.event_manager.emit(
                    EventType.STATE_CHANGE,
                    element_id=f"migration_{schema_name}",
                    state_info={
                        "phase": ProcessingPhase.DISCOVERY,
                        "state": ProcessingState.COMPLETED,
                        "element_id": f"migration_{schema_name}"
                    }
                )

                return ValidationResult(is_valid=True, messages=[])

            except Exception as e:
                self.logger.error(f"Error registering migration: {str(e)}")
                return ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path="",
                        message=f"Migration registration error: {str(e)}",
                        severity=ValidationSeverity.ERROR,
                        code="registration_error"
                    )]
                )

    def migrate_schema(
        self,
        schema: Dict[str, Any],
        schema_name: str,
        from_version: Union[str, SchemaVersion],
        to_version: Union[str, SchemaVersion],
        migration_type: MigrationType = MigrationType.UPGRADE
    ) -> MigrationResult:
        """
        Migrate schema from one version to another.

        Args:
            schema: Schema to migrate
            schema_name: Schema name
            from_version: Starting version
            to_version: Target version
            migration_type: Type of migration

        Returns:
            MigrationResult: Migration result with validation
        """
        # Initialize version variables outside try block
        start_version = (
            SchemaVersion.from_string(from_version)
            if isinstance(from_version, str)
            else from_version
        )
        target_version = (
            SchemaVersion.from_string(to_version)
            if isinstance(to_version, str)
            else to_version
        )

        try:
            # Get migration path
            migration_path = self._get_migration_path(
                schema_name,
                start_version,
                target_version,
                migration_type
            )

            if not migration_path:
                return MigrationResult(
                    success=False,
                    version=start_version,
                    changes=[],
                    validation_result=ValidationResult(
                        is_valid=False,
                        messages=[ValidationMessage(
                            path="",
                            message=f"No migration path found from {start_version} to {target_version}",
                            severity=ValidationSeverity.ERROR,
                            code="no_migration_path"
                        )]
                    )
                )

            # Apply migrations sequentially
            current_schema = schema.copy()
            applied_changes = []

            for step in migration_path.steps:
                # Apply migration step
                try:
                    current_schema = step.transform(current_schema)
                    applied_changes.extend(step.changes)

                    # Validate after each step
                    validation_result = self.schema_validator.validate_schema_structure(
                        current_schema,
                        step.required_types
                    )

                    if not validation_result.is_valid:
                        return MigrationResult(
                            success=False,
                            version=step.from_version,
                            changes=applied_changes,
                            validation_result=validation_result
                        )

                except Exception as step_error:
                    self.logger.error(f"Error applying migration step: {str(step_error)}")
                    return MigrationResult(
                        success=False,
                        version=step.from_version,
                        changes=applied_changes,
                        validation_result=ValidationResult(
                            is_valid=False,
                            messages=[ValidationMessage(
                                path="",
                                message=f"Migration step error: {str(step_error)}",
                                severity=ValidationSeverity.ERROR,
                                code="step_error"
                            )]
                        )
                    )

            # Final validation
            final_validation = self.validate_migration_result(
                current_schema,
                migration_path,
                applied_changes
            )

            # Always return a MigrationResult
            return MigrationResult(
                success=final_validation.is_valid,
                version=target_version,
                changes=applied_changes,
                validation_result=final_validation,
                metadata={
                    "migration_type": migration_type.value,
                    "completed_at": datetime.now().isoformat()
                }
            )

        except Exception as e:
            self.logger.error(f"Error migrating schema: {str(e)}")
            return MigrationResult(
                success=False,
                version=start_version,
                changes=[],
                validation_result=ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path="",
                        message=f"Migration error: {str(e)}",
                        severity=ValidationSeverity.ERROR,
                        code="migration_error"
                    )]
                )
            )
    def _get_migration_path(
        self,
        schema_name: str,
        from_version: SchemaVersion,
        to_version: SchemaVersion,
        migration_type: MigrationType
    ) -> Optional[MigrationPath]:
        """Get optimal migration path between versions."""
        try:
            # Check cache first
            cache_key = f"migration_path_{schema_name}_{from_version}_{to_version}_{migration_type.value}"
            if cached := self.cache.get(cache_key, CacheEntryType.CONFIG):
                return cached

            migrations = self._migrations.get(schema_name, [])
            if not migrations:
                return None

            # Build path based on migration type
            if migration_type == MigrationType.UPGRADE:
                steps = self._build_upgrade_path(migrations, from_version, to_version)
            elif migration_type == MigrationType.DOWNGRADE:
                steps = self._build_downgrade_path(migrations, from_version, to_version)
            else:
                steps = self._build_feature_path(migrations, from_version, to_version)

            if not steps:
                return None

            path = MigrationPath(
                start_version=from_version,
                target_version=to_version,
                steps=steps
            )

            # Cache path
            self.cache.set(
                key=cache_key,
                data=path,
                entry_type=CacheEntryType.CONFIG,
                element_type=ElementType.UNKNOWN,
                phase=ProcessingPhase.DISCOVERY
            )

            return path

        except Exception as e:
            self.logger.error(f"Error getting migration path: {str(e)}")
            return None

    def _build_upgrade_path(
        self,
        migrations: List[MigrationStep],
        from_version: SchemaVersion,
        to_version: SchemaVersion
    ) -> List[MigrationStep]:
        """Build upgrade migration path."""
        path = []
        current = from_version

        while current < to_version:
            next_step = None
            for migration in migrations:
                if (migration.from_version == current and
                    migration.to_version <= to_version):
                    if not next_step or migration.to_version > next_step.to_version:
                        next_step = migration
            if not next_step:
                break
            path.append(next_step)
            current = next_step.to_version

        return path

    def _build_downgrade_path(
        self,
        migrations: List[MigrationStep],
        from_version: SchemaVersion,
        to_version: SchemaVersion
    ) -> List[MigrationStep]:
        """Build downgrade migration path."""
        path = []
        current = from_version

        while current > to_version:
            prev_step = None
            for migration in migrations:
                if (migration.to_version == current and
                    migration.from_version >= to_version):
                    if not prev_step or migration.from_version < prev_step.from_version:
                        prev_step = migration
            if not prev_step:
                break
            path.append(prev_step)
            current = prev_step.from_version

        return path

    def _build_feature_path(
        self,
        migrations: List[MigrationStep],
        from_version: SchemaVersion,
        to_version: SchemaVersion
    ) -> List[MigrationStep]:
        """Build feature-specific migration path."""
        # Feature paths can include both upgrades and downgrades
        path = []
        current = from_version

        while current != to_version:
            next_step = None
            for migration in migrations:
                if migration.from_version == current:
                    if not next_step or self._is_better_feature_step(
                        migration, next_step, to_version
                    ):
                        next_step = migration
            if not next_step:
                break
            path.append(next_step)
            current = next_step.to_version

        return path

    def _is_better_feature_step(
        self,
        step: MigrationStep,
        current_best: MigrationStep,
        target: SchemaVersion
    ) -> bool:
        """Determine if a step is better for feature migration."""
        return (
            abs(step.to_version.major - target.major) <
            abs(current_best.to_version.major - target.major)
        )

    def validate_migration_structure(
        self,
        migration: MigrationStep
    ) -> ValidationResult:
        """Validate migration step structure."""
        try:
            messages = []

            # Validate version ordering
            if migration.to_version < migration.from_version:
                messages.append(ValidationMessage(
                    path="versions",
                    message="Target version cannot be lower than source version",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_version_order"
                ))

            # Validate changes list
            if not migration.changes:
                messages.append(ValidationMessage(
                    path="changes",
                    message="Migration must include changes",
                    severity=ValidationSeverity.ERROR,
                    code="no_changes"
                ))

            # Validate transform callable
            if not callable(migration.transform):
                messages.append(ValidationMessage(
                    path="transform",
                    message="Transform must be callable",
                    severity=ValidationSeverity.ERROR,
                    code="invalid_transform"
                ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating migration structure: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Migration validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def validate_migration_path(
        self,
        path: MigrationPath
    ) -> ValidationResult:
        """Validate complete migration path."""
        try:
            messages = []

            # Validate step sequence
            current_version = path.start_version
            for step in path.steps:
                if step.from_version != current_version:
                    messages.append(ValidationMessage(
                        path="steps",
                        message=f"Invalid step sequence at version {step.from_version}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_sequence"
                    ))
                current_version = step.to_version

            # Validate final version
            if current_version != path.target_version:
                messages.append(ValidationMessage(
                    path="target_version",
                    message="Migration path does not reach target version",
                    severity=ValidationSeverity.ERROR,
                    code="incomplete_path"
                ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating migration path: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Path validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def validate_migration_result(
        self,
        schema: Dict[str, Any],
        path: MigrationPath,
        changes: List[Dict[str, Any]]
    ) -> ValidationResult:
        """Validate migrated schema."""
        try:
            messages = []

            # Validate schema structure
            structure_result = self.schema_validator.validate_schema_structure(schema)
            messages.extend(structure_result.messages)

            # Validate version
            version_result = self.schema_validator.validate_schema_version(schema)
            messages.extend(version_result.messages)

            # Check compatibility if backwards compatible
            if all(step.backwards_compatible for step in path.steps):
                compat_result = self.compatibility_checker.check_version_compatibility(
                    str(path.start_version),
                    str(path.target_version)
                )
                if not compat_result:
                    messages.append(ValidationMessage(
                        path="compatibility",
                        message="Migration breaks backwards compatibility",
                        severity=ValidationSeverity.ERROR,
                        code="compatibility_error"
                    ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating migration result: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="",
                    message=f"Result validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def cleanup(self) -> None:
        """Clean up migrator resources."""
        try:
            self._migrations.clear()
            self._pending_migrations.clear()
            self.cache.invalidate_by_pattern("migration_*")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise
