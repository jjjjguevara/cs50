"""Feature flag management and resolution for DITA processing."""
from typing import Dict, Optional, Any, List, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from ...models.types import (
    Feature,
    FeatureScope,
    ValidationResult,
    ValidationMessage,
    ValidationSeverity,
    ProcessingContext
)
from ...event_manager import EventManager, EventType
from ...utils.cache import ContentCache, CacheEntryType
from ...utils.logger import DITALogger

class FeatureError(Exception):
    """Custom exception for feature-related errors."""
    pass

@dataclass
class FeatureState:
    """Track feature state."""
    enabled: bool
    scope: FeatureScope
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class FeatureManager:
    """
    Manages feature flags with dependency resolution and inheritance.
    Handles feature state management and scope-based resolution.
    """

    def __init__(
        self,
        event_manager: EventManager,
        cache: ContentCache,
        logger: Optional[DITALogger] = None
    ):
        """Initialize feature manager."""
        self.event_manager = event_manager
        self.cache = cache
        self.logger = logger or DITALogger(name=__name__)

        # Feature storage
        self._feature_registry: Dict[str, Feature] = {}
        self._feature_states: Dict[str, FeatureState] = {}
        self._feature_dependencies: Dict[str, Set[str]] = {}
        self._feature_conflicts: Dict[str, Set[str]] = {}

        # Scope tracking
        self._scope_features: Dict[FeatureScope, Set[str]] = {
            scope: set() for scope in FeatureScope
        }

        # Register event handlers
        self._register_event_handlers()

    def _register_event_handlers(self) -> None:
        """Register for relevant events."""
        self.event_manager.subscribe(
            EventType.STATE_CHANGE,
            self._handle_state_change
        )
        self.event_manager.subscribe(
            EventType.FEATURE_UPDATED,
            self._handle_feature_update
        )

    def register_feature(
        self,
        name: str,
        scope: FeatureScope,
        default: bool = False,
        description: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        conflicts: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Register a new feature with dependency tracking."""
        try:
            # Validate feature doesn't exist
            if name in self._feature_registry:
                return ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path=name,
                        message=f"Feature already exists: {name}",
                        severity=ValidationSeverity.ERROR,
                        code="duplicate_feature"
                    )]
                )

            # Create feature
            feature = Feature(
                name=name,
                scope=scope,
                default=default,
                description=description,
                dependencies=dependencies or [],
                conflicts=conflicts or [],
                metadata=metadata or {}
            )

            # Store feature
            self._feature_registry[name] = feature
            self._scope_features[scope].add(name)

            # Initialize state
            self._feature_states[name] = FeatureState(
                enabled=default,
                scope=scope
            )

            # Track dependencies and conflicts
            if dependencies:
                self._feature_dependencies[name] = set(dependencies)
            if conflicts:
                self._feature_conflicts[name] = set(conflicts)

            # Emit event
            self.event_manager.emit(
                EventType.FEATURE_UPDATED,
                feature_name=name,
                new_state=default
            )

            return ValidationResult(is_valid=True, messages=[])

        except Exception as e:
            self.logger.error(f"Error registering feature {name}: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path=name,
                    message=str(e),
                    severity=ValidationSeverity.ERROR,
                    code="registration_error"
                )]
            )

    def update_feature(
        self,
        name: str,
        enabled: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Update feature state with dependency validation."""
        try:
            # Validate feature exists
            if name not in self._feature_registry:
                return ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path=name,
                        message=f"Unknown feature: {name}",
                        severity=ValidationSeverity.ERROR,
                        code="unknown_feature"
                    )]
                )

            # Check dependencies if enabling
            if enabled:
                validation_result = self._validate_dependencies(name)
                if not validation_result.is_valid:
                    return validation_result

                # Check conflicts
                validation_result = self._validate_conflicts(name)
                if not validation_result.is_valid:
                    return validation_result

            # Update state
            self._feature_states[name].enabled = enabled
            self._feature_states[name].timestamp = datetime.now()
            if metadata:
                self._feature_states[name].metadata.update(metadata)

            # Emit event
            self.event_manager.emit(
                EventType.FEATURE_UPDATED,
                feature_name=name,
                new_state=enabled
            )

            return ValidationResult(is_valid=True, messages=[])

        except Exception as e:
            self.logger.error(f"Error updating feature {name}: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path=name,
                    message=str(e),
                    severity=ValidationSeverity.ERROR,
                    code="update_error"
                )]
            )

    def get_feature_state(
        self,
        name: str,
        context: Optional[ProcessingContext] = None
    ) -> bool:
        """Get current feature state with context awareness."""
        try:
            # Get base state
            state = self._feature_states.get(name)
            if not state:
                return False

            # Apply context overrides if provided
            if context:
                # Create cache key after we know we have a valid state
                cache_key = f"feature_{name}_{context.context_id}"

                # Check cache
                if cached := self.cache.get(cache_key, CacheEntryType.FEATURE):
                    return cached

                # Resolve state
                state_value = self._resolve_context_state(name, state, context)

                # Cache resolved state with known cache_key
                self.cache.set(
                    key=cache_key,
                    data=state_value,
                    entry_type=CacheEntryType.FEATURE,
                    element_type=context.element_type,
                    phase=context.state_info.phase
                )
                return state_value

            return state.enabled

        except Exception as e:
            self.logger.error(f"Error getting feature state for {name}: {str(e)}")
            return False

    def _resolve_context_state(
        self,
        name: str,
        state: FeatureState,
        context: ProcessingContext
    ) -> bool:
        """Resolve feature state based on context."""
        try:
            # Get feature
            feature = self._feature_registry[name]

            # Check context scope
            if feature.scope == FeatureScope.GLOBAL:
                return state.enabled

            # Check context-specific overrides
            scope_features = context.features.get(feature.scope.value, {})
            if isinstance(scope_features, dict) and name in scope_features:
                return bool(scope_features[name])  # Ensure boolean return

            return state.enabled

        except Exception as e:
            self.logger.error(f"Error resolving context state for {name}: {str(e)}")
            return state.enabled

    def _validate_dependencies(self, name: str) -> ValidationResult:
        """Validate feature dependencies."""
        if deps := self._feature_dependencies.get(name):
            messages = []
            for dep in deps:
                if not self._feature_states.get(dep, FeatureState(False, FeatureScope.GLOBAL)).enabled:
                    messages.append(ValidationMessage(
                        path=name,
                        message=f"Dependency not enabled: {dep}",
                        severity=ValidationSeverity.ERROR,
                        code="dependency_not_enabled"
                    ))
            return ValidationResult(
                is_valid=len(messages) == 0,
                messages=messages
            )
        return ValidationResult(is_valid=True, messages=[])

    def _validate_conflicts(self, name: str) -> ValidationResult:
        """Validate feature conflicts."""
        if conflicts := self._feature_conflicts.get(name):
            messages = []
            for conflict in conflicts:
                if self._feature_states.get(conflict, FeatureState(False, FeatureScope.GLOBAL)).enabled:
                    messages.append(ValidationMessage(
                        path=name,
                        message=f"Conflicts with enabled feature: {conflict}",
                        severity=ValidationSeverity.ERROR,
                        code="feature_conflict"
                    ))
            return ValidationResult(
                is_valid=len(messages) == 0,
                messages=messages
            )
        return ValidationResult(is_valid=True, messages=[])

    def _handle_state_change(self, **event_data: Any) -> None:
        """Handle state change events."""
        if feature_name := event_data.get("feature_name"):
            self.cache.invalidate_by_pattern(f"feature_{feature_name}_*")

    def _handle_feature_update(self, **event_data: Any) -> None:
        """Handle feature update events."""
        if feature_name := event_data.get("feature_name"):
            self.cache.invalidate_by_pattern(f"feature_{feature_name}_*")

    def cleanup(self) -> None:
        """Clean up manager resources."""
        try:
            self._feature_registry.clear()
            self._feature_states.clear()
            self._feature_dependencies.clear()
            self._feature_conflicts.clear()
            self._scope_features = {scope: set() for scope in FeatureScope}
            self.cache.invalidate_by_pattern("feature_*")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise
