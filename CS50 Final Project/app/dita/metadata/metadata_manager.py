"""Metadata management orchestration for DITA content processing."""
from typing import Dict, Optional, Any, Generator, List, Union, TYPE_CHECKING, Type
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
import logging
from datetime import datetime
import importlib
import re

from dita.dtd.dtd_models import(
    ValidationState,
    ValidationSeverity,
    ValidationMessage
)

# Type checking imports
if TYPE_CHECKING:
    from ..config.config_manager import ConfigManager
    from ..context_manager import ContextManager
    from ..key_manager import KeyManager

# Direct imports
from .extractor import MetadataExtractor
from .storage import MetadataStorage
from ..event_manager import EventManager, EventType
from ..utils.cache import ContentCache, CacheEntryType
from ..utils.logger import DITALogger

# Type imports
from ..models.types import (
    ContentElement,
    ProcessingContext,
    ProcessingPhase,
    ProcessingState,
    ElementType,
    ContentScope,
    MetadataTransaction,
    ValidationResult,
    MetadataState
)


class MetadataManager:
    """
    Orchestrates metadata operations throughout the processing pipeline.
    Coordinates between storage, extraction, and key resolution systems.
    """
    def __init__(
        self,
        db_path: Union[str, Path],
        cache: ContentCache,
        event_manager: EventManager,
        context_manager: Optional['ContextManager'] = None,
        config_manager: Optional['ConfigManager'] = None,
        logger: Optional[DITALogger] = None
    ):
        # Initialize logger first
        self.logger = logger if isinstance(logger, DITALogger) else DITALogger(name=__name__)

        # Store core dependencies
        self.db_path = Path(db_path)
        self.cache = cache
        self.event_manager = event_manager

        # Store optional dependencies
        self._context_manager = context_manager
        self._config_manager = config_manager

        # Initialize storage with required dependencies
        self.storage = MetadataStorage(
            db_path=self.db_path,
            cache=self.cache,
            event_manager=self.event_manager,
            logger=self.logger
        )

        # Initialize placeholders for dependent components
        self.extractor = None
        self.key_manager = None

        # Register event handlers
        self._register_event_handlers()


    @property
    def context_manager(self) -> Optional['ContextManager']:
        return self._context_manager

    @context_manager.setter
    def context_manager(self, manager: 'ContextManager') -> None:
        self._context_manager = manager
        self._initialize_dependent_components()

    @property
    def config_manager(self) -> Optional['ConfigManager']:
        return self._config_manager

    @config_manager.setter
    def config_manager(self, manager: 'ConfigManager') -> None:
        self._config_manager = manager
        self._initialize_dependent_components()


    def _initialize_dependent_components(self) -> None:
        """Initialize components that require both config and context managers."""
        if self._config_manager is not None and self._context_manager is not None:
            # Initialize extractor
            self.extractor = MetadataExtractor(
                cache=self.cache,
                config_manager=self._config_manager,
                logger=self.logger
            )

            # Import KeyManager dynamically to avoid circular imports
            try:
                key_manager_module = importlib.import_module('..key_manager', __package__)
                KeyManager: Type = getattr(key_manager_module, 'KeyManager')

                # Initialize key manager
                self.key_manager = KeyManager(
                    event_manager=self.event_manager,
                    cache=self.cache,
                    config_manager=self._config_manager,
                    context_manager=self._context_manager,
                    metadata_storage=self.storage,
                    logger=self.logger
                )
            except ImportError as e:
                self.logger.error(f"Failed to import KeyManager: {str(e)}")
                raise
            except Exception as e:
                self.logger.error(f"Failed to initialize KeyManager: {str(e)}")
                raise

    def store_metadata(
            self,
            content_id: str,
            metadata: Dict[str, Any]
        ) -> None:
            """Store metadata using storage component."""
            self.storage.store_bulk_metadata([(content_id, metadata)])

    def get_content_relationships(self, content_id: str) -> List[Dict[str, Any]]:
        """Get content relationships from storage."""
        with self.storage.transaction(content_id) as txn:
            return self.storage.get_metadata(content_id).get('relationships', [])

    @contextmanager
    def transaction(
        self,
        content_id: str
    ) -> Generator[MetadataTransaction, None, None]:
        """Create metadata transaction."""
        with self.storage.transaction(content_id) as txn:
            yield txn

    def _register_event_handlers(self) -> None:
        """Register for metadata-related events."""
        self.event_manager.subscribe(
            EventType.STATE_CHANGE,
            self._handle_state_change
        )
        self.event_manager.subscribe(
            EventType.CACHE_INVALIDATE,
            self._handle_cache_invalidation
        )

    def process_metadata(
        self,
        element: ContentElement,
        context: ProcessingContext,
        phase: ProcessingPhase
    ) -> Dict[str, Any]:
        """
        Process element metadata through the pipeline.

        Args:
            element: Element being processed
            context: Current processing context
            phase: Processing phase

        Returns:
            Dict containing processed metadata
        """
        try:
            # Validate required components
            if self.extractor is None:
                raise ValueError("MetadataExtractor not initialized")
            if self.key_manager is None:
                raise ValueError("KeyManager not initialized")
            if self._config_manager is None:
                raise ValueError("ConfigManager not initialized")

            # Extract base metadata
            metadata = self.extractor.extract_metadata(
                element=element,
                context=context,
                phase=phase
            )

            # Process key references if present
            if keyref := metadata.get("keyref"):
                if resolved_key := self.key_manager.resolve_key(
                    key=keyref,
                    context_map=context.navigation.root_map
                ):
                    metadata.update(resolved_key.metadata)
                    metadata["resolved_keyref"] = resolved_key.key

            # Get processing rules
            rules = self._config_manager.get_processing_rules(
                element_type=element.type,
                context=context
            )

            # Apply processing rules
            if rules:
                metadata = self._apply_processing_rules(
                    metadata=metadata,
                    rules=rules,
                    context=context
                )

            # Store metadata
            with self.storage.transaction(element.id) as txn:
                txn.updates = metadata

            return metadata

        except Exception as e:
            self.logger.error(f"Error processing metadata: {str(e)}")
            raise

    def _apply_processing_rules(
        self,
        metadata: Dict[str, Any],
        rules: Dict[str, Any],
        context: ProcessingContext
    ) -> Dict[str, Any]:
        """Apply processing rules to metadata."""
        try:
            processed = metadata.copy()

            for rule in rules.get("metadata_rules", []):
                rule_type = rule.get("type")
                if rule_type == "transform":
                    if target := rule.get("target"):
                        if source_value := processed.get(target):
                            processed[target] = self._transform_value(
                                value=source_value,
                                transform=rule.get("transform", {}),
                                context=context
                            )
                elif rule_type == "inherit":
                    if parent_id := context.navigation.parent_id:
                        if parent_metadata := self.storage.get_metadata(parent_id):
                            processed.update(
                                self._inherit_metadata(
                                    parent_metadata,
                                    rule.get("fields", [])
                                )
                            )

            return processed

        except Exception as e:
            self.logger.error(f"Error applying processing rules: {str(e)}")
            return metadata

    def _transform_value(
        self,
        value: Any,
        transform: Dict[str, Any],
        context: ProcessingContext
    ) -> Any:
        """Transform metadata value based on rules."""
        try:
            transform_type = transform.get("type")

            if transform_type == "map":
                mapping = transform.get("mapping", {})
                return mapping.get(str(value), value)

            elif transform_type == "format":
                format_str = transform.get("format", "{}")
                return format_str.format(value=value)

            elif transform_type == "conditional":
                conditions = transform.get("conditions", [])
                for condition in conditions:
                    if self._evaluate_condition(condition, context):
                        return condition.get("value", value)

            return value

        except Exception as e:
            self.logger.error(f"Error transforming value: {str(e)}")
            return value

    def _inherit_metadata(
        self,
        parent_metadata: Dict[str, Any],
        fields: List[str]
    ) -> Dict[str, Any]:
        """Inherit specific fields from parent metadata."""
        return {
            field: parent_metadata[field]
            for field in fields
            if field in parent_metadata
        }

    def _evaluate_condition(
        self,
        condition: Dict[str, Any],
        context: ProcessingContext
    ) -> bool:
        """Evaluate a condition against context."""
        try:
            if context_path := condition.get("context_path"):
                value = context
                for part in context_path.split('.'):
                    if hasattr(value, part):
                        value = getattr(value, part)
                        if value is None:
                            return False
                    else:
                        return False

                operator = condition.get("operator", "eq")
                target = condition.get("value")

                # Ensure type safety
                if value is None or target is None:
                    return False

                if operator == "eq":
                    return str(value) == str(target)
                elif operator == "contains":
                    # Check if value is a container type
                    if isinstance(value, (str, list, dict, set)):
                        return str(target) in str(value)
                    return False
                elif operator == "startswith":
                    return str(value).startswith(str(target))

            return False

        except Exception as e:
            self.logger.error(f"Error evaluating condition: {str(e)}")
            return False

    def _handle_state_change(self, **event_data) -> None:
        """Handle element state changes."""
        try:
            element_id = event_data.get("element_id")
            state_info = event_data.get("state_info")

            if element_id and state_info:
                # Update metadata state tracking
                if state_info.state == ProcessingState.COMPLETED:
                    self.storage.commit_transaction(element_id)
                elif state_info.state == ProcessingState.ERROR:
                    self.storage.rollback_transaction(element_id)

        except Exception as e:
            self.logger.error(f"Error handling state change: {str(e)}")

    def _handle_cache_invalidation(self, **event_data) -> None:
        """Handle cache invalidation events."""
        try:
            element_id = event_data.get("element_id")
            if element_id:
                # Invalidate in cache
                self.cache.invalidate(
                    key=f"metadata_{element_id}",
                    entry_type=CacheEntryType.METADATA
                )
                # Invalidate in storage
                with self.storage.transaction(element_id) as txn:
                    txn.updates = {}  # Clear metadata

        except Exception as e:
            self.logger.error(f"Error handling cache invalidation: {str(e)}")

    @contextmanager
    def metadata_transaction(
        self,
        content_id: str
    ) -> Generator[MetadataTransaction, None, None]:
        """Create metadata transaction."""
        with self.storage.transaction(content_id) as txn:
            yield txn

    def get_metadata(
        self,
        element_id: str,
        scope: Optional[ContentScope] = None
    ) -> Dict[str, Any]:
        """
        Get metadata with scope filtering.

        Args:
            element_id: Element ID to get metadata for
            scope: Optional scope to filter metadata

        Returns:
            Dict[str, Any]: Filtered metadata
        """
        try:
            # Get base metadata from storage
            base_metadata = self.storage.get_metadata(element_id)

            # If no scope filtering needed or no context manager, return all metadata
            if not scope or not self._context_manager:
                return base_metadata

            # Use context manager to filter by scope
            return {
                k: v for k, v in base_metadata.items()
                if self._context_manager.validate_scope(k, scope)
            }

        except Exception as e:
            self.logger.error(f"Error getting metadata for {element_id}: {str(e)}")
            return {}


    #########################
    # Validation Management #
    #########################

    def validate_metadata(
        self,
        metadata: Dict[str, Any],
        element_type: ElementType,
        context: Optional[ProcessingContext] = None
    ) -> ValidationResult:
        """
        Validate metadata against rules.

        Args:
            metadata: Metadata to validate
            element_type: Type of element metadata belongs to
            context: Optional processing context

        Returns:
            ValidationResult indicating validation status
        """
        try:
            # Validate required components
            if self._config_manager is None:
                return ValidationResult(
                    is_valid=False,
                    messages=[ValidationMessage(
                        path="metadata",
                        message="Config manager not initialized",
                        severity=ValidationSeverity.ERROR,
                        code="missing_config"
                    )]
                )

            # Get metadata rules for element type and context
            rules = self._config_manager.get_metadata_rules(
                phase=ProcessingPhase.VALIDATION,
                element_type=element_type,
                context=context
            )

            messages = []

            # Validate against rules
            messages.extend(self._validate_metadata_content(metadata, rules))

            # Validate metadata state if context provided
            if context:
                messages.extend(self._validate_metadata_state(
                    metadata, context.metadata_state
                ))

            return ValidationResult(
                is_valid=not any(
                    msg.severity == ValidationSeverity.ERROR for msg in messages
                ),
                messages=messages
            )

        except Exception as e:
            self.logger.error(f"Error validating metadata: {str(e)}")
            return ValidationResult(
                is_valid=False,
                messages=[ValidationMessage(
                    path="metadata",
                    message=f"Metadata validation error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    code="validation_error"
                )]
            )

    def _validate_metadata_content(
        self,
        metadata: Dict[str, Any],
        rules: Dict[str, Any]
    ) -> List[ValidationMessage]:
        """Validate metadata content against rules."""
        messages = []

        try:
            # Required fields validation
            if required := rules.get('required_fields', []):
                for field in required:
                    if field not in metadata:
                        messages.append(ValidationMessage(
                            path=f"metadata.{field}",
                            message=f"Missing required field: {field}",
                            severity=ValidationSeverity.ERROR,
                            code="missing_required"
                        ))

            # Field type validation
            if field_types := rules.get('field_types', {}):
                for field, field_type in field_types.items():
                    if field in metadata:
                        if not self._validate_field_type(
                            metadata[field],
                            field_type
                        ):
                            messages.append(ValidationMessage(
                                path=f"metadata.{field}",
                                message=f"Invalid type for field {field}",
                                severity=ValidationSeverity.ERROR,
                                code="invalid_type"
                            ))

            # Field value validation
            if value_rules := rules.get('value_rules', {}):
                for field, field_rules in value_rules.items():
                    if field in metadata:
                        field_messages = self._validate_field_value(
                            field,
                            metadata[field],
                            field_rules
                        )
                        messages.extend(field_messages)

            return messages

        except Exception as e:
            self.logger.error(f"Error validating metadata content: {str(e)}")
            messages.append(ValidationMessage(
                path="metadata",
                message=f"Content validation error: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="content_validation_error"
            ))
            return messages

    def _validate_metadata_state(
        self,
        metadata: Dict[str, Any],
        state: MetadataState
    ) -> List[ValidationMessage]:
        """Validate metadata state."""
        messages = []

        try:
            # Validate phase consistency
            if state.phase == ProcessingPhase.ERROR:
                messages.append(ValidationMessage(
                    path="metadata.state",
                    message="Metadata in error state",
                    severity=ValidationSeverity.ERROR,
                    code="error_state"
                ))

            # Validate key references
            if key_refs := state.key_references:
                messages.extend(self._validate_key_references(key_refs))

            # Validate metadata references
            if metadata_refs := metadata.get('metadata_refs', {}):
                messages.extend(self._validate_metadata_references(
                    metadata_refs,
                    state
                ))

            return messages

        except Exception as e:
            self.logger.error(f"Error validating metadata state: {str(e)}")
            messages.append(ValidationMessage(
                path="metadata.state",
                message=f"State validation error: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="state_validation_error"
            ))
            return messages

    def _validate_field_type(self, value: Any, expected_type: str) -> bool:
        """Validate field value type."""
        try:
            type_map = {
                "string": str,
                "number": (int, float),
                "integer": int,
                "boolean": bool,
                "array": list,
                "object": dict,
                "null": type(None)
            }

            expected = type_map.get(expected_type)
            if not expected:
                return True  # Skip validation for unknown types

            return isinstance(value, expected)

        except Exception as e:
            self.logger.error(f"Error validating field type: {str(e)}")
            return False

    def _validate_field_value(
        self,
        field: str,
        value: Any,
        rules: Dict[str, Any]
    ) -> List[ValidationMessage]:
        """Validate field value against rules."""
        messages = []

        try:
            # Pattern validation
            if pattern := rules.get('pattern'):
                if not re.match(pattern, str(value)):
                    messages.append(ValidationMessage(
                        path=f"metadata.{field}",
                        message=f"Value does not match pattern: {pattern}",
                        severity=ValidationSeverity.ERROR,
                        code="pattern_mismatch"
                    ))

            # Range validation
            if isinstance(value, (int, float)):
                if min_val := rules.get('min'):
                    if value < min_val:
                        messages.append(ValidationMessage(
                            path=f"metadata.{field}",
                            message=f"Value below minimum: {min_val}",
                            severity=ValidationSeverity.ERROR,
                            code="below_minimum"
                        ))

                if max_val := rules.get('max'):
                    if value > max_val:
                        messages.append(ValidationMessage(
                            path=f"metadata.{field}",
                            message=f"Value above maximum: {max_val}",
                            severity=ValidationSeverity.ERROR,
                            code="above_maximum"
                        ))

            # Enum validation
            if allowed := rules.get('allowed_values'):
                if value not in allowed:
                    messages.append(ValidationMessage(
                        path=f"metadata.{field}",
                        message=f"Invalid value. Allowed: {allowed}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_value"
                    ))

            return messages

        except Exception as e:
            self.logger.error(f"Error validating field value: {str(e)}")
            messages.append(ValidationMessage(
                path=f"metadata.{field}",
                message=f"Value validation error: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="value_validation_error"
            ))
            return messages

    def _validate_key_references(
        self,
        key_refs: List[str]
    ) -> List[ValidationMessage]:
        """Validate key references."""
        messages = []

        try:
            if self.key_manager is None:
                messages.append(ValidationMessage(
                    path="metadata.key_refs",
                    message="Key manager not initialized",
                    severity=ValidationSeverity.ERROR,
                    code="missing_key_manager"
                ))
                return messages

            for key in key_refs:
                if not self.key_manager.get_key_definition(key):
                    messages.append(ValidationMessage(
                        path=f"metadata.key_refs.{key}",
                        message=f"Invalid key reference: {key}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_key_ref"
                    ))

            return messages

        except Exception as e:
            self.logger.error(f"Error validating key references: {str(e)}")
            messages.append(ValidationMessage(
                path="metadata.key_refs",
                message=f"Key reference validation error: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="key_ref_validation_error"
            ))
            return messages

    def _validate_metadata_references(
        self,
        metadata_refs: Dict[str, str],
        state: MetadataState
    ) -> List[ValidationMessage]:
        """Validate metadata references."""
        messages = []

        try:
            for ref_id, ref_type in metadata_refs.items():
                # Check reference exists
                if not self.storage.get_metadata(ref_id):
                    messages.append(ValidationMessage(
                        path=f"metadata.refs.{ref_id}",
                        message=f"Invalid metadata reference: {ref_id}",
                        severity=ValidationSeverity.ERROR,
                        code="invalid_metadata_ref"
                    ))

            return messages

        except Exception as e:
            self.logger.error(f"Error validating metadata references: {str(e)}")
            messages.append(ValidationMessage(
                path="metadata.refs",
                message=f"Metadata reference validation error: {str(e)}",
                severity=ValidationSeverity.ERROR,
                code="metadata_ref_validation_error"
            ))
            return messages


    def cleanup(self) -> None:
        """Clean up manager resources."""
        try:
            # Clean up storage
            self.storage.cleanup()

            # Clean up key manager if initialized
            if self.key_manager is not None:
                self.key_manager.cleanup()

            # Clear cache
            self.cache.clear()
            self.logger.debug("Metadata manager cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise
