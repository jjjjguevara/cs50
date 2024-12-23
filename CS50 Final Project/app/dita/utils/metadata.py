# app/dita/utils/metadata.py

from typing import(
    ContextManager,
    Generator,
    Optional,
    Callable,
    TypeVar,
    Tuple,
    Dict,
    List,
    Any,
    Set,
)
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, field
import sqlite3
from pathlib import Path
import logging
import yaml
from uuid import uuid4


from lxml import etree
from lxml.etree import _Element
import frontmatter
import json
import re
from enum import Enum
from app.dita.models.types import(
    PathLike,
    ElementType,
    ProcessingContext,
    ProcessingPhase,
    ProcessingState,
    ContentType,
    MetadataField,
    MetadataTransaction,
    TrackedElement,
    ValidationSeverity,
    ValidationMessage,
    ValidationResult,
    YAMLFrontmatter,
    KeyDefinition,
    LogContext
)

T = TypeVar('T')

# Global config
from app_config import DITAConfig
from app.dita.models.types import ElementType, ProcessingPhase
from app.dita.utils.logger import DITALogger
from app.dita.utils.cache import ContentCache
from app.dita.event_manager import EventManager, EventType


class MetadataManager:
    """
    Centralized metadata management system.
    Handles both transient and persistent metadata with event-driven updates.
    """
    def __init__(
        self,
        db_path: Path,
        cache: ContentCache,
        event_manager: EventManager,
        context_manager: ContextManager,
        config_manager: ConfigManager,
        logger: Optional[DITALogger] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.db_path = db_path
        self.cache = cache
        self.event_manager = event_manager
        self.context_manager = context_manager
        self.config_manager = config_manager

        # Initialize components
        self.storage = MetadataStorage(
            db_path=db_path,
            cache=cache,
            event_manager=event_manager,
            logger=logger
        )
        self.extractor = MetadataExtractor(
            cache=cache,
            config_manager=config_manager,
            logger=logger
        )
        self.validator = MetadataValidator(
            config_manager=config_manager,
            logger=logger
        )

        # Register event handlers
        self._register_event_handlers()

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

    def _handle_state_change(self, **event_data) -> None:
        """Handle element state changes."""
        try:
            element_id = event_data.get("element_id")
            state_info = event_data.get("state_info")

            if element_id and state_info:
                # Update metadata state tracking
                if state_info.state == ProcessingState.COMPLETED:
                    # Persist metadata when processing completes
                    self.storage.commit_metadata(element_id)
                elif state_info.state == ProcessingState.ERROR:
                    # Rollback on error
                    self.storage.rollback_metadata(element_id)

        except Exception as e:
            self.logger.error(f"Error handling state change: {str(e)}")

    def _handle_cache_invalidation(self, **event_data) -> None:
        """Handle cache invalidation events."""
        try:
            element_id = event_data.get("element_id")
            if element_id:
                self.storage.invalidate_metadata(element_id)
        except Exception as e:
            self.logger.error(f"Error handling cache invalidation: {str(e)}")

    def process_metadata(
        self,
        element: TrackedElement,
        context: ProcessingContext,
        phase: ProcessingPhase
    ) -> Dict[str, Any]:
        """
        Process element metadata based on phase.

        Args:
            element: Element being processed
            context: Current processing context
            phase: Processing phase

        Returns:
            Dict containing processed metadata
        """
        try:
            # Extract base metadata
            metadata = self.extractor.extract_metadata(
                element=element,
                context=context,
                phase=phase
            )

            # Validate metadata
            validation_result = self.validator.validate_metadata(
                metadata=metadata,
                element_type=element.type,
                phase=phase
            )

            if not validation_result.is_valid:
                self.logger.warning(
                    f"Metadata validation failed for {element.id}: "
                    f"{validation_result.messages}"
                )

            # Store metadata
            with self.storage.transaction(element.id) as txn:
                txn.updates = metadata

            return metadata

        except Exception as e:
            self.logger.error(f"Error processing metadata: {str(e)}")
            raise

    def get_metadata(
        self,
        element_id: str,
        scope: Optional[ContentScope] = None
    ) -> Dict[str, Any]:
        """Get metadata with optional scope filtering."""
        return self.storage.get_metadata(element_id, scope)

    def store_metadata(
        self,
        element_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Store metadata with transaction handling."""
        with self.storage.transaction(element_id) as txn:
            txn.updates = metadata

    @contextmanager
    def metadata_transaction(
        self,
        element_id: str
    ) -> Generator[MetadataTransaction, None, None]:
        """Context manager for metadata transactions."""
        with self.storage.transaction(element_id) as txn:
            yield txn

    def cleanup(self) -> None:
        """Clean up manager resources."""
        try:
            self.storage.cleanup()
            self.cache.clear()
            self.logger.debug("Metadata manager cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise
