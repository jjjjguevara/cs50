# app/dita/event_manager.py

from typing import Dict, Set, Callable, List, Optional, Any
from enum import Enum
import logging
from datetime import datetime
from functools import wraps

from app.dita.utils.cache import ContentCache
from app.dita.models.types import ProcessingPhase, ProcessingState, TrackedElement

class EventType(Enum):
    """Event types for the processing pipeline."""
    PHASE_START = "phase_start"
    PHASE_END = "phase_end"
    STATE_CHANGE = "state_change"
    ERROR = "error"
    CACHE_UPDATE = "cache_update"
    CACHE_INVALIDATE = "cache_invalidate"

class EventManager:
    """
    Centralized event management for the DITA processing pipeline.
    Coordinates processing phases, state changes, and caching.
    """
    def __init__(self, cache: ContentCache):
        self.logger = logging.getLogger(__name__)
        self.cache = cache
        self._handlers: Dict[EventType, List[Callable]] = {
            event_type: [] for event_type in EventType
        }
        self._processed_elements: Set[str] = set()
        self._active_phases: Dict[str, ProcessingPhase] = {}

    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        """Register an event handler."""
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Callable) -> None:
        """Remove an event handler."""
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    def emit(self, event_type: EventType, **data) -> None:
        """Emit an event to registered handlers."""
        try:
            for handler in self._handlers[event_type]:
                handler(**data)
        except Exception as e:
            self.logger.error(f"Error emitting {event_type.value} event: {str(e)}")

    def start_phase(self, element_id: str, phase: ProcessingPhase) -> None:
        """Start a processing phase for an element."""
        if element_id in self._active_phases:
            current_phase = self._active_phases[element_id]
            self.logger.warning(
                f"Element {element_id} already in {current_phase.value} phase"
            )
            return

        self._active_phases[element_id] = phase
        self.emit(EventType.PHASE_START, element_id=element_id, phase=phase)

    def end_phase(self, element_id: str, phase: ProcessingPhase) -> None:
        """End a processing phase for an element."""
        if self._active_phases.get(element_id) != phase:
            self.logger.error(
                f"Phase mismatch for {element_id}: expected {phase.value}"
            )
            return

        del self._active_phases[element_id]
        self.emit(EventType.PHASE_END, element_id=element_id, phase=phase)

    def track_element(self, element: TrackedElement) -> bool:
        """
        Track element processing to prevent duplicates.

        Args:
            element: TrackedElement to track

        Returns:
            bool: False if element was already tracked, True if newly tracked
        """
        if element.id in self._processed_elements:
            return False
        self._processed_elements.add(element.id)
        return True

    def update_element_state(
        self, element: TrackedElement, new_state: ProcessingState
    ) -> None:
        """Update element processing state with event emission."""
        old_state = element.state
        element.state = new_state
        self.emit(
            EventType.STATE_CHANGE,
            element_id=element.id,
            old_state=old_state,
            new_state=new_state
        )

    def clear_tracked_elements(self) -> None:
        """Clear tracked elements set."""
        self._processed_elements.clear()

    def is_element_processed(self, element_id: str) -> bool:
        """Check if element has been processed."""
        return element_id in self._processed_elements
