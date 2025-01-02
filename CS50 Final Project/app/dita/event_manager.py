# app/dita/event_manager.py

from typing import Dict, Set, Callable, List, Optional, Any
from enum import Enum
import logging
from datetime import datetime
from functools import wraps

from app.dita.utils.cache import ContentCache, CacheEntryType
from app.dita.models.types import(
    ContentScope,
    ProcessingPhase,
    ProcessingState,
    ProcessingStateInfo,
    TrackedElement,
    ElementType
)
class EventType(Enum):
    """Event types for the processing pipeline."""
    PHASE_START = "phase_start"
    PHASE_END = "phase_end"
    STATE_CHANGE = "state_change"
    ERROR = "error"
    CACHE_UPDATE = "cache_update"
    CACHE_INVALIDATE = "cache_invalidate"
    CACHE_INVALIDATED = "cache_invalidated"
    FEATURE_UPDATED = "feature_updated"
    RULE_UPDATED = "rule_updated"
    CONFIG_UPDATE = "config_update"
    VALIDATION_FAILED = "validation_failed"
    CACHE_PATTERN_MATCHED = "cache_pattern_matched"
    RULE_RESOLVED = "rule_resolved"

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
        self._state_history: Dict[str, List[ProcessingState]] = {}
        self._event_stack: List[str] = []  # Event stack tracking
        self._max_event_depth = 10  # Maximum event nesting depth

    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        """Register an event handler."""
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Callable) -> None:
        """Remove an event handler."""
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    def emit(self, event_type: EventType, **data) -> None:
        """Emit an event with recursion protection."""
        try:
            # Create unique event identifier
            event_id = f"{event_type.value}_{data.get('element_id', '')}"

            # Check event depth
            if len(self._event_stack) >= self._max_event_depth:
                self.logger.warning(f"Maximum event depth reached, skipping event: {event_id}")
                return

            # Check for recursive events
            if event_id in self._event_stack:
                self.logger.debug(f"Skipping recursive event: {event_id}")
                return

            # Push event to stack
            self._event_stack.append(event_id)

            try:
                # Process event handlers
                for handler in self._handlers.get(event_type, []):
                    try:
                        handler(**data)
                    except Exception as e:
                        self.logger.error(f"Error in event handler: {str(e)}")
            finally:
                # Pop event from stack
                self._event_stack.pop()

        except Exception as e:
            self.logger.error(f"Error emitting event: {str(e)}")

    def start_phase(self, element_id: str, phase: ProcessingPhase) -> None:
            """Start a processing phase with proper state management."""
            try:
                if element_id in self._active_phases:
                    current_phase = self._active_phases[element_id]
                    self.logger.warning(
                        f"Element {element_id} already in {current_phase.value} phase"
                    )
                    return

                # Update phase tracking
                self._active_phases[element_id] = phase

                # Create state info
                state_info = ProcessingStateInfo(
                    phase=phase,
                    state=ProcessingState.PENDING,
                    element_id=element_id,
                    timestamp=datetime.now()
                )

                # Emit phase start event with state info
                self.emit(
                    EventType.PHASE_START,
                    element_id=element_id,
                    state_info=state_info
                )

            except Exception as e:
                self.logger.error(f"Error starting phase for {element_id}: {str(e)}")
                raise

    def end_phase(self, element_id: str, phase: ProcessingPhase) -> None:
        """End a processing phase with state cleanup."""
        try:
            if self._active_phases.get(element_id) != phase:
                self.logger.error(
                    f"Phase mismatch for {element_id}: expected {phase.value}"
                )
                return

            # Update state tracking
            current_state = self._state_history.get(element_id, [])[-1]
            state_info = ProcessingStateInfo(
                phase=phase,
                state=ProcessingState.COMPLETED,
                element_id=element_id,
                timestamp=datetime.now(),
                previous_state=current_state
            )

            # Cleanup phase tracking
            del self._active_phases[element_id]

            # Emit phase end event with final state
            self.emit(
                EventType.PHASE_END,
                element_id=element_id,
                state_info=state_info
            )

        except Exception as e:
            self.logger.error(f"Error ending phase for {element_id}: {str(e)}")
            raise

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
        self,
        element_id: str,
        new_state: ProcessingState,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update element state with history tracking."""
        try:
            # Get current state and phase
            current_phase = self._active_phases.get(element_id, ProcessingPhase.DISCOVERY)  # Default to DISCOVERY
            current_state = (
                self._state_history.get(element_id, [])[-1]
                if self._state_history.get(element_id)
                else ProcessingState.PENDING  # Default to PENDING
            )

            # Create state info with proper defaults
            state_info = ProcessingStateInfo(
                element_id=element_id,
                phase=current_phase,  # Now always has a value
                state=new_state,
                previous_state=current_state,
                timestamp=datetime.now()
            )

            # Update state history
            if element_id not in self._state_history:
                self._state_history[element_id] = []
            self._state_history[element_id].append(new_state)

            # Emit state change event
            self.emit(
                EventType.STATE_CHANGE,
                element_id=element_id,
                state_info=state_info,
                metadata=metadata
            )

        except Exception as e:
            self.logger.error(f"Error updating state for {element_id}: {str(e)}")
            raise

    def handle_cache_update(
        self,
        key: str,
        data: Any,
        entry_type: CacheEntryType,
        element_type: ElementType,
        phase: ProcessingPhase,
        scope: ContentScope = ContentScope.LOCAL,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Handle cache updates with event emission."""
        try:
            # Update cache
            self.cache.set(
                key=key,
                data=data,
                entry_type=entry_type,
                element_type=element_type,
                phase=phase,
                scope=scope,
                ttl=ttl,
                metadata=metadata
            )

            # Emit cache update event
            self.emit(
                EventType.CACHE_UPDATE,
                cache_key=key,
                entry_type=entry_type,
                element_type=element_type,
                phase=phase,
                scope=scope
            )

        except Exception as e:
            self.logger.error(f"Error updating cache for {key}: {str(e)}")
            raise

    def handle_cache_invalidate(self, pattern: str) -> None:
        """Handle cache invalidation with event emission."""
        try:
            # Invalidate cache entries
            self.cache.invalidate_by_pattern(pattern)

            # Emit cache invalidation event
            self.emit(
                EventType.CACHE_INVALIDATE,
                pattern=pattern
            )

        except Exception as e:
            self.logger.error(f"Error invalidating cache pattern {pattern}: {str(e)}")
            raise


    def clear_tracked_elements(self) -> None:
        """Clear tracked elements set."""
        self._processed_elements.clear()

    def is_element_processed(self, element_id: str) -> bool:
        """Check if element has been processed."""
        return element_id in self._processed_elements
