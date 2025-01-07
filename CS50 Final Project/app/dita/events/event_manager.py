# app/dita/event_manager.py

from typing import Dict, Set, Callable, List, Optional, Any
import logging
from datetime import datetime

from ..cache.cache import ContentCache
from ..types import(
    EventType,
    ContentScope,
    ProcessingPhase,
    ProcessingState,
    ProcessingStatus,
    ContentElement,
    ElementType,
    CacheEntryType
)


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
        self._active_validations: Set[str] = set()
        self._active_phases: Dict[str, ProcessingPhase] = {}
        self._state_history: Dict[str, List[ProcessingState]] = {}
        self._event_stack: List[str] = []  # Event stack tracking
        self._max_event_depth = 10  # Maximum event nesting depth

        # Register internal handlers
        self._register_internal_handlers()

    def _register_internal_handlers(self) -> None:
            """Register internal event handlers."""
            # Register DTD validation handlers
            dtd_events = [
                EventType.DTD_VALIDATION_START,
                EventType.DTD_VALIDATION_END,
                EventType.DTD_VALIDATION_ERROR
            ]

            for event_type in dtd_events:
                self.subscribe(
                    event_type,
                    lambda **kwargs: self._handle_dtd_validation_events(event_type, **kwargs)
                )

    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[..., None]
    ) -> None:
        """Register an event handler."""
        if not isinstance(event_type, EventType):
            raise TypeError(f"event_type must be an EventType, got {type(event_type)}")

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
                state_info = ProcessingStatus(
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
            state_info = ProcessingStatus(
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

    def track_element(self, element: ContentElement) -> bool:
        """
        Track element processing to prevent duplicates.

        Args:
            element: ContentElement to track

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
            state_info = ProcessingStatus(
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

    def _handle_state_change(self, **event_data: Any) -> None:
            """Enhanced state change handling with DTD awareness."""
            try:
                element_id = event_data.get("element_id")
                state_info = event_data.get("state_info")

                if element_id and state_info:
                    # Check for DTD validation
                    if dtd_validation := event_data.get('dtd_validation'):
                        self.emit(
                            EventType.DTD_VALIDATION_COMPLETE,
                            element_id=element_id,
                            validation_result=dtd_validation
                        )

                    # Continue with normal state change handling
                    # ...

            except Exception as e:
                self.logger.error(f"Error handling state change: {str(e)}")

    def clear_tracked_elements(self) -> None:
        """Clear tracked elements set."""
        self._processed_elements.clear()

    def is_element_processed(self, element_id: str) -> bool:
        """Check if element has been processed."""
        return element_id in self._processed_elements

    def _handle_dtd_validation_events(self, event_type: EventType, **event_data: Any) -> None:
        """
        Handle DTD validation events centrally.

        Args:
            event_type: Type of DTD validation event
            **event_data: Event data including:
                - element_id: ID of element being validated
                - validation_context: Validation context
                - validation_result: Optional validation result
                - error: Optional error information
        """
        # Initialize element_id outside try block
        element_id = None

        try:
            # Get element ID with type checking
            element_id = event_data.get("element_id")
            if not element_id:
                self.logger.error("Missing element_id in DTD validation event")
                return

            # Now element_id is confirmed to be a string
            context = event_data.get("validation_context", {})

            if event_type == EventType.DTD_VALIDATION_START:
                # Initialize validation tracking
                self._active_validations.add(element_id)

                # Cache validation context
                self.cache.set(
                    key=f"dtd_context_{element_id}",
                    data=context,
                    entry_type=CacheEntryType.VALIDATION,
                    element_type=ElementType.UNKNOWN,
                    phase=ProcessingPhase.VALIDATION
                )

                self.logger.debug(
                    f"Starting DTD validation for {element_id}"
                )

            elif event_type == EventType.DTD_VALIDATION_END:
                # Clear validation tracking
                self._active_validations.discard(element_id)

                # Process validation result
                if validation_result := event_data.get("validation_result"):
                    if validation_result.is_valid:
                        self.logger.debug(f"DTD validation successful for {element_id}")
                    else:
                        self.logger.warning(
                            f"DTD validation failed for {element_id} with "
                            f"{len(validation_result.messages)} issues"
                        )

                # Clear validation context
                self.cache.invalidate(
                    key=f"dtd_context_{element_id}",
                    entry_type=CacheEntryType.VALIDATION
                )

            elif event_type == EventType.DTD_VALIDATION_ERROR:
                # Handle validation error
                error = event_data.get("error")
                self.logger.error(
                    f"DTD validation error for {element_id}: {error}"
                )

                # Cache error state
                error_state = {
                    "error": str(error),
                    "context": context,
                    "timestamp": datetime.now().isoformat()
                }
                self.cache.set(
                    key=f"dtd_error_{element_id}",
                    data=error_state,
                    entry_type=CacheEntryType.VALIDATION,
                    element_type=ElementType.UNKNOWN,
                    phase=ProcessingPhase.VALIDATION
                )

                # Clear validation tracking
                self._active_validations.discard(element_id)

        except Exception as e:
            self.logger.error(f"Error handling DTD validation event: {str(e)}")
            # Avoid recursive error emission
            if event_type != EventType.DTD_VALIDATION_ERROR:
                # Only emit error if we have a valid element_id
                if isinstance(element_id, str):
                    self.emit(
                        EventType.DTD_VALIDATION_ERROR,
                        element_id=element_id,
                        error=str(e)
                    )
