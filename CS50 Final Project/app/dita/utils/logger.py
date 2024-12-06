# app/dita/utils/logger.py

import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
from functools import wraps

from .types import ProcessingPhase, ProcessingError, LogContext

class DITALogger:
    """Centralized logging for DITA processing pipeline."""

    def __init__(self):
        self.logger = logging.getLogger("dita")
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Configure logging with proper formatters."""
        # Set base logging level
        self.logger.setLevel(logging.DEBUG)

        # Create console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # Create file handler
        file_handler = logging.FileHandler("dita_processing.log")
        file_handler.setLevel(logging.DEBUG)

        # Create formatters and add them to handlers
        console_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        console.setFormatter(console_format)
        file_handler.setFormatter(file_format)

        # Add handlers to logger
        self.logger.addHandler(console)
        self.logger.addHandler(file_handler)

    def log_phase_start(self, phase: ProcessingPhase, context: LogContext) -> None:
        """Log the start of a processing phase."""
        self.logger.info(
            f"Starting {phase.value} phase - "
            f"Map: {context.map_id or 'N/A'}, "
            f"Topic: {context.topic_id or 'N/A'}"
        )

    def log_phase_end(self, phase: ProcessingPhase, context: LogContext) -> None:
        """Log the end of a processing phase."""
        self.logger.info(
            f"Completed {phase.value} phase - "
            f"Map: {context.map_id or 'N/A'}, "
            f"Topic: {context.topic_id or 'N/A'}"
        )

    def log_error(self, error: ProcessingError, phase: Optional[ProcessingPhase] = None) -> None:
        """Log processing error with context."""
        error_msg = (
            f"Error during {phase.value if phase else 'processing'}: "
            f"{error.message}\n"
            f"Context: {error.context}\n"
            f"Element: {error.element_id or 'N/A'}"
        )

        if error.stacktrace:
            error_msg += f"\nStacktrace:\n{error.stacktrace}"

        self.logger.error(error_msg)

    def log_debug_info(self, info: Dict[str, Any], context: LogContext) -> None:
        """Log debug information with context."""
        debug_msg = [
            f"Debug info for {context.phase.value}:",
            f"Map ID: {context.map_id or 'N/A'}",
            f"Topic ID: {context.topic_id or 'N/A'}",
            f"Element ID: {context.element_id or 'N/A'}",
            "Details:"
        ]

        for key, value in info.items():
            debug_msg.append(f"  {key}: {value}")

        self.logger.debug("\n".join(debug_msg))

    def log_state_change(self,
                        element_id: str,
                        old_state: str,
                        new_state: str,
                        context: LogContext) -> None:
        """Log element state changes."""
        self.logger.debug(
            f"State change for {element_id}: "
            f"{old_state} -> {new_state} "
            f"(Phase: {context.phase.value})"
        )

    def create_error_log(self, e: Exception, context: Dict[str, Any]) -> None:
        """Create comprehensive error log entry."""
        self.logger.error(
            "Error Details:\n"
            f"Timestamp: {datetime.now().isoformat()}\n"
            f"Error Type: {type(e).__name__}\n"
            f"Message: {str(e)}\n"
            f"Context: {context}\n"
            f"Stacktrace:\n{traceback.format_exc()}"
        )

def log_processing_phase(phase: ProcessingPhase):
    """Decorator for logging processing phases."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            context = LogContext(
                phase=phase,
                map_id=getattr(self, 'current_map_id', None),
                topic_id=getattr(self, 'current_topic_id', None)
            )

            self.logger.log_phase_start(phase, context)
            try:
                result = func(self, *args, **kwargs)
                self.logger.log_phase_end(phase, context)
                return result
            except Exception as e:
                if isinstance(e, ProcessingError):
                    self.logger.log_error(e, phase)
                else:
                    self.logger.create_error_log(e, {
                        'phase': phase.value,
                        'function': func.__name__,
                        'args': args,
                        'kwargs': kwargs
                    })
                raise
        return wrapper
    return decorator
