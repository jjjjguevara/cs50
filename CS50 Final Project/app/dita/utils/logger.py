from typing import Dict, Optional, Any, Union, TextIO
from pathlib import Path
import logging
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
import traceback
from functools import wraps
import inspect

from ..models.types import (
    ProcessingPhase,
    ProcessingError,
    LogContext,
    ValidationResult,
    ProcessingState
)

@dataclass
class LogEntry:
    """Structured log entry for JSON serialization."""
    timestamp: str
    level: str
    message: str
    function: str
    module: str
    line_number: int
    context: Dict[str, Any] = field(default_factory=dict)
    traceback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary for JSON serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}

class JSONFormatter(logging.Formatter):
    """Custom formatter for JSON log output."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Get call frame information
        frame = inspect.currentframe()
        while frame:
            if frame.f_code.co_name != 'format':
                break
            frame = frame.f_back

        module = frame.f_code.co_name if frame else ''
        line_no = frame.f_lineno if frame else 0

        # Create log entry
        entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created).isoformat(),
            level=record.levelname,
            message=record.getMessage(),
            function=record.funcName,
            module=module,
            line_number=line_no,
            context=getattr(record, 'context', {}),
            traceback=record.exc_text if record.exc_text else None,
            metadata=getattr(record, 'metadata', {})
        )

        return json.dumps(entry.to_dict())

class DITALogger:
    """
    Enhanced logger with JSON support and backward compatibility.
    """

    def __init__(
        self,
        name: str = "dita",
        log_file: Optional[Union[str, Path]] = None,
        log_level: int = logging.DEBUG,
        enable_json: bool = True,
        console_output: bool = True
    ):
        """Initialize logger with backward compatibility."""
        # Create base logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.propagate = False

        # Clear existing handlers
        self.logger.handlers.clear()

        # Create formatters
        json_formatter = JSONFormatter()
        text_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Configure console output
        if console_output:
            console = logging.StreamHandler()
            console.setFormatter(json_formatter if enable_json else text_formatter)
            console.setLevel(logging.INFO)
            self.logger.addHandler(console)

        # Configure file output
        if log_file:
            file_path = Path(log_file)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # JSON log file
            if enable_json:
                json_handler = logging.FileHandler(
                    str(file_path.with_suffix('.json'))
                )
                json_handler.setFormatter(json_formatter)
                json_handler.setLevel(log_level)
                self.logger.addHandler(json_handler)

            # Standard log file (for backward compatibility)
            text_handler = logging.FileHandler(
                str(file_path.with_suffix('.log'))
            )
            text_handler.setFormatter(text_formatter)
            text_handler.setLevel(log_level)
            self.logger.addHandler(text_handler)

    def _log_with_context(
        self,
        level: int,
        msg: str,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """Internal method for context-aware logging."""
        if self.logger.isEnabledFor(level):
            extra = {
                'context': context or {},
                'metadata': metadata or {}
            }
            self.logger.log(level, msg, *args, extra=extra, **kwargs)

    # Maintain standard logging interface
    def debug(
        self,
        msg: str,
        context: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """Log debug message with optional context."""
        self._log_with_context(logging.DEBUG, msg, context, None, *args, **kwargs)

    def info(
        self,
        msg: str,
        context: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """Log info message with optional context."""
        self._log_with_context(logging.INFO, msg, context, None, *args, **kwargs)

    def warning(
        self,
        msg: str,
        context: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """Log warning message with optional context."""
        self._log_with_context(logging.WARNING, msg, context, None, *args, **kwargs)

    def error(
        self,
        msg: str,
        context: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """Log error message with optional context."""
        self._log_with_context(logging.ERROR, msg, context, None, *args, **kwargs)

    def critical(
        self,
        msg: str,
        context: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """Log critical message with optional context."""
        self._log_with_context(logging.CRITICAL, msg, context, None, *args, **kwargs)

    # Enhanced logging methods
    def log_phase(
        self,
        phase: ProcessingPhase,
        context: LogContext
    ) -> None:
        """Log phase transition with structured context."""
        self._log_with_context(
            logging.INFO,
            f"Processing phase: {phase.value}",
            asdict(context),
            {"phase": phase.value}
        )

    def log_error(
        self,
        error: ProcessingError,
        phase: Optional[ProcessingPhase] = None
    ) -> None:
        """Log processing error with context."""
        context = {
            "phase": phase.value if phase else None,
            "element_id": error.element_id,
            "error_type": error.error_type,
            "context": str(error.context)
        }

        self._log_with_context(
            logging.ERROR,
            error.message,
            context,
            {"stacktrace": error.stacktrace}
        )

    def log_validation(
        self,
        result: ValidationResult,
        context: Dict[str, Any]
    ) -> None:
        """Log validation result with context."""
        self._log_with_context(
            logging.INFO if result.is_valid else logging.WARNING,
            f"Validation {'succeeded' if result.is_valid else 'failed'}",
            context,
            {"validation_messages": [asdict(msg) for msg in result.messages]}
        )

def get_logger(name: str = "dita") -> DITALogger:
    """Get or create a logger instance."""
    return DITALogger(name)
