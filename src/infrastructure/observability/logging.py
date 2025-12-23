"""Structured JSON logging with context support.

Provides consistent log formatting with:
- Request/session ID tracking
- JSON-structured output
- Context propagation across async calls
"""
import logging
import json
from datetime import datetime
from contextvars import ContextVar
from typing import Optional, Any

# Context variables for request tracking
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
session_id_var: ContextVar[str] = ContextVar('session_id', default='')


class StructuredLogger:
    """
    JSON structured logger with context support.
    
    Usage:
        logger = StructuredLogger("my_module")
        
        # Set context for request
        request_id_var.set("req-123")
        
        # Log with automatic context inclusion
        logger.info("Processing query", query="What is my portfolio?")
    """
    
    def __init__(self, name: str, level: int = logging.INFO):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name (usually module name)
            level: Logging level
        """
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._name = name
        
        # Add handler if not already configured
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            self._logger.addHandler(handler)
    
    def _create_log_entry(self, level: str, message: str, **kwargs) -> dict:
        """Create a structured log entry."""
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "logger": self._name,
            "message": message,
        }
        
        # Add context if available
        request_id = request_id_var.get('')
        session_id = session_id_var.get('')
        
        if request_id:
            entry["request_id"] = request_id
        if session_id:
            entry["session_id"] = session_id
        
        # Add any extra fields
        if kwargs:
            entry["data"] = kwargs
        
        return entry
    
    def _log(self, level: str, log_method, message: str, **kwargs) -> None:
        """Internal logging method."""
        entry = self._create_log_entry(level, message, **kwargs)
        log_method(json.dumps(entry))
    
    def info(self, message: str, **kwargs) -> None:
        """Log at INFO level."""
        self._log("INFO", self._logger.info, message, **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log at ERROR level with optional exception."""
        if exception:
            kwargs["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception)
            }
        self._log("ERROR", self._logger.error, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log at WARNING level."""
        self._log("WARNING", self._logger.warning, message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log at DEBUG level."""
        self._log("DEBUG", self._logger.debug, message, **kwargs)


# Global logger instances
_loggers: dict[str, StructuredLogger] = {}


def get_logger(name: str) -> StructuredLogger:
    """Get or create a structured logger by name."""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name)
    return _loggers[name]


def set_request_context(request_id: str, session_id: Optional[str] = None) -> None:
    """Set request context for logging."""
    request_id_var.set(request_id)
    if session_id:
        session_id_var.set(session_id)


def clear_request_context() -> None:
    """Clear request context."""
    request_id_var.set('')
    session_id_var.set('')
