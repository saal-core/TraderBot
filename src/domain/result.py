"""Result pattern for type-safe error handling.

This module provides a Result type that represents either success or failure,
eliminating the need for inconsistent tuple returns and exception handling.
"""
from dataclasses import dataclass, field
from typing import Generic, TypeVar, Callable, Optional
from enum import Enum, auto

T = TypeVar('T')


class ErrorCode(Enum):
    """Error codes for categorizing failures."""
    
    # Database errors
    CONNECTION_FAILED = auto()
    QUERY_TIMEOUT = auto()
    INVALID_SQL = auto()
    NO_RESULTS = auto()
    
    # LLM errors
    LLM_UNAVAILABLE = auto()
    CLASSIFICATION_FAILED = auto()
    GENERATION_FAILED = auto()
    
    # Market data errors
    SYMBOL_NOT_FOUND = auto()
    API_RATE_LIMITED = auto()
    DATA_UNAVAILABLE = auto()
    
    # Validation errors
    INVALID_INPUT = auto()
    SECURITY_VIOLATION = auto()
    
    # General errors
    UNKNOWN_ERROR = auto()


@dataclass(frozen=True)
class Error:
    """Represents an error with code, message, and optional details."""
    
    code: ErrorCode
    message: str
    details: Optional[dict] = None
    
    def __str__(self) -> str:
        return f"[{self.code.name}] {self.message}"


@dataclass(frozen=True)
class Result(Generic[T]):
    """
    A Result type that represents either success or failure.
    
    Usage:
        # Success case
        result = Result.ok(data)
        if result.is_success:
            process(result.value)
            
        # Failure case  
        result = Result.fail_with(ErrorCode.NO_RESULTS, "No data found")
        if result.is_failure:
            handle_error(result.error)
            
        # Chaining operations
        result = get_data().map(transform).map(validate)
        
        # Default values
        value = result.unwrap_or(default_value)
    """
    
    _value: Optional[T]
    _error: Optional[Error]
    
    @classmethod
    def ok(cls, value: T) -> "Result[T]":
        """Create a successful result with the given value."""
        return cls(_value=value, _error=None)
    
    @classmethod
    def fail(cls, error: Error) -> "Result[T]":
        """Create a failed result with the given error."""
        return cls(_value=None, _error=error)
    
    @classmethod
    def fail_with(
        cls,
        code: ErrorCode,
        message: str,
        details: Optional[dict] = None
    ) -> "Result[T]":
        """Create a failed result with error code and message."""
        return cls.fail(Error(code=code, message=message, details=details))
    
    @property
    def is_success(self) -> bool:
        """Check if the result is successful."""
        return self._error is None
    
    @property
    def is_failure(self) -> bool:
        """Check if the result is a failure."""
        return self._error is not None
    
    @property
    def value(self) -> T:
        """
        Get the value from a successful result.
        
        Raises:
            ValueError: If the result is a failure.
        """
        if self._error:
            raise ValueError(f"Cannot get value from failed result: {self._error}")
        return self._value
    
    @property
    def error(self) -> Error:
        """
        Get the error from a failed result.
        
        Raises:
            ValueError: If the result is successful.
        """
        if self._error is None:
            raise ValueError("Cannot get error from successful result")
        return self._error
    
    def map(self, fn: Callable[[T], "Result[T]"]) -> "Result[T]":
        """
        Chain operations on successful results.
        
        If this result is a failure, returns self without calling fn.
        If this result is a success, applies fn to the value.
        """
        if self.is_failure:
            return self
        return fn(self._value)
    
    def unwrap_or(self, default: T) -> T:
        """Get value or return default if failed."""
        return self._value if self.is_success else default
    
    def unwrap_or_else(self, fn: Callable[[Error], T]) -> T:
        """Get value or compute from error."""
        return self._value if self.is_success else fn(self._error)
