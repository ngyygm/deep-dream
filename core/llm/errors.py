"""LLM 客户端异常类型（供 pipeline 与各 mixin 按类型捕获）。"""

from __future__ import annotations
from typing import Optional, Any


class LLMError(RuntimeError):
    """Base class for all LLM-related errors."""

    def __init__(self, message: str, *, retry_after: Optional[float] = None,
                 suggested_action: Optional[str] = None, context: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.retry_after = retry_after  # Suggested retry delay in seconds
        self.suggested_action = suggested_action  # Human-readable action suggestion
        self.context = context or {}


class LLMTransientError(LLMError):
    """Base class for transient (retryable) LLM errors.

    These errors should trigger retry with exponential backoff.
    Subclasses: LLMConnectionError, LLMTimeoutError, LLMRateLimitError
    """
    pass


class LLMPermanentError(LLMError):
    """Base class for permanent (non-retryable) LLM errors.

    These errors should fail immediately without retry.
    Subclasses: LLMContextBudgetExceeded, LLMValidationError
    """
    pass


class LLMConnectionError(LLMTransientError):
    """Connection failed or was lost during the request.

    Examples: connection refused, DNS failure, network timeout
    """

    def __init__(self, message: str, *, original_error: Optional[Exception] = None,
                 retry_after: Optional[float] = None):
        suggested_action = "Check network connectivity and LLM service status"
        context = {"original_error_type": type(original_error).__name__} if original_error else None
        super().__init__(message, retry_after=retry_after or 3.0,
                        suggested_action=suggested_action, context=context)
        self.original_error = original_error


class LLMTimeoutError(LLMTransientError):
    """Request timed out before completion.

    This could be due to network latency or slow model response.
    """

    def __init__(self, message: str, *, timeout_seconds: Optional[float] = None,
                 retry_after: Optional[float] = None):
        suggested_action = "Increase timeout or reduce request complexity"
        context = {"timeout_seconds": timeout_seconds} if timeout_seconds else None
        super().__init__(message, retry_after=retry_after or 5.0,
                        suggested_action=suggested_action, context=context)
        self.timeout_seconds = timeout_seconds


class LLMRateLimitError(LLMTransientError):
    """Rate limit exceeded (HTTP 429 or TPM/RPM limit).

    The service is throttling requests; retry after specified delay.
    """

    def __init__(self, message: str, *, retry_after: Optional[float] = None,
                 limit_type: Optional[str] = None):
        suggested_action = "Reduce request rate or increase quota"
        context = {"limit_type": limit_type} if limit_type else None
        # Rate limit errors typically need longer retry delays
        default_retry = 60.0 if limit_type == "tpm" else 10.0
        super().__init__(message, retry_after=retry_after or default_retry,
                        suggested_action=suggested_action, context=context)
        self.limit_type = limit_type


class LLMContextBudgetExceeded(LLMPermanentError):
    """Input tokens exceed the model's context window.

    This is a permanent error for the given input; retrying won't help.
    """

    def __init__(self, message: str, *, context_window: Optional[int] = None,
                 estimated_tokens: Optional[int] = None):
        suggested_action = "Reduce input length or use a model with larger context"
        context = {}
        if context_window:
            context["context_window"] = context_window
        if estimated_tokens:
            context["estimated_tokens"] = estimated_tokens
        super().__init__(message, suggested_action=suggested_action, context=context)
        self.context_window = context_window
        self.estimated_tokens = estimated_tokens


class LLMValidationError(LLMPermanentError):
    """Invalid request parameters or content.

    Examples: invalid API key, unsupported model, malformed request
    """

    def __init__(self, message: str, *, field: Optional[str] = None,
                 value: Optional[Any] = None):
        suggested_action = "Check request parameters and configuration"
        context = {}
        if field:
            context["field"] = field
        if value is not None:
            context["value"] = str(value)[:100]  # Truncate long values
        super().__init__(message, suggested_action=suggested_action, context=context)
        self.field = field
        self.value = value


class LLMResponseError(LLMError):
    """Invalid or malformed response from LLM.

    This could indicate a model hallucination or service issue.
    """

    def __init__(self, message: str, *, response_snippet: Optional[str] = None,
                 is_retryable: bool = True):
        suggested_action = "Retry or adjust prompt to encourage valid output"
        context = {}
        if response_snippet:
            context["response_snippet"] = response_snippet[:200]
        # Use appropriate base class based on retryability
        if is_retryable:
            super().__init__(message, retry_after=1.0,
                           suggested_action=suggested_action, context=context)
        else:
            super().__init__(message, suggested_action=suggested_action, context=context)
        self.response_snippet = response_snippet
        self.is_retryable = is_retryable


def classify_error(error: Exception) -> LLMError:
    """Convert a generic exception into an appropriate LLMError subclass.

    Args:
        error: The original exception

    Returns:
        An LLMError subclass instance with appropriate context
    """
    error_str = str(error).lower()
    error_type = type(error).__name__

    # Already an LLMError - return as-is
    if isinstance(error, LLMError):
        return error

    # Rate limit detection
    if hasattr(error, "status_code") and error.status_code == 429:
        return LLMRateLimitError(
            f"Rate limit exceeded: {error}",
            retry_after=getattr(error, "retry_after", None),
            limit_type="tpm" if "tpm" in error_str else "rpm"
        )

    if "rate_limit" in error_str or "tpm" in error_str or "throttl" in error_str:
        return LLMRateLimitError(str(error))

    # Connection error detection
    conn_keywords = (
        "connection refused", "connectionerror",
        "failed to establish a new connection", "newconnectionerror",
        "temporarily unreachable", "temporary failure in name resolution",
        "name or service not known", "connection aborted",
        "connection reset", "errno 111", "timed out"
    )
    if any(kw in error_str for kw in conn_keywords):
        return LLMConnectionError(str(error), original_error=error)

    # Timeout detection
    if "timeout" in error_str or "timed out" in error_str:
        return LLMTimeoutError(str(error))

    # Context budget exceeded
    context_keywords = (
        "context length", "maximum context", "max context", "context window",
        "token limit", "too many tokens", "maximum tokens", "exceeds the maximum",
        "prompt is too long", "input is too long", "input length", "length limit"
    )
    if any(kw in error_str for kw in context_keywords):
        return LLMContextBudgetExceeded(str(error))

    # Default: generic error (assume transient for resilience)
    return LLMTransientError(str(error), context={"original_type": error_type})
