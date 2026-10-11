
import time
from typing import Any, Callable


def is_retryable_api_error(error: Exception) -> bool:
    """Identify network and temporary server failures."""
    retryable_names = {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
    }

    return (
        isinstance(error, (TimeoutError, ConnectionError))
        or type(error).__name__ in retryable_names
    )


def execute_with_retry(
    operation: Callable[[], dict[str, Any]],
    max_retries: int = 1,
) -> tuple[dict[str, Any], int]:
    """Execute a tool with bounded retries for execution failures."""

    if max_retries < 0:
        raise ValueError("max_retries cannot be negative.")

    for attempt in range(max_retries + 1):
        try:
            result = operation()

        except Exception:
            result = {
                "success": False,
                "error": {
                    "type": "execution_error",
                    "message": "Tool execution raised an exception.",
                },
            }

        if result.get("success"):
            return result, attempt + 1

        error = result.get("error", {})

        # Invalid arguments and unknown tools won't improve with retries.
        if error.get("type") != "execution_error":
            return result, attempt + 1

        if attempt == max_retries:
            return result, attempt + 1

        time.sleep(min(0.2 * (2 ** attempt), 1.0))

    raise RuntimeError("Unreachable retry state.")
