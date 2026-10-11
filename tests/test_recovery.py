
import pytest

from src.agents.recovery import (
    execute_with_retry,
    is_retryable_api_error,
)


def test_retries_execution_error_then_succeeds():
    state = {"attempts": 0}

    def flaky_tool():
        state["attempts"] += 1

        if state["attempts"] == 1:
            return {
                "success": False,
                "error": {"type": "execution_error"},
            }

        return {"success": True, "result": 42}

    result, attempts = execute_with_retry(
        flaky_tool,
        max_retries=1,
    )

    assert result["success"] is True
    assert result["result"] == 42
    assert attempts == 2


def test_stops_after_retry_limit():
    def failing_tool():
        raise RuntimeError("Tool crashed")

    result, attempts = execute_with_retry(
        failing_tool,
        max_retries=2,
    )

    assert result["success"] is False
    assert attempts == 3
    assert result["error"]["type"] == "execution_error"


def test_validation_error_is_not_retried():
    state = {"attempts": 0}

    def invalid_tool():
        state["attempts"] += 1
        return {
            "success": False,
            "error": {"type": "validation_error"},
        }

    result, attempts = execute_with_retry(
        invalid_tool,
        max_retries=3,
    )

    assert result["success"] is False
    assert attempts == 1
    assert state["attempts"] == 1


def test_connection_error_is_retryable():
    assert is_retryable_api_error(ConnectionError())


def test_timeout_is_retryable():
    assert is_retryable_api_error(TimeoutError())


def test_authentication_error_is_not_retryable():
    class AuthenticationError(Exception):
        pass

    assert not is_retryable_api_error(AuthenticationError())


def test_negative_retry_limit_is_rejected():
    with pytest.raises(ValueError):
        execute_with_retry(lambda: {"success": True}, max_retries=-1)
