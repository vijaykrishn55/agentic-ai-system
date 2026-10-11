
import pytest

from src.tools.registry import create_default_registry


@pytest.fixture
def registry():
    return create_default_registry()


def test_add_tool(registry):
    result = registry.execute("add", {"a": 10, "b": 5})

    assert result["success"] is True
    assert result["result"] == 15


def test_multiply_tool(registry):
    result = registry.execute("multiply", {"a": 4, "b": 3})

    assert result["success"] is True
    assert result["result"] == 12


def test_invalid_argument_type(registry):
    result = registry.execute("add", {"a": "invalid", "b": 5})

    assert result["success"] is False
    assert result["error"]["type"] == "validation_error"


def test_missing_argument(registry):
    result = registry.execute("add", {"a": 10})

    assert result["success"] is False
    assert result["error"]["type"] == "validation_error"


def test_extra_argument_rejected(registry):
    result = registry.execute(
        "add",
        {"a": 10, "b": 5, "c": 20},
    )

    assert result["success"] is False


def test_unknown_tool(registry):
    result = registry.execute("divide", {"a": 10, "b": 5})

    assert result["success"] is False
    assert result["error"]["type"] == "unknown_tool"


def test_non_object_arguments(registry):
    result = registry.execute("add", ["10", "5"])

    assert result["success"] is False
