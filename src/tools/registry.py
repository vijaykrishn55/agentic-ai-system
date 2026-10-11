
import json
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class AddInput(BaseModel):
    """Arguments for adding two numbers."""

    model_config = ConfigDict(extra="forbid")

    a: float = Field(description="The first number")
    b: float = Field(description="The second number")


class MultiplyInput(BaseModel):
    """Arguments for multiplying two numbers."""

    model_config = ConfigDict(extra="forbid")

    a: float = Field(description="The first number")
    b: float = Field(description="The second number")


class ToolRegistry:
    """Register, validate, and execute tools dynamically."""

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        description: str,
        schema: type[BaseModel],
        function: Callable[..., Any],
    ) -> None:
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")

        self._tools[name] = {
            "description": description,
            "schema": schema,
            "function": function,
        }

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return tool definitions in the format Groq expects."""
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool["description"],
                    "parameters": tool["schema"].model_json_schema(),
                },
            }
            for name, tool in self._tools.items()
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate arguments, execute a registered tool, and return JSON-safe output."""
        if name not in self._tools:
            return {
                "success": False,
                "error": {
                    "type": "unknown_tool",
                    "message": f"Tool not found: {name}",
                },
            }

        try:
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object.")

            tool = self._tools[name]
            validated = tool["schema"].model_validate(arguments)
            result = tool["function"](**validated.model_dump())

            return {
                "success": True,
                "tool": name,
                "result": result,
            }

        except ValidationError as exc:
            return {
                "success": False,
                "tool": name,
                "error": {
                    "type": "validation_error",
                    "message": "Invalid tool arguments.",
                    "details": exc.errors(include_url=False),
                },
            }

        except (ValueError, TypeError) as exc:
            return {
                "success": False,
                "tool": name,
                "error": {
                    "type": "invalid_arguments",
                    "message": str(exc),
                },
            }

        except Exception:
            return {
                "success": False,
                "tool": name,
                "error": {
                    "type": "execution_error",
                    "message": "The tool failed during execution.",
                },
            }


def add(a: float, b: float) -> float:
    return a + b


def multiply(a: float, b: float) -> float:
    return a * b


def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        name="add",
        description="Add two numbers.",
        schema=AddInput,
        function=add,
    )

    registry.register(
        name="multiply",
        description="Multiply two numbers.",
        schema=MultiplyInput,
        function=multiply,
    )

    return registry


if __name__ == "__main__":
    registry = create_default_registry()

    print(json.dumps(
        registry.execute("add", {"a": 10, "b": 5}),
        indent=2,
    ))

    print(json.dumps(
        registry.execute("multiply", {"a": 4, "b": 3}),
        indent=2,
    ))

    print(json.dumps(
        registry.execute("add", {"a": "invalid", "b": 5}),
        indent=2,
    ))
