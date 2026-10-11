from src.tools.registry import create_default_registry


def test_registered_tool_schemas():
    registry = create_default_registry()
    schemas = registry.get_schemas()

    assert len(schemas) == 2
    assert all(item["type"] == "function" for item in schemas)

    names = {
        item["function"]["name"]
        for item in schemas
    }
    assert names == {"add", "multiply"}


def test_add_schema_requires_both_arguments():
    registry = create_default_registry()
    schemas = registry.get_schemas()

    add_schema = next(
        item["function"]["parameters"]
        for item in schemas
        if item["function"]["name"] == "add"
    )

    assert set(add_schema["required"]) == {"a", "b"}
    assert add_schema["additionalProperties"] is False