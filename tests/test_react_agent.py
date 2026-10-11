import json
from types import SimpleNamespace
from unittest.mock import Mock

import src.agents.react_agent as react_module


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self, exclude_none=True):
        calls = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in self.tool_calls
        ]

        result = {
            "role": "assistant",
            "content": self.content,
        }

        if calls:
            result["tool_calls"] = calls

        return result


def response(content=None, tool_calls=None):
    message = FakeMessage(content, tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)]
    )


def tool_call(name, arguments):
    return SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        ),
    )


class FakeSemanticMemory:
    def __init__(self):
        self.saved = []

    def search(self, query, n_results=3):
        return []

    def add_memory(self, text, metadata=None):
        self.saved.append(text)
        return "fake-id"


def setup_agent(monkeypatch, responses, max_iterations=5):
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=Mock(side_effect=responses)
            )
        )
    )

    monkeypatch.setattr(
        react_module,
        "Groq",
        lambda **kwargs: fake_client,
    )

    fake_memory = FakeSemanticMemory()

    monkeypatch.setattr(
        react_module,
        "SemanticMemoryStore",
        lambda: fake_memory,
    )

    agent = react_module.ReActAgent(
        max_iterations=max_iterations,
    )

    return agent, fake_client, fake_memory


def test_react_executes_tool_and_finishes(monkeypatch):
    agent, client, _ = setup_agent(
        monkeypatch,
        [
            response(tool_calls=[
                tool_call("add", '{"a": 2, "b": 2}')
            ]),
            response("The result is 4."),
        ],
    )

    result = agent.run("Add 2 and 2.")

    assert result["success"] is True
    assert "4" in result["answer"]

    observations = [
        item for item in result["history"]
        if item["event"] == "tool_observation"
    ]

    assert len(observations) == 1
    assert observations[0]["observation"]["result"] == 4.0
    assert client.chat.completions.create.call_count == 2


def test_react_handles_invalid_tool_json(monkeypatch):
    agent, _, _ = setup_agent(
        monkeypatch,
        [
            response(tool_calls=[
                tool_call("add", "{invalid-json")
            ]),
            response("The tool arguments were invalid."),
        ],
    )

    result = agent.run("Add two numbers.")

    assert result["success"] is True

    observation = next(
        item for item in result["history"]
        if item["event"] == "tool_observation"
    )

    assert observation["observation"]["error"]["type"] == "invalid_json"


def test_react_enforces_max_iterations(monkeypatch):
    agent, client, _ = setup_agent(
        monkeypatch,
        [
            response(tool_calls=[
                tool_call("add", '{"a": 2, "b": 2}')
            ]),
        ],
        max_iterations=1,
    )

    result = agent.run("Add two and two.")

    assert result["success"] is False
    assert "Maximum ReAct iterations" in result["error"]
    assert client.chat.completions.create.call_count == 1


def test_react_saves_successful_conversation(monkeypatch):
    agent, _, semantic_memory = setup_agent(
        monkeypatch,
        [response("Hello!")],
    )

    result = agent.run("Hi")

    assert result["success"] is True
    assert "User message: Hi" in semantic_memory.saved
    assert any(
        "Hello!" in item for item in semantic_memory.saved
    )