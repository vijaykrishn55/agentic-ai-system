from types import SimpleNamespace
from unittest.mock import Mock

import src.agents.multi_agent as multi_module


def response(content):
    message = SimpleNamespace(content=content)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)]
    )


class FakeMathWorker:
    def run(self, task):
        return "The calculation result is 8."


def setup_supervisor(monkeypatch, responses):
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=Mock(side_effect=responses)
            )
        )
    )

    worker = FakeMathWorker()

    monkeypatch.setattr(
        multi_module,
        "Groq",
        lambda **kwargs: fake_client,
    )
    monkeypatch.setattr(
        multi_module,
        "ReActAgent",
        lambda: worker,
    )

    return multi_module.SupervisorAgent(), fake_client


def test_supervisor_delegates_to_math(monkeypatch):
    supervisor, client = setup_supervisor(
        monkeypatch,
        [
            response('{"worker": "math"}'),
            response("The answer is 8."),
        ],
    )

    result = supervisor.run("Add 4 and 4.")

    assert result["success"] is True
    assert result["state"]["worker"] == "math"
    assert result["state"]["status"] == "completed"
    assert result["answer"] == "The answer is 8."

    events = [
        item["event"]
        for item in result["state"]["history"]
    ]

    assert "delegation" in events
    assert "worker_completed" in events
    assert "synthesis_completed" in events
    assert client.chat.completions.create.call_count == 2


def test_supervisor_rejects_invalid_worker(monkeypatch):
    supervisor, _ = setup_supervisor(
        monkeypatch,
        [response('{"worker": "unknown"}')],
    )

    result = supervisor.run("Do something.")

    assert result["success"] is False
    assert result["state"]["status"] == "failed"
    assert result["state"]["history"][-1]["event"] == (
        "orchestration_error"
    )