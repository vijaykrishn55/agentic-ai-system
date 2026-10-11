
from src.memory.manager import MemoryManager


def test_working_memory():
    memory = MemoryManager()

    memory.set_working_memory("topic", "Python")
    assert memory.get_working_memory("topic") == "Python"


def test_conversation_history():
    memory = MemoryManager()

    memory.add_message("user", "Hello")
    memory.add_message("assistant", "Hi!")

    context = memory.get_context()

    assert len(context) == 2
    assert context[0]["content"] == "Hello"


def test_sliding_window():
    memory = MemoryManager(window_size=2)

    memory.add_message("user", "First")
    memory.add_message("assistant", "Second")
    memory.add_message("user", "Third")

    assert len(memory.history) == 2
    assert memory.history[0]["content"] == "Second"
    assert "First" in memory.summary


def test_summary_buffer_is_bounded():
    memory = MemoryManager(
        window_size=2,
        summary_max_chars=50,
    )

    for i in range(10):
        memory.add_message("user", f"Message number {i}")

    assert len(memory.summary) <= 50


def test_clear_memory():
    memory = MemoryManager()
    memory.set_working_memory("task", "calculate")
    memory.add_message("user", "Hello")

    memory.clear()

    assert memory.working_memory == {}
    assert memory.history == []
    assert memory.summary == ""
