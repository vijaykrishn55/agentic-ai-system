
import pytest

from src.memory.semantic_store import SemanticMemoryStore


@pytest.fixture
def store(tmp_path):
    return SemanticMemoryStore(
        persist_directory=str(tmp_path / "chroma")
    )


def test_add_and_count_memory(store):
    store.add_memory("The user prefers Python.")

    assert store.count() == 1


def test_semantic_search(store):
    store.add_memory(
        "The user prefers Python for backend development."
    )
    store.add_memory(
        "The user's favorite food is pizza."
    )

    results = store.search(
        "Which programming language is preferred for server-side coding?",
        n_results=1,
    )

    assert len(results) == 1
    assert "Python" in results[0]["text"]


def test_empty_search_returns_no_results(store):
    assert store.search("Anything about my preferences?") == []


def test_empty_memory_is_rejected(store):
    with pytest.raises(ValueError):
        store.add_memory("   ")


def test_memory_persists_after_reopening(tmp_path):
    path = str(tmp_path / "persistent_chroma")

    first_store = SemanticMemoryStore(persist_directory=path)
    first_store.add_memory("The user is learning AI engineering.")

    second_store = SemanticMemoryStore(persist_directory=path)

    assert second_store.count() == 1
    assert "AI engineering" in second_store.search(
        "What field is the user studying?"
    )[0]["text"]
