
from pathlib import Path
from typing import Any
from uuid import uuid4

import chromadb


class SemanticMemoryStore:
    """Persistent long-term memory with semantic search."""

    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "agent_long_term_memory",
    ) -> None:
        Path(persist_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_memory(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if not text.strip():
            raise ValueError("Memory text cannot be empty.")

        memory_id = uuid4().hex

        safe_metadata = {
            key: value
            for key, value in (metadata or {}).items()
            if isinstance(value, (str, int, float, bool))
            and value is not None
        }

        if not safe_metadata:
            safe_metadata = {"type": "general"}

        self.collection.add(
            ids=[memory_id],
            documents=[text],
            metadatas=[safe_metadata],
        )

        return memory_id

    def search(
        self,
        query: str,
        n_results: int = 3,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        if n_results < 1:
            raise ValueError("n_results must be at least 1.")

        total = self.collection.count()

        if total == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, total),
            include=["documents", "metadatas", "distances"],
        )

        memories = []

        for i, document in enumerate(results["documents"][0]):
            memories.append({
                "id": results["ids"][0][i],
                "text": document,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        return memories

    def count(self) -> int:
        return self.collection.count()


if __name__ == "__main__":
    store = SemanticMemoryStore()

    memory_id = store.add_memory(
        "The user prefers Python for backend development.",
        metadata={"category": "preference"},
    )

    print(f"Saved memory: {memory_id}")

    results = store.search(
        "Which programming language does the user prefer for server-side coding?"
    )

    for memory in results:
        print(memory)
