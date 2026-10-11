
from typing import Any


class MemoryManager:
    """Manage working memory, recent history, and a summary buffer."""

    def __init__(
        self,
        window_size: int = 6,
        summary_max_chars: int = 2000,
    ) -> None:
        if window_size < 2 or summary_max_chars < 1:
            raise ValueError("Invalid memory limits.")

        self.window_size = window_size
        self.summary_max_chars = summary_max_chars

        self.working_memory: dict[str, Any] = {}
        self.history: list[dict[str, str]] = []
        self.summary = ""

    def set_working_memory(self, key: str, value: Any) -> None:
        self.working_memory[key] = value

    def get_working_memory(self, key: str, default=None):
        return self.working_memory.get(key, default)

    def add_message(self, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("Role must be user or assistant.")

        self.history.append({
            "role": role,
            "content": content,
        })

        self._prune_history()

    def _prune_history(self) -> None:
        while len(self.history) > self.window_size:
            old_message = self.history.pop(0)

            entry = f"{old_message['role']}: {old_message['content']}"
            combined = f"{self.summary}\n{entry}".strip()

            # Keep the summary bounded as well.
            self.summary = combined[-self.summary_max_chars:]

    def get_context(self) -> list[dict[str, str]]:
        """Build the conversation context supplied to an LLM."""
        context = []

        if self.summary:
            context.append({
                "role": "system",
                "content": (
                    "Summary of earlier conversation. Use this as background, "
                    "not as a replacement for the user's current request:\n"
                    + self.summary
                ),
            })

        context.extend(self.history)
        return context

    def clear(self) -> None:
        self.working_memory.clear()
        self.history.clear()
        self.summary = ""
