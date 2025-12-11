"""
Short-term memory implementation.

Provides a FIFO buffer for recent observations with optional persistence.
"""

from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

from .base import BaseMemory


class ShortTermMemory(BaseMemory):
    """
    Short-term memory using a FIFO buffer.

    Stores recent observations with configurable capacity.
    Optionally persists to markdown file.
    """

    def __init__(
        self,
        capacity: int = 100,
        storage_path: Optional[str] = None,
    ):
        """
        Initialize short-term memory.

        Args:
            capacity: Maximum number of items to store
            storage_path: Optional path for markdown persistence
        """
        self.capacity = capacity
        self.storage_path = Path(storage_path) if storage_path else None
        self._buffer: deque = deque(maxlen=capacity)

        if self.storage_path:
            self._load_from_markdown()

    def add(self, content: str, tool_name: Optional[str] = None, **kwargs) -> str:
        """
        Add an observation to short-term memory.

        Args:
            content: The observation content
            tool_name: Optional tool that generated this observation
            **kwargs: Additional metadata

        Returns:
            The memory ID
        """
        memory_id = self._generate_id()
        timestamp = datetime.now().isoformat()

        memory = {
            "id": memory_id,
            "content": content,
            "timestamp": timestamp,
            "tool_name": tool_name,
            **kwargs,
        }

        self._buffer.append(memory)

        if self.storage_path:
            self._persist_to_markdown()

        return memory_id

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """
        Retrieve relevant observations by keyword search.

        Args:
            query: Search query
            top_k: Maximum number of results

        Returns:
            List of observation strings
        """
        query_words = set(query.lower().split())
        scored = []

        for memory in self._buffer:
            content = memory.get("content", "")
            content_words = set(content.lower().split())
            score = len(query_words & content_words)
            if score > 0:
                scored.append((score, content))

        # Sort by relevance
        scored.sort(reverse=True)
        return [content for _, content in scored[:top_k]]

    def get_recent(self, limit: int = 10) -> list[dict]:
        """
        Get the most recent observations.

        Args:
            limit: Maximum number of observations

        Returns:
            List of memory dictionaries (most recent first)
        """
        recent = list(self._buffer)[-limit:]
        recent.reverse()
        return recent

    def get_recent_contents(self, limit: int = 10) -> list[str]:
        """
        Get content of the most recent observations.

        Args:
            limit: Maximum number of observations

        Returns:
            List of content strings (most recent first)
        """
        return [m.get("content", "") for m in self.get_recent(limit)]

    def clear(self) -> None:
        """Clear all short-term memories."""
        self._buffer.clear()
        if self.storage_path:
            self._persist_to_markdown()

    def _load_from_markdown(self):
        """Load memories from markdown file."""
        buffer_path = self.storage_path / "short_term" / "buffer.md"
        if not buffer_path.exists():
            return

        content = buffer_path.read_text()

        # Parse observations from markdown
        import re
        for match in re.finditer(
            r"- \*\*([^*]+)\*\*(?:\s*\[([^\]]+)\])?: (.+)",
            content
        ):
            timestamp, tool_name, obs_content = match.groups()
            self._buffer.append({
                "id": self._generate_id(),
                "content": obs_content,
                "timestamp": timestamp,
                "tool_name": tool_name,
            })

    def _persist_to_markdown(self):
        """Persist memories to markdown file."""
        buffer_path = self.storage_path / "short_term" / "buffer.md"
        buffer_path.parent.mkdir(parents=True, exist_ok=True)

        lines = ["# Short-Term Memory Buffer\n"]
        lines.append("<!-- Recent observations (FIFO, most recent first) -->\n")

        # Write observations (most recent first)
        for memory in reversed(list(self._buffer)):
            timestamp = memory.get("timestamp", "")
            tool_name = memory.get("tool_name", "")
            content = memory.get("content", "")

            tool_info = f" [{tool_name}]" if tool_name else ""
            # Truncate long content
            display_content = content[:200] + ("..." if len(content) > 200 else "")
            lines.append(f"- **{timestamp}**{tool_info}: {display_content}")

        lines.append(f"\n---\n*Last updated: {datetime.now().isoformat()}*")

        buffer_path.write_text("\n".join(lines))

    def __len__(self) -> int:
        """Return the number of items in memory."""
        return len(self._buffer)

    def __iter__(self):
        """Iterate over memories (oldest first)."""
        return iter(self._buffer)
