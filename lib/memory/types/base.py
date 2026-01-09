"""
Base memory class for the memory system.

Defines the interface that all memory types must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
import uuid


class BaseMemory(ABC):
    """Abstract base class for memory implementations."""

    @abstractmethod
    def add(self, content: str, **kwargs) -> str:
        """
        Add a memory.

        Args:
            content: The memory content
            **kwargs: Additional metadata

        Returns:
            The memory ID
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve relevant memories.

        Args:
            query: Search query
            top_k: Maximum number of results

        Returns:
            List of memory strings
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all memories."""
        pass

    @staticmethod
    def _generate_id() -> str:
        """Generate a unique memory ID."""
        return str(uuid.uuid4())[:8]
