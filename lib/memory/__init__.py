"""
Memory System for Coding Agents

Provides persistent memory capabilities for long-running agents:
- Short-term memory (recent observations)
- Long-term memory (accumulated knowledge)
- Session management (checkpoints and resume)

All storage uses markdown files for human-readable, agent-friendly persistence.
"""

from .manager import MemoryManager
from .session import SessionManager

__all__ = ["MemoryManager", "SessionManager"]
