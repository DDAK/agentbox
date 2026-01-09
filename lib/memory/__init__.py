"""
Memory System for Coding Agents

Provides persistent memory capabilities for long-running agents:
- Short-term memory (recent observations)
- Long-term memory (accumulated knowledge)
- Session management (checkpoints and resume)
- Hook-based memory integration

All storage uses markdown files for human-readable, agent-friendly persistence.

Hook Integration:
    The MemoryHooks class provides automatic memory management through lifecycle hooks.
    Register it with your HookManager to enable:
    - SessionStart: Initialize/restore sessions
    - UserPromptSubmit: Retrieve relevant context
    - PostToolUse: Store observations
    - Stop: Checkpoint session state

Example:
    from lib.memory import MemoryManager, MemoryHooks
    from lib.hooks import get_hook_manager

    memory_manager = MemoryManager(storage_path=".agent_memory")
    memory_hooks = MemoryHooks(memory_manager)
    memory_hooks.register_all(get_hook_manager())
"""

from .manager import MemoryManager
from .session import SessionManager
from .hooks import MemoryHooks, create_memory_hooks

__all__ = ["MemoryManager", "SessionManager", "MemoryHooks", "create_memory_hooks"]
