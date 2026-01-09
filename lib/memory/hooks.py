"""
Memory Hooks Module - Connect memory system with lifecycle hooks.

This module provides hook handlers that integrate the memory system with
the agent's lifecycle events. By registering these hooks, memory operations
are automatically triggered at the appropriate points in the agent execution.

Hook Events and Memory Operations:
- SessionStart: Initialize memory, load previous session state if resuming
- UserPromptSubmit: Retrieve relevant context for the query
- PostToolUse: Store observations from tool executions
- Stop: Checkpoint session state, consolidate memories

Usage:
    from lib.memory.hooks import MemoryHooks
    from lib.hooks import get_hook_manager

    memory_manager = MemoryManager(storage_path=".agent_memory")
    memory_hooks = MemoryHooks(memory_manager)
    memory_hooks.register_all(get_hook_manager())
"""

import json
from typing import Optional, Any, Dict, List
from datetime import datetime

from ..hooks.types import HookEvent, HookContext, HookResult
from ..hooks.manager import HookManager
from .manager import MemoryManager
from .integration import extract_observation, inject_memories


class MemoryHooks:
    """
    Hook handlers for memory system integration.

    Provides handlers for all lifecycle events that need memory operations.
    Can be registered with a HookManager to enable automatic memory management.
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        checkpoint_interval: int = 100,
        auto_retrieve: bool = True,
        top_k_context: int = 10,
    ):
        """
        Initialize memory hooks.

        Args:
            memory_manager: The MemoryManager instance to use
            checkpoint_interval: Steps between automatic checkpoints
            auto_retrieve: Whether to automatically retrieve context on UserPromptSubmit
            top_k_context: Number of context items to retrieve
        """
        self.memory_manager = memory_manager
        self.checkpoint_interval = checkpoint_interval
        self.auto_retrieve = auto_retrieve
        self.top_k_context = top_k_context

        # Track state for checkpointing
        self._current_task: Optional[str] = None
        self._step_count: int = 0
        self._last_query: Optional[str] = None
        self._retrieved_context: Optional[List[str]] = None

    def register_all(self, hook_manager: HookManager) -> None:
        """
        Register all memory hooks with the given hook manager.

        Args:
            hook_manager: The HookManager to register with
        """
        hook_manager.register(HookEvent.SessionStart, self.on_session_start)
        hook_manager.register(HookEvent.UserPromptSubmit, self.on_user_prompt_submit)
        hook_manager.register(HookEvent.PostToolUse, self.on_post_tool_use)
        hook_manager.register(HookEvent.Stop, self.on_stop)

    def unregister_all(self, hook_manager: HookManager) -> None:
        """
        Unregister all memory hooks from the given hook manager.

        Args:
            hook_manager: The HookManager to unregister from
        """
        hook_manager.unregister(HookEvent.SessionStart, self.on_session_start)
        hook_manager.unregister(HookEvent.UserPromptSubmit, self.on_user_prompt_submit)
        hook_manager.unregister(HookEvent.PostToolUse, self.on_post_tool_use)
        hook_manager.unregister(HookEvent.Stop, self.on_stop)

    def on_session_start(self, context: HookContext) -> HookResult:
        """
        Handle SessionStart event - initialize or resume session.

        On session start:
        - If session_id provided, attempt to restore previous session
        - Otherwise, start a new session
        - Reset step counter

        Args:
            context: The hook context with session information

        Returns:
            HookResult with session metadata
        """
        session_id = context.session_id
        task = context.metadata.get("task", "")
        resume_id = context.metadata.get("resume_id")

        # Start or resume session
        if resume_id:
            # Attempt to restore previous session
            restored_state = self.memory_manager.restore_session(resume_id)
            if restored_state:
                self._step_count = restored_state.get("step", 0)
                self._current_task = restored_state.get("task", "")
                return HookResult(
                    metadata={
                        "session_restored": True,
                        "session_id": resume_id,
                        "restored_step": self._step_count,
                        "messages": restored_state.get("messages", []),
                    }
                )

        # Start new session
        new_session_id = self.memory_manager.start_session(task=task)
        self._step_count = 0
        self._current_task = task

        return HookResult(
            metadata={
                "session_started": True,
                "session_id": new_session_id,
            }
        )

    def on_user_prompt_submit(self, context: HookContext) -> HookResult:
        """
        Handle UserPromptSubmit event - retrieve relevant context.

        Before sending query to LLM:
        - Retrieve relevant memories for the query
        - Store context for potential use by the agent

        Args:
            context: The hook context with user query

        Returns:
            HookResult with retrieved context in metadata
        """
        query = context.query or ""
        self._last_query = query

        # Update task if this is the first query or a significant new one
        if not self._current_task and query:
            self._current_task = query[:200]

        if not self.auto_retrieve:
            return HookResult()

        try:
            # Retrieve relevant context
            relevant_memories = self.memory_manager.retrieve_context(
                query=query,
                top_k=self.top_k_context,
            )

            self._retrieved_context = relevant_memories

            return HookResult(
                metadata={
                    "retrieved_context": relevant_memories,
                    "context_count": len(relevant_memories),
                }
            )
        except Exception as e:
            return HookResult(
                metadata={
                    "retrieval_error": str(e),
                }
            )

    def on_post_tool_use(self, context: HookContext) -> HookResult:
        """
        Handle PostToolUse event - store observation in memory.

        After each tool execution:
        - Extract observation from tool result
        - Store in short-term memory
        - Increment step counter

        Args:
            context: The hook context with tool info and result

        Returns:
            HookResult with memory storage metadata
        """
        tool_name = context.tool_name or "unknown"
        result = context.result
        arguments = context.arguments or {}

        self._step_count += 1

        try:
            # Convert result to string for observation extraction
            if isinstance(result, dict):
                result_str = json.dumps(result)
            else:
                result_str = str(result) if result else ""

            # Convert arguments to string
            args_str = json.dumps(arguments) if arguments else ""

            # Extract and store observation
            observation = extract_observation(tool_name, result_str, args_str)
            memory_id = self.memory_manager.add_memory(
                observation,
                memory_type="observation",
                tool_name=tool_name,
            )

            return HookResult(
                metadata={
                    "observation_stored": True,
                    "memory_id": memory_id,
                    "step": self._step_count,
                }
            )
        except Exception as e:
            return HookResult(
                metadata={
                    "observation_error": str(e),
                    "step": self._step_count,
                }
            )

    def on_stop(self, context: HookContext) -> HookResult:
        """
        Handle Stop event - checkpoint and consolidate.

        When agent finishes:
        - Save checkpoint if interval reached or if important
        - End or pause session based on completion status

        Args:
            context: The hook context with messages and metadata

        Returns:
            HookResult with checkpoint metadata
        """
        messages = context.messages or []
        metadata = context.metadata or {}

        total_steps = metadata.get("total_steps", self._step_count)
        is_complete = metadata.get("is_complete", False)

        checkpoint_saved = False
        session_ended = False

        try:
            # Save checkpoint
            if self._should_checkpoint(total_steps):
                self.memory_manager.checkpoint(
                    step=total_steps,
                    messages=messages,
                    task=self._current_task or "",
                    progress=f"Step {total_steps}",
                )
                checkpoint_saved = True

            # End session if marked complete
            if is_complete:
                summary = self._generate_session_summary()
                self.memory_manager.end_session(summary=summary)
                session_ended = True

            return HookResult(
                metadata={
                    "checkpoint_saved": checkpoint_saved,
                    "session_ended": session_ended,
                    "final_step": total_steps,
                }
            )
        except Exception as e:
            return HookResult(
                metadata={
                    "checkpoint_error": str(e),
                }
            )

    def _should_checkpoint(self, step: int) -> bool:
        """
        Determine if a checkpoint should be saved.

        Args:
            step: Current step number

        Returns:
            True if checkpoint should be saved
        """
        if step <= 0:
            return False

        # Always checkpoint on significant steps
        if step % self.checkpoint_interval == 0:
            return True

        # Checkpoint on first few steps for safety
        if step <= 5:
            return True

        return False

    def _generate_session_summary(self) -> str:
        """
        Generate a summary of the current session.

        Returns:
            Summary string
        """
        summary_parts = []

        if self._current_task:
            summary_parts.append(f"Task: {self._current_task}")

        summary_parts.append(f"Steps completed: {self._step_count}")

        # Get memory stats
        stats = self.memory_manager.get_stats()
        summary_parts.append(f"Observations stored: {stats.get('short_term_count', 0)}")
        summary_parts.append(f"Knowledge items: {stats.get('long_term_count', 0)}")

        return " | ".join(summary_parts)

    def get_retrieved_context(self) -> Optional[List[str]]:
        """
        Get the context retrieved during the last UserPromptSubmit.

        Returns:
            List of context strings, or None if not retrieved
        """
        return self._retrieved_context

    def get_current_step(self) -> int:
        """
        Get the current step count.

        Returns:
            Current step number
        """
        return self._step_count

    def reset(self) -> None:
        """Reset the memory hooks state."""
        self._current_task = None
        self._step_count = 0
        self._last_query = None
        self._retrieved_context = None


def create_memory_hooks(
    storage_path: str = ".agent_memory",
    llm_client: Optional[Any] = None,
    checkpoint_interval: int = 100,
    auto_retrieve: bool = True,
    top_k_context: int = 10,
) -> MemoryHooks:
    """
    Factory function to create MemoryHooks with a new MemoryManager.

    Args:
        storage_path: Path for memory storage
        llm_client: Optional LLM client for summarization
        checkpoint_interval: Steps between checkpoints
        auto_retrieve: Whether to auto-retrieve context
        top_k_context: Number of context items to retrieve

    Returns:
        Configured MemoryHooks instance
    """
    memory_manager = MemoryManager(
        storage_path=storage_path,
        llm_client=llm_client,
    )

    return MemoryHooks(
        memory_manager=memory_manager,
        checkpoint_interval=checkpoint_interval,
        auto_retrieve=auto_retrieve,
        top_k_context=top_k_context,
    )
