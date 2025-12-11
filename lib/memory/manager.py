"""
Memory Manager - Central orchestrator for the memory system.

Coordinates short-term and long-term memory, provides unified interface
for memory operations, and handles consolidation and retrieval.
"""

import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from .types.short_term import ShortTermMemory
from .types.long_term import LongTermMemory
from .session import SessionManager
from .persistence.markdown_store import MarkdownMemoryStore


class MemoryManager:
    """
    Central memory manager coordinating all memory operations.

    Provides:
    - Unified interface for adding and retrieving memories
    - Automatic consolidation from short-term to long-term
    - Session management with checkpointing
    - Thread-safe operations
    """

    def __init__(
        self,
        storage_path: str = ".agent_memory",
        llm_client: Optional[Any] = None,
        summary_model: str = "gpt-4o-mini",
        short_term_capacity: int = 100,
        consolidation_threshold: int = 50,
    ):
        """
        Initialize the memory manager.

        Args:
            storage_path: Root directory for memory storage
            llm_client: Optional LLM client for summarization
            summary_model: Model to use for summarization
            short_term_capacity: Maximum items in short-term memory
            consolidation_threshold: Trigger consolidation at this count
        """
        self.storage_path = Path(storage_path)
        self.llm_client = llm_client
        self.summary_model = summary_model
        self.consolidation_threshold = consolidation_threshold

        # Initialize lock for thread safety
        self._lock = threading.RLock()

        # Initialize memory components
        self.store = MarkdownMemoryStore(storage_path)
        self.short_term = ShortTermMemory(
            capacity=short_term_capacity,
            storage_path=storage_path,
        )
        self.long_term = LongTermMemory(storage_path=storage_path)
        self.session_manager = SessionManager(storage_path)

    def add_memory(
        self,
        content: str,
        memory_type: str = "observation",
        tool_name: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Add a memory to the appropriate store.

        Args:
            content: The memory content
            memory_type: Type of memory (observation, fact, pattern, learning)
            tool_name: Optional tool that generated this memory
            **kwargs: Additional metadata

        Returns:
            The memory ID
        """
        with self._lock:
            if memory_type == "observation":
                memory_id = self.short_term.add(content, tool_name=tool_name, **kwargs)

                # Check if consolidation is needed
                if len(self.short_term) >= self.consolidation_threshold:
                    self._consolidate()

            elif memory_type == "fact":
                memory_id = self.long_term.add_fact(content)

            elif memory_type == "pattern":
                memory_id = self.long_term.add_pattern(content)

            elif memory_type == "learning":
                memory_id = self.long_term.add_learning(content)

            elif memory_type == "api_info":
                memory_id = self.long_term.add_api_info(content)

            else:
                # Default to observation
                memory_id = self.short_term.add(content, tool_name=tool_name, **kwargs)

            return memory_id

    def retrieve_context(
        self,
        query: str,
        top_k: int = 10,
        include_recent: bool = True,
        include_knowledge: bool = True,
    ) -> list[str]:
        """
        Retrieve relevant context for a query.

        Args:
            query: The query to find relevant context for
            top_k: Maximum number of items to return
            include_recent: Include recent observations
            include_knowledge: Include long-term knowledge

        Returns:
            List of relevant memory strings
        """
        with self._lock:
            results = []

            # Get recent observations
            if include_recent:
                recent = self.short_term.get_recent_contents(limit=5)
                results.extend([f"[Recent] {r}" for r in recent])

            # Search long-term memory
            if include_knowledge:
                knowledge = self.long_term.retrieve(query, top_k=top_k)
                results.extend(knowledge)

            # If we have too many results, prioritize by relevance
            if len(results) > top_k:
                # Simple deduplication and truncation
                seen = set()
                unique = []
                for r in results:
                    if r not in seen:
                        seen.add(r)
                        unique.append(r)
                results = unique[:top_k]

            return results

    def checkpoint(
        self,
        step: int,
        messages: list[dict],
        task: str = "",
        progress: str = "",
    ) -> str:
        """
        Save a checkpoint of the current state.

        Args:
            step: Current step number
            messages: Conversation messages
            task: Current task description
            progress: Progress description

        Returns:
            Path to the checkpoint file
        """
        with self._lock:
            return self.session_manager.checkpoint(
                messages=messages,
                task=task,
                progress=progress,
            )

    def start_session(self, resume_id: Optional[str] = None, task: str = "") -> str:
        """
        Start a new session or resume an existing one.

        Args:
            resume_id: Optional session ID to resume
            task: Task description for new sessions

        Returns:
            The session ID
        """
        with self._lock:
            return self.session_manager.start_session(resume_id=resume_id, task=task)

    def end_session(self, summary: str = "") -> Optional[str]:
        """
        End the current session.

        Args:
            summary: Final summary of the session

        Returns:
            Path to the finalized session file
        """
        with self._lock:
            return self.session_manager.end_session(summary=summary)

    def restore_session(self, session_id: str) -> Optional[dict]:
        """
        Restore a session from checkpoint.

        Args:
            session_id: Session identifier

        Returns:
            Session state dictionary or None
        """
        with self._lock:
            return self.session_manager.restore(session_id)

    def list_sessions(self, status: Optional[str] = None) -> list[dict]:
        """
        List available sessions.

        Args:
            status: Optional status filter

        Returns:
            List of session metadata
        """
        return self.session_manager.list_sessions(status=status)

    def _consolidate(self):
        """
        Consolidate short-term memories into long-term.

        This is called automatically when short-term memory reaches threshold.
        Uses LLM to synthesize patterns if available.
        """
        recent = self.short_term.get_recent(limit=self.consolidation_threshold)
        if not recent:
            return

        # Extract content
        contents = [m.get("content", "") for m in recent]

        # Generate summary
        summary = self._summarize_with_llm(contents)

        # Store as pattern in long-term memory
        self.long_term.add(
            summary,
            category="Patterns",
            source="consolidation",
            timestamp=datetime.now().isoformat(),
        )

    def _summarize_with_llm(self, memories: list[str]) -> str:
        """
        Use LLM to synthesize memories into insight.

        Args:
            memories: List of memory strings to summarize

        Returns:
            Synthesized summary string
        """
        if not self.llm_client:
            # Fallback: simple concatenation with truncation
            return "Summary: " + " | ".join(memories)[:500]

        prompt = f"""Synthesize these observations into a concise insight:

{chr(10).join(f'- {m}' for m in memories[:20])}

Provide a 1-2 sentence summary capturing the key information and any patterns noticed."""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.summary_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback on error
            return f"Summary: {' | '.join(memories[:5])[:500]}"

    def reflect(self, topic: Optional[str] = None) -> str:
        """
        Generate a reflection on accumulated memories.

        Args:
            topic: Optional topic to focus reflection on

        Returns:
            Reflection string
        """
        with self._lock:
            # Gather memories
            recent = self.short_term.get_recent_contents(limit=20)
            knowledge = self.long_term.get_all()

            if topic:
                # Filter to relevant memories
                relevant_knowledge = self.long_term.retrieve(topic, top_k=10)
            else:
                # Use all knowledge categories
                relevant_knowledge = []
                for category, items in knowledge.items():
                    relevant_knowledge.extend([f"[{category}] {item}" for item in items[:5]])

            all_memories = recent + relevant_knowledge

            return self._summarize_with_llm(all_memories)

    def clear_short_term(self):
        """Clear short-term memory."""
        with self._lock:
            self.short_term.clear()

    def clear_all(self):
        """Clear all memories (use with caution)."""
        with self._lock:
            self.short_term.clear()
            self.long_term.clear()

    @property
    def current_session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self.session_manager.current_session_id

    @property
    def current_step(self) -> int:
        """Get the current step number."""
        return self.session_manager.current_step

    def get_stats(self) -> dict:
        """
        Get memory statistics.

        Returns:
            Dictionary with memory stats
        """
        return {
            "short_term_count": len(self.short_term),
            "long_term_count": len(self.long_term),
            "current_session": self.session_manager.current_session_id,
            "current_step": self.session_manager.current_step,
            "storage_path": str(self.storage_path),
        }
