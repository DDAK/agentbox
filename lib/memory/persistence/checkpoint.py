"""
Checkpoint Manager for agent session state.

Provides functionality to save and restore agent state for resumable sessions.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .markdown_store import MarkdownMemoryStore


class CheckpointManager:
    """Manages checkpoints for long-running agent sessions."""

    def __init__(self, store: MarkdownMemoryStore):
        """
        Initialize the checkpoint manager.

        Args:
            store: The markdown memory store for persistence
        """
        self.store = store
        self.current_session_id: Optional[str] = None
        self.last_checkpoint_step: int = 0

    def save_checkpoint(
        self,
        session_id: str,
        step: int,
        messages: list[dict],
        task: str = "",
        progress: str = "",
        sandbox_state: Optional[dict] = None,
    ) -> str:
        """
        Save a complete agent state checkpoint.

        Args:
            session_id: Unique session identifier
            step: Current step number
            messages: Conversation messages (will be compressed)
            task: Current task description
            progress: Progress description (markdown list)
            sandbox_state: Optional sandbox state (file manifest, etc.)

        Returns:
            Path to the checkpoint file
        """
        # Compress messages for storage
        messages_compressed = self._compress_messages(messages)

        # Build state dictionary
        state = {
            "started": getattr(self, "_session_start_time", datetime.now().isoformat()),
            "step": step,
            "status": "active",
            "task": task,
            "progress": progress,
            "summary": self._generate_summary(messages),
            "messages_compressed": messages_compressed,
        }

        if sandbox_state:
            state["sandbox_hash"] = sandbox_state.get("hash", "")
            state["working_dir"] = sandbox_state.get("working_dir", "")

        self.current_session_id = session_id
        self.last_checkpoint_step = step

        return self.store.save_session(session_id, state)

    def restore_checkpoint(self, session_id: str) -> Optional[dict]:
        """
        Restore agent state from a checkpoint.

        Args:
            session_id: Session identifier to restore

        Returns:
            Restored state dictionary or None if not found
        """
        state = self.store.load_session(session_id)
        if state:
            self.current_session_id = session_id
            self.last_checkpoint_step = state.get("step", 0)
            self._session_start_time = state.get("started")

            # Decompress messages if available
            if "messages_compressed" in state:
                state["messages"] = self._decompress_messages(state["messages_compressed"])

        return state

    def finalize_session(self, session_id: str, summary: str = "") -> str:
        """
        Mark a session as completed.

        Args:
            session_id: Session identifier
            summary: Final summary of the session

        Returns:
            Path to the finalized session file
        """
        state = self.store.load_session(session_id) or {}
        state["status"] = "completed"
        state["ended"] = datetime.now().isoformat()
        if summary:
            state["summary"] = summary

        return self.store.save_session(session_id, state)

    def list_checkpoints(self, status: Optional[str] = None) -> list[dict]:
        """
        List available checkpoints.

        Args:
            status: Optional filter by status (active, completed, paused)

        Returns:
            List of checkpoint metadata
        """
        sessions = self.store.list_sessions()
        if status:
            sessions = [s for s in sessions if s.get("status") == status]
        return sessions

    def _compress_messages(self, messages: list[dict]) -> str:
        """
        Compress messages for storage.

        Keeps key information while reducing size:
        - User queries
        - Assistant responses (truncated)
        - Tool calls and results (summarized)
        """
        compressed_lines = []

        for msg in messages[-50:]:  # Keep last 50 messages
            if isinstance(msg, dict):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        compressed_lines.append(f"USER: {content[:200]}")
                elif msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        compressed_lines.append(f"ASSISTANT: {content[:200]}")
                elif msg.get("type") == "function_call":
                    compressed_lines.append(f"TOOL_CALL: {msg.get('name', 'unknown')}")
                elif msg.get("type") == "function_call_output":
                    output = msg.get("output", "")
                    if isinstance(output, str):
                        compressed_lines.append(f"TOOL_RESULT: {output[:100]}")

        return "\n".join(compressed_lines)

    def _decompress_messages(self, compressed: str) -> list[dict]:
        """
        Decompress messages from storage format.

        Note: This returns a simplified format since original messages
        can't be fully reconstructed from compressed format.
        """
        messages = []

        for line in compressed.split("\n"):
            if line.startswith("USER: "):
                messages.append({"role": "user", "content": line[6:]})
            elif line.startswith("ASSISTANT: "):
                messages.append({"role": "assistant", "content": line[11:]})
            elif line.startswith("TOOL_CALL: "):
                messages.append({"type": "function_call", "name": line[11:]})
            elif line.startswith("TOOL_RESULT: "):
                messages.append({"type": "function_call_output", "output": line[13:]})

        return messages

    def _generate_summary(self, messages: list[dict]) -> str:
        """
        Generate a brief summary of the conversation.

        This is a simple extraction; can be enhanced with LLM summarization.
        """
        # Extract user queries
        queries = []
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content:
                    queries.append(content[:100])

        if queries:
            return f"User requested: {queries[-1]}"
        return "No summary available"
