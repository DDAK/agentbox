"""
Session Manager for agent sessions.

Manages agent session lifecycle including:
- Starting new sessions
- Resuming existing sessions
- Checkpointing state
- Listing and managing sessions
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from .persistence.markdown_store import MarkdownMemoryStore
from .persistence.checkpoint import CheckpointManager


class SessionManager:
    """Manages agent session lifecycle using markdown checkpoints."""

    def __init__(self, storage_path: str):
        """
        Initialize the session manager.

        Args:
            storage_path: Root directory for memory storage
        """
        self.storage_path = Path(storage_path)
        self.store = MarkdownMemoryStore(storage_path)
        self.checkpoint_manager = CheckpointManager(self.store)

        self.current_session_id: Optional[str] = None
        self.session_start_time: Optional[str] = None
        self.current_step: int = 0

    def start_session(self, resume_id: Optional[str] = None, task: str = "") -> str:
        """
        Start a new session or resume an existing one.

        Args:
            resume_id: Optional session ID to resume
            task: Task description for new sessions

        Returns:
            The session ID
        """
        if resume_id:
            # Attempt to resume existing session
            state = self.checkpoint_manager.restore_checkpoint(resume_id)
            if state:
                self.current_session_id = resume_id
                self.session_start_time = state.get("started")
                self.current_step = state.get("step", 0)
                return resume_id
            # Fall through to create new session if resume fails

        # Create new session
        self.current_session_id = self._generate_session_id()
        self.session_start_time = datetime.now().isoformat()
        self.current_step = 0

        # Save initial checkpoint
        self.checkpoint_manager.save_checkpoint(
            session_id=self.current_session_id,
            step=0,
            messages=[],
            task=task,
            progress="- Session started",
        )

        return self.current_session_id

    def checkpoint(
        self,
        messages: List[dict],
        task: str = "",
        progress: str = "",
        sandbox_state: Optional[dict] = None,
    ) -> str:
        """
        Save a checkpoint at the current step.

        Args:
            messages: Current conversation messages
            task: Current task description
            progress: Progress description
            sandbox_state: Optional sandbox state

        Returns:
            Path to the checkpoint file
        """
        if not self.current_session_id:
            self.start_session(task=task)

        self.current_step += 1

        return self.checkpoint_manager.save_checkpoint(
            session_id=self.current_session_id,
            step=self.current_step,
            messages=messages,
            task=task,
            progress=progress,
            sandbox_state=sandbox_state,
        )

    def end_session(self, summary: str = "") -> Optional[str]:
        """
        Finalize and close the current session.

        Args:
            summary: Final summary of the session

        Returns:
            Path to the finalized session file, or None if no active session
        """
        if not self.current_session_id:
            return None

        path = self.checkpoint_manager.finalize_session(
            self.current_session_id,
            summary=summary,
        )

        self.current_session_id = None
        self.session_start_time = None
        self.current_step = 0

        return path

    def restore(self, session_id: str) -> Optional[dict]:
        """
        Restore session state from a checkpoint.

        Args:
            session_id: Session identifier to restore

        Returns:
            Session state dictionary or None if not found
        """
        state = self.checkpoint_manager.restore_checkpoint(session_id)
        if state:
            self.current_session_id = session_id
            self.session_start_time = state.get("started")
            self.current_step = state.get("step", 0)
        return state

    def list_sessions(self, status: Optional[str] = None) -> List[dict]:
        """
        List all available sessions.

        Args:
            status: Optional filter by status (active, completed, paused)

        Returns:
            List of session metadata dictionaries
        """
        return self.checkpoint_manager.list_checkpoints(status=status)

    def get_session_info(self, session_id: Optional[str] = None) -> Optional[dict]:
        """
        Get information about a specific session.

        Args:
            session_id: Session ID (defaults to current session)

        Returns:
            Session metadata dictionary or None
        """
        sid = session_id or self.current_session_id
        if not sid:
            return None

        return self.store.load_session(sid)

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and its checkpoint.

        Args:
            session_id: Session identifier to delete

        Returns:
            True if deleted, False if not found
        """
        if session_id == self.current_session_id:
            self.current_session_id = None
            self.session_start_time = None
            self.current_step = 0

        return self.store.delete_session(session_id)

    def pause_session(self) -> Optional[str]:
        """
        Pause the current session (mark as paused).

        Returns:
            Path to the paused session file, or None if no active session
        """
        if not self.current_session_id:
            return None

        state = self.store.load_session(self.current_session_id) or {}
        state["status"] = "paused"
        return self.store.save_session(self.current_session_id, state)

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique = str(uuid.uuid4())[:6]
        return f"session_{timestamp}_{unique}"

    @property
    def is_active(self) -> bool:
        """Check if there's an active session."""
        return self.current_session_id is not None

    @property
    def session_duration(self) -> Optional[float]:
        """Get the duration of the current session in seconds."""
        if not self.session_start_time:
            return None

        start = datetime.fromisoformat(self.session_start_time)
        return (datetime.now() - start).total_seconds()
