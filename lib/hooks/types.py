"""
Hook types and data structures for the lifecycle hooks system.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional, Dict
from datetime import datetime


class HookEvent(Enum):
    """
    Lifecycle hook events that can be triggered during agent execution.

    Each event corresponds to a specific point in the execution flow:
    - SessionStart: When the CLI boots or a session is cleared
    - UserPromptSubmit: After user hits Enter, before prompt is sent to LLM
    - PreToolUse: Before agent executes a tool (e.g., bash, write_file)
    - PostToolUse: Immediately after a tool finishes
    - Stop: When agent finishes its turn and prepares to hand control back
    """
    SessionStart = "session_start"
    UserPromptSubmit = "user_prompt_submit"
    PreToolUse = "pre_tool_use"
    PostToolUse = "post_tool_use"
    Stop = "stop"


@dataclass
class HookContext:
    """
    Context object passed to hook handlers.

    Contains information about the current state of the agent and
    the specific event that triggered the hook.

    Attributes:
        event: The hook event type
        timestamp: When the event occurred
        session_id: Current session ID (if available)
        tool_name: Name of the tool (for PreToolUse/PostToolUse)
        arguments: Tool arguments (for PreToolUse/PostToolUse)
        result: Tool execution result (for PostToolUse)
        query: User query (for UserPromptSubmit)
        messages: Conversation history
        metadata: Additional event-specific data
    """
    event: HookEvent
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None
    tool_name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    query: Optional[str] = None
    messages: Optional[list] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            'event': self.event.value,
            'timestamp': self.timestamp.isoformat(),
            'session_id': self.session_id,
            'tool_name': self.tool_name,
            'arguments': self.arguments,
            'result': self.result,
            'query': self.query,
            'messages': self.messages,
            'metadata': self.metadata,
        }


@dataclass
class HookResult:
    """
    Result returned by a hook handler.

    Hooks can optionally return a HookResult to modify agent behavior.

    Attributes:
        block: If True, blocks the current action (for PreToolUse)
        reason: Explanation for blocking or modification
        modified_arguments: Modified tool arguments (for PreToolUse)
        modified_result: Modified tool result (for PostToolUse)
        skip_remaining: If True, skip remaining hooks for this event
        metadata: Additional result data
    """
    block: bool = False
    reason: Optional[str] = None
    modified_arguments: Optional[Dict[str, Any]] = None
    modified_result: Optional[Any] = None
    skip_remaining: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'block': self.block,
            'reason': self.reason,
            'modified_arguments': self.modified_arguments,
            'modified_result': self.modified_result,
            'skip_remaining': self.skip_remaining,
            'metadata': self.metadata,
        }

    @classmethod
    def allow(cls) -> 'HookResult':
        """Create a result that allows the action to proceed."""
        return cls(block=False)

    @classmethod
    def deny(cls, reason: str) -> 'HookResult':
        """Create a result that blocks the action."""
        return cls(block=True, reason=reason)
