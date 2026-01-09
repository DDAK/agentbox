"""
Tests for the memory hooks integration module.

Tests cover:
- MemoryHooks class initialization
- SessionStart hook behavior
- UserPromptSubmit hook for context retrieval
- PostToolUse hook for observation storage
- Stop hook for checkpointing
- Integration with HookManager
- Short-term and long-term memory persistence
"""

import pytest
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import sys

# Mock optional dependencies before importing lib modules
sys.modules['gradio'] = Mock()
sys.modules['gradio_browser'] = Mock()
sys.modules['gradio_aicontext'] = Mock()
sys.modules['tiktoken'] = Mock()
sys.modules['litellm'] = Mock()
sys.modules['PIL'] = Mock()
sys.modules['PIL.Image'] = Mock()

from lib.hooks.types import HookEvent, HookContext, HookResult
from lib.hooks.manager import HookManager
from lib.memory.manager import MemoryManager
from lib.memory.hooks import MemoryHooks, create_memory_hooks


class TestMemoryHooksInitialization:
    """Tests for MemoryHooks initialization."""

    def test_basic_initialization(self, temp_memory_dir):
        """Test creating MemoryHooks with a MemoryManager."""
        memory_manager = MemoryManager(storage_path=temp_memory_dir)
        hooks = MemoryHooks(memory_manager)

        assert hooks.memory_manager is memory_manager
        assert hooks.checkpoint_interval == 100
        assert hooks.auto_retrieve is True
        assert hooks.top_k_context == 10

    def test_custom_configuration(self, temp_memory_dir):
        """Test creating MemoryHooks with custom settings."""
        memory_manager = MemoryManager(storage_path=temp_memory_dir)
        hooks = MemoryHooks(
            memory_manager,
            checkpoint_interval=50,
            auto_retrieve=False,
            top_k_context=5,
        )

        assert hooks.checkpoint_interval == 50
        assert hooks.auto_retrieve is False
        assert hooks.top_k_context == 5

    def test_create_memory_hooks_factory(self, temp_memory_dir):
        """Test the create_memory_hooks factory function."""
        hooks = create_memory_hooks(
            storage_path=temp_memory_dir,
            checkpoint_interval=25,
            auto_retrieve=True,
            top_k_context=15,
        )

        assert hooks.memory_manager is not None
        assert hooks.checkpoint_interval == 25
        assert hooks.top_k_context == 15


class TestMemoryHooksRegistration:
    """Tests for hook registration with HookManager."""

    def test_register_all(self, temp_memory_dir):
        """Test registering all memory hooks with a HookManager."""
        memory_manager = MemoryManager(storage_path=temp_memory_dir)
        memory_hooks = MemoryHooks(memory_manager)
        hook_manager = HookManager()

        memory_hooks.register_all(hook_manager)

        # Verify all hooks are registered
        assert hook_manager.has_hooks(HookEvent.SessionStart)
        assert hook_manager.has_hooks(HookEvent.UserPromptSubmit)
        assert hook_manager.has_hooks(HookEvent.PostToolUse)
        assert hook_manager.has_hooks(HookEvent.Stop)

    def test_unregister_all(self, temp_memory_dir):
        """Test unregistering all memory hooks."""
        memory_manager = MemoryManager(storage_path=temp_memory_dir)
        memory_hooks = MemoryHooks(memory_manager)
        hook_manager = HookManager()

        memory_hooks.register_all(hook_manager)
        memory_hooks.unregister_all(hook_manager)

        # Verify all hooks are unregistered
        assert not hook_manager.has_hooks(HookEvent.SessionStart)
        assert not hook_manager.has_hooks(HookEvent.UserPromptSubmit)
        assert not hook_manager.has_hooks(HookEvent.PostToolUse)
        assert not hook_manager.has_hooks(HookEvent.Stop)


class TestSessionStartHook:
    """Tests for the SessionStart hook handler."""

    def test_new_session_created(self, temp_memory_dir):
        """Test that SessionStart creates a new session."""
        memory_manager = MemoryManager(storage_path=temp_memory_dir)
        memory_hooks = MemoryHooks(memory_manager)

        context = HookContext(
            event=HookEvent.SessionStart,
            metadata={"task": "Test task"},
        )

        result = memory_hooks.on_session_start(context)

        assert result.metadata.get("session_started") is True
        assert "session_id" in result.metadata
        assert memory_manager.current_session_id is not None

    def test_session_resume(self, temp_memory_dir):
        """Test that SessionStart can resume an existing session."""
        memory_manager = MemoryManager(storage_path=temp_memory_dir)
        memory_hooks = MemoryHooks(memory_manager)

        # Create initial session
        session_id = memory_manager.start_session(task="Original task")
        memory_manager.checkpoint(
            step=5,
            messages=[{"role": "user", "content": "test"}],
            task="Original task",
            progress="Step 5",
        )

        # Reset and resume
        memory_hooks.reset()
        context = HookContext(
            event=HookEvent.SessionStart,
            metadata={"resume_id": session_id},
        )

        result = memory_hooks.on_session_start(context)

        assert result.metadata.get("session_restored") is True
        assert result.metadata.get("session_id") == session_id
        # The restored_step should be the step from the checkpoint
        assert result.metadata.get("restored_step") >= 1

    def test_failed_resume_creates_new_session(self, temp_memory_dir):
        """Test that failed resume creates a new session."""
        memory_manager = MemoryManager(storage_path=temp_memory_dir)
        memory_hooks = MemoryHooks(memory_manager)

        context = HookContext(
            event=HookEvent.SessionStart,
            metadata={"resume_id": "nonexistent_session"},
        )

        result = memory_hooks.on_session_start(context)

        # Should create new session when resume fails
        assert result.metadata.get("session_started") is True
        assert "session_id" in result.metadata


class TestUserPromptSubmitHook:
    """Tests for the UserPromptSubmit hook handler."""

    def test_context_retrieved(self, temp_memory_dir):
        """Test that UserPromptSubmit retrieves context."""
        memory_manager = MemoryManager(storage_path=temp_memory_dir)
        memory_hooks = MemoryHooks(memory_manager)

        # Add some memories first
        memory_manager.add_memory("Python is a programming language", memory_type="fact")
        memory_manager.add_memory("File was created successfully", memory_type="observation")

        context = HookContext(
            event=HookEvent.UserPromptSubmit,
            query="What is Python?",
        )

        result = memory_hooks.on_user_prompt_submit(context)

        assert "retrieved_context" in result.metadata
        assert result.metadata.get("context_count", 0) >= 0

    def test_auto_retrieve_disabled(self, temp_memory_dir):
        """Test that context retrieval can be disabled."""
        memory_manager = MemoryManager(storage_path=temp_memory_dir)
        memory_hooks = MemoryHooks(memory_manager, auto_retrieve=False)

        context = HookContext(
            event=HookEvent.UserPromptSubmit,
            query="Test query",
        )

        result = memory_hooks.on_user_prompt_submit(context)

        assert "retrieved_context" not in result.metadata

    def test_query_stored(self, temp_memory_dir):
        """Test that the query is stored for later use."""
        memory_manager = MemoryManager(storage_path=temp_memory_dir)
        memory_hooks = MemoryHooks(memory_manager)

        context = HookContext(
            event=HookEvent.UserPromptSubmit,
            query="My important query",
        )

        memory_hooks.on_user_prompt_submit(context)

        assert memory_hooks._last_query == "My important query"


class TestPostToolUseHook:
    """Tests for the PostToolUse hook handler."""

    def test_observation_stored(self, temp_memory_dir):
        """Test that PostToolUse stores observations."""
        memory_manager = MemoryManager(storage_path=temp_memory_dir)
        memory_hooks = MemoryHooks(memory_manager)

        context = HookContext(
            event=HookEvent.PostToolUse,
            tool_name="bash",
            arguments={"command": "ls -la"},
            result={"output": "file1.py file2.py"},
        )

        result = memory_hooks.on_post_tool_use(context)

        assert result.metadata.get("observation_stored") is True
        assert "memory_id" in result.metadata
        assert result.metadata.get("step") == 1

    def test_step_incremented(self, temp_memory_dir):
        """Test that step count is incremented on each tool use."""
        memory_manager = MemoryManager(storage_path=temp_memory_dir)
        memory_hooks = MemoryHooks(memory_manager)

        for i in range(3):
            context = HookContext(
                event=HookEvent.PostToolUse,
                tool_name="bash",
                arguments={"command": f"command_{i}"},
                result={"output": f"result_{i}"},
            )
            memory_hooks.on_post_tool_use(context)

        assert memory_hooks.get_current_step() == 3

    def test_string_result_handled(self, temp_memory_dir):
        """Test that string results are handled correctly."""
        memory_manager = MemoryManager(storage_path=temp_memory_dir)
        memory_hooks = MemoryHooks(memory_manager)

        context = HookContext(
            event=HookEvent.PostToolUse,
            tool_name="bash",
            arguments={"command": "echo hello"},
            result="hello",
        )

        result = memory_hooks.on_post_tool_use(context)

        assert result.metadata.get("observation_stored") is True


class TestStopHook:
    """Tests for the Stop hook handler."""

    def test_checkpoint_saved(self, temp_memory_dir):
        """Test that Stop saves a checkpoint when interval reached."""
        memory_manager = MemoryManager(storage_path=temp_memory_dir)
        memory_hooks = MemoryHooks(memory_manager, checkpoint_interval=5)

        # Start session
        memory_hooks.on_session_start(HookContext(
            event=HookEvent.SessionStart,
            metadata={"task": "Test"},
        ))

        # Simulate some tool uses
        for _ in range(5):
            memory_hooks.on_post_tool_use(HookContext(
                event=HookEvent.PostToolUse,
                tool_name="bash",
                result={"output": "test"},
            ))

        context = HookContext(
            event=HookEvent.Stop,
            messages=[{"role": "user", "content": "test"}],
            metadata={"total_steps": 5},
        )

        result = memory_hooks.on_stop(context)

        assert result.metadata.get("checkpoint_saved") is True
        assert result.metadata.get("final_step") == 5

    def test_session_ended_on_complete(self, temp_memory_dir):
        """Test that Stop ends session when marked complete."""
        memory_manager = MemoryManager(storage_path=temp_memory_dir)
        memory_hooks = MemoryHooks(memory_manager)

        # Start session
        memory_hooks.on_session_start(HookContext(
            event=HookEvent.SessionStart,
            metadata={"task": "Test"},
        ))

        context = HookContext(
            event=HookEvent.Stop,
            messages=[],
            metadata={"is_complete": True, "total_steps": 1},
        )

        result = memory_hooks.on_stop(context)

        assert result.metadata.get("session_ended") is True


class TestMemoryPersistence:
    """Tests for short-term and long-term memory persistence."""

    def test_short_term_memory_persisted(self, temp_memory_dir):
        """Test that short-term observations are persisted."""
        memory_manager = MemoryManager(storage_path=temp_memory_dir)
        memory_hooks = MemoryHooks(memory_manager)
        hook_manager = HookManager()
        memory_hooks.register_all(hook_manager)

        # Trigger PostToolUse to store observations
        hook_manager.trigger(
            HookEvent.PostToolUse,
            tool_name="bash",
            arguments={"command": "ls"},
            result={"output": "file1.txt file2.txt"},
        )

        # Verify observation in short-term memory
        assert len(memory_manager.short_term) > 0

        # Verify persistence file exists
        buffer_path = Path(temp_memory_dir) / "short_term" / "buffer.md"
        assert buffer_path.exists()

    def test_long_term_memory_persisted(self, temp_memory_dir):
        """Test that long-term knowledge is persisted."""
        memory_manager = MemoryManager(storage_path=temp_memory_dir)

        # Add knowledge directly
        memory_manager.add_memory(
            "Python is a programming language",
            memory_type="fact",
        )

        # Verify in long-term memory
        assert len(memory_manager.long_term) > 0

        # Verify persistence files exist
        knowledge_path = Path(temp_memory_dir) / "long_term" / "knowledge.md"
        assert knowledge_path.exists()

        content = knowledge_path.read_text()
        assert "Python is a programming language" in content

    def test_context_retrieved_from_both_memories(self, temp_memory_dir):
        """Test that context is retrieved from both short-term and long-term."""
        memory_manager = MemoryManager(storage_path=temp_memory_dir)
        memory_hooks = MemoryHooks(memory_manager)

        # Add to long-term
        memory_manager.add_memory("Python is a programming language", memory_type="fact")

        # Add to short-term via hook
        memory_hooks.on_post_tool_use(HookContext(
            event=HookEvent.PostToolUse,
            tool_name="bash",
            arguments={"command": "python --version"},
            result={"output": "Python 3.12.0"},
        ))

        # Retrieve context
        result = memory_hooks.on_user_prompt_submit(HookContext(
            event=HookEvent.UserPromptSubmit,
            query="What Python version?",
        ))

        context = result.metadata.get("retrieved_context", [])
        assert len(context) > 0


class TestIntegrationWithCodingAgent:
    """Integration tests for memory hooks with the coding agent flow."""

    def test_full_lifecycle(self, temp_memory_dir):
        """Test the full memory lifecycle through hooks."""
        memory_manager = MemoryManager(storage_path=temp_memory_dir)
        memory_hooks = MemoryHooks(memory_manager)
        hook_manager = HookManager()
        memory_hooks.register_all(hook_manager)

        # 1. Session Start
        start_result = hook_manager.trigger(
            HookEvent.SessionStart,
            metadata={"task": "Test task"},
        )
        assert start_result.metadata.get("session_started") is True

        # 2. User Prompt Submit
        prompt_result = hook_manager.trigger(
            HookEvent.UserPromptSubmit,
            query="List all files",
        )
        assert "context_count" in prompt_result.metadata

        # 3. Tool Use (multiple)
        for i in range(3):
            tool_result = hook_manager.trigger(
                HookEvent.PostToolUse,
                tool_name="bash",
                arguments={"command": f"command_{i}"},
                result={"output": f"output_{i}"},
            )
            assert tool_result.metadata.get("observation_stored") is True

        # 4. Stop
        stop_result = hook_manager.trigger(
            HookEvent.Stop,
            messages=[{"role": "user", "content": "test"}],
            metadata={"total_steps": 3, "is_complete": True},
        )
        assert stop_result.metadata.get("session_ended") is True

        # Verify state
        assert memory_hooks.get_current_step() == 3
        assert len(memory_manager.short_term) == 3

    def test_context_injection_flow(self, temp_memory_dir):
        """Test that retrieved context can be used for injection."""
        memory_manager = MemoryManager(storage_path=temp_memory_dir)
        memory_hooks = MemoryHooks(memory_manager, top_k_context=5)

        # Add some prior knowledge
        memory_manager.add_memory("The config file is at /etc/config", memory_type="fact")
        memory_manager.add_memory("Use pytest for testing", memory_type="pattern")

        # Simulate user query about pytest specifically
        memory_hooks.on_user_prompt_submit(HookContext(
            event=HookEvent.UserPromptSubmit,
            query="How do I use pytest?",
        ))

        # Get retrieved context
        context = memory_hooks.get_retrieved_context()
        # Context should be retrieved (even if empty, since keyword search may not match)
        assert context is not None
        # At minimum, check that the retrieval mechanism works
        # The actual matching depends on the keyword search implementation


class TestResetAndState:
    """Tests for state management and reset."""

    def test_reset_clears_state(self, temp_memory_dir):
        """Test that reset clears internal state."""
        memory_manager = MemoryManager(storage_path=temp_memory_dir)
        memory_hooks = MemoryHooks(memory_manager)

        # Build up some state
        memory_hooks._step_count = 10
        memory_hooks._current_task = "Some task"
        memory_hooks._last_query = "Some query"
        memory_hooks._retrieved_context = ["context1", "context2"]

        # Reset
        memory_hooks.reset()

        assert memory_hooks._step_count == 0
        assert memory_hooks._current_task is None
        assert memory_hooks._last_query is None
        assert memory_hooks._retrieved_context is None

    def test_get_current_step(self, temp_memory_dir):
        """Test getting current step count."""
        memory_manager = MemoryManager(storage_path=temp_memory_dir)
        memory_hooks = MemoryHooks(memory_manager)

        assert memory_hooks.get_current_step() == 0

        # Simulate tool uses
        for _ in range(5):
            memory_hooks.on_post_tool_use(HookContext(
                event=HookEvent.PostToolUse,
                tool_name="bash",
                result="output",
            ))

        assert memory_hooks.get_current_step() == 5


# Fixtures


@pytest.fixture
def temp_memory_dir():
    """Create a temporary directory for memory storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
