"""
Tests for the lifecycle hooks module.

Tests cover:
- HookEvent enum
- HookContext dataclass
- HookResult dataclass
- HookManager class
  - Registration (decorator and direct)
  - Triggering hooks
  - Result merging
  - Script hook execution (mock)
  - Blocking behavior
"""

import pytest
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sys

# Mock optional dependencies before importing lib modules
sys.modules['gradio'] = Mock()
sys.modules['gradio_browser'] = Mock()
sys.modules['gradio_aicontext'] = Mock()
sys.modules['tiktoken'] = Mock()
sys.modules['litellm'] = Mock()
sys.modules['PIL'] = Mock()
sys.modules['PIL.Image'] = Mock()

# Now import hooks directly
from lib.hooks.types import HookEvent, HookContext, HookResult
from lib.hooks.manager import HookManager


class TestHookEvent:
    """Tests for the HookEvent enum."""

    def test_all_events_exist(self):
        """Test that all required hook events are defined."""
        assert HookEvent.SessionStart.value == "session_start"
        assert HookEvent.UserPromptSubmit.value == "user_prompt_submit"
        assert HookEvent.PreToolUse.value == "pre_tool_use"
        assert HookEvent.PostToolUse.value == "post_tool_use"
        assert HookEvent.Stop.value == "stop"

    def test_event_count(self):
        """Test that we have exactly 5 hook events."""
        assert len(HookEvent) == 5


class TestHookContext:
    """Tests for the HookContext dataclass."""

    def test_basic_creation(self):
        """Test creating a HookContext with minimal data."""
        ctx = HookContext(event=HookEvent.SessionStart)
        assert ctx.event == HookEvent.SessionStart
        assert isinstance(ctx.timestamp, datetime)
        assert ctx.session_id is None
        assert ctx.tool_name is None
        assert ctx.arguments is None
        assert ctx.result is None
        assert ctx.query is None
        assert ctx.messages is None
        assert ctx.metadata == {}

    def test_creation_with_all_fields(self):
        """Test creating a HookContext with all fields populated."""
        timestamp = datetime.now()
        ctx = HookContext(
            event=HookEvent.PreToolUse,
            timestamp=timestamp,
            session_id="test-session-123",
            tool_name="bash",
            arguments={"command": "ls -la"},
            result=None,
            query="list files",
            messages=[{"role": "user", "content": "hello"}],
            metadata={"custom_key": "custom_value"},
        )
        assert ctx.event == HookEvent.PreToolUse
        assert ctx.timestamp == timestamp
        assert ctx.session_id == "test-session-123"
        assert ctx.tool_name == "bash"
        assert ctx.arguments == {"command": "ls -la"}
        assert ctx.query == "list files"
        assert ctx.messages == [{"role": "user", "content": "hello"}]
        assert ctx.metadata == {"custom_key": "custom_value"}

    def test_to_dict(self):
        """Test serialization of HookContext to dictionary."""
        ctx = HookContext(
            event=HookEvent.PostToolUse,
            tool_name="write_file",
            result={"status": "success"},
        )
        data = ctx.to_dict()
        assert data["event"] == "post_tool_use"
        assert data["tool_name"] == "write_file"
        assert data["result"] == {"status": "success"}
        assert "timestamp" in data


class TestHookResult:
    """Tests for the HookResult dataclass."""

    def test_default_values(self):
        """Test default HookResult values (allow action)."""
        result = HookResult()
        assert result.block is False
        assert result.reason is None
        assert result.modified_arguments is None
        assert result.modified_result is None
        assert result.skip_remaining is False
        assert result.metadata == {}

    def test_allow_factory(self):
        """Test HookResult.allow() factory method."""
        result = HookResult.allow()
        assert result.block is False

    def test_deny_factory(self):
        """Test HookResult.deny() factory method."""
        result = HookResult.deny("Dangerous command blocked")
        assert result.block is True
        assert result.reason == "Dangerous command blocked"

    def test_to_dict(self):
        """Test serialization of HookResult to dictionary."""
        result = HookResult(
            block=True,
            reason="test reason",
            modified_arguments={"arg1": "modified"},
            metadata={"key": "value"},
        )
        data = result.to_dict()
        assert data["block"] is True
        assert data["reason"] == "test reason"
        assert data["modified_arguments"] == {"arg1": "modified"}
        assert data["metadata"] == {"key": "value"}


class TestHookManager:
    """Tests for the HookManager class."""

    def test_initialization(self):
        """Test HookManager initializes with empty hooks for all events."""
        manager = HookManager()
        for event in HookEvent:
            assert manager.has_hooks(event) is False

    def test_register_decorator(self):
        """Test registering a hook using the decorator."""
        manager = HookManager()

        @manager.on(HookEvent.SessionStart)
        def on_start(ctx):
            return HookResult()

        assert manager.has_hooks(HookEvent.SessionStart)

    def test_register_direct(self):
        """Test registering a hook using direct method."""
        manager = HookManager()

        def my_handler(ctx):
            return HookResult()

        manager.register(HookEvent.Stop, my_handler)
        assert manager.has_hooks(HookEvent.Stop)

    def test_unregister(self):
        """Test unregistering a hook handler."""
        manager = HookManager()

        def my_handler(ctx):
            return HookResult()

        manager.register(HookEvent.Stop, my_handler)
        assert manager.has_hooks(HookEvent.Stop)

        success = manager.unregister(HookEvent.Stop, my_handler)
        assert success is True
        assert manager.has_hooks(HookEvent.Stop) is False

    def test_unregister_nonexistent(self):
        """Test unregistering a handler that doesn't exist."""
        manager = HookManager()

        def my_handler(ctx):
            return HookResult()

        success = manager.unregister(HookEvent.Stop, my_handler)
        assert success is False

    def test_clear_specific_event(self):
        """Test clearing hooks for a specific event."""
        manager = HookManager()

        @manager.on(HookEvent.SessionStart)
        def handler1(ctx):
            pass

        @manager.on(HookEvent.Stop)
        def handler2(ctx):
            pass

        manager.clear(HookEvent.SessionStart)
        assert manager.has_hooks(HookEvent.SessionStart) is False
        assert manager.has_hooks(HookEvent.Stop) is True

    def test_clear_all_events(self):
        """Test clearing all hooks."""
        manager = HookManager()

        @manager.on(HookEvent.SessionStart)
        def handler1(ctx):
            pass

        @manager.on(HookEvent.Stop)
        def handler2(ctx):
            pass

        manager.clear()
        assert manager.has_hooks(HookEvent.SessionStart) is False
        assert manager.has_hooks(HookEvent.Stop) is False

    def test_trigger_no_handlers(self):
        """Test triggering an event with no handlers returns default result."""
        manager = HookManager()
        result = manager.trigger(HookEvent.SessionStart)
        assert result.block is False

    def test_trigger_with_handler(self):
        """Test triggering an event executes the handler."""
        manager = HookManager()
        handler_called = []

        @manager.on(HookEvent.SessionStart)
        def on_start(ctx):
            handler_called.append(ctx.event)
            return HookResult()

        manager.trigger(HookEvent.SessionStart)
        assert HookEvent.SessionStart in handler_called

    def test_trigger_with_context_kwargs(self):
        """Test triggering with context keyword arguments."""
        manager = HookManager()
        received_context = []

        @manager.on(HookEvent.PreToolUse)
        def on_tool(ctx):
            received_context.append(ctx)
            return HookResult()

        manager.trigger(
            HookEvent.PreToolUse,
            tool_name="bash",
            arguments={"command": "echo hello"}
        )

        assert len(received_context) == 1
        assert received_context[0].tool_name == "bash"
        assert received_context[0].arguments == {"command": "echo hello"}

    def test_trigger_with_provided_context(self):
        """Test triggering with a pre-built HookContext."""
        manager = HookManager()
        received_context = []

        @manager.on(HookEvent.UserPromptSubmit)
        def on_prompt(ctx):
            received_context.append(ctx)
            return HookResult()

        ctx = HookContext(
            event=HookEvent.UserPromptSubmit,
            query="test query"
        )
        manager.trigger(HookEvent.UserPromptSubmit, context=ctx)

        assert len(received_context) == 1
        assert received_context[0].query == "test query"

    def test_blocking_hook(self):
        """Test that a blocking hook returns block=True."""
        manager = HookManager()

        @manager.on(HookEvent.PreToolUse)
        def block_dangerous(ctx):
            if ctx.tool_name == "bash":
                cmd = ctx.arguments.get("command", "")
                if "rm -rf" in cmd:
                    return HookResult.deny("Dangerous command blocked")
            return HookResult.allow()

        # Test blocked command
        result = manager.trigger(
            HookEvent.PreToolUse,
            tool_name="bash",
            arguments={"command": "rm -rf /"}
        )
        assert result.block is True
        assert "Dangerous" in result.reason

        # Test allowed command
        result = manager.trigger(
            HookEvent.PreToolUse,
            tool_name="bash",
            arguments={"command": "ls"}
        )
        assert result.block is False

    def test_multiple_handlers(self):
        """Test that multiple handlers are all executed."""
        manager = HookManager()
        call_order = []

        @manager.on(HookEvent.Stop)
        def handler1(ctx):
            call_order.append(1)
            return HookResult()

        @manager.on(HookEvent.Stop)
        def handler2(ctx):
            call_order.append(2)
            return HookResult()

        manager.trigger(HookEvent.Stop)
        assert call_order == [1, 2]

    def test_skip_remaining(self):
        """Test that skip_remaining stops further handlers."""
        manager = HookManager()
        call_order = []

        @manager.on(HookEvent.Stop)
        def handler1(ctx):
            call_order.append(1)
            return HookResult(skip_remaining=True)

        @manager.on(HookEvent.Stop)
        def handler2(ctx):
            call_order.append(2)
            return HookResult()

        manager.trigger(HookEvent.Stop)
        assert call_order == [1]  # handler2 should not be called

    def test_result_merging(self):
        """Test that results from multiple handlers are merged."""
        manager = HookManager()

        @manager.on(HookEvent.PreToolUse)
        def modify_args1(ctx):
            return HookResult(modified_arguments={"arg1": "value1"})

        @manager.on(HookEvent.PreToolUse)
        def modify_args2(ctx):
            return HookResult(modified_arguments={"arg2": "value2"})

        result = manager.trigger(HookEvent.PreToolUse)
        assert result.modified_arguments == {"arg1": "value1", "arg2": "value2"}

    def test_block_takes_precedence(self):
        """Test that block=True takes precedence in merged results."""
        manager = HookManager()

        @manager.on(HookEvent.PreToolUse)
        def allow_handler(ctx):
            return HookResult(block=False)

        @manager.on(HookEvent.PreToolUse)
        def block_handler(ctx):
            return HookResult(block=True, reason="Blocked by second handler")

        result = manager.trigger(HookEvent.PreToolUse)
        assert result.block is True
        assert result.reason == "Blocked by second handler"

    def test_handler_returning_dict(self):
        """Test that handlers can return dict instead of HookResult."""
        manager = HookManager()

        @manager.on(HookEvent.PreToolUse)
        def dict_handler(ctx):
            return {"block": True, "reason": "Dict-based blocking"}

        result = manager.trigger(HookEvent.PreToolUse)
        assert result.block is True
        assert result.reason == "Dict-based blocking"

    def test_handler_returning_none(self):
        """Test that handlers can return None."""
        manager = HookManager()

        @manager.on(HookEvent.Stop)
        def none_handler(ctx):
            return None

        result = manager.trigger(HookEvent.Stop)
        assert result.block is False

    def test_handler_exception_is_caught(self):
        """Test that exceptions in handlers are caught and logged."""
        manager = HookManager()

        @manager.on(HookEvent.Stop)
        def error_handler(ctx):
            raise ValueError("Test error")

        @manager.on(HookEvent.Stop)
        def normal_handler(ctx):
            return HookResult(metadata={"reached": True})

        # Should not raise, exception is caught
        result = manager.trigger(HookEvent.Stop)
        assert result.metadata.get("reached") is True

    def test_list_hooks(self):
        """Test listing registered hooks."""
        manager = HookManager()

        @manager.on(HookEvent.SessionStart)
        def on_start(ctx):
            pass

        @manager.on(HookEvent.Stop)
        def on_stop(ctx):
            pass

        hooks = manager.list_hooks()
        assert "session_start" in hooks
        assert "stop" in hooks
        assert "on_start" in hooks["session_start"]
        assert "on_stop" in hooks["stop"]

    def test_list_hooks_specific_event(self):
        """Test listing hooks for a specific event."""
        manager = HookManager()

        @manager.on(HookEvent.SessionStart)
        def on_start(ctx):
            pass

        @manager.on(HookEvent.Stop)
        def on_stop(ctx):
            pass

        hooks = manager.list_hooks(HookEvent.SessionStart)
        assert "session_start" in hooks
        assert "stop" not in hooks


class TestScriptHooks:
    """Tests for script-based hooks (bash/python)."""

    def test_register_script_hook(self):
        """Test registering a script hook."""
        manager = HookManager()
        manager.register_script(
            HookEvent.SessionStart,
            "/path/to/script.sh",
            "bash"
        )
        assert manager.has_hooks(HookEvent.SessionStart)

    def test_register_script_invalid_type(self):
        """Test that invalid script types raise an error."""
        manager = HookManager()
        with pytest.raises(ValueError, match="Invalid script type"):
            manager.register_script(
                HookEvent.SessionStart,
                "/path/to/script.rb",
                "ruby"
            )

    @patch('subprocess.run')
    def test_execute_bash_script(self, mock_run):
        """Test executing a bash script hook."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"block": false}',
            stderr=""
        )

        manager = HookManager()
        manager.register_script(HookEvent.SessionStart, "/path/to/hook.sh", "bash")
        result = manager.trigger(HookEvent.SessionStart)

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0] == ["bash", "/path/to/hook.sh"]
        assert result.block is False

    @patch('subprocess.run')
    def test_execute_python_script(self, mock_run):
        """Test executing a python script hook."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"block": true, "reason": "Python blocked"}',
            stderr=""
        )

        manager = HookManager()
        manager.register_script(HookEvent.PreToolUse, "/path/to/hook.py", "python")
        result = manager.trigger(HookEvent.PreToolUse)

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0] == ["python", "/path/to/hook.py"]
        assert result.block is True
        assert result.reason == "Python blocked"

    @patch('subprocess.run')
    def test_script_receives_context_json(self, mock_run):
        """Test that script receives context as JSON via stdin."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="",
            stderr=""
        )

        manager = HookManager()
        manager.register_script(HookEvent.PreToolUse, "/path/to/hook.sh", "bash")
        manager.trigger(
            HookEvent.PreToolUse,
            tool_name="bash",
            arguments={"command": "test"}
        )

        call_kwargs = mock_run.call_args[1]
        input_json = json.loads(call_kwargs["input"])
        assert input_json["event"] == "pre_tool_use"
        assert input_json["tool_name"] == "bash"
        assert input_json["arguments"] == {"command": "test"}

    @patch('subprocess.run')
    def test_script_timeout(self, mock_run):
        """Test that script timeout is handled gracefully."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("bash", 30)

        manager = HookManager()
        manager.register_script(HookEvent.SessionStart, "/path/to/slow.sh", "bash")

        # Should not raise, timeout is caught
        result = manager.trigger(HookEvent.SessionStart)
        assert result.block is False

    @patch('subprocess.run')
    def test_script_not_found(self, mock_run):
        """Test that missing script is handled gracefully."""
        mock_run.side_effect = FileNotFoundError()

        manager = HookManager()
        manager.register_script(HookEvent.SessionStart, "/nonexistent.sh", "bash")

        # Should not raise, error is caught
        result = manager.trigger(HookEvent.SessionStart)
        assert result.block is False


class TestIntegrationScenarios:
    """Integration tests for common hook usage scenarios."""

    def test_dangerous_command_blocking(self):
        """Test blocking dangerous bash commands."""
        manager = HookManager()

        DANGEROUS_PATTERNS = [
            "rm -rf /",
            "rm -rf /*",
            ":(){:|:&};:",  # Fork bomb
            "dd if=/dev/zero of=/dev/sda",
        ]

        @manager.on(HookEvent.PreToolUse)
        def block_dangerous(ctx):
            if ctx.tool_name == "bash":
                cmd = ctx.arguments.get("command", "")
                for pattern in DANGEROUS_PATTERNS:
                    if pattern in cmd:
                        return HookResult.deny(f"Blocked dangerous command: {pattern}")
            return HookResult.allow()

        # Test each dangerous command is blocked
        for pattern in DANGEROUS_PATTERNS:
            result = manager.trigger(
                HookEvent.PreToolUse,
                tool_name="bash",
                arguments={"command": pattern}
            )
            assert result.block is True, f"Should block: {pattern}"

        # Test safe command is allowed
        result = manager.trigger(
            HookEvent.PreToolUse,
            tool_name="bash",
            arguments={"command": "ls -la"}
        )
        assert result.block is False

    def test_session_logging(self):
        """Test logging session start and stop."""
        manager = HookManager()
        log = []

        @manager.on(HookEvent.SessionStart)
        def log_start(ctx):
            log.append(f"Session started at {ctx.timestamp}")
            return HookResult()

        @manager.on(HookEvent.Stop)
        def log_stop(ctx):
            log.append(f"Session stopped at {ctx.timestamp}")
            return HookResult()

        manager.trigger(HookEvent.SessionStart)
        manager.trigger(HookEvent.Stop)

        assert len(log) == 2
        assert "started" in log[0]
        assert "stopped" in log[1]

    def test_query_modification(self):
        """Test modifying user query before sending to LLM."""
        manager = HookManager()

        @manager.on(HookEvent.UserPromptSubmit)
        def add_context(ctx):
            # Could modify the query or add context
            return HookResult(metadata={"enhanced_query": f"[enhanced] {ctx.query}"})

        result = manager.trigger(
            HookEvent.UserPromptSubmit,
            query="What is the weather?"
        )

        assert result.metadata.get("enhanced_query") == "[enhanced] What is the weather?"

    def test_tool_result_modification(self):
        """Test modifying tool results after execution."""
        manager = HookManager()

        @manager.on(HookEvent.PostToolUse)
        def sanitize_result(ctx):
            if ctx.tool_name == "bash":
                # Could sanitize or format the result
                return HookResult(
                    modified_result=f"[sanitized] {ctx.result}",
                    metadata={"was_sanitized": True}
                )
            return HookResult()

        result = manager.trigger(
            HookEvent.PostToolUse,
            tool_name="bash",
            result="raw output"
        )

        assert result.modified_result == "[sanitized] raw output"
        assert result.metadata.get("was_sanitized") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
