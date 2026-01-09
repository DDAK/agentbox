"""
Integration tests for hooks in the coding_agent module.

Tests that hooks are properly triggered during the agent execution lifecycle:
- UserPromptSubmit: When user query is received
- PreToolUse: Before tool execution
- PostToolUse: After tool execution
- Stop: When agent finishes
"""

import pytest
import json
import sys
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

# Mock optional dependencies before importing lib modules
sys.modules['gradio'] = Mock()
sys.modules['gradio_browser'] = Mock()
sys.modules['gradio_aicontext'] = Mock()
sys.modules['tiktoken'] = Mock()
sys.modules['litellm'] = Mock()
sys.modules['PIL'] = Mock()
sys.modules['PIL.Image'] = Mock()

# Import hooks directly
from lib.hooks.types import HookEvent, HookContext, HookResult
from lib.hooks.manager import HookManager

from lib.coding_agent import (
    coding_agent,
    get_hook_manager,
    set_hook_manager,
)


class MockSandbox:
    """Mock sandbox for testing."""
    pass


class MockLLMResponse:
    """Mock LLM response for testing."""
    def __init__(self, output, total_tokens=100):
        self.output = output
        self.usage = Mock(total_tokens=total_tokens)


class MockOutputPart:
    """Mock output part from LLM response."""
    def __init__(self, part_type, **kwargs):
        self.type = part_type
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        result = {"type": self.type}
        if self.type == "function_call":
            result.update({
                "name": self.name,
                "arguments": self.arguments,
                "call_id": self.call_id,
            })
        elif self.type == "message":
            result["content"] = [{"text": getattr(self, "text", "")}]
        return result


@pytest.fixture
def hook_manager():
    """Create a fresh HookManager for each test."""
    manager = HookManager()
    return manager


@pytest.fixture
def mock_client():
    """Create a mock LLM client."""
    client = Mock()
    # Default: return a simple message (no tool calls)
    client.responses.create.return_value = MockLLMResponse(
        output=[MockOutputPart("message", text="Hello!")]
    )
    return client


@pytest.fixture
def mock_sandbox():
    """Create a mock sandbox."""
    return MockSandbox()


@pytest.fixture
def tools():
    """Create mock tools."""
    def mock_bash(command):
        return {"output": f"Executed: {command}"}

    def mock_write_file(path, content):
        return {"status": "success", "path": path}

    return {
        "bash": mock_bash,
        "write_file": mock_write_file,
    }


@pytest.fixture
def tools_schemas():
    """Create mock tool schemas."""
    return [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute bash command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"}
                    },
                    "required": ["command"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["path", "content"]
                }
            }
        }
    ]


class TestUserPromptSubmitHook:
    """Tests for UserPromptSubmit hook triggering."""

    def test_hook_triggered_on_query(self, hook_manager, mock_client, mock_sandbox, tools, tools_schemas):
        """Test that UserPromptSubmit hook is triggered when query is submitted."""
        triggered_events = []

        @hook_manager.on(HookEvent.UserPromptSubmit)
        def on_prompt(ctx):
            triggered_events.append({
                "event": ctx.event,
                "query": ctx.query,
            })
            return HookResult()

        # Run the agent
        gen = coding_agent(
            client=mock_client,
            sbx=mock_sandbox,
            query="Test query",
            tools=tools,
            tools_schemas=tools_schemas,
            hook_manager=hook_manager,
        )

        # Consume the generator
        list(gen)

        assert len(triggered_events) == 1
        assert triggered_events[0]["event"] == HookEvent.UserPromptSubmit
        assert triggered_events[0]["query"] == "Test query"

    def test_hook_can_block_query(self, hook_manager, mock_client, mock_sandbox, tools, tools_schemas):
        """Test that UserPromptSubmit hook can block the query."""
        @hook_manager.on(HookEvent.UserPromptSubmit)
        def block_query(ctx):
            if "blocked" in ctx.query.lower():
                return HookResult.deny("Query contains blocked content")
            return HookResult.allow()

        # Run with blocked query
        gen = coding_agent(
            client=mock_client,
            sbx=mock_sandbox,
            query="This should be blocked",
            tools=tools,
            tools_schemas=tools_schemas,
            hook_manager=hook_manager,
        )

        results = list(gen)

        # Should return early with blocked message
        assert len(results) == 1
        assert "blocked" in results[0][0]["content"].lower()

        # LLM should not be called
        mock_client.responses.create.assert_not_called()


class TestPreToolUseHook:
    """Tests for PreToolUse hook triggering."""

    def test_hook_triggered_before_tool(self, hook_manager, mock_client, mock_sandbox, tools, tools_schemas):
        """Test that PreToolUse hook is triggered before tool execution."""
        triggered_events = []

        @hook_manager.on(HookEvent.PreToolUse)
        def on_pre_tool(ctx):
            triggered_events.append({
                "event": ctx.event,
                "tool_name": ctx.tool_name,
                "arguments": ctx.arguments,
            })
            return HookResult()

        # Setup client to return a tool call
        mock_client.responses.create.return_value = MockLLMResponse(
            output=[
                MockOutputPart(
                    "function_call",
                    name="bash",
                    arguments={"command": "ls -la"},
                    call_id="call_123"
                )
            ]
        )

        gen = coding_agent(
            client=mock_client,
            sbx=mock_sandbox,
            query="List files",
            tools=tools,
            tools_schemas=tools_schemas,
            hook_manager=hook_manager,
            max_steps=1,
        )

        list(gen)

        assert len(triggered_events) == 1
        assert triggered_events[0]["event"] == HookEvent.PreToolUse
        assert triggered_events[0]["tool_name"] == "bash"
        assert triggered_events[0]["arguments"] == {"command": "ls -la"}

    def test_hook_can_block_tool(self, hook_manager, mock_client, mock_sandbox, tools, tools_schemas):
        """Test that PreToolUse hook can block tool execution."""
        @hook_manager.on(HookEvent.PreToolUse)
        def block_dangerous(ctx):
            if ctx.tool_name == "bash":
                cmd = ctx.arguments.get("command", "")
                if "rm -rf" in cmd:
                    return HookResult.deny("Dangerous command blocked")
            return HookResult.allow()

        # Setup client to return a dangerous command
        mock_client.responses.create.return_value = MockLLMResponse(
            output=[
                MockOutputPart(
                    "function_call",
                    name="bash",
                    arguments={"command": "rm -rf /"},
                    call_id="call_123"
                )
            ]
        )

        gen = coding_agent(
            client=mock_client,
            sbx=mock_sandbox,
            query="Delete everything",
            tools=tools,
            tools_schemas=tools_schemas,
            hook_manager=hook_manager,
            max_steps=1,
        )

        results = list(gen)

        # Find the function_call_output
        outputs = [r for r in results if r[0].get("type") == "function_call_output"]
        assert len(outputs) == 1
        output = json.loads(outputs[0][0]["output"])
        assert "error" in output
        assert "blocked" in output["error"].lower()

    def test_hook_can_modify_arguments(self, hook_manager, mock_client, mock_sandbox, tools, tools_schemas):
        """Test that PreToolUse hook can modify tool arguments."""
        @hook_manager.on(HookEvent.PreToolUse)
        def modify_args(ctx):
            if ctx.tool_name == "bash":
                # Add safety flag
                cmd = ctx.arguments.get("command", "")
                return HookResult(modified_arguments={"command": f"safe_{cmd}"})
            return HookResult()

        # Track what was actually executed
        executed_commands = []
        def capturing_bash(command, **kwargs):
            executed_commands.append(command)
            return {"output": f"Executed: {command}"}, {}

        tools["bash"] = capturing_bash

        mock_client.responses.create.return_value = MockLLMResponse(
            output=[
                MockOutputPart(
                    "function_call",
                    name="bash",
                    arguments={"command": "ls"},
                    call_id="call_123"
                )
            ]
        )

        gen = coding_agent(
            client=mock_client,
            sbx=mock_sandbox,
            query="List files",
            tools=tools,
            tools_schemas=tools_schemas,
            hook_manager=hook_manager,
            max_steps=1,
        )

        list(gen)

        # Check that modified command was executed
        assert len(executed_commands) == 1
        assert executed_commands[0] == "safe_ls"


class TestPostToolUseHook:
    """Tests for PostToolUse hook triggering."""

    def test_hook_triggered_after_tool(self, hook_manager, mock_client, mock_sandbox, tools, tools_schemas):
        """Test that PostToolUse hook is triggered after tool execution."""
        triggered_events = []

        @hook_manager.on(HookEvent.PostToolUse)
        def on_post_tool(ctx):
            triggered_events.append({
                "event": ctx.event,
                "tool_name": ctx.tool_name,
                "result": ctx.result,
            })
            return HookResult()

        mock_client.responses.create.return_value = MockLLMResponse(
            output=[
                MockOutputPart(
                    "function_call",
                    name="bash",
                    arguments={"command": "echo hello"},
                    call_id="call_123"
                )
            ]
        )

        gen = coding_agent(
            client=mock_client,
            sbx=mock_sandbox,
            query="Echo hello",
            tools=tools,
            tools_schemas=tools_schemas,
            hook_manager=hook_manager,
            max_steps=1,
        )

        list(gen)

        assert len(triggered_events) == 1
        assert triggered_events[0]["event"] == HookEvent.PostToolUse
        assert triggered_events[0]["tool_name"] == "bash"
        assert triggered_events[0]["result"] is not None

    def test_hook_can_modify_result(self, hook_manager, mock_client, mock_sandbox, tools, tools_schemas):
        """Test that PostToolUse hook can modify tool result."""
        @hook_manager.on(HookEvent.PostToolUse)
        def modify_result(ctx):
            if ctx.tool_name == "bash":
                return HookResult(modified_result={"output": "Modified result"})
            return HookResult()

        mock_client.responses.create.return_value = MockLLMResponse(
            output=[
                MockOutputPart(
                    "function_call",
                    name="bash",
                    arguments={"command": "echo hello"},
                    call_id="call_123"
                )
            ]
        )

        gen = coding_agent(
            client=mock_client,
            sbx=mock_sandbox,
            query="Echo hello",
            tools=tools,
            tools_schemas=tools_schemas,
            hook_manager=hook_manager,
            max_steps=1,
        )

        results = list(gen)

        # Find the function_call_output
        outputs = [r for r in results if r[0].get("type") == "function_call_output"]
        assert len(outputs) == 1
        output = json.loads(outputs[0][0]["output"])
        assert output == {"output": "Modified result"}


class TestStopHook:
    """Tests for Stop hook triggering."""

    def test_hook_triggered_on_stop(self, hook_manager, mock_client, mock_sandbox, tools, tools_schemas):
        """Test that Stop hook is triggered when agent finishes."""
        triggered_events = []

        @hook_manager.on(HookEvent.Stop)
        def on_stop(ctx):
            triggered_events.append({
                "event": ctx.event,
                "metadata": ctx.metadata,
            })
            return HookResult()

        gen = coding_agent(
            client=mock_client,
            sbx=mock_sandbox,
            query="Test query",
            tools=tools,
            tools_schemas=tools_schemas,
            hook_manager=hook_manager,
        )

        # Must consume entire generator
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        assert len(triggered_events) == 1
        assert triggered_events[0]["event"] == HookEvent.Stop
        assert "total_steps" in triggered_events[0]["metadata"]
        assert "usage" in triggered_events[0]["metadata"]


class TestGlobalHookManager:
    """Tests for global hook manager functionality."""

    def test_get_hook_manager_creates_singleton(self):
        """Test that get_hook_manager returns a singleton."""
        # Reset global state
        set_hook_manager(None)

        manager1 = get_hook_manager()
        manager2 = get_hook_manager()

        assert manager1 is manager2

    def test_set_hook_manager_replaces_global(self):
        """Test that set_hook_manager replaces the global instance."""
        custom_manager = HookManager()
        set_hook_manager(custom_manager)

        assert get_hook_manager() is custom_manager

        # Clean up
        set_hook_manager(None)


class TestSessionIdPropagation:
    """Tests for session ID propagation to hooks."""

    def test_session_id_passed_to_hooks(self, hook_manager, mock_client, mock_sandbox, tools, tools_schemas):
        """Test that session_id is passed to all hooks."""
        received_session_ids = []

        @hook_manager.on(HookEvent.UserPromptSubmit)
        def on_prompt(ctx):
            received_session_ids.append(("prompt", ctx.session_id))
            return HookResult()

        @hook_manager.on(HookEvent.Stop)
        def on_stop(ctx):
            received_session_ids.append(("stop", ctx.session_id))
            return HookResult()

        gen = coding_agent(
            client=mock_client,
            sbx=mock_sandbox,
            query="Test query",
            tools=tools,
            tools_schemas=tools_schemas,
            hook_manager=hook_manager,
            session_id="test-session-123",
        )

        # Consume generator
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        assert len(received_session_ids) == 2
        assert all(sid == "test-session-123" for _, sid in received_session_ids)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
