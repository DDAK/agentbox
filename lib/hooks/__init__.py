"""
Lifecycle Hooks Module

This module provides a deterministic hook system for the agent lifecycle.
Hooks are procedural code (Python callables) that run at specific points in the execution flow.

Available Hook Events:
- SessionStart: When the agent initializes or a session is cleared
- UserPromptSubmit: After user input, but before sending to LLM
- PreToolUse: Before executing a tool (e.g., bash, write_file)
- PostToolUse: Immediately after a tool finishes
- Stop: When the agent finishes its turn

Example usage:
    from lib.hooks import HookManager, HookEvent

    hooks = HookManager()

    @hooks.on(HookEvent.SessionStart)
    def on_session_start(context):
        print(f"Session started at {context.get('timestamp')}")

    @hooks.on(HookEvent.PreToolUse)
    def block_dangerous_commands(context):
        if context.get('tool_name') == 'bash':
            cmd = context.get('arguments', {}).get('command', '')
            if 'rm -rf /' in cmd:
                return {'block': True, 'reason': 'Dangerous command blocked'}
        return None
"""

from .types import HookEvent, HookContext, HookResult
from .manager import HookManager

__all__ = ['HookEvent', 'HookContext', 'HookResult', 'HookManager']
