# Add Lifecycle Hooks: 
## ADD The Event Loop
Hooks are deterministic triggers. Unlike the LLM’s reasoning, hooks are procedural code (Bash or Python) that runs at specific points in the execution flow:

| Hook Event | Technical Trigger Point |
|------------|-------------------------|
| SessionStart | When the CLI boots or a session is cleared. |
| UserPromptSubmit | After you hit Enter, but before the prompt is sent to the LLM. |
| PreToolUse | Before agentbox executes a tool (e.g., bash, write_file). Can be used to block dangerous commands. |
| PostToolUse | Immediately after a tool finishes. This is where auto-formatters (Prettier) or auto-testers run. |
| Stop | When agentbox finishes its turn and prepares to hand control back to the user. |

## ADD Trigger Point
SessionStart:	When the CLI boots or a session is cleared.
UserPromptSubmit:	After you hit Enter, but before the prompt is sent to the LLM.
PreToolUse:	Before agentbox executes a tool (e.g., bash, write_file). Can be used to block dangerous commands.
PostToolUse:	Immediately after a tool finishes. This is where auto-formatters (Prettier) or auto-testers run.
Stop:	When agentbox finishes its turn and prepares to hand control back to the user.

## ADD tests
This test the hooks and triggers

## MARK DONE WHEN
1. When test are a success
2. Hooks work and are triggered successfuly

---

## Implementation Progress

### Iteration 1: Created hooks module with types and manager [COMPLETED]
- Created `lib/hooks/` module with:
  - `__init__.py` - Module initialization and exports
  - `types.py` - HookEvent enum, HookContext and HookResult dataclasses
  - `manager.py` - HookManager class for registering and triggering hooks

**Features implemented:**
- HookEvent enum with all 5 event types (SessionStart, UserPromptSubmit, PreToolUse, PostToolUse, Stop)
- HookContext dataclass with event-specific fields (tool_name, arguments, result, query, etc.)
- HookResult dataclass with blocking, modification, and metadata support
- HookManager with:
  - Decorator-based hook registration (`@hooks.on(event)`)
  - Direct registration (`hooks.register()`)
  - Script hook support (bash/python scripts)
  - Hook triggering with context merging
  - Result merging for multiple handlers

### Iteration 2: Integrated hooks into coding_agent and added tests [COMPLETED]
- Integrated hooks into `lib/coding_agent.py`:
  - Added `get_hook_manager()` and `set_hook_manager()` for global hook manager access
  - Added `hook_manager` and `session_id` parameters to `coding_agent()` function
  - Added `UserPromptSubmit` hook trigger after query received (with blocking support)
  - Added `PreToolUse` hook trigger before tool execution (with blocking and argument modification)
  - Added `PostToolUse` hook trigger after tool execution (with result modification)
  - Added `Stop` hook trigger when agent finishes

- Created comprehensive test suite:
  - `tests/test_hooks.py` - 41 unit tests for hooks module
  - `tests/test_hooks_integration.py` - 11 integration tests for hooks in coding_agent

**All 52 tests passing!**

**Hook Integration Points:**
| Hook Event | Integration Point | Capabilities |
|------------|-------------------|--------------|
| UserPromptSubmit | After query received, before LLM call | Can block requests |
| PreToolUse | Before execute_tool() | Can block or modify arguments |
| PostToolUse | After execute_tool() | Can modify results |
| Stop | When agent loop completes | Logging, cleanup |

**Usage Example:**
```python
from lib import HookManager, HookEvent, HookResult, get_hook_manager

hooks = get_hook_manager()

@hooks.on(HookEvent.PreToolUse)
def block_dangerous_commands(ctx):
    if ctx.tool_name == "bash" and "rm -rf" in ctx.arguments.get("command", ""):
        return HookResult.deny("Dangerous command blocked")
    return HookResult.allow()
```