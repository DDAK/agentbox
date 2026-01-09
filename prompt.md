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

### Iteration 1: Created hooks module with types and manager
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

**Next steps:**
- Integrate hooks into CodingAgent class
- Integrate hooks into coding_agent loop
- Write tests