# Universal AI Agent Codebase Research

**Date**: 2025-12-11 (Updated with Memory System)
**Repository**: coding_agents

---

## Summary

The `coding_agents` codebase is a **universal AI agent** that supports multiple LLM providers (OpenAI, Anthropic, Google Gemini) via LiteLLM with a tool-augmented "Reason then Act" agent loop. It can handle:

- **Research**: Web search (DuckDuckGo) and content fetching
- **Coding**: Python code and bash command execution
- **File Operations**: Read, write, search, and modify files

It supports both local and Docker sandbox environments for safe code execution, with a Gradio web interface or CLI mode for user interaction.

---

## Architecture

```
coding_agents/
├── main.py              # Entry point with CLI argument parsing
├── agent.py             # CodingAgent class - main interface
├── helper.py            # Environment and API key utilities
├── plan.md              # Implementation plan for memory system
└── lib/
    ├── __init__.py      # Module exports
    ├── coding_agent.py  # Core agent loop with tool execution
    ├── llm_client.py    # LiteLLM abstraction layer
    ├── model_config.py  # Model registry and provider config
    ├── sandbox.py       # Local and Docker sandbox implementations
    ├── tools.py         # Tool implementations (file ops, code exec, web research)
    ├── tools_schemas.py # OpenAI function calling schemas
    ├── ui.py            # Gradio web interface
    ├── prompts.py       # System prompts for different modes
    ├── logger.py        # Rich-based logging utilities
    ├── utils.py         # Sandbox factory utilities
    └── memory/          # Persistent memory system (NEW)
        ├── __init__.py          # Package exports
        ├── manager.py           # MemoryManager - central orchestrator
        ├── session.py           # SessionManager - session lifecycle
        ├── integration.py       # Agent loop integration functions
        ├── types/
        │   ├── __init__.py
        │   ├── base.py          # BaseMemory abstract class
        │   ├── short_term.py    # ShortTermMemory - FIFO buffer
        │   └── long_term.py     # LongTermMemory - markdown knowledge
        └── persistence/
            ├── __init__.py
            ├── markdown_store.py  # MarkdownMemoryStore - file backend
            └── checkpoint.py      # CheckpointManager - session snapshots
```

---

## Components

### 1. Entry Point: main.py

**Purpose**: CLI entry point that parses arguments and launches the agent.

**Key Functions**:
- `main()` - Parses arguments, creates agent, launches UI or CLI
- `run_cli_mode(agent)` - Interactive command-line loop

**CLI Options**:
| Option | Description |
|--------|-------------|
| `--sandbox` | `local` (default) or `docker` |
| `--working-dir` | Working directory for sandbox |
| `--model` | LLM model (default: `gpt-4.1-mini`) - see Supported Models |
| `--max-steps` | Maximum agent steps (default: 100) |
| `--cli` | Run in CLI mode instead of UI |
| `--memory-path` | Path for persistent memory storage (default: `.agent_memory`) |
| `--no-persistence` | Disable persistent memory |
| `--resume` | Session ID to resume from previous run |
| `--checkpoint-interval` | Steps between checkpoints (default: 100) |

**Session Management Subcommands**:
```bash
python main.py sessions list              # List available sessions
python main.py sessions show <session_id> # Show session details
python main.py sessions delete <session_id> # Delete a session
```

---

### 2. CodingAgent Class: agent.py

**Purpose**: Main user-facing interface for the universal AI agent. Manages agent lifecycle, sandbox, and persistent memory. Despite the name, it now handles research, coding, and file operations.

**Initialization**:
```python
CodingAgent(
    sandbox_type="local",      # "local" or "docker"
    working_dir=None,          # defaults to current directory
    model="gpt-4.1-mini",      # supports OpenAI, Anthropic, Gemini
    max_steps=100,
    system_prompt=None,
    docker_image=None,
    # Memory/Session parameters (NEW)
    memory_path=".agent_memory",     # Path for persistent memory storage
    enable_persistence=True,         # Enable/disable persistent memory
    resume_session=None,             # Session ID to resume
    checkpoint_interval=100,         # Steps between checkpoints
)
```

**Methods**:
| Method | Description |
|--------|-------------|
| `setup_sandbox()` | Create and configure sandbox environment |
| `run(query)` | Run agent with query, returns generator |
| `run_with_logging(query)` | Run with console logging |
| `launch_ui()` | Launch Gradio web interface |
| `cleanup()` | Kill sandbox and free resources |
| `list_sessions()` | List available sessions (NEW) |
| `get_current_session()` | Get current session ID (NEW) |
| `end_session(summary)` | End session with optional summary (NEW) |

---

### 3. Core Agent Loop: lib/coding_agent.py

**Purpose**: The heart of the system - implements the "Reason then Act" cycle with integrated memory hooks.

#### Function Signature
```python
def coding_agent(
    client,                              # LLM client with .responses.create()
    sbx: BaseSandbox,
    query: str,
    tools: dict[str, Callable],
    tools_schemas: list[dict],
    max_steps: int = 5,
    system: Optional[str] = None,
    messages: Optional[list[dict]] = None,
    model: str = "gpt-4.1-mini",
    # Memory parameters (NEW)
    memory_manager=None,                 # MemoryManager instance for persistence
    checkpoint_interval: int = 100,      # Steps between checkpoints
) -> Generator[tuple[dict, dict, int], None, tuple[list[dict], int]]
```

#### Loop Execution Steps

1. **Receive Query** - User message appended to `messages` list
2. **Retrieve Memories** - (NEW) Inject relevant past memories into context
3. **Maybe Compress** - If tokens > 42k, compress older messages into state snapshot
4. **Call LLM API** - Send messages + tool schemas via `client.responses.create()`
5. **Process Response** - For each part in response:
   - If `function_call`: execute tool via `execute_tool()`
   - **Store Observation** - (NEW) Save tool result to memory
   - Yield `(part_dict, messages, usage)` tuple
6. **Checkpoint** - (NEW) Save state every `checkpoint_interval` steps
7. **Loop Control** - Continue until no tool calls or `max_steps` reached

#### Memory Integration Points

```
┌─────────────────────────────────────────────────────────┐
│                  AGENT LOOP WITH MEMORY                  │
│                                                          │
│  while steps < max_steps:                                │
│      │                                                   │
│      ├─▶ retrieve_context()        # Get relevant past  │
│      │   └─▶ inject_memories()     # Add to messages    │
│      │                                                   │
│      ├─▶ maybe_compress_messages()                       │
│      ├─▶ client.responses.create()                       │
│      │                                                   │
│      ├─▶ for part in response.output:                    │
│      │       └─▶ if function_call:                       │
│      │               execute_tool()                      │
│      │               extract_observation()  # NEW        │
│      │               add_memory()           # NEW        │
│      │                                                   │
│      └─▶ if should_checkpoint(step):       # NEW        │
│              checkpoint(messages, task, progress)        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

#### Context Compression

When token usage exceeds 42k (70% of 60k limit):
- Compresses oldest messages using provider-specific compression model
- Creates XML state snapshot with: goal, key knowledge, file state, actions, plan

---

### 4. LLM Client Abstraction: lib/llm_client.py (NEW)

**Purpose**: Provides a unified interface for multiple LLM providers via LiteLLM.

**Key Classes**:
| Class | Description |
|-------|-------------|
| `LLMClient` | Main client that wraps LiteLLM with response normalization |
| `ResponsesAPI` | Provides `client.responses.create()` interface for backward compatibility |
| `NormalizedResponse` | Wraps LiteLLM response to match expected format |
| `ToolCall` | Normalized tool call with `.type`, `.name`, `.arguments`, `.call_id` |

**Factory Function**:
```python
from lib.llm_client import create_llm_client

client = create_llm_client(model="claude-3-5-sonnet")
response = client.responses.create(
    model="claude-3-5-sonnet",
    input=[...],
    tools=[...]
)
```

**Message Format Conversion**:
- Converts `"developer"` role to `"system"` for non-OpenAI providers
- Converts `function_call`/`function_call_output` to LiteLLM's tool format
- Normalizes LiteLLM responses back to expected format

---

### 5. Model Configuration: lib/model_config.py (NEW)

**Purpose**: Registry of supported models and their provider-specific configurations.

**ModelConfig Fields**:
| Field | Description |
|-------|-------------|
| `provider` | `"openai"`, `"anthropic"`, or `"gemini"` |
| `litellm_model` | LiteLLM model identifier (e.g., `"anthropic/claude-3-5-sonnet-20241022"`) |
| `supports_tool_calling` | Whether native tool calling is supported |
| `context_window` | Token limit for the model |
| `system_role` | `"developer"` for OpenAI, `"system"` for others |
| `compression_model` | Model to use for context compression |

**Usage**:
```python
from lib.model_config import get_model_config

config = get_model_config("claude-3-5-sonnet")
# config.litellm_model = "anthropic/claude-3-5-sonnet-20241022"
# config.system_role = "system"
```

---

### 6. Tools System: lib/tools.py

**Purpose**: Implements all tools for file operations, code execution, and web research.

#### Available Tools

| Tool | Description |
|------|-------------|
| `execute_code` | Execute Python code in sandbox |
| `execute_bash` | Execute bash commands in sandbox |
| `list_directory` | List directory contents with pagination |
| `read_file` | Read file content with offset/limit |
| `write_file` | Write content to file, creates directories |
| `replace_in_file` | Search and replace in file |
| `search_file_content` | Search files (literal/regex/fuzzy) |
| `glob` | Find files by glob pattern |
| `web_search` | Search the web using DuckDuckGo (no API key required) |
| `web_fetch` | Fetch and extract content from URLs as markdown |

#### Security

`secure_path()` function ensures all paths stay within working directory, preventing directory traversal attacks.

---

### 7. Tool Schemas: lib/tools_schemas.py

**Purpose**: Defines OpenAI-compatible function-calling JSON schemas for each tool.

Exports `tools_schemas` list containing all 10 tool schemas in OpenAI function format. LiteLLM handles converting these to provider-specific formats.

---

### 8. Sandbox System: lib/sandbox.py

**Purpose**: Provides isolated environments for code execution.

#### LocalSandbox
- Executes directly on host machine
- No isolation - use with caution
- 300 second timeout

#### DockerSandbox
- Executes in Docker container
- Default image: `python:3.12-slim`
- Mounts working directory at `/workspace`
- Host network mode

---

### 9. User Interface: lib/ui.py

**Purpose**: Gradio-based web interface for the agent.

**Features**:
- Chat window with message history
- Tool call visualization with collapsible panels
- AIContext panel showing raw messages
- Optional browser preview

---

### 10. System Prompts: lib/prompts.py

**Available Prompts**:
| Prompt | Purpose |
|--------|---------|
| `SYSTEM_PROMPT_UNIVERSAL` | Universal agent prompt (default) - handles research, coding, and file ops |
| `SYSTEM_PROMPT_COMPRESS_MESSAGES` | Compress history into state snapshot |
| `SYSTEM_PROMPT_WEB_DEV` | Next.js/TypeScript/Tailwind development |

---

### 11. Memory System: lib/memory/ (NEW)

**Purpose**: Provides persistent memory storage and session management for long-running, resumable agents.

#### Components

| Component | File | Purpose |
|-----------|------|---------|
| `MemoryManager` | `manager.py` | Central orchestrator for all memory operations |
| `SessionManager` | `session.py` | Session lifecycle and checkpoint management |
| `ShortTermMemory` | `types/short_term.py` | FIFO buffer for recent observations |
| `LongTermMemory` | `types/long_term.py` | Persistent knowledge storage (facts, patterns, learnings) |
| `MarkdownMemoryStore` | `persistence/markdown_store.py` | Markdown file backend |
| `CheckpointManager` | `persistence/checkpoint.py` | Session state snapshots |

#### Storage Structure

```
.agent_memory/
├── index.md                 # Quick lookup index
├── sessions/
│   └── session_xxx.md       # Session checkpoints (YAML frontmatter)
├── long_term/
│   ├── knowledge.md         # Facts, patterns, API info
│   └── learnings.md         # Error resolutions, successful approaches
└── short_term/
    └── buffer.md            # Recent observations (FIFO)
```

#### Memory Types

| Type | Storage | Purpose |
|------|---------|---------|
| `observation` | Short-term buffer | Tool results, recent actions |
| `fact` | `knowledge.md` | Project facts and configuration |
| `pattern` | `knowledge.md` | Discovered patterns and best practices |
| `api_info` | `knowledge.md` | API endpoints, configurations |
| `learning` | `learnings.md` | Error resolutions, successful approaches |

#### MemoryManager API

```python
from lib.memory import MemoryManager

manager = MemoryManager(
    storage_path=".agent_memory",
    llm_client=client,           # For LLM-based summarization
    short_term_capacity=100,     # FIFO buffer size
    consolidation_threshold=50,  # Trigger consolidation at this count
)

# Add memories
manager.add_memory("Found main.py in project root", memory_type="observation")
manager.add_memory("Project uses Python 3.12", memory_type="fact")
manager.add_memory("Use absolute imports for this codebase", memory_type="learning")

# Retrieve relevant context
memories = manager.retrieve_context(query="What files are in this project?", top_k=10)

# Session management
session_id = manager.start_session(task="Implement authentication feature")
manager.checkpoint(step=100, messages=messages, task="Auth feature", progress="JWT done")
manager.end_session(summary="Completed JWT auth implementation")

# Resume session
session_id = manager.start_session(resume_id="session_abc123")
state = manager.restore_session("session_abc123")
```

#### Integration Functions (lib/memory/integration.py)

```python
from lib.memory.integration import inject_memories, extract_observation, should_checkpoint

# Inject memories into conversation
messages = inject_memories(messages, relevant_memories)

# Extract observation from tool result
observation = extract_observation("execute_code", result, arguments)

# Check if checkpoint is needed
if should_checkpoint(step=150, interval=100):
    manager.checkpoint(...)
```

---

### 12. Supporting Files

#### lib/logger.py
Rich-based logging with emoji indicators (✨ INFO, ❌ ERROR, 🤖 TOOL)

#### lib/utils.py
Sandbox factory utilities: `create_sandbox()`, `clear_sandboxes()`

#### helper.py
Environment loading and API key retrieval functions:
- `setup_api_keys_for_litellm()` - Load all API keys for LiteLLM
- `get_api_key_for_model(model)` - Get provider-specific API key
- `get_openai_api_key()`, `get_anthropic_api_key()`, `get_google_api_key()`

---

## Component Interaction Diagram

```
┌─────────────────┐
│   User Input    │  (CLI or Gradio UI)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   main.py       │  Argument parsing, mode selection
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  CodingAgent    │────▶│     ui.py       │     │  lib/memory/    │
│   (agent.py)    │     └─────────────────┘     │  MemoryManager  │
└────────┬────────┘                             └────────┬────────┘
         │                                               │
         ▼                                               │
┌─────────────────┐     ┌─────────────────┐              │
│  llm_client.py  │────▶│ model_config.py │              │
│ (LiteLLM wrap)  │     │ (model registry)│              │
└────────┬────────┘     └─────────────────┘              │
         │                                               │
         ▼                                               │
┌─────────────────┐                                      │
│ coding_agent()  │ ◀──── LiteLLM ─────────────────┬─────┘
│ (Agent Loop)    │ ◀──── Memory retrieval ────────┘
└────────┬────────┘
         │ executes tools
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    tools.py     │────▶│   sandbox.py    │     │  .agent_memory/ │
│ (10 tools)      │     │ (code execution)│     │ (markdown files)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## Agent Loop Detail

```
┌─────────────────────────────────────────────────────────┐
│                    AGENT LOOP                           │
│                                                         │
│  while steps < max_steps:                               │
│      │                                                  │
│      ├─▶ retrieve_context()         # Get past memories │
│      │   └─▶ inject_memories()      # Add to context    │
│      │                                                  │
│      ├─▶ maybe_compress_messages()  # Token management  │
│      │                                                  │
│      ├─▶ client.responses.create()  # LiteLLM API call  │
│      │                                                  │
│      ├─▶ for part in response.output:                   │
│      │       │                                          │
│      │       ├─▶ yield (part, messages, usage)          │
│      │       │                                          │
│      │       └─▶ if function_call:                      │
│      │               execute_tool()                     │
│      │               extract_observation()  # NEW       │
│      │               add_memory()           # NEW       │
│      │               yield (result, messages, usage)    │
│      │                                                  │
│      ├─▶ if should_checkpoint(step): checkpoint()  # NEW│
│      │                                                  │
│      └─▶ if no function_calls: break                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Memory Management

The agent uses a **two-tier memory system** combining:
1. **In-session message history** with automatic context compression
2. **Persistent markdown-based memory** for cross-session knowledge (NEW)

### Message History

All conversation turns are stored in a `messages` list that persists across the session:

```python
messages = [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"type": "function_call", "name": "...", "arguments": "...", "call_id": "..."},
    {"type": "function_call_output", "call_id": "...", "output": "..."},
    {"type": "message", "content": [{"type": "text", "text": "..."}]},
    # ... continues growing with each turn
]
```

### Context Compression

When token usage exceeds the threshold, older messages are compressed into a state snapshot:

| Setting | Value |
|---------|-------|
| `TOKEN_LIMIT` | 60,000 tokens |
| `COMPRESS_THRESHOLD` | 0.7 (70%) |
| **Trigger Point** | 42,000 tokens |

#### Compression Flow

```
┌─────────────────────────────────────────────────────────┐
│               COMPRESSION PROCESS                        │
│                                                          │
│  1. Check: usage > TOKEN_LIMIT * 0.7 (42k)?             │
│     └─▶ No: Return messages unchanged                    │
│     └─▶ Yes: Continue to step 2                          │
│                                                          │
│  2. Find compression index (oldest 70% of messages)      │
│                                                          │
│  3. Split messages:                                      │
│     ├─▶ to_compress: messages[0:compress_index]         │
│     └─▶ to_keep: messages[compress_index:]              │
│                                                          │
│  4. Call compression model with to_compress messages     │
│     └─▶ Uses provider-specific compression_model         │
│                                                          │
│  5. Extract <state_snapshot> from response               │
│                                                          │
│  6. Return: [snapshot_messages] + to_keep               │
└─────────────────────────────────────────────────────────┘
```

#### State Snapshot Format

The compression model generates an XML state snapshot:

```xml
<state_snapshot>
  <goal>What the user is trying to accomplish</goal>
  <key_knowledge>
    - Important facts discovered
    - File locations and structures
    - API endpoints or configurations
  </key_knowledge>
  <file_state>
    - Files created or modified
    - Current file contents summary
  </file_state>
  <actions_taken>
    - Tools called and their results
    - Code executed and outputs
  </actions_taken>
  <next_steps>
    - Remaining tasks
    - Current plan
  </next_steps>
</state_snapshot>
```

#### Compression Models by Provider

| Provider | Main Model | Compression Model |
|----------|------------|-------------------|
| OpenAI | `gpt-4o` | `gpt-4o-mini` |
| OpenAI | `gpt-4.1-mini` | `gpt-4o-mini` |
| Anthropic | `claude-3-5-sonnet` | `claude-3-5-haiku-20241022` |
| Google | `gemini-1.5-pro` | `gemini/gemini-1.5-flash` |

### Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `maybe_compress_messages()` | `lib/coding_agent.py:103` | Check if compression needed, trigger if so |
| `compress_messages()` | `lib/coding_agent.py:32` | Call compression model, extract snapshot |
| `get_compress_message_index()` | `lib/coding_agent.py:79` | Calculate where to split messages |
| `format_messages()` | `lib/coding_agent.py:63` | Format messages for compression prompt |

### Persistent Memory System (NEW)

In addition to context compression, the agent now supports **persistent markdown-based memory** for:
- Cross-session knowledge retention
- Session checkpointing and resume
- Long-term learning and pattern storage

#### Persistent Memory Features

| Feature | Description |
|---------|-------------|
| **Short-term buffer** | Recent observations in FIFO buffer (`buffer.md`) |
| **Long-term knowledge** | Facts, patterns, API info (`knowledge.md`) |
| **Learnings** | Error resolutions, successful approaches (`learnings.md`) |
| **Session checkpoints** | Messages and state at intervals (`sessions/`) |

#### Memory Retrieval Flow

```
┌─────────────────────────────────────────────────────────┐
│              MEMORY RETRIEVAL PROCESS                    │
│                                                          │
│  1. User query received                                  │
│                                                          │
│  2. Retrieve relevant memories:                          │
│     ├─▶ Recent observations from short-term buffer       │
│     └─▶ Knowledge matching query keywords                │
│                                                          │
│  3. Inject memories as <assistant_notes> in messages     │
│                                                          │
│  4. LLM has access to past relevant experience           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

#### Session Checkpoint Format

Session checkpoints use YAML frontmatter in markdown:

```markdown
---
id: session_abc123
started: 2024-01-15T10:30:00
updated: 2024-01-15T14:45:00
step: 150
status: active
---

# Session: session_abc123

## Current Task
Implementing user authentication feature

## Progress
- JWT token generation complete
- Working on refresh token implementation

## Compressed State
<compressed_messages>
[Base64 encoded JSON messages]
</compressed_messages>
```

### Memory Characteristics

| Aspect | In-Session | Persistent Memory |
|--------|------------|-------------------|
| **Persistence** | Within session only | Across sessions (NEW) |
| **Storage** | In-memory list | Markdown files (NEW) |
| **Compression** | Lossy (LLM summary) | Lossless checkpoints (NEW) |
| **Automatic** | Context compression | Observation storage, checkpoints (NEW) |
| **Resume** | Not supported | `--resume <session_id>` (NEW) |

---

## Supported Models

### OpenAI
| Model | LiteLLM ID |
|-------|------------|
| `gpt-4.1-mini` | `gpt-4.1-mini` |
| `gpt-4o` | `gpt-4o` |
| `gpt-4o-mini` | `gpt-4o-mini` |
| `gpt-5-mini` | `gpt-5-mini` |
| `gpt-5-nano` | `gpt-5-nano` |

### Anthropic
| Model | LiteLLM ID |
|-------|------------|
| `claude-3-5-sonnet` | `anthropic/claude-3-5-sonnet-20241022` |
| `claude-3-5-haiku` | `anthropic/claude-3-5-haiku-20241022` |
| `claude-3-opus` | `anthropic/claude-3-opus-20240229` |

### Google Gemini
| Model | LiteLLM ID |
|-------|------------|
| `gemini-1.5-pro` | `gemini/gemini-1.5-pro` |
| `gemini-1.5-flash` | `gemini/gemini-1.5-flash` |
| `gemini-2.0-flash` | `gemini/gemini-2.0-flash-exp` |

---

## Configuration Defaults

| Setting | Default |
|---------|---------|
| Model | `gpt-4.1-mini` |
| Max Steps | 100 |
| Token Limit | 60,000 |
| Compress Threshold | 70% (42,000 tokens) |
| Execution Timeout | 300 seconds |
| Docker Image | `python:3.12-slim` |
| Memory Path | `.agent_memory` (NEW) |
| Enable Persistence | `true` (NEW) |
| Checkpoint Interval | 100 steps (NEW) |
| Short-term Capacity | 100 items (NEW) |
| Consolidation Threshold | 50 items (NEW) |

---

## Environment Variables

| Variable | Provider | Required |
|----------|----------|----------|
| `OPENAI_API_KEY` | OpenAI | For OpenAI models |
| `ANTHROPIC_API_KEY` | Anthropic | For Claude models |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Google | For Gemini models |

---

## Security Considerations

1. **Path Security**: `secure_path()` prevents directory traversal
2. **Local Sandbox**: No isolation - executes with user's permissions
3. **Docker Sandbox**: Container isolation with mounted working directory
4. **Timeout**: 300-second limit prevents runaway processes

---

## Dependencies

| Package | Purpose |
|---------|---------|
| litellm | Multi-provider LLM abstraction |
| openai | OpenAI API client (used by LiteLLM) |
| gradio | Web UI framework |
| python-dotenv | Environment variables |
| rich | Terminal formatting |
| tiktoken | Token counting |
| duckduckgo-search | Web search via DuckDuckGo (no API key) |
| beautifulsoup4 | HTML parsing for web_fetch |
| html2text | HTML to markdown conversion |
| requests | HTTP requests for web_fetch |
| pyyaml | YAML frontmatter parsing for session files (NEW) |

---

## Usage Examples

### Research Tasks
```bash
python main.py --cli
> What are the latest features in Python 3.13?
> Research best practices for REST API design
```

### Coding Tasks
```bash
python main.py --cli
> Create a Python script that fetches weather data from an API
> Find all TODO comments in the codebase and list them
```

### Mixed Research + Coding
```bash
python main.py --cli
> Research OAuth 2.0 implementation best practices and create a sample Flask app
> Look up the BeautifulSoup documentation and write a web scraper
```

### Using Different Models
```bash
# Use Claude for research
python main.py --model claude-3-5-sonnet --cli

# Use Gemini for general tasks
python main.py --model gemini-2.0-flash --cli

# Use GPT-4o for coding
python main.py --model gpt-4o --cli
```

### Session Management (NEW)
```bash
# List all available sessions
python main.py sessions list

# Show details of a specific session
python main.py sessions show session_abc123

# Resume a previous session
python main.py --resume session_abc123 --cli

# Delete an old session
python main.py sessions delete session_abc123

# Disable persistent memory (for one-off tasks)
python main.py --no-persistence --cli

# Custom memory storage path
python main.py --memory-path ./my_project_memory --cli

# Adjust checkpoint frequency
python main.py --checkpoint-interval 50 --cli
```
