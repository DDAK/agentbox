<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen?style=for-the-badge" alt="PRs Welcome">
  <img src="https://img.shields.io/badge/Tests-75%2B%20Passing-success?style=for-the-badge" alt="Tests">
</p>

<h1 align="center">Agentbox</h1>

<p align="center">
  <strong>A production-ready AI coding agent with sandboxed execution</strong>
</p>

<p align="center">
  Build, test, and deploy code using natural language. <br/>
  Supports OpenAI, Anthropic, and Google models with local or Docker sandboxing.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#usage">Usage</a> •
  <a href="#api">API</a> •
  <a href="#architecture">Architecture</a>
</p>

---

## Why Agentbox?

| Feature | Agentbox | Other Tools |
|---------|----------|-------------|
| **Safe Execution** | Docker sandboxing built-in | Often local-only |
| **Multi-Provider LLM** | OpenAI, Anthropic, Google | Usually single provider |
| **Memory & Context** | Automatic compression + persistence | Context window limits |
| **Extensibility** | Hook-based event system | Hard to customize |
| **UI + CLI + API** | All three included | Usually one mode |

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/DDAK/agentbox.git && cd agentbox
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure (add your API key)
cp .env.example .env && nano .env

# Launch!
python main.py
```

Open `http://localhost:7860` and start coding with AI.

---

## Features

### Core Capabilities

- **Multi-Provider LLM Support** — Use GPT-4, Claude, or Gemini interchangeably
- **Dual Sandbox Modes** — Local execution for speed, Docker for isolation
- **13 Built-in Tools** — File operations, code execution, web search, and more
- **Context Management** — Automatic message compression at 60k tokens
- **Session Persistence** — Save and resume coding sessions

### Advanced Features

- **Lifecycle Hooks** — Intercept and modify agent behavior at 5 key points
- **Memory System** — Short-term and long-term memory with automatic retrieval
- **Web Interface** — Gradio-based UI with real-time chat and browser preview
- **CLI Mode** — Headless operation for servers and automation

---

## Usage

### Web UI (Default)

```bash
python main.py                          # Local sandbox
python main.py --sandbox=docker         # Docker sandbox (isolated)
python main.py --model=claude-3-5-sonnet  # Use Claude
```

### CLI Mode

```bash
python main.py --cli
```

**CLI Commands:**
- `exit` / `quit` — Exit the agent
- `help` — Show available commands
- `pwd` — Show working directory

### All Options

```bash
python main.py --help

Options:
  --sandbox {local,docker}  Execution environment (default: local)
  --working-dir PATH        Working directory for sandbox
  --docker-image IMAGE      Docker image (default: python:3.12-slim)
  --model MODEL             LLM model (default: gpt-4.1-mini)
  --max-steps N             Max agent iterations (default: 100)
  --cli                     Run in CLI mode
  --no-share                Don't create public Gradio link
  --cleanup                 Clear all sandboxes and exit
```

---

## API

### Programmatic Usage

```python
from agent import create_agent

# Create an agent
agent = create_agent(
    sandbox_type="docker",
    working_dir="/path/to/project",
    model="gpt-4.1-mini",
    max_steps=100
)

# Run a task
messages, usage = agent.run_with_logging("Create a REST API with FastAPI")

# Or launch the UI
agent.launch_ui(share=True)

# Cleanup
agent.cleanup()
```

### CodingAgent Methods

| Method | Description |
|--------|-------------|
| `setup_sandbox()` | Initialize the sandbox environment |
| `run(query)` | Run agent with query (returns generator) |
| `run_with_logging(query)` | Run with console output |
| `launch_ui(share)` | Launch Gradio web interface |
| `cleanup()` | Free resources and kill sandbox |

### Available Tools

| Tool | Description |
|------|-------------|
| `execute_code` | Execute Python code in sandbox |
| `execute_bash` | Execute bash commands |
| `list_directory` | List directory contents (paginated) |
| `read_file` | Read file contents |
| `write_file` | Write/create files |
| `replace_in_file` | Search and replace |
| `search_file_content` | Search with regex/literal/fuzzy |
| `glob_search` | Find files by pattern |
| `web_search` | DuckDuckGo search |
| `web_fetch` | Fetch URL content |

---

## Supported Models

| Provider | Models |
|----------|--------|
| **OpenAI** | `gpt-4.1-mini`, `gpt-4o`, `gpt-4o-mini`, `gpt-5-mini`, `gpt-5-nano` |
| **Anthropic** | `claude-3-5-sonnet`, `claude-3-5-haiku`, `claude-3-opus`, `claude-sonnet-4` |
| **Google** | `gemini-1.5-pro`, `gemini-1.5-flash`, `gemini-2.0-flash-exp` |

---

## Architecture

```
User Input (CLI or Gradio UI)
         │
         ▼
    CodingAgent
    ├─ Manages sandbox lifecycle
    ├─ Configures model & prompts
    └─ Triggers lifecycle hooks
         │
         ▼
    Agent Loop
    ├─ SessionStart hook (load context)
    ├─ UserPromptSubmit hook (retrieve memories)
    ├─ Compress messages if >60k tokens
    ├─ Call LLM with tool schemas
    │   │
    │   ├─ PreToolUse hook (can block/modify)
    │   ├─ Execute tool
    │   └─ PostToolUse hook (can modify results)
    │
    └─ Stop hook (checkpoint state)
         │
         ▼
    Sandbox Execution
    ├─ LocalSandbox (direct, fast)
    └─ DockerSandbox (isolated, safe)
```

### Project Structure

```
agentbox/
├── main.py              # CLI entry point
├── agent.py             # CodingAgent class
├── lib/
│   ├── coding_agent.py  # Core agent loop
│   ├── sandbox.py       # Local & Docker sandboxes
│   ├── tools.py         # 13 tool implementations
│   ├── llm_client.py    # Multi-provider LLM interface
│   ├── ui.py            # Gradio web interface
│   ├── hooks/           # Lifecycle hooks system
│   └── memory/          # Session & memory management
└── tests/               # 75+ tests
```

---

## Lifecycle Hooks

Extend agent behavior without modifying core code:

```python
from lib.hooks import HookManager, HookEvent

hooks = HookManager()

@hooks.register(HookEvent.PRE_TOOL_USE)
def block_dangerous_commands(context):
    if "rm -rf" in str(context.data.get("args", {})):
        return {"blocked": True, "reason": "Dangerous command blocked"}
    return {"blocked": False}
```

**Available Hooks:**
- `SessionStart` — Initialize/resume session
- `UserPromptSubmit` — Before LLM call (can block)
- `PreToolUse` — Before tool execution (can block/modify)
- `PostToolUse` — After tool execution (can modify results)
- `Stop` — Agent finishes turn (checkpoint)

---

## Memory System

Agentbox includes persistent memory across sessions:

- **Short-term Memory** — Recent context and observations
- **Long-term Memory** — Important facts and code patterns
- **Automatic Retrieval** — Relevant memories injected into prompts
- **Checkpointing** — State saved at configurable intervals

---

## Security

| Sandbox | Isolation | Use Case |
|---------|-----------|----------|
| **Local** | None | Trusted code, development |
| **Docker** | Container | Untrusted code, production |

**Security Features:**
- Path validation (sandbox escape prevention)
- 300-second execution timeout
- Working directory confinement

---

## Installation

### Prerequisites

- Python 3.12+
- Docker (optional, for isolated execution)
- API key (OpenAI, Anthropic, or Google)

### Steps

```bash
# 1. Clone
git clone https://github.com/DDAK/agentbox.git
cd agentbox

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
# or: uv sync

# 4. Configure API key
cp .env.example .env
# Add your OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>If Agentbox helps you, consider giving it a star!</strong><br/>
  <a href="https://github.com/DDAK/agentbox/stargazers">
    <img src="https://img.shields.io/github/stars/DDAK/agentbox?style=social" alt="GitHub Stars">
  </a>
</p>
