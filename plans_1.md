# General-Purpose Agent Plan

Transform the coding-focused agent into a general-purpose agent with configurable agent types.

---

## Research Summary

Based on analysis of Manus AI, Claude Code, Replit, and Perplexity:

| Capability | Manus | Claude Code | Replit | Current Agent |
|------------|-------|-------------|--------|---------------|
| File operations | ✓ | ✓ | ✓ | ✓ |
| Code execution | ✓ | ✓ | ✓ | ✓ |
| Web search | ✓ | ✓ | ✓ | ✗ |
| Web fetch/browse | ✓ | ✓ | ✓ | ✗ |
| Browser automation | ✓✓✓ | ✗ | ✓ | ✗ |
| User messaging | ✓ | ✗ | ✗ | ✗ |
| Deployment | ✓ | ✗ | ✓ | ✗ |

**Key gaps in current agent:** Web search, web fetch, user interaction tools.

---

## Proposed Architecture

### Option A: Agent Types (Recommended)

Pass `agent_type` parameter to load specific prompts and tools:

```python
CodingAgent(
    agent_type="general",  # "coding", "research", "web_dev", "general"
    model="claude-3-5-sonnet",
    ...
)
```

Each agent type has:
- Specific system prompt
- Specific tool set
- Shared core (LLM client, sandbox, agent loop)

### Option B: Universal Agent

Single agent with all tools, universal prompt. Simpler but less focused.

---

## Implementation Plan

### Phase 1: Add New Tools

#### 1.1 Web Search Tool
```python
web_search_schema = {
    "type": "function",
    "name": "web_search",
    "description": "Search the web for information. Returns titles, URLs, and snippets.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "num_results": {"type": "integer", "description": "Number of results (default: 5, max: 10)"},
        },
        "required": ["query"],
    },
}
```

Implementation options:
- **SerpAPI** - Paid, reliable
- **DuckDuckGo** - Free, `duckduckgo-search` package
- **Tavily** - AI-focused search API

#### 1.2 Web Fetch Tool
```python
web_fetch_schema = {
    "type": "function",
    "name": "web_fetch",
    "description": "Fetch and extract content from a URL. Returns markdown-formatted text.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "extract_mode": {"type": "string", "enum": ["full", "main_content", "summary"]},
        },
        "required": ["url"],
    },
}
```

Implementation: `requests` + `beautifulsoup4` + `html2text` for markdown conversion.

#### 1.3 User Messaging Tools (optional)
```python
notify_user_schema = {
    "type": "function",
    "name": "notify_user",
    "description": "Send a non-blocking notification to the user.",
    "parameters": {...}
}

ask_user_schema = {
    "type": "function",
    "name": "ask_user",
    "description": "Ask the user a question and wait for response.",
    "parameters": {...}
}
```

---

### Phase 2: Create Agent Type System

#### 2.1 New File: `lib/agent_types.py`

```python
from dataclasses import dataclass
from typing import Literal

AgentType = Literal["coding", "research", "web_dev", "general"]

@dataclass
class AgentTypeConfig:
    name: str
    system_prompt: str
    tools: list[str]  # Tool names to include
    description: str

AGENT_TYPES = {
    "coding": AgentTypeConfig(
        name="Coding Agent",
        system_prompt=SYSTEM_PROMPT_CODING,
        tools=["execute_code", "execute_bash", "read_file", "write_file",
               "replace_in_file", "list_directory", "search_file_content", "glob"],
        description="Software development and code modification"
    ),
    "research": AgentTypeConfig(
        name="Research Agent",
        system_prompt=SYSTEM_PROMPT_RESEARCH,
        tools=["web_search", "web_fetch", "read_file", "write_file",
               "list_directory", "glob"],
        description="Web research and information gathering"
    ),
    "web_dev": AgentTypeConfig(
        name="Web Dev Agent",
        system_prompt=SYSTEM_PROMPT_WEB_DEV,
        tools=["execute_code", "execute_bash", "read_file", "write_file",
               "replace_in_file", "list_directory", "search_file_content", "glob",
               "web_fetch"],
        description="Next.js/React web application development"
    ),
    "general": AgentTypeConfig(
        name="General Agent",
        system_prompt=SYSTEM_PROMPT_GENERAL,
        tools=["execute_code", "execute_bash", "read_file", "write_file",
               "replace_in_file", "list_directory", "search_file_content", "glob",
               "web_search", "web_fetch"],
        description="General-purpose assistant with all capabilities"
    ),
}
```

#### 2.2 New System Prompts

**SYSTEM_PROMPT_GENERAL:**
```
You are a general-purpose AI assistant. You can accomplish a wide variety of tasks using the available tools.

You MUST follow a strict 'Reason then Act' cycle:

1. **Reason:** Think step-by-step in a <scratchpad> block.
2. **Act:** Use one of your available tools to execute the next step.

Your capabilities include:
- **Research**: Search the web and fetch content from URLs
- **File Operations**: Read, write, search, and modify files
- **Code Execution**: Run Python code and bash commands
- **Analysis**: Process and analyze data, documents, and information

When researching:
- Use web_search to find relevant sources
- Use web_fetch to read full content from URLs
- Synthesize information from multiple sources
- Cite sources with URLs

When coding:
- Read existing code before modifying
- Test changes when possible
- Explain your approach

If you complete the task, provide a final answer without tool calls.
```

**SYSTEM_PROMPT_RESEARCH:**
```
You are a research assistant specialized in finding and synthesizing information.

You MUST follow a strict 'Reason then Act' cycle:

1. **Reason:** Think step-by-step in a <scratchpad> block.
2. **Act:** Use tools to gather information.

Your approach:
1. Break down complex questions into searchable queries
2. Search for multiple perspectives on a topic
3. Fetch and read full articles for depth
4. Synthesize findings into clear, well-organized responses
5. Always cite sources with URLs

Prioritize:
- Authoritative sources (official docs, academic, reputable news)
- Recent information when timeliness matters
- Multiple sources to verify facts
```

---

### Phase 3: Update Core Files

#### 3.1 `lib/tools.py` - Add new tool implementations

```python
def web_search(query: str, num_results: int = 5, **kwargs) -> tuple[dict, dict]:
    """Search the web using DuckDuckGo."""
    from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=num_results))

    return {
        "results": [
            {"title": r["title"], "url": r["href"], "snippet": r["body"]}
            for r in results
        ],
        "query": query,
        "count": len(results)
    }, {}


def web_fetch(url: str, extract_mode: str = "main_content", **kwargs) -> tuple[dict, dict]:
    """Fetch and extract content from a URL."""
    import requests
    from bs4 import BeautifulSoup
    import html2text

    response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script/style elements
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()

    if extract_mode == "main_content":
        # Try to find main content
        main = soup.find("main") or soup.find("article") or soup.find("body")
        text = main.get_text(separator="\n", strip=True) if main else ""
    else:
        text = soup.get_text(separator="\n", strip=True)

    # Convert to markdown
    h = html2text.HTML2Text()
    h.ignore_links = False
    markdown = h.handle(str(soup))

    return {
        "url": url,
        "title": soup.title.string if soup.title else "",
        "content": markdown[:10000],  # Limit content size
        "content_length": len(markdown)
    }, {}
```

#### 3.2 `lib/tools_schemas.py` - Add new schemas

Add `web_search_schema`, `web_fetch_schema` to the file.

Create tool registry:
```python
ALL_TOOLS_SCHEMAS = {
    "execute_code": execute_code_schema,
    "execute_bash": execute_bash_schema,
    "read_file": read_file_schema,
    # ... existing tools
    "web_search": web_search_schema,
    "web_fetch": web_fetch_schema,
}

def get_tools_for_agent_type(agent_type: str) -> list[dict]:
    """Get tool schemas for a specific agent type."""
    from .agent_types import AGENT_TYPES
    config = AGENT_TYPES[agent_type]
    return [ALL_TOOLS_SCHEMAS[name] for name in config.tools]
```

#### 3.3 `agent.py` - Add agent_type parameter

```python
class CodingAgent:  # Consider renaming to just "Agent"
    def __init__(
        self,
        agent_type: str = "coding",  # NEW
        sandbox_type: Literal["local", "docker"] = "local",
        working_dir: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        max_steps: int = 100,
        system_prompt: Optional[str] = None,  # Override agent type prompt
        docker_image: Optional[str] = None,
    ):
        from lib.agent_types import AGENT_TYPES, get_tools_for_agent_type

        self.agent_type = agent_type
        self.agent_config = AGENT_TYPES[agent_type]

        # Use agent type prompt unless overridden
        self.system_prompt = system_prompt or self.agent_config.system_prompt

        # Get tools for this agent type
        self.tools_schemas = get_tools_for_agent_type(agent_type)
        self.tools = get_tools_dict_for_agent_type(agent_type)
```

#### 3.4 `main.py` - Add CLI option

```python
parser.add_argument(
    "--agent-type",
    type=str,
    default="coding",
    choices=["coding", "research", "web_dev", "general"],
    help="Agent type: coding, research, web_dev, or general"
)
```

---

### Phase 4: Update Dependencies

**requirements.txt additions:**
```
duckduckgo-search>=6.0.0
beautifulsoup4>=4.12.0
html2text>=2024.0.0
requests>=2.31.0
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `lib/agent_types.py` | **CREATE** - Agent type configs and prompts |
| `lib/tools.py` | **MODIFY** - Add web_search, web_fetch |
| `lib/tools_schemas.py` | **MODIFY** - Add new schemas, tool registry |
| `lib/prompts.py` | **MODIFY** - Add SYSTEM_PROMPT_GENERAL, SYSTEM_PROMPT_RESEARCH |
| `agent.py` | **MODIFY** - Add agent_type parameter |
| `main.py` | **MODIFY** - Add --agent-type CLI option |
| `requirements.txt` | **MODIFY** - Add new dependencies |

---

## Usage After Implementation

```bash
# Coding tasks (default)
python main.py --cli

# Research tasks
python main.py --agent-type research --cli

# General purpose
python main.py --agent-type general --model claude-3-5-sonnet --cli

# Web development
python main.py --agent-type web_dev --cli
```

---

## Testing Plan

1. Test web_search tool with various queries
2. Test web_fetch tool with different websites
3. Test each agent type loads correct tools
4. Test research agent can search + synthesize
5. Test general agent can do both coding and research

---

## Future Enhancements (Out of Scope)

- Browser automation (Playwright/Selenium)
- Communication tools (Slack, email via APIs)
- Deployment tools (Docker, cloud providers)
- Multi-agent orchestration
