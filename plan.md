# Implementation Plan: Persistent Memory for Long-Running Agents

## Executive Summary

This plan builds a memory system inside `coding_agents/lib/memory/` to enable agents that can execute millions of steps with persistent memory and intelligent context management. The goal is to transform ephemeral, session-bound agents into persistent, resumable agents capable of long-running autonomous tasks.

**Key Architecture Decisions:**
- Memory system location: `coding_agents/lib/memory/` (integrated, not separate package)
- Storage backend: **Markdown files** (human-readable, agent-friendly, no database)
- Minimal dependencies: Standard library + PyYAML only

---

## Current State Analysis

### coding_agents Current Memory Model

| Aspect | Current State | Limitation |
|--------|---------------|------------|
| Message History | In-memory list | Lost on process termination |
| Context Management | Token-based compression at 42k | Lossy summarization, no semantic retrieval |
| Persistence | None | Cannot resume sessions |
| Long-term Knowledge | State snapshot only | No accumulated knowledge across sessions |
| Step Limit | 100 steps default | Not designed for millions of steps |

### Memory System Capabilities (To Be Built)

| Capability | Description | Implementation |
|------------|-------------|----------------|
| Short-Term Memory | FIFO buffer, 50-100 items | Markdown file (buffer.md) |
| Long-Term Memory | Keyword-based retrieval | Markdown files (knowledge.md, learnings.md) |
| Reflection | LLM summarization | Real LLM integration |
| Session Management | Checkpoints and resume | Markdown with YAML frontmatter |
| Context Retrieval | Combined STM + LTM | Integration with agent loop |

---

## Architecture Design

### Target Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CodingAgent                                  │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    coding_agent() Loop                        │   │
│  │                                                               │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │   │
│  │  │ LLM Client  │    │   Tools     │    │  Sandbox    │      │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘      │   │
│  │                                                               │   │
│  └─────────────────────────┬───────────────────────────────────┘   │
│                            │                                        │
│                            ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              MemoryManager (lib/memory/manager.py)            │   │
│  │                                                               │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │   │
│  │  │ ShortTerm    │  │ LongTerm     │  │ Session      │       │   │
│  │  │ Memory       │  │ Memory       │  │ Manager      │       │   │
│  │  │ (buffer.md)  │  │ (knowledge)  │  │ (sessions/)  │       │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │   │
│  │         │                 │                 │                │   │
│  │         └─────────────────┼─────────────────┘                │   │
│  │                           ▼                                   │   │
│  │              ┌─────────────────────────┐                     │   │
│  │              │   Markdown File Store   │                     │   │
│  │              │   (.agent_memory/)      │                     │   │
│  │              └─────────────────────────┘                     │   │
│  │                                                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Memory Storage Structure (Markdown Files)

```
.agent_memory/
├── index.md                        # Quick lookup index
├── sessions/
│   └── session_xxx.md              # Session checkpoints with YAML frontmatter
├── long_term/
│   ├── knowledge.md                # Facts, patterns, API info
│   └── learnings.md                # What agent learned from errors
└── short_term/
    └── buffer.md                   # Recent observations (FIFO)
```

### Memory Types for Long-Running Agents

| Memory Type | Purpose | Persistence | Capacity |
|-------------|---------|-------------|----------|
| **Short-Term** | Recent observations, current context | buffer.md | 50-100 items |
| **Long-Term** | Accumulated knowledge, patterns | knowledge.md, learnings.md | Unlimited |
| **Session** | Session checkpoints, resumable state | sessions/session_xxx.md | Per-session |

---

## Implementation Phases

### Phase 1: Markdown-Based Persistence Layer

**Objective**: Add disk persistence using markdown files for human-readable, agent-friendly storage.

#### 1.1 Markdown Store for Long-Term Memory

**File**: `coding_agents/lib/memory/persistence/markdown_store.py`

```python
class MarkdownMemoryStore:
    """Markdown-based storage for agent memories - human-readable and agent-friendly."""

    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self._init_structure()

    def _init_structure(self):
        """Create directory structure for memories."""
        (self.storage_path / "sessions").mkdir(parents=True, exist_ok=True)
        (self.storage_path / "long_term").mkdir(exist_ok=True)
        (self.storage_path / "short_term").mkdir(exist_ok=True)

    def add(self, content: str, memory_type: str, category: str = "general") -> str:
        """Add memory to appropriate markdown file."""
        pass

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """Retrieve by keyword search in markdown files."""
        pass

    def save_session(self, session_id: str, state: dict) -> str:
        """Save session checkpoint as markdown with YAML frontmatter."""
        pass

    def load_session(self, session_id: str) -> dict:
        """Load session from markdown checkpoint."""
        pass
```

#### 1.2 Session Checkpointing with Markdown

**File**: `coding_agents/lib/memory/session.py`

```python
class SessionManager:
    """Manages agent session lifecycle using markdown checkpoints."""

    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.current_session_id = None

    def start_session(self, resume_id: str = None) -> str:
        """Start new session or resume existing one."""
        pass

    def checkpoint(self, state: dict, step: int):
        """Save checkpoint to sessions/session_xxx.md."""
        pass

    def end_session(self, summary: str = None):
        """Finalize session with summary."""
        pass

    def list_sessions(self) -> list[dict]:
        """List all available sessions from sessions/ directory."""
        pass
```

---

### Phase 2: Integration with coding_agents

**Objective**: Connect memory module to the coding_agent loop.

#### 2.1 Add MemoryManager to CodingAgent

**File**: `coding_agents/agent.py`

Changes:
```python
from lib.memory import MemoryManager

class CodingAgent:
    def __init__(
        self,
        # existing params...
        memory_path: str = ".agent_memory",  # NEW
        enable_persistence: bool = True,      # NEW
        resume_session: str = None,           # NEW
    ):
        # Initialize memory system
        self.memory_manager = None
        if enable_persistence:
            self.memory_manager = MemoryManager(memory_path)
        if resume_session:
            self._restore_session(resume_session)
```

#### 2.2 Modify Agent Loop for Memory Integration

**File**: `coding_agents/lib/coding_agent.py`

Integration points:
1. **Before LLM call**: Retrieve relevant context from long-term memory
2. **After tool execution**: Store observations in short-term memory
3. **Periodic reflection**: Consolidate short-term to long-term
4. **Checkpointing**: Save state every N steps

```python
from lib.memory.integration import inject_memories

def coding_agent(
    client,
    sbx,
    query,
    tools,
    tools_schemas,
    max_steps=5,
    system=None,
    messages=None,
    model="gpt-4.1-mini",
    memory_manager=None,      # NEW
    checkpoint_interval=100,   # NEW
):
    step = 0
    while step < max_steps:
        # NEW: Inject relevant memories into context
        if memory_manager:
            relevant_memories = memory_manager.retrieve_context(query, top_k=10)
            messages = inject_memories(messages, relevant_memories)

        # Existing: maybe_compress_messages
        messages = maybe_compress_messages(client, messages, usage, model)

        # Existing: LLM call and tool execution
        response = client.responses.create(...)

        for part in response.output:
            if part.type == "function_call":
                result = execute_tool(...)

                # NEW: Store observation in memory
                if memory_manager:
                    memory_manager.add_memory(
                        f"Tool: {part.name}, Result: {result[:500]}"
                    )

            yield (part_dict, messages, usage)

        # NEW: Periodic checkpointing
        if memory_manager and step % checkpoint_interval == 0:
            memory_manager.checkpoint(step, messages)

        step += 1
```

---

### Phase 3: Scaling for Millions of Steps

**Objective**: Optimize for extreme scale while maintaining coherence.

#### 3.1 Hierarchical Memory Consolidation

```
Step 1-100:      Raw observations → Short-term buffer
Step 100:        Reflection → "Completed setup phase, installed deps"
Step 100-1000:   More observations → Short-term buffer
Step 1000:       Meta-reflection → "Project structure established"
Step 1000-10000: ...
Step 10000:      High-level summary → "Phase 1 complete: core implemented"
```

**Implementation**:
```python
class HierarchicalMemory:
    """Multi-level memory consolidation for extreme scale."""

    def __init__(self):
        self.levels = {
            "observations": [],      # Raw, step-level
            "summaries": [],         # Every 100 steps
            "meta_summaries": [],    # Every 1000 steps
            "milestones": [],        # Every 10000 steps
        }

    def consolidate(self, step: int, llm_client):
        """Consolidate memories based on current step."""
        if step % 100 == 0:
            self._create_summary(llm_client)
        if step % 1000 == 0:
            self._create_meta_summary(llm_client)
        if step % 10000 == 0:
            self._create_milestone(llm_client)
```

#### 3.2 Checkpoint and Resume System

```python
class CheckpointManager:
    """Manages checkpoints for long-running tasks."""

    def __init__(self, storage_path: str):
        self.storage_path = storage_path

    def save_checkpoint(
        self,
        session_id: str,
        step: int,
        messages: list[dict],
        memory_state: dict,
        sandbox_state: dict,
    ):
        """Save complete agent state for later resumption."""
        checkpoint = {
            "session_id": session_id,
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "messages_compressed": self._compress_messages(messages),
            "memory_state": memory_state,
            "sandbox_hash": sandbox_state.get("hash"),
            "working_dir_manifest": sandbox_state.get("manifest"),
        }
        # Save to SQLite + file system
        pass

    def restore_checkpoint(self, session_id: str, step: int = None) -> dict:
        """Restore agent state from checkpoint."""
        # If step not specified, use latest
        pass
```

#### 3.3 Adaptive Reflection Triggers

Don't reflect on fixed intervals; reflect when meaningful:

```python
class AdaptiveReflectionTrigger:
    """Triggers reflection based on content, not just count."""

    def should_reflect(self, memory_buffer: list[str]) -> bool:
        # Trigger on:
        # - Error recovery (learned something)
        # - Task completion (milestone)
        # - Context switch (new subtask)
        # - Novel discovery (new file/pattern)
        # - Confusion detected (repeated failures)
        pass
```

---

### Phase 4: Additional Features

#### 4.1 LLM Integration for Summarization

**File**: `coding_agents/lib/memory/manager.py`

```python
def _summarize_with_llm(self, memories: list[str]) -> str:
    """Use LLM to synthesize memories into insight."""
    if not self.llm_client:
        # Fallback: simple concatenation with truncation
        return "Summary: " + " | ".join(memories)[:500]

    prompt = f"""Synthesize these observations into a concise insight:

{chr(10).join(f'- {m}' for m in memories)}

Provide a 1-2 sentence summary capturing the key information."""

    response = self.llm_client.chat.completions.create(
        model=self.summary_model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

#### 4.2 Semantic Retrieval (Optional)

Add embedding support for better retrieval:

**File**: `coding_agents/lib/memory/embeddings.py`

```python
def retrieve(self, query: str, top_k: int = 5, use_embeddings: bool = True) -> list[str]:
    if use_embeddings and self.embedding_provider:
        return self._similarity_search(query, top_k)
    else:
        return self._keyword_search(query, top_k)
```

#### 4.3 Thread Safety

Add locking for concurrent access:

**File**: `coding_agents/lib/memory/manager.py`

```python
import threading

class MemoryManager:
    def __init__(self, ...):
        self._lock = threading.RLock()

    def add_memory(self, content: str):
        with self._lock:
            # ... existing logic
```

---

## File Changes Summary

### New Files - Inside coding_agents/lib/memory/

```
coding_agents/lib/memory/
├── __init__.py                     # Package exports
├── manager.py                      # MemoryManager - central orchestrator
├── session.py                      # SessionManager - lifecycle & checkpoints
├── embeddings.py                   # EmbeddingProvider - semantic retrieval (optional)
├── consolidation.py                # HierarchicalMemory - scale optimization
├── types/
│   ├── __init__.py
│   ├── base.py                     # BaseMemory abstract class
│   ├── short_term.py               # ShortTermMemory - FIFO buffer
│   └── long_term.py                # LongTermMemory - markdown-based knowledge
├── persistence/
│   ├── __init__.py
│   ├── markdown_store.py           # MarkdownMemoryStore - markdown file backend
│   └── checkpoint.py               # CheckpointManager - session snapshots
└── integration.py                  # Memory injection and context management
```

### Modified Files

| File | Changes |
|------|---------|
| `coding_agents/agent.py` | Add memory initialization, resume support |
| `coding_agents/lib/coding_agent.py` | Integrate memory into loop |
| `coding_agents/main.py` | Add CLI flags for memory/resume |

---

## CLI Interface Changes

```bash
# New flags for persistence
python main.py --cli \
    --memory-path .agent_memory \     # Where to store memories
    --enable-persistence \             # Enable persistent memory
    --resume <session-id> \            # Resume previous session
    --checkpoint-interval 100 \        # Steps between checkpoints
    --max-steps 1000000                # Support millions of steps

# Session management
python main.py sessions list          # List all sessions
python main.py sessions show <id>     # Show session details
python main.py sessions resume <id>   # Resume session
python main.py sessions delete <id>   # Delete session
```

---

## Success Criteria

### Functional Requirements

- [ ] Agent can run for 1,000,000+ steps without context degradation
- [ ] Sessions can be paused and resumed across process restarts
- [ ] Relevant past experiences are retrieved for current tasks
- [ ] Memory consolidation happens automatically at scale
- [ ] Checkpoints are created at configurable intervals

### Performance Requirements

- [ ] Memory operations add < 100ms latency per step
- [ ] Checkpoint save < 1 second
- [ ] Checkpoint restore < 5 seconds
- [ ] SQLite database handles 10M+ memories efficiently

### Quality Requirements

- [ ] Backward compatible: existing usage without memory still works
- [ ] Graceful degradation: if persistence fails, continue in-memory
- [ ] Observable: logging for memory operations and consolidation

---

## Implementation Order

1. **Week 1**: Phase 1 - Persistence layer in memory_system
2. **Week 2**: Phase 2 - Integration with coding_agents
3. **Week 3**: Phase 3 - Scaling optimizations
4. **Week 4**: Phase 4 - Memory system fixes + testing

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM costs for reflection | High at scale | Batch reflections, use cheaper models |
| SQLite performance at scale | Medium | Add indexes, consider PostgreSQL for enterprise |
| Embedding model latency | Medium | Local models (sentence-transformers), caching |
| Checkpoint size growth | Medium | Incremental checkpoints, compression |
| Context coherence loss | High | Hierarchical summarization, key fact extraction |

---

## Dependencies to Add

### coding_agents (add to requirements.txt)

```
pyyaml>=6.0                        # YAML frontmatter parsing for session files
# Optional: sentence-transformers>=2.2.0  # Local embeddings (if semantic search needed)
```

**Note:** Memory system is integrated into `coding_agents/lib/memory/` - no external package needed. Markdown-based storage requires minimal dependencies (standard library + PyYAML).

---

## Appendix: Memory Retrieval Strategies

### For Different Query Types

| Query Type | Retrieval Strategy |
|------------|-------------------|
| "What did I learn about X?" | Semantic search in summaries |
| "Show recent tool outputs" | Recency-based from short-term |
| "What errors have I seen?" | Filtered search by memory type |
| "Continue from where I left off" | Checkpoint + episodic memory |
| "What patterns have I noticed?" | Meta-summary retrieval |

### Memory Injection Format

```
<assistant_notes>
## Relevant Past Experience
- Previously encountered similar error in file X, fixed by Y
- User prefers approach Z for this type of task

## Current Task Progress
- Completed: Steps A, B, C
- In Progress: Step D
- Remaining: Steps E, F

## Key Learnings This Session
- API endpoint moved to /v2/...
- Config requires ENABLE_FEATURE=true
</assistant_notes>
```

---

## Conclusion

This implementation plan enables coding_agents to:

1. **Persist memory** across sessions using SQLite
2. **Scale to millions of steps** through hierarchical consolidation
3. **Resume interrupted tasks** via checkpointing
4. **Retrieve relevant context** using semantic search
5. **Maintain coherence** through intelligent summarization

The integration preserves backward compatibility while adding powerful new capabilities for long-running autonomous agents.
