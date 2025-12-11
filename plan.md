# Implementation Plan: Persistent Memory for Long-Running Agents

## Executive Summary

This plan integrates the `memory_system` framework into `coding_agents` to enable agents that can execute millions of steps with persistent memory and intelligent context management. The goal is to transform ephemeral, session-bound agents into persistent, resumable agents capable of long-running autonomous tasks.

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

### memory_system Capabilities

| Capability | Description | Gap for Integration |
|------------|-------------|---------------------|
| Short-Term Memory | FIFO buffer, 10 items | In-memory only, needs persistence |
| Long-Term Memory | Keyword-based retrieval | In-memory dict, needs database backend |
| Reflection | LLM summarization | Placeholder implementation |
| Todo Management | File-based | Already persistent |
| Context Retrieval | Combined STM + LTM | Needs integration with agent loop |

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
│  │                  MemoryManager (NEW)                          │   │
│  │                                                               │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │   │
│  │  │ ShortTerm    │  │ LongTerm     │  │ Episode      │       │   │
│  │  │ Memory       │  │ Memory       │  │ Memory (NEW) │       │   │
│  │  │ (Working)    │  │ (Knowledge)  │  │ (Sessions)   │       │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │   │
│  │         │                 │                 │                │   │
│  │         └─────────────────┼─────────────────┘                │   │
│  │                           ▼                                   │   │
│  │              ┌─────────────────────────┐                     │   │
│  │              │   Persistence Layer     │                     │   │
│  │              │   (SQLite / Arrow)      │                     │   │
│  │              └─────────────────────────┘                     │   │
│  │                                                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Memory Types for Long-Running Agents

| Memory Type | Purpose | Persistence | Capacity |
|-------------|---------|-------------|----------|
| **Short-Term** | Recent observations, current context | Session file | 50-100 items |
| **Long-Term** | Accumulated knowledge, patterns | SQLite + embeddings | Unlimited |
| **Episodic** | Session checkpoints, resumable state | SQLite | Per-session |
| **Procedural** | Learned tool sequences, workflows | JSON files | Per-task type |

---

## Implementation Phases

### Phase 1: Persistence Layer for memory_system

**Objective**: Add disk persistence to memory_system without breaking existing API.

#### 1.1 SQLite Backend for Long-Term Memory

**File**: `memory_system/persistence/sqlite_store.py`

```python
class SQLiteMemoryStore:
    """Persistent storage backend for long-term memories."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self):
        """Create tables for memories, embeddings, and metadata."""
        # memories: id, content, timestamp, memory_type, embedding_id
        # embeddings: id, vector (blob), model
        # sessions: id, start_time, end_time, summary, checkpoint_path
        pass

    def add(self, content: str, memory_type: str, embedding: list[float] = None) -> str:
        """Add memory with optional embedding."""
        pass

    def retrieve(self, query: str, top_k: int = 10, embedding: list[float] = None) -> list[dict]:
        """Retrieve by keyword or embedding similarity."""
        pass

    def checkpoint(self, session_id: str, state: dict) -> str:
        """Save session checkpoint for resumption."""
        pass

    def restore(self, session_id: str) -> dict:
        """Restore session from checkpoint."""
        pass
```

#### 1.2 Update LongTermMemory to Use Persistence

**File**: `memory_system/memory_types/long_term.py`

Changes:
- Add `storage_backend` parameter (default: in-memory for backward compatibility)
- Implement `save()` and `load()` methods
- Add embedding support for semantic retrieval

#### 1.3 Session Checkpointing

**File**: `memory_system/session.py`

```python
class SessionManager:
    """Manages agent session lifecycle and checkpoints."""

    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.current_session_id = None

    def start_session(self, resume_id: str = None) -> str:
        """Start new session or resume existing one."""
        pass

    def checkpoint(self, state: dict, step: int):
        """Save checkpoint at current step."""
        pass

    def end_session(self, summary: str = None):
        """Finalize and close session."""
        pass

    def list_sessions(self) -> list[dict]:
        """List all available sessions."""
        pass
```

---

### Phase 2: Integration with coding_agents

**Objective**: Connect memory_system to the coding_agent loop.

#### 2.1 Add MemoryManager to CodingAgent

**File**: `coding_agents/agent.py`

Changes:
```python
class CodingAgent:
    def __init__(
        self,
        # existing params...
        memory_path: str = ".agent_memory",  # NEW
        enable_persistence: bool = True,      # NEW
        resume_session: str = None,           # NEW
    ):
        # Initialize memory system
        self.memory_manager = self._init_memory(memory_path, enable_persistence)
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

#### 2.3 Replace Compression with Intelligent Context Management

Current compression is lossy. Replace with:

1. **Semantic Retrieval**: Query long-term memory for relevant past experiences
2. **Hierarchical Summarization**: Multiple levels of abstraction
3. **Selective Retention**: Keep tool results and key decisions in detail

```python
def manage_context(
    messages: list[dict],
    memory_manager: MemoryManager,
    token_limit: int = 60000,
    preserve_recent: int = 10,
) -> list[dict]:
    """
    Intelligent context management that preserves important information.

    Strategy:
    1. Always keep: system prompt, recent N messages, current tool calls
    2. Summarize: older conversation turns
    3. Retrieve: relevant long-term memories for current task
    4. Inject: retrieved memories as "assistant notes"
    """
    pass
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

### Phase 4: Memory System Fixes Required

Based on analysis, the following fixes are needed in `memory_system`:

#### 4.1 LLM Integration (Critical)

The `_summarize_with_llm()` is currently a placeholder. Need real implementation:

**File**: `memory_system/manager.py`

```python
def _summarize_with_llm(self, memories: list[str]) -> str:
    """Use LLM to synthesize memories into insight."""
    if not self.llm_client:
        # Fallback: simple concatenation with truncation
        return "Summary: " + " | ".join(memories)[:500]

    prompt = f"""Synthesize these observations into a concise insight:

{chr(10).join(f'- {m}' for m in memories)}

Provide a 1-2 sentence summary capturing the key information."""

    response = self.llm_client.generate(prompt)
    return response.text
```

#### 4.2 Persistence Backend (Critical)

Current LongTermMemory uses in-memory dict. Add SQLite:

**File**: `memory_system/memory_types/long_term.py`

```python
class LongTermMemory(BaseMemory):
    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path
        if storage_path:
            self._init_sqlite()
        else:
            self.memories = {}  # Backward compatible
```

#### 4.3 Semantic Retrieval (Important)

Keyword matching is weak. Add embedding support:

```python
def retrieve(self, query: str, top_k: int = 5, use_embeddings: bool = True) -> list[str]:
    if use_embeddings and self.embedding_model:
        query_embedding = self.embedding_model.encode(query)
        return self._similarity_search(query_embedding, top_k)
    else:
        return self._keyword_search(query, top_k)
```

#### 4.4 Thread Safety (Important for Scale)

Add locking for concurrent access:

```python
import threading

class ThreadSafeMemoryManager:
    def __init__(self, ...):
        self._lock = threading.RLock()

    def add_memory(self, content: str):
        with self._lock:
            # ... existing logic
```

---

## File Changes Summary

### New Files

| File | Purpose |
|------|---------|
| `memory_system/persistence/__init__.py` | Persistence module |
| `memory_system/persistence/sqlite_store.py` | SQLite backend |
| `memory_system/persistence/checkpoint.py` | Checkpoint management |
| `memory_system/session.py` | Session lifecycle management |
| `memory_system/embeddings.py` | Embedding model integration |
| `coding_agents/lib/memory_integration.py` | Integration layer |

### Modified Files

| File | Changes |
|------|---------|
| `memory_system/memory_types/long_term.py` | Add persistence, embeddings |
| `memory_system/manager.py` | Real LLM integration, session support |
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

### memory_system

```toml
[project.dependencies]
# Existing
pyarrow = ">=22.0.0"

# New
sentence-transformers = ">=2.2.0"  # Local embeddings
sqlalchemy = ">=2.0"               # Database abstraction
```

### coding_agents

```toml
[project.dependencies]
# Add reference to memory_system
memory-system = { path = "../memory_system" }
# Or if published: memory-system = ">=1.0.0"
```

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
