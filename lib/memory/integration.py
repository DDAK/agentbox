"""
Integration layer for connecting memory system to coding_agents.

Provides functions for:
- Injecting memories into conversation context
- Extracting observations from tool executions
- Managing context intelligently
"""

from typing import Optional, List


def inject_memories(messages: List[dict], memories: List[str]) -> List[dict]:
    """
    Inject retrieved memories into the conversation context.

    Memories are added as assistant notes before the conversation
    to provide relevant context without disrupting the flow.

    Args:
        messages: Current conversation messages
        memories: List of relevant memory strings to inject

    Returns:
        Modified messages list with memories injected
    """
    if not memories:
        return messages

    # Format memories as assistant notes
    memory_lines = "\n".join(f"- {m}" for m in memories)
    memory_content = f"""<assistant_notes>
## Relevant Past Experience

{memory_lines}

</assistant_notes>"""

    # Create memory message
    memory_message = {
        "role": "user",
        "content": memory_content,
        "_internal": True,  # Mark as internal (will be stripped before LLM call)
    }

    # Insert after system message (if present) or at the beginning
    if messages and messages[0].get("role") == "system":
        return [messages[0], memory_message] + messages[1:]
    else:
        return [memory_message] + messages


def extract_observation(
    tool_name: str,
    result: str,
    arguments: Optional[str] = None,
    max_length: int = 500,
) -> str:
    """
    Extract an observation from a tool execution result.

    Args:
        tool_name: Name of the tool that was executed
        result: The result from the tool
        arguments: Optional arguments that were passed
        max_length: Maximum length of the observation

    Returns:
        Formatted observation string
    """
    # Truncate long results
    if len(result) > max_length:
        result = result[:max_length] + "..."

    # Format based on tool type
    if tool_name in ("read_file", "list_directory", "glob"):
        # File operations - extract key info
        return f"Read/listed: {result[:200]}"

    elif tool_name in ("write_file", "replace_in_file"):
        # Write operations - note what was written
        if arguments:
            return f"Wrote to file: {arguments[:100]}"
        return f"File write: {result[:100]}"

    elif tool_name in ("execute_code", "execute_bash"):
        # Code execution - capture output
        return f"Executed code, output: {result[:300]}"

    elif tool_name in ("web_search", "web_fetch"):
        # Web operations - note what was found
        return f"Web: {result[:300]}"

    else:
        # Generic observation
        return f"Tool {tool_name}: {result[:200]}"


def format_context_summary(
    task: str,
    progress: List[str],
    recent_actions: List[str],
    key_learnings: List[str],
) -> str:
    """
    Format a context summary for checkpoint or injection.

    Args:
        task: Current task description
        progress: List of completed items
        recent_actions: Recent actions taken
        key_learnings: Key learnings discovered

    Returns:
        Formatted context summary string
    """
    progress_str = "\n".join(f"- [x] {p}" for p in progress) if progress else "- No progress yet"
    actions_str = "\n".join(f"- {a}" for a in recent_actions[-5:]) if recent_actions else "- No recent actions"
    learnings_str = "\n".join(f"- {l}" for l in key_learnings) if key_learnings else "- No learnings yet"

    return f"""## Current Task
{task}

## Progress
{progress_str}

## Recent Actions
{actions_str}

## Key Learnings
{learnings_str}
"""


def should_checkpoint(
    step: int,
    interval: int = 100,
    force_on_error: bool = True,
    had_error: bool = False,
) -> bool:
    """
    Determine if a checkpoint should be saved.

    Args:
        step: Current step number
        interval: Checkpoint interval
        force_on_error: Whether to checkpoint after errors
        had_error: Whether an error occurred

    Returns:
        True if checkpoint should be saved
    """
    if force_on_error and had_error:
        return True

    return step > 0 and step % interval == 0


def extract_task_from_query(query: str) -> str:
    """
    Extract a task description from a user query.

    Args:
        query: The user's query

    Returns:
        Extracted task description
    """
    # Simple extraction - take first sentence or first 200 chars
    if "." in query:
        first_sentence = query.split(".")[0]
        if len(first_sentence) < 200:
            return first_sentence.strip()

    return query[:200].strip()


def merge_messages_with_context(
    messages: List[dict],
    context: List[str],
    recent_limit: int = 10,
) -> List[dict]:
    """
    Merge messages with retrieved context intelligently.

    Preserves:
    - System prompt
    - Recent messages (last N)
    - Current tool calls

    Injects:
    - Retrieved context as assistant notes

    Args:
        messages: Current conversation messages
        context: Retrieved context strings
        recent_limit: Number of recent messages to preserve

    Returns:
        Merged messages list
    """
    if not messages:
        return []

    result = []

    # Keep system message if present
    if messages[0].get("role") == "system":
        result.append(messages[0])
        messages = messages[1:]

    # Inject context
    if context:
        result = inject_memories(result, context)

    # Keep recent messages
    recent = messages[-recent_limit:] if len(messages) > recent_limit else messages
    result.extend(recent)

    return result


def clean_messages_for_storage(messages: List[dict]) -> List[dict]:
    """
    Clean messages for storage by removing internal markers.

    Args:
        messages: Messages to clean

    Returns:
        Cleaned messages list
    """
    cleaned = []
    for msg in messages:
        if msg.get("_internal"):
            continue

        clean_msg = {k: v for k, v in msg.items() if not k.startswith("_")}
        cleaned.append(clean_msg)

    return cleaned
