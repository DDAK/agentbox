"""
Markdown-based storage for agent memories.

Provides human-readable, agent-friendly persistent storage using markdown files.
"""

import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class MarkdownMemoryStore:
    """Markdown-based storage for agent memories - human-readable and agent-friendly."""

    def __init__(self, storage_path: str):
        """
        Initialize the markdown memory store.

        Args:
            storage_path: Root directory for memory storage (e.g., ".agent_memory")
        """
        self.storage_path = Path(storage_path)
        self._init_structure()

    def _init_structure(self):
        """Create directory structure for memories."""
        # Create directories
        (self.storage_path / "sessions").mkdir(parents=True, exist_ok=True)
        (self.storage_path / "long_term").mkdir(exist_ok=True)
        (self.storage_path / "short_term").mkdir(exist_ok=True)

        # Initialize index.md if it doesn't exist
        index_path = self.storage_path / "index.md"
        if not index_path.exists():
            self._create_index()

        # Initialize knowledge.md if it doesn't exist
        knowledge_path = self.storage_path / "long_term" / "knowledge.md"
        if not knowledge_path.exists():
            self._create_knowledge_file()

        # Initialize learnings.md if it doesn't exist
        learnings_path = self.storage_path / "long_term" / "learnings.md"
        if not learnings_path.exists():
            self._create_learnings_file()

        # Initialize buffer.md if it doesn't exist
        buffer_path = self.storage_path / "short_term" / "buffer.md"
        if not buffer_path.exists():
            self._create_buffer_file()

    def _create_index(self):
        """Create the initial index.md file."""
        content = """# Memory Index

## Sessions
<!-- Sessions will be listed here -->

## Knowledge Topics
<!-- Knowledge topics will be linked here -->

## Recent Observations
<!-- Recent observations will be listed here -->

---
*Last updated: {timestamp}*
""".format(timestamp=datetime.now().isoformat())

        (self.storage_path / "index.md").write_text(content)

    def _create_knowledge_file(self):
        """Create the initial knowledge.md file."""
        content = """# Long-Term Knowledge

## Facts
<!-- Project facts and configuration will be stored here -->

## Patterns
<!-- Discovered patterns and best practices -->

## API Information
<!-- API endpoints, configurations, etc. -->

---
*Last updated: {timestamp}*
""".format(timestamp=datetime.now().isoformat())

        (self.storage_path / "long_term" / "knowledge.md").write_text(content)

    def _create_learnings_file(self):
        """Create the initial learnings.md file."""
        content = """# Agent Learnings

## Error Resolutions
<!-- How errors were resolved -->

## Successful Approaches
<!-- What worked well -->

## Things to Avoid
<!-- What didn't work -->

---
*Last updated: {timestamp}*
""".format(timestamp=datetime.now().isoformat())

        (self.storage_path / "long_term" / "learnings.md").write_text(content)

    def _create_buffer_file(self):
        """Create the initial buffer.md file."""
        content = """# Short-Term Memory Buffer

<!-- Recent observations (FIFO, most recent first) -->

---
*Last updated: {timestamp}*
""".format(timestamp=datetime.now().isoformat())

        (self.storage_path / "short_term" / "buffer.md").write_text(content)

    def add_observation(self, content: str, tool_name: Optional[str] = None) -> str:
        """
        Add an observation to short-term memory buffer.

        Args:
            content: The observation content
            tool_name: Optional tool that generated this observation

        Returns:
            The observation ID
        """
        observation_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()

        # Read existing buffer
        buffer_path = self.storage_path / "short_term" / "buffer.md"
        existing = buffer_path.read_text()

        # Parse existing observations
        lines = existing.split("\n")
        header_end = 0
        for i, line in enumerate(lines):
            if line.startswith("<!-- Recent observations"):
                header_end = i + 1
                break

        # Create new observation entry
        tool_info = f" [{tool_name}]" if tool_name else ""
        new_entry = f"- **{timestamp}**{tool_info}: {content[:200]}{'...' if len(content) > 200 else ''}"

        # Insert new observation after header
        lines.insert(header_end + 1, new_entry)

        # Keep only last 100 observations (FIFO)
        observation_lines = [l for l in lines if l.startswith("- **")]
        if len(observation_lines) > 100:
            # Remove oldest observations
            for old_line in observation_lines[100:]:
                if old_line in lines:
                    lines.remove(old_line)

        # Update timestamp
        new_content = "\n".join(lines)
        new_content = re.sub(
            r"\*Last updated: .*\*",
            f"*Last updated: {timestamp}*",
            new_content
        )

        buffer_path.write_text(new_content)
        return observation_id

    def add_knowledge(self, content: str, category: str = "Facts") -> str:
        """
        Add knowledge to long-term memory.

        Args:
            content: The knowledge content
            category: Category (Facts, Patterns, API Information)

        Returns:
            The knowledge ID
        """
        knowledge_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()

        knowledge_path = self.storage_path / "long_term" / "knowledge.md"
        existing = knowledge_path.read_text()

        # Find the appropriate section
        section_header = f"## {category}"
        if section_header not in existing:
            # Add new section before the footer
            footer_match = re.search(r"\n---\n\*Last updated:", existing)
            if footer_match:
                insert_pos = footer_match.start()
                new_section = f"\n{section_header}\n"
                existing = existing[:insert_pos] + new_section + existing[insert_pos:]

        # Add entry under the section
        lines = existing.split("\n")
        for i, line in enumerate(lines):
            if line == section_header:
                # Insert after section header and any comment
                insert_idx = i + 1
                while insert_idx < len(lines) and lines[insert_idx].startswith("<!--"):
                    insert_idx += 1
                lines.insert(insert_idx, f"- {content}")
                break

        # Update timestamp
        new_content = "\n".join(lines)
        new_content = re.sub(
            r"\*Last updated: .*\*",
            f"*Last updated: {timestamp}*",
            new_content
        )

        knowledge_path.write_text(new_content)
        return knowledge_id

    def add_learning(self, content: str, category: str = "Successful Approaches") -> str:
        """
        Add a learning to long-term memory.

        Args:
            content: The learning content
            category: Category (Error Resolutions, Successful Approaches, Things to Avoid)

        Returns:
            The learning ID
        """
        learning_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()

        learnings_path = self.storage_path / "long_term" / "learnings.md"
        existing = learnings_path.read_text()

        # Find the appropriate section
        section_header = f"## {category}"
        if section_header in existing:
            lines = existing.split("\n")
            for i, line in enumerate(lines):
                if line == section_header:
                    insert_idx = i + 1
                    while insert_idx < len(lines) and lines[insert_idx].startswith("<!--"):
                        insert_idx += 1
                    lines.insert(insert_idx, f"- [{timestamp[:10]}] {content}")
                    break
            existing = "\n".join(lines)

        # Update timestamp
        new_content = re.sub(
            r"\*Last updated: .*\*",
            f"*Last updated: {timestamp}*",
            existing
        )

        learnings_path.write_text(new_content)
        return learning_id

    def get_recent_observations(self, limit: int = 10) -> List[str]:
        """
        Get recent observations from short-term memory.

        Args:
            limit: Maximum number of observations to return

        Returns:
            List of observation strings
        """
        buffer_path = self.storage_path / "short_term" / "buffer.md"
        content = buffer_path.read_text()

        observations = []
        for line in content.split("\n"):
            if line.startswith("- **"):
                # Extract content after timestamp
                match = re.match(r"- \*\*[^*]+\*\*(?:\s*\[[^\]]+\])?: (.+)", line)
                if match:
                    observations.append(match.group(1))
                if len(observations) >= limit:
                    break

        return observations

    def get_knowledge(self, category: Optional[str] = None) -> List[str]:
        """
        Get knowledge from long-term memory.

        Args:
            category: Optional category filter

        Returns:
            List of knowledge strings
        """
        knowledge_path = self.storage_path / "long_term" / "knowledge.md"
        content = knowledge_path.read_text()

        knowledge = []
        current_section = None

        for line in content.split("\n"):
            if line.startswith("## "):
                current_section = line[3:]
            elif line.startswith("- ") and not line.startswith("<!--"):
                if category is None or current_section == category:
                    knowledge.append(line[2:])

        return knowledge

    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """
        Retrieve relevant memories by keyword search.

        Args:
            query: Search query
            top_k: Maximum number of results

        Returns:
            List of relevant memory strings
        """
        query_words = set(query.lower().split())
        results = []

        # Search in observations
        for obs in self.get_recent_observations(50):
            obs_words = set(obs.lower().split())
            score = len(query_words & obs_words)
            if score > 0:
                results.append((score, f"[Recent] {obs}"))

        # Search in knowledge
        for knowledge in self.get_knowledge():
            knowledge_words = set(knowledge.lower().split())
            score = len(query_words & knowledge_words)
            if score > 0:
                results.append((score, f"[Knowledge] {knowledge}"))

        # Sort by relevance and return top_k
        results.sort(reverse=True)
        return [content for _, content in results[:top_k]]

    def save_session(self, session_id: str, state: dict) -> str:
        """
        Save a session checkpoint as markdown with YAML frontmatter.

        Args:
            session_id: Unique session identifier
            state: Session state dictionary

        Returns:
            Path to the session file
        """
        timestamp = datetime.now().isoformat()
        session_path = self.storage_path / "sessions" / f"{session_id}.md"

        # Build YAML frontmatter
        frontmatter = {
            "id": session_id,
            "started": state.get("started", timestamp),
            "updated": timestamp,
            "step": state.get("step", 0),
            "status": state.get("status", "active"),
        }

        # Build markdown content
        if HAS_YAML:
            frontmatter_str = yaml.dump(frontmatter, default_flow_style=False)
        else:
            frontmatter_str = "\n".join(f"{k}: {v}" for k, v in frontmatter.items())

        content = f"""---
{frontmatter_str}---

# Session: {session_id}

## Current Task
{state.get('task', 'No task specified')}

## Progress
{state.get('progress', '- No progress recorded')}

## Context Summary
{state.get('summary', 'No summary available')}

## Recent Messages (compressed)
```
{state.get('messages_compressed', 'No messages')}
```

---
*Last checkpoint: {timestamp}*
"""
        session_path.write_text(content)

        # Update index
        self._update_session_index(session_id, frontmatter)

        return str(session_path)

    def load_session(self, session_id: str) -> Optional[dict]:
        """
        Load a session from markdown checkpoint.

        Args:
            session_id: Session identifier

        Returns:
            Session state dictionary or None if not found
        """
        session_path = self.storage_path / "sessions" / f"{session_id}.md"
        if not session_path.exists():
            return None

        content = session_path.read_text()

        # Parse YAML frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter_str = parts[1].strip()
                if HAS_YAML:
                    state = yaml.safe_load(frontmatter_str)
                else:
                    state = {}
                    for line in frontmatter_str.split("\n"):
                        if ": " in line:
                            key, value = line.split(": ", 1)
                            state[key.strip()] = value.strip()

                # Extract sections from markdown body
                body = parts[2]

                # Extract task
                task_match = re.search(r"## Current Task\n(.+?)(?=\n##|\n---|\Z)", body, re.DOTALL)
                if task_match:
                    state["task"] = task_match.group(1).strip()

                # Extract progress
                progress_match = re.search(r"## Progress\n(.+?)(?=\n##|\n---|\Z)", body, re.DOTALL)
                if progress_match:
                    state["progress"] = progress_match.group(1).strip()

                # Extract summary
                summary_match = re.search(r"## Context Summary\n(.+?)(?=\n##|\n---|\Z)", body, re.DOTALL)
                if summary_match:
                    state["summary"] = summary_match.group(1).strip()

                # Extract compressed messages
                messages_match = re.search(r"## Recent Messages \(compressed\)\n```\n(.+?)\n```", body, re.DOTALL)
                if messages_match:
                    state["messages_compressed"] = messages_match.group(1).strip()

                return state

        return None

    def list_sessions(self) -> List[dict]:
        """
        List all available sessions.

        Returns:
            List of session metadata dictionaries
        """
        sessions = []
        sessions_dir = self.storage_path / "sessions"

        for session_file in sessions_dir.glob("*.md"):
            session_id = session_file.stem
            state = self.load_session(session_id)
            if state:
                sessions.append({
                    "id": session_id,
                    "started": state.get("started"),
                    "updated": state.get("updated"),
                    "step": state.get("step", 0),
                    "status": state.get("status", "unknown"),
                })

        # Sort by updated time (most recent first)
        sessions.sort(key=lambda x: x.get("updated", ""), reverse=True)
        return sessions

    def _update_session_index(self, session_id: str, metadata: dict):
        """Update the index.md with session information."""
        index_path = self.storage_path / "index.md"
        content = index_path.read_text()

        # Find sessions section
        sessions_section = "## Sessions"
        if sessions_section in content:
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if line == sessions_section:
                    # Check if session already listed
                    session_entry = f"- [{session_id}]"
                    found = False
                    for j in range(i + 1, len(lines)):
                        if lines[j].startswith(session_entry):
                            # Update existing entry
                            lines[j] = f"- [{session_id}](sessions/{session_id}.md) - Step {metadata.get('step', 0)}, {metadata.get('status', 'active')}"
                            found = True
                            break
                        if lines[j].startswith("##"):
                            break

                    if not found:
                        # Add new entry
                        insert_idx = i + 1
                        while insert_idx < len(lines) and lines[insert_idx].startswith("<!--"):
                            insert_idx += 1
                        lines.insert(insert_idx, f"- [{session_id}](sessions/{session_id}.md) - Step {metadata.get('step', 0)}, {metadata.get('status', 'active')}")
                    break

            content = "\n".join(lines)
            index_path.write_text(content)

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session checkpoint.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        session_path = self.storage_path / "sessions" / f"{session_id}.md"
        if session_path.exists():
            session_path.unlink()
            return True
        return False
