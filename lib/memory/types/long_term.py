"""
Long-term memory implementation.

Provides persistent knowledge storage using markdown files.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

from .base import BaseMemory


class LongTermMemory(BaseMemory):
    """
    Long-term memory using markdown files for persistence.

    Stores accumulated knowledge in categorized markdown files:
    - knowledge.md: Facts, patterns, API information
    - learnings.md: Error resolutions, successful approaches
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize long-term memory.

        Args:
            storage_path: Path to memory storage directory
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self._cache: dict = {}  # In-memory cache

        if self.storage_path:
            self._load_from_markdown()

    def add(self, content: str, category: str = "Facts", **kwargs) -> str:
        """
        Add knowledge to long-term memory.

        Args:
            content: The knowledge content
            category: Category (Facts, Patterns, API Information, Error Resolutions, etc.)
            **kwargs: Additional metadata

        Returns:
            The memory ID
        """
        memory_id = self._generate_id()
        timestamp = datetime.now().isoformat()

        # Store in cache
        if category not in self._cache:
            self._cache[category] = []

        self._cache[category].append({
            "id": memory_id,
            "content": content,
            "timestamp": timestamp,
            **kwargs,
        })

        # Persist to markdown
        if self.storage_path:
            self._persist_to_category(content, category, timestamp)

        return memory_id

    def add_fact(self, content: str) -> str:
        """Add a fact to knowledge."""
        return self.add(content, category="Facts")

    def add_pattern(self, content: str) -> str:
        """Add a discovered pattern."""
        return self.add(content, category="Patterns")

    def add_api_info(self, content: str) -> str:
        """Add API information."""
        return self.add(content, category="API Information")

    def add_learning(self, content: str, learning_type: str = "Successful Approaches") -> str:
        """
        Add a learning to the learnings file.

        Args:
            content: The learning content
            learning_type: Type (Error Resolutions, Successful Approaches, Things to Avoid)

        Returns:
            The memory ID
        """
        return self.add(content, category=learning_type)

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve relevant knowledge by keyword search.

        Args:
            query: Search query
            top_k: Maximum number of results

        Returns:
            List of knowledge strings
        """
        query_words = set(query.lower().split())
        results = []

        for category, memories in self._cache.items():
            for memory in memories:
                content = memory.get("content", "")
                content_words = set(content.lower().split())
                score = len(query_words & content_words)
                if score > 0:
                    results.append((score, f"[{category}] {content}"))

        # Sort by relevance
        results.sort(reverse=True)
        return [content for _, content in results[:top_k]]

    def get_by_category(self, category: str) -> List[str]:
        """
        Get all knowledge in a category.

        Args:
            category: The category to retrieve

        Returns:
            List of content strings
        """
        return [m.get("content", "") for m in self._cache.get(category, [])]

    def get_all(self) -> Dict[str, List[str]]:
        """
        Get all knowledge organized by category.

        Returns:
            Dictionary mapping category to list of content strings
        """
        return {
            category: [m.get("content", "") for m in memories]
            for category, memories in self._cache.items()
        }

    def clear(self) -> None:
        """Clear all long-term memories."""
        self._cache.clear()
        if self.storage_path:
            self._reinitialize_files()

    def _load_from_markdown(self):
        """Load memories from markdown files."""
        knowledge_path = self.storage_path / "long_term" / "knowledge.md"
        learnings_path = self.storage_path / "long_term" / "learnings.md"

        if knowledge_path.exists():
            self._parse_markdown_file(knowledge_path)

        if learnings_path.exists():
            self._parse_markdown_file(learnings_path)

    def _parse_markdown_file(self, file_path: Path):
        """Parse a markdown file and populate cache."""
        content = file_path.read_text()
        current_section = None

        for line in content.split("\n"):
            if line.startswith("## "):
                current_section = line[3:].strip()
                if current_section not in self._cache:
                    self._cache[current_section] = []
            elif line.startswith("- ") and current_section:
                # Extract content (may have timestamp prefix)
                item_content = line[2:]
                # Remove date prefix if present [YYYY-MM-DD]
                item_content = re.sub(r"^\[\d{4}-\d{2}-\d{2}\]\s*", "", item_content)

                if item_content and not item_content.startswith("<!--"):
                    self._cache[current_section].append({
                        "id": self._generate_id(),
                        "content": item_content,
                        "timestamp": datetime.now().isoformat(),
                    })

    def _persist_to_category(self, content: str, category: str, timestamp: str):
        """Persist content to appropriate markdown file."""
        # Determine which file to use based on category
        learning_categories = {"Error Resolutions", "Successful Approaches", "Things to Avoid"}

        if category in learning_categories:
            file_path = self.storage_path / "long_term" / "learnings.md"
        else:
            file_path = self.storage_path / "long_term" / "knowledge.md"

        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Read existing content or create new
        if file_path.exists():
            existing = file_path.read_text()
        else:
            existing = self._get_initial_content(category in learning_categories)

        # Find or create section
        section_header = f"## {category}"
        if section_header not in existing:
            # Add new section before footer
            footer_match = re.search(r"\n---\n\*Last updated:", existing)
            if footer_match:
                insert_pos = footer_match.start()
                new_section = f"\n{section_header}\n\n"
                existing = existing[:insert_pos] + new_section + existing[insert_pos:]

        # Add entry under section
        lines = existing.split("\n")
        for i, line in enumerate(lines):
            if line == section_header:
                # Find insertion point (after section header and comments)
                insert_idx = i + 1
                while insert_idx < len(lines) and (
                    lines[insert_idx].startswith("<!--") or lines[insert_idx] == ""
                ):
                    insert_idx += 1

                # Add date prefix for learnings
                if category in learning_categories:
                    entry = f"- [{timestamp[:10]}] {content}"
                else:
                    entry = f"- {content}"

                lines.insert(insert_idx, entry)
                break

        # Update timestamp
        new_content = "\n".join(lines)
        new_content = re.sub(
            r"\*Last updated: .*\*",
            f"*Last updated: {timestamp}*",
            new_content
        )

        file_path.write_text(new_content)

    def _get_initial_content(self, is_learnings: bool) -> str:
        """Get initial content for a new markdown file."""
        timestamp = datetime.now().isoformat()

        if is_learnings:
            return f"""# Agent Learnings

## Error Resolutions
<!-- How errors were resolved -->

## Successful Approaches
<!-- What worked well -->

## Things to Avoid
<!-- What didn't work -->

---
*Last updated: {timestamp}*
"""
        else:
            return f"""# Long-Term Knowledge

## Facts
<!-- Project facts and configuration -->

## Patterns
<!-- Discovered patterns and best practices -->

## API Information
<!-- API endpoints, configurations, etc. -->

---
*Last updated: {timestamp}*
"""

    def _reinitialize_files(self):
        """Reinitialize markdown files after clear."""
        if not self.storage_path:
            return

        knowledge_path = self.storage_path / "long_term" / "knowledge.md"
        learnings_path = self.storage_path / "long_term" / "learnings.md"

        knowledge_path.write_text(self._get_initial_content(False))
        learnings_path.write_text(self._get_initial_content(True))

    def __len__(self) -> int:
        """Return total number of memories."""
        return sum(len(memories) for memories in self._cache.values())
