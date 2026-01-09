"""
Tools for the coding agent.

Provides file system operations and code execution capabilities.
"""

import json
import os
import fnmatch
import glob as glob_module
import re
from typing import Callable, Optional, List, Dict, Any, Tuple

from .sandbox import BaseSandbox


class ToolError(Exception):
    """Custom exception for tool failures."""
    pass


def _paginate_results(
    results: List[Any], offset: int = 0, limit: Optional[int] = 16
) -> Dict[str, Any]:
    """Slice results and return pagination metadata."""
    total = len(results)
    start = max(0, offset)
    limit = min(limit or 16, 64)
    end = start + limit if limit else total
    page = results[start:end]

    return {
        "pagination": {
            "total": total,
            "offset": start,
            "limit": limit,
            "has_more": end < total,
        },
        "results": page,
    }


def secure_path(requested_path: str, working_dir: str = None) -> str:
    """Keep paths locked to working_dir."""
    working_dir = working_dir or os.getcwd()
    wd_real = os.path.realpath(working_dir)

    if not requested_path:
        return wd_real

    if os.path.isabs(requested_path):
        target_real = os.path.realpath(requested_path)
    else:
        target_real = os.path.realpath(os.path.join(wd_real, requested_path))

    if not target_real.startswith(wd_real + os.sep) and target_real != wd_real:
        raise ToolError(
            f"Path '{requested_path}' escapes working directory. You can read/edit only files in '{working_dir}'."
        )

    return target_real


def execute_code(sbx: BaseSandbox, code: str, language: str = "python") -> Tuple[dict, dict]:
    """Execute code in the sandbox."""
    execution = sbx.run_code(code, language)
    metadata = {}
    return execution.to_json(), metadata


def execute_bash(sbx: BaseSandbox, code: str) -> Tuple[dict, dict]:
    """Execute bash command in the sandbox."""
    return execute_code(sbx, code, language="bash")


def list_directory(
    sbx: BaseSandbox,
    path: str = ".",
    ignore: Optional[List[str]] = None,
    offset: int = 0,
    limit: Optional[int] = 16,
) -> Tuple[dict, dict]:
    """List directory contents with pagination."""
    working_dir = sbx.working_dir
    path = secure_path(path, working_dir)

    if not os.path.exists(path):
        return {"error": f"Path does not exist: {path}"}, {}
    if not os.path.isdir(path):
        return {"error": f"Path is not a directory: {path}"}, {}

    entries = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if ignore and any(fnmatch.fnmatch(item, pattern) for pattern in ignore):
            continue

        try:
            stat = os.stat(item_path)
            entries.append({
                "name": item,
                "type": "directory" if os.path.isdir(item_path) else "file",
                "size": stat.st_size,
                "modified": stat.st_mtime,
            })
        except OSError:
            continue

    entries.sort(key=lambda x: (x["type"] != "directory", x["name"]))
    result = {**_paginate_results(entries, offset, limit), "path": path}
    return result, {}


def read_file(
    sbx: BaseSandbox,
    file_path: str,
    limit: Optional[int] = None,
    offset: int = 0,
) -> Tuple[dict, dict]:
    """Read file content with optional offset and limit."""
    working_dir = sbx.working_dir
    file_path = secure_path(file_path, working_dir)

    if not os.path.exists(file_path):
        return {"error": f"File does not exist: {file_path}"}, {}
    if not os.path.isfile(file_path):
        return {"error": f"Path is not a file: {file_path}"}, {}

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            if offset > 0:
                f.seek(offset)
            content = f.read(limit) if limit else f.read()
        return {"content": content, "size": len(content), "path": file_path}, {}
    except PermissionError:
        return {"error": f"Permission denied: {file_path}"}, {}
    except Exception as e:
        return {"error": str(e)}, {}


def write_file(
    sbx: BaseSandbox,
    content: str,
    file_path: str,
) -> Tuple[dict, dict]:
    """Write content to file, creating directories if needed."""
    working_dir = sbx.working_dir
    file_path = secure_path(file_path, working_dir)

    try:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        file_size = os.path.getsize(file_path)
        return {
            "message": f"Written {file_size} bytes to {file_path}",
            "size": file_size,
            "path": file_path,
        }, {}
    except PermissionError:
        return {"error": f"Permission denied: {file_path}"}, {}
    except Exception as e:
        return {"error": str(e)}, {}


def replace_in_file(
    sbx: BaseSandbox,
    file_path: str,
    old_string: str,
    new_string: str,
    expected_replacements: int = 1,
) -> Tuple[dict, dict]:
    """Replace text in file with validation."""
    working_dir = sbx.working_dir
    file_path = secure_path(file_path, working_dir)

    if not os.path.exists(file_path):
        return {"error": f"File does not exist: {file_path}"}, {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        actual_count = content.count(old_string)
        if actual_count != expected_replacements:
            return {
                "error": f"Expected {expected_replacements} occurrences, found {actual_count}"
            }, {}

        new_content = content.replace(old_string, new_string, expected_replacements)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        return {
            "replacements": expected_replacements,
            "message": f"Replaced {expected_replacements} occurrences in {file_path}",
        }, {}
    except Exception as e:
        return {"error": str(e)}, {}


def search_file_content(
    sbx: BaseSandbox,
    pattern: str,
    include: Optional[str] = None,
    path: str = ".",
    use_regex: bool = False,
    fuzzy_threshold: Optional[int] = None,
    offset: int = 0,
    limit: Optional[int] = 16,
) -> Tuple[dict, dict]:
    """Search for pattern in file contents with pagination."""
    working_dir = sbx.working_dir
    path = secure_path(path, working_dir)

    results = []
    total_files_searched = 0

    regex_pattern = None
    if use_regex:
        try:
            regex_pattern = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return {"error": f"Invalid regex pattern: {e}"}, {}

    # Try to use rapidfuzz for fuzzy matching if available
    fuzz = None
    if fuzzy_threshold is not None:
        try:
            from rapidfuzz import fuzz as rapidfuzz
            fuzz = rapidfuzz
        except ImportError:
            return {"error": "Fuzzy matching requires rapidfuzz. Install with: pip install rapidfuzz"}, {}

    for root, dirs, files in os.walk(path):
        for file in files:
            if include and not fnmatch.fnmatch(file, include):
                continue

            filepath = os.path.join(root, file)
            total_files_searched += 1

            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        line_stripped = line.strip()
                        match_data = {
                            "file": filepath,
                            "line": line_num,
                            "content": line_stripped,
                        }

                        if fuzzy_threshold is not None and fuzz:
                            similarity = fuzz.partial_ratio(pattern.lower(), line.lower())
                            if similarity >= fuzzy_threshold:
                                match_data["similarity"] = similarity
                                results.append(match_data)
                        elif use_regex:
                            if regex_pattern.search(line):
                                results.append(match_data)
                        else:
                            if pattern.lower() in line.lower():
                                results.append(match_data)
            except Exception:
                continue

    if fuzzy_threshold is not None:
        results.sort(key=lambda x: x.get("similarity", 0), reverse=True)

    paginated = _paginate_results(results, offset, limit)
    return {**paginated, "files_searched": total_files_searched}, {}


def glob_search(
    sbx: BaseSandbox,
    pattern: str,
    path: str = ".",
    ignore: Optional[List[str]] = None,
    offset: int = 0,
    limit: Optional[int] = 16,
) -> Tuple[dict, dict]:
    """Find files matching glob pattern with pagination."""
    working_dir = sbx.working_dir
    path = secure_path(path, working_dir)
    original_cwd = os.getcwd()

    try:
        os.chdir(path)

        # Auto-load .gitignore patterns if ignore is None
        if ignore is None:
            gitignore_path = os.path.join(path, ".gitignore")
            if os.path.isfile(gitignore_path):
                ignore = []
                try:
                    with open(gitignore_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and not line.startswith("/"):
                                if line.endswith("/"):
                                    line = line[:-1]
                                ignore.append(line)
                except Exception:
                    ignore = None

        matches = glob_module.glob(pattern, recursive=True)

        results = []
        for match in matches:
            if ignore:
                should_ignore = False
                path_parts = match.split(os.sep)

                for ignore_pattern in ignore:
                    if any(fnmatch.fnmatch(part, ignore_pattern) for part in path_parts):
                        should_ignore = True
                        break
                    if match.startswith(ignore_pattern + os.sep) or match == ignore_pattern:
                        should_ignore = True
                        break
                    if fnmatch.fnmatch(match, ignore_pattern):
                        should_ignore = True
                        break

                if should_ignore:
                    continue

            abs_path = os.path.abspath(match)
            if os.path.isfile(abs_path):
                try:
                    stat = os.stat(abs_path)
                    results.append({
                        "path": abs_path,
                        "relative_path": match,
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                    })
                except OSError:
                    continue

        results.sort(key=lambda x: x["modified"], reverse=True)
        return {**_paginate_results(results, offset, limit), "pattern": pattern}, {}

    finally:
        os.chdir(original_cwd)


# Web tools (no sandbox needed)
def web_search(query: str, num_results: int = 5, **kwargs) -> Tuple[dict, dict]:
    """Search the web using DuckDuckGo.

    Args:
        query: Search query string
        num_results: Number of results to return (default: 5, max: 10)

    Returns:
        Dictionary with search results containing titles, URLs, and snippets
    """
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=min(num_results, 10)))

        return {
            "results": [
                {"title": r["title"], "url": r["href"], "snippet": r["body"]}
                for r in results
            ],
            "query": query,
            "count": len(results)
        }, {}
    except ImportError:
        return {"error": "duckduckgo-search package not installed. Install with: pip install duckduckgo-search"}, {}
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}, {}


def web_fetch(url: str, max_length: int = 10000, **kwargs) -> Tuple[dict, dict]:
    """Fetch and extract text content from a URL.

    Args:
        url: The URL to fetch content from
        max_length: Maximum content length to return (default: 10000 chars)

    Returns:
        Dictionary with URL, title, content (as markdown), and metadata
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        import html2text

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, timeout=30, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            element.decompose()

        # Get title
        title = soup.title.string if soup.title else ""

        # Convert to markdown
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.body_width = 0  # No line wrapping
        markdown = h.handle(str(soup))

        # Truncate if needed
        content = markdown[:max_length]
        truncated = len(markdown) > max_length

        return {
            "url": url,
            "title": title,
            "content": content,
            "content_length": len(markdown),
            "truncated": truncated
        }, {}
    except ImportError as e:
        missing = str(e).split("'")[1] if "'" in str(e) else "required packages"
        return {"error": f"Missing package: {missing}. Install with: pip install beautifulsoup4 html2text requests"}, {}
    except requests.exceptions.Timeout:
        return {"error": f"Request timed out for URL: {url}"}, {}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}, {}
    except Exception as e:
        return {"error": f"Fetch failed: {str(e)}"}, {}


# Tools registry - maps tool names to functions
tools = {
    "execute_code": execute_code,
    "execute_bash": execute_bash,
    "list_directory": list_directory,
    "read_file": read_file,
    "write_file": write_file,
    "replace_in_file": replace_in_file,
    "search_file_content": search_file_content,
    "glob": glob_search,
    # Web tools
    "web_search": web_search,
    "web_fetch": web_fetch,
}


def execute_tool(name: str, args: str, tools: Dict[str, Callable], **kwargs) -> Tuple[dict, dict]:
    """Execute a tool by name with the given arguments."""
    metadata = {}
    try:
        parsed_args = json.loads(args)
        if name not in tools:
            return {"error": f"Tool {name} doesn't exist."}, metadata
        result, metadata = tools[name](**parsed_args, **kwargs)
        return result, metadata
    except json.JSONDecodeError as e:
        return {"error": f"{name} failed to parse arguments: {str(e)}"}, metadata
    except KeyError as e:
        return {"error": f"Missing key in arguments: {str(e)}"}, metadata
    except ToolError as e:
        return {"error": str(e)}, metadata
    except Exception as e:
        return {"error": str(e)}, metadata
