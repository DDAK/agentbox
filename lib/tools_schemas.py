execute_code_schema = {
    "type": "function",
    "name": "execute_code",
    "description": "Execute Python code and return the result or error.",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute as a string.",
            }
        },
        "required": ["code"],
        "additionalProperties": False,
    },
}

execute_bash_schema = {
    "type": "function",
    "name": "execute_bash",
    "description": "Execute a bash command and return the result or error.",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "A valid bash command to be executed as string.",
            },
        },
        "required": ["code"],
        "additionalProperties": False,
    },
}

list_directory_schema = {
    "type": "function",
    "name": "list_directory",
    "description": "Lists the names of files and subdirectories directly within a specified directory path. Can optionally ignore entries matching provided glob patterns. Returns paginated results.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list contents from",
            },
            "ignore": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of glob patterns to ignore when listing directory contents (e.g., ['*.tmp', '.DS_Store'])",
            },
            "offset": {
                "type": "integer",
                "description": "Starting position for pagination. Defaults to 0",
                "minimum": 0,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of entries to return. Between 0 and 64, defaults to 16.",
                "minimum": 1,
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
}

read_file_schema = {
    "type": "function",
    "name": "read_file",
    "description": "Reads content from a file with optional offset and limit for large files.",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to read",
            },
            "limit": {
                "type": "number",
                "description": "Maximum number of characters to read",
            },
            "offset": {
                "type": "number",
                "description": "Starting position in the file",
            },
        },
        "required": ["file_path"],
        "additionalProperties": False,
    },
}

write_file_schema = {
    "type": "function",
    "name": "write_file",
    "description": "Writes content to a file, creating directories if needed.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
            "file_path": {
                "type": "string",
                "description": "Absolute path where the file will be written",
            },
        },
        "required": ["content", "file_path"],
        "additionalProperties": False,
    },
}

search_file_content_schema = {
    "type": "function",
    "name": "search_file_content",
    "description": "Searches for a pattern within file contents using exact string matching, regex, or fuzzy matching. Can filter files by glob pattern. Returns paginated matching lines with file paths, line numbers, and similarity scores (for fuzzy mode).",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Pattern to search for within file contents",
            },
            "include": {
                "type": "string",
                "description": "Optional glob pattern to filter files (e.g., '*.py', '*.txt')",
            },
            "path": {
                "type": "string",
                "description": "Directory path to search in. Defaults to current working directory",
            },
            "use_regex": {
                "type": "boolean",
                "description": "Whether to treat pattern as regex. Defaults to false for exact string matching",
            },
            "fuzzy_threshold": {
                "type": "integer",
                "description": "Optional fuzzy matching threshold (0-100). If provided, enables fuzzy search mode",
                "minimum": 0,
                "maximum": 100,
            },
            "offset": {
                "type": "integer",
                "description": "Starting position for pagination. Defaults to 0",
                "minimum": 0,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of entries to return. Between 0 and 64, defaults to 16.",
                "minimum": 1,
            },
        },
        "required": ["pattern"],
        "additionalProperties": False,
    },
}

replace_in_file_schema = {
    "type": "function",
    "name": "replace_in_file",
    "description": "Replaces text within a file. By default, replaces a single occurrence, but can replace multiple occurrences when expected_replacements is specified. This tool requires providing significant context around the change to ensure precise targeting. Always use the read_file tool to examine the file's current content before attempting a text replacement.",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file where text replacement will be performed",
            },
            "new_string": {
                "type": "string",
                "description": "The new text that will replace the old text",
            },
            "old_string": {
                "type": "string",
                "description": "The existing text to be replaced. Should include significant context to ensure precise targeting",
            },
            "expected_replacements": {
                "type": "number",
                "description": "Optional number of expected replacements to perform. If not provided, replaces a single occurrence by default",
            },
        },
        "required": ["file_path", "new_string", "old_string"],
        "additionalProperties": False,
    },
}

glob_schema = {
    "type": "function",
    "name": "glob",
    "description": "Efficiently finds files matching specific glob patterns (e.g., src/**/*.ts, **/*.md), returning paginated absolute paths sorted by modification time (newest first). Ideal for quickly locating files based on their name or path structure, especially in large codebases.",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match files (e.g., 'src/**/*.ts', '**/*.md', '*.py')",
            },
            "path": {
                "type": "string",
                "description": "Optional directory path to search in. If not provided, searches from current working directory",
            },
            "ignore": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of glob patterns to ignore. If not provided, loads patterns from .gitignore if it exists",
            },
            "offset": {
                "type": "integer",
                "description": "Starting position for pagination. Defaults to 0",
                "minimum": 0,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of entries to return. Between 0 and 64, defaults to 16.",
                "minimum": 1,
            },
        },
        "required": ["pattern"],
        "additionalProperties": False,
    },
}

web_search_schema = {
    "type": "function",
    "name": "web_search",
    "description": "Search the web for information using DuckDuckGo. Use this to find current information, documentation, tutorials, news, or answers to questions. Returns titles, URLs, and snippets from search results.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query - be specific and include relevant keywords for best results",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default: 5, max: 10)",
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

web_fetch_schema = {
    "type": "function",
    "name": "web_fetch",
    "description": "Fetch and extract text content from a URL. Use this to read full articles, documentation pages, blog posts, or any web content. Returns the page content converted to markdown format.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch content from",
            },
            "max_length": {
                "type": "integer",
                "description": "Maximum content length to return in characters (default: 10000, max: 50000)",
                "minimum": 1000,
                "maximum": 50000,
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    },
}

tools_schemas = [
    execute_code_schema,
    execute_bash_schema,
    list_directory_schema,
    read_file_schema,
    write_file_schema,
    search_file_content_schema,
    replace_in_file_schema,
    glob_schema,
    # Web tools
    web_search_schema,
    web_fetch_schema,
]
