"""Workspace Tools Plugin for NanoMA.

Provides 9 structured workspace tools that satisfy the Orthogonal Minimalism
principle — each tool exists because shell cannot reliably replicate its
interaction model.

Tool Set:
- 3 File I/O (create, append, read) — shell heredocs break on special chars
- 3 File Editing (replace, multi-replace, V4A patch) — atomic multi-step ops
- 3 Code Search (grep, outline, read symbol) — structured results no CLI gives

See DESIGN.md for the full rationale.

Usage:
    from nanoma.plugins.workspace_tools import WORKSPACE_TOOLS, get_tool_schemas

    # Merge into the runtime's tool registry:
    all_tools = {**basic_tools, **WORKSPACE_TOOLS, **META_TOOLS}
"""

from __future__ import annotations

from typing import Any

from nanoma.plugins.workspace_tools.file_ops import (
    tool_create_file,
    tool_append_file,
    tool_read_file_advanced,
)
from nanoma.plugins.workspace_tools.file_edit import (
    tool_replace_string,
    tool_multi_replace_string,
    tool_apply_patch,
    find_and_replace,
    count_occurrences,
    parse_patch,
    apply_hunks,
)
from nanoma.plugins.workspace_tools.code_search import (
    tool_grep_search,
    tool_code_outline,
    tool_read_symbol,
    escape_regex,
    is_binary_file,
    find_block_end,
    parse_file_symbols,
)

__all__ = [
    "WORKSPACE_TOOLS",
    "get_tool_schemas",
    # File I/O
    "tool_create_file", "tool_append_file", "tool_read_file_advanced",
    # File edit
    "tool_replace_string", "tool_multi_replace_string", "tool_apply_patch",
    "find_and_replace", "count_occurrences", "parse_patch", "apply_hunks",
    # Code search
    "tool_grep_search", "tool_code_outline", "tool_read_symbol",
    "escape_regex", "is_binary_file", "find_block_end", "parse_file_symbols",
]


# ─── Tool registry (NanoMA format) ──────────────────────────────────────────

WORKSPACE_TOOLS: dict[str, dict[str, Any]] = {
    # ─── File I/O (Reliability: shell heredocs break on special chars) ────
    "ws_create_file": {
        "handler": tool_create_file,
        "schema": {"type": "function", "function": {
            "name": "ws_create_file",
            "description": "Create a new file with content. Auto-creates parent directories. Fails if file exists unless overwrite=true. Use this instead of shell heredocs when content contains backticks, $, or backslashes.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "File path (relative to workspace)"},
                "content": {"type": "string", "description": "File content to write"},
                "overwrite": {"type": "boolean", "description": "If true, overwrite existing file", "default": False},
            }, "required": ["path", "content"]},
        }},
    },
    "ws_append_file": {
        "handler": tool_append_file,
        "schema": {"type": "function", "function": {
            "name": "ws_append_file",
            "description": "Append content to a file (creates if not exists). Use for incremental writes or large files that exceed argument limits.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "File path"},
                "content": {"type": "string", "description": "Content to append"},
            }, "required": ["path", "content"]},
        }},
    },
    "ws_read_file": {
        "handler": tool_read_file_advanced,
        "schema": {"type": "function", "function": {
            "name": "ws_read_file",
            "description": "Read file content with line-based pagination. Returns {content, total_lines, read_from, read_to}. Lines are 1-indexed. Default limit is 2000 lines.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "File path"},
                "offset": {"type": "integer", "description": "Start line number (1-based)"},
                "limit": {"type": "integer", "description": "Max lines to read (default: 2000)"},
            }, "required": ["path"]},
        }},
    },

    # ─── File Editing (Atomicity: multi-step ops impossible in sed) ───────
    "ws_replace_string": {
        "handler": tool_replace_string,
        "schema": {"type": "function", "function": {
            "name": "ws_replace_string",
            "description": "Find and replace a string in a file. Uses 4-tier matching: exact → trimmed → indent-agnostic → normalized whitespace. The old_string must match exactly once. File is unchanged if no match found.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "File path"},
                "old_string": {"type": "string", "description": "Text to find (must appear exactly once)"},
                "new_string": {"type": "string", "description": "Replacement text"},
            }, "required": ["path", "old_string", "new_string"]},
        }},
    },
    "ws_multi_replace": {
        "handler": tool_multi_replace_string,
        "schema": {"type": "function", "function": {
            "name": "ws_multi_replace",
            "description": "Execute multiple find-replace operations on a file atomically. All replacements must succeed or the file is unchanged (all-or-nothing).",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "File path"},
                "replacements": {"type": "array", "items": {"type": "object", "properties": {
                    "old_string": {"type": "string"}, "new_string": {"type": "string"},
                }, "required": ["old_string", "new_string"]}, "description": "List of {old_string, new_string} pairs"},
            }, "required": ["path", "replacements"]},
        }},
    },
    "ws_apply_patch": {
        "handler": tool_apply_patch,
        "schema": {"type": "function", "function": {
            "name": "ws_apply_patch",
            "description": "Apply a V4A diff/patch. Supports Add, Update, Delete file operations with context-based matching. For complex multi-file edits.",
            "parameters": {"type": "object", "properties": {
                "input": {"type": "string", "description": "V4A patch content (must include *** Begin Patch / *** End Patch)"},
                "explanation": {"type": "string", "description": "Brief description of the change"},
            }, "required": ["input", "explanation"]},
        }},
    },

    # ─── Code Search (Structured I/O: no CLI equivalent) ─────────────────
    "ws_grep": {
        "handler": tool_grep_search,
        "schema": {"type": "function", "function": {
            "name": "ws_grep",
            "description": "Fast text or regex search across workspace files. Returns structured results with file path, line number, content, and match positions. Auto-skips binary files and noise directories (.git, node_modules, etc.).",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "Search text or ERE regex pattern"},
                "is_regexp": {"type": "boolean", "description": "Whether query is a regex", "default": False},
                "case_sensitive": {"type": "boolean", "description": "Case-sensitive search", "default": False},
                "include_pattern": {"type": "string", "description": "Glob to filter files (e.g. '*.py')"},
                "path": {"type": "string", "description": "Optional directory or file path to search. Supports $SHARED/..."},
                "root": {"type": "string", "description": "Alias for path. Optional directory or file path to search."},
                "max_results": {"type": "integer", "description": "Max results (default: 100)", "default": 100},
            }, "required": ["query"]},
        }},
    },
    "ws_code_outline": {
        "handler": tool_code_outline,
        "schema": {"type": "function", "function": {
            "name": "ws_code_outline",
            "description": "Get the symbol outline of a source file. Returns all functions, classes, interfaces, etc. with their line ranges. Use to understand file structure before reading specific symbols.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "Path to source file"},
            }, "required": ["path"]},
        }},
    },
    "ws_read_symbol": {
        "handler": tool_read_symbol,
        "schema": {"type": "function", "function": {
            "name": "ws_read_symbol",
            "description": "Read a specific symbol's source code from a file by name. More precise than reading the whole file — extracts just the function/class/struct you need.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "Path to source file"},
                "symbol_name": {"type": "string", "description": "Exact name of the symbol to read"},
            }, "required": ["path", "symbol_name"]},
        }},
    },
}


def get_tool_schemas() -> list[dict[str, Any]]:
    """Get all tool schemas for LLM function calling."""
    return [t["schema"] for t in WORKSPACE_TOOLS.values()]
