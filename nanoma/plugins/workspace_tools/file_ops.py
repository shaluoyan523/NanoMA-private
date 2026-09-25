"""Workspace Tools Plugin - File Operations Module.

Implements the following tools:
- create_file: Create a new file with auto-created parent directories
- append_file: Append content to a file (create if not exists)
- read_file: Read file content with offset/limit support
- rename_file: Rename or move a file/directory
- delete_file: Permanently delete a file or directory
- create_directory: Create directory recursively (mkdir -p)
- list_dir: List directory contents (with optional recursion)
- read_project_structure: Read project tree structure
- file_search: Search files by glob pattern
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

from nanoma.plugins.workspace_tools.path_utils import resolve_workspace_path


async def tool_create_file(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """Create a new file. Auto-creates parent directories.
    
    Args:
        path: File path (relative to workspace or absolute under workspace root)
        content: File content to write
        overwrite: If true, overwrite existing file (default: false)
    """
    file_path = args.get("path", "")
    content = args.get("content", "")
    overwrite = args.get("overwrite", False)

    if not file_path or not file_path.strip():
        return {"error": "path is required and must not be empty"}
    if content is None:
        return {"error": "content is required"}

    try:
        path = resolve_workspace_path(file_path, workspace, ctx)
    except ValueError:
        return {"error": "Access denied: path is outside workspace root"}

    if path.exists() and not overwrite:
        return {"error": f"File already exists: {path}. Pass overwrite=true to replace."}

    # Auto-create parent directories
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    return {"path": str(path), "bytes_written": len(content.encode("utf-8"))}


async def tool_append_file(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """Append content to a file. Creates the file if it doesn't exist.

    Args:
        path: File path
        content: Content to append
    """
    file_path = args.get("path", "")
    content = args.get("content", "")

    if not file_path or not file_path.strip():
        return {"error": "path is required"}
    if content is None:
        return {"error": "content is required"}

    try:
        path = resolve_workspace_path(file_path, workspace, ctx)
    except ValueError:
        return {"error": "Access denied: path is outside workspace root"}

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(content)

    return {"path": str(path), "bytes_written": len(content.encode("utf-8"))}


async def tool_read_file_advanced(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """Read file content with offset/limit support (1-indexed lines).

    Args:
        path: File path
        offset: Starting line number (1-based, optional)
        limit: Maximum number of lines to read (optional, default 2000)
    """
    file_path = args.get("path", "")
    offset = args.get("offset")
    limit = args.get("limit")

    if not file_path or not file_path.strip():
        return {"error": "path is required"}

    try:
        path = resolve_workspace_path(file_path, workspace, ctx)
    except ValueError:
        return {"error": "Access denied: path is outside workspace root"}

    if not path.exists():
        return {"error": f"File not found: {path}"}
    if path.is_dir():
        return {"error": f"Path is a directory, not a file: {path}"}

    content = path.read_text(encoding="utf-8", errors="replace")
    all_lines = content.split("\n")
    total_lines = len(all_lines)

    # Compute reading range
    start_line = max(1, offset) if offset else 1
    max_lines = limit if limit else 2000
    end_line = min(start_line + max_lines - 1, total_lines)

    selected_lines = all_lines[start_line - 1:end_line]
    result_content = "\n".join(selected_lines)

    return {
        "content": result_content,
        "total_lines": total_lines,
        "read_from": start_line,
        "read_to": end_line,
    }
