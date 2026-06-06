"""Work tools: shell, file_read, file_write, file_list, grep."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from nanoma.core import ToolContext


async def tool_shell(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """Execute a shell command."""
    from nanoma.sandbox import shell_exec
    cmd = args.get("command", "")
    timeout = args.get("timeout", 30)
    max_output = ctx.shell_max_output

    if ctx.sandbox is not None:
        result = await ctx.sandbox.exec(cmd, workspace, ctx.shared_dir, timeout)
    else:
        result = await shell_exec(cmd, workspace, ctx.shared_dir, timeout)

    if max_output > 0:
        for key in ("stdout", "stderr"):
            val = result.get(key, "")
            if len(val) > max_output:
                out_file = workspace / f".output_{key}_{hash(cmd) % 100000:05d}.txt"
                out_file.write_text(val)
                result[key] = val[:max_output] + f"\n...(truncated {len(val)} chars, full: {out_file.name})"
    return result


async def tool_file_read(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """Read a file."""
    path = Path(args.get("path", ""))
    if not path.is_absolute():
        path = workspace / path
    # Sandbox: must be under workspace root
    try:
        path.resolve().relative_to(ctx.workspace_root.resolve())
    except ValueError:
        return {"error": f"Access denied: outside workspace"}
    if not path.exists():
        return {"error": f"Not found: {path}"}
    if not path.is_file():
        return {"error": f"Not a file: {path}"}
    content = path.read_text(errors="replace")
    max_chars = ctx.file_read_max_chars
    if max_chars > 0 and len(content) > max_chars:
        return {"content": content[:max_chars], "size": path.stat().st_size, "truncated": True}
    return {"content": content, "size": path.stat().st_size, "truncated": False}


async def tool_file_write(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """Write a file."""
    path = Path(args.get("path", ""))
    if not path.is_absolute():
        path = workspace / path
    try:
        path.resolve().relative_to(ctx.workspace_root.resolve())
    except ValueError:
        return {"error": "Access denied: outside workspace"}
    content = args.get("content", "")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return {"path": str(path), "bytes": len(content.encode())}


async def tool_file_list(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """List directory."""
    path = Path(args.get("path", "."))
    if not path.is_absolute():
        path = workspace / path
    if not path.exists():
        return {"error": f"Not found: {path}"}
    entries = []
    for e in sorted(path.iterdir()):
        entries.append({"name": e.name + ("/" if e.is_dir() else ""), "size": e.stat().st_size if e.is_file() else None})
    max_entries = ctx.file_list_max_entries
    if max_entries > 0 and len(entries) > max_entries:
        return {"entries": entries[:max_entries], "truncated": True, "total": len(entries)}
    return {"entries": entries}


async def tool_grep(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """Grep files."""
    pattern = args.get("pattern", "")
    path = args.get("path", ".")
    if not Path(path).is_absolute():
        path = str(workspace / path)
    try:
        proc = await asyncio.create_subprocess_exec(
            "grep", "-rn", "--include=*", pattern, path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        all_lines = [l for l in stdout.decode(errors="replace").strip().split("\n") if l]
        max_results = ctx.grep_max_results
        if max_results > 0 and len(all_lines) > max_results:
            return {"matches": all_lines[:max_results], "count": max_results, "total_matches": len(all_lines), "truncated": True}
        return {"matches": all_lines, "count": len(all_lines), "total_matches": len(all_lines), "truncated": False}
    except asyncio.TimeoutError:
        return {"error": "Timeout", "matches": []}
    except Exception as e:
        return {"error": str(e), "matches": []}


# --- Registry ---

WORK_TOOLS: dict[str, dict[str, Any]] = {
    "shell": {
        "handler": tool_shell,
        "schema": {"type": "function", "function": {
            "name": "shell",
            "description": "Execute a shell command in your workspace directory. Returns {stdout, stderr, returncode}. CWD is your private workspace. Use $SHARED to reference the shared directory. Long outputs are truncated and saved to a file.",
            "parameters": {"type": "object", "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {"type": "integer", "description": "Max execution time in seconds", "default": 30},
            }, "required": ["command"]},
        }},
    },
    "file_read": {
        "handler": tool_file_read,
        "schema": {"type": "function", "function": {
            "name": "file_read",
            "description": "Read a file's content. Relative paths resolve from your workspace. You can also read from shared/ or other agents' workspaces (under the workspace root). Returns {content, size, truncated}.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "File path (relative to workspace, or absolute under workspace root)"},
            }, "required": ["path"]},
        }},
    },
    "file_write": {
        "handler": tool_file_write,
        "schema": {"type": "function", "function": {
            "name": "file_write",
            "description": "Write content to a file (creates or overwrites). Parent directories are created automatically. Write to shared/ to make files visible to other agents.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "File path (relative to workspace). Use 'shared/filename' to write to the shared directory."},
                "content": {"type": "string", "description": "File content to write"},
            }, "required": ["path", "content"]},
        }},
    },
    "file_list": {
        "handler": tool_file_list,
        "schema": {"type": "function", "function": {
            "name": "file_list",
            "description": "List directory contents. Returns [{name, size}]. Names ending with / are directories. Use to discover files written by other agents in shared/.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "Directory path (default: your workspace root)", "default": "."},
            }},
        }},
    },
    "grep": {
        "handler": tool_grep,
        "schema": {"type": "function", "function": {
            "name": "grep",
            "description": "Recursively search files for a regex pattern. Returns matching lines with file paths and line numbers. Useful for finding content across shared/ or your workspace.",
            "parameters": {"type": "object", "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "Directory to search (default: workspace root)", "default": "."},
            }, "required": ["pattern"]},
        }},
    },
}
