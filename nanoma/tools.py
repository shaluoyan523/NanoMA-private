"""Work tools: shell, files, grep, and protocol helpers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
from pathlib import Path
from typing import Any, TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from nanoma.core import ToolContext


async def tool_shell(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """Execute a shell command."""
    if not _tool_enabled("shell", ctx):
        return {"error": "Tool disabled by runtime policy: shell"}
    from nanoma.sandbox import shell_exec
    cmd = args.get("command", "")
    timeout = args.get("timeout", 30)
    max_output = ctx.shell_max_output
    mode = getattr(ctx, "shell_mode", "unrestricted")
    effective_workspace = workspace
    if mode == "disabled":
        return {"error": "Tool disabled by runtime policy: shell"}
    if mode == "controlled":
        normalized = _normalize_controlled_shell_command(cmd, workspace, ctx)
        if isinstance(normalized, str):
            return {"error": normalized, "policy": "controlled_shell"}
        cmd = normalized.command
        effective_workspace = normalized.workspace
        policy_error = _controlled_shell_policy_error(cmd, ctx)
        if policy_error:
            return {"error": policy_error, "policy": "controlled_shell"}

    if ctx.sandbox is not None:
        result = await ctx.sandbox.exec(cmd, effective_workspace, ctx.shared_dir, timeout)
    else:
        result = await shell_exec(cmd, effective_workspace, ctx.shared_dir, timeout)

    if max_output > 0:
        for key in ("stdout", "stderr"):
            val = result.get(key, "")
            if len(val) > max_output:
                out_file = workspace / f".output_{key}_{hash(cmd) % 100000:05d}.txt"
                out_file.write_text(val)
                result[key] = val[:max_output] + f"\n...(truncated {len(val)} chars, full: {out_file.name})"
    return result


class ControlledShellCommand(NamedTuple):
    command: str
    workspace: Path


CONTROLLED_SHELL_DEFAULT_COMMANDS = {
    "python",
    "python3",
    "pytest",
    "git",
    "python3.10",
    "python3.11",
    "python3.12",
}
CONTROLLED_SHELL_BLOCK_TOKENS = {
    "|",
    "||",
    "&",
    "&&",
    ";",
    ">",
    ">>",
    "<",
    "<<",
    "$(",
    "`",
}
CONTROLLED_SHELL_BLOCK_COMMANDS = {
    "bash",
    "sh",
    "curl",
    "wget",
    "nc",
    "ncat",
    "ssh",
    "scp",
    "rsync",
    "pip",
    "pip3",
    "python -m pip",
    "python3 -m pip",
    "rm",
    "mv",
    "cp",
    "chmod",
    "chown",
    "sudo",
}


def _normalize_controlled_shell_command(cmd: str, workspace: Path, ctx: "ToolContext") -> ControlledShellCommand | str:
    cmd = str(cmd or "").strip()
    if not cmd:
        return "Controlled shell rejected empty command"
    try:
        parts = shlex.split(cmd)
    except ValueError as e:
        return f"Controlled shell rejected unparsable command: {e}"
    if not parts:
        return "Controlled shell rejected empty command"
    stripped_stderr_merge = False
    if parts[-1] == "2>&1":
        parts = parts[:-1]
        stripped_stderr_merge = True
        if not parts:
            return "Controlled shell rejected empty command"
    if "&&" not in parts:
        command = shlex.join(parts) if stripped_stderr_merge else cmd
        return ControlledShellCommand(command=command, workspace=workspace)
    if parts.count("&&") != 1 or len(parts) < 4 or parts[0] != "cd" or parts[2] != "&&":
        return "Controlled shell rejected command containing shell operators/redirection"
    target = _resolve_workspace_path(parts[1], workspace, ctx)
    if target is None:
        return "Controlled shell rejected cd path outside workspace/shared roots"
    if not target.exists() or not target.is_dir():
        return f"Controlled shell rejected cd path that is not a directory: {parts[1]}"
    command = shlex.join(parts[3:])
    if not command:
        return "Controlled shell rejected empty command"
    return ControlledShellCommand(command=command, workspace=target)


def _controlled_shell_policy_error(cmd: str, ctx: "ToolContext") -> str | None:
    cmd = str(cmd or "").strip()
    if not cmd:
        return "Controlled shell rejected empty command"
    if any(token in cmd for token in CONTROLLED_SHELL_BLOCK_TOKENS):
        return "Controlled shell rejected command containing shell operators/redirection"
    try:
        parts = shlex.split(cmd)
    except ValueError as e:
        return f"Controlled shell rejected unparsable command: {e}"
    if not parts:
        return "Controlled shell rejected empty command"
    exe = Path(parts[0]).name
    two_part = " ".join(parts[:3]).lower()
    if two_part.startswith(("python -m pip", "python3 -m pip")):
        return "Controlled shell rejected package installation command"
    if exe in CONTROLLED_SHELL_BLOCK_COMMANDS:
        return f"Controlled shell rejected command: {exe}"
    allowed = getattr(ctx, "controlled_shell_allowed_commands", None) or CONTROLLED_SHELL_DEFAULT_COMMANDS
    if exe not in allowed:
        return f"Controlled shell only allows: {', '.join(sorted(allowed))}"
    return None


async def tool_file_read(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """Read a file."""
    if not _tool_enabled("file_read", ctx):
        return {"error": "Tool disabled by runtime policy: file_read"}
    path = _resolve_workspace_path(_path_arg(args), workspace, ctx)
    if path is None:
        return {"error": "Access denied: outside workspace"}
    if not path.exists():
        return {"error": f"Not found: {path}"}
    if not path.is_file():
        return {"error": f"Not a file: {path}"}
    content = path.read_text(errors="replace")
    sha256 = hashlib.sha256(content.encode()).hexdigest()
    total_lines = None
    offset = args.get("offset", args.get("start_line"))
    limit = args.get("limit", args.get("lines"))
    if offset is not None or limit is not None:
        try:
            start_line = max(1, int(offset or 1))
            line_limit = int(limit) if limit not in (None, "") else 200
        except (TypeError, ValueError):
            return {"error": "offset/start_line and limit/lines must be integers"}
        if line_limit <= 0:
            return {"error": "limit/lines must be positive"}
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)
        start_idx = start_line - 1
        end_idx = min(total_lines, start_idx + line_limit)
        content = "".join(lines[start_idx:end_idx]) if start_idx < total_lines else ""
        return {
            "content": content,
            "size": path.stat().st_size,
            "sha256": sha256,
            "truncated": end_idx < total_lines,
            "offset": start_line,
            "limit": line_limit,
            "lines_returned": max(0, end_idx - start_idx),
            "total_lines": total_lines,
        }
    max_chars = ctx.file_read_max_chars
    if max_chars > 0 and len(content) > max_chars:
        return {
            "content": content[:max_chars],
            "size": path.stat().st_size,
            "sha256": sha256,
            "truncated": True,
            "total_lines": len(content.splitlines()),
            "range_hint": "Use file_read with offset/start_line and limit/lines to inspect a smaller line range.",
        }
    return {"content": content, "size": path.stat().st_size, "sha256": sha256, "truncated": False, "total_lines": total_lines}


async def tool_file_write(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """Write a file."""
    if not _tool_enabled("file_write", ctx):
        return {"error": "Tool disabled by runtime policy: file_write"}
    path = _resolve_workspace_path(_path_arg(args), workspace, ctx)
    if path is None:
        return {"error": "Access denied: outside workspace"}
    content = args.get("content", "")
    overwrite = bool(args.get("overwrite", False))
    expected_sha256 = args.get("expected_sha256")
    if path.exists() and path.is_file():
        current = path.read_text(errors="replace")
        current_hash = hashlib.sha256(current.encode()).hexdigest()
        if expected_sha256 and str(expected_sha256) != current_hash:
            return {
                "error": "Refusing to overwrite: expected_sha256 does not match current file",
                "path": str(path),
                "current_sha256": current_hash,
            }
        large_existing = path.stat().st_size > 8192 or len(current.splitlines()) > 200
        if large_existing and not overwrite and not expected_sha256:
            return {
                "error": (
                    "Refusing to overwrite existing large file without overwrite=true "
                    "or expected_sha256. Use file_replace for targeted edits."
                ),
                "path": str(path),
                "bytes": path.stat().st_size,
                "lines": len(current.splitlines()),
                "sha256": current_hash,
            }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return {"path": str(path), "bytes": len(content.encode()), "sha256": hashlib.sha256(content.encode()).hexdigest()}


async def tool_file_replace(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """Replace an exact text range in a file."""
    if not _tool_enabled("file_replace", ctx):
        return {"error": "Tool disabled by runtime policy: file_replace"}
    path = _resolve_workspace_path(_path_arg(args), workspace, ctx)
    if path is None:
        return {"error": "Access denied: outside workspace"}
    if not path.exists():
        return {"error": f"Not found: {path}"}
    if not path.is_file():
        return {"error": f"Not a file: {path}"}

    old = args.get("old", "")
    new = args.get("new", "")
    expected_sha256 = args.get("expected_sha256")
    count = args.get("count", 1)
    try:
        count = int(count or 1)
    except (TypeError, ValueError):
        return {"error": "count must be an integer"}
    if count <= 0:
        return {"error": "count must be positive"}
    if not old:
        return {"error": "old text is required"}

    content = path.read_text(errors="replace")
    current_hash = hashlib.sha256(content.encode()).hexdigest()
    if expected_sha256 and str(expected_sha256) != current_hash:
        return {
            "error": "Refusing to replace: expected_sha256 does not match current file",
            "path": str(path),
            "current_sha256": current_hash,
        }
    occurrences = content.count(old)
    if occurrences == 0:
        return {"error": "old text not found", "path": str(path), "sha256": current_hash}
    if occurrences > count and not args.get("allow_multiple", False):
        return {
            "error": "old text appears more times than count; refine old text or pass allow_multiple=true",
            "path": str(path),
            "occurrences": occurrences,
            "sha256": current_hash,
        }
    updated = content.replace(old, new, count)
    path.write_text(updated)
    return {
        "path": str(path),
        "replacements": min(occurrences, count),
        "bytes": len(updated.encode()),
        "sha256": hashlib.sha256(updated.encode()).hexdigest(),
    }


async def tool_file_list(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """List directory."""
    if not _tool_enabled("file_list", ctx):
        return {"error": "Tool disabled by runtime policy: file_list"}
    path = _resolve_workspace_path(_path_arg(args, "."), workspace, ctx)
    if path is None:
        return {"error": "Access denied: outside workspace"}
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
    if not _tool_enabled("grep", ctx):
        return {"error": "Tool disabled by runtime policy: grep"}
    pattern = args.get("pattern", "")
    path_obj = _resolve_workspace_path(_path_arg(args, "."), workspace, ctx)
    if path_obj is None:
        return {"error": "Access denied: outside workspace", "matches": []}
    path = str(path_obj)
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


async def tool_bt_aggregate(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """Aggregate pairwise winner/loser JSON records with a Bradley-Terry model."""
    if not _tool_enabled("bt_aggregate", ctx):
        return {"error": "Tool disabled by runtime policy: bt_aggregate"}

    paths_result = _bt_input_paths(args, workspace, ctx)
    if "error" in paths_result:
        return paths_result
    paths: list[Path] = paths_result["paths"]
    if not paths:
        return {"error": "No comparison files found"}

    records: list[dict[str, str]] = []
    loaded_files: list[str] = []
    for path in paths:
        if not path.exists():
            return {"error": f"Not found: {path}"}
        if not path.is_file():
            return {"error": f"Not a file: {path}"}
        try:
            data = json.loads(path.read_text(errors="replace"))
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON in {path}: {e}"}
        extracted = _extract_bt_records(data, str(path))
        if not extracted:
            return {"error": f"No winner/loser comparison records found in {path}"}
        records.extend(extracted)
        loaded_files.append(str(path))

    aggregate = _bradley_terry_aggregate(records)
    if "error" in aggregate:
        return aggregate

    ranking = aggregate["ranking"]
    retain_arg = args.get("retain")
    if retain_arg in (None, ""):
        retain = len(ranking)
    else:
        try:
            retain = int(retain_arg)
        except (TypeError, ValueError):
            return {"error": "retain must be an integer"}
        retain = max(0, min(retain, len(ranking)))

    result: dict[str, Any] = {
        "method": "bradley_terry_mm",
        "files": loaded_files,
        "comparison_count": len(records),
        "candidate_count": len(aggregate["candidates"]),
        "candidates": aggregate["candidates"],
        "ranking": ranking,
        "scores": aggregate["scores"],
        "strengths": aggregate["strengths"],
        "win_counts": aggregate["win_counts"],
        "loss_counts": aggregate["loss_counts"],
        "retained": ranking[:retain],
        "discarded": ranking[retain:],
        "elite": ranking[0] if ranking else None,
        "bottom": ranking[-1] if ranking else None,
        "iterations": aggregate["iterations"],
        "converged": aggregate["converged"],
    }

    output_path = args.get("output_path")
    if output_path:
        resolved = _resolve_workspace_path(output_path, workspace, ctx)
        if resolved is None:
            return {"error": "Access denied: outside workspace"}
        payload = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(payload)
        result["output_path"] = str(resolved)
        result["bytes"] = len(payload.encode())
        result["sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    return result


def _tool_enabled(name: str, ctx: "ToolContext") -> bool:
    return ctx.enabled_work_tools is None or name in ctx.enabled_work_tools


def _path_arg(args: dict[str, Any], default: str = "") -> Any:
    return args.get("path", args.get("file_path", args.get("filepath", default)))


def _resolve_workspace_path(raw_path: str | Path, workspace: Path, ctx: "ToolContext") -> Path | None:
    raw = str(raw_path or ".")
    workspace_root = ctx.workspace_root.resolve()
    shared_dir = ctx.shared_dir.resolve()
    workspace_dir = workspace.resolve()
    if raw == "$SHARED" or raw == "${SHARED}":
        path = shared_dir
    elif raw.startswith("$SHARED/"):
        path = shared_dir / raw[len("$SHARED/"):]
    elif raw.startswith("${SHARED}/"):
        path = shared_dir / raw[len("${SHARED}/"):]
    elif raw == "$WORKSPACE" or raw == "${WORKSPACE}":
        path = workspace_dir
    elif raw.startswith("$WORKSPACE/"):
        path = workspace_dir / raw[len("$WORKSPACE/"):]
    elif raw.startswith("${WORKSPACE}/"):
        path = workspace_dir / raw[len("${WORKSPACE}/"):]
    else:
        raw = os.path.expandvars(raw)
        path = Path(raw)
        if path.is_absolute() and path.parts and path.parts[1:2] == (workspace_root.name,):
            path = workspace_root.joinpath(*path.parts[2:])
        if not path.is_absolute() and path.parts and path.parts[0] == shared_dir.name:
            path = shared_dir.joinpath(*path.parts[1:])
        elif not path.is_absolute() and path.parts and path.parts[0] == workspace_root.name:
            path = workspace_root.joinpath(*path.parts[1:])
        elif not path.is_absolute():
            path = workspace_dir / path
    try:
        resolved = path.resolve()
        resolved.relative_to(workspace_root)
        return resolved
    except ValueError:
        return None


def _bt_input_paths(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    paths: list[Path] = []
    seen: set[Path] = set()

    explicit = args.get("comparisons") or args.get("files") or []
    if isinstance(explicit, str):
        explicit = [explicit]
    if not isinstance(explicit, list):
        return {"error": "comparisons/files must be a list of paths"}
    for raw_path in explicit:
        path = _resolve_workspace_path(raw_path, workspace, ctx)
        if path is None:
            return {"error": "Access denied: outside workspace"}
        if path not in seen:
            paths.append(path)
            seen.add(path)

    directory = args.get("directory")
    if directory:
        root = _resolve_workspace_path(directory, workspace, ctx)
        if root is None:
            return {"error": "Access denied: outside workspace"}
        if not root.exists():
            return {"error": f"Not found: {root}"}
        if not root.is_dir():
            return {"error": f"Not a directory: {root}"}
        pattern = str(args.get("pattern") or "*.json")
        recursive = bool(args.get("recursive", False))
        matched = root.rglob(pattern) if recursive else root.glob(pattern)
        for path in sorted(p for p in matched if p.is_file()):
            if path not in seen:
                paths.append(path)
                seen.add(path)

    return {"paths": paths}


def _extract_bt_records(data: Any, source: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if isinstance(data, list):
        for idx, item in enumerate(data):
            records.extend(_extract_bt_records(item, f"{source}[{idx}]"))
        return records

    if not isinstance(data, dict):
        return records

    if "winner" in data and "loser" in data:
        winner = str(data["winner"]).strip()
        loser = str(data["loser"]).strip()
        if winner and loser and winner != loser:
            records.append({"winner": winner, "loser": loser, "source": source})

    for key in ("pairwise_results", "comparisons", "results", "judgments"):
        nested = data.get(key)
        if isinstance(nested, dict):
            for nested_key, item in nested.items():
                records.extend(_extract_bt_records(item, f"{source}:{nested_key}"))
        elif isinstance(nested, list):
            for idx, item in enumerate(nested):
                records.extend(_extract_bt_records(item, f"{source}:{key}[{idx}]"))

    return records


def _bradley_terry_aggregate(records: list[dict[str, str]]) -> dict[str, Any]:
    candidates = sorted({record["winner"] for record in records} | {record["loser"] for record in records})
    if len(candidates) < 2:
        return {"error": "At least two candidates are required"}

    index = {candidate: idx for idx, candidate in enumerate(candidates)}
    n = len(candidates)
    wins = [[0.0 for _ in range(n)] for _ in range(n)]
    win_counts = {candidate: 0 for candidate in candidates}
    loss_counts = {candidate: 0 for candidate in candidates}
    for record in records:
        winner = record["winner"]
        loser = record["loser"]
        if winner not in index or loser not in index or winner == loser:
            continue
        wins[index[winner]][index[loser]] += 1.0
        win_counts[winner] += 1
        loss_counts[loser] += 1

    total_wins = [sum(row) for row in wins]
    strengths = [1.0 for _ in range(n)]
    converged = False
    iterations = 0
    floor = 1e-12

    for iterations in range(1, 1001):
        updated: list[float] = []
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if i == j:
                    continue
                comparisons = wins[i][j] + wins[j][i]
                if comparisons:
                    denom += comparisons / max(strengths[i] + strengths[j], floor)
            value = floor if total_wins[i] <= 0 or denom <= 0 else total_wins[i] / denom
            updated.append(max(value, floor))

        scale = sum(updated) / n
        if scale > 0:
            updated = [value / scale for value in updated]
        delta = max(abs(updated[i] - strengths[i]) for i in range(n))
        strengths = updated
        if delta < 1e-9:
            converged = True
            break

    total_strength = sum(strengths) or 1.0
    scores = {candidate: strengths[index[candidate]] / total_strength for candidate in candidates}
    raw_strengths = {candidate: strengths[index[candidate]] for candidate in candidates}
    ranking = sorted(
        candidates,
        key=lambda candidate: (
            -scores[candidate],
            -win_counts[candidate],
            loss_counts[candidate],
            candidate,
        ),
    )
    return {
        "candidates": candidates,
        "ranking": ranking,
        "scores": scores,
        "strengths": raw_strengths,
        "win_counts": win_counts,
        "loss_counts": loss_counts,
        "iterations": iterations,
        "converged": converged,
    }


# --- Registry ---

WORK_TOOLS: dict[str, dict[str, Any]] = {
    "shell": {
        "handler": tool_shell,
        "schema": {"type": "function", "function": {
            "name": "shell",
            "description": "Execute a shell command in your workspace directory. Returns {stdout, stderr, exit_code}. CWD is your private workspace. Use $SHARED to reference the shared directory. Long outputs are truncated and saved to a file.",
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
            "description": "Read a file's content. Relative paths resolve from your workspace. You can also read from shared/ or other agents' workspaces under the workspace root. Use offset/start_line and limit/lines for 1-based line-range reads after grep; limit<=40 is usually enough for source inspection. Returns content, size, truncated, and line metadata.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "File path (relative to workspace, or absolute under workspace root)"},
                "offset": {"type": "integer", "description": "Optional 1-based start line for a line-range read"},
                "start_line": {"type": "integer", "description": "Alias for offset"},
                "limit": {"type": "integer", "description": "Optional maximum number of lines to read"},
                "lines": {"type": "integer", "description": "Alias for limit"},
            }, "required": ["path"]},
        }},
    },
    "file_write": {
        "handler": tool_file_write,
        "schema": {"type": "function", "function": {
            "name": "file_write",
            "description": "Write content to a file. Existing large files require overwrite=true or expected_sha256; prefer file_replace for source edits. Parent directories are created automatically. Write to shared/ to make files visible to other agents.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "File path (relative to workspace). Use 'shared/filename' to write to the shared directory."},
                "content": {"type": "string", "description": "File content to write"},
                "overwrite": {"type": "boolean", "description": "Required to overwrite existing large files without expected_sha256", "default": False},
                "expected_sha256": {"type": "string", "description": "Optional current file hash guard for safe overwrite"},
            }, "required": ["path", "content"]},
        }},
    },
    "file_replace": {
        "handler": tool_file_replace,
        "schema": {"type": "function", "function": {
            "name": "file_replace",
            "description": "Replace exact text in an existing file. Safer than whole-file writes for source edits. Use a distinctive old text block and optionally expected_sha256 from file_read/file_write output.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "File path (relative to workspace, shared/, $SHARED, or absolute under workspace root)"},
                "old": {"type": "string", "description": "Exact text to replace"},
                "new": {"type": "string", "description": "Replacement text"},
                "count": {"type": "integer", "description": "Maximum replacements", "default": 1},
                "allow_multiple": {"type": "boolean", "description": "Allow replacement when old text appears more often than count", "default": False},
                "expected_sha256": {"type": "string", "description": "Optional current file hash guard"},
            }, "required": ["path", "old", "new"]},
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
    "bt_aggregate": {
        "handler": tool_bt_aggregate,
        "schema": {"type": "function", "function": {
            "name": "bt_aggregate",
            "description": "Aggregate pairwise comparison JSON files with a Bradley-Terry model. Provide directory for the comparison JSON directory; use pattern/recursive to control discovery. Inputs may be files containing {winner, loser}, a list of such records, or nested pairwise_results/comparisons/results/judgments. Use this after parallel judge agents produce winner/loser decisions.",
            "parameters": {"type": "object", "properties": {
                "comparisons": {"type": "array", "items": {"type": "string"}, "description": "Optional explicit JSON comparison file paths"},
                "files": {"type": "array", "items": {"type": "string"}, "description": "Alias for comparisons"},
                "directory": {"type": "string", "description": "Optional directory of JSON comparison files"},
                "pattern": {"type": "string", "description": "Glob pattern when directory is used", "default": "*.json"},
                "recursive": {"type": "boolean", "description": "Recursively search directory for pattern", "default": False},
                "retain": {"type": "integer", "description": "Number of top-ranked candidates to keep in retained"},
                "output_path": {"type": "string", "description": "Optional path to write aggregate JSON, usually under shared/"},
            }},
        }},
    },
}
