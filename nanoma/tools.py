"""Work tools: shell.

The shell tool is NanoMA's universal primitive — the "escape hatch" that covers
any operation not handled by the 9 structured workspace tools or 12 meta tools.

Design principle: everything that CAN be done via a one-liner shell command
SHOULD use shell. Dedicated tools exist only when shell cannot be reliable.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from nanoma.core import ToolContext


WEB_BLOCK_OUTPUT_PATTERNS = [
    r"Your Request Originates from an Undeclared Automated Tool",
    r"Access Denied",
    r"temporarily blocked",
    r"request has been identified as part of a network of automated tools",
]


_PKILL_COMMAND_RE = re.compile(
    r"(?P<exe>(?<![\w.-])(?:sudo\s+)?(?:(?:/usr)?/bin/)?pkill)"
    r"(?P<args>[^;&|\n]*)",
    flags=re.IGNORECASE,
)


def _guard_process_control_command(cmd: str) -> tuple[str, str | None]:
    """Keep task cleanup commands from terminating the NanoMA runtime.

    `pkill -f` matches full command lines.  EdgeBench task paths are present in
    the NanoMA runner's own argv, so cleanup such as `pkill -f dcss` can kill the
    runner and its parent before either one records an exit.  procps' `-A`
    option excludes every ancestor of pkill while preserving its intended task
    cleanup behavior.

    Direct `kill` is guarded at execution time in `sandbox.shell_exec`, where the
    actual ancestor PIDs are known.  Bypasses that cannot be wrapped safely are
    rejected here.
    """

    bypass = re.search(
        r"(?<![\w.-])(?:sudo\s+)(?:(?:/usr)?/bin/)?kill(?!all)\b"
        r"|(?<![\w.-])(?:(?:/usr)?/bin/)kill(?:all)?\b"
        r"|(?<![\w.-])(?:sudo\s+)?(?:(?:/usr)?/bin/)?killall\b",
        cmd,
        flags=re.IGNORECASE,
    )
    if bypass:
        return cmd, (
            "Broad or privileged process termination is blocked because it can "
            "kill the NanoMA runtime. Use plain `kill <explicit child pid>` or "
            "`pkill -f <task-specific pattern>`; those forms protect runtime "
            "ancestor processes automatically."
        )

    def protect(match: re.Match[str]) -> str:
        args = match.group("args")
        if re.search(r"(?:^|\s)--ignore-ancestors(?:\s|$)|(?:^|\s)-[^\s]*A", args):
            return match.group(0)
        return f"{match.group('exe')} --ignore-ancestors{args}"

    return _PKILL_COMMAND_RE.sub(protect, cmd), None


def _command_access_is_limited_to_runtime_workspace(cmd: str, workspace: Path, shared_dir: Path) -> bool:
    """Allow benchmark run paths only when they are this task's own workspace."""
    workspace = workspace.resolve()
    shared_dir = shared_dir.resolve()
    workspace_root = shared_dir.parent.resolve()
    task_root = workspace_root.parent.resolve()
    text = (
        str(cmd)
        .replace(str(workspace), "$WORKSPACE")
        .replace(str(shared_dir), "$SHARED")
        .replace(str(workspace_root), "$TASK_WORKSPACE")
        .replace(str(task_root), "$TASK_ROOT")
    )
    protected_run_roots = (
        "/data_storage/nanoma/runs",
        "/data/workspace/NanoMA/runs",
    )
    return not any(root in text for root in protected_run_roots)


def _clean_shell_command(cmd: str, workspace: Path | None = None, shared_dir: Path | None = None) -> str:
    cmd = str(cmd).strip()
    cmd = re.sub(r"^```(?:bash|sh|shell)?\s*", "", cmd, flags=re.IGNORECASE)
    cmd = re.sub(r"\s*```\s*$", "", cmd).strip()
    cmd = re.sub(r"(?m)^\s*```[A-Za-z0-9_-]*\s*$", "", cmd)
    cmd = re.sub(r"(?is)^\s*<parameter(?:\s+[^>]*)?>\s*(?:command\s*>\s*)?", "", cmd)
    cmd = re.sub(r"(?is)\s*</parameter>\s*$", "", cmd).strip()
    cmd = re.sub(r"(?im)^\s*<parameter(?:\s+[^>]*)?>\s*(?:command\s*>)?\s*$", "", cmd)
    cmd = re.sub(r"(?im)^\s*</parameter>\s*$", "", cmd)
    cmd = re.sub(r"</?(?:command|command_sequence|tool_call)[^>]*>", "", cmd, flags=re.IGNORECASE)
    cleaned_lines = []
    for line in cmd.splitlines():
        stripped = line.strip()
        if stripped.lower() in {"shellrun", "shell run", "run shell", "shell", "command", "command_sequence"}:
            continue
        if (
            re.search(r"\bmkdir\s+-p\s+/data_storage/.*/runs/widesearch", stripped)
            and (
                workspace is None
                or shared_dir is None
                or not _command_access_is_limited_to_runtime_workspace(stripped, workspace, shared_dir)
            )
        ):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _looks_like_web_block_output(result: dict[str, Any]) -> str | None:
    text = "\n".join(str(result.get(key, "")) for key in ("stdout", "stderr"))
    for pattern in WEB_BLOCK_OUTPUT_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return pattern
    return None


async def tool_shell(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """Execute a shell command in the agent's workspace directory.

    Environment variables:
        WORKSPACE — agent's private workspace path
        SHARED    — shared directory visible to all agents
    """
    from nanoma.sandbox import shell_exec
    from nanoma.core import classify_shell_capability
    from nanoma import shell_memory

    cmd = args.get("command", "")
    timeout = args.get("timeout")
    max_output = ctx.shell_max_output

    if not cmd or not cmd.strip():
        return {"error": "command is required"}
    cmd = _clean_shell_command(cmd, workspace=workspace, shared_dir=ctx.shared_dir)
    original_cmd = cmd
    cmd, process_control_block = _guard_process_control_command(cmd)
    if process_control_block:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": process_control_block,
            "blocked": True,
            "blocked_capability": "process_control",
            "instruction": process_control_block,
        }
    process_guard_applied = cmd != original_cmd

    try:
        requested_timeout = int(timeout) if timeout is not None else int(getattr(ctx, "shell_max_timeout", 30) or 30)
    except (TypeError, ValueError):
        requested_timeout = int(getattr(ctx, "shell_max_timeout", 30) or 30)
    requested_timeout = max(1, requested_timeout)
    timeout = requested_timeout
    max_timeout = int(getattr(ctx, "shell_max_timeout", 30) or 0)
    if max_timeout > 0 and timeout > max_timeout:
        timeout = max_timeout

    capability = classify_shell_capability(cmd)
    allowed = getattr(ctx, "allowed_shell_capabilities", None)
    if allowed and capability not in allowed:
        instruction = (
            "This shell command was blocked because that shell capability is no longer "
            "available under the current runtime constraints. Do not retry the same "
            "blocked capability. Use local files, prior evidence, and available "
            "tools to produce the required deliverable, then call set_status(done, "
            "result=...)."
        )
        if capability == "web":
            instruction = (
                "Additional web/network shell commands are closed for this task. "
                "Do not retry curl, wget, browser/search, or urllib/requests commands. "
                "Use evidence already collected plus local shell/file tools to write "
                "the final deliverable, then call set_status(done, result=...)."
            )
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Blocked shell capability by constraint: {capability}",
            "blocked": True,
            "blocked_capability": capability,
            "allowed_shell_capabilities": sorted(allowed),
            "instruction": instruction,
        }

    for pattern in ctx.blocked_shell_patterns:
        if re.search(pattern, cmd, flags=re.IGNORECASE) and not _command_access_is_limited_to_runtime_workspace(
            cmd,
            workspace,
            ctx.shared_dir,
        ):
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Blocked shell command by policy pattern: {pattern}",
                "blocked": True,
                "pattern": pattern,
                "blocked_capability": capability if capability == "web" else None,
            }

    # Memory is claimed here rather than assumed at spawn time. The per-child cap
    # budgets agent processes; what exhausts a container is what those agents run.
    memory = shell_memory.arbiter()
    ticket = await memory.acquire(cmd, timeout=float(timeout))
    if not ticket["granted"]:
        wanted_mb = (ticket.get("wanted") or 0) / (1024 * 1024)
        held_mb = (ticket.get("held") or 0) / (1024 * 1024)
        budget_mb = (ticket.get("budget") or 0) / (1024 * 1024)
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": (
                f"Not enough memory to start this command within {timeout}s: it has "
                f"needed about {wanted_mb:.0f}MB, {held_mb:.0f}MB of the {budget_mb:.0f}MB "
                "budget is held by commands already running."
            ),
            "memory_deferred": True,
            "instruction": (
                "This was not started, so nothing has changed and nothing was measured. "
                "It is queued behind other work in this container, not broken. Do "
                "something that does not need this command, or wait and run it again. "
                "Do not shrink what you are measuring to make it fit."
            ),
        }
    result: dict[str, Any] = {}
    try:
        result = await shell_exec(cmd, workspace, ctx.shared_dir, timeout)
    finally:
        await memory.release(cmd, ticket, result.get("peak_rss_bytes"))
    if ticket.get("waited", 0) >= 1.0:
        result["memory_waited_seconds"] = round(ticket["waited"], 1)
    if process_guard_applied:
        result["process_guard_applied"] = True

    if timeout < requested_timeout:
        # Silently clamping taught agents the wrong lesson. One asked for 620s,
        # wrapping its command in `timeout 600`, was killed at 300 and told only
        # "Timeout after 300s" — a limit it could not see and so could not plan
        # around. Saying so lets it split the work or run it in the background
        # instead of shrinking what it was measuring until the answer fit.
        result["timeout_limit"] = timeout
        result["timeout_requested"] = requested_timeout
        if result.get("timed_out"):
            result["instruction"] = (
                f"This ran under the {timeout}s ceiling on a single shell call, not "
                f"the {requested_timeout}s you asked for, and any output above is "
                "what it printed before being stopped. For work that legitimately "
                "takes longer, start it in the background and poll it "
                "(`nohup … > out.log 2>&1 &`, then read out.log), or run it in "
                "pieces. Do not reduce what you are measuring in order to fit."
            )

    if matched_block := _looks_like_web_block_output(result):
        result["blocked"] = True
        result["blocked_capability"] = "web"
        result["pattern"] = matched_block
        result["instruction"] = (
            "The retrieved web content is an access-block or anti-automation page. "
            "Do not retry the same host with different curl headers. Use Serper/search "
            "snippets, accessible investor-relations mirrors, cached local evidence, "
            "or the evidence already collected, then produce the final deliverable."
        )

    # Truncate large outputs and spill to file
    for key in ("stdout", "stderr"):
        val = result.get(key, "")
        if max_output > 0 and len(val) > max_output:
            # Use uuid for unique, collision-free file naming
            out_dir = workspace / ".cmd-output"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{uuid.uuid4().hex[:8]}_{key}.txt"
            out_file.write_text(val)
            result[key] = val[:max_output] + f"\n...(truncated {len(val)} chars, full: {out_file.name})"
            result[f"{key}_file"] = str(out_file.relative_to(workspace))

    return result


# --- Registry ---

WORK_TOOLS: dict[str, dict[str, Any]] = {
    "shell": {
        "handler": tool_shell,
        "schema": {"type": "function", "function": {
            "name": "shell",
            "description": "Execute a shell command in your workspace directory. Returns {stdout, stderr, exit_code}. CWD is your private workspace. $SHARED references the shared directory. Long outputs are truncated and saved to a file. Use shell for: mkdir -p, rm -rf, mv, ls, find, tree, git, pip, curl, and any other system command.",
            "parameters": {"type": "object", "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {"type": "integer", "description": "Max execution time in seconds (default: 30)", "default": 30},
            }, "required": ["command"]},
        }},
    },
}
