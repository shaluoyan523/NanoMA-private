from __future__ import annotations

import json
import os
import re
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


SOURCE_ROOT = Path(os.environ["NANOMA_SOURCE_ROOT"]).resolve()
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from nanoma import Runtime, RuntimeConfig  # noqa: E402
from nanoma.meta import META_TOOLS  # noqa: E402
from nanoma.plugins.workspace_tools import WORKSPACE_TOOLS  # noqa: E402
from nanoma.tools import WORK_TOOLS  # noqa: E402


_REPOZERO_RUNTIME_PROFILE = {
    "budget": 1000.0,
    "time_limit": 7200.0,
    "max_turns": 1000,
    "max_agents": sys.maxsize,
    "max_depth": sys.maxsize,
    "max_concurrent_llm": 16,
    "max_parallel_children": None,
    "max_total_tokens": 0,
    "llm_admission_control": False,
    "llm_min_start_spacing": 0.0,
    "llm_large_context_tokens": 32000,
    "llm_large_context_spacing": 0.0,
    "llm_overload_cooldown_seconds": 0.0,
    "llm_admission_max_delay": 120.0,
    "tool_policy_mode": "off",
    "tool_policy_delivery_enabled": False,
    "tool_policy_delivery_prepare_enabled": False,
    "tool_policy_delivery_verify_enabled": False,
    "spawn_judge_enabled": True,
}

# Preserve NanoMA's node planning, lifecycle, and message passing. Host-side
# work, generic verification, experiment, and submit tools must not leak into
# an official Terminal-Bench sandbox.
_NANOMA_AGENT_TOOLS = frozenset({
    "task_create",
    "task_list",
    "task_update",
    "query",
    "send",
    "wait",
    "kill",
    "deliver_to_parent",
    "deliveries",
    "set_bio",
    "get_task_context",
    "get_cost",
    "set_status",
})

_SPAWN_JUDGE_INSTRUCTION = (
    "Planning authority belongs to every node, not only the root. Approve spawning for "
    "genuinely separable implementation components, materially different hypotheses, or "
    "independent verification. A child may spawn again when its own assignment separates. "
    "At a root node's first nontrivial planning boundary prefer two or three distinct "
    "children; at child nodes create only genuinely distinct work. Never reject only because "
    "an ancestor already delegated."
)

_NANOMA_PLANNING_PROTOCOL = """
NanoMA planning and delivery protocol:
- Every node owns its next planning decision. At the first meaningful planning boundary, call
  task_create before doing substantive work whenever implementation, edge-case analysis, or
  verification can be separated. This applies equally to root and child nodes.
- Children may call task_create and recursively delegate distinct subproblems or independent
  audits. Avoid duplicate branches, keep assignments bounded, and reconcile concrete evidence.
- After a candidate exists, open a fresh planning boundary for independent review when capacity
  remains. Do not treat a single self-review as equivalent to independent verification.
- All agents share the same official benchmark environment through the tb_* tools. A child that
  changes the benchmark environment must report the exact change and evidence to its parent.
- Only non-root children call deliver_to_parent. The root completes with set_status(done) only
  after validating the benchmark deliverable in the official environment.
"""


_BLOCKED = (
    re.compile(r"(^|[\s'\";])/?tests(/|[\s'\";]|$)", re.I),
    re.compile(r"/logs/verifier", re.I),
    re.compile(r"reward\.txt|ctrf\.json|solution/solve\.sh", re.I),
)


def _blocked(text: str) -> str | None:
    for pattern in _BLOCKED:
        if pattern.search(text):
            return pattern.pattern
    return None


def _disabled_nanoma_tools() -> set[str]:
    """Remove non-benchmark work tools while preserving agent control tools."""

    builtins = set(WORK_TOOLS) | set(WORKSPACE_TOOLS) | set(META_TOOLS)
    return builtins - set(_NANOMA_AGENT_TOOLS)


class NanoMAHarborAgent(BaseAgent):
    """Run NanoMA as an external Harbor agent against the official task sandbox."""

    SUPPORTS_ATIF = False

    @staticmethod
    def name() -> str:
        return "nanoma"

    def version(self) -> str | None:
        return "0.9.2-harbor-repozero-profile.1"

    async def setup(self, environment: BaseEnvironment) -> None:
        return

    def _runtime_config(
        self,
        *,
        model: str,
        workspace: Path,
        events: Path,
        environment: BaseEnvironment,
    ) -> RuntimeConfig:
        """Build the RepoZero-equivalent runtime around benchmark-native tools."""

        return RuntimeConfig(
            budget=_REPOZERO_RUNTIME_PROFILE["budget"],
            time_limit=_REPOZERO_RUNTIME_PROFILE["time_limit"],
            max_turns=_REPOZERO_RUNTIME_PROFILE["max_turns"],
            max_agents=_REPOZERO_RUNTIME_PROFILE["max_agents"],
            max_depth=_REPOZERO_RUNTIME_PROFILE["max_depth"],
            max_concurrent_llm=_REPOZERO_RUNTIME_PROFILE["max_concurrent_llm"],
            max_parallel_children=_REPOZERO_RUNTIME_PROFILE["max_parallel_children"],
            max_total_tokens=_REPOZERO_RUNTIME_PROFILE["max_total_tokens"],
            llm_admission_control=_REPOZERO_RUNTIME_PROFILE["llm_admission_control"],
            llm_min_start_spacing=_REPOZERO_RUNTIME_PROFILE["llm_min_start_spacing"],
            llm_large_context_tokens=_REPOZERO_RUNTIME_PROFILE["llm_large_context_tokens"],
            llm_large_context_spacing=_REPOZERO_RUNTIME_PROFILE["llm_large_context_spacing"],
            llm_overload_cooldown_seconds=_REPOZERO_RUNTIME_PROFILE["llm_overload_cooldown_seconds"],
            llm_admission_max_delay=_REPOZERO_RUNTIME_PROFILE["llm_admission_max_delay"],
            allowed_models=[model],
            default_model=model,
            workspace_root=workspace.resolve(),
            log_dir=events.resolve(),
            disabled_tools=_disabled_nanoma_tools(),
            shell_max_timeout=60,
            shell_max_output=40000,
            file_read_max_chars=200000,
            file_list_max_entries=5000,
            grep_max_results=500,
            tool_policy_mode=_REPOZERO_RUNTIME_PROFILE["tool_policy_mode"],
            tool_policy_delivery_enabled=_REPOZERO_RUNTIME_PROFILE["tool_policy_delivery_enabled"],
            tool_policy_delivery_prepare_enabled=_REPOZERO_RUNTIME_PROFILE["tool_policy_delivery_prepare_enabled"],
            tool_policy_delivery_verify_enabled=_REPOZERO_RUNTIME_PROFILE["tool_policy_delivery_verify_enabled"],
            spawn_judge_enabled=_REPOZERO_RUNTIME_PROFILE["spawn_judge_enabled"],
            spawn_judge_instruction=_SPAWN_JUDGE_INSTRUCTION,
            system_extra_instructions=(
                "You are solving a Terminal-Bench task inside Harbor's official task environment.\n"
                "- Use tb_shell, tb_read_file, and tb_write_file for all task inspection and edits.\n"
                "- The benchmark working directory and required output artifacts are under /app.\n"
                "- Your private NanoMA workspace is scratch space only; edits there are not graded.\n"
                "- Do not use the host shell for benchmark work.\n"
                "- Do not inspect /tests, verifier logs, reward files, hidden tests, or solution scripts.\n"
                "- No peer exists unless task_create or query returns its exact NanoMA ID. Never invent agent IDs.\n"
                "- Use deliver_to_parent only when you are a non-root child. Root results are graded from /app.\n"
                "- Complete and validate the requested artifacts in /app before calling set_status(status=\"done\").\n"
            ),
            extra_tools=self._tools(environment),
        )

    def _tools(self, environment: BaseEnvironment) -> dict[str, dict[str, Any]]:
        async def tb_shell(args: dict[str, Any], workspace: Path, ctx: Any) -> dict[str, Any]:
            command = str(args.get("command") or "")
            if not command.strip():
                return {"error": "command is required"}
            if pattern := _blocked(command):
                return {
                    "return_code": -1,
                    "stdout": "",
                    "stderr": f"Blocked verifier/solution access: {pattern}",
                    "blocked": True,
                }
            timeout = max(1, min(int(args.get("timeout") or 60), 1800))
            result = await environment.exec(command=command, cwd="/app", timeout_sec=timeout)
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            max_output = 50000
            if len(stdout) > max_output:
                stdout = stdout[:max_output] + f"\n...(truncated {len(stdout)} chars)"
            if len(stderr) > max_output:
                stderr = stderr[:max_output] + f"\n...(truncated {len(stderr)} chars)"
            return {"return_code": result.return_code, "stdout": stdout, "stderr": stderr}

        async def tb_read_file(args: dict[str, Any], workspace: Path, ctx: Any) -> dict[str, Any]:
            raw_path = str(args.get("path") or "")
            if not raw_path:
                return {"error": "path is required"}
            path = raw_path if raw_path.startswith("/") else f"/app/{raw_path}"
            if pattern := _blocked(path):
                return {"error": f"Blocked verifier/solution access: {pattern}", "blocked": True}
            max_bytes = max(1, min(int(args.get("max_bytes") or 200000), 500000))
            command = (
                f"test -f {shlex.quote(path)} && "
                f"wc -c < {shlex.quote(path)} && head -c {max_bytes} {shlex.quote(path)}"
            )
            result = await environment.exec(command=command, cwd="/app", timeout_sec=60)
            if result.return_code != 0:
                return {"error": f"not found or unreadable: {path}", "stderr": result.stderr or ""}
            stdout = result.stdout or ""
            first, _, content = stdout.partition("\n")
            try:
                size = int(first.strip())
            except ValueError:
                size = len(content.encode("utf-8", errors="replace"))
            return {"path": path, "bytes": size, "truncated": size > max_bytes, "content": content}

        async def tb_write_file(args: dict[str, Any], workspace: Path, ctx: Any) -> dict[str, Any]:
            raw_path = str(args.get("path") or "")
            if not raw_path:
                return {"error": "path is required"}
            path = raw_path if raw_path.startswith("/") else f"/app/{raw_path}"
            if pattern := _blocked(path):
                return {"error": f"Blocked verifier/solution access: {pattern}", "blocked": True}
            content = str(args.get("content") or "")
            parent = str(Path(path).parent)
            mkdir = await environment.exec(
                command=f"mkdir -p {shlex.quote(parent)}", cwd="/app", timeout_sec=30
            )
            if mkdir.return_code != 0:
                return {"error": mkdir.stderr or mkdir.stdout or "failed to create parent directory"}
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
                handle.write(content)
                source = Path(handle.name)
            try:
                await environment.upload_file(source, path)
            finally:
                source.unlink(missing_ok=True)
            chmod = await environment.exec(
                command=f"chmod 0644 {shlex.quote(path)}", cwd="/app", timeout_sec=30
            )
            if chmod.return_code != 0:
                return {"error": chmod.stderr or chmod.stdout or "failed to set artifact permissions"}
            return {"path": path, "bytes_written": len(content.encode("utf-8"))}

        return {
            "tb_shell": {
                "handler": tb_shell,
                "schema": {"type": "function", "function": {
                    "name": "tb_shell",
                    "description": "Execute a shell command in the official Terminal-Bench task environment. The working directory is /app.",
                    "parameters": {"type": "object", "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer", "default": 60},
                    }, "required": ["command"]},
                }},
            },
            "tb_read_file": {
                "handler": tb_read_file,
                "schema": {"type": "function", "function": {
                    "name": "tb_read_file",
                    "description": "Read a file from the official task environment. Relative paths resolve under /app.",
                    "parameters": {"type": "object", "properties": {
                        "path": {"type": "string"},
                        "max_bytes": {"type": "integer", "default": 200000},
                    }, "required": ["path"]},
                }},
            },
            "tb_write_file": {
                "handler": tb_write_file,
                "schema": {"type": "function", "function": {
                    "name": "tb_write_file",
                    "description": "Write a UTF-8 file into the official task environment. Relative paths resolve under /app.",
                    "parameters": {"type": "object", "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    }, "required": ["path", "content"]},
                }},
            },
        }

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        model = (self.model_name or os.environ.get("NANOMA_MODEL") or "gpt-5.6-luna").split("/", 1)[-1]
        run_root = self.logs_dir / "nanoma"
        workspace = run_root / "workspace"
        events = run_root / "events"
        workspace.mkdir(parents=True, exist_ok=True)
        events.mkdir(parents=True, exist_ok=True)

        cfg = self._runtime_config(
            model=model,
            workspace=workspace,
            events=events,
            environment=environment,
        )
        runtime = Runtime(config=cfg)
        result = ""
        error: str | None = None
        try:
            result = await runtime.run(
                instruction.rstrip() + "\n\n" + _NANOMA_PLANNING_PROTOCOL.strip() + "\n",
                model=model,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            stats = runtime.stats()
            payload = {
                "model": model,
                "result": result,
                "error": error,
                "stats": stats,
                "runtime_profile": _REPOZERO_RUNTIME_PROFILE,
                "agent_tools": sorted(_NANOMA_AGENT_TOOLS),
                "benchmark_tools": sorted(self._tools(environment)),
                "disabled_nanoma_tools": sorted(_disabled_nanoma_tools()),
            }
            (run_root / "summary.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            llm_events = [e for e in getattr(runtime, "_events", []) if e.get("event") == "llm_done"]
            context.n_input_tokens = sum(int(e.get("data", {}).get("input_tokens") or 0) for e in llm_events)
            context.n_cache_tokens = sum(int(e.get("data", {}).get("cached") or 0) for e in llm_events)
            context.n_output_tokens = sum(int(e.get("data", {}).get("output_tokens") or 0) for e in llm_events)
            overview = stats.get("overview", {}) if isinstance(stats, dict) else {}
            context.cost_usd = float(overview.get("total_cost_usd_with_supervisor") or overview.get("total_cost_usd") or 0)
            context.metadata = {
                "nanoma_result": result,
                "nanoma_error": error,
                "nanoma_stats": stats,
                "nanoma_source": str(SOURCE_ROOT),
                "nanoma_runtime_profile": _REPOZERO_RUNTIME_PROFILE,
                "nanoma_agent_tools": sorted(_NANOMA_AGENT_TOOLS),
                "nanoma_benchmark_tools": sorted(self._tools(environment)),
            }
