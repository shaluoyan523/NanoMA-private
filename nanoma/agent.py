"""General-purpose NanoMA agent API.

This module is benchmark-agnostic. It turns the runtime into a small reusable
agent interface for research, coding, analysis, and artifact-producing tasks.
Benchmark integrations may provide extra tools, but none are required here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from nanoma.core import Runtime, RuntimeConfig


EventHandler = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class AgentRun:
    """Result and inspectable runtime state from one general-agent run."""

    result: str | None
    stats: dict[str, Any]
    workspace_root: Path
    shared_dir: Path
    log_dir: Path | None
    runtime: Runtime

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-safe public portion of the run."""
        return {
            "result": self.result,
            "stats": self.stats,
            "workspace_root": str(self.workspace_root),
            "shared_dir": str(self.shared_dir),
            "log_dir": str(self.log_dir) if self.log_dir else None,
        }


def prepare_general_config(
    config: RuntimeConfig | None = None,
    *,
    project_dir: str | Path | None = None,
    instructions: str = "",
) -> RuntimeConfig:
    """Return an isolated config for a normal, non-benchmark task.

    ``project_dir`` is optional. When supplied, agents are told that it is the
    user's target directory and structured workspace tools are allowed to read
    it. The internal agent workspaces and logs remain separate.
    """
    base = config or RuntimeConfig()
    workspace_root = Path(base.workspace_root).expanduser().resolve()
    log_dir = Path(base.log_dir).expanduser().resolve() if base.log_dir else None
    extra_roots = [Path(path).expanduser().resolve() for path in base.workspace_extra_roots]
    extra_sections = [str(base.system_extra_instructions or "").strip()]

    if project_dir is not None:
        project = Path(project_dir).expanduser().resolve()
        if not project.is_dir():
            raise ValueError(f"project directory does not exist: {project}")
        if project not in extra_roots:
            extra_roots.append(project)
        extra_sections.append(
            "User project directory: " + str(project) + "\n"
            "This is the target directory supplied by the user. Inspect it before "
            "changing it, keep edits scoped to the task, and run relevant checks there. "
            "Your private NanoMA workspace is for scratch work; final project edits belong "
            "in this target directory."
        )

    if instructions.strip():
        extra_sections.append(instructions.strip())

    return replace(
        base,
        workspace_root=workspace_root,
        log_dir=log_dir,
        workspace_extra_roots=extra_roots,
        system_extra_instructions="\n\n".join(section for section in extra_sections if section),
    )


async def run_agent(
    task: str,
    *,
    config: RuntimeConfig | None = None,
    model: str | None = None,
    project_dir: str | Path | None = None,
    instructions: str = "",
    on_event: EventHandler | None = None,
) -> AgentRun:
    """Run NanoMA as a general-purpose agent.

    Every node uses its own working model for planning decisions. When a node
    chooses to delegate, its children inherit that same model. Direct spawn
    primitives remain internal to the runtime.
    """
    task = str(task or "").strip()
    if not task:
        raise ValueError("task must not be empty")

    prepared = prepare_general_config(
        config,
        project_dir=project_dir,
        instructions=instructions,
    )
    selected_model = str(model or prepared.default_model).strip()
    if not selected_model:
        raise ValueError("model must not be empty")
    if selected_model != prepared.default_model:
        prepared = replace(prepared, default_model=selected_model)

    runtime = Runtime(config=prepared, on_event=on_event)
    result = await runtime.run(task, model=selected_model)
    return AgentRun(
        result=result,
        stats=runtime.stats(),
        workspace_root=prepared.workspace_root,
        shared_dir=prepared.workspace_root / prepared.shared_dir,
        log_dir=prepared.log_dir,
        runtime=runtime,
    )


__all__ = ["AgentRun", "EventHandler", "prepare_general_config", "run_agent"]
