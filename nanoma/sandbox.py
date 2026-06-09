"""Shell execution through Codex-provided sandboxing."""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


SandboxBackend = Literal["codex", "host"]


@dataclass
class SandboxConfig:
    backend: SandboxBackend = "codex"
    codex_bin: str = "codex"
    network: bool = False


async def _communicate(
    argv: list[str],
    *,
    timeout: int,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd) if cwd else None,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return {
            "exit_code": proc.returncode,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        return {"exit_code": -1, "stdout": "", "stderr": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"exit_code": -1, "stdout": "", "stderr": str(e)}


async def shell_exec(
    cmd: str,
    workspace: Path,
    shared_dir: Path,
    timeout: int = 30,
) -> dict[str, Any]:
    """Execute a shell command directly on the host.

    This is intentionally kept only for explicit host mode and focused tests.
    Runtime-created agents default to CodexSandboxSession instead.
    """
    env = _agent_env(workspace, shared_dir, inherit_host=True)
    return await _communicate(["/bin/sh", "-lc", cmd], timeout=timeout, cwd=workspace, env=env)


class SandboxSession:
    def __init__(self, config: SandboxConfig, workspace_root: Path):
        self.config = config
        self.workspace_root = workspace_root.resolve()
        self.backend: SandboxBackend | None = None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self.backend = self.config.backend
        if self.backend == "codex":
            await self._probe_codex_sandbox()
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def exec(
        self,
        cmd: str,
        workspace: Path,
        shared_dir: Path,
        timeout: int = 30,
    ) -> dict[str, Any]:
        await self.start()
        if self.backend == "host":
            return await shell_exec(cmd, workspace, shared_dir, timeout)
        if self.backend == "codex":
            return await self._exec_codex(cmd, workspace, shared_dir, timeout)
        return {"exit_code": -1, "stdout": "", "stderr": "Sandbox backend was not initialized"}

    async def _probe_codex_sandbox(self) -> None:
        codex_path = shutil.which(self.config.codex_bin)
        if not codex_path:
            raise RuntimeError(
                f"Codex sandbox backend requested, but '{self.config.codex_bin}' is not on PATH."
            )
        result = await self._exec_codex(
            "printf codex-sandbox-ready",
            self.workspace_root,
            self.workspace_root / "shared",
            timeout=10,
        )
        if result["exit_code"] != 0 or "codex-sandbox-ready" not in result["stdout"]:
            details = (result.get("stderr") or result.get("stdout") or "unknown error").strip()
            raise RuntimeError(
                "Codex sandbox backend is installed but not usable in this environment. "
                "NanoMA will not run agent shell commands unsandboxed. "
                f"codex sandbox failed: {details}"
            )

    async def _exec_codex(
        self,
        cmd: str,
        workspace: Path,
        shared_dir: Path,
        timeout: int,
    ) -> dict[str, Any]:
        workspace = workspace.resolve()
        shared_dir = shared_dir.resolve()
        _require_inside(workspace, self.workspace_root)
        _require_inside(shared_dir, self.workspace_root)

        outer_env = _codex_env()
        command_env = _agent_env(workspace, shared_dir, inherit_host=False)
        argv = [
            self.config.codex_bin,
            "sandbox",
            "--permissions-profile",
            ":workspace",
            "-C",
            str(workspace),
        ]
        if self.config.network:
            argv.extend(["-c", "sandbox_workspace_write.network_access=true"])
        argv.extend([
            "/usr/bin/env",
            "-i",
            *[f"{key}={value}" for key, value in sorted(command_env.items())],
            "/bin/sh",
            "-lc",
            cmd,
        ])
        return await _communicate(argv, timeout=timeout, cwd=workspace, env=outer_env)


def _require_inside(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as e:
        raise ValueError(f"Path is outside workspace root: {path}") from e


def _agent_env(workspace: Path, shared_dir: Path, *, inherit_host: bool) -> dict[str, str]:
    base = os.environ.copy() if inherit_host else {}
    base.update({
        "HOME": "/tmp",
        "PATH": os.environ.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
        "WORKSPACE": str(workspace.resolve()),
        "SHARED": str(shared_dir.resolve()),
    })
    return base


def _codex_env() -> dict[str, str]:
    home = os.environ.get("HOME") or str(Path.home())
    env = {
        "HOME": home,
        "PATH": os.environ.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
    }
    codex_home = os.environ.get("CODEX_HOME") or str(Path(home) / ".codex")
    env["CODEX_HOME"] = codex_home
    return env
