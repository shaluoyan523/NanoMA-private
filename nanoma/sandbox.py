"""Shell execution — no sandboxing."""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path
from typing import Any


async def shell_exec(
    cmd: str,
    workspace: Path,
    shared_dir: Path,
    timeout: int = 30,
) -> dict[str, Any]:
    """Execute a shell command in the workspace directory."""
    env = {**os.environ, "WORKSPACE": str(workspace), "SHARED": str(shared_dir)}

    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workspace),
            env=env,
        )
        communicate_task = asyncio.create_task(proc.communicate())
        stdout, stderr = await asyncio.wait_for(asyncio.shield(communicate_task), timeout=timeout)
        return {
            "exit_code": proc.returncode,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }
    except asyncio.TimeoutError:
        proc.kill()
        with contextlib.suppress(asyncio.CancelledError):
            await communicate_task
        await proc.wait()
        return {"exit_code": -1, "stdout": "", "stderr": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"exit_code": -1, "stdout": "", "stderr": str(e)}
