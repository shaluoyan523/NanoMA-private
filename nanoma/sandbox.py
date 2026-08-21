"""Shell execution — no sandboxing."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from pathlib import Path
from typing import Any


# Ceiling on what one stream may accumulate in memory. The caller truncates for
# the agent anyway; this only stops a runaway producer from being held in full
# inside a memory-capped container.
_MAX_STREAM_BYTES = 64 * 1024 * 1024


async def _drain(stream: asyncio.StreamReader | None, sink: bytearray) -> None:
    """Accumulate a pipe as it is written, so a kill does not take it with it.

    Keeps the head rather than the tail: a command that dies after producing too
    much is being read for how far it got, and its early output is what says so.
    """
    if stream is None:
        return
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            return
        if len(sink) < _MAX_STREAM_BYTES:
            sink.extend(chunk[: _MAX_STREAM_BYTES - len(sink)])


# How often the resident size of a running command is sampled. A command that
# allocates and frees between two samples is undercounted; the interval trades
# that against scanning /proc, and a build that peaks for less than a second is
# not what exhausts a container.
_RSS_SAMPLE_SECONDS = 1.0


def _ancestor_pids(pid: int | None = None) -> set[int]:
    """Return this process and its ancestors from procfs."""
    current = int(pid or os.getpid())
    ancestors: set[int] = set()
    while current > 1 and current not in ancestors:
        ancestors.add(current)
        try:
            raw = Path(f"/proc/{current}/stat").read_bytes()
            cut = raw.rfind(b")")
            fields = raw[cut + 2:].split() if cut >= 0 else []
            parent = int(fields[1]) if len(fields) >= 2 else 0
        except (OSError, ValueError):
            break
        current = parent
    if current == 1:
        ancestors.add(1)
    return ancestors


def _kill_guard_prelude(protected_pids: set[int]) -> str:
    """Define a shell `kill` wrapper that refuses to target runtime ancestors."""
    protected = "|".join(str(pid) for pid in sorted(protected_pids)) or "0"
    return f"""
kill() {{
  for _nanoma_target in "$@"; do
    case "$_nanoma_target" in
      {protected})
        printf '%s\\n' "Blocked kill of protected NanoMA runtime pid $_nanoma_target" >&2
        return 126
        ;;
    esac
  done
  command kill "$@"
}}
export -f kill
""".strip()


def process_group_rss_bytes(pgid: int) -> int:
    """Resident bytes held right now by every process in one group.

    `shell_exec` starts each command in its own session, so the whole tree a
    command builds shares its leader's group id and nothing else does. That makes
    the group the unit to measure: the memory belongs to the `python` a command
    launched, not to the shell that is waiting on it.
    """
    page_size = os.sysconf("SC_PAGE_SIZE")
    total = 0
    try:
        entries = os.listdir("/proc")
    except OSError:
        return 0
    for name in entries:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as handle:
                raw = handle.read()
        except OSError:
            continue  # exited between the listing and the read
        # comm is parenthesised and may contain spaces, so fields are counted
        # from after the last ')': index 0 is state, i.e. field 3.
        cut = raw.rfind(b")")
        if cut < 0:
            continue
        fields = raw[cut + 2:].split()
        if len(fields) < 22:
            continue
        try:
            if int(fields[2]) != pgid:  # field 5, pgrp
                continue
            total += int(fields[21]) * page_size  # field 24, rss in pages
        except ValueError:
            continue
    return total


async def _sample_peak_rss(pgid: int, into: dict[str, int]) -> None:
    """Track the largest resident total this command's tree is seen holding."""
    while True:
        try:
            current = process_group_rss_bytes(pgid)
        except Exception:
            return
        if current > into.get("peak_rss_bytes", 0):
            into["peak_rss_bytes"] = current
        await asyncio.sleep(_RSS_SAMPLE_SECONDS)


async def shell_exec(
    cmd: str,
    workspace: Path,
    shared_dir: Path,
    timeout: int = 30,
) -> dict[str, Any]:
    """Execute a shell command in the workspace directory.

    Output is accumulated as it arrives rather than collected at the end, so a
    command that runs out of time still returns what it managed to print. It used
    to `communicate()` and discard the buffer on timeout, which meant a long
    measurement returned `stdout: ""` and the single line "Timeout after 300s".

    That is not a small loss. On a task whose benchmark sweep takes 30-70 minutes,
    every attempt to run it returned literally nothing, and the only way for an
    agent to see any number at all was to shrink the work until it fit — which on
    one run meant deleting the parameters from the scored config that had produced
    its best result. Half the shell time in that run's first iteration, and 80% of
    the second's, went into calls that came back empty.
    """
    protected_pids = _ancestor_pids()
    env = {
        **os.environ,
        "WORKSPACE": str(workspace),
        "SHARED": str(shared_dir),
        "NANOMA_PROTECTED_PIDS": " ".join(str(pid) for pid in sorted(protected_pids)),
    }
    guarded_cmd = f"{_kill_guard_prelude(protected_pids)}\n{cmd}"
    proc: asyncio.subprocess.Process | None = None
    out, err = bytearray(), bytearray()
    readers: list[asyncio.Task] = []
    sampler: asyncio.Task | None = None
    # Filled in while the command runs. What it costs to run is a property of the
    # command, and until it was measured nothing in the runtime knew that one call
    # could hold two thirds of the container on its own.
    observed: dict[str, int] = {}

    def decoded(timed_out: bool) -> dict[str, Any]:
        stderr = err.decode(errors="replace")
        if timed_out:
            note = f"Timeout after {timeout}s"
            stderr = f"{stderr}\n{note}" if stderr else note
        return {
            "exit_code": -1 if timed_out else proc.returncode,
            "stdout": out.decode(errors="replace"),
            "stderr": stderr,
            **({"timed_out": True} if timed_out else {}),
            **(
                {"peak_rss_bytes": observed["peak_rss_bytes"]}
                if observed.get("peak_rss_bytes")
                else {}
            ),
        }

    try:
        proc = await asyncio.create_subprocess_shell(
            guarded_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workspace),
            env=env,
            start_new_session=True,
            executable="/bin/bash",
        )
        sampler = asyncio.create_task(_sample_peak_rss(proc.pid, observed))
        readers = [
            asyncio.create_task(_drain(proc.stdout, out)),
            asyncio.create_task(_drain(proc.stderr, err)),
        ]
        try:
            _, pending = await asyncio.wait(readers, timeout=timeout)
        finally:
            sampler.cancel()
        if pending:
            for task in pending:
                task.cancel()
            await _terminate_process_tree(proc)
            return decoded(timed_out=True)
        # The pipes are closed, but a process can outlive them; the wait is
        # bounded so a lingering child cannot hold the agent here indefinitely.
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            await _terminate_process_tree(proc)
        return decoded(timed_out=False)
    except Exception as e:
        for task in readers:
            task.cancel()
        if sampler is not None:
            sampler.cancel()
        if proc is not None and proc.returncode is None:
            await _terminate_process_tree(proc)
        result = decoded(timed_out=False)
        result["exit_code"] = -1
        result["stderr"] = f"{result['stderr']}\n{e}" if result["stderr"] else str(e)
        return result


async def _terminate_process_tree(proc: asyncio.subprocess.Process | None) -> None:
    """Terminate the shell and any children it started."""
    if proc is None or proc.returncode is not None:
        return

    def kill_group(sig: signal.Signals) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, sig)

    kill_group(signal.SIGTERM)
    wait_task = asyncio.create_task(proc.wait())
    done, _ = await asyncio.wait({wait_task}, timeout=2)
    if not done:
        kill_group(signal.SIGKILL)
        await wait_task
