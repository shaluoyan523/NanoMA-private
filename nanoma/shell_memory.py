"""Memory as something a shell command has to be admitted against.

The runtime already caps how many children may run at once, derived from the
container's memory limit divided by an assumed per-child figure. That accounts
for the wrong thing. A child agent is a Python loop holding an LLM client — a
couple of hundred megabytes. The memory is spent by what the child *runs*: on
2026-07-31 a run with a correctly enforced cap of 15 children was OOM-killed
29 minutes in, because single `shell` calls building a vector index held
16.8 GB, 14.8 GB and 12.0 GB. Fifteen of those is not a limit, it is a wish.

So the resource is claimed where it is spent. A command's cost is not declared,
it is remembered: the first run of something new is observed, and later runs of
the same shape have to fit in what is left before they start. Commands that
have never been seen are admitted against the container's current usage
instead, which is the only thing known about them.

Nothing here is specific to a benchmark. Any task whose agents run builds, test
suites or data loads has the same exposure, and the numbers come from the run
itself rather than from configuration.
"""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

# Left for the runtime, the agents' own interpreters, and the page cache the
# kernel will not give back instantly. Reserving nothing would admit a command
# that fits only if everything else in the container is already gone.
_RESERVE_FRACTION = 0.15

# Above this share of the limit, a command whose cost is unknown waits. It is
# the only guard available for something never measured: refusing to start work
# into a nearly full container is cheap, and being wrong costs a delay.
_UNKNOWN_HEADROOM_FRACTION = 0.75


def _env_bytes(name: str) -> int | None:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        return int(float(raw) * 1024 * 1024)
    except ValueError:
        return None


def container_memory_limit_bytes() -> int | None:
    """Best-effort memory-cgroup limit for this process (v2 then v1)."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if not raw or raw == "max":
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if 0 < value < (1 << 62):
            return value
    return None


def container_memory_usage_bytes() -> int | None:
    """What the cgroup is holding right now, page cache included.

    The cache counts: it is reclaimable, but it is also what the kernel is
    holding when it decides whether an allocation can be served.
    """
    for path in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return None


def command_shape(cmd: str) -> str:
    """A key that groups commands which cost about the same to run.

    Runs of digits are dropped, so sweeping a parameter does not look like a
    different command every time — `--ef 40` and `--ef 60` build the same index
    and hold the same memory. Names are kept, so `python small.py` and
    `python big.py` stay apart.
    """
    return re.sub(r"\d+", "#", " ".join(cmd.split()))[:400]


class ShellMemoryArbiter:
    """Admits shell commands against a memory budget learnt from the run.

    One instance per process: every agent runs in the same event loop and the
    same cgroup, so the budget they compete for is shared and there is nothing
    to synchronise across threads.
    """

    def __init__(self) -> None:
        self._observed: dict[str, int] = {}
        self._reserved: int = 0
        self._in_flight: int = 0
        self._room = asyncio.Condition()
        self.waits: int = 0  # for tests and for the run's own record

    # ── what is known ────────────────────────────────────────────────────────

    def budget_bytes(self) -> int | None:
        """What all concurrent shell commands together may hold."""
        override = _env_bytes("NANOMA_SHELL_MEMORY_BUDGET_MB")
        if override is not None:
            return override if override > 0 else None
        limit = container_memory_limit_bytes()
        if not limit:
            return None  # unlimited or unknowable: nothing to admit against
        return int(limit * (1.0 - _RESERVE_FRACTION))

    def estimate_bytes(self, cmd: str) -> int | None:
        """What this command has been seen to hold, or None if never measured.

        The largest observation wins rather than the average. Admitting on the
        average means half the runs of a command do not fit the room made for
        them, and the failure mode is not a slow run, it is a dead container.
        """
        return self._observed.get(command_shape(cmd))

    def record(self, cmd: str, peak_rss_bytes: int | None) -> None:
        """Remember what a command turned out to cost."""
        if not peak_rss_bytes or peak_rss_bytes <= 0:
            return
        key = command_shape(cmd)
        if peak_rss_bytes > self._observed.get(key, 0):
            self._observed[key] = int(peak_rss_bytes)

    # ── admission ────────────────────────────────────────────────────────────

    def _fits(self, want: int, budget: int) -> bool:
        if self._reserved + want <= budget:
            return True
        # A command bigger than the whole budget can still be the right thing to
        # run; it just cannot share. Refusing it outright would make the task
        # impossible rather than slow.
        return want > budget and self._reserved == 0

    def _unknown_may_start(self, budget: int) -> bool:
        usage = container_memory_usage_bytes()
        if usage is None:
            return True  # nothing to go on; do not invent a reason to block
        limit = container_memory_limit_bytes() or budget
        return usage < limit * _UNKNOWN_HEADROOM_FRACTION

    async def acquire(self, cmd: str, timeout: float) -> dict:
        """Wait for room to run `cmd`. Returns how the wait went.

        `granted` is False only when the wait ran out, which the caller reports
        to the agent rather than raising: a busy machine is a condition to work
        around, not a broken tool.
        """
        budget = self.budget_bytes()
        if budget is None:
            self._in_flight += 1
            return {"granted": True, "reserved": 0, "waited": 0.0}

        want = self.estimate_bytes(cmd)
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + max(0.0, timeout)

        async with self._room:
            while True:
                ready = (
                    self._fits(want, budget) if want is not None
                    else self._unknown_may_start(budget)
                )
                if ready:
                    reserved = want or 0
                    self._reserved += reserved
                    self._in_flight += 1
                    waited = loop.time() - started
                    if waited > 0:
                        self.waits += 1
                    return {"granted": True, "reserved": reserved, "waited": waited}
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return {
                        "granted": False,
                        "reserved": 0,
                        "waited": loop.time() - started,
                        "wanted": want,
                        "held": self._reserved,
                        "budget": budget,
                    }
                try:
                    await asyncio.wait_for(self._room.wait(), timeout=min(remaining, 5.0))
                except asyncio.TimeoutError:
                    # Re-check anyway: an unknown command is admitted on the
                    # cgroup's usage, which falls without anyone notifying us.
                    continue

    async def release(self, cmd: str, ticket: dict, peak_rss_bytes: int | None) -> None:
        self.record(cmd, peak_rss_bytes)
        async with self._room:
            self._reserved = max(0, self._reserved - int(ticket.get("reserved") or 0))
            self._in_flight = max(0, self._in_flight - 1)
            self._room.notify_all()

    def snapshot(self) -> dict:
        """For the run's own record, and for tests."""
        budget = self.budget_bytes()
        return {
            "budget_bytes": budget,
            "reserved_bytes": self._reserved,
            "in_flight": self._in_flight,
            "commands_measured": len(self._observed),
            "waits": self.waits,
        }


_arbiter: ShellMemoryArbiter | None = None


def arbiter() -> ShellMemoryArbiter:
    global _arbiter
    if _arbiter is None:
        _arbiter = ShellMemoryArbiter()
    return _arbiter


def reset_for_tests() -> None:
    global _arbiter
    _arbiter = None
