"""Resource admission changes are opt-in and affect commands, not agents."""
import asyncio
import sys

import pytest

from nanoma import Runtime, RuntimeConfig
from nanoma.core import ToolContext
from nanoma import shell_memory
from nanoma.tools import tool_shell


def arbiter(monkeypatch, usage=100):
    memory = shell_memory.ShellMemoryArbiter()
    monkeypatch.setattr(memory, "budget_bytes", lambda: 1000)
    monkeypatch.setattr(shell_memory, "container_memory_limit_bytes", lambda: 1200)
    monkeypatch.setattr(shell_memory, "container_memory_usage_bytes", lambda: usage)
    return memory


@pytest.mark.asyncio
async def test_unknown_commands_reserve_before_their_rss_grows(monkeypatch):
    memory = arbiter(monkeypatch)
    tickets = [await memory.acquire(f"build-{letter}", 0, strict=True) for letter in "abc"]
    assert all(t["granted"] and t["reserved"] == 250 for t in tickets)
    denied = await memory.acquire("another-new-build", 0, strict=True)
    assert not denied["granted"]
    await memory.release("build-a", tickets[0], 100)
    assert (await memory.acquire("another-new-build", 0, strict=True))["granted"]


@pytest.mark.asyncio
async def test_known_commands_account_for_external_usage_without_double_counting(monkeypatch):
    memory = arbiter(monkeypatch)
    memory.record("measured", 500)
    ticket = await memory.acquire("measured", 0, strict=True)
    ticket["rss"] = 500
    monkeypatch.setattr(shell_memory, "container_memory_usage_bytes", lambda: 800)
    memory.record("small", 100)
    assert (await memory.acquire("small", 0, strict=True))["granted"]
    memory.record("larger", 300)
    assert not (await memory.acquire("larger", 0, strict=True))["granted"]


@pytest.mark.asyncio
async def test_measured_small_commands_have_no_fixed_concurrency_cap(monkeypatch):
    memory = arbiter(monkeypatch)
    memory.record("small", 20)
    tickets = [await memory.acquire("small", 0, strict=True) for _ in range(20)]
    assert all(ticket["granted"] for ticket in tickets)
    assert memory.snapshot()["in_flight"] == 20


@pytest.mark.asyncio
async def test_default_legacy_admission_is_unchanged(monkeypatch):
    memory = arbiter(monkeypatch)
    tickets = [await memory.acquire("unknown", 0) for _ in range(8)]
    assert all(t["granted"] and t["reserved"] == 0 for t in tickets)
    assert RuntimeConfig().shell_memory_strict_admission is False


@pytest.mark.asyncio
async def test_cancelled_wait_does_not_leak_reservations(monkeypatch):
    memory = arbiter(monkeypatch)
    tickets = [await memory.acquire("unknown", 0, strict=True) for _ in range(3)]
    waiting = asyncio.create_task(memory.acquire("fourth", 10, strict=True))
    await asyncio.sleep(.01)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert memory.snapshot()["reserved_bytes"] == 750
    for ticket in tickets:
        await memory.release("unknown", ticket, 100)
    assert memory.snapshot()["in_flight"] == 0


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="production process tracking uses /proc")
async def test_background_process_retains_ticket_until_it_exits(tmp_path, monkeypatch):
    memory = shell_memory.ShellMemoryArbiter()
    monkeypatch.setattr(memory, "budget_bytes", lambda: 64 * 1024 * 1024)
    monkeypatch.setattr(shell_memory, "container_memory_usage_bytes", lambda: 0)
    monkeypatch.setattr(shell_memory, "_arbiter", memory)
    shared = tmp_path / "shared"
    shared.mkdir()
    ctx = ToolContext(shared_dir=shared, workspace_root=tmp_path,
                      shell_memory_strict_admission=True)
    result = await tool_shell({"command": "sleep 1.5 > /dev/null 2>&1 &", "timeout": 3}, tmp_path, ctx)
    assert result["exit_code"] == 0
    assert memory.snapshot()["in_flight"] == 1
    await asyncio.wait_for(asyncio.gather(*memory._background), 4)
    assert memory.snapshot()["in_flight"] == 0


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="production process tracking uses /proc")
async def test_cancelled_foreground_releases_resources(tmp_path, monkeypatch):
    memory = shell_memory.ShellMemoryArbiter()
    monkeypatch.setattr(memory, "budget_bytes", lambda: 64 * 1024 * 1024)
    monkeypatch.setattr(shell_memory, "container_memory_usage_bytes", lambda: 0)
    monkeypatch.setattr(shell_memory, "_arbiter", memory)
    (tmp_path / "shared").mkdir()
    ctx = ToolContext(shared_dir=tmp_path / "shared", workspace_root=tmp_path,
                      shell_memory_strict_admission=True)
    call = asyncio.create_task(tool_shell({"command": "sleep 60", "timeout": 60}, tmp_path, ctx))
    await asyncio.sleep(.15)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    if memory._background:
        await asyncio.wait_for(asyncio.gather(*memory._background), 4)
    assert memory.snapshot()["in_flight"] == 0
