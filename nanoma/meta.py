"""Meta-tools: spawn, kill, send, query, wait, transfer, set_bio, get_cost, set_status, rebirth, submit, batch."""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from nanoma.core import Agent, Runtime


# ─── spawn ───────────────────────────────────────────────────────────────────

async def meta_spawn(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    """Create a child agent."""
    from nanoma.core import ResourceQuota

    task = args.get("task", "")
    model = args.get("model")
    delegate = args.get("delegate", False)

    if not task:
        return {"error": "task is required"}
    if agent.depth + 1 > runtime.config.max_depth:
        return {"error": f"Max depth ({runtime.config.max_depth}) exceeded"}
    if len(runtime.agents) >= runtime.config.max_agents:
        return {"error": f"Max agents ({runtime.config.max_agents}) reached"}
    violation = runtime.spawn_policy_violation(agent)
    if violation:
        event = {
            "tool": "spawn",
            "reason": violation,
        }
        if runtime.config.tool_policy_log_events:
            event["policy"] = runtime.current_tool_policy(agent)
        runtime._emit(agent.id, "tool_policy_block", event)
        return {
            "error": f"spawn blocked by tool policy: {violation}",
        }

    # Model selection
    if not model:
        if runtime.router:
            model = runtime.router(task, runtime.ledger.remaining(), allowed_models=runtime.config.allowed_models)
        else:
            model = runtime.config.default_model
    if runtime.config.allowed_models and model not in runtime.config.allowed_models:
        model = runtime.config.allowed_models[0]

    child_quota = ResourceQuota(budget=float("inf"), time_limit=agent.quota.time_limit, max_turns=agent.quota.max_turns)
    child = runtime.create_agent(task=task, model=model, quota=child_quota, parent=agent.id, depth=agent.depth + 1)
    runtime.start_agent(child)

    # Emit spawn event from parent's perspective (for viewer)
    runtime._emit(agent.id, "spawn", {"child": child.id, "task": task[:100], "model": model})

    if delegate:
        agent.status = "done"
        agent.result = f"[Delegated to {child.id}]"

    return {"agent_id": child.id, "model": model}


# ─── kill ────────────────────────────────────────────────────────────────────

async def meta_kill(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    """Terminate an agent."""
    target_id = args.get("agent_id", "")
    target = runtime.agents.get(target_id)
    if not target:
        return {"error": f"Agent '{target_id}' not found"}

    # Permission: self or descendant
    if target_id != agent.id and not _is_descendant(target_id, agent.id, runtime):
        return {"error": "Cannot kill — not a descendant"}

    target.status = "done"
    if target._task and not target._task.done():
        target._task.cancel()

    return {"killed": target_id}


# ─── send ────────────────────────────────────────────────────────────────────

async def meta_send(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    """Send a message to agent(s). No broadcast — must specify IDs."""
    from nanoma.core import Envelope
    from nanoma.llm import estimate_tokens

    to = args.get("to", "")
    message = args.get("message", "")
    mode = args.get("mode", "queue")

    if not to:
        return {"error": "'to' required (agent_id or list of IDs)"}
    if not message:
        return {"error": "'message' required"}
    if mode not in ("immediate", "steer", "queue"):
        return {"error": f"Invalid mode '{mode}'"}

    recipients = [r.strip() for r in to.split(",")] if isinstance(to, str) else to
    msg_tokens = estimate_tokens(message)
    delivered = 0

    for rid in recipients:
        if rid not in runtime.agents:
            continue
        await runtime.deliver(Envelope(
            from_id=agent.id, to_id=rid, content=message,
            tokens=msg_tokens, timestamp=time.time(), mode=mode,
        ))
        delivered += 1
        # Emit send event for viewer
        runtime._emit(agent.id, "send", {
            "from": agent.id, "to": rid, "mode": mode,
            "tokens": msg_tokens, "message": message,
            "message_chars": len(message),
        })

    return {"delivered": delivered, "tokens": msg_tokens, "mode": mode}


# ─── query ───────────────────────────────────────────────────────────────────

async def meta_query(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    """Query agents. messages=N for last N messages, messages=-1 for all."""
    target_id = args.get("agent_id")
    messages_n = args.get("messages", 0)  # 0=meta only, N=last N, -1=all

    if target_id:
        target = runtime.agents.get(target_id)
        if not target:
            return {"error": f"Agent '{target_id}' not found"}
        result: dict[str, Any] = {
            "id": target.id,
            "status": target.status,
            "bio": target.bio,
            "model": target.model,
            "parent": target.parent,
            "children": list(target.children),
            "result": target.result if target.result else None,
            "artifacts": [a.path for a in target.artifacts],
        }
        # Include messages if requested
        if messages_n != 0:
            history = target.history[1:]  # skip system prompt
            if messages_n > 0:
                history = history[-messages_n:]
            # Serialize compactly
            result["messages"] = [
                {"role": m.get("role", ""), "content": (m.get("content") or "")[:500]}
                for m in history if m.get("role") in ("user", "assistant")
            ]
        return result

    # List all agents (lightweight)
    agents_list = []
    for a in runtime.agents.values():
        agents_list.append({
            "id": a.id, "status": a.status, "bio": a.bio,
        })
    return {"agents": agents_list, "count": len(agents_list)}


# ─── wait ────────────────────────────────────────────────────────────────────

async def meta_wait(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    """Block until agents finish. mode='all' waits for everyone, mode='any' returns on first completion."""
    target_ids = args.get("agent_ids", [])
    timeout = args.get("timeout", 120.0)
    mode = args.get("mode", "all")  # "all" or "any"

    if mode not in ("all", "any"):
        return {"error": "mode must be 'all' or 'any'"}

    if not target_ids:
        target_ids = list(agent.children)
    if not target_ids:
        return {"completed": [], "pending": [], "note": "Nothing to wait for"}

    # Cap timeout vs time limit
    if agent.quota.time_limit > 0:
        remaining = agent.quota.time_limit - (time.time() - runtime._start_time)
        timeout = min(timeout, max(0, remaining * 0.9))  # leave 10% margin for cleanup

    completed = []
    interrupted = False
    reason = None

    try:
        async with asyncio.timeout(timeout):
            while True:
                if not agent._immediate_inbox.empty() or not agent._steer_inbox.empty():
                    interrupted = True
                    reason = "message_received"
                    break

                for tid in target_ids:
                    t = runtime.agents.get(tid)
                    if t and t.status in ("done", "failed"):
                        if not any(c["id"] == tid for c in completed):
                            completed.append({
                                "id": tid, "status": t.status,
                                "result": (t.result or "")[:500],
                                "artifacts": [a.path for a in t.artifacts],
                            })

                # mode="any": return as soon as at least one completes
                if mode == "any" and completed:
                    break
                # mode="all": wait for every target
                if mode == "all":
                    all_done = all(
                        any(c["id"] == tid for c in completed)
                        for tid in target_ids
                    )
                    if all_done:
                        break

                await asyncio.sleep(0.5)
    except (asyncio.TimeoutError, TimeoutError):
        interrupted = True
        reason = "timeout"

    # Snapshot pending agents
    pending = []
    for tid in target_ids:
        if not any(c["id"] == tid for c in completed):
            t = runtime.agents.get(tid)
            pending.append({"id": tid, "status": t.status if t else "not_found"})

    result = {"completed": completed, "pending": pending}
    if interrupted:
        result["interrupted"] = True
        result["reason"] = reason
    return result


# ─── transfer ────────────────────────────────────────────────────────────────

async def meta_transfer(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    """Transfer files between agents. Push or Pull."""
    src = args.get("src", "")
    to = args.get("to", "")
    from_agent = args.get("from_agent", "")
    dest = args.get("dest", "")

    if not src:
        return {"error": "src required"}

    shared_dir = runtime._tool_context.shared_dir
    src_list = [src] if isinstance(src, str) else src

    if from_agent:
        # Pull
        source_dir = shared_dir if from_agent == "shared" else (
            runtime.agents[from_agent].workspace if from_agent in runtime.agents else None)
        if not source_dir:
            return {"error": f"Agent '{from_agent}' not found"}
        copied = _copy_files(src_list, source_dir, agent.workspace / dest if dest else agent.workspace)
        return {"pulled": copied, "from": from_agent}

    if not to:
        return {"error": "'to' or 'from_agent' required"}

    # Push
    if to == "shared":
        target_dir = shared_dir
    elif to in runtime.agents:
        target_dir = runtime.agents[to].workspace
    else:
        return {"error": f"Agent '{to}' not found"}

    dest_dir = target_dir / dest if dest else target_dir
    copied = _copy_files(src_list, agent.workspace, dest_dir)
    return {"pushed": copied, "to": to}


def _copy_files(patterns: list[str], src_dir: Path, dest_dir: Path) -> list[str]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for pat in patterns:
        matches = list(src_dir.glob(pat))
        if not matches:
            p = src_dir / pat
            if p.exists():
                matches = [p]
        for src_path in matches:
            target = dest_dir / src_path.name
            if src_path.is_dir():
                shutil.copytree(src_path, target, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, target)
            copied.append(src_path.name)
    return copied


# ─── set_bio ─────────────────────────────────────────────────────────────────

async def meta_set_bio(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    """Update self-description."""
    bio = args.get("bio", "")
    if not bio:
        return {"error": "bio required"}
    agent.bio = bio
    return {"bio": bio}


# ─── get_cost ────────────────────────────────────────────────────────────────

async def meta_get_cost(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    """Get resource status."""
    elapsed = time.time() - runtime._start_time
    context_pct = round(agent.context_tokens / max(1, agent.context_limit) * 100, 1)
    coordination_tools = {"spawn", "send", "wait", "query", "kill", "transfer", "set_bio"}
    has_coordination_tools = bool(coordination_tools - runtime.config.disabled_tools)
    result = {
        "agent_id": agent.id,
        "bio": agent.bio,
        "context_tokens": agent.context_tokens,
        "context_limit": agent.context_limit,
        "context_usage_pct": context_pct,
        "tokens_consumed": agent.tokens_consumed,
        "turns_used": agent._turns,
        "max_turns": agent.quota.max_turns,
        "budget_total": runtime.ledger.total_budget,
        "budget_spent": round(runtime.ledger.total_spent, 6),
        "budget_remaining": round(runtime.ledger.remaining(), 6),
        "elapsed_seconds": round(elapsed, 1),
        "total_agents": len(runtime.agents),
    }
    if has_coordination_tools:
        result["spawned_by"] = agent.parent
        result["sub_agents"] = list(agent.children)
    else:
        result["parent"] = agent.parent
        result["children"] = list(agent.children)
    if agent.quota.time_limit > 0:
        result["time_remaining"] = round(max(0, agent.quota.time_limit - elapsed), 1)
    return result


# ─── set_status ──────────────────────────────────────────────────────────────

async def meta_set_status(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    """Mark done or idle."""
    status = args.get("status", "done")
    result = args.get("result", "")
    if status not in ("done", "idle"):
        return {"error": "status must be 'done' or 'idle'"}
    agent.status = status
    if result:
        agent.result = result
    return {"status": status}


# ─── rebirth ─────────────────────────────────────────────────────────────────

async def meta_rebirth(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    """Context reset with summary."""
    summary = args.get("summary", "")
    if not summary:
        return {"error": "summary required"}
    agent._rebirth_pending = {
        "summary": summary,
        "files": args.get("files", []),
        "new_task": args.get("new_task"),
        "new_bio": args.get("new_bio"),
    }
    return {"scheduled": True, "note": "Context resets next turn."}


# ─── submit ──────────────────────────────────────────────────────────────────

async def meta_submit(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    """Submit a file as artifact."""
    from nanoma.core import Artifact
    path_str = args.get("path", "")
    if not path_str:
        return {"error": "path required"}
    path = Path(path_str)
    if not path.is_absolute():
        path = agent.workspace / path
    if not path.exists():
        return {"error": f"Not found: {path}"}

    # Copy to shared
    shared = runtime._tool_context.shared_dir
    shared.mkdir(parents=True, exist_ok=True)
    dest = shared / path.name
    if dest.exists() and dest.is_dir():
        return {"error": f"Shared destination is a directory: {dest}"}
    shutil.copy2(path, dest)

    artifact = Artifact(path=path_str, absolute_path=path, description=args.get("description", ""), agent_id=agent.id)
    agent.artifacts.append(artifact)
    return {"submitted": path_str, "shared_copy": str(dest)}


# ─── batch ───────────────────────────────────────────────────────────────────

async def meta_batch(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    """Execute tool calls from a JSON file. File format: [{"tool": "...", "args": {...}}, ...]"""
    path_str = args.get("path", "")
    if not path_str:
        return {"error": "path required"}
    path = Path(path_str)
    if not path.is_absolute():
        path = agent.workspace / path
    if not path.exists():
        return {"error": f"Not found: {path}"}

    try:
        calls = json.loads(path.read_text())
    except Exception as e:
        return {"error": f"Parse error: {e}"}

    if not isinstance(calls, list):
        return {"error": "File must contain a JSON array of {tool, args} objects"}

    from nanoma.tools import WORK_TOOLS
    from nanoma.plugins.workspace_tools import WORKSPACE_TOOLS
    all_tools = {
        name: tool
        for name, tool in {**WORK_TOOLS, **WORKSPACE_TOOLS, **META_TOOLS}.items()
        if name not in runtime.config.disabled_tools
    }
    all_tools, _ = runtime._apply_state_tool_policy(agent, all_tools)
    results = []

    for i, call in enumerate(calls):
        tool_name = call.get("tool", "")
        tool_args = call.get("args", {})
        tool_info = all_tools.get(tool_name)
        if not tool_info:
            results.append({"index": i, "error": f"Unknown tool: {tool_name}"})
            continue
        try:
            handler = tool_info["handler"]
            if tool_info.get("is_meta"):
                r = await handler(tool_args, agent, runtime)
            else:
                r = await handler(tool_args, agent.workspace, runtime._tool_context)
            results.append({"index": i, "tool": tool_name, "result": r})
        except Exception as e:
            results.append({"index": i, "error": str(e)})

    return {"executed": len(results), "results": results}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _is_descendant(target_id: str, ancestor_id: str, runtime: "Runtime") -> bool:
    visited = set()
    stack = list(runtime.agents[ancestor_id].children) if ancestor_id in runtime.agents else []
    while stack:
        current = stack.pop()
        if current == target_id:
            return True
        if current in visited:
            continue
        visited.add(current)
        if current in runtime.agents:
            stack.extend(runtime.agents[current].children)
    return False


# ─── Registry ────────────────────────────────────────────────────────────────

META_TOOLS: dict[str, dict[str, Any]] = {
    "spawn": {"handler": meta_spawn, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "spawn",
        "description": "Create a new agent that starts immediately and runs in parallel. The child agent gets its own workspace, fresh context, and the same tools. It cannot see your conversation history — include all necessary context in the task description. You become its parent and receive a notification when it finishes.",
        "parameters": {"type": "object", "properties": {
            "task": {"type": "string", "description": "Complete task description for the new agent. Include all context it needs — it cannot see your history."},
            "model": {"type": "string", "description": "LLM model override (omit to use default)"},
            "delegate": {"type": "boolean", "description": "If true, you terminate immediately and the child inherits your role", "default": False},
        }, "required": ["task"]},
    }}},
    "kill": {"handler": meta_kill, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "kill",
        "description": "Terminate an agent. You can only kill yourself or your descendants (children, grandchildren, etc). The agent stops immediately.",
        "parameters": {"type": "object", "properties": {
            "agent_id": {"type": "string", "description": "ID of the agent to terminate"},
        }, "required": ["agent_id"]},
    }}},
    "send": {"handler": meta_send, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "send",
        "description": "Send a message to one or more agents. The message appears in their conversation as a user message. Use to communicate results, give instructions, or coordinate. Modes: 'queue' (delivered next turn, default), 'steer' (delivered after current tool calls), 'immediate' (interrupts current processing). Sending to an 'idle' agent wakes it up.",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string", "description": "Recipient agent ID. For multicast, pass comma-separated IDs."},
            "message": {"type": "string", "description": "Message content to deliver"},
            "mode": {"type": "string", "enum": ["immediate", "steer", "queue"], "description": "Delivery priority: queue (next turn), steer (after current tools), immediate (interrupts)", "default": "queue"},
        }, "required": ["to", "message"]},
    }}},
    "query": {"handler": meta_query, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "query",
        "description": "Discover agents in the system. Without agent_id: returns list of all agents with {id, status, bio}. With agent_id: returns detailed info including parent, children, result, artifacts. Use messages=N to peek at an agent's recent conversation (N messages, or -1 for all).",
        "parameters": {"type": "object", "properties": {
            "agent_id": {"type": "string", "description": "Query a specific agent for detailed info (omit to list all)"},
            "messages": {"type": "integer", "description": "Include last N messages from agent's history (0=none, -1=all)", "default": 0},
        }},
    }}},
    "wait": {"handler": meta_wait, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "wait",
        "description": "Block execution until agents finish. Returns {completed: [{id, status, result, artifacts}], pending: [{id, status}]}. Interrupted early if you receive a message or timeout expires. Use mode='any' in a loop to process results as they arrive incrementally.",
        "parameters": {"type": "object", "properties": {
            "agent_ids": {"type": "array", "items": {"type": "string"}, "description": "Which agents to wait for (default: all your children)"},
            "timeout": {"type": "number", "description": "Max seconds to wait before returning", "default": 120},
            "mode": {"type": "string", "enum": ["all", "any"], "description": "'all' = block until every agent finishes, 'any' = return as soon as one finishes", "default": "all"},
        }},
    }}},
    "transfer": {"handler": meta_transfer, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "transfer",
        "description": "Copy files between agent workspaces. Push: transfer(src=file, to=agent_id) copies from your workspace to theirs. Pull: transfer(src=file, from_agent=agent_id) copies from theirs to yours. Use to='shared' or from_agent='shared' for the shared directory. Supports glob patterns.",
        "parameters": {"type": "object", "properties": {
            "src": {"type": "string", "description": "Source file path or glob pattern"},
            "to": {"type": "string", "description": "Push destination: agent_id or 'shared'"},
            "from_agent": {"type": "string", "description": "Pull source: agent_id or 'shared'"},
            "dest": {"type": "string", "description": "Subdirectory at destination to place files into"},
        }, "required": ["src"]},
    }}},
    "set_bio": {"handler": meta_set_bio, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "set_bio",
        "description": "Set your self-description. Other agents see this when they call query(). Use to advertise your role, capabilities, or current status so others can find and communicate with you.",
        "parameters": {"type": "object", "properties": {
            "bio": {"type": "string", "description": "Short description of your role, expertise, or current status"},
        }, "required": ["bio"]},
    }}},
    "get_cost": {"handler": meta_get_cost, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "get_cost",
        "description": "Get your identity and resource status. Returns: your agent_id, bio, parent, children list, context_tokens/limit, turns used/max, elapsed time, and remaining budget.",
        "parameters": {"type": "object", "properties": {}},
    }}},
    "set_status": {"handler": meta_set_status, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "set_status",
        "description": "Change your lifecycle status. 'done': terminate and notify parent with your result (parent receives '[Agent X finished: done] result'). 'idle': pause your loop and sleep — you wake automatically when any message arrives in your inbox.",
        "parameters": {"type": "object", "properties": {
            "status": {"type": "string", "enum": ["done", "idle"], "description": "'done' = terminate, 'idle' = sleep until messaged"},
            "result": {"type": "string", "description": "Summary of your output (sent to parent on 'done')"},
        }, "required": ["status"]},
    }}},
    "rebirth": {"handler": meta_rebirth, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "rebirth",
        "description": "Reset your context window to save memory. Your entire conversation history is wiped and replaced with just the summary you provide. Same agent ID, same workspace, same tools — but fresh context. Use when your context is getting full (check via get_cost).",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "description": "Everything you need to remember — this is ALL you'll have after rebirth"},
            "files": {"type": "array", "items": {"type": "string"}, "description": "Key file paths to reference in the new context"},
            "new_task": {"type": "string", "description": "Optionally replace your task description"},
            "new_bio": {"type": "string", "description": "Optionally update your bio"},
        }, "required": ["summary"]},
    }}},
    "submit": {"handler": meta_submit, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "submit",
        "description": "Mark a file as a final deliverable/artifact. The file is copied to the shared/ directory so all agents and the user can access it. Use for outputs that represent completed work.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path to the file to submit (relative to your workspace)"},
            "description": {"type": "string", "description": "Brief description of what this deliverable is"},
        }, "required": ["path"]},
    }}},
    "batch": {"handler": meta_batch, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "batch",
        "description": "Execute multiple tool calls from a JSON file. File format: [{\"tool\": \"name\", \"args\": {...}}, ...]. Results are returned in order. Useful for programmatic or bulk operations when you need to make many calls at once.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path to JSON file containing array of {tool, args} objects"},
        }, "required": ["path"]},
    }}},
}
