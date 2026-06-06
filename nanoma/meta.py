"""Meta-tools: spawn, kill, send, query, wait, transfer, set_bio, get_cost, set_status, rebirth, compact, submit, batch."""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

from nanoma.memory import ExperienceCard

if TYPE_CHECKING:
    from nanoma.core import Agent, Runtime


ALLOWED_SET_STATUS_ACTIONS = {"done", "self_stop", "create", "stop", "read", "message", "work", "compact"}


def _normalize_workflow_prior(value: str | None) -> str:
    if value in (None, ""):
        return ""
    if value not in {"research_loop", "critic_review_loop", "synthesis_loop"}:
        return value
    return value


def _agent_snapshot(target: "Agent", runtime: "Runtime") -> dict[str, Any]:
    return {
        "id": target.id,
        "status": target.status,
        "bio": target.bio,
        "model": target.model,
        "parent": target.parent,
        "children": list(target.children),
        "result": target.result if target.result else None,
        "artifacts": [a.path for a in target.artifacts],
        "role": target.role,
        "create_type": target.create_type,
        "relationship": target.relationship,
        "created_by": target.created_by,
        "group_id": target.group_id,
        "workflow_prior": target.workflow_prior,
        "action_state": target.action_state,
        "state_board": runtime.state_board_get(target.id),
        "public_memory": runtime.memory.serialize(target.id),
        "memory": target.memory,
    }


async def meta_spawn(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    from nanoma.core import ResourceQuota

    task = args.get("task", "")
    model = args.get("model")
    delegate = args.get("delegate", False)
    role = args.get("role", "")
    create_type = args.get("create_type", "")
    relationship = args.get("relationship", "")
    group_id = args.get("group_id", "")
    workflow_prior = _normalize_workflow_prior(args.get("workflow_prior"))
    current_task_tags = list(args.get("current_task_tags", []))

    if not task:
        return {"error": "task is required"}
    if agent.depth + 1 > runtime.config.max_depth:
        return {"error": f"Max depth ({runtime.config.max_depth}) exceeded"}
    if len(runtime.agents) >= runtime.config.max_agents:
        return {"error": f"Max agents ({runtime.config.max_agents}) reached"}

    if not model:
        if runtime.router:
            model = runtime.router(task, runtime.ledger.remaining(), allowed_models=runtime.config.allowed_models)
        else:
            model = runtime.config.default_model
    if runtime.config.allowed_models and model not in runtime.config.allowed_models:
        model = runtime.config.allowed_models[0]

    if create_type == "peer_agent":
        relationship = relationship or "peer"
        created_by = agent.id
    else:
        created_by = None

    child_quota = ResourceQuota(budget=float("inf"), time_limit=agent.quota.time_limit, max_turns=agent.quota.max_turns)
    child = runtime.create_agent(
        task=task,
        model=model,
        quota=child_quota,
        parent=agent.id,
        depth=agent.depth + 1,
        role=role,
        create_type=create_type,
        relationship=relationship,
        created_by=created_by,
        group_id=group_id,
        workflow_prior=workflow_prior,
        current_task_tags=current_task_tags,
    )
    runtime.start_agent(child)
    runtime.state_board_update(child.id, action_state="create", current_task_tags=current_task_tags)

    runtime._emit(agent.id, "spawn", {
        "child": child.id,
        "task": task[:100],
        "model": model,
        "role": role,
        "create_type": create_type,
        "relationship": relationship,
        "created_by": created_by,
        "group_id": group_id,
        "workflow_prior": workflow_prior,
    })

    if delegate:
        agent.status = "done"
        agent.action_state = "stop"
        agent.result = f"[Delegated to {child.id}]"
        runtime.state_board_sync(agent.id)

    return {
        "agent_id": child.id,
        "model": model,
        "role": role,
        "create_type": create_type,
        "relationship": relationship,
        "created_by": created_by,
        "group_id": group_id,
        "workflow_prior": workflow_prior,
    }


async def meta_create_agent(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    return await meta_spawn(args, agent, runtime)


async def meta_kill(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    target_id = args.get("agent_id", "")
    emergency = bool(args.get("emergency", False))
    target = runtime.agents.get(target_id)
    if not target:
        return {"error": f"Agent '{target_id}' not found"}
    if not emergency:
        return {"error": "kill is emergency-only; pass emergency=True"}
    if target_id != agent.id and not _is_descendant(target_id, agent.id, runtime):
        return {"error": "Cannot kill — not a descendant"}

    target.action_state = "stop"
    target.status = "done"
    runtime.memory.update(
        target.id,
        clear_active_task=True,
        add_experience=ExperienceCard(summary=target.result or "Emergency stop", tags=target.current_task_tags, artifacts=[a.path for a in target.artifacts]),
    )
    target.memory = {"public_memory": runtime.memory.serialize(target.id)}
    runtime.state_board_sync(target.id)
    if target._task and not target._task.done():
        target._task.cancel()
    return {"killed": target_id, "emergency": True}


async def meta_send(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    from nanoma.core import Envelope
    from nanoma.llm import estimate_tokens

    to = args.get("to", "")
    message = args.get("message", "")
    mode = args.get("mode", "queue")
    message_type = args.get("message_type", "text")
    payload = args.get("payload")
    requires_ack = bool(args.get("requires_ack", False))
    urgency = args.get("urgency", "normal")

    if not to:
        return {"error": "'to' required (agent_id or list of IDs)"}
    if not message and payload is None:
        return {"error": "'message' or 'payload' required"}
    if mode not in ("immediate", "steer", "queue"):
        return {"error": f"Invalid mode '{mode}'"}

    recipients = [r.strip() for r in to.split(",")] if isinstance(to, str) else to
    content = message or json.dumps(payload, ensure_ascii=False)
    msg_tokens = estimate_tokens(content)
    delivered = 0
    runtime.state_board_update(agent.id, action_state="message")

    for rid in recipients:
        if rid not in runtime.agents:
            continue
        await runtime.deliver(Envelope(
            from_id=agent.id,
            to_id=rid,
            content=content,
            tokens=msg_tokens,
            timestamp=time.time(),
            message_type=message_type,
            payload=payload,
            requires_ack=requires_ack,
            urgency=urgency,
            mode=mode,
        ))
        delivered += 1
        runtime._emit(agent.id, "send", {
            "from": agent.id,
            "to": rid,
            "mode": mode,
            "tokens": msg_tokens,
            "message": content,
            "message_chars": len(content),
            "message_type": message_type,
            "payload": payload,
            "requires_ack": requires_ack,
            "urgency": urgency,
        })

    return {
        "delivered": delivered,
        "tokens": msg_tokens,
        "mode": mode,
        "message_type": message_type,
        "requires_ack": requires_ack,
        "urgency": urgency,
    }


async def meta_query(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    target_id = args.get("agent_id")
    messages_n = args.get("messages", 0)
    if target_id:
        target = runtime.agents.get(target_id)
        if not target:
            return {"error": f"Agent '{target_id}' not found"}
        result = _agent_snapshot(target, runtime)
        if messages_n != 0:
            history = target.history[1:]
            if messages_n > 0:
                history = history[-messages_n:]
            result["messages"] = [
                {"role": m.get("role", ""), "content": (m.get("content") or "")[:500]}
                for m in history if m.get("role") in ("user", "assistant", "system")
            ]
        return result

    return {
        "agents": [_agent_snapshot(a, runtime) for a in runtime.agents.values()],
        "count": len(runtime.agents),
        "state_board": runtime.state_board_list(),
        "public_memory": runtime.memory.list(),
    }


async def meta_wait(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    target_ids = args.get("agent_ids", [])
    timeout = args.get("timeout", 120.0)
    mode = args.get("mode", "all")
    if mode not in ("all", "any"):
        return {"error": "mode must be 'all' or 'any'"}
    if not target_ids:
        target_ids = list(agent.children)
    if not target_ids:
        return {"completed": [], "pending": [], "note": "Nothing to wait for"}

    if agent.quota.time_limit > 0:
        remaining = agent.quota.time_limit - (time.time() - runtime._start_time)
        timeout = min(timeout, max(0, remaining * 0.9))

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
                    if t and t.status in ("done", "failed") and not any(c["id"] == tid for c in completed):
                        completed.append({
                            "id": tid,
                            "status": t.status,
                            "result": (t.result or "")[:500],
                            "artifacts": [a.path for a in t.artifacts],
                        })
                if mode == "any" and completed:
                    break
                if mode == "all" and all(any(c["id"] == tid for c in completed) for tid in target_ids):
                    break
                await asyncio.sleep(0.5)
    except (asyncio.TimeoutError, TimeoutError):
        interrupted = True
        reason = "timeout"

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


async def meta_transfer(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    src = args.get("src", "")
    to = args.get("to", "")
    from_agent = args.get("from_agent", "")
    dest = args.get("dest", "")
    if not src:
        return {"error": "src required"}

    shared_dir = runtime._tool_context.shared_dir
    src_list = [src] if isinstance(src, str) else src
    if from_agent:
        source_dir = shared_dir if from_agent == "shared" else (runtime.agents[from_agent].workspace if from_agent in runtime.agents else None)
        if not source_dir:
            return {"error": f"Agent '{from_agent}' not found"}
        copied = _copy_files(src_list, source_dir, agent.workspace / dest if dest else agent.workspace)
        return {"pulled": copied, "from": from_agent}
    if not to:
        return {"error": "'to' or 'from_agent' required"}
    if to == "shared":
        target_dir = shared_dir
    elif to in runtime.agents:
        target_dir = runtime.agents[to].workspace
    else:
        return {"error": f"Agent '{to}' not found"}
    dest_dir = target_dir / dest if dest else target_dir
    copied = _copy_files(src_list, agent.workspace, dest_dir)
    return {"pushed": copied, "to": to}


async def meta_set_bio(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    bio = args.get("bio", "")
    if not bio:
        return {"error": "bio required"}
    agent.bio = bio
    runtime.memory.update(agent.id, public_summary=runtime.memory.serialize(agent.id).get("public_summary", ""))
    agent.memory = {"public_memory": runtime.memory.serialize(agent.id)}
    return {"bio": bio}


async def meta_get_cost(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    elapsed = time.time() - runtime._start_time
    context_pct = round(agent.context_tokens / max(1, agent.context_limit) * 100, 1)
    result = {
        "agent_id": agent.id,
        "bio": agent.bio,
        "spawned_by": agent.parent,
        "sub_agents": list(agent.children),
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
        "action_state": agent.action_state,
        "state_board": runtime.state_board_get(agent.id),
    }
    if agent.quota.time_limit > 0:
        result["time_remaining"] = round(max(0, agent.quota.time_limit - elapsed), 1)
    return result


async def meta_set_status(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    action = args.get("action")
    status = args.get("status")
    result = args.get("result", "")
    compact_before_stop = bool(args.get("compact_before_stop", False))
    compact_summary = args.get("compact_summary", "")
    current_task_tags = args.get("current_task_tags")
    work_outline = args.get("work_outline")

    if action is None:
        if status in ("done", "idle"):
            action = "stop" if status == "done" else "read"
        else:
            return {"error": "action is required unless status is 'done' or 'idle'"}
    if action not in ALLOWED_SET_STATUS_ACTIONS:
        return {"error": f"Unknown action '{action}'"}

    if current_task_tags is not None or work_outline is not None or action in {"create", "read", "message", "work", "compact", "stop"}:
        runtime.state_board_update(
            agent.id,
            action_state="stop" if action in {"done", "self_stop", "stop"} else action,
            current_task_tags=current_task_tags,
            work_outline=work_outline,
        )

    if action in {"create", "read", "message", "work", "compact"}:
        return {"status": agent.status, "action_state": agent.action_state, "state_board": runtime.state_board_get(agent.id)}

    if compact_before_stop:
        if not compact_summary:
            return {"error": "compact_summary required when compact_before_stop=true"}
        agent._compact_pending = {
            "summary": compact_summary,
            "files": args.get("files", []),
            "tags": list(current_task_tags or agent.current_task_tags),
            "experience": args.get("experience") or result or compact_summary,
            "new_task": args.get("new_task"),
            "new_bio": args.get("new_bio"),
            "work_outline": work_outline or agent.work_outline,
            "stop_after": True,
            "stop_result": result,
        }
        runtime.state_board_update(agent.id, action_state="compact", current_task_tags=current_task_tags, work_outline=work_outline)
        return {"scheduled": True, "action_state": "compact"}

    agent.action_state = "stop"
    agent.status = "done"
    if result:
        agent.result = result
    runtime.memory.update(
        agent.id,
        clear_active_task=True,
        add_experience=ExperienceCard(summary=agent.result or result or action, tags=agent.current_task_tags, artifacts=[a.path for a in agent.artifacts]),
        tags=agent.current_task_tags,
    )
    agent.memory = {"public_memory": runtime.memory.serialize(agent.id)}
    runtime.state_board_sync(agent.id)
    return {"status": agent.status, "action": action, "action_state": agent.action_state}


async def meta_rebirth(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    summary = args.get("summary", "")
    if not summary:
        return {"error": "summary required"}
    agent._rebirth_pending = {
        "summary": summary,
        "files": args.get("files", []),
        "new_task": args.get("new_task"),
        "new_bio": args.get("new_bio"),
        "tags": list(args.get("tags", agent.current_task_tags)),
        "experience": args.get("experience"),
    }
    runtime.state_board_update(agent.id, action_state="compact", current_task_tags=args.get("tags"), work_outline=args.get("work_outline"))
    return {"scheduled": True, "note": "Context resets next turn."}


async def meta_compact(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    summary = args.get("summary", "")
    if not summary:
        return {"error": "summary required"}
    agent._compact_pending = {
        "summary": summary,
        "files": args.get("files", []),
        "tags": list(args.get("tags", agent.current_task_tags)),
        "experience": args.get("experience"),
        "new_task": args.get("new_task"),
        "new_bio": args.get("new_bio"),
        "work_outline": args.get("work_outline", agent.work_outline),
        "stop_after": False,
    }
    runtime.state_board_update(agent.id, action_state="compact", current_task_tags=args.get("tags"), work_outline=args.get("work_outline"))
    return {"scheduled": True, "action_state": "compact"}


async def meta_submit(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    from nanoma.core import Artifact

    path_str = args.get("path", "")
    if not path_str:
        return {"error": "path required"}
    path = Path(path_str)
    if not path.is_absolute():
        path = agent.workspace / path
    if not path.exists():
        return {"error": f"Not found: {path}"}

    shared = runtime._tool_context.shared_dir
    shared.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, shared / path.name)

    artifact = Artifact(path=path_str, absolute_path=path, description=args.get("description", ""), agent_id=agent.id)
    agent.artifacts.append(artifact)
    runtime.memory.update(agent.id, add_artifacts=[path_str])
    agent.memory = {"public_memory": runtime.memory.serialize(agent.id)}
    return {"submitted": path_str, "shared_copy": str(shared / path.name)}


async def meta_batch(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
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
    all_tools = {**WORK_TOOLS, **META_TOOLS}
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


META_TOOLS: dict[str, dict[str, Any]] = {
    "spawn": {"handler": meta_spawn, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "spawn",
        "description": "Create a new agent that starts immediately and runs in parallel. Optional peer-first metadata fields are role/create_type/relationship/group_id/workflow_prior. created_by is runtime-populated when create_type='peer_agent'. workflow_prior is guidance, not a workflow engine.",
        "parameters": {"type": "object", "properties": {
            "task": {"type": "string"},
            "model": {"type": "string"},
            "delegate": {"type": "boolean", "default": False},
            "role": {"type": "string"},
            "create_type": {"type": "string"},
            "relationship": {"type": "string"},
            "group_id": {"type": "string"},
            "workflow_prior": {"type": "string"},
            "current_task_tags": {"type": "array", "items": {"type": "string"}},
        }, "required": ["task"]},
    }}} ,
    "create_agent": {"handler": meta_create_agent, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "create_agent",
        "description": "Alias of spawn with the same peer-first creation metadata; created_by is runtime-populated for peer_agent.",
        "parameters": {"type": "object", "properties": {
            "task": {"type": "string"},
            "model": {"type": "string"},
            "delegate": {"type": "boolean", "default": False},
            "role": {"type": "string"},
            "create_type": {"type": "string"},
            "relationship": {"type": "string"},
            "group_id": {"type": "string"},
            "workflow_prior": {"type": "string"},
            "current_task_tags": {"type": "array", "items": {"type": "string"}},
        }, "required": ["task"]},
    }}} ,
    "kill": {"handler": meta_kill, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "kill",
        "description": "Emergency-only terminate an agent. Only self or descendants may be killed, and only when emergency=true.",
        "parameters": {"type": "object", "properties": {
            "agent_id": {"type": "string"},
            "emergency": {"type": "boolean", "default": False},
        }, "required": ["agent_id"]},
    }}} ,
    "send": {"handler": meta_send, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "send",
        "description": "Send a message to one or more agents. Supports structured messaging with message_type/payload/requires_ack/urgency while remaining backward-compatible with plain text. Use stop_request as a structured message type for stop coordination.",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string"},
            "message": {"type": "string"},
            "mode": {"type": "string", "enum": ["immediate", "steer", "queue"], "default": "queue"},
            "message_type": {"type": "string", "default": "text"},
            "payload": {"type": "object"},
            "requires_ack": {"type": "boolean", "default": False},
            "urgency": {"type": "string", "default": "normal"},
        }, "required": ["to"]},
    }}} ,
    "query": {"handler": meta_query, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "query",
        "description": "Query one agent or list all agents. Returns action_state/state_board and structured public_memory in addition to identity and artifacts.",
        "parameters": {"type": "object", "properties": {
            "agent_id": {"type": "string"},
            "messages": {"type": "integer", "default": 0},
        }},
    }}} ,
    "wait": {"handler": meta_wait, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "wait",
        "description": "Block execution until agents finish.",
        "parameters": {"type": "object", "properties": {
            "agent_ids": {"type": "array", "items": {"type": "string"}},
            "timeout": {"type": "number", "default": 120},
            "mode": {"type": "string", "enum": ["all", "any"], "default": "all"},
        }},
    }}} ,
    "transfer": {"handler": meta_transfer, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "transfer",
        "description": "Copy files between agent workspaces.",
        "parameters": {"type": "object", "properties": {
            "src": {"type": "string"},
            "to": {"type": "string"},
            "from_agent": {"type": "string"},
            "dest": {"type": "string"},
        }, "required": ["src"]},
    }}} ,
    "set_bio": {"handler": meta_set_bio, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "set_bio",
        "description": "Set your self-description.",
        "parameters": {"type": "object", "properties": {
            "bio": {"type": "string"},
        }, "required": ["bio"]},
    }}} ,
    "get_cost": {"handler": meta_get_cost, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "get_cost",
        "description": "Get identity and resource status, including action_state and state_board.",
        "parameters": {"type": "object", "properties": {}},
    }}} ,
    "set_status": {"handler": meta_set_status, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "set_status",
        "description": "Update action_state through the state board or stop/self-stop. Unknown actions are errors. compact_before_stop can compact first when compact_summary is provided.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string"},
            "status": {"type": "string"},
            "result": {"type": "string"},
            "current_task_tags": {"type": "array", "items": {"type": "string"}},
            "work_outline": {"type": "string"},
            "compact_before_stop": {"type": "boolean", "default": False},
            "compact_summary": {"type": "string"},
            "files": {"type": "array", "items": {"type": "string"}},
            "experience": {"type": "string"},
            "new_task": {"type": "string"},
            "new_bio": {"type": "string"},
        }},
    }}} ,
    "rebirth": {"handler": meta_rebirth, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "rebirth",
        "description": "Reset your context window and sync structured public memory.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string"},
            "files": {"type": "array", "items": {"type": "string"}},
            "new_task": {"type": "string"},
            "new_bio": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "experience": {"type": "string"},
            "work_outline": {"type": "string"},
        }, "required": ["summary"]},
    }}} ,
    "compact": {"handler": meta_compact, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "compact",
        "description": "Immediately compact live history into a compact summary and update public memory. summary is required; files/tags/experience/new_task/new_bio/work_outline are optional.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string"},
            "files": {"type": "array", "items": {"type": "string"}},
            "tags": {"type": "array", "items": {"type": "string"}},
            "experience": {"type": "string"},
            "new_task": {"type": "string"},
            "new_bio": {"type": "string"},
            "work_outline": {"type": "string"},
        }, "required": ["summary"]},
    }}} ,
    "submit": {"handler": meta_submit, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "submit",
        "description": "Mark a file as a deliverable and update artifact index in public memory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "description": {"type": "string"},
        }, "required": ["path"]},
    }}} ,
    "batch": {"handler": meta_batch, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "batch",
        "description": "Execute multiple tool calls from a JSON file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
        }, "required": ["path"]},
    }}} ,
}
