"""Meta-tools: spawn, kill, send, query, wait, delivery, lifecycle, and artifacts."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from nanoma.core import Agent, Runtime


def _looks_like_question_restatement(value: str) -> bool:
    candidate = str(value or "").strip()
    return bool(
        not re.search(r"\d", candidate)
        and re.search(
            r"(?:^|\b(?:final\s+)?answer\s*:\s*)"
            r"the\s+(?:absolute\s+)?difference\s+(?:in|between)\b.*\bbetween\b",
            candidate,
            flags=re.IGNORECASE,
        )
    )


def _extract_candidate_answer(text: str) -> str:
    """Recover an answer-only value from a legacy free-text handoff."""
    value = str(text or "").strip()
    if not value:
        return ""
    patterns = (
        r"final\s+answer\s*[:=\-]\s*[`\"']?([^\n`\"']+)",
        r"[\"']answer[\"']\s*:\s*[\"']([^\"']+)",
        r"(?m)^\s*answer\s*:\s*[`\"']?([^\n`\"']+)",
        r"(?:ball\s+with\s+highest[^\n:]{0,100}|most\s+likely\s+ball|best\s+ball)\s*:\s*(?:ball\s*)?([-+]?\d+(?:\.\d+)?)",
        r"(?:winner|winning\s+ball)\s*:\s*(?:ball\s*)?([-+]?\d+(?:\.\d+)?)",
        r"found\s+the\s+answer\s*:\s*(?:the\s+\w+\s+is\s+)?[`\"']?([^\n`\"'.]+)",
        r"answer\s+(?:is|should\s+be)\s*[:=\-]?\s*[`\"']?([^\n`\"'.]+)",
        r"(?:it\s+is|it(?:'|’)s)\s+([-+]?\d+(?:\.\d+)?)\s+clicks?\b",
        r"(?:duration|time\s+span)[^\n]{0,120}?(?:of|is|=)\s+(?:approximately\s+)?([-+]?\d+(?:\.\d+)?)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if not match:
            continue
        answer = re.split(r"\\[nrt]", match.group(1), maxsplit=1)[0]
        answer = answer.strip().rstrip(" .;,:)")
        compact_number = re.fullmatch(
            r"(?:ball|number|page\s+links?)\s*#?\s*([-+]?\d+(?:\.\d+)?)",
            answer,
            flags=re.IGNORECASE,
        )
        if compact_number:
            answer = compact_number.group(1)
        if _looks_like_question_restatement(answer):
            continue
        if answer:
            return answer
    if "\n" not in value and len(value) <= 160:
        fallback = value.strip(" `\"'")
        if _looks_like_question_restatement(fallback):
            return ""
        return fallback
    return ""


# ─── spawn ───────────────────────────────────────────────────────────────────

_SPAWN_TASK_KEYS = (
    "task", "assignment", "instruction", "prompt",
    "message", "description", "unit", "query",
)


def _spawn_task_from_args(args: dict[str, Any]) -> str:
    """Accept common task-description aliases from less strict tool callers."""
    for key in _SPAWN_TASK_KEYS:
        if key not in args:
            continue
        value = args.get(key)
        if value:
            return str(value).strip()
    return ""


def _strip_root_only_spawn_override(task: str) -> tuple[str, bool]:
    """Remove a force-spawn instruction that a root copied into a child task."""
    cleaned, count = re.subn(
        r"(?:^|\n+)\[Probe branch tool override\]\s*.*?Do not call shell\.\s*",
        "\n\n",
        str(task or ""),
        flags=re.IGNORECASE | re.DOTALL,
    )
    return cleaned.strip(), bool(count)

def _inherit_forced_spawn_scope(
    task: str,
    agent: "Agent",
    runtime: "Runtime",
) -> tuple[str, int]:
    """Preserve an explicit per-child evidence count when the root narrows an assignment."""
    if (
        not runtime.config.force_spawn_many
        or agent.parent is not None
    ):
        return task, 0
    inherited = str(runtime.config.system_extra_instructions or "")
    if not re.search(
        r"\b(?:each|every|both)\s+(?:spawned\s+)?(?:child|children|agents?)\b.{0,240}"
        r"\b(?:independently|each|produce|deliver|complete)\b",
        inherited,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        return task, 0
    required_items = runtime._explicit_expected_evidence_count(inherited)
    assigned_items = runtime._explicit_expected_evidence_count(task)
    if required_items <= 1 or assigned_items >= required_items:
        return task, 0
    scope_contract = (
        "\n\n[RUNTIME INHERITED SCOPE CONTRACT]\n"
        f"The supervisor explicitly requires every child to independently produce {required_items} "
        "evidence rows/items. Do not reduce this assignment to metadata-only, one-side research, or a "
        "single row. Complete all required items end to end, include the measurements or calculations "
        "needed to derive an answer candidate, and only then deliver to the parent."
    )
    return task.rstrip() + scope_contract, required_items


def _inherit_spawn_research_protocol(
    task: str,
    agent: "Agent",
    runtime: "Runtime",
) -> tuple[str, bool]:
    """Append an explicitly marked supervisor protocol to root-spawned child tasks."""
    if agent.parent is not None:
        return task, False
    inherited = str(runtime.config.system_extra_instructions or "")
    match = re.search(
        r"\[CHILD RESEARCH PROTOCOL\]\s*(.*?)\s*\[/CHILD RESEARCH PROTOCOL\]",
        inherited,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return task, False
    protocol = match.group(1).strip()
    if not protocol or protocol in task:
        return task, False
    protocol_contract = (
        "\n\n[RUNTIME INHERITED RESEARCH PROTOCOL]\n"
        "This protocol was supplied by the supervisor for every research child. Follow it in addition "
        "to the assignment above; the assignment does not override or abbreviate it.\n"
        + protocol
    )
    return task.rstrip() + protocol_contract, True


def _rewrite_conflicting_arxiv_identifier_month_filter(
    task: str,
    agent: "Agent",
    runtime: "Runtime",
) -> tuple[str, bool]:
    """Remove a child assignment that contradicts an inherited version-history protocol."""
    if agent.parent is not None:
        return task, False
    inherited = str(runtime.config.system_extra_instructions or "")
    if not (
        "identifier prefix records v1 only" in inherited.lower()
        and "version" in inherited.lower()
    ):
        return task, False
    rewritten, count = re.subn(
        r"\bfilter\s+(?:the\s+)?(?:ids?|results?|candidates?)?\s*(?:for|by)\s+"
        r"[^.\r\n]{0,120}?\(?\s*yymm\s*=\s*\d{4}\s*\)?",
        "inspect every candidate's version history for the requested month "
        "(do not filter by identifier YYMM)",
        str(task or ""),
        flags=re.IGNORECASE,
    )
    return rewritten, bool(count)

async def meta_spawn(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    """Create a child agent."""
    from nanoma.core import ResourceQuota

    task = _spawn_task_from_args(args)
    original_task = task
    original_task_preview = original_task[:200]
    requested_model = args.get("model")
    model = requested_model
    delegate = args.get("delegate", False)

    task, stripped_root_override = _strip_root_only_spawn_override(task)
    if stripped_root_override:
        runtime._emit(agent.id, "spawn_root_override_stripped", {
            "original_task_preview": original_task_preview,
            "clean_task_preview": task[:200],
        })
    if not task:
        return {"error": "task is required"}
    task, identifier_filter_rewritten = _rewrite_conflicting_arxiv_identifier_month_filter(
        task,
        agent,
        runtime,
    )
    if identifier_filter_rewritten:
        runtime._emit(agent.id, "spawn_identifier_month_filter_rewritten", {
            "original_task_preview": original_task_preview,
            "clean_task_preview": task[:200],
        })
    task, inherited_items = _inherit_forced_spawn_scope(task, agent, runtime)
    if inherited_items:
        runtime._emit(agent.id, "spawn_scope_inherited", {
            "required_items": inherited_items,
            "original_task_preview": original_task_preview,
        })
    task, inherited_protocol = _inherit_spawn_research_protocol(task, agent, runtime)
    if inherited_protocol:
        runtime._emit(agent.id, "spawn_protocol_inherited", {
            "protocol_chars": len(task) - len(original_task),
            "original_task_preview": original_task_preview,
        })
    if agent.depth + 1 > runtime.config.max_depth:
        return {"error": f"Max depth ({runtime.config.max_depth}) exceeded"}
    if len(runtime.agents) >= runtime.config.max_agents:
        return {"error": f"Max agents ({runtime.config.max_agents}) reached"}
    memory_block = runtime._spawn_memory_block_reason()
    if memory_block:
        runtime._emit(agent.id, "spawn_blocked_memory", {
            "reason": memory_block,
            "active_children": runtime._active_children_total(),
        })
        return {"error": f"spawn blocked by {memory_block}"}
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

    # Model selection. Supervisor-authorized spawn is a topology intervention,
    # not a model-routing intervention; keep worker execution on the runtime default.
    if agent.id in runtime._strategy_spawn_authorized:
        model = runtime.config.default_model
    elif not model:
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
    runtime._emit(agent.id, "spawn", {
        "child": child.id,
        "task": task[:100],
        "model": model,
        "requested_model": requested_model,
    })

    if delegate:
        agent.status = "done"
        agent.result = f"[Delegated to {child.id}]"

    return {"agent_id": child.id, "model": model, "requested_model": requested_model}


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

    grace_applies = bool(
        runtime.config.child_kill_delivery_grace_enabled
        and target_id != agent.id
        and target.status in {"running", "idle"}
        and runtime._child_has_material_delivery_evidence(target)
        and not runtime._child_has_formal_delivery(target)
    )
    if grace_applies:
        marker_prefix = f"kill_delivery_grace:{agent.id}:"
        marker = next(
            (
                str(item) for item in target._notified_thresholds
                if str(item).startswith(marker_prefix)
            ),
            "",
        )
        requested_turn = target._turns
        if marker:
            try:
                requested_turn = int(marker.removeprefix(marker_prefix))
            except ValueError:
                requested_turn = target._turns
        if not marker:
            from nanoma.core import Envelope
            from nanoma.llm import estimate_tokens

            target._notified_thresholds.add(f"{marker_prefix}{requested_turn}")
            message = (
                "[Runtime delivery grace] Stop new research now. You already have material evidence; "
                "summarize the best answer candidate and call deliver_to_parent on this turn. Do not "
                "start another search or merely report that work is complete."
            )
            await runtime.deliver(Envelope(
                from_id=agent.id,
                to_id=target.id,
                content=message,
                tokens=estimate_tokens(message),
                timestamp=time.time(),
                mode="steer",
            ))
            runtime._emit(agent.id, "child_kill_delivery_grace", {
                "child": target.id,
                "requested_turn": requested_turn,
                "candidate_like_outputs": target._shell_activity.candidate_like_outputs,
                "material_gain_calls": target._shell_activity.material_gain_calls,
            })
        force = bool(args.get("force", False))
        if not force or target._turns <= requested_turn:
            return {
                "error": (
                    "kill deferred: child has material evidence but has not delivered it. "
                    "Wait/query for deliver_to_parent. Only after the child completes another turn may "
                    "you retry with force=true if delivery is still impossible."
                ),
                "deferred": target_id,
                "requested_turn": requested_turn,
                "current_turn": target._turns,
                "required_behavior": ["wait", "query"],
            }

    target.status = "done"
    if target._task and not target._task.done() and target._task is not asyncio.current_task():
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


# ─── deliver_to_parent ──────────────────────────────────────────────────────

async def meta_deliver_to_parent(
    args: dict[str, Any], agent: "Agent", runtime: "Runtime"
) -> dict[str, Any]:
    """Deliver a structured candidate to the parent and persist it in the runtime ledger."""
    from nanoma.core import Envelope
    from nanoma.llm import estimate_tokens

    if not agent.parent or agent.parent not in runtime.agents:
        return {"error": "deliver_to_parent is only available to an agent with a live parent"}

    answer = str(args.get("answer", "") or "").strip()
    evidence = str(args.get("evidence", "") or "").strip()
    method = str(args.get("method", "") or "").strip()
    if not answer:
        return {"error": "'answer' required; provide the concise answer-only candidate"}
    fixed_spec = runtime._fixed_agent_specs.get(agent.id)
    answer_pattern = (
        str(fixed_spec.delivery_answer_regex or "").strip()
        if fixed_spec is not None
        else ""
    )
    if answer_pattern and re.fullmatch(answer_pattern, answer) is None:
        return {
            "error": "answer does not match this fixed role's delivery contract",
            "required_answer_pattern": answer_pattern,
            "required_behavior": (
                "Deliver only the intermediate or final answer shape assigned to this "
                "role; keep downstream conclusions in evidence rather than answer."
            ),
        }

    try:
        confidence = float(args.get("confidence", 0.7))
    except (TypeError, ValueError):
        return {"error": "'confidence' must be a number between 0 and 1"}
    confidence = max(0.0, min(1.0, confidence))

    record = runtime._record_candidate_delivery(
        agent,
        parent_id=agent.parent,
        answer=answer,
        evidence=evidence,
        confidence=confidence,
        method=method,
        source="deliver_to_parent",
    )
    agent.result = answer
    message = (
        f"[Candidate delivery from {agent.id}]\n"
        f"answer: {answer}\n"
        f"confidence: {confidence:.2f}\n"
        f"method: {method or 'unspecified'}\n"
        f"evidence: {evidence or 'not supplied'}"
    )
    tokens = estimate_tokens(message)
    await runtime.deliver(Envelope(
        from_id=agent.id,
        to_id=agent.parent,
        content=message,
        tokens=tokens,
        timestamp=time.time(),
        mode="steer",
    ))
    runtime._emit(agent.id, "candidate_delivered", {
        "to": agent.parent,
        "answer": answer[:300],
        "confidence": confidence,
        "method": method[:120],
        "evidence": evidence[:500],
        "record_seq": record.get("seq") if record else None,
    })
    auto_completed = bool(
        runtime.config.child_delivery_auto_complete
        and agent.status == "running"
    )
    if auto_completed:
        agent.status = "done"
        runtime._emit(agent.id, "candidate_delivery_auto_completed", {
            "to": agent.parent,
            "answer": answer[:300],
            "record_seq": record.get("seq") if record else None,
        })
    return {
        "delivered": True,
        "to": agent.parent,
        "answer": answer,
        "confidence": confidence,
        "record_seq": record.get("seq") if record else None,
        "agent_completed": auto_completed,
    }


# ─── query ───────────────────────────────────────────────────────────────────

async def meta_query(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    """Query agents. messages=N for last N messages, messages=-1 for all."""
    target_id = args.get("agent_id")
    messages_n = args.get("messages", 0)  # 0=meta only, N=last N, -1=all

    if isinstance(messages_n, str):
        stripped = messages_n.strip()
        if re.fullmatch(r"-?\d+", stripped):
            messages_n = int(stripped)
        elif target_id == agent.parent and stripped:
            recovered_answer = _extract_candidate_answer(stripped)
            if recovered_answer:
                result = await meta_deliver_to_parent({
                    "answer": recovered_answer,
                    "evidence": stripped,
                    "confidence": 0.65,
                    "method": "recovered legacy query.messages handoff",
                }, agent, runtime)
                result["recovered_from"] = "query.messages"
                result["warning"] = (
                    "query.messages accepts an integer; the text payload was recovered as a "
                    "parent candidate. Use deliver_to_parent next time."
                )
                return result
            send_result = await meta_send({
                "to": agent.parent,
                "message": (
                    "[Recovered legacy query.messages report; no answer-only value was parsed]\n"
                    + stripped
                ),
                "mode": "steer",
            }, agent, runtime)
            agent.result = stripped
            return {
                **send_result,
                "recovered_from": "query.messages",
                "warning": (
                    "query.messages accepts an integer; the report was forwarded to the parent. "
                    "Use deliver_to_parent(answer=..., evidence=...) next time."
                ),
            }
        else:
            return {
                "error": "'messages' must be an integer count, not message text",
                "required_behavior": (
                    "Use send(to=..., message=...) for peer communication or "
                    "deliver_to_parent(answer=..., evidence=...) for final child results."
                ),
            }
    if isinstance(messages_n, list) and target_id:
        contents = []
        for item in messages_n:
            if isinstance(item, dict):
                content = str(item.get("content", "") or "").strip()
            else:
                content = str(item or "").strip()
            if content:
                contents.append(content)
        if contents:
            result = await meta_send({
                "to": str(target_id),
                "message": "\n".join(contents),
                "mode": "steer",
            }, agent, runtime)
            return {
                **result,
                "recovered_from": "query.messages_list",
                "warning": (
                    "query.messages accepts an integer; the message list was routed through send. "
                    "Use send(to=..., message=...) for peer communication."
                ),
            }
        return {"error": "'messages' list did not contain any non-empty content"}
    if isinstance(messages_n, bool) or not isinstance(messages_n, int):
        return {
            "error": "'messages' must be an integer count (0, a positive count, or -1)",
            "received_type": type(messages_n).__name__,
        }

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
    timeout_raw = args.get("timeout", 120.0)
    if isinstance(timeout_raw, str):
        match = re.search(r"\d+(?:\.\d+)?", timeout_raw)
        timeout = float(match.group(0)) if match else 120.0
    else:
        try:
            timeout = float(timeout_raw)
        except (TypeError, ValueError):
            timeout = 120.0
    mode = args.get("mode", "all")  # "all" or "any"

    if mode not in ("all", "any"):
        return {"error": "mode must be 'all' or 'any'"}

    if not target_ids:
        target_ids = list(agent.children)
    else:
        target_ids = [str(tid) for tid in target_ids if str(tid).strip()]
        existing = [tid for tid in target_ids if tid in runtime.agents]
        missing = [tid for tid in target_ids if tid not in runtime.agents]
        if missing and agent.children:
            target_ids = list(agent.children)
            mode = "any"
    if not target_ids:
        return {"completed": [], "pending": [], "note": "Nothing to wait for"}

    # Cap timeout vs time limit
    if agent.quota.time_limit > 0:
        remaining = agent.quota.time_limit - runtime.effective_elapsed()
        timeout = min(timeout, max(0, remaining * 0.9))  # leave 10% margin for cleanup

    completed = []
    interrupted = False
    reason = None

    try:
        async with asyncio.timeout(timeout):
            while True:
                if runtime._wait_is_interrupted(agent, target_ids):
                    interrupted = True
                    reason = "message_received"
                    break
                if agent.parent is None and runtime._harvest_child_candidates(agent):
                    interrupted = True
                    reason = "candidate_recovered"
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
    """Transfer files between agents. Push or Pull; plain copy or diff-merge."""
    src = args.get("src", "")
    to = args.get("to", "")
    from_agent = args.get("from_agent", "")
    dest = args.get("dest", "")
    mode = str(args.get("mode", "copy") or "copy").lower()

    if not src:
        return {"error": "src required"}

    shared_dir = runtime._tool_context.shared_dir
    src_list = [src] if isinstance(src, str) else src

    if mode == "merge":
        return await _transfer_merge(
            args, agent, runtime, src_list, to, from_agent, dest, shared_dir
        )

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


def _build_uniqueness_reminder(
    from_who: str,
    changed: list[str],
    deleted: list[str],
    overlap: dict[str, list[str]],
    active_others: list[str],
) -> str:
    lines = [
        f"[merge] Received a diff from '{from_who}': "
        f"{len(changed)} file(s) changed, {len(deleted)} deleted."
    ]
    if changed:
        shown = ", ".join(changed[:20]) + (" …" if len(changed) > 20 else "")
        lines.append(f"  changed: {shown}")
    if deleted:
        shown = ", ".join(deleted[:20]) + (" …" if len(deleted) > 20 else "")
        lines.append(f"  deleted: {shown}")
    if overlap:
        who = "; ".join(f"{a} ({len(fs)} shared file(s))" for a, fs in overlap.items())
        lines.append(
            f"⚠ Other ACTIVE agents are editing SOME OF THE SAME files and have NOT delivered yet: "
            f"{who}. Before you rely on or finalize this merge, query them "
            f"(query(agent_id='<id>', messages=-1)) to confirm this contribution is unique and will "
            f"not clobber / be clobbered by their concurrent work; reconcile if it is not."
        )
    elif active_others:
        lines.append(
            "Other agents are still working: "
            + ", ".join(active_others)
            + ". Consider query-ing them to check none is producing an overlapping, "
            "undelivered diff before you finalize."
        )
    else:
        lines.append("No other active contributors detected — this diff appears unique.")
    return "\n".join(lines)


async def _transfer_merge(
    args: dict[str, Any],
    agent: "Agent",
    runtime: "Runtime",
    src_list: list[str],
    to: str,
    from_agent: str,
    dest: str,
    shared_dir: Path,
) -> dict[str, Any]:
    """Apply the DIFF of a source tree onto a destination tree (union-merge),
    then remind the receiver to verify the diff is unique among live agents."""
    from nanoma.core import Envelope
    from nanoma.llm import estimate_tokens

    base = str(args.get("base", "") or "")

    # Resolve source-side owner and destination tree.
    receiver_id: str | None
    target_agent = None
    if from_agent:  # pull: source is another agent, destination is caller
        owner = shared_dir if from_agent == "shared" else (
            runtime.agents[from_agent].workspace if from_agent in runtime.agents else None)
        if owner is None:
            return {"error": f"Agent '{from_agent}' not found"}
        dest_dir = (agent.workspace / dest) if dest else agent.workspace
        receiver_id = agent.id
        from_who = from_agent
    else:  # push: source is caller, destination is another agent / shared
        if not to:
            return {"error": "'to' or 'from_agent' required"}
        owner = agent.workspace
        if to == "shared":
            dest_dir = (shared_dir / dest) if dest else shared_dir
            receiver_id = None
        elif to in runtime.agents:
            target_agent = runtime.agents[to]
            dest_dir = (target_agent.workspace / dest) if dest else target_agent.workspace
            receiver_id = to
        else:
            return {"error": f"Agent '{to}' not found"}
        from_who = agent.id

    if len(src_list) != 1:
        return {"error": "merge mode requires exactly one source directory (not a glob/list)"}
    src_dir = Path(src_list[0]) if Path(src_list[0]).is_absolute() else (owner / src_list[0])
    if not src_dir.is_dir():
        return {"error": f"merge source must be a directory: {src_dir}"}

    # Resolve the ancestor to diff against (optional).
    base_dir: Path | None = None
    if base:
        if base == "baseline":
            base_dir = getattr(runtime, "_merge_baseline_path", None)
        elif Path(base).is_absolute():
            base_dir = Path(base)
        else:
            base_dir = owner / base
        if base_dir is None or not Path(base_dir).is_dir():
            return {"error": f"merge base not found: {base!r}"}

    dest_dir.mkdir(parents=True, exist_ok=True)
    if base_dir is not None:
        # True diff vs common ancestor: adds/modifies AND deletions.
        changed, deleted = runtime._merge_diff(src_dir, Path(base_dir))
    else:
        # No ancestor: only carry files that differ from the destination; never
        # delete destination files (source may be a partial contribution).
        changed, _ = runtime._merge_diff(src_dir, dest_dir)
        deleted = set()

    if not changed and not deleted:
        return {"merged": [], "deleted": [], "note": "no differing files to merge"}

    runtime._merge_apply(changed, deleted, dest_dir)
    changed_files = sorted(changed.keys())
    deleted_files = sorted(deleted)

    # Uniqueness / concurrency check against still-running agents.
    exclude = {agent.id, receiver_id, from_agent, "shared"}
    touched = set(changed_files) | set(deleted_files)
    overlap = runtime._agents_touching_files(touched, exclude_ids=exclude)
    active_others = [a for a in runtime._active_contributor_ids(exclude_ids=exclude) if a not in overlap]
    reminder = _build_uniqueness_reminder(from_who, changed_files, deleted_files, overlap, active_others)

    # Push to a live agent: nudge the receiver in-band.
    if target_agent is not None and receiver_id and receiver_id != agent.id:
        await runtime.deliver(Envelope(
            from_id="system", to_id=receiver_id, content=reminder,
            tokens=estimate_tokens(reminder), timestamp=time.time(), mode="steer",
        ))

    result: dict[str, Any] = {
        "merged": changed_files,
        "deleted": deleted_files,
        "dest": str(dest_dir),
        "base": str(base_dir) if base_dir else "(destination)",
        "uniqueness_check": reminder,
    }
    if overlap:
        result["overlapping_agents"] = {a: fs for a, fs in overlap.items()}
    if active_others:
        result["other_active_agents"] = active_others
    return result


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
    elapsed = runtime.effective_elapsed()
    context_pct = round(agent.context_tokens / max(1, agent.context_limit) * 100, 1)
    coordination_tools = {"spawn", "send", "deliver_to_parent", "wait", "query", "kill", "transfer", "set_bio"}
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
    if result is not None and result != "":
        agent.result = str(result)
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
        for name, tool in {
            **WORK_TOOLS,
            **WORKSPACE_TOOLS,
            **META_TOOLS,
            **runtime.config.extra_tools,
        }.items()
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
    "kill": {"handler": meta_kill, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "kill",
        "description": "Terminate an agent. A descendant with material evidence receives a delivery grace turn before termination.",
        "parameters": {"type": "object", "properties": {
            "agent_id": {"type": "string", "description": "ID of the agent to terminate"},
            "force": {"type": "boolean", "description": "After a delivery grace turn, terminate even if no formal handoff arrived", "default": False},
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
    "deliver_to_parent": {"handler": meta_deliver_to_parent, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "deliver_to_parent",
        "description": "Deliver a completed answer candidate to your parent through a durable structured channel. Use this exactly once when your evidence supports an answer, then call set_status(done, result=<same answer>). Do not use query or send for final result delivery.",
        "parameters": {"type": "object", "properties": {
            "answer": {"type": "string", "description": "Concise answer-only candidate expected by the task"},
            "evidence": {"type": "string", "description": "Brief decisive evidence or calculation supporting the answer"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1, "description": "Confidence from 0 to 1", "default": 0.7},
            "method": {"type": "string", "description": "Short method label, such as independent calculation, source lookup, or elimination"},
        }, "required": ["answer"]},
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
        "description": (
            "Move work between agent workspaces. Two modes:\n"
            "• mode='copy' (default): copy files/globs. Push: transfer(src=file, to=agent_id); "
            "Pull: transfer(src=file, from_agent=agent_id). Use to='shared'/from_agent='shared' for the shared dir.\n"
            "• mode='merge': treat src as a DIRECTORY and apply only its DIFF onto the destination tree "
            "(adds/modifies changed files; also deletes files if you pass base=<ancestor dir> or base='baseline'). "
            "This folds one agent's changes into another's tree without clobbering unrelated files. "
            "The receiver is automatically reminded to verify the diff is unique — i.e. to query whether any "
            "other still-running agent is editing the same files but has not delivered yet — and the result "
            "lists any overlapping/active agents."
        ),
        "parameters": {"type": "object", "properties": {
            "src": {"type": "string", "description": "Source file/glob (copy mode) or a single source DIRECTORY (merge mode)"},
            "to": {"type": "string", "description": "Push destination: agent_id or 'shared'"},
            "from_agent": {"type": "string", "description": "Pull source: agent_id or 'shared'"},
            "dest": {"type": "string", "description": "Subdirectory at destination to place files into"},
            "mode": {"type": "string", "enum": ["copy", "merge"], "description": "'copy' (default) = plain file copy; 'merge' = apply the diff of a source directory onto the destination tree", "default": "copy"},
            "base": {"type": "string", "description": "merge mode only: ancestor to diff against — a directory path or 'baseline' (the pre-fan-out snapshot). If omitted, only files that differ from the destination are carried and nothing is deleted."},
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
        "description": "Change your lifecycle status. Child agents should first call deliver_to_parent with their answer and evidence, then call done with the same answer in result. 'idle' pauses until a message arrives.",
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


# ─── Planning and optional add-on tools ──────────────────────────────────────
# Per-node planning is a built-in NanoMA capability and is always available in
# installed packages. Evaluation-oriented verification and delivery helpers
# remain optional add-ons. Any of them can be disabled with
# RuntimeConfig.disabled_tools.

def _register_optimization_tools() -> None:
    import os
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from nanoma.planning import TODO_TOOLS

    for name, tool in TODO_TOOLS.items():
        META_TOOLS.setdefault(name, tool)

    for module, attribute in (
        ("optimizations.verify_tool", "VERIFY_TOOLS"),
        ("optimizations.experiment_ledger", "LEDGER_TOOLS"),
        ("optimizations.merge_submit", "DELIVERY_TOOLS"),
    ):
        try:
            tools = getattr(__import__(module, fromlist=[attribute]), attribute)
        except Exception:
            continue
        for name, tool in tools.items():
            META_TOOLS.setdefault(name, tool)


_register_optimization_tools()
