"""Meta-tools: spawn, kill, send, query, wait, transfer, set_bio, get_cost, set_status, rebirth, compact, submit, submit_answer, batch."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

from nanoma.memory import build_memory_card, normalize_tags

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
    expected_outputs = runtime.expected_outputs(target)
    missing_outputs = runtime.missing_expected_outputs(target)
    answer_required = runtime.answer_submission_required(target)
    answer_submitted = bool(target.submitted_answer_path)
    progress = {
        "turns": target._turns,
        "tool_calls": target._tool_calls,
        "artifacts_count": len(target.artifacts),
        "expected_outputs": expected_outputs[:20],
        "missing_outputs": missing_outputs[:20],
        "outputs_complete": runtime.outputs_complete(target),
        "answer_submission_required": answer_required,
        "answer_submitted": answer_submitted,
        "submitted_answer_path": target.submitted_answer_path,
        "completion_score": _completion_score(target, expected_outputs, missing_outputs),
    }
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
        "progress": progress,
        "state_board": runtime.state_board_get(target.id),
        "task_ledger": runtime.task_ledger_item_snapshot(target),
        "public_memory": runtime.memory.serialize(target.id),
        "memory": target.memory,
    }


def _completion_score(target: "Agent", expected_outputs: list[str], missing_outputs: list[str]) -> float:
    score = 0.0
    if target.status == "done":
        score += 3.0
    elif target.status in {"running", "idle"}:
        score += 1.0
    score += min(len(target.artifacts), 5) * 0.5
    if expected_outputs:
        score += (len(expected_outputs) - len(missing_outputs)) / max(1, len(expected_outputs)) * 3.0
    score += min(target._turns, 20) * 0.05
    return round(score, 3)


def _normalize_string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _parse_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        try:
            return json.loads(stripped, strict=False)
        except json.JSONDecodeError:
            return value


def _recover_raw_tool_args(args: dict[str, Any]) -> dict[str, Any]:
    raw = args.get("_raw") if isinstance(args, dict) else None
    if not raw:
        return args
    parsed = _parse_json_string(raw)
    if not isinstance(parsed, dict):
        return args
    recovered = dict(parsed)
    for key, value in args.items():
        if key != "_raw" and key not in recovered:
            recovered[key] = value
    return recovered


def _parse_spawn_many_payload(value: Any) -> tuple[Any, dict[str, Any]]:
    parsed = _parse_json_string(value)
    if not isinstance(parsed, str):
        return parsed, {}
    stripped = parsed.strip()
    if not stripped.startswith("["):
        return parsed, {}
    decoder = json.JSONDecoder()
    try:
        items, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
        items = _parse_json_array_prefix(stripped)
        return (items, {}) if items else (parsed, {})
    extras: dict[str, Any] = {}
    tail = stripped[end:].strip()
    if tail.startswith(","):
        tail = tail[1:].strip()
    if tail:
        try:
            tail_obj = json.loads("{" + tail.rstrip(",") + "}")
            if isinstance(tail_obj, dict):
                defaults = tail_obj.get("defaults")
                if isinstance(defaults, dict):
                    extras = defaults
        except json.JSONDecodeError:
            pass
    return items, extras


def _parse_json_array_prefix(value: str) -> list[Any]:
    stripped = value.strip()
    if not stripped.startswith("["):
        return []
    decoder = json.JSONDecoder()
    items: list[Any] = []
    idx = 1
    while idx < len(stripped):
        while idx < len(stripped) and stripped[idx].isspace():
            idx += 1
        if idx < len(stripped) and stripped[idx] == "]":
            return items
        if idx < len(stripped) and stripped[idx] == ",":
            idx += 1
            continue
        try:
            item, end = decoder.raw_decode(stripped, idx)
        except json.JSONDecodeError:
            break
        items.append(item)
        idx = end
    return items


def _looks_like_placeholder_task(task: Any) -> bool:
    text = str(task or "").strip().lower()
    if not text:
        return True
    placeholder_terms = (
        "i'm sorry",
        "i cannot provide",
        "cannot provide the content",
        "placeholder",
        "no task",
    )
    return any(term in text for term in placeholder_terms)


def _float01(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


_DOWNSTREAM_FOCUS_TERMS = (
    "verify",
    "verification",
    "validate",
    "validation",
    "review",
    "cross-check",
    "corroborate",
    "confidence",
    "critique",
    "critic",
    "synthesis",
    "synthesize",
    "summarize",
    "summary",
    "integrate",
    "integration",
    "qa",
    "final",
)
_UPSTREAM_INPUT_TERMS = (
    "after",
    "based on",
    "before final",
    "confidence",
    "dependency",
    "depends",
    "evidence",
    "from peers",
    "input",
    "partial",
    "peer",
    "peers",
    "source",
    "upstream",
    "worker",
    "workers",
)
_DOWNSTREAM_STRONG_TERMS = (
    "after",
    "based on",
    "before final",
    "cross-check",
    "integrate",
    "integration",
    "review",
    "synthesis",
    "synthesize",
    "verify",
    "verifier",
    "verification",
)
_EVIDENCE_COLLECTION_TERMS = (
    "collect evidence",
    "evidence report",
    "find",
    "identify",
    "investigate",
    "look up",
    "lookup",
    "research",
    "search",
    "source",
    "sources",
)


def _looks_like_downstream_focus(*parts: Any) -> bool:
    text = "\n".join(str(part or "") for part in parts).lower()
    if not text.strip():
        return False
    if any(term in text for term in _EVIDENCE_COLLECTION_TERMS) and not any(
        term in text for term in _DOWNSTREAM_STRONG_TERMS
    ):
        return False
    return (
        any(term in text for term in _DOWNSTREAM_STRONG_TERMS)
        and any(term in text for term in _UPSTREAM_INPUT_TERMS)
    )


def _identity_terms_for_dependency_match(agent: "Agent") -> set[str]:
    terms = {
        str(agent.id or "").lower(),
        str(agent.role or "").lower(),
        str(agent.group_id or "").lower(),
        str(agent.create_type or "").lower(),
        str(agent.relationship or "").lower(),
    }
    terms.update(str(tag or "").lower() for tag in agent.current_task_tags)
    expanded = set(terms)
    for term in terms:
        if ":" in term:
            expanded.add(term.split(":", 1)[1])
    return {term for term in expanded if term}


def _agent_matches_dependency_terms(agent: "Agent", depends_on: list[str]) -> bool:
    wanted = {str(item).strip().lower() for item in depends_on if str(item).strip()}
    if not wanted:
        return True
    terms = _identity_terms_for_dependency_match(agent)
    for item in wanted:
        if item in terms:
            return True
        if ":" in item and item.split(":", 1)[1] in terms:
            return True
        if any(term.endswith(item) or item.endswith(term) for term in terms if len(term) >= 4):
            return True
    return False


def _spawn_dependency_score(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    depends_on = _normalize_string_list(args.get("depends_on"))
    explicit_readiness = args.get("readiness")
    configured_threshold = _float01(runtime.config.spawn_readiness_threshold, 0.62)
    requested_threshold = _float01(args.get("readiness_threshold"), configured_threshold)
    downstream_focus = _looks_like_downstream_focus(
        args.get("task", ""),
        args.get("role", ""),
        args.get("workflow_prior", ""),
        " ".join(_normalize_string_list(args.get("current_task_tags"))),
    )
    threshold = max(configured_threshold, requested_threshold) if depends_on or downstream_focus else requested_threshold
    dependency_like = bool(depends_on) or explicit_readiness is not None or downstream_focus
    inferred_readiness = 1.0
    if dependency_like:
        peers = [runtime.agents[child_id] for child_id in agent.children if child_id in runtime.agents]
        if (
            agent.parent
            and (
                not depends_on
                or _agent_matches_dependency_terms(agent, depends_on)
                or str(agent.id).lower() in {item.lower() for item in depends_on}
            )
        ):
            peers.append(agent)
        if depends_on:
            peers = [
                peer
                for peer in peers
                if peer.id in depends_on or peer.role in depends_on or peer.group_id in depends_on
                or set(depends_on).intersection(peer.current_task_tags)
            ]
        if not peers:
            inferred_readiness = 0.0
        else:
            scores = []
            for peer in peers:
                expected = runtime.expected_outputs(peer)
                missing = runtime.missing_expected_outputs(peer)
                if peer.status == "done":
                    score = 1.0
                elif (
                    downstream_focus
                    and expected
                    and peer.artifacts
                    and any(path in {artifact.path for artifact in peer.artifacts} for path in expected)
                ):
                    score = 0.75
                elif expected:
                    score = (len(expected) - len(missing)) / max(1, len(expected))
                elif peer.artifacts or peer.result:
                    score = 0.85
                elif peer._tool_calls >= 3:
                    score = 0.35
                else:
                    score = 0.1
                scores.append(score)
            inferred_readiness = sum(scores) / max(1, len(scores))
    if depends_on:
        readiness = inferred_readiness
    elif explicit_readiness is not None:
        readiness = max(_float01(explicit_readiness, 0.0), inferred_readiness if dependency_like else 0.0)
    else:
        readiness = inferred_readiness
    return {
        "readiness": round(readiness, 3),
        "inferred_readiness": round(inferred_readiness, 3),
        "threshold": round(threshold, 3),
        "depends_on": depends_on,
        "dependency_like": dependency_like,
        "downstream_focus": downstream_focus,
        "ready": readiness >= threshold,
    }


def _defer_spawn_request(args: dict[str, Any], agent: "Agent", runtime: "Runtime", readiness: dict[str, Any]) -> dict[str, Any]:
    request = {
        "task": args.get("task", ""),
        "role": args.get("role", ""),
        "create_type": args.get("create_type", ""),
        "relationship": args.get("relationship", ""),
        "group_id": args.get("group_id", ""),
        "workflow_prior": args.get("workflow_prior", ""),
        "current_task_tags": list(args.get("current_task_tags") or []),
        "orchestration_preference": args.get("orchestration_preference"),
        "model": args.get("model"),
        "depends_on": readiness.get("depends_on") or _normalize_string_list(args.get("depends_on")),
        "readiness": readiness.get("readiness", 0.0),
        "readiness_threshold": readiness.get("threshold", runtime.config.spawn_readiness_threshold),
        "reason": "spawn_readiness_below_threshold",
        "downstream_focus": bool(readiness.get("downstream_focus")),
        "created_at_turn": agent._turns,
    }
    agent._deferred_spawn_requests.append(request)
    runtime._emit(agent.id, "spawn_deferred", request)
    return {
        "deferred": True,
        "reason": request["reason"],
        "readiness": request["readiness"],
        "readiness_threshold": request["readiness_threshold"],
        "depends_on": request["depends_on"],
        "downstream_focus": request["downstream_focus"],
    }


def _matches_query_filter(snapshot: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if expected in (None, "", [], {}):
            continue
        memory_tags = set((snapshot.get("public_memory") or {}).get("tags") or [])
        board_tags = set((snapshot.get("state_board") or {}).get("current_task_tags") or [])
        all_tags = memory_tags.union(board_tags)
        if key in {"tags", "current_task_tags"}:
            expected_tags = set(normalize_tags(_normalize_string_list(expected)))
            if not expected_tags.intersection(all_tags):
                return False
            continue
        if key == "has_artifacts":
            has_artifacts = bool(snapshot.get("artifacts"))
            if bool(expected) != has_artifacts:
                return False
            continue
        if key in {"agent_id", "id"}:
            actual = snapshot.get("id")
            if isinstance(expected, list):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
            continue
        actual = snapshot.get(key)
        if key == "status" and expected == "active":
            expected = ["running", "idle"]
        if key == "role" and expected == "agent":
            continue
        if key == "role" and expected == "peer_agent":
            actual = snapshot.get("create_type")
        elif key == "role" and isinstance(expected, str):
            role_tag = f"role:{expected}"
            if snapshot.get("role") == expected or role_tag in all_tags:
                continue
        if key == "group_id" and isinstance(expected, str):
            actual_text = str(actual or "")
            if actual_text and (
                actual_text == expected
                or actual_text.endswith(expected)
                or expected.endswith(actual_text)
            ):
                continue
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _query_help(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    ids = sorted({snap.get("id") for snap in snapshots if snap.get("id")})
    roles = sorted({snap.get("role") for snap in snapshots if snap.get("role")})
    groups = sorted({snap.get("group_id") for snap in snapshots if snap.get("group_id")})
    tags = sorted({
        tag
        for snap in snapshots
        for tag in (
            ((snap.get("public_memory") or {}).get("tags") or [])
            + ((snap.get("state_board") or {}).get("current_task_tags") or [])
        )
    })
    return {
        "available_agent_ids": ids[:50],
        "available_roles": roles[:30],
        "available_group_ids": groups[:30],
        "available_tags": tags[:50],
        "tips": [
            "Use filter.group_id with the exact group_id shown here to query a wave of peer agents.",
            "Use tags=['feature:N'] or filter.tags=['feature:N'] for feature-scoped lookup.",
            "Use filter.status='running' or 'active'; active matches running and idle.",
            "create_type='peer_agent' is not a role; query filter.role='peer_agent' is treated as create_type for compatibility.",
        ],
    }


async def meta_spawn(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    from nanoma.core import (
        ResourceQuota,
        _inject_expected_outputs_if_needed,
        _inject_local_deliverable_if_needed,
        _strip_submit_answer_protocol_from_child_task,
    )

    task = args.get("task", "")
    model = args.get("model")
    delegate = args.get("delegate", False)
    role = args.get("role", "")
    create_type = args.get("create_type", "")
    relationship = args.get("relationship", "")
    group_id = args.get("group_id", "")
    workflow_prior = _normalize_workflow_prior(args.get("workflow_prior"))
    current_task_tags = list(args.get("current_task_tags", []))
    expected_outputs_arg = args.get("expected_outputs") or args.get("target_outputs") or args.get("target_output_path")
    orchestration_preference = args.get("orchestration_preference")
    readiness = _spawn_dependency_score(args, agent, runtime)

    if not task:
        return {"error": "task is required"}
    if _looks_like_placeholder_task(task):
        return {"error": "task appears to be a placeholder/refusal, not a concrete child task"}
    original_task = str(task)
    child_submit_granted = runtime.should_grant_submit_answer_to_child(
        agent,
        task=original_task,
        role=str(role or ""),
        group_id=str(group_id or ""),
        workflow_prior=str(workflow_prior or ""),
        current_task_tags=current_task_tags,
    )
    if not child_submit_granted:
        task = _strip_submit_answer_protocol_from_child_task(original_task)
    if task != original_task:
        runtime._emit(agent.id, "spawn_task_sanitized", {
            "reason": "removed_global_submit_protocol",
            "original_preview": original_task[:200],
            "sanitized_preview": str(task)[:200],
        })
    task, explicit_expected_outputs = _inject_expected_outputs_if_needed(str(task), expected_outputs_arg)
    if explicit_expected_outputs:
        runtime._emit(agent.id, "spawn_expected_outputs_injected", {
            "paths": explicit_expected_outputs[:20],
            "role": role,
            "group_id": group_id,
            "task_preview": str(task)[:200],
        })
    task, auto_local_output = _inject_local_deliverable_if_needed(
        str(task),
        parent_task=agent.task,
        role=str(role or ""),
        group_id=str(group_id or ""),
        current_task_tags=current_task_tags,
        child_submit_granted=child_submit_granted,
    )
    if auto_local_output:
        runtime._emit(agent.id, "spawn_local_output_injected", {
            "path": auto_local_output,
            "role": role,
            "group_id": group_id,
            "task_preview": str(task)[:200],
        })
    if agent.depth + 1 > runtime.config.max_depth:
        return {"error": f"Max depth ({runtime.config.max_depth}) exceeded"}
    if len(runtime.agents) >= runtime.config.max_agents:
        return {"error": f"Max agents ({runtime.config.max_agents}) reached"}
    if readiness["dependency_like"] and not readiness["ready"]:
        return _defer_spawn_request(args, agent, runtime, readiness)
    covered = runtime.covered_spawn_request(
        agent,
        task=str(task),
        role=str(role or ""),
        group_id=str(group_id or ""),
        workflow_prior=str(workflow_prior or ""),
        current_task_tags=current_task_tags,
    )
    if covered:
        covered_by = covered.get("covered_by", [])
        agent._last_spawn_skipped_turn = agent._turns
        agent._last_spawn_skipped_reason = str(covered.get("reason") or "")
        agent._last_spawn_skipped_covered_by = [
            str(item.get("id"))
            for item in covered_by
            if isinstance(item, dict) and item.get("id")
        ]
        agent._last_spawn_skipped_covered_paths = [
            str(path)
            for path in covered.get("covered_paths", [])
            if str(path)
        ]
        agent._last_create_resolution = "skipped_covered"
        runtime._emit(agent.id, "spawn_skipped", {
            "reason": covered.get("reason"),
            "task": str(task)[:160],
            "role": role,
            "group_id": group_id,
            "covered_by": covered_by,
            "covered_paths": covered.get("covered_paths", []),
            "request_focus": covered.get("request_focus"),
            "coverage_focuses": covered.get("coverage_focuses", []),
            "role_compatible": covered.get("role_compatible"),
        })
        return {
            "skipped": True,
            "reason": covered.get("reason"),
            "covered_by": covered_by,
            "covered_paths": covered.get("covered_paths", []),
            "request_focus": covered.get("request_focus"),
            "coverage_focuses": covered.get("coverage_focuses", []),
            "role_compatible": covered.get("role_compatible"),
            "advice": covered.get("advice"),
        }

    if not model:
        if runtime.router:
            model = runtime.router(task, runtime.ledger.remaining(), allowed_models=runtime.config.allowed_models)
        else:
            model = runtime.config.default_model
    requested_model = model
    model_fallback_reason = ""
    if runtime.config.allowed_models and model not in runtime.config.allowed_models:
        model = runtime.config.allowed_models[0]
        model_fallback_reason = "not_in_allowed_models"
    elif not getattr(runtime.config, "allow_agent_model_override", False) and model != runtime.config.default_model:
        model = runtime.config.default_model
        model_fallback_reason = "agent_model_override_disabled"
    else:
        try:
            from nanoma.models import get_registry
            registry = get_registry()
            if registry.models and not registry.get(str(model)):
                model = runtime.config.default_model
                model_fallback_reason = "unknown_model"
        except Exception:
            pass

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
        orchestration_preference=orchestration_preference,
    )
    if child_submit_granted:
        runtime.grant_submit_answer(child)
    runtime.start_agent(child)
    runtime.state_board_update(child.id, action_state="create", current_task_tags=child.current_task_tags)

    runtime._emit(agent.id, "spawn", {
        "child": child.id,
        "task": task[:100],
        "model": model,
        "requested_model": requested_model,
        "model_fallback_reason": model_fallback_reason,
        "role": child.role,
        "role_description": child.bio,
        "create_type": create_type,
        "relationship": relationship,
        "created_by": created_by,
        "group_id": group_id,
        "workflow_prior": workflow_prior,
        "current_task_tags": list(child.current_task_tags),
        "submit_answer_granted": child_submit_granted,
        "auto_local_output": auto_local_output,
        "expected_outputs": explicit_expected_outputs,
        "requested_orchestration_preference": orchestration_preference,
        "orchestration_preference": child.orchestration_preference,
        "readiness": readiness["readiness"],
        "readiness_threshold": readiness["threshold"],
        "depends_on": readiness["depends_on"],
    })

    if delegate:
        agent.status = "done"
        agent.action_state = "stop"
        agent.result = f"[Delegated to {child.id}]"
        runtime.state_board_sync(agent.id)

    return {
        "agent_id": child.id,
        "model": model,
        "requested_model": requested_model,
        "model_fallback_reason": model_fallback_reason,
        "role": child.role,
        "role_description": child.bio,
        "create_type": create_type,
        "relationship": relationship,
        "created_by": created_by,
        "group_id": group_id,
        "workflow_prior": workflow_prior,
        "current_task_tags": list(child.current_task_tags),
        "submit_answer_granted": child_submit_granted,
        "auto_local_output": auto_local_output,
        "expected_outputs": explicit_expected_outputs,
        "requested_orchestration_preference": orchestration_preference,
        "orchestration_preference": child.orchestration_preference,
        "readiness": readiness["readiness"],
        "readiness_threshold": readiness["threshold"],
        "depends_on": readiness["depends_on"],
    }


async def meta_create_agent(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    return await meta_spawn(args, agent, runtime)


async def meta_spawn_many(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    args = _recover_raw_tool_args(args)
    agent_items, embedded_defaults = _parse_spawn_many_payload(args.get("agents") or [])
    task_items, task_embedded_defaults = _parse_spawn_many_payload(args.get("tasks") or [])
    items: Any
    if isinstance(agent_items, list) and isinstance(task_items, list):
        items = agent_items + task_items
    else:
        items = agent_items if agent_items else task_items
    defaults = _parse_json_string(args.get("defaults") or {})
    if not isinstance(defaults, dict):
        defaults = {}
    defaults = {**embedded_defaults, **task_embedded_defaults, **defaults}
    for key in (
        "model",
        "delegate",
        "role",
        "create_type",
        "relationship",
        "group_id",
        "workflow_prior",
        "current_task_tags",
        "orchestration_preference",
        "depends_on",
        "readiness",
        "readiness_threshold",
        "expected_outputs",
        "target_output_path",
    ):
        if key in args and key not in defaults:
            defaults[key] = args[key]
    if not isinstance(items, list):
        return {"error": "agents/tasks must be a list"}
    if not items:
        return {"error": "agents/tasks must not be empty"}

    results = []
    for i, item in enumerate(items):
        if isinstance(item, str):
            spawn_args = {**defaults, "task": item}
        elif isinstance(item, dict):
            spawn_args = {**defaults, **item}
        else:
            results.append({"index": i, "error": "agent item must be a task string or object"})
            continue
        if spawn_args.get("orchestration_preference") == "solo":
            spawn_args.pop("orchestration_preference", None)
            spawn_args["_ignored_orchestration_preference"] = "solo"
        result = await meta_spawn(spawn_args, agent, runtime)
        results.append({"index": i, "result": result})
        if "error" in result and "Max agents" in result["error"]:
            break
    created = sum(1 for r in results if "agent_id" in r.get("result", {}))
    deferred = sum(1 for r in results if r.get("result", {}).get("deferred"))
    skipped = sum(1 for r in results if r.get("result", {}).get("skipped"))
    return {"created": created, "deferred": deferred, "skipped": skipped, "results": results}


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
        add_experience=build_memory_card(
            target.result or "Emergency stop",
            tags=target.current_task_tags,
            artifacts=[a.path for a in target.artifacts],
            memory_kind="terminal",
            memory_source="kill",
            stop_after=True,
        ),
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
        if message_type == "prune_request":
            agent._prune_requests_sent_targets.add(rid)

    return {
        "delivered": delivered,
        "tokens": msg_tokens,
        "mode": mode,
        "message_type": message_type,
        "requires_ack": requires_ack,
        "urgency": urgency,
    }


async def meta_query(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    agent._last_query_turn = max(agent._last_query_turn, agent._turns or 1)
    target_id = args.get("agent_id")
    messages_n = args.get("messages", 0)
    filters = args.get("filter") or {}
    tags = normalize_tags(_normalize_string_list(args.get("tags")))
    memory_intent = args.get("memory_intent", "")
    include_memory = bool(args.get("include_memory", True))
    limit = args.get("limit", 0)
    if not isinstance(filters, dict):
        return {"error": "filter must be an object"}
    try:
        limit = int(limit or 0)
    except (TypeError, ValueError):
        return {"error": "limit must be an integer"}

    if target_id:
        target = runtime.agents.get(target_id)
        if not target:
            if filters or tags:
                target_id = ""
            else:
                return {"error": f"Agent '{target_id}' not found"}
    if target_id:
        target = runtime.agents.get(target_id)
        if not target:
            return {"error": f"Agent '{target_id}' not found"}
        result = _agent_snapshot(target, runtime)
        runtime._emit(agent.id, "query", {
            "scope": "agent",
            "target": target_id,
            "targets": [target_id],
            "filter": filters,
            "tags": tags,
            "memory_intent": memory_intent,
            "result_count": 1,
        })
        if messages_n != 0:
            history = target.history[1:]
            if messages_n > 0:
                history = history[-messages_n:]
            result["messages"] = [
                {"role": m.get("role", ""), "content": (m.get("content") or "")[:500]}
                for m in history if m.get("role") in ("user", "assistant", "system")
            ]
        return result

    all_snapshots = [_agent_snapshot(a, runtime) for a in runtime.agents.values()]
    snapshots = list(all_snapshots)
    if tags:
        filters = {**filters, "tags": tags}
    if filters:
        snapshots = [snap for snap in snapshots if _matches_query_filter(snap, filters)]
    snapshots.sort(key=lambda snap: (
        -snap.get("progress", {}).get("completion_score", 0),
        snap["id"],
    ))
    if limit > 0:
        snapshots = snapshots[:limit]

    memory_read = None
    if tags or memory_intent:
        memory_read = runtime.memory.read(intent=memory_intent, seed_terms=tags)

    targets = [snap["id"] for snap in snapshots]
    runtime._emit(agent.id, "query", {
        "scope": "agents",
        "target": None,
        "targets": targets[:100],
        "filter": filters,
        "tags": tags,
        "memory_intent": memory_intent,
        "result_count": len(snapshots),
        "limited": limit > 0,
    })

    return {
        "agents": snapshots,
        "count": len(snapshots),
        "total_agents": len(runtime.agents),
        "state_board": runtime.state_board_list(),
        "task_ledger": runtime.task_ledger_snapshot(agent, include_all=bool(args.get("include_task_ledger_all", False))),
        "public_memory": runtime.memory.list() if include_memory else None,
        "memory_read": memory_read,
        "query_help": _query_help(all_snapshots) if not snapshots else None,
    }


async def meta_ledger_read(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    include_all = bool(args.get("include_all", False))
    item_id = str(args.get("item_id") or "").strip() or None
    limit = args.get("limit", 20)
    try:
        limit = int(limit or 20)
    except (TypeError, ValueError):
        return {"error": "limit must be an integer"}
    result = runtime.task_ledger_snapshot(agent, include_all=include_all, item_id=item_id, limit=limit)
    runtime._emit(agent.id, "task_ledger_read", {
        "include_all": include_all,
        "item_id": item_id,
        "count": result.get("count"),
        "total_items": result.get("total_items"),
    })
    return result


async def meta_ledger_update(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    return runtime.task_ledger_update(agent, args)


async def meta_wait(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    target_ids = _parse_json_string(args.get("agent_ids", []))
    timeout = args.get("timeout", 120.0)
    mode = args.get("mode", "all")
    if mode not in ("all", "any"):
        return {"error": "mode must be 'all' or 'any'"}
    if isinstance(target_ids, str):
        target_ids = _normalize_string_list(target_ids)
    if not isinstance(target_ids, list):
        return {"error": "agent_ids must be a list or comma-separated string"}
    try:
        timeout = float(timeout)
    except (TypeError, ValueError):
        return {"error": "timeout must be a number"}
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

    blockers = runtime.stop_blockers(agent) if action in {"done", "self_stop", "stop"} else []
    if blockers and not runtime.can_stop_with_active_dependencies(agent, blockers):
        runtime.state_board_update(agent.id, action_state="work", current_task_tags=current_task_tags, work_outline=work_outline)
        return {
            "blocked": "active_children",
            "message": (
                "Cannot stop blindly while dependent agents are still active. Use query() or wait() first, "
                "then stop if your summary can explicitly record which dependencies are still running."
            ),
            "active_children": blockers,
        }

    completion_blockers = (
        runtime.completion_blockers(agent, tags=list(current_task_tags or agent.current_task_tags), result=result)
        if action in {"done", "self_stop", "stop"}
        else []
    )
    if completion_blockers:
        runtime.state_board_update(agent.id, action_state="work", current_task_tags=current_task_tags, work_outline=work_outline)
        return {
            "blocked": "completion_evidence",
            "message": (
                "Cannot stop yet because required deliverable evidence is missing. Continue with the indicated work action, "
                "then retry stop/compact after source changes, tests, and explicit outputs are complete."
            ),
            "blockers": completion_blockers,
        }

    if compact_before_stop:
        if not compact_summary:
            return {"error": "compact_summary required when compact_before_stop=true"}
        agent._compact_pending = {
            "summary": compact_summary,
            "files": _normalize_string_list(args.get("files", [])),
            "tags": list(current_task_tags or agent.current_task_tags),
            "experience": args.get("experience") or result or compact_summary,
            "new_task": args.get("new_task"),
            "new_bio": args.get("new_bio"),
            "work_outline": work_outline or agent.work_outline,
            "stop_after": True,
            "stop_result": result,
            "memory_kind": args.get("memory_kind"),
            "memory_source": args.get("memory_source"),
            "durable": args.get("durable"),
            "evidence": _normalize_string_list(args.get("evidence", [])),
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
        add_experience=build_memory_card(
            agent.result or result or action,
            tags=agent.current_task_tags,
            artifacts=[a.path for a in agent.artifacts],
            memory_kind="terminal",
            memory_source="set_status",
            stop_after=True,
        ),
        tags=agent.current_task_tags,
    )
    agent.memory = {"public_memory": runtime.memory.serialize(agent.id)}
    runtime.state_board_sync(agent.id)
    runtime._refresh_task_ledger_agent_item(agent)
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
    stop_after = bool(args.get("stop_after", False) or args.get("done", False))
    stop_result = args.get("result") or args.get("stop_result") or ""
    tags = _normalize_string_list(args.get("tags", agent.current_task_tags)) or list(agent.current_task_tags)
    blockers = runtime.stop_blockers(agent) if stop_after else []
    if blockers and not runtime.can_stop_with_active_dependencies(agent, blockers):
        runtime.state_board_update(agent.id, action_state="work", current_task_tags=tags, work_outline=args.get("work_outline"))
        return {
            "blocked": "active_children",
            "message": (
                "Cannot compact-and-stop blindly while dependent agents are still active. Use query() or wait() first, "
                "then compact with stop_after=true if your summary records active dependencies and remaining gaps."
            ),
            "active_children": blockers,
        }
    completion_blockers = runtime.completion_blockers(agent, tags=tags, result=stop_result) if stop_after else []
    if completion_blockers:
        runtime.state_board_update(agent.id, action_state="work", current_task_tags=tags, work_outline=args.get("work_outline"))
        return {
            "blocked": "completion_evidence",
            "message": (
                "Cannot compact-and-stop yet because required deliverable evidence is missing. Continue the task, "
                "then compact with stop_after=true after the blockers are resolved."
            ),
            "blockers": completion_blockers,
        }
    agent._compact_pending = {
        "summary": summary,
        "files": _normalize_string_list(args.get("files", [])),
        "tags": tags,
        "experience": args.get("experience") or (stop_result if stop_after else None),
        "new_task": args.get("new_task"),
        "new_bio": args.get("new_bio"),
        "work_outline": args.get("work_outline", agent.work_outline),
        "stop_after": stop_after,
        "stop_result": stop_result,
        "memory_kind": args.get("memory_kind"),
        "memory_source": args.get("memory_source"),
        "durable": args.get("durable"),
        "evidence": _normalize_string_list(args.get("evidence", [])),
    }
    runtime.state_board_update(agent.id, action_state="compact", current_task_tags=tags, work_outline=args.get("work_outline"))
    return {"scheduled": True, "action_state": "compact", "stop_after": stop_after}


async def meta_submit(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    from nanoma.core import Artifact

    path_str = args.get("path", "")
    if not path_str:
        return {"error": "path required"}
    path = _resolve_workspace_path(path_str, agent.workspace, runtime)
    if path is None:
        return {"error": "Access denied: outside workspace"}
    if not path.exists():
        return {"error": f"Not found: {path}"}
    if not path.is_file():
        return {"error": f"Not a file: {path}"}

    shared = runtime._tool_context.shared_dir
    shared.mkdir(parents=True, exist_ok=True)
    shared_copy = path
    try:
        path.resolve().relative_to(shared.resolve())
    except ValueError:
        shared_copy = shared / path.name
        shutil.copy2(path, shared_copy)

    artifact = Artifact(path=path_str, absolute_path=path, description=args.get("description", ""), agent_id=agent.id)
    agent.artifacts.append(artifact)
    runtime.memory.update(agent.id, add_artifacts=[path_str])
    agent.memory = {"public_memory": runtime.memory.serialize(agent.id)}
    runtime._refresh_task_ledger_agent_item(agent)
    return {"submitted": path_str, "shared_copy": str(shared_copy)}


async def meta_submit_answer(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    from nanoma.core import Artifact

    answer = args.get("answer")
    if answer is None:
        return {"error": "answer required"}
    answer_text = str(answer)
    path_str = str(args.get("path") or runtime.final_answer_path_for(agent))
    permission = runtime.can_submit_answer(agent, path_str)
    if not permission.get("ok"):
        runtime._emit(agent.id, "submit_answer_denied", permission)
        return {"error": "submit_answer_denied", **permission}
    if not runtime.answer_submission_required(agent) and not path_str:
        path_str = f"{runtime.config.shared_dir}/answer.json"

    path = _resolve_workspace_path(path_str, agent.workspace, runtime)
    if path is None:
        return {"error": "Access denied: outside workspace"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"answer": answer_text}, ensure_ascii=False) + "\n", encoding="utf-8")

    rel_path = runtime._normalize_workspace_relative_path(path_str, agent.workspace)
    artifact = Artifact(
        path=rel_path,
        absolute_path=path,
        description=args.get("description") or "final answer submission",
        agent_id=agent.id,
    )
    if rel_path not in {existing.path for existing in agent.artifacts}:
        agent.artifacts.append(artifact)
    agent.submitted_answer_path = rel_path
    agent.action_state = "stop"
    agent.status = "done"
    agent.result = answer_text

    tags = normalize_tags(_normalize_string_list(args.get("tags", agent.current_task_tags)) or list(agent.current_task_tags))
    evidence_refs = _normalize_string_list(args.get("evidence_refs", args.get("evidence", [])))
    memory_text = answer_text
    if evidence_refs:
        memory_text += "\nEvidence refs:\n" + "\n".join(f"- {ref}" for ref in evidence_refs[:20])
    confidence = args.get("confidence")
    if confidence:
        memory_text += f"\nConfidence: {confidence}"
    runtime.memory.update(
        agent.id,
        public_summary=answer_text,
        clear_active_task=True,
        add_artifacts=[rel_path],
        add_experience=build_memory_card(
            memory_text,
            tags=tags,
            artifacts=[rel_path],
            memory_kind="terminal",
            memory_source="submit_answer",
            stop_after=True,
        ),
        tags=tags,
    )
    agent.memory = {"public_memory": runtime.memory.serialize(agent.id)}
    runtime.state_board_sync(agent.id)
    runtime._refresh_task_ledger_agent_item(agent)
    if agent.parent is None and runtime.answer_submission_required(agent):
        running = [
            peer
            for peer in runtime.agents.values()
            if peer.id != agent.id and peer.status in {"running", "idle"}
        ]
        await runtime._request_remaining_agents_prune_after_terminal_submission(agent, running)
    return {
        "submitted_answer": rel_path,
        "answer": answer_text,
        "status": agent.status,
        "confidence": confidence,
    }


async def meta_batch(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    path_str = args.get("path", "")
    if not path_str:
        return {"error": "path required"}
    path = _resolve_workspace_path(path_str, agent.workspace, runtime)
    if path is None:
        return {"error": "Access denied: outside workspace"}
    if not path.exists():
        return {"error": f"Not found: {path}"}
    if not path.is_file():
        return {"error": f"Not a file: {path}"}
    try:
        calls = json.loads(path.read_text())
    except Exception as e:
        return {"error": f"Parse error: {e}"}
    if not isinstance(calls, list):
        return {"error": "File must contain a JSON array of {tool, args} objects"}

    all_tools = {**runtime._available_work_tools(), **META_TOOLS}
    results = []
    for i, call in enumerate(calls):
        tool_name, tool_args, normalize_error = _normalize_batch_call(call)
        if normalize_error:
            results.append({"index": i, "error": normalize_error})
            continue
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


def _normalize_batch_call(call: Any) -> tuple[str, dict[str, Any], str | None]:
    if not isinstance(call, dict):
        return "", {}, "Batch item must be an object"

    fn = call.get("function")
    if isinstance(fn, dict):
        tool_name = fn.get("name") or call.get("tool") or call.get("name") or ""
        tool_args = fn.get("arguments", call.get("arguments", call.get("args", call.get("params", {}))))
    else:
        tool_name = call.get("tool") or call.get("name") or (fn if isinstance(fn, str) else "")
        tool_args = call.get("args", call.get("params", call.get("arguments", {})))

    if not tool_name:
        return "", {}, "Batch item missing tool/name/function.name"

    if isinstance(tool_args, str):
        if not tool_args.strip():
            tool_args = {}
        else:
            try:
                tool_args = json.loads(tool_args)
            except Exception as e:
                return str(tool_name), {}, f"Arguments parse error: {e}"

    if tool_args is None:
        tool_args = {}
    if not isinstance(tool_args, dict):
        return str(tool_name), {}, "Arguments must be an object"
    return str(tool_name), tool_args, None


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


def _resolve_workspace_path(path_str: str, workspace: Path, runtime: "Runtime") -> Path | None:
    shared_dir = runtime._tool_context.shared_dir
    raw = str(path_str or ".")
    raw = raw.replace("$SHARED", str(shared_dir)).replace("${SHARED}", str(shared_dir))
    raw = raw.replace("$WORKSPACE", str(workspace)).replace("${WORKSPACE}", str(workspace))
    raw = os.path.expandvars(raw)
    path = Path(raw)
    if not path.is_absolute() and path.parts and path.parts[0] == shared_dir.name:
        path = shared_dir.joinpath(*path.parts[1:])
    elif not path.is_absolute() and path.parts and path.parts[0] == runtime._tool_context.workspace_root.name:
        path = runtime._tool_context.workspace_root.joinpath(*path.parts[1:])
    elif not path.is_absolute():
        path = workspace / path
    try:
        resolved = path.resolve()
        resolved.relative_to(runtime._tool_context.workspace_root.resolve())
        return resolved
    except ValueError:
        return None


META_TOOLS: dict[str, dict[str, Any]] = {
    "spawn": {"handler": meta_spawn, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "spawn",
        "description": "Create a new agent when its input is ready. Optional peer-first metadata fields are role/create_type/relationship/group_id/workflow_prior. For downstream verification/review/synthesis agents, provide depends_on and a 0-1 readiness score; runtime defers low-readiness spawns instead of starting empty agents. created_by is runtime-populated when create_type='peer_agent'. workflow_prior is guidance, not a workflow engine.",
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
            "orchestration_preference": {"type": "string", "enum": ["solo", "balanced", "parallel", "aggressive"]},
            "depends_on": {"type": "array", "items": {"type": "string"}, "description": "Agent ids, roles, group ids, or tags whose partial outputs this downstream agent depends on."},
            "readiness": {"type": "number", "description": "0-1 estimate that dependencies already have enough output for this agent to start useful work."},
            "readiness_threshold": {"type": "number", "description": "Optional 0-1 threshold for immediate start; defaults to runtime spawn_readiness_threshold."},
            "expected_outputs": {"type": "array", "items": {"type": "string"}, "description": "Optional canonical shared output paths this child must produce or repair. Use for recovery/verifier handoffs so the ledger tracks the same output slot."},
            "target_output_path": {"type": "string", "description": "Shortcut for one expected output path."},
        }, "required": ["task"]},
    }}} ,
    "create_agent": {"handler": meta_create_agent, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "create_agent",
        "description": "Alias of spawn with the same peer-first creation metadata and readiness/dependency controls.",
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
            "orchestration_preference": {"type": "string", "enum": ["solo", "balanced", "parallel", "aggressive"]},
            "depends_on": {"type": "array", "items": {"type": "string"}},
            "readiness": {"type": "number"},
            "readiness_threshold": {"type": "number"},
            "expected_outputs": {"type": "array", "items": {"type": "string"}},
            "target_output_path": {"type": "string"},
        }, "required": ["task"]},
    }}} ,
    "spawn_many": {"handler": meta_spawn_many, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "spawn_many",
        "description": "Create a peer wave of multiple agents in one tool call. Use early for independent worker lanes. For downstream verification/review/synthesis lanes, provide depends_on and readiness; runtime defers low-readiness entries so workers can publish partial outputs before verifier agents start. Put shared metadata in defaults and per-agent task/role/tags in agents.",
        "parameters": {"type": "object", "properties": {
            "defaults": {"type": "object", "description": "Optional fields applied to every spawn, such as create_type='peer_agent', relationship='peer', group_id, workflow_prior, orchestration_preference, model, depends_on, readiness, readiness_threshold, or expected_outputs."},
            "agents": {"type": "array", "items": {"type": "object"}, "description": "Agent definitions. Each object may include task, role, create_type, relationship, group_id, workflow_prior, orchestration_preference, current_task_tags, model, delegate, depends_on, readiness, readiness_threshold, expected_outputs, or target_output_path."},
            "tasks": {"type": "array", "items": {"type": "string"}, "description": "Shortcut list of task strings; defaults are applied to each."},
        }},
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
        "description": "Send a message to one or more agents. Supports structured messaging with message_type/payload/requires_ack/urgency while remaining backward-compatible with plain text. Use prune_request after query shows a key output is already covered and duplicate agents should compact their partial state and self-prune; use stop_request only for explicit stop coordination.",
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
        "description": "Discover agents by status, role, group_id, parent, tags, or public memory. Use filter={} when intentionally querying all agents. Only set agent_id when you already know an exact existing agent id; for group/status/tag discovery leave agent_id unset and use filter/tags. Use this to inspect peer state_board/public_memory and summarize from observed agent state instead of relying on parent-directed reports.",
        "parameters": {"type": "object", "properties": {
            "agent_id": {"type": "string", "description": "Exact existing agent id for a direct lookup. Leave unset when using filter or tags."},
            "messages": {"type": "integer", "default": 0},
            "filter": {"type": "object", "description": "Optional filters such as status, action_state, role, group_id, parent, created_by, relationship, workflow_prior, tags, current_task_tags, or has_artifacts."},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Match agents whose public memory or state_board contains any of these tags."},
            "memory_intent": {"type": "string", "description": "Intent label for tag-based memory discovery."},
            "include_memory": {"type": "boolean", "default": True},
            "include_task_ledger_all": {"type": "boolean", "default": False},
            "limit": {"type": "integer", "default": 0},
        }},
    }}} ,
    "ledger_read": {"handler": meta_ledger_read, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "ledger_read",
        "description": "Read the shared task ledger. The loop card already shows a short ledger slice every turn; use this tool for more items or a specific item. The ledger tracks task ownership, blockers, expected outputs, and coverage, while memory/query hold detailed evidence.",
        "parameters": {"type": "object", "properties": {
            "item_id": {"type": "string", "description": "Optional ledger item id, usually an agent id."},
            "include_all": {"type": "boolean", "default": False},
            "limit": {"type": "integer", "default": 20},
        }},
    }}} ,
    "ledger_update": {"handler": meta_ledger_update, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "ledger_update",
        "description": "Update the shared task ledger after producing, blocking, verifying, or covering a task item/output. Do not store long evidence here; put detailed evidence in artifacts or compacted memory and reference it from note/evidence_refs.",
        "parameters": {"type": "object", "properties": {
            "item_id": {"type": "string", "description": "Ledger item id to update; defaults to this agent id."},
            "status": {"type": "string", "enum": ["open", "claimed", "in_progress", "partial", "blocked", "ready_for_review", "verified", "covered", "pruned", "failed"]},
            "output_path": {"type": "string"},
            "output_status": {"type": "string", "enum": ["missing", "draft", "partial", "unverified", "placeholder", "evidence_gap", "evidence_attached", "verified", "covered", "submitted", "rejected"]},
            "covered_by": {"type": "array", "items": {"type": "string"}},
            "blockers": {"type": "array", "items": {"type": "object"}},
            "note": {"type": "string"},
            "evidence_refs": {"type": "array", "items": {"type": "string"}},
        }},
    }}} ,
    "wait": {"handler": meta_wait, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "wait",
        "description": "Block execution until agents finish. Use agent_ids=[] to wait for all current child agents.",
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
        "description": "Update action_state through the state board or stop/self-stop. Prefer action for state transitions and use action='stop' instead of status='done' when intentionally stopping. Unknown actions are errors. compact_before_stop can compact first when compact_summary is provided. If peer query shows your lane is redundant, stop with tags such as status:pruned and reason:peer_ahead via compact_before_stop or compact(..., stop_after=true).",
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
            "memory_kind": {"type": "string", "enum": ["active", "working", "summary", "experience", "durable", "evidence", "terminal", "handoff", "status", "recent_input"]},
            "memory_source": {"type": "string"},
            "durable": {"type": "boolean"},
            "evidence": {"type": "array", "items": {"type": "string"}},
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
        "description": "Compact live history into public memory. summary is required. Use stop_after=true or done=true with result when compaction is the final step; plain compact means continue next turn. Keep fine-grained task tags, and optionally set memory_kind to evidence/summary/experience/durable/terminal so peers can query the knowledge base by both tags and memory layer. When pruning yourself after peer query, include tags like status:pruned, reason:peer_ahead, and task tags so peers can discover the summary.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string"},
            "files": {"type": "array", "items": {"type": "string"}},
            "tags": {"type": "array", "items": {"type": "string"}},
            "experience": {"type": "string"},
            "memory_kind": {"type": "string", "enum": ["active", "working", "summary", "experience", "durable", "evidence", "terminal", "handoff", "status", "recent_input"]},
            "memory_source": {"type": "string"},
            "durable": {"type": "boolean"},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "new_task": {"type": "string"},
            "new_bio": {"type": "string"},
            "work_outline": {"type": "string"},
            "stop_after": {"type": "boolean", "default": False},
            "done": {"type": "boolean", "default": False},
            "result": {"type": "string"},
            "stop_result": {"type": "string"},
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
    "submit_answer": {"handler": meta_submit_answer, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "submit_answer",
        "description": "Final protocol for answer-style benchmark tasks. Writes the concise answer to the configured shared answer JSON file and marks this agent done. This tool does not judge correctness; submit only the answer you intend the benchmark to evaluate.",
        "parameters": {"type": "object", "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "string"},
            "evidence_refs": {"type": "array", "items": {"type": "string"}},
            "tags": {"type": "array", "items": {"type": "string"}},
            "path": {"type": "string"},
            "description": {"type": "string"},
        }, "required": ["answer"]},
    }}} ,
    "batch": {"handler": meta_batch, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "batch",
        "description": "Execute multiple tool calls from a JSON array file. Items may use {tool,args}, {name,params}, or {function:{name,arguments}}.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
        }, "required": ["path"]},
    }}} ,
}
