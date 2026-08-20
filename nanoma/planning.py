"""Same-model planning-list meta tools for NanoMA.

The planning path does not expose direct spawn tools to the worker model.
Instead, the worker maintains a per-agent task list. At a fresh planning node,
`task_create` asks that worker's model whether the next phase should fan out;
the runtime creates children with the same model only when it approves.

This module adds three coordination meta tools that mirror Claude Code's
`TaskCreate` / `TaskUpdate` / `TaskList`, adapted to NanoMA conventions:

    task_create  -> create a pending task in the current agent's list
    task_update  -> change a task's status/fields (pending/in_progress/completed/cancelled)
    task_list    -> read the current agent's task list

Design notes
------------
* Naming: NanoMA tools are snake_case (`task_create`, `set_bio`, `ws_read_file`), so
  these use `task_create` etc. rather than CamelCase `TaskCreate`. The tool
  *descriptions* keep Claude's "When to Use / When NOT to Use" policy text,
  because our log analysis showed that description text — not any runtime
  event — is what actually gates when a model emits these calls.
* State: the list lives on the Agent instance as `agent._todos` (a list of
  dicts) plus `agent._todo_seq` (an int counter). Agent is a plain dataclass
  without __slots__, so attaching these lazily needs no core edit. State is
  per-agent; judge-created children start with an empty list.
* These are `is_meta` tools: handlers receive (args, agent, runtime) and may
  emit viewer events via `runtime._emit`, exactly like `send`.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from nanoma.core import Agent, Runtime


VALID_STATUSES = ("pending", "in_progress", "completed", "cancelled")
ACTIVE_STATUSES = ("in_progress", "pending")
TERMINAL_STATUSES = ("completed", "cancelled")

# Reminder-loop tuning (state re-injection + nudges).
# Override via env NANOMA_TODO_STALE_AFTER / NANOMA_TODO_EMPTY_HINT_AFTER.
STALE_NUDGE_AFTER_TURNS = 3   # tasks open but no create/update for this many turns
EMPTY_HINT_AFTER_TURNS = 4    # no tasks created after this many turns -> one-time hint


# ─── State helpers ───────────────────────────────────────────────────────────

def _todos(agent: "Agent") -> list[dict[str, Any]]:
    """Return the agent's task list, creating it on first use."""
    todos = getattr(agent, "_todos", None)
    if todos is None:
        todos = []
        setattr(agent, "_todos", todos)
    return todos


def _next_id(agent: "Agent") -> int:
    seq = getattr(agent, "_todo_seq", 0) + 1
    setattr(agent, "_todo_seq", seq)
    return seq


def _mark_mutated(agent: "Agent") -> None:
    """Record the agent turn at which the task list last changed (for staleness)."""
    setattr(agent, "_todo_last_mutation_turn", int(getattr(agent, "_turns", 0) or 0))


def _find(agent: "Agent", task_id: Any) -> dict[str, Any] | None:
    try:
        tid = int(task_id)
    except (TypeError, ValueError):
        return None
    for todo in _todos(agent):
        if todo["id"] == tid:
            return todo
    return None


def _counts(agent: "Agent") -> dict[str, int]:
    counts = {status: 0 for status in VALID_STATUSES}
    for todo in _todos(agent):
        counts[todo["status"]] = counts.get(todo["status"], 0) + 1
    return counts


def _render(agent: "Agent") -> list[dict[str, Any]]:
    """Compact, model-friendly view of the list."""
    return [
        {
            "id": todo["id"],
            "status": todo["status"],
            "subject": todo["subject"],
        }
        for todo in _todos(agent)
    ]


# ─── Handlers ────────────────────────────────────────────────────────────────

async def meta_task_create(
    args: dict[str, Any], agent: "Agent", runtime: "Runtime"
) -> dict[str, Any]:
    """Create a new pending task in the current agent's task list."""
    subject = str(args.get("subject", "") or "").strip()
    description = str(args.get("description", "") or "").strip()
    active_form = str(args.get("activeForm", "") or args.get("active_form", "") or "").strip()

    if not subject:
        return {"error": "'subject' required (a brief, actionable title in imperative form)"}

    # Same-model spawn-judge hook: at a fresh planning moment, decide
    # whether to fan this next phase out to parallel children BEFORE materializing
    # the todolist — so the list is only ever created on the no-spawn path
    # (decide first, plan second). The same model sees the parent's full context
    # and owns the split; on spawn we suppress local task creation entirely.
    planning_enabled = bool(
        getattr(getattr(runtime, "config", None), "node_autonomous_planning", False)
    )
    if planning_enabled:
        cur = int(getattr(agent, "_turns", 0) or 0)
        _delegated = {
            "delegated": True,
            "note": (
                "This planning step was delegated to parallel child agents by the "
                "spawn judge; no local task was created. The children can query your "
                "context; integrate their results when they finish."
            ),
        }
        if getattr(agent, "_spawn_decided_turn", None) == cur:
            # Already judged this turn: suppress sibling task_create calls if we spawned.
            if getattr(agent, "_spawn_decided_spawned", False):
                return _delegated
        else:
            # Treat as a fresh planning moment only when no pending task is open yet
            # (the start of a new plan, not mid-plan appends).
            has_pending = any(t.get("status") == "pending" for t in _todos(agent))
            if not has_pending:
                hint = subject if not description else f"{subject}: {description}"
                try:
                    spawned = bool(await runtime._spawn_judge_at_plan(agent, hint))
                except Exception:
                    spawned = False
                setattr(agent, "_spawn_decided_turn", cur)
                setattr(agent, "_spawn_decided_spawned", spawned)
                if spawned:
                    return _delegated

    todo = {
        "id": _next_id(agent),
        "subject": subject,
        "description": description,
        "activeForm": active_form,
        "status": "pending",
    }
    _todos(agent).append(todo)
    _mark_mutated(agent)

    try:
        runtime._emit(agent.id, "task_create", {
            "task_id": todo["id"],
            "subject": subject,
            "description": description,
            "status": "pending",
            "total": len(_todos(agent)),
        })
    except Exception:
        pass

    return {
        "task_id": todo["id"],
        "created": subject,
        "status": "pending",
        "total": len(_todos(agent)),
        "counts": _counts(agent),
    }


async def meta_task_update(
    args: dict[str, Any], agent: "Agent", runtime: "Runtime"
) -> dict[str, Any]:
    """Update a task's status or fields."""
    if "task_id" not in args and "id" not in args:
        return {"error": "'task_id' required"}
    task_id = args.get("task_id", args.get("id"))
    todo = _find(agent, task_id)
    if todo is None:
        return {
            "error": f"task_id {task_id!r} not found",
            "task_list": _render(agent),
        }

    changed: list[str] = []

    status = args.get("status")
    if status is not None:
        status = str(status).strip()
        if status not in VALID_STATUSES:
            return {"error": f"invalid status {status!r}; must be one of {VALID_STATUSES}"}
        todo["status"] = status
        changed.append("status")

    for field in ("subject", "description", "activeForm"):
        alt = "active_form" if field == "activeForm" else field
        if field in args or alt in args:
            value = str(args.get(field, args.get(alt, "")) or "").strip()
            if value:
                todo[field] = value
                changed.append(field)

    if not changed:
        return {
            "error": "no updatable fields provided (status/subject/description/activeForm)",
            "task": {"id": todo["id"], "status": todo["status"], "subject": todo["subject"]},
        }

    _mark_mutated(agent)

    try:
        runtime._emit(agent.id, "task_update", {
            "task_id": todo["id"],
            "changed": changed,
            "status": todo["status"],
            "subject": todo["subject"],
            "counts": _counts(agent),
        })
    except Exception:
        pass

    return {
        "task_id": todo["id"],
        "changed": changed,
        "status": todo["status"],
        "counts": _counts(agent),
    }


async def meta_task_list(
    args: dict[str, Any], agent: "Agent", runtime: "Runtime"
) -> dict[str, Any]:
    """Return the current agent's task list."""
    status_filter = args.get("status")
    todos = _todos(agent)
    if status_filter:
        status_filter = str(status_filter).strip()
        items = [t for t in _render(agent) if t["status"] == status_filter]
    else:
        items = _render(agent)
    return {
        "tasks": items,
        "total": len(todos),
        "counts": _counts(agent),
    }


# ─── Per-turn reminder / nudge (state re-injection) ──────────────────────────
# Mirrors Claude Code's behavior of keeping the current task list visible in
# context every turn, plus a staleness nudge. Returned text is injected as an
# ephemeral trailing user message by the runtime — it is NOT persisted to the
# agent's history, so it always reflects current state and does not accumulate.

def _env_int(name: str, default: int) -> int:
    import os
    try:
        return int(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _active_todos(agent: "Agent") -> list[dict[str, Any]]:
    return [t for t in _todos(agent) if t["status"] in ACTIVE_STATUSES]


def _reflect_delegated_status(agent: "Agent", runtime: "Runtime | None") -> None:
    """Read-only: mark delegated tasks completed once their child agent is done.

    Never touches or shares the child's own task list — only checks the child's
    lifecycle status via the runtime's agent registry. Best-effort.
    """
    if runtime is None:
        return
    for t in _todos(agent):
        cid = t.get("child_id")
        if not cid or t.get("status") != "in_progress":
            continue
        try:
            child = runtime.agents.get(cid)
        except Exception:
            child = None
        if child is not None and getattr(child, "status", None) == "done":
            t["status"] = "completed"


def render_todo_reminder(agent: "Agent", runtime: "Runtime | None" = None) -> str | None:
    """Build the ephemeral per-turn task-list reminder for `agent`.

    Returns None when there is nothing worth injecting (keeps quiet on trivial
    turns). Three cases:
      * open tasks exist        -> re-inject the list (+ stale nudge if idle).
      * no tasks, early turns    -> stay silent.
      * no tasks, after N turns  -> one-time gentle hint (then never again).
    """
    _reflect_delegated_status(agent, runtime)
    todos = _todos(agent)
    current_turn = int(getattr(agent, "_turns", 0) or 0)
    stale_after = _env_int("NANOMA_TODO_STALE_AFTER", STALE_NUDGE_AFTER_TURNS)
    empty_after = _env_int("NANOMA_TODO_EMPTY_HINT_AFTER", EMPTY_HINT_AFTER_TURNS)

    active = _active_todos(agent)

    # Empty-list one-time hint.
    if not todos:
        if current_turn >= empty_after and not getattr(agent, "_todo_empty_hint_sent", False):
            setattr(agent, "_todo_empty_hint_sent", True)
            return (
                "<task-list-hint>\n"
                "You have no task list yet. If your current work involves 3+ distinct "
                "steps, call task_create to plan and track it (and mark tasks "
                "in_progress/completed as you go). Skip this for trivial, single-step work.\n"
                "</task-list-hint>"
            )
        return None

    # No open tasks left -> nothing to re-inject.
    if not active:
        return None

    done = sum(1 for t in todos if t["status"] == "completed")
    lines = []
    for t in todos:
        if t["status"] in TERMINAL_STATUSES and t["status"] != "completed":
            continue  # hide cancelled from the working view
        marker = f"[{t['status']}]".ljust(14)
        annotation = f"  (delegated -> {t['child_id']})" if t.get("child_id") else ""
        lines.append(f"  {marker} #{t['id']} {t['subject']}{annotation}")

    body = (
        "<task-list>\n"
        "Your current task list (mark a task in_progress before starting it and "
        "completed as soon as it is done; add newly discovered work with task_create):\n"
        + "\n".join(lines)
        + f"\nProgress: {done} completed / {len(todos)} total."
    )

    last_mut = int(getattr(agent, "_todo_last_mutation_turn", 0) or 0)
    idle_turns = current_turn - last_mut
    if last_mut and idle_turns >= stale_after:
        body += (
            f"\nREMINDER: the task list has not changed in {idle_turns} turns. "
            "Update statuses (complete finished tasks, set the next one to "
            "in_progress) or add any newly discovered tasks."
        )

    body += "\n</task-list>"
    return body


# ─── Descriptions ────────────────────────────────────────────────────────────
# The "When to Use / When NOT to Use" text below is adapted from Claude Code's
# TaskCreate description because that policy text is what actually drives when a
# model chooses to build a task list (confirmed via proxy-log analysis of the
# observed planning-workflow runs).

_TASK_CREATE_DESC = (
    "Create a structured task in your current session's task list. This helps you "
    "plan and track progress on complex, multi-step work and makes your intended "
    "steps visible.\n\n"
    "## When to Use\n"
    "- Complex multi-step tasks requiring 3 or more distinct steps.\n"
    "- Non-trivial work that benefits from explicit planning.\n"
    "- After receiving new instructions — capture the requirements as tasks.\n"
    "- When you discover follow-up work while executing — add it as a new task.\n\n"
    "## When NOT to Use\n"
    "- A single, straightforward task, or work completable in <3 trivial steps.\n"
    "- Purely conversational or informational requests.\n"
    "In those cases just do the work directly instead of tracking it.\n\n"
    "Note: this task list is a lightweight self-checklist for THIS agent. At a "
    "fresh task_create planning node, the same model decides whether "
    "independent work should be delegated to parallel children."
)

_TASK_UPDATE_DESC = (
    "Update a task in your task list. Set `status` to 'in_progress' BEFORE you "
    "start working on it, and 'completed' as soon as it is done. Use 'cancelled' "
    "for tasks that are no longer needed. You may also revise subject/description."
)

_TASK_LIST_DESC = (
    "Read your current task list with each task's id, status, and subject. "
    "Optionally filter by `status`. Check this before creating tasks to avoid "
    "duplicates and to decide what to work on next."
)

# ─── Registry ────────────────────────────────────────────────────────────────

TODO_TOOLS: dict[str, dict[str, Any]] = {
    "task_create": {"handler": meta_task_create, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "task_create",
        "description": _TASK_CREATE_DESC,
        "parameters": {"type": "object", "properties": {
            "subject": {"type": "string", "description": "A brief, actionable title in imperative form (e.g., 'Fix auth bug in login flow')."},
            "description": {"type": "string", "description": "What needs to be done, with enough detail to act on."},
            "activeForm": {"type": "string", "description": "Present-continuous label shown while in_progress (e.g., 'Fixing auth bug'). Optional."},
        }, "required": ["subject"]},
    }}},
    "task_update": {"handler": meta_task_update, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "task_update",
        "description": _TASK_UPDATE_DESC,
        "parameters": {"type": "object", "properties": {
            "task_id": {"type": "integer", "description": "The id of the task to update (from task_create/task_list)."},
            "status": {"type": "string", "enum": list(VALID_STATUSES), "description": "New status."},
            "subject": {"type": "string", "description": "Revised title (optional)."},
            "description": {"type": "string", "description": "Revised description (optional)."},
            "activeForm": {"type": "string", "description": "Revised in-progress label (optional)."},
        }, "required": ["task_id"]},
    }}},
    "task_list": {"handler": meta_task_list, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "task_list",
        "description": _TASK_LIST_DESC,
        "parameters": {"type": "object", "properties": {
            "status": {"type": "string", "enum": list(VALID_STATUSES), "description": "Optional: only return tasks with this status."},
        }},
    }}},
}
