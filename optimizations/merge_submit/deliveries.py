"""Outstanding deliveries as something a parent can read and act on.

A parent that submits while its children are mid-task waits for them, so the
submission describes the whole workspace rather than a half-written one. That
wait used to be the parent's only option, and it was invisible: it happened
inside the `submit` call, for up to fifteen minutes, with nothing the parent
could do in the meantime and no way to ask who it was waiting for.

One run blocked its root for 2702 of its 6240 seconds that way — three separate
waits on the same three children holding the same two files — and reached its
deadline having submitted once. The wait itself was reasonable; being unable to
see it or choose against it was not.

`deliveries` answers the two questions the parent needed and could not ask:
who still owes work, and can what they have already written be folded in now.
"""
from __future__ import annotations

from typing import Any

_DELIVERIES_DESC = """See which of your children still owe you work, and fold in what is already written.

Call this before `submit` when you have children running. `submit` waits for them
on your behalf, but that wait is bounded and spends a budget for the whole run —
once it is gone, a submission goes out describing whatever is on disk. Knowing
who is holding what lets you decide rather than find out afterwards.

Each outstanding child reports its status and the files it has changed but not
handed over. From there you can:
  - wait, if a child is close to finishing something worth having;
  - `collect: true`, to merge everything already written to disk right now,
    without waiting for anyone to finish — each child's changes are still
    verified before they are kept, exactly as when they deliver themselves;
  - `kill` a child that is not converging, which releases its work to be folded
    in, and submit.

`collect` is safe to call repeatedly: a child whose changes have not moved since
the last fold-in is skipped."""


async def meta_deliveries(args: dict, agent: Any, runtime: Any) -> dict:
    if not runtime._merge_active():
        return {
            "error": (
                "This run has no shared submission path, so children do not deliver "
                "files and there is nothing outstanding to report."
            )
        }

    collect = bool((args or {}).get("collect", False))
    outstanding = runtime._outstanding_deliveries(agent)

    result: dict[str, Any] = {
        "outstanding": {
            cid: {
                "status": info["status"],
                "undelivered_files": info["undelivered_files"],
            }
            for cid, info in sorted(outstanding.items())
        },
        "wait_budget_remaining_seconds": max(
            0.0,
            runtime._AGGREGATE_WAIT_SECONDS
            - runtime._aggregate_wait_spent.get(agent.id, 0.0),
        ),
    }

    if collect:
        # The same fold-in `submit` performs after its wait, reachable without
        # the wait. Each child's diff is still verified against the merged
        # result before it is kept, so this is not a way around the gate.
        try:
            merged = await runtime._merge_promote_pending()
        except Exception as exc:
            return {**result, "collect_error": str(exc)[:300]}
        result["collected"] = merged
        result["outstanding_after_collect"] = sorted(
            runtime._outstanding_deliveries(agent)
        )

    if not result["outstanding"]:
        result["note"] = (
            "Nothing is outstanding: every child has either finished and handed "
            "over its work or is parked with nothing pending. A submission now "
            "describes the whole workspace."
        )
    elif result["wait_budget_remaining_seconds"] <= 0:
        result["note"] = (
            "The run's wait budget is spent, so `submit` will no longer block for "
            "these children. Either collect what they have written, kill the ones "
            "that are not converging, or accept that their work is not included."
        )
    return result


DELIVERY_TOOLS: dict[str, dict[str, Any]] = {
    "deliveries": {
        "handler": meta_deliveries,
        "is_meta": True,
        "schema": {
            "type": "function",
            "function": {
                "name": "deliveries",
                "description": _DELIVERIES_DESC,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collect": {
                            "type": "boolean",
                            "description": (
                                "Merge everything children have already written to "
                                "disk, without waiting for them to finish. Each "
                                "child's changes are still verified before being "
                                "kept. Default false (report only)."
                            ),
                        }
                    },
                    "required": [],
                },
            },
        },
    }
}
