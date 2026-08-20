"""Read the run's measurements back, and return to a state that measured well.

An agent optimizing something re-derives its own history from the filesystem,
which does not hold it: one run wrote a configuration, ran it under a shell
timeout too short to finish, and so never learned what that configuration
scored. It then narrowed the configuration to fit the timeout, deleting the
parameters that had produced the best result on record.

The runtime already sees every judge verdict and can fingerprint the state each
one applies to, so the history exists — it just had no reader and no way back.
This is both.
"""
from __future__ import annotations

import time
from typing import Any

from nanoma.core import Agent, Runtime

_DESC = (
    "What this run has measured so far, newest first: each entry is a score and "
    "the state of the submission path it was measured on. Use it before changing "
    "the submission path, and to go back to a state that scored better than the "
    "one you have now — pass restore with a state id to put that state back."
)


async def meta_experiments(
    args: dict[str, Any], agent: Agent, runtime: Runtime
) -> dict[str, Any]:
    restore = str(args.get("restore") or "").strip()
    limit = int(args.get("limit") or 20)

    entries = runtime._ledger_entries()
    if not entries:
        return {
            "measurements": [],
            "note": (
                "nothing measured yet. Scores are recorded when the judge returns a "
                "verdict, so the first submission is what fills this in."
            ),
        }

    if restore:
        known = {e.get("state") for e in entries}
        if restore not in known:
            return {
                "restored": False,
                "reason": f"no measurement was recorded for state {restore}",
                "states": sorted(s for s in known if s),
            }
        if not runtime._ledger_restore(restore):
            return {
                "restored": False,
                "reason": (
                    f"state {restore} was measured but not kept — only states that "
                    "improved on the record are snapshotted, and states too large to "
                    "copy are not"
                ),
            }
        runtime._emit(agent.id, "ledger_restore_requested", {"state": restore})
        return {
            "restored": True,
            "state": restore,
            "note": (
                "the submission path now holds that state. It is worth re-measuring "
                "before submitting: anything outside the submission path is unchanged."
            ),
        }

    target = runtime._merge_target()
    current = (
        runtime._ledger_digest(runtime._merge_scope_signature(target))
        if target is not None
        else None
    )
    best = runtime._ledger_best()
    now = time.time()

    return {
        "current_state": current,
        "current_state_measured": bool(runtime._ledger_state_metrics(current)),
        "best": (
            {
                "state": best.get("state"),
                "metric": best.get("metric"),
                "round": best.get("round"),
                "is_current": best.get("state") == current,
                "restorable": bool(
                    best.get("state")
                    and runtime._ledger_snapshot_dir(best["state"]).is_dir()
                ),
            }
            if best is not None
            else None
        ),
        "measurements": [
            {
                "metric": e.get("metric"),
                "state": e.get("state"),
                "source": e.get("source"),
                "round": e.get("round"),
                "valid": e.get("valid"),
                "is_current": e.get("state") == current,
                "seconds_ago": round(now - float(e.get("at") or now)),
            }
            for e in sorted(entries, key=lambda e: float(e.get("at") or 0), reverse=True)[
                :limit
            ]
        ],
    }


LEDGER_TOOLS: dict[str, dict[str, Any]] = {
    "experiments": {
        "handler": meta_experiments,
        "is_meta": True,
        "schema": {
            "type": "function",
            "function": {
                "name": "experiments",
                "description": _DESC,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "restore": {
                            "type": "string",
                            "description": (
                                "State id to put back into the submission path. "
                                "Omit to just read the record."
                            ),
                        },
                        "limit": {
                            "type": "number",
                            "description": "How many measurements to return (default 20).",
                        },
                    },
                },
            },
        },
    },
}
