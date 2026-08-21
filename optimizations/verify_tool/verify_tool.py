"""Agent-owned verification: a check the runtime can reproduce.

Agents already measure their own work constantly — in one ann_vector_search_qps
run, 39 of 40 shell calls were benchmark runs. The problem was never that they
do not verify; it is that each measurement was a throwaway `python -c` whose
verdict survived only as prose in a message. Nobody could reproduce it, and the
artifact that actually got submitted — a config merged from three children —
was never run by anyone before it replaced a working solution.

`verify` turns a check into a runtime-owned object: the agent declares the
command, the runtime executes it and reads a machine-readable verdict, and the
command is stored with its working root replaced by a placeholder so the runtime
can later re-run the very same check against a merged result.
"""
from __future__ import annotations

import time
from typing import Any

_VERIFY_DESC = """Run your own check on your work and register it with the runtime.

Use this whenever you have something worth measuring: a build, a test suite, the
task's own benchmark. The runtime runs the command in your working directory and
keeps it, so it can re-run the exact same check later — including against the
merged result of several agents' work, which is the state that actually gets
submitted and the one nobody thinks to measure.

Your command MUST end by printing a single line of JSON with the verdict:
    {"ok": true, "metric": 3760.0}
`ok` is correctness (tests pass, constraints such as a recall threshold are met);
`metric` is the number to maximize (throughput, score, accuracy). Print `ok:
false` when the artifact is broken — a failing check always loses, so no metric,
however good, can carry a broken artifact into the submission.

The command must REGENERATE what it measures rather than read a result file an
earlier run left behind. The runtime re-runs it against the merged submission
path, which carries only the files being submitted — not your benchmark output,
not datasets you downloaded — so a check that reads `results/...` produces a
verdict in your copy and nothing at all there. Such a check is refused: it would
make the gate reject every merge for the rest of the run, which is exactly how
one run threw away its best result.

The command must actually execute the build, tests or benchmark. Do not register
a constant `echo`/`printf` verdict. Pipelines run with `pipefail`, so do not mask
a failing build with `|| true` or otherwise replace its exit status.

A registered check is what makes your result evidence rather than a claim. A
number you only state in a message is not verifiable by anyone else, and results
that nobody else can reproduce have already cost this system a whole run."""


async def meta_verify(args: dict, agent: Any, runtime: Any) -> dict:
    command = str((args or {}).get("command") or "").strip()
    if not command:
        return {"error": "command is required: the shell command that checks your work"}
    if runtime._verification_command_is_constant(command):
        reason = (
            "it only prints a fixed verdict and does not execute a build, test, "
            "or benchmark."
        )
        agent._verify_failures = getattr(agent, "_verify_failures", 0) + 1
        agent._verify_last_failure = reason
        runtime._emit(agent.id, "verify_rejected", {
            "reason": "constant_verdict",
            "command": command[:300],
        })
        return {
            "ok": False,
            "metric": None,
            "measured": "none",
            "seconds": 0.0,
            "exit_code": None,
            "timed_out": False,
            "registered": False,
            "note": f"Not registered: {reason}",
            "output": "",
        }
    timeout = float((args or {}).get("timeout") or 900)
    higher_is_better = bool((args or {}).get("higher_is_better", True))
    label = str((args or {}).get("name") or "").strip()

    target = str((args or {}).get("target") or "auto").lower()
    if target not in ("auto", "own", "submission"):
        return {"error": "target must be 'auto', 'own' or 'submission'"}
    # An agent with nothing of its own in its copy is measuring somebody else's
    # work, so the thing it means to measure is the shared submission. One that
    # has been editing means its own copy. Read off the copy, not off a role.
    if target == "auto":
        own_changed, own_deleted = runtime._merge_own_changes(agent)
        target = "own" if (own_changed or own_deleted) else "submission"

    portable = runtime._verify_portable_command(command, agent)
    if target == "submission" and runtime._merge_active() and runtime._merge_target():
        workdir = runtime._merge_target()
        # Under the lock, so a promote cannot land halfway through the run and
        # leave the measurement describing a tree that never existed.
        async with runtime._merge_get_lock():
            result = await runtime._run_verification(portable, workdir, timeout)
    else:
        workdir = runtime._verify_working_root(agent) or agent.workspace
        result = await runtime._run_verification(portable, workdir, timeout)

    if result["metric"] is None and result["ok"]:
        result["ok"] = False
        result["note"] = (
            "The command succeeded but printed no verdict line, so there is nothing "
            'to record. End it with a line like {"ok": true, "metric": 123.4}.'
        )
    if not result["ok"]:
        # Remember why, so the follow-up reminder can name the actual problem
        # instead of repeating the generic ask.
        if result["timed_out"]:
            reason = f"it hit the {timeout:.0f}s timeout before finishing."
        elif result["metric"] is None and result["exit_code"] == 0:
            reason = "it ran fine but printed no verdict line the runtime could read."
        else:
            tail = " ".join((result["output"] or "").split())[-200:]
            reason = f"it exited with code {result['exit_code']}: {tail}"
        agent._verify_failures = getattr(agent, "_verify_failures", 0) + 1
        agent._verify_last_failure = reason
    # A check passed where the agent ran it, but the gate re-runs it against the
    # merged submission path. Registration is earned there, not here: a check
    # that cannot even produce a verdict against that tree makes the gate answer
    # "no" to every promotion for the rest of the run.
    unreproducible: dict | None = None
    if result["ok"] and target == "own" and runtime._merge_active():
        probe = await runtime._verify_probe_at_target(portable, timeout)
        if probe is not None and not probe.get("readable"):
            unreproducible = probe

    if result["ok"] and unreproducible is not None:
        tail = " ".join((unreproducible["output"] or "").split())[-300:]
        agent._verify_failures = getattr(agent, "_verify_failures", 0) + 1
        agent._verify_last_failure = (
            "it cannot run against the shared submission path, only against your "
            f"own copy: {tail}"
        )
        runtime._emit(agent.id, "verify_unreproducible", {
            "name": label,
            "exit_code": unreproducible["exit_code"],
            "timed_out": unreproducible["timed_out"],
            "output_tail": tail,
        })
        return {
            "ok": result["ok"],
            "metric": result["metric"],
            "measured": target,
            "seconds": result["seconds"],
            "exit_code": result["exit_code"],
            "timed_out": result["timed_out"],
            "registered": False,
            "note": (
                "Not registered: it passed in your own copy but could not produce a "
                "verdict against the shared submission path, which is where the "
                "runtime has to re-run it. Your copy holds files the merge does not "
                "carry — benchmark output under results/, datasets you downloaded — "
                "so a check that reads them cannot speak for the merged result. Make "
                "the check regenerate what it measures instead of reading an artifact "
                f"from an earlier run. It failed there with: {tail}"
            ),
            "output": result["output"][-2000:],
        }

    if result["ok"]:
        spec = {
            "command": portable,
            "timeout": timeout,
            "higher_is_better": higher_is_better,
            "name": label,
            "agent": agent.id,
            "at": time.time(),
        }
        agent._verify_spec = spec
        runtime._verify_register_spec(spec)
        rank = runtime._verify_rank(result["ok"], result["metric"], higher_is_better)
        # A check the parent ran in the submission path measures the submission
        # itself, so it is the state worth keeping if everything later regresses.
        if (
            rank is not None
            and runtime._merge_active()
            and not getattr(agent, "_merge_copy", None)
        ):
            runtime._record_verified_best(result["metric"], rank)

    runtime._verify_note_metric(
        portable, result["metric"], at_target=(target == "submission")
    )
    runtime._emit(agent.id, "verify", {
        "name": label,
        "ok": result["ok"],
        "metric": result["metric"],
        "seconds": result["seconds"],
        "registered": bool(result["ok"]),
        "target": target,
    })
    return {
        "ok": result["ok"],
        "metric": result["metric"],
        "measured": target,
        "seconds": result["seconds"],
        "exit_code": result["exit_code"],
        "timed_out": result["timed_out"],
        "registered": bool(result["ok"]),
        "note": result.get("note") or (
            "Registered. The runtime will re-run this check against merged results."
            if result["ok"] else
            "Not registered: a check has to pass before it can speak for your work."
        ),
        "output": result["output"][-2000:],
    }


VERIFY_TOOLS: dict[str, dict[str, Any]] = {
    "verify": {"handler": meta_verify, "is_meta": True, "schema": {"type": "function", "function": {
        "name": "verify",
        "description": _VERIFY_DESC,
        "parameters": {"type": "object", "properties": {
            "command": {
                "type": "string",
                "description": (
                    "Shell command run in your working directory. Must print a final "
                    'JSON line with the verdict, e.g. {"ok": true, "metric": 3760.0}.'
                ),
            },
            "name": {"type": "string", "description": "Short label for this check. Optional."},
            "timeout": {
                "type": "number",
                "description": "Seconds before the check is killed (default 900).",
            },
            "higher_is_better": {
                "type": "boolean",
                "description": "Whether a larger metric is better (default true).",
            },
            "target": {
                "type": "string",
                "enum": ["auto", "own", "submission"],
                "description": (
                    "What to measure. 'own' is your working copy; 'submission' is the "
                    "shared submission as it stands right now, measured while no other "
                    "agent can change it. Defaults to whichever fits your assignment."
                ),
            },
        }, "required": ["command"]},
    }}},
}
