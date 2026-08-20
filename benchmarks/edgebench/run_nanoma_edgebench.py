#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import tarfile
import time
from pathlib import Path

from nanoma import Runtime, RuntimeConfig


def _runtime_prompt(prompt: str, task_cwd: Path, *, iteration: int, wall_seconds: float, deadline: float) -> str:
    if wall_seconds > 0:
        remaining = max(0.0, deadline - time.time())
        lifecycle = f"""12-hour benchmark protocol:
- This SForge task is running under a {wall_seconds:.0f}s wall-clock benchmark budget.
- Current NanoMA improvement iteration: {iteration}; approximate remaining runner budget: {remaining:.0f}s.
- Do not treat one candidate submission as the end of the benchmark. Use this iteration to inspect the workspace, improve the current solution, run focused validation, and submit a better candidate when meaningful.
- When this iteration is complete, call set_status(status="done", result="iteration complete"). The outer NanoMA EdgeBench runner will start another iteration until the wall-clock budget is nearly exhausted.
"""
    else:
        lifecycle = "- When finished, call set_status(status=\"done\", result=\"task complete\")."

    return f"""You are solving an EdgeBench task inside an SForge work container.

SForge task directory: {task_cwd}
NanoMA private workspaces are separate from the SForge task directory. Use shell commands such as `cd {task_cwd} && ...` when you need to inspect or modify the benchmark workspace.

Important protocol:
- Solve the task described below.
- Work in the SForge task directory unless you intentionally need private scratch space.
- Submit progress when you have a meaningful candidate. In NanoMA, the `submit` tool is wired to the official SForge `sforge-submit`; shell `cd {task_cwd} && sforge-submit` is also valid.
- Do not inspect judge internals, hidden tests, reward files, or solution files.
- Use task_create when you reach a fresh planning node. The runtime asks this
  same model whether that phase should fan out and creates children through its
  internal executor; direct spawn tools are intentionally not exposed to you.
{lifecycle}

Official EdgeBench/SForge prompt:

{prompt}
    """


_COOLDOWN_RE = re.compile(r"wait\s+(\d+)s", re.IGNORECASE)


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _cooldown_wait_seconds(text: str) -> int | None:
    match = _COOLDOWN_RE.search(text or "")
    if not match:
        return None
    try:
        return max(0, int(match.group(1)))
    except ValueError:
        return None


def _submission_budget_exhausted(text: str) -> bool:
    lowered = (text or "").lower()
    return "submission limit reached" in lowered or "submission budget exhausted" in lowered


def _requested_seed_paths() -> list[str]:
    raw = os.environ.get("NANOMA_EDGE_SEED_PATHS", "").replace("|", ",")
    paths: list[str] = []
    for item in raw.split(","):
        candidate = item.strip().replace("\\", "/")
        if candidate.startswith("/") or ".." in Path(candidate).parts:
            raise ValueError(f"unsafe seed path: {item!r}")
        normalized = candidate.lstrip("./")
        if not normalized:
            continue
        paths.append(normalized)
    return list(dict.fromkeys(paths))


def _stage_seed_archive(task_cwd: Path, workspace: Path) -> tuple[dict, list[tuple[Path, Path]]]:
    archive_value = os.environ.get("NANOMA_EDGE_SEED_ARCHIVE_CONTAINER", "").strip()
    paths = _requested_seed_paths()
    expected_count = max(
        0,
        int(os.environ.get("NANOMA_EDGE_SEED_EXPECTED_COUNT", "0") or 0),
    )
    if expected_count and len(paths) != expected_count:
        return {
            "enabled": True,
            "accepted": False,
            "paths": paths,
            "expected_count": expected_count,
            "error": f"expected {expected_count} seed paths, received {len(paths)}",
        }, []
    if not archive_value or not paths:
        return {"enabled": False, "accepted": False, "paths": []}, []
    archive = Path(archive_value)
    if not archive.is_file():
        return {
            "enabled": True,
            "accepted": False,
            "paths": paths,
            "error": f"seed archive not found: {archive}",
        }, []

    task_root = task_cwd.resolve()
    backup_root = workspace / "seed-original"
    backup_root.mkdir(parents=True, exist_ok=True)
    backups: list[tuple[Path, Path]] = []
    applied: list[str] = []
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            regular_members = [member for member in bundle.getmembers() if member.isfile()]
            for relative in paths:
                matches = [
                    member
                    for member in regular_members
                    if member.name.replace("\\", "/").lstrip("./") == relative
                    or member.name.replace("\\", "/").lstrip("./").endswith("/" + relative)
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"expected exactly one archive member for {relative}, found {len(matches)}"
                    )
                target = (task_root / relative).resolve()
                target.relative_to(task_root)
                if not target.is_file():
                    raise FileNotFoundError(f"seed target does not exist: {target}")
                backup = backup_root / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)
                source = bundle.extractfile(matches[0])
                if source is None:
                    raise ValueError(f"could not read seed member for {relative}")
                target.write_bytes(source.read())
                backups.append((target, backup))
                applied.append(relative)
    except Exception as exc:
        for target, backup in reversed(backups):
            shutil.copy2(backup, target)
        return {
            "enabled": True,
            "accepted": False,
            "paths": paths,
            "applied": applied,
            "error": f"{type(exc).__name__}: {exc}",
        }, []
    return {
        "enabled": True,
        "accepted": False,
        "paths": paths,
        "applied": applied,
        "archive": str(archive),
    }, backups


async def _prepare_seed_archive(task_cwd: Path, workspace: Path) -> dict:
    target_root_value = os.environ.get("NANOMA_EDGE_SEED_TARGET_ROOT", "").strip()
    target_root = Path(target_root_value) if target_root_value else task_cwd
    record, backups = _stage_seed_archive(target_root, workspace)
    record["target_root"] = str(target_root)
    if not backups:
        return record
    timeout = max(
        30.0,
        float(os.environ.get("NANOMA_EDGE_SEED_BUILD_TIMEOUT", "900") or 900),
    )
    started = time.time()
    proc = await asyncio.create_subprocess_exec(
        "lake",
        "build",
        cwd=str(target_root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    timed_out = False
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        timed_out = True
        proc.kill()
        stdout, stderr = await proc.communicate()
    exit_code = -1 if timed_out else int(proc.returncode or 0)
    accepted = exit_code == 0
    if not accepted:
        for target, backup in reversed(backups):
            shutil.copy2(backup, target)
    record.update({
        "accepted": accepted,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "elapsed_seconds": round(time.time() - started, 3),
        "outcome": "seed_green" if accepted else "seed_rejected_baseline_restored",
        "stdout_tail": stdout.decode(errors="replace")[-12000:],
        "stderr_tail": stderr.decode(errors="replace")[-12000:],
    })
    return record


async def _run_sforge_submit_once(task_cwd: Path, *, timeout: float) -> dict:
    try:
        proc = await asyncio.create_subprocess_exec(
            "sforge-submit",
            cwd=str(task_cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return {
            "exit_code": proc.returncode,
            "stdout": stdout.decode(errors="replace")[-4000:],
            "stderr": stderr.decode(errors="replace")[-4000:],
        }
    except Exception as exc:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


async def _sforge_submit_with_retry(task_cwd: Path, *, attempts: int, timeout: float) -> dict:
    attempts = max(1, attempts)
    history: list[dict] = []
    for attempt in range(1, attempts + 1):
        result = await _run_sforge_submit_once(task_cwd, timeout=timeout)
        result["attempt"] = attempt
        # Store a detached snapshot. Appending result itself and then assigning
        # result["attempts"] = history makes the submit payload self-referential
        # and crashes JSON serialization in the runtime.
        history.append(dict(result))

        combined = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
        if int(result.get("exit_code") or 0) == 0:
            result["attempts"] = history
            return result
        if _submission_budget_exhausted(combined):
            result["attempts"] = history
            return result

        wait_s = _cooldown_wait_seconds(combined)
        if wait_s is None or attempt >= attempts:
            result["attempts"] = history
            return result
        await asyncio.sleep(wait_s + 2)

    return history[-1] if history else {"exit_code": -1, "stdout": "", "stderr": "no submit attempts"}


async def _final_sforge_submit(task_cwd: Path, runtime=None) -> dict:
    """The submission that closes an iteration, through the runtime's own gate.

    This is the decisive submission of a round — it is the one the agent's last
    state is scored on — and it used to be the only one that answered to nothing.
    The runtime installs its submission gates on the tool table it hands the
    model each turn, so calling `sforge-submit` from out here skipped all of
    them: the regression check that refuses a state already measured worse, the
    incomplete-aggregate check, and the calibration step that reads the judge's
    verdict back into the experiment ledger. On the 2026-07-29 run this path
    shipped `agent-2` and its verdict was never recorded, so the run's own
    record of what it had scored was missing its most recent entry.

    Falls back to submitting directly when the runtime cannot do it, because a
    round that fails to submit at all scores nothing.
    """
    if runtime is not None and hasattr(runtime, "submit_official"):
        try:
            result = await runtime.submit_official(reason="iteration final submit")
        except Exception as exc:  # the fallback below is the point
            print(f"NANOMA_FINAL_SUBMIT_GATE_ERROR={exc}", flush=True)
            result = None
        if isinstance(result, dict):
            if result.get("blocked"):
                # A refusal is an outcome, not a failure to submit: the gate has
                # decided this state is worse than one already on record, and
                # re-submitting it outside the gate is exactly the bypass this
                # change removes.
                print(
                    "NANOMA_FINAL_SUBMIT_BLOCKED="
                    + json.dumps(result, ensure_ascii=False, default=str),
                    flush=True,
                )
            return result

    attempts = int(os.environ.get("NANOMA_EDGE_SUBMIT_RETRY_ATTEMPTS", "4") or 4)
    return await _sforge_submit_with_retry(task_cwd, attempts=attempts, timeout=900)


def _failed_before_work(stats: dict) -> bool:
    overview = stats.get("overview", {})
    agents = stats.get("agents", {})
    status_breakdown = agents.get("status_breakdown", {})
    return (
        int(overview.get("total_tokens") or 0) == 0
        and int(agents.get("total_spawned") or 0) > 0
        and int(status_breakdown.get("failed") or 0) == int(agents.get("total_spawned") or 0)
    )


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


async def _run_single_iteration(args: argparse.Namespace) -> int:
    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    task_cwd = Path(args.task_cwd)
    workspace = Path(args.workspace) if args.workspace else task_cwd / ".nanoma-task-work"
    log_dir = Path(args.log_dir) if args.log_dir else workspace / "logs"
    workspace.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    wall_seconds = float(args.wall_seconds)
    deadline = float(args.deadline_epoch)
    iteration = int(args.iteration)
    iter_log_dir = log_dir / f"iter_{iteration:03d}"
    iter_log_dir.mkdir(parents=True, exist_ok=True)
    iteration_results = log_dir / "nanoma_iteration_results.jsonl"

    seed_record = await _prepare_seed_archive(task_cwd, workspace)
    (iter_log_dir / "seed_record.json").write_text(
        json.dumps(seed_record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("NANOMA_SEED_JSON=" + json.dumps(seed_record, ensure_ascii=False), flush=True)
    seed_submit_result = None
    submit_seed = os.environ.get("NANOMA_EDGE_SUBMIT_SEED", "").strip().lower()
    if seed_record.get("accepted") and submit_seed in {"1", "true", "yes", "on"}:
        seed_submit_result = await _sforge_submit_with_retry(
            task_cwd,
            attempts=int(os.environ.get("NANOMA_EDGE_SEED_SUBMIT_ATTEMPTS", "3") or 3),
            timeout=float(os.environ.get("NANOMA_EDGE_SEED_SUBMIT_TIMEOUT", "900") or 900),
        )
        print(
            "NANOMA_SEED_SUBMIT_JSON="
            + json.dumps(seed_submit_result, ensure_ascii=False, default=str),
            flush=True,
        )

    async def edgebench_submit(args: dict, agent, runtime) -> dict:
        attempts = int(os.environ.get("NANOMA_EDGE_TOOL_SUBMIT_RETRY_ATTEMPTS", "3") or 3)
        timeout = float(os.environ.get("NANOMA_EDGE_TOOL_SUBMIT_TIMEOUT", "900") or 900)
        result = await _sforge_submit_with_retry(task_cwd, attempts=attempts, timeout=timeout)
        result["submitted"] = "sforge-submit"
        result["reason"] = args.get("reason", "")
        if int(result.get("exit_code") or 0) != 0:
            result["error"] = result.get("stderr") or result.get("stdout") or "sforge-submit failed"
        return result

    resume_probe = Path(args.resume_probe).expanduser().resolve() if args.resume_probe else None
    if resume_probe is not None and not resume_probe.is_file():
        raise FileNotFoundError(f"NanoMA resume probe not found: {resume_probe}")
    auto_checkpoint_enabled = _env_enabled("NANOMA_AUTO_CHECKPOINT_ENABLED")
    auto_checkpoint_value = os.environ.get("NANOMA_AUTO_CHECKPOINT_DIR", "").strip()
    auto_checkpoint_dir = (
        Path(auto_checkpoint_value).expanduser()
        if auto_checkpoint_value
        else (Path("/tmp/nanoma-checkpoints") if auto_checkpoint_enabled else None)
    )

    config = RuntimeConfig(
        budget=args.budget,
        time_limit=args.time_limit,
        max_agents=1_000_000,
        max_depth=1_000_000,
        max_concurrent_llm=int(os.environ.get("NANOMA_MAX_CONCURRENT_LLM", "50")),
        default_model=args.model,
        allowed_models=[args.model],
        workspace_root=workspace,
        workspace_extra_roots=[task_cwd],
        log_dir=iter_log_dir,
        probe_dir=resume_probe.parents[1] if resume_probe is not None else None,
        auto_checkpoint_enabled=auto_checkpoint_enabled,
        auto_checkpoint_dir=auto_checkpoint_dir,
        probe_resume_instruction=os.environ.get(
            "NANOMA_PROBE_RESUME_INSTRUCTION",
            (
                "Resume the interrupted task from this saved node. The previous LLM request did not "
                "produce a committed response; inspect the latest tool result, then continue without "
                "repeating completed work."
                if resume_probe is not None
                else ""
            ),
        ),
        tool_policy_mode=os.environ.get("NANOMA_TOOL_POLICY_MODE", "off"),
        tool_policy_log_events=True,
        extra_tools={
            "submit": {
                "handler": edgebench_submit,
                "is_meta": True,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "submit",
                        "description": "Run the official EdgeBench SForge submission for the current task directory and return judge feedback.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "reason": {
                                    "type": "string",
                                    "description": "Short note describing the candidate being submitted."
                                },
                                "path": {
                                    "type": "string",
                                    "description": "Optional compatibility field; EdgeBench submits the configured task paths, not this file."
                                }
                            },
                            "required": []
                        },
                    },
                },
            },
        },
        shell_max_timeout=int(os.environ.get("NANOMA_SHELL_MAX_TIMEOUT", "300")),
        shell_max_output=int(os.environ.get("NANOMA_SHELL_MAX_OUTPUT", "40000")),
        file_read_max_chars=int(os.environ.get("NANOMA_FILE_READ_MAX_CHARS", "200000")),
        file_list_max_entries=int(os.environ.get("NANOMA_FILE_LIST_MAX_ENTRIES", "5000")),
        grep_max_results=int(os.environ.get("NANOMA_GREP_MAX_RESULTS", "500")),
        system_extra_instructions=(
            f"EdgeBench task directory is {task_cwd}. "
            f"Use structured ws_* tools or `cd {task_cwd} && ...` for benchmark work. "
            f"Use {workspace} for private NanoMA scratch files. "
            f"Structured ws_* tools may access both {workspace} and {task_cwd}; "
            f"use absolute paths under {task_cwd} when editing the task repository. "
            "Avoid /tmp unless the public task instruction explicitly requires it."
        ),
    )
    runtime = Runtime(config=config)
    iter_record = {
        "iteration": iteration,
        "started_at": time.time(),
        "wall_seconds": wall_seconds,
        "remaining_seconds": max(0.0, deadline - time.time()) if wall_seconds > 0 else None,
        "pid": os.getpid(),
        "resume_probe": str(resume_probe) if resume_probe is not None else None,
        "seed": seed_record,
        "seed_submit": seed_submit_result,
    }
    print("NANOMA_ITERATION_START=" + json.dumps(iter_record, ensure_ascii=False), flush=True)

    try:
        if resume_probe is not None:
            result = await runtime.continue_from_probe(resume_probe)
        else:
            result = await runtime.run(
                _runtime_prompt(
                    prompt,
                    task_cwd,
                    iteration=iteration,
                    wall_seconds=wall_seconds,
                    deadline=deadline,
                ),
                model=args.model,
            )
        stats = runtime.stats()
        (iter_log_dir / "nanoma_stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (iter_log_dir / "nanoma_result_preview.txt").write_text(str(result)[-8000:], encoding="utf-8")
        failed_before_work = _failed_before_work(stats)
        submit_result = None
        branch_manifest = None
        if failed_before_work:
            print(
                "NANOMA_ITERATION_SKIP_SUBMIT="
                + json.dumps({"iteration": iteration, "reason": "llm_failed_before_work"}, ensure_ascii=False),
                flush=True,
            )
        else:
            submit_result = await _final_sforge_submit(task_cwd, runtime)

        # Freeze candidates only after the closing submission has returned. The
        # selected snapshot must describe exactly what was finally delivered,
        # while every child still points at its untouched private worktree. This
        # is archival only: no branch score is visible to this runtime process.
        if hasattr(runtime, "preserve_final_branches"):
            try:
                branch_manifest = runtime.preserve_final_branches(
                    iter_log_dir / "candidate_branches"
                )
                print(
                    "NANOMA_BRANCH_MANIFEST_JSON="
                    + json.dumps({
                        "path": branch_manifest.get("output_dir"),
                        "logical_candidates": branch_manifest.get("logical_candidate_count", 0),
                        "unique_artifacts": branch_manifest.get("unique_artifact_count", 0),
                        "selected_candidate_id": branch_manifest.get("selected_candidate_id"),
                    }, ensure_ascii=False, default=str),
                    flush=True,
                )
            except Exception as exc:
                branch_manifest = {
                    "preserved": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(
                    "NANOMA_BRANCH_PRESERVE_ERROR=" + branch_manifest["error"],
                    file=sys.stderr,
                    flush=True,
                )

        iter_record.update({
            "finished_at": time.time(),
            "ok": True,
            "failed_before_work": failed_before_work,
            "stats": stats,
            "submit": submit_result,
            "branches": branch_manifest,
            "result_preview": str(result)[-2000:],
        })
        _append_jsonl(iteration_results, iter_record)
        print("NANOMA_ITERATION_JSON=" + json.dumps(iter_record, ensure_ascii=False, default=str), flush=True)
        if submit_result is not None:
            print("NANOMA_ITERATION_SUBMIT_JSON=" + json.dumps(submit_result, ensure_ascii=False, default=str), flush=True)
        return 0 if not failed_before_work else 3
    except Exception as exc:
        iter_record.update({
            "finished_at": time.time(),
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        })
        _append_jsonl(iteration_results, iter_record)
        print(f"NanoMA EdgeBench iteration {iteration} failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        print("NANOMA_ITERATION_JSON=" + json.dumps(iter_record, ensure_ascii=False, default=str), flush=True)
        return 2


async def _run_outer_loop(args: argparse.Namespace) -> int:
    task_cwd = Path(args.task_cwd)
    workspace = Path(args.workspace) if args.workspace else task_cwd / ".nanoma-task-work"
    log_dir = Path(args.log_dir) if args.log_dir else workspace / "logs"
    workspace.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    wall_seconds = float(os.environ.get("NANOMA_EDGE_WALL_SECONDS", "0") or 0)
    final_safety = float(os.environ.get("NANOMA_EDGE_FINAL_SAFETY_SECONDS", "90") or 90)
    retry_sleep = float(os.environ.get("NANOMA_EDGE_FAILED_LLM_SLEEP_SECONDS", "30") or 30)
    resume_probe = args.resume_probe or os.environ.get("NANOMA_EDGE_RESUME_PROBE", "").strip()
    deadline = time.time() + wall_seconds if wall_seconds > 0 else 0.0
    process_results = log_dir / "nanoma_runner_process_results.jsonl"

    print(
        "NANOMA_EDGE_RUNNER_START="
        + json.dumps({
            "pid": os.getpid(),
            "model": args.model,
            "wall_seconds": wall_seconds,
            "workspace": str(workspace),
            "log_dir": str(log_dir),
        }, ensure_ascii=False),
        flush=True,
    )

    iteration = 0
    last_return_code = 0
    barren = 0
    barren_backoff_cap = float(
        os.environ.get("NANOMA_EDGE_BARREN_BACKOFF_CAP_SECONDS", "600") or 600
    )
    while True:
        if wall_seconds > 0 and time.time() >= deadline - final_safety:
            print(
                "NANOMA_EDGE_WALL_BUDGET_EXHAUSTED="
                + json.dumps({"iterations": iteration, "wall_seconds": wall_seconds}, ensure_ascii=False),
                flush=True,
            )
            break

        iteration += 1
        iter_log_dir = log_dir / f"iter_{iteration:03d}"
        iter_log_dir.mkdir(parents=True, exist_ok=True)
        remaining = max(0.0, deadline - time.time() - final_safety) if wall_seconds > 0 else None
        cmd = [
            sys.executable,
            __file__,
            "--single-iteration",
            "--prompt-file", args.prompt_file,
            "--task-cwd", args.task_cwd,
            "--model", args.model,
            "--workspace", str(workspace),
            "--log-dir", str(log_dir),
            "--budget", str(args.budget),
            "--time-limit", str(args.time_limit),
            "--iteration", str(iteration),
            "--wall-seconds", str(wall_seconds),
            "--deadline-epoch", str(deadline),
        ]
        if iteration == 1 and resume_probe:
            cmd.extend(["--resume-probe", resume_probe])
        launch_record = {
            "iteration": iteration,
            "started_at": time.time(),
            "remaining_seconds": remaining,
            "resume_probe": resume_probe if iteration == 1 and resume_probe else None,
        }
        print("NANOMA_EDGE_ITERATION_LAUNCH=" + json.dumps(launch_record, ensure_ascii=False), flush=True)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(task_cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            if remaining is not None:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=max(1.0, remaining))
            else:
                stdout, stderr = await proc.communicate()
        except asyncio.TimeoutError:
            proc.kill()
            stdout, stderr = await proc.communicate()
            last_return_code = 124
            record = {
                **launch_record,
                "finished_at": time.time(),
                "exit_code": -9,
                "timed_out": True,
            }
            _append_jsonl(process_results, record)
            (iter_log_dir / "runner_stdout.txt").write_bytes(stdout[-200000:])
            (iter_log_dir / "runner_stderr.txt").write_bytes(stderr[-200000:])
            print("NANOMA_EDGE_ITERATION_PROCESS_JSON=" + json.dumps(record, ensure_ascii=False), flush=True)
            break

        (iter_log_dir / "runner_stdout.txt").write_bytes(stdout[-200000:])
        (iter_log_dir / "runner_stderr.txt").write_bytes(stderr[-200000:])
        record = {
            **launch_record,
            "finished_at": time.time(),
            "exit_code": proc.returncode,
            "stdout_tail": stdout.decode(errors="replace")[-4000:],
            "stderr_tail": stderr.decode(errors="replace")[-4000:],
        }
        _append_jsonl(process_results, record)
        print("NANOMA_EDGE_ITERATION_PROCESS_JSON=" + json.dumps(record, ensure_ascii=False, default=str), flush=True)
        last_return_code = int(proc.returncode or 0)

        if wall_seconds <= 0:
            break
        if proc.returncode in (2, 3):
            # An iteration that died before doing any work usually means the model
            # endpoint is refusing us, and that lasts minutes to hours. A flat retry
            # turned one such outage into 219 identical crashes that ate a whole 2h
            # budget, so back off further each time while still probing for recovery.
            barren += 1
            backoff = min(retry_sleep * (2 ** (barren - 1)), barren_backoff_cap)
            print(
                "NANOMA_EDGE_ITERATION_BARREN="
                + json.dumps({
                    "iteration": iteration,
                    "consecutive": barren,
                    "exit_code": proc.returncode,
                    "sleeping_seconds": backoff,
                }, ensure_ascii=False),
                flush=True,
            )
            await asyncio.sleep(min(backoff, max(0.0, deadline - time.time() - final_safety)))
        else:
            barren = 0

    return last_return_code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--task-cwd", required=True)
    parser.add_argument("--model", default=os.environ.get("NANOMA_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--workspace", default=os.environ.get("NANOMA_WORKSPACE", ""))
    parser.add_argument("--log-dir", default=os.environ.get("NANOMA_LOG_DIR", ""))
    parser.add_argument("--budget", type=float, default=float(os.environ.get("NANOMA_BUDGET", "100000")))
    parser.add_argument("--time-limit", type=float, default=float(os.environ.get("NANOMA_TIME_LIMIT", "0")))
    parser.add_argument("--single-iteration", action="store_true")
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--wall-seconds", type=float, default=0.0)
    parser.add_argument("--deadline-epoch", type=float, default=0.0)
    parser.add_argument(
        "--resume-probe",
        default=os.environ.get("NANOMA_EDGE_RESUME_PROBE", ""),
    )
    return parser


async def _main() -> int:
    args = _build_parser().parse_args()
    if args.single_iteration:
        return await _run_single_iteration(args)
    return await _run_outer_loop(args)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
