#!/usr/bin/env python3
"""Run two small official RepoZero Hard tasks with NanoMA.

The generation phase exposes only the public task source and the official
black-box executable. Hidden test cases are loaded only after NanoMA exits.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from nanoma import DeliveryContract, DeliveryTree, Runtime, RuntimeConfig
from nanoma.llm import RetryConfig, openai_compatible_call


NODE = Path("/data/workspace/.tools/node-v22.14.0-linux-x64/bin/node")
SYSTEM_LOADER = Path("/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2")

TASKS: dict[str, dict[str, Any]] = {
    "py2js": {
        "benchmark": "RepoZero-Py2JS",
        "difficulty": "Hard",
        "relative_path": "schedule/test2.py",
        "source_asset": "selected_assets/schedule/test2.py",
        "oracle_asset": "selected_assets/extracted/py/test2_executable",
        "testcases": "evaluate/testcases/py2js/testcase_schedule_merged.jsonl",
        "output_source": "output/test2.mjs",
        "official_case_count": 70,
        "selection": "225-byte source; smallest practical Hard task in schedule",
    },
    "c2rust": {
        "benchmark": "RepoZero-C2Rust",
        "difficulty": "Hard",
        "relative_path": "earcut.hpp/tests/test5.cpp",
        "source_asset": "selected_assets/CppLarge/earcut.hpp/tests/test5.cpp",
        "oracle_asset": "selected_assets/extracted/c/test5",
        "testcases": "evaluate/testcases/c2rust/cleaned_test_cases.jsonl",
        "output_source": "output/test5.rs",
        "output_binary": "output/test5",
        "official_case_count": 36,
        "lib_total_lines": 904,
        "covered_ranges": 4,
        "selection": "fewest official cases among the near-boundary Hard candidates",
    },
}

BLOCKED_SHELL_PATTERNS = [
    r"RepoZero/evaluate",
    r"evaluate/testcases",
    r"cleaned_test_cases",
    r"testcase_.*merged",
    r"selection_metadata",
    r"difficulty\.json",
    r"raw\.githubusercontent\.com/mapbox/earcut",
    r"github\.com/mapbox/earcut",
]

PUBLIC_C_PROBES = ["0", "1", "2", "3", "4", "5", "8", "13", "21", "34", "55"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repozero-root", type=Path, default=Path("/data/workspace/RepoZero"))
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--tasks", default="py2js,c2rust")
    parser.add_argument("--time-limit", type=float, default=7200)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--max-concurrent-llm", type=int, default=4)
    parser.add_argument(
        "--seed-c2rust-dir",
        type=Path,
        help="Prior NanoMA workspace containing public candidate test5.rs files",
    )
    parser.add_argument(
        "--inherit-env-pid",
        type=int,
        help="Read only NanoMA API credentials from this already-authorized process",
    )
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def inherit_runtime_credentials(pid: int | None) -> None:
    if pid is None:
        return
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    inherited: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        name = key.decode("utf-8", errors="surrogateescape")
        if name in {"NANOMA_API_KEY", "NANOMA_LLM_BASE_URL"}:
            inherited[name] = value.decode("utf-8", errors="surrogateescape")
    missing = sorted({"NANOMA_API_KEY", "NANOMA_LLM_BASE_URL"} - set(inherited))
    if missing:
        raise RuntimeError(f"Credential source process is missing required keys: {missing}")
    os.environ.update(inherited)


async def smoke_model(model: str) -> None:
    response = await openai_compatible_call(
        [{"role": "user", "content": "Reply with exactly OK."}],
        model,
        tools=None,
        temperature=0.0,
        max_tokens=1024,
        retry_config=RetryConfig(http_timeout=120),
    )
    if not (response.content or "").strip():
        raise RuntimeError("API smoke test returned empty content")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_process(command: list[str], timeout: float = 10) -> dict[str, Any]:
    started = time.time()
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "RUST_BACKTRACE": "0"},
        )
        return {
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "wall_seconds": round(time.time() - started, 4),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "stdout": str(exc.stdout or ""),
            "stderr": str(exc.stderr or ""),
            "timeout": True,
            "wall_seconds": round(time.time() - started, 4),
        }
    except OSError as exc:
        return {
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "os_error": type(exc).__name__,
            "wall_seconds": round(time.time() - started, 4),
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_task(kind: str, spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Path]:
    task_dir = (args.run_dir / kind).resolve()
    input_dir = task_dir / "input"
    output_dir = task_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = input_dir / Path(spec["relative_path"]).name
    shutil.copy2(args.repozero_root / spec["source_asset"], source)

    if kind == "py2js":
        oracle = input_dir / "test2_executable"
        shutil.copy2(args.repozero_root / spec["oracle_asset"], oracle)
        oracle.chmod(oracle.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    else:
        cpp_binary = input_dir / "test5_cpp"
        shutil.copy2(args.repozero_root / spec["oracle_asset"], cpp_binary)
        cpp_binary.chmod(cpp_binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        oracle = input_dir / "oracle"
        oracle.write_text(
            "#!/bin/sh\n"
            f"exec {SYSTEM_LOADER} \"$(dirname \"$0\")/test5_cpp\" \"$@\"\n",
            encoding="utf-8",
        )
        oracle.chmod(0o755)

    return {
        "task_dir": task_dir,
        "source": source,
        "oracle": oracle,
        "output": task_dir / spec["output_source"],
        "binary": task_dir / spec.get("output_binary", "output/unused"),
    }


def validate_c_candidate(source: Path, binary: Path, oracle: Path) -> dict[str, Any]:
    compile_result = run_process(
        ["rustc", "--edition=2021", "-O", str(source), "-o", str(binary)],
        timeout=180,
    )
    probes: list[dict[str, Any]] = []
    passed = 0
    if compile_result.get("returncode") == 0:
        for value in PUBLIC_C_PROBES:
            reference = run_process([str(oracle), value], timeout=10)
            candidate = run_process([str(binary), value], timeout=10)
            matched = (
                reference.get("returncode") == 0
                and candidate.get("returncode") == 0
                and c_outputs_equal(reference.get("stdout", ""), candidate.get("stdout", ""))
            )
            passed += int(matched)
            probes.append({"n": value, "matched": matched})
    return {
        "source": str(source),
        "compile": compile_result,
        "public_probes_passed": passed,
        "public_probes_total": len(PUBLIC_C_PROBES),
        "probes": probes,
    }


def select_c_candidate(
    paths: dict[str, Path],
    *,
    seed_dir: Path | None = None,
    include_workspace: bool = False,
    record_name: str,
) -> dict[str, Any]:
    checkpoint_dir = paths["task_dir"] / "checkpoints"
    build_dir = paths["task_dir"] / "candidate_builds" / record_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[Path] = []
    if paths["output"].is_file():
        candidates.append(paths["output"])
    if seed_dir is not None:
        seed_dir = seed_dir.expanduser().resolve()
        if not seed_dir.is_dir():
            raise FileNotFoundError(f"C2Rust seed directory not found: {seed_dir}")
        for source in sorted(seed_dir.rglob("test5.rs")):
            label = source.parent.name
            destination = checkpoint_dir / f"{label}_test5.rs"
            suffix = 2
            while destination.exists() and destination.read_bytes() != source.read_bytes():
                destination = checkpoint_dir / f"{label}_{suffix}_test5.rs"
                suffix += 1
            if not destination.exists():
                shutil.copy2(source, destination)
            candidates.append(destination)
    if include_workspace:
        candidates.extend(sorted((paths["task_dir"] / "nanoma_workspace").rglob("test5.rs")))

    unique: list[Path] = []
    seen: set[bytes] = set()
    for source in candidates:
        if not source.is_file():
            continue
        content = source.read_bytes()
        if content in seen:
            continue
        seen.add(content)
        unique.append(source)

    validations: list[dict[str, Any]] = []
    for index, source in enumerate(unique):
        binary = build_dir / f"candidate_{index:03d}"
        validations.append(validate_c_candidate(source, binary, paths["oracle"]))
    ranked = sorted(
        validations,
        key=lambda item: (
            item["public_probes_passed"],
            int((item.get("compile") or {}).get("returncode") == 0),
        ),
        reverse=True,
    )
    selected = ranked[0] if ranked else None
    if selected and (selected.get("compile") or {}).get("returncode") == 0:
        selected_source = Path(selected["source"])
        if selected_source.resolve() != paths["output"].resolve():
            shutil.copy2(selected_source, paths["output"])
        selected["final_compile"] = run_process(
            ["rustc", "--edition=2021", "-O", str(paths["output"]), "-o", str(paths["binary"])],
            timeout=180,
        )
    record = {
        "selection_basis": "fixed public black-box probes only; hidden tests were not loaded",
        "public_probes": PUBLIC_C_PROBES,
        "selected": selected,
        "candidates": validations,
        "official_output_exists": paths["output"].is_file(),
        "official_binary_exists": paths["binary"].is_file(),
    }
    atomic_json(paths["task_dir"] / f"{record_name}.json", record)
    return record


def build_prompt(kind: str, spec: dict[str, Any], paths: dict[str, Path]) -> str:
    source_code = paths["source"].read_text(encoding="utf-8")
    shared = f"""NanoMA protocol:
- At the first real planning boundary, call task_create with a short plan. The runtime may fan out
  independent algorithm hypotheses. Every child has the same planning ability and may itself call
  task_create when its own work has genuinely separable uncertainties.
- Keep a working baseline, use children for distinct hypotheses or audits, and reconcile concrete
  evidence before replacing it. Avoid duplicate workers that pursue the same hypothesis.
- You may query the supplied black-box executable with arbitrary inputs. Do not inspect RepoZero
  evaluator code, hidden tests, difficulty metadata, or library implementation source.
- Compile or execute the final artifact yourself and compare it with the oracle on diverse inputs.
- Call set_status(status=\"done\", result=...) only after the required output exists and has been
  tested. Children should deliver concrete implementations, observations, or counterexamples.
"""
    if kind == "py2js":
        return f"""You are a senior Python-to-Node.js migration engineer.

--- Source Code ({paths['source']}) ---
{source_code}
---

Official task requirements:
1. Write pure JavaScript for Node.js using ES Modules. Use import/export; require() and
   module.exports are prohibited. Generated JavaScript files must use the .mjs suffix.
2. Match the source program's CLI argument names, defaults, required fields, logic, numeric
   precision, and printed output.
3. Use no npm packages and do not embed or invoke Python. Node.js built-ins and local .mjs files
   are allowed; local imports must include their complete suffix.
4. This is a black-box implementation. The Python package source is unavailable. Query the
   precompiled executable directly as `{paths['oracle']} --arg value`.
5. Save the final entry file exactly at `{paths['output']}`. Node is available at `{NODE}`.

{shared}"""

    checkpoint_note = ""
    checkpoint_dir = paths["task_dir"] / "checkpoints"
    if checkpoint_dir.is_dir() and any(checkpoint_dir.glob("*.rs")):
        checkpoint_note = f"""
Continuation checkpoint:
- Prior candidate sources are available in `{checkpoint_dir}`. The strongest candidate on a fixed
  public black-box probe set has already been copied to `{paths['output']}` and compiled.
- Continue from these artifacts instead of restarting. Compare concrete functions and observed
  oracle mismatches, then preserve every improvement in the official output path.
- At regular milestones and before waiting for children, copy the best compiling candidate to
  `{paths['output']}` and rebuild `{paths['binary']}`. A partial candidate is preferable to no
  deliverable at the deadline.
"""

    return f"""You are an expert C++-to-Rust migration engineer.

--- Source Code ({paths['source']}) ---
{source_code}
---

Official task requirements:
1. Write pure Rust using the 2021 edition. The program must compile with rustc.
2. Accept exactly the same command-line arguments as the C++ binary. Parse them with std::env::args.
3. Match the source logic, numerical precision, and stdout byte for byte.
4. Use only the Rust standard library; crates.io dependencies are prohibited. Implement required
   earcut behavior yourself without reading the C++ library source.
5. Save the entry source exactly at `{paths['output']}` and compile the executable exactly at
   `{paths['binary']}`.
6. Query the compiled C++ black box directly as `{paths['oracle']} <arguments>`.

{checkpoint_note}

{shared}"""


async def generate(kind: str, spec: dict[str, Any], paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    log_dir = paths["task_dir"] / "nanoma_logs"
    workspace = paths["task_dir"] / "nanoma_workspace"
    log_dir.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)

    config = RuntimeConfig(
        max_agents=sys.maxsize,
        max_depth=sys.maxsize,
        max_concurrent_llm=args.max_concurrent_llm,
        budget=1000.0,
        max_total_tokens=0,
        time_limit=args.time_limit,
        max_turns=args.max_turns,
        allowed_models=[args.model],
        default_model=args.model,
        log_dir=log_dir,
        workspace_root=workspace,
        workspace_extra_roots=[paths["task_dir"]],
        delivery_contract=DeliveryContract(
            target_root=paths["task_dir"],
            trees=(
                DeliveryTree(
                    target="output",
                    candidates=("output", "Py2JS/output"),
                    required=(paths["output"].name,),
                ),
            ),
        ),
        blocked_shell_patterns=BLOCKED_SHELL_PATTERNS,
        shell_max_timeout=300,
        shell_max_output=40000,
        file_read_max_chars=200000,
        file_list_max_entries=5000,
        grep_max_results=500,
        tool_policy_mode="off",
        tool_policy_delivery_enabled=False,
        tool_policy_delivery_prepare_enabled=False,
        tool_policy_delivery_verify_enabled=False,
        system_extra_instructions=(
            f"RepoZero task directory is {paths['task_dir']}. Work only inside this task directory. "
            "The input source and black-box executable are public task materials. Hidden tests, "
            "evaluator internals, difficulty metadata, and original library source are forbidden."
        ),
    )
    runtime = Runtime(config=config)
    started = time.time()
    result = await runtime.run(build_prompt(kind, spec, paths), model=args.model)
    return {
        "model": args.model,
        "wall_seconds": round(time.time() - started, 3),
        "result": result,
        "stats": runtime.stats(),
        "agents": [
            {
                "id": agent.id,
                "parent": agent.parent,
                "depth": agent.depth,
                "status": agent.status,
                "turns": agent._turns,
                "tokens": agent.tokens_consumed,
            }
            for agent in runtime.agents.values()
        ],
    }


def normalize_py_output(text: str) -> list[str]:
    return ["".join(line.split()) for line in text.strip().splitlines() if line.strip()]


def score_py(spec: dict[str, Any], paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    cases = [
        row for row in read_jsonl(args.repozero_root / spec["testcases"])
        if row.get("filename") == spec["relative_path"]
    ]
    failures: list[dict[str, Any]] = []
    passed = 0
    for index, row in enumerate(cases):
        params = {key: value for key, value in row.items() if key != "filename"}
        cli: list[str] = []
        for key, value in params.items():
            cli.extend([f"--{key}", str(value)])
        reference = run_process([str(paths["oracle"]), *cli], timeout=5)
        candidate = run_process([str(NODE), str(paths["output"]), *cli], timeout=5)
        ref_lines = normalize_py_output(reference.get("stdout", ""))
        got_lines = normalize_py_output(candidate.get("stdout", ""))
        matched = (
            reference.get("returncode") == 0
            and candidate.get("returncode") == 0
            and len(ref_lines) == len(got_lines)
            and len(ref_lines) > 0
            and ref_lines == got_lines
        )
        passed += int(matched)
        if not matched and len(failures) < 12:
            failures.append({
                "index": index,
                "params": params,
                "reference": ref_lines[:4],
                "candidate": got_lines[:4],
                "candidate_returncode": candidate.get("returncode"),
                "candidate_stderr": candidate.get("stderr", "")[-1000:],
            })
    return score_record(spec, cases, passed, failures)


def c_outputs_equal(left: str, right: str) -> bool:
    left, right = left.strip(), right.strip()
    if left == right:
        return True
    try:
        return float(left) == float(right)
    except ValueError:
        return False


def score_c(spec: dict[str, Any], paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    compile_result = None
    if paths["output"].is_file():
        compile_result = run_process(
            ["rustc", "--edition=2021", "-O", str(paths["output"]), "-o", str(paths["binary"])],
            timeout=120,
        )
    cases = [
        row for row in read_jsonl(args.repozero_root / spec["testcases"])
        if row.get("file_name") == spec["relative_path"]
    ]
    failures: list[dict[str, Any]] = []
    passed = 0
    for index, row in enumerate(cases):
        cli = [str(value) for key, value in row.items() if key != "file_name"]
        reference = run_process([str(paths["oracle"]), *cli], timeout=5)
        candidate = run_process([str(paths["binary"]), *cli], timeout=5)
        matched = (
            reference.get("returncode") == 0
            and candidate.get("returncode") == 0
            and c_outputs_equal(reference.get("stdout", ""), candidate.get("stdout", ""))
        )
        passed += int(matched)
        if not matched and len(failures) < 12:
            failures.append({
                "index": index,
                "args": cli,
                "reference": reference.get("stdout", "")[-1000:],
                "candidate": candidate.get("stdout", "")[-1000:],
                "candidate_returncode": candidate.get("returncode"),
                "candidate_stderr": candidate.get("stderr", "")[-1000:],
            })
    record = score_record(spec, cases, passed, failures)
    record["compile"] = compile_result
    record["output_source_exists"] = paths["output"].is_file()
    record["output_binary_exists"] = paths["binary"].is_file()
    return record


def score_record(
    spec: dict[str, Any],
    cases: list[dict[str, Any]],
    passed: int,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(cases)
    return {
        "benchmark": spec["benchmark"],
        "difficulty": spec["difficulty"],
        "relative_path": spec["relative_path"],
        "passed": passed,
        "total": total,
        "pass_rate": passed / total if total else None,
        "all_pass": bool(total and passed == total),
        "official_file_score": int(bool(total and passed == total)),
        "failures": failures,
        "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def git_revision(path: Path) -> str | None:
    result = run_process(["git", "-C", str(path), "rev-parse", "HEAD"])
    return result["stdout"].strip() if result.get("returncode") == 0 else None


async def main() -> None:
    args = parse_args()
    inherit_runtime_credentials(args.inherit_env_pid)
    if args.smoke_only:
        await smoke_model(args.model)
        print(f"SMOKE_OK model={args.model}", flush=True)
        return
    selected = [item.strip() for item in args.tasks.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(TASKS))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")
    if not NODE.is_file():
        raise FileNotFoundError(NODE)
    if "c2rust" in selected and shutil.which("rustc") is None:
        raise RuntimeError("rustc is required for c2rust")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pid": os.getpid(),
        "model": args.model,
        "tasks": {kind: TASKS[kind] for kind in selected},
        "seed_c2rust_dir": str(args.seed_c2rust_dir) if args.seed_c2rust_dir else None,
        "nanoma_revision": git_revision(Path(__file__).resolve().parents[2]),
        "repozero_revision": git_revision(args.repozero_root),
        "generation_protocol": "public source + official black-box executable only",
        "grading_protocol": "official hidden JSONL loaded only after NanoMA exits",
    }
    atomic_json(args.run_dir / "manifest.json", manifest)
    print(f"RUN_DIR={args.run_dir}", flush=True)
    print(f"MODEL={args.model}", flush=True)

    summaries: dict[str, Any] = {}
    for kind in selected:
        spec = TASKS[kind]
        paths = prepare_task(kind, spec, args)
        if kind == "c2rust" and args.seed_c2rust_dir is not None:
            seed_selection = select_c_candidate(
                paths,
                seed_dir=args.seed_c2rust_dir,
                record_name="seed_selection",
            )
            print(
                "[c2rust] SEED "
                f"public_probes={((seed_selection.get('selected') or {}).get('public_probes_passed'))}/"
                f"{len(PUBLIC_C_PROBES)} output_ready={paths['binary'].is_file()}",
                flush=True,
            )
        print(f"[{kind}] START {spec['relative_path']}", flush=True)
        try:
            generation = await generate(kind, spec, paths, args)
            if kind == "c2rust":
                generation["delivery_recovery"] = select_c_candidate(
                    paths,
                    include_workspace=True,
                    record_name="delivery_recovery",
                )
            atomic_json(paths["task_dir"] / "generation.json", generation)
            score = score_py(spec, paths, args) if kind == "py2js" else score_c(spec, paths, args)
            atomic_json(paths["task_dir"] / "score.json", score)
            summaries[kind] = score
            print(
                f"[{kind}] SCORE {score['passed']}/{score['total']} "
                f"all_pass={score['all_pass']}",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "type": type(exc).__name__,
                "message": str(exc),
                "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            atomic_json(paths["task_dir"] / "failure.json", failure)
            summaries[kind] = {"error": failure}
            print(f"[{kind}] ERROR {type(exc).__name__}: {exc}", flush=True)

    summary = {
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "tasks": summaries,
    }
    atomic_json(args.run_dir / "summary.json", summary)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
