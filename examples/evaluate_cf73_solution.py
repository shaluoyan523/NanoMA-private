"""Evaluate a selected C++ solution against a staged CF-73 Polygon package."""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one CF-73 selected solution")
    parser.add_argument("--problem-dir", required=True, help="Public problem directory containing metadata.json")
    parser.add_argument("--judge-dir", help="Private problem directory containing polygon_package.zip")
    parser.add_argument("--package", help="Explicit private Polygon package path")
    parser.add_argument("--solution", required=True, help="C++ solution file to evaluate")
    parser.add_argument("--solution-std", default="gnu++17", help="C++ standard for submitted solution, or `auto`")
    parser.add_argument("--max-tests", type=int, default=0, help="Optional cap for smoke evaluation")
    parser.add_argument("--timeout", type=float, default=15.0, help="Per process timeout in seconds")
    parser.add_argument("--verbose", action="store_true", help="Let generator/checker/interactor output stream to the terminal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    problem_dir = Path(args.problem_dir)
    solution = Path(args.solution)
    metadata = json.loads((problem_dir / "metadata.json").read_text())
    package = resolve_package(args, problem_dir)
    if not solution.is_file():
        raise SystemExit(f"solution not found: {solution}")

    with tempfile.TemporaryDirectory(prefix="cf73_eval_") as td:
        work = Path(td)
        with zipfile.ZipFile(package) as zf:
            zf.extractall(work)
        commands = generated_test_commands(work / "problem.xml")
        compile_required(work, solution, commands, bool(metadata.get("interactive")), args.solution_std)
        if args.max_tests > 0:
            commands = commands[: args.max_tests]
        passed = 0
        failures = []
        for idx, command in enumerate(commands, start=1):
            ok, error = run_one_test(work, idx, command, args.timeout, bool(metadata.get("interactive")), args.verbose)
            if ok:
                passed += 1
            else:
                failures.append({"test": idx, "cmd": command, "error": error})
                break
        print(json.dumps({
            "status": "passed" if passed == len(commands) else "failed",
            "problem": metadata["problem_id"],
            "tests_total": len(commands),
            "tests_passed": passed,
            "failures": failures,
        }, indent=2))


def resolve_package(args: argparse.Namespace, problem_dir: Path) -> Path:
    if args.package:
        package = Path(args.package)
    elif args.judge_dir:
        package = Path(args.judge_dir) / "polygon_package.zip"
    else:
        raise SystemExit(
            "private Polygon package is required; pass --judge-dir or --package. "
            "It must not be stored in the public agent-visible problem directory."
        )
    if not package.is_file():
        raise SystemExit(f"Polygon package not found: {package}")
    return package


def generated_test_commands(problem_xml: Path) -> list[str]:
    root = ET.fromstring(problem_xml.read_text(errors="replace"))
    return [test.attrib["cmd"] for test in root.findall("./judging/testset/tests/test") if test.attrib.get("method") == "generated"]


def compile_required(work: Path, solution: Path, commands: list[str], interactive: bool, solution_std: str) -> None:
    include = work / "files"
    for generator in sorted(generator_names(commands)):
        generator_src = work / "files" / f"{generator}.cpp"
        if not generator_src.is_file():
            raise SystemExit(f"generator source missing for `{generator}`: {generator_src}")
        compile_cpp(generator_src, work / generator, include)

    if interactive:
        interactor = find_interactor(work / "problem.xml", work)
        compile_cpp(interactor, work / "interactor", include, flexible=True)
    else:
        compile_cpp(find_checker(work / "problem.xml", work), work / "check", include)
        compile_cpp(find_main_solution(work / "problem.xml", work), work / "official", include, flexible=True)

    compile_cpp(solution, work / "submitted", include, standard=solution_std, flexible=solution_std == "auto")


def compile_cpp(src: Path, out: Path, include: Path, *, standard: str = "gnu++17", flexible: bool = False) -> None:
    compilers = available_cpp_compilers()
    attempts = [(compiler, std) for compiler in compilers for std in ("gnu++17", "gnu++20", "gnu++23", "gnu++2b")] if flexible else [(compilers[0], standard)]
    errors = []
    for compiler, std in attempts:
        result = subprocess.run(
            [compiler, f"-std={std}", "-O2", "-pipe", "-I", str(include), str(src), "-o", str(out)],
            capture_output=True,
            text=True,
            preexec_fn=raise_stack_limit,
        )
        if result.returncode == 0:
            return
        stderr = result.stderr.strip()
        errors.append(f"{compiler} -std={std}: {stderr[-1600:]}")
    detail = "\n\n".join(errors[-3:]) if errors else "no C++ compiler found"
    raise SystemExit(f"compile failed for {src}\n{detail}")


def available_cpp_compilers() -> list[str]:
    candidates = [
        "/opt/OpenCloudOS/gcc-toolset-14/root/usr/bin/g++",
        "/opt/rh/gcc-toolset-14/root/usr/bin/g++",
        "/opt/OpenCloudOS/gcc-toolset-13/root/usr/bin/g++",
        "/opt/rh/gcc-toolset-13/root/usr/bin/g++",
        "g++-14",
        "g++",
        "clang++",
    ]
    compilers = []
    seen = set()
    for candidate in candidates:
        resolved = candidate if Path(candidate).is_file() else shutil.which(candidate)
        if not resolved or resolved in seen:
            continue
        seen.add(resolved)
        compilers.append(resolved)
    if not compilers:
        raise SystemExit("no C++ compiler found")
    return compilers


def generator_names(commands: list[str]) -> set[str]:
    names = set()
    for command in commands:
        parts = command.split()
        if parts:
            names.add(parts[0])
    return names


def find_checker(problem_xml: Path, work: Path) -> Path:
    root = ET.fromstring(problem_xml.read_text(errors="replace"))
    checker = root.find("./assets/checker/source")
    if checker is not None and checker.attrib.get("path"):
        candidate = work / checker.attrib["path"]
        if candidate.is_file():
            return candidate
    return work / "check.cpp"


def find_interactor(problem_xml: Path, work: Path) -> Path:
    root = ET.fromstring(problem_xml.read_text(errors="replace"))
    interactor = root.find("./assets/interactor/source")
    if interactor is not None and interactor.attrib.get("path"):
        candidate = work / interactor.attrib["path"]
        if candidate.is_file():
            return candidate
    return work / "files" / "i.cpp"


def find_main_solution(problem_xml: Path, work: Path) -> Path:
    root = ET.fromstring(problem_xml.read_text(errors="replace"))
    for solution in root.findall("./assets/solutions/solution"):
        if solution.attrib.get("tag") == "main":
            source = solution.find("./source")
            if source is not None and source.attrib.get("path"):
                candidate = work / source.attrib["path"]
                if candidate.is_file():
                    return candidate
    fallback = work / "solutions" / "ac1.cpp"
    if fallback.is_file():
        return fallback
    raise SystemExit("main official solution source not found in Polygon package")


def run_one_test(work: Path, idx: int, command: str, timeout: float, interactive: bool, verbose: bool) -> tuple[bool, str]:
    tests = work / "tests"
    tests.mkdir(exist_ok=True)
    input_path = tests / f"{idx:02d}"
    answer_path = tests / f"{idx:02d}.a"
    output_path = tests / f"{idx:02d}.out"
    parts = command.split()
    if not parts:
        return False, f"unsupported generator command: {command}"
    generator = work / parts[0]
    if not generator.exists():
        return False, f"generator binary missing for command: {command}"
    stderr_target = None if verbose else subprocess.PIPE
    try:
        with input_path.open("wb") as out:
            subprocess.run(
                [str(generator), *parts[1:]],
                cwd=work,
                stdout=out,
                stderr=stderr_target,
                check=True,
                timeout=timeout,
                preexec_fn=raise_stack_limit,
            )
        if interactive:
            return run_interactive_test(work, input_path, output_path, timeout)
        with input_path.open("rb") as inp, answer_path.open("wb") as out:
            subprocess.run(
                [str(work / "official")],
                cwd=work,
                stdin=inp,
                stdout=out,
                stderr=stderr_target,
                check=True,
                timeout=timeout,
                preexec_fn=raise_stack_limit,
            )
        with input_path.open("rb") as inp, output_path.open("wb") as out:
            subprocess.run(
                [str(work / "submitted")],
                cwd=work,
                stdin=inp,
                stdout=out,
                stderr=stderr_target,
                check=True,
                timeout=timeout,
                preexec_fn=raise_stack_limit,
            )
        subprocess.run(
            [str(work / "check"), str(input_path), str(output_path), str(answer_path)],
            cwd=work,
            stdout=None if verbose else subprocess.PIPE,
            stderr=None if verbose else subprocess.PIPE,
            check=True,
            timeout=timeout,
            preexec_fn=raise_stack_limit,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except subprocess.CalledProcessError as e:
        return False, format_process_error(e)
    return True, ""


def format_process_error(error: subprocess.CalledProcessError) -> str:
    parts = [f"exit {error.returncode}: {' '.join(map(str, error.cmd))}"]
    for label, payload in (("stdout", error.stdout), ("stderr", error.stderr)):
        text = decode_process_output(payload).strip()
        if text:
            parts.append(f"{label}: {text[-1200:]}")
    return "\n".join(parts)


def decode_process_output(payload) -> str:
    if payload is None:
        return ""
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")
    return str(payload)


def run_interactive_test(work: Path, input_path: Path, output_path: Path, timeout: float) -> tuple[bool, str]:
    interactor = subprocess.Popen(
        [str(work / "interactor"), str(input_path), str(output_path)],
        cwd=work,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=raise_stack_limit,
    )
    submitted = subprocess.Popen(
        [str(work / "submitted")],
        cwd=work,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=raise_stack_limit,
    )

    threads = [
        threading.Thread(target=pump, args=(interactor.stdout, submitted.stdin), daemon=True),
        threading.Thread(target=pump, args=(submitted.stdout, interactor.stdin), daemon=True),
    ]
    for thread in threads:
        thread.start()

    deadline = time.time() + timeout
    while time.time() < deadline:
        if interactor.poll() is not None and submitted.poll() is not None:
            break
        time.sleep(0.02)
    else:
        kill_process(interactor)
        kill_process(submitted)
        return False, "timeout"

    for thread in threads:
        thread.join(timeout=1)
    interactor_rc = interactor.wait(timeout=1)
    submitted_rc = submitted.wait(timeout=1)
    interactor_err = (interactor.stderr.read() or b"").decode("utf-8", errors="replace").strip()
    submitted_err = (submitted.stderr.read() or b"").decode("utf-8", errors="replace").strip()
    if interactor_rc != 0:
        return False, f"interactor exit {interactor_rc}: {interactor_err}"
    if submitted_rc != 0:
        return False, f"submitted exit {submitted_rc}: {submitted_err}"
    return True, ""


def pump(src, dst) -> None:
    if src is None or dst is None:
        return
    try:
        while True:
            chunk = src.readline()
            if not chunk:
                break
            dst.write(chunk)
            dst.flush()
    except (BrokenPipeError, ValueError):
        pass
    finally:
        try:
            dst.close()
        except Exception:
            pass


def kill_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.kill()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass


def raise_stack_limit() -> None:
    try:
        resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except (OSError, ValueError):
        pass


if __name__ == "__main__":
    main()
