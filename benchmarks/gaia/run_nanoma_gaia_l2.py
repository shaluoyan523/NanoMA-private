#!/usr/bin/env python3
"""Run NanoMA on GAIA validation Level 2 with BenchAgent-style settings.

GAIA is a gated HuggingFace dataset. Export HF_TOKEN or
HUGGINGFACE_HUB_TOKEN after accepting dataset access:

    set -a; . ./.env; set +a
    export HF_TOKEN=...
    python3 benchmarks/gaia/run_nanoma_gaia_l2.py --limit 1 --clean

The harness uses BenchAgent's reported model-call settings for the benchmark:
temperature=0.2, top_p=1.0, max_tokens=8192.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import shutil
import string
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HF_DATASET = "gaia-benchmark/GAIA"
HF_REVISION = "main"
GAIA_SUBDIR = "2023/validation"
LEVEL2_PARQUET = f"{GAIA_SUBDIR}/metadata.level2.parquet"
LEVEL2_CONFIG = "2023_level2"
GAIA_DIR = ROOT / "benchmarks" / "gaia"
GAIA_DATA_DIR = GAIA_DIR / "data"
GAIA_RUNS_DIR = GAIA_DIR / "runs"
DEFAULT_OUTPUT_ROOT = ROOT / "runs" / "gaia"

BENCHAGENT_TEMPERATURE = 0.2
BENCHAGENT_TOP_P = 1.0
BENCHAGENT_MAX_TOKENS = 8192
ANSWER_SOURCE_BLOCK_PATTERNS = [
    r"gaia-benchmark/GAIA",
    r"huggingface\.co/(api/)?datasets/gaia-benchmark/GAIA",
    r"metadata(?:\.level\d+)?\.parquet",
    r"Final answer",
    r"Annotator Metadata",
    r"ground truth",
    r"answer key",
    r"benchmark answer",
    r"answer leak",
    re.escape(str(GAIA_DATA_DIR.resolve())),
    re.escape(str(GAIA_RUNS_DIR.resolve())),
    r"benchmarks/gaia/data",
    r"benchmarks/gaia/runs",
]


def hf_token() -> str:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or ""


def hf_headers(token: str | None = None) -> dict[str, str]:
    token = token or hf_token()
    return {"Authorization": f"Bearer {token}"} if token else {}


def token_diagnostic(token: str) -> str:
    if not token:
        return "No HF token is configured."
    try:
        resp = requests.get("https://huggingface.co/api/whoami-v2", headers=hf_headers(token), timeout=20)
        if resp.status_code != 200:
            return f"HF token whoami status={resp.status_code}."
        data = resp.json()
        access = data.get("auth", {}).get("accessToken", {})
        fine = access.get("fineGrained", {})
        can_read_gated = fine.get("canReadGatedRepos")
        role = access.get("role")
        display = access.get("displayName")
        return (
            f"HF token displayName={display!r}, role={role!r}, "
            f"canReadGatedRepos={can_read_gated!r}. "
            "For GAIA, use a token from an account approved for gaia-benchmark/GAIA "
            "and enable gated repo read access."
        )
    except Exception as exc:
        return f"HF token diagnostic failed: {type(exc).__name__}: {exc}"


def hf_url(repo_file: str) -> str:
    repo = quote(HF_DATASET, safe="/")
    path = quote(repo_file, safe="/")
    return f"https://huggingface.co/datasets/{repo}/resolve/{HF_REVISION}/{path}"


def hf_parquet_index_url() -> str:
    repo = quote(HF_DATASET, safe="/")
    return f"https://huggingface.co/api/datasets/{repo}/parquet"


def download_file(repo_file: str, dest: Path, *, token: str, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = hf_url(repo_file)
    with requests.get(url, headers=hf_headers(token), stream=True, timeout=timeout) as resp:
        if resp.status_code == 401:
            raise RuntimeError(
                f"GAIA dataset access denied for {repo_file}. "
                "Set HF_TOKEN/HUGGINGFACE_HUB_TOKEN for an account with access to gaia-benchmark/GAIA."
            )
        if resp.status_code == 403:
            raise RuntimeError(
                f"GAIA dataset access forbidden for {repo_file}. "
                f"Check dataset access approval and token permissions. {token_diagnostic(token)}"
            )
        resp.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(dest)


def download_generated_parquet(dest: Path, *, token: str, timeout: int = 120) -> None:
    """Download the parquet produced by HuggingFace's dataset parquet API."""
    index_resp = requests.get(hf_parquet_index_url(), headers=hf_headers(token), timeout=timeout)
    index_resp.raise_for_status()
    index = index_resp.json()
    urls = index.get(LEVEL2_CONFIG, {}).get("validation", [])
    if not urls:
        raise RuntimeError(f"No generated parquet URL for {LEVEL2_CONFIG}/validation")

    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(urls[0], headers=hf_headers(token), stream=True, timeout=timeout) as resp:
        if resp.status_code == 401:
            raise RuntimeError(
                "GAIA generated parquet access denied. "
                "Set HF_TOKEN/HUGGINGFACE_HUB_TOKEN for an account with access to gaia-benchmark/GAIA."
            )
        if resp.status_code == 403:
            raise RuntimeError(
                "GAIA generated parquet access forbidden. "
                f"Check dataset access approval and token permissions. {token_diagnostic(token)}"
            )
        resp.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(dest)


def snapshot_metadata_path(snapshot_dir: Path) -> Path:
    return snapshot_dir / LEVEL2_PARQUET


def ensure_level2_metadata(
    data_dir: Path,
    *,
    token: str,
    force: bool = False,
    download_method: str = "auto",
    snapshot_dir: Path | None = None,
) -> Path:
    if snapshot_dir:
        metadata = snapshot_metadata_path(snapshot_dir)
        if not metadata.exists():
            raise FileNotFoundError(f"Missing GAIA Level 2 metadata in snapshot: {metadata}")
        return metadata

    metadata = data_dir / LEVEL2_PARQUET
    if force or not metadata.exists():
        errors: list[str] = []
        if download_method in {"auto", "api"}:
            try:
                download_generated_parquet(metadata, token=token)
                return metadata
            except Exception as exc:
                errors.append(f"api: {type(exc).__name__}: {exc}")
                if download_method == "api":
                    raise
        if download_method in {"auto", "repo"}:
            try:
                download_file(LEVEL2_PARQUET, metadata, token=token)
                return metadata
            except Exception as exc:
                errors.append(f"repo: {type(exc).__name__}: {exc}")
                if download_method == "repo":
                    raise
        raise RuntimeError("Unable to download GAIA-L2 metadata via available methods:\n" + "\n".join(errors))
    return metadata


def table_to_records(path: Path) -> list[dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    records: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        record = {str(k): v for k, v in row.items()}
        record.setdefault("_row_index", i)
        records.append(record)
    return records


def value_for(record: dict[str, Any], *names: str, default: Any = None) -> Any:
    lowered = {k.lower(): k for k in record}
    for name in names:
        if name in record:
            return record[name]
        key = lowered.get(name.lower())
        if key is not None:
            return record[key]
    return default


def task_id(record: dict[str, Any], index: int) -> str:
    value = value_for(record, "task_id", "id", "question_id", default=None)
    if value:
        return str(value)
    return f"gaia_l2_{index:03d}"


def file_name(record: dict[str, Any]) -> str:
    value = value_for(record, "file_name", "file", "filename", "attachment", default="")
    if value is None:
        return ""
    return str(value)


def final_answer(record: dict[str, Any]) -> str:
    return str(value_for(record, "Final answer", "final_answer", "answer", default=""))


def question(record: dict[str, Any]) -> str:
    return str(value_for(record, "Question", "question", default=""))


def ensure_attachment(
    record: dict[str, Any],
    data_dir: Path,
    *,
    token: str,
    force: bool = False,
    snapshot_dir: Path | None = None,
) -> Path | None:
    name = file_name(record)
    if not name or name.lower() in {"nan", "none", "null"}:
        return None
    if snapshot_dir:
        path_value = str(value_for(record, "file_path", default="") or "")
        local = snapshot_dir / (path_value if path_value else f"{GAIA_SUBDIR}/{name}")
        if not local.exists():
            raise FileNotFoundError(f"Missing GAIA attachment in snapshot: {local}")
        return local
    local = data_dir / GAIA_SUBDIR / name
    if force or not local.exists():
        download_file(f"{GAIA_SUBDIR}/{name}", local, token=token, timeout=300)
    return local


def stage_attachment(attachment: Path | None, workspace: Path) -> Path | None:
    """Copy the task attachment into the run workspace so agents need no dataset path access."""
    if attachment is None:
        return None
    dest_dir = workspace / "shared" / "attachments"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / attachment.name
    if not dest.exists() or dest.stat().st_size != attachment.stat().st_size:
        shutil.copy2(attachment, dest)
    return Path("$SHARED") / "attachments" / attachment.name


def normalize_number_str(number_str: str) -> float:
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")


def split_string(value: str, char_list: list[str] | None = None) -> list[str]:
    if char_list is None:
        char_list = [",", ";"]
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, value)


def normalize_str(value: str, *, remove_punct: bool = True) -> str:
    no_spaces = re.sub(r"\s", "", value)
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    return no_spaces.lower()


def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def gaia_score(model_answer: str, gold: str) -> bool:
    """GAIA official-style exact scorer."""
    model_answer = str(model_answer).strip()
    gold = str(gold).strip()
    if is_float(gold):
        return normalize_number_str(model_answer) == float(gold)
    if any(char in gold for char in [",", ";"]):
        gold_elems = split_string(gold)
        model_elems = split_string(model_answer)
        if len(gold_elems) != len(model_elems):
            return False
        comparisons = []
        for ma_elem, gt_elem in zip(model_elems, gold_elems):
            if is_float(gt_elem):
                comparisons.append(normalize_number_str(ma_elem) == float(gt_elem))
            else:
                comparisons.append(
                    normalize_str(ma_elem, remove_punct=False)
                    == normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)
    return normalize_str(model_answer) == normalize_str(gold)


def extract_answer(text: str) -> str:
    """Extract the answer-only contract from NanoMA output."""
    text = (text or "").strip()
    candidates: list[str] = []

    for path in re.findall(r"answer\.json", text):
        candidates.append(path)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            for key in ("answer", "final_answer", "FINAL ANSWER"):
                if key in parsed:
                    return str(parsed[key]).strip()
    except Exception:
        pass

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    for block in fenced:
        try:
            parsed = json.loads(block)
        except Exception:
            continue
        if isinstance(parsed, dict):
            for key in ("answer", "final_answer", "FINAL ANSWER"):
                if key in parsed:
                    return str(parsed[key]).strip()

    match = re.search(r"FINAL ANSWER\s*:?\s*(.+)", text, flags=re.I | re.S)
    if match:
        return match.group(1).strip().strip("`")
    return text


def read_answer_json(workspace: Path) -> tuple[str | None, Path | None, dict[str, Any] | None]:
    for path in sorted(workspace.glob("**/answer.json"), key=lambda p: (len(p.parts), str(p))):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            for key in ("answer", "final_answer"):
                if key in data:
                    return str(data[key]).strip(), path, data
    return None, None, None


def prompt_for(
    record: dict[str, Any],
    attachment: Path | None,
    *,
    allow_spawn: bool = True,
    block_answer_sources: bool = False,
) -> str:
    attach_text = "No attachment is provided for this task."
    if attachment:
        attach_text = (
            f"Attachment path: {attachment}\n"
            "You may inspect it with shell tools. Use Python or command-line tools when useful."
        )
    agent_rule = (
        "- You may spawn helper agents for parallel investigation, but keep the final answer contract strict."
        if allow_spawn
        else "- Work as a single agent. Do not delegate, spawn helpers, or rely on multi-agent coordination."
    )
    leakage_rule = (
        "- Do not access GAIA benchmark metadata, answer keys, ground-truth files, `Final answer` fields, leaderboard answer dumps, or `gaia-benchmark/GAIA` dataset files. Use only the task question, provided attachment, and ordinary public evidence needed to solve the task."
        if block_answer_sources
        else ""
    )

    return f"""You are evaluating one GAIA validation Level 2 task under a BenchAgent-style protocol.

Question:
{question(record)}

{attach_text}

Rules:
- Solve the task using available tools. Search the web only through shell commands if needed.
{agent_rule}
{leakage_rule}
- Do not include reasoning in the final submitted answer.
- Before finishing, write a JSON file named answer.json in the shared directory with exactly this shape:
  {{"answer": "<answer-only string>"}}
- Then call set_status("done", result="<answer-only string>").

Final answer requirements:
- Return only the concise answer expected by GAIA.
- If the answer is a list, preserve the order and use commas.
- If the answer is numeric, omit extra units unless the question requires them.
"""


def benchagent_llm_call(model_settings: dict[str, Any]):
    from nanoma.llm import default_llm_call

    async def call(messages, model, tools=None, **kwargs):
        kwargs["temperature"] = model_settings["temperature"]
        kwargs["top_p"] = model_settings["top_p"]
        kwargs["max_tokens"] = model_settings["max_tokens"]
        return await default_llm_call(messages, model, tools, **kwargs)

    return call


async def run_one(args: argparse.Namespace, record: dict[str, Any], index: int, attachment: Path | None) -> dict[str, Any]:
    from nanoma.core import Runtime, RuntimeConfig

    tid = task_id(record, index)
    workspace = args.workspace_root / tid
    log_dir = args.log_root / tid

    if args.clean:
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(log_dir, ignore_errors=True)
    staged_attachment = stage_attachment(attachment, workspace)

    model_settings = {
        "temperature": BENCHAGENT_TEMPERATURE,
        "top_p": BENCHAGENT_TOP_P,
        "max_tokens": args.max_tokens,
    }
    config_kwargs: dict[str, Any] = {}
    if args.max_agents is not None:
        config_kwargs["max_agents"] = args.max_agents

    config = RuntimeConfig(
        max_depth=args.max_depth,
        max_concurrent_llm=args.max_concurrent_llm,
        budget=args.budget,
        max_total_tokens=args.max_total_tokens,
        tool_policy_soft_total_tokens=args.tool_policy_soft_total_tokens,
        time_limit=args.time_limit,
        max_turns=args.max_turns,
        default_model=args.model,
        workspace_root=workspace,
        log_dir=log_dir,
        shell_max_output=args.shell_max_output,
        file_read_max_chars=args.file_read_max_chars,
        disabled_tools=set(args.disable_tool),
        blocked_shell_patterns=ANSWER_SOURCE_BLOCK_PATTERNS if args.block_answer_sources else [],
        tool_policy_mode=args.tool_policy_mode,
        tool_policy_dependency_window=args.tool_policy_dependency_window,
        tool_policy_prune_tools=args.tool_policy_prune,
        tool_policy_prune_min_tools=args.tool_policy_prune_min_tools,
        tool_policy_prune_pressure_start=args.tool_policy_prune_pressure_start,
        tool_policy_prune_pressure_end=args.tool_policy_prune_pressure_end,
        tool_policy_prune_shell_capabilities=args.tool_policy_prune_shell_capabilities,
        tool_policy_shell_capability_pressure_start=args.tool_policy_shell_capability_pressure_start,
        tool_policy_shell_capability_pressure_end=args.tool_policy_shell_capability_pressure_end,
        tool_policy_web_saturation_enabled=args.tool_policy_web_saturation_enabled,
        tool_policy_web_saturation_min_calls=args.tool_policy_web_saturation_min_calls,
        tool_policy_web_saturation_threshold=args.tool_policy_web_saturation_threshold,
        tool_policy_web_saturation_finalize_after_blocks=args.tool_policy_web_saturation_finalize_after_blocks,
        tool_policy_log_events=args.tool_policy_log_events,
        **config_kwargs,
    )

    def on_event(e: dict[str, Any]) -> None:
        if args.verbose:
            print(json.dumps(e, ensure_ascii=False), flush=True)
            return
        if e["event"] in {"agent_new", "spawn", "done", "failed", "llm_done"}:
            data = e.get("data", {})
            compact = {
                k: data[k]
                for k in ("task", "child", "status", "turns", "tokens", "tool_calls", "has_content")
                if k in data
            }
            print(f"[{tid}][{e['event']}] {e['agent']} {compact}", flush=True)

    runtime = Runtime(config=config, llm_call=benchagent_llm_call(model_settings), on_event=on_event)
    started = time.time()
    result = await runtime.run(
        prompt_for(
            record,
            staged_attachment,
            allow_spawn="spawn" not in set(args.disable_tool),
            block_answer_sources=args.block_answer_sources,
        ),
        model=args.model,
    )
    answer, answer_path, answer_json = read_answer_json(workspace)
    if answer is None:
        answer = extract_answer(result)
    gold = final_answer(record)
    passed = gaia_score(answer, gold)

    return {
        "task_id": tid,
        "row_index": index,
        "question": question(record),
        "file_name": file_name(record),
        "gold": gold,
        "answer": answer,
        "passed": passed,
        "raw_result": result,
        "answer_path": str(answer_path) if answer_path else None,
        "answer_json": answer_json,
        "stats": runtime.stats(),
        "elapsed_s": round(time.time() - started, 3),
        "source_attachment": str(attachment) if attachment else None,
        "attachment": str(staged_attachment) if staged_attachment else None,
        "workspace": str(workspace),
        "logs": str(log_dir),
        "model_settings": model_settings,
    }


def select_records(records: list[dict[str, Any]], args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(records))
    if args.task_id:
        wanted = set(args.task_id)
        indexed = [(i, r) for i, r in indexed if task_id(r, i) in wanted]
    if args.offset:
        indexed = indexed[args.offset:]
    if args.limit:
        indexed = indexed[: args.limit]
    return indexed


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task_id",
        "row_index",
        "passed",
        "answer",
        "gold",
        "elapsed_s",
        "tokens",
        "cost_usd",
        "agents",
        "peak_concurrent",
        "attachment",
        "source_attachment",
        "workspace",
        "logs",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            stats = row.get("stats", {})
            overview = stats.get("overview", {})
            agents = stats.get("agents", {})
            writer.writerow({
                "task_id": row.get("task_id"),
                "row_index": row.get("row_index"),
                "passed": row.get("passed"),
                "answer": row.get("answer"),
                "gold": row.get("gold"),
                "elapsed_s": row.get("elapsed_s"),
                "tokens": overview.get("total_tokens"),
                "cost_usd": overview.get("total_cost_usd"),
                "agents": agents.get("total_spawned"),
                "peak_concurrent": agents.get("peak_concurrent"),
                "workspace": row.get("workspace"),
                "logs": row.get("logs"),
            })


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    total = len(rows)
    passed = sum(1 for row in rows if row.get("passed"))
    total_tokens = sum(row.get("stats", {}).get("overview", {}).get("total_tokens", 0) for row in rows)
    total_cost = sum(row.get("stats", {}).get("overview", {}).get("total_cost_usd", 0.0) for row in rows)
    return {
        "benchmark": "GAIA validation Level 2",
        "completed_tasks": total,
        "passed": passed,
        "pass_at_1": (passed / total) if total else 0.0,
        "total_tokens": total_tokens,
        "avg_tokens": int(total_tokens / total) if total else 0,
        "total_cost_usd": round(total_cost, 4),
        "model": args.model,
        "benchagent_settings": {
            "temperature": BENCHAGENT_TEMPERATURE,
            "top_p": BENCHAGENT_TOP_P,
            "max_tokens": args.max_tokens,
        },
        "runtime": {
            "max_agents": args.max_agents,
            "max_depth": args.max_depth,
            "max_concurrent_llm": args.max_concurrent_llm,
            "max_turns": args.max_turns,
            "max_total_tokens": args.max_total_tokens,
            "tool_policy_soft_total_tokens": args.tool_policy_soft_total_tokens,
            "time_limit": args.time_limit,
            "disabled_tools": sorted(set(args.disable_tool)),
            "block_answer_sources": args.block_answer_sources,
            "blocked_shell_patterns": ANSWER_SOURCE_BLOCK_PATTERNS if args.block_answer_sources else [],
            "tool_policy_mode": args.tool_policy_mode,
            "tool_policy_dependency_window": args.tool_policy_dependency_window,
            "tool_policy_prune": args.tool_policy_prune,
            "tool_policy_prune_min_tools": args.tool_policy_prune_min_tools,
            "tool_policy_prune_pressure_start": args.tool_policy_prune_pressure_start,
            "tool_policy_prune_pressure_end": args.tool_policy_prune_pressure_end,
            "tool_policy_prune_shell_capabilities": args.tool_policy_prune_shell_capabilities,
            "tool_policy_shell_capability_pressure_start": args.tool_policy_shell_capability_pressure_start,
            "tool_policy_shell_capability_pressure_end": args.tool_policy_shell_capability_pressure_end,
            "tool_policy_web_saturation_enabled": args.tool_policy_web_saturation_enabled,
            "tool_policy_web_saturation_min_calls": args.tool_policy_web_saturation_min_calls,
            "tool_policy_web_saturation_threshold": args.tool_policy_web_saturation_threshold,
            "tool_policy_web_saturation_finalize_after_blocks": args.tool_policy_web_saturation_finalize_after_blocks,
            "tool_policy_log_events": args.tool_policy_log_events,
        },
    }


def preflight_attachments(
    selected: list[tuple[int, dict[str, Any]]],
    args: argparse.Namespace,
    token: str,
) -> dict[str, Path | None]:
    attachments: dict[str, Path | None] = {}
    missing: list[str] = []
    for index, record in selected:
        tid = task_id(record, index)
        try:
            attachments[tid] = ensure_attachment(
                record,
                args.data_dir,
                token=token,
                force=args.force_download,
                snapshot_dir=args.snapshot_dir,
            )
        except Exception as exc:
            missing.append(f"{tid} ({file_name(record)}): {type(exc).__name__}: {exc}")
    if missing:
        raise SystemExit(
            "GAIA attachment preflight failed before starting NanoMA:\n"
            + "\n".join(missing)
        )
    return attachments


async def run_selected(args: argparse.Namespace, selected: list[tuple[int, dict[str, Any]]], data_dir: Path, token: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    attachments = (
        preflight_attachments(selected, args, token)
        if args.preflight_attachments and not args.skip_download
        else {}
    )
    for ordinal, (index, record) in enumerate(selected, start=1):
        tid = task_id(record, index)
        print(f"=== GAIA-L2 {ordinal}/{len(selected)} task_id={tid} ===", flush=True)
        attachment = attachments.get(tid)
        if tid not in attachments:
            attachment = ensure_attachment(
                record,
                data_dir,
                token=token,
                force=args.force_download,
                snapshot_dir=args.snapshot_dir,
            )
        if args.dry_run:
            rows.append({
                "task_id": tid,
                "row_index": index,
                "question": question(record),
                "file_name": file_name(record),
                "gold": final_answer(record),
                "attachment": str(attachment) if attachment else None,
                "dry_run": True,
            })
            continue
        try:
            row = await run_one(args, record, index, attachment)
        except Exception as exc:
            row = {
                "task_id": tid,
                "row_index": index,
                "question": question(record),
                "file_name": file_name(record),
                "gold": final_answer(record),
                "answer": "",
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[{tid}] ERROR {row['error']}", flush=True)
        rows.append(row)
        args.results_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.results_dir / "results.jsonl", rows)
        write_csv(args.results_dir / "results.csv", rows)
        (args.results_dir / "summary.json").write_text(
            json.dumps(summarize(rows, args), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("NANOMA_MODEL", "deepseek-v4-pro"))
    ap.add_argument("--data-dir", type=Path, default=GAIA_DATA_DIR)
    ap.add_argument("--snapshot-dir", type=Path, help="Use an already downloaded GAIA repo snapshot directory.")
    ap.add_argument("--download-method", choices=["auto", "api", "repo"], default="auto")
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "nanoma_l2")
    ap.add_argument("--workspace-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "workspace")
    ap.add_argument("--log-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "logs")
    ap.add_argument("--task-id", action="append", help="Run specific GAIA task id. Repeatable.")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 means all selected tasks.")
    ap.add_argument("--budget", type=float, default=200.0)
    ap.add_argument("--max-total-tokens", type=int, default=0, help="0 disables the runtime token cap.")
    ap.add_argument("--tool-policy-soft-total-tokens", type=int, default=0, help="0 disables the policy-only token pressure cap.")
    ap.add_argument("--time-limit", type=float, default=7200)
    ap.add_argument("--max-turns", type=int, default=0)
    ap.add_argument("--max-agents", type=int, default=None)
    ap.add_argument("--max-depth", type=int, default=1)
    ap.add_argument("--max-concurrent-llm", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=BENCHAGENT_MAX_TOKENS)
    ap.add_argument("--tool-policy-mode", choices=["off", "adaptive", "enforce"], default="adaptive")
    ap.add_argument("--tool-policy-dependency-window", type=int, default=8)
    ap.add_argument("--tool-policy-prune", action="store_true")
    ap.add_argument("--tool-policy-prune-min-tools", type=int, default=6)
    ap.add_argument("--tool-policy-prune-pressure-start", type=float, default=0.65)
    ap.add_argument("--tool-policy-prune-pressure-end", type=float, default=0.95)
    ap.add_argument("--tool-policy-prune-shell-capabilities", action="store_true")
    ap.add_argument("--tool-policy-shell-capability-pressure-start", type=float, default=0.55)
    ap.add_argument("--tool-policy-shell-capability-pressure-end", type=float, default=0.95)
    ap.add_argument("--tool-policy-web-saturation-enabled", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tool-policy-web-saturation-min-calls", type=int, default=10)
    ap.add_argument("--tool-policy-web-saturation-threshold", type=float, default=0.85)
    ap.add_argument("--tool-policy-web-saturation-finalize-after-blocks", type=int, default=2)
    ap.add_argument("--tool-policy-log-events", action="store_true")
    ap.add_argument("--shell-max-output", type=int, default=50000)
    ap.add_argument("--file-read-max-chars", type=int, default=120000)
    ap.add_argument(
        "--disable-tool",
        action="append",
        default=[],
        help="Hide a tool from the runtime and reject it inside batch. Repeatable.",
    )
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument(
        "--preflight-attachments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download/check all selected task attachments before any NanoMA runs.",
    )
    ap.add_argument(
        "--block-answer-sources",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Block shell commands that try to access GAIA metadata, answer keys, prior runs/logs, or benchmark answer sources.",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    token = hf_token()
    if not token and not args.skip_download and not args.snapshot_dir:
        raise SystemExit(
            "Missing HF_TOKEN/HUGGINGFACE_HUB_TOKEN. GAIA is gated; accept access on HuggingFace and export a token."
        )

    metadata = snapshot_metadata_path(args.snapshot_dir) if args.snapshot_dir else args.data_dir / LEVEL2_PARQUET
    if not args.skip_download:
        metadata = ensure_level2_metadata(
            args.data_dir,
            token=token,
            force=args.force_download,
            download_method=args.download_method,
            snapshot_dir=args.snapshot_dir,
        )
    if not metadata.exists():
        raise SystemExit(f"Missing metadata parquet: {metadata}")

    records = table_to_records(metadata)
    selected = select_records(records, args)
    if not selected:
        raise SystemExit("No GAIA-L2 records selected.")

    print(f"GAIA-L2 records loaded: {len(records)}; selected: {len(selected)}", flush=True)
    print(
        "BenchAgent-style call settings: "
        f"temperature={BENCHAGENT_TEMPERATURE}, top_p={BENCHAGENT_TOP_P}, max_tokens={args.max_tokens}",
        flush=True,
    )
    rows = asyncio.run(run_selected(args, selected, args.data_dir, token))
    summary = summarize(rows, args)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.results_dir / "results.jsonl", rows)
    write_csv(args.results_dir / "results.csv", rows)
    (args.results_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"results={args.results_dir}", flush=True)
    if args.dry_run:
        return 0
    return 0 if summary["completed_tasks"] and summary["pass_at_1"] >= 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
