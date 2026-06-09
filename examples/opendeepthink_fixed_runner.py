"""Fixed OpenDeepThink-style runner for one CF-73 problem.

This reproduces the paper/official-code flow with a local OpenAI-compatible
LLM endpoint:
  gen0 n samples -> K-round pairwise judge/BT/mutate for T generations ->
  final M-round pairwise judge/BT -> selected_solution.cpp.

Private judge evaluation is optional and only runs after selection.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from nanoma.cost import CostLedger, UsageRecord
from nanoma.env import load_dotenv
from nanoma.llm import LLMResponse, RetryConfig, openai_compatible_call, set_log_dir


SYSTEM_PROMPT = (
    "You are an expert competitive programmer.\n"
    "Output your solution as a single ```cpp ... ``` block, preceded by brief reasoning."
)

GENERATION_PROMPT = "{problem}"
OFFICIAL_AC_PATH = "solutions/ac1.cpp"

PAIRWISE_PROMPT = """You are a competitive programming expert.

## Problem Statement
{problem}

## Solution A
```cpp
{code_a}
```

## Solution B
```cpp
{code_b}
```

Which solution is more likely to receive an Accepted verdict from an online judge -- meaning it produces correct output within the time and memory limits for all valid inputs?

If both solutions appear incorrect (wrong answer, TLE, or other issues), choose the one that requires fewer modifications to become Accepted.

If they are fundamentally identical or equally likely to be Accepted, output TIE.

Respond with a JSON object and nothing else, in exactly this format:
{{
  "feedback_a": "one sentence on Solution A's key strength or critical flaw",
  "feedback_b": "one sentence on Solution B's key strength or critical flaw",
  "winner": "A or B or TIE"
}}
"""

MUTATION_PROMPT_WITH_FEEDBACK = """## Problem
{problem}

## Solution
```cpp
{code}
```

## Pairwise Feedback
This solution was compared against other solutions multiple times:

{feedback_sections}
## Task
Write a solution that maximizes the probability of Accepted.
You may refine the existing solution or take a different approach if the current one is fundamentally flawed.

Think briefly, then output your final solution as a single ```cpp ... ``` block."""

MUTATION_PROMPT_NO_FEEDBACK = """## Problem
{problem}

## Solution
```cpp
{code}
```

## Task
Write a solution that maximizes the probability of Accepted.
You may refine the existing solution or take a different approach if the current one is fundamentally flawed.

Think briefly, then output your final solution as a single ```cpp ... ``` block."""


@dataclass
class Candidate:
    idx: int
    md: str
    origin: str


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        load_dotenv()
        self.args = args
        self.pid = args.pid
        self.model = args.model or os.environ.get("NANOMA_MODEL", "deepseek-v4-flash")
        self.problem_dir = Path(args.problem_dir)
        self.private_judge_dir = Path(args.private_judge_dir) if args.private_judge_dir else None
        self.out_dir = Path(args.out_dir)
        self.problem_text = (self.problem_dir / "statement.md").read_text(encoding="utf-8")
        self.meta = json.loads((self.problem_dir / "metadata.json").read_text(encoding="utf-8"))
        self.rng = random.Random(args.seed)
        self.sem = asyncio.Semaphore(args.workers)
        self.ledger = CostLedger(total_budget=math.inf)
        self.calls: list[dict[str, Any]] = []
        self.out_dir.mkdir(parents=True, exist_ok=True)
        set_log_dir(self.out_dir / "llm_raw")

    async def llm(
        self,
        *,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        kind: str,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        async with self.sem:
            t0 = time.monotonic()
            response: LLMResponse = await openai_compatible_call(
                messages,
                self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                retry_config=RetryConfig(max_retries=self.args.retries, http_timeout=self.args.http_timeout),
            )
            elapsed = round(time.monotonic() - t0, 3)
        cost = self.ledger.record(kind, response.usage)
        self.calls.append({
            "kind": kind,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
            "cost": cost,
            "latency_s": elapsed,
        })
        return response.content or ""

    async def generate_one(self, idx: int) -> Candidate:
        md = await self.llm(
            prompt=GENERATION_PROMPT.format(problem=self.problem_text),
            system_prompt=SYSTEM_PROMPT,
            temperature=1.0,
            max_tokens=self.args.max_tokens_generate,
            kind="generate",
        )
        return Candidate(idx=idx, md=md, origin=f"gen0:{idx:02d}")

    async def mutate_one(self, gen: int, candidate: Candidate, comparisons: list[dict[str, Any]]) -> Candidate:
        code = extract_code(candidate.md)
        wins, ties, losses = collect_feedback(candidate.idx, comparisons)
        if wins or ties or losses:
            prompt = MUTATION_PROMPT_WITH_FEEDBACK.format(
                problem=self.problem_text,
                code=code,
                feedback_sections=build_feedback_sections(wins, ties, losses),
            )
        else:
            prompt = MUTATION_PROMPT_NO_FEEDBACK.format(problem=self.problem_text, code=code)
        md = await self.llm(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=1.0,
            max_tokens=self.args.max_tokens_generate,
            kind=f"mutate_gen{gen}",
        )
        return Candidate(idx=candidate.idx, md=md, origin=f"gen{gen}:mutated:{candidate.idx:02d}")

    async def judge_pair(self, gen_label: str, rnd: int, ia: int, ib: int, a: Candidate, b: Candidate) -> dict[str, Any]:
        flip = self.rng.choice([False, True])
        left, right = (b, a) if flip else (a, b)
        prompt = PAIRWISE_PROMPT.format(
            problem=self.problem_text,
            code_a=extract_code(left.md),
            code_b=extract_code(right.md),
        )
        text = await self.llm(
            prompt=prompt,
            system_prompt=None,
            temperature=0.0,
            max_tokens=self.args.max_tokens_judge,
            kind=f"judge_{gen_label}",
        )
        parsed = parse_json_object(text)
        winner = str(parsed.get("winner", "TIE")).strip().upper()
        if winner not in {"A", "B", "TIE"}:
            winner = "TIE"
        if flip:
            mapped = "B" if winner == "A" else "A" if winner == "B" else "TIE"
            feedback_a = str(parsed.get("feedback_b", ""))
            feedback_b = str(parsed.get("feedback_a", ""))
        else:
            mapped = winner
            feedback_a = str(parsed.get("feedback_a", ""))
            feedback_b = str(parsed.get("feedback_b", ""))
        return {
            "round": rnd,
            "idx_a": ia,
            "idx_b": ib,
            "winner": mapped,
            "feedback_a": feedback_a,
            "feedback_b": feedback_b,
            "presentation_flipped": flip,
            "raw_winner": winner,
        }

    def pair_jobs(self, n: int, rounds: int) -> list[tuple[int, int, int]]:
        jobs: list[tuple[int, int, int]] = []
        for rnd in range(rounds):
            indices = list(range(n))
            self.rng.shuffle(indices)
            for k in range(0, n - 1, 2):
                jobs.append((rnd, indices[k], indices[k + 1]))
        return jobs

    async def run_tournament(self, gen_label: str, candidates: list[Candidate], rounds: int) -> list[dict[str, Any]]:
        cmp_dir = self.out_dir / "comparisons"
        cmp_dir.mkdir(exist_ok=True)
        path = cmp_dir / f"{gen_label}.jsonl"
        jobs = self.pair_jobs(len(candidates), rounds)
        print(f"[{gen_label}] judging {len(jobs)} pairs", flush=True)
        tasks = [
            self.judge_pair(gen_label, rnd, ia, ib, candidates[ia], candidates[ib])
            for rnd, ia, ib in jobs
        ]
        comparisons: list[dict[str, Any]] = []
        for done, coro in enumerate(asyncio.as_completed(tasks), start=1):
            record = await coro
            comparisons.append(record)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if done % 10 == 0 or done == len(tasks):
                print(f"[{gen_label}] judge {done}/{len(tasks)}", flush=True)
        return comparisons

    async def run(self) -> None:
        start = time.monotonic()
        print(f"[setup] pid={self.pid} model={self.model} out={self.out_dir}", flush=True)
        print(f"[gen0] generating {self.args.pop} candidates", flush=True)
        candidates = await asyncio.gather(*(self.generate_one(i) for i in range(self.args.pop)))
        candidates = sorted(candidates, key=lambda c: c.idx)
        self.save_population(0, candidates)

        rankings: dict[str, Any] = {}
        for gen in range(1, self.args.generations + 1):
            comparisons = await self.run_tournament(f"gen{gen}", candidates, self.args.k_rounds)
            scores, ranking = compute_bt_scores(len(candidates), comparisons)
            rankings[f"gen{gen}"] = self.save_ranking(f"gen{gen}", scores, ranking, comparisons)
            top5 = ranking[: self.args.elite]
            top_mutate = ranking[: self.args.pop - self.args.eliminate]
            print(f"[gen{gen}] top5={top5} mutate={top_mutate}", flush=True)
            mutations = await asyncio.gather(*(self.mutate_one(gen, candidates[i], comparisons) for i in top_mutate))
            next_candidates = list(candidates)
            by_idx = {m.idx: m for m in mutations}
            for i in top_mutate:
                next_candidates[i] = by_idx[i]
            for bottom_i, elite_i in zip(ranking[-self.args.elite:], top5):
                next_candidates[bottom_i] = Candidate(
                    idx=bottom_i,
                    md=candidates[elite_i].md,
                    origin=f"gen{gen}:elite_copy:{elite_i:02d}",
                )
            candidates = next_candidates
            self.save_population(gen, candidates)

        final_comparisons = await self.run_tournament("final", candidates, self.args.m_rounds)
        final_scores, final_ranking = compute_bt_scores(len(candidates), final_comparisons)
        rankings["final"] = self.save_ranking("final", final_scores, final_ranking, final_comparisons)
        winner_idx = final_ranking[0]
        selected = extract_code(candidates[winner_idx].md)
        selected_path = self.out_dir / "selected_solution.cpp"
        selected_path.write_text(selected + "\n", encoding="utf-8")
        print(f"[final] winner=sol{winner_idx:02d} selected={selected_path}", flush=True)

        eval_summary = {}
        if self.args.evaluate:
            eval_summary["selected"] = self.evaluate_solution(selected_path, "selected_solution")
            if self.args.evaluate_gen0:
                eval_summary["gen0"] = self.evaluate_population(self.out_dir / "solutions" / "gen0", limit=self.args.evaluate_gen0_limit)
        if self.args.evaluate_official_ac:
            eval_summary["official_ac"] = self.evaluate_official_ac()

        summary = {
            "pid": self.pid,
            "problem": self.meta,
            "model": self.model,
            "paper_protocol": {
                "n": self.args.pop,
                "K": self.args.k_rounds,
                "T": self.args.generations,
                "M": self.args.m_rounds,
                "expected_llm_calls": self.args.pop
                + self.args.generations * (self.args.pop * self.args.k_rounds // 2 + (self.args.pop - self.args.eliminate))
                + self.args.pop * self.args.m_rounds // 2,
            },
            "actual_llm_calls": len(self.calls),
            "tokens": {
                "input": sum(c["input_tokens"] for c in self.calls),
                "output": sum(c["output_tokens"] for c in self.calls),
                "total": sum(c["total_tokens"] for c in self.calls),
            },
            "cost_usd": round(self.ledger.total_spent, 6),
            "wall_time_s": round(time.monotonic() - start, 3),
            "winner_idx": winner_idx,
            "rankings": rankings,
            "evaluation": eval_summary,
            "evaluation_summary": summarize_evaluation(eval_summary),
        }
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.out_dir / "calls.jsonl").write_text(
            "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in self.calls),
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False)[:4000], flush=True)

    def save_population(self, gen: int, candidates: list[Candidate]) -> None:
        root = self.out_dir / "solutions" / f"gen{gen}"
        root.mkdir(parents=True, exist_ok=True)
        for c in candidates:
            (root / f"sol{c.idx:02d}.md").write_text(c.md, encoding="utf-8")
            (root / f"sol{c.idx:02d}.cpp").write_text(extract_code(c.md) + "\n", encoding="utf-8")

    def save_ranking(self, label: str, scores: list[float], ranking: list[int], comparisons: list[dict[str, Any]]) -> dict[str, Any]:
        data = {
            "label": label,
            "ranking": ranking,
            "scores": {str(i): scores[i] for i in range(len(scores))},
            "top5": ranking[:5],
            "bottom5": ranking[-5:],
            "comparison_count": len(comparisons),
        }
        root = self.out_dir / "rankings"
        root.mkdir(exist_ok=True)
        (root / f"{label}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return data

    def evaluate_population(self, root: Path, *, limit: int = 0) -> list[dict[str, Any]]:
        results = []
        files = sorted(root.glob("sol*.cpp"))
        if limit > 0:
            files = files[:limit]
        for path in files:
            results.append(self.evaluate_solution(path, path.stem))
        return results

    def evaluate_solution(self, solution_path: Path, label: str) -> dict[str, Any]:
        if not self.private_judge_dir:
            payload = {"label": label, "error": "private judge dir not configured"}
            self.write_eval_payload(label, payload)
            return payload
        cmd = [
            sys.executable,
            "examples/evaluate_cf73_solution.py",
            "--problem-dir",
            str(self.problem_dir),
            "--judge-dir",
            str(self.private_judge_dir),
            "--solution",
            str(solution_path),
            "--timeout",
            str(self.args.eval_timeout),
        ]
        if self.args.eval_max_tests:
            cmd.extend(["--max-tests", str(self.args.eval_max_tests)])
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.args.eval_process_timeout)
        payload: dict[str, Any]
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            payload = {"stdout": proc.stdout[-3000:], "stderr": proc.stderr[-3000:]}
        payload.update({"label": label, "returncode": proc.returncode, "solution": str(solution_path)})
        self.write_eval_payload(label, payload)
        print(f"[eval] {label}: {payload.get('status')} rc={proc.returncode}", flush=True)
        return payload

    def evaluate_official_ac(self) -> dict[str, Any]:
        label = "official_ac"
        if not self.private_judge_dir:
            payload = {
                "label": label,
                "error": "private judge dir not configured",
                "path_in_package": OFFICIAL_AC_PATH,
            }
            self.write_eval_payload(label, payload)
            print(f"[eval] {label}: error private judge dir not configured", flush=True)
            return payload

        package = self.private_judge_dir / "polygon_package.zip"
        with tempfile.TemporaryDirectory(prefix="opendeepthink_official_ac_") as td:
            extracted = Path(td) / "official_ac.cpp"
            extraction = extract_official_ac_source(package, extracted)
            if extraction.get("error"):
                payload = {"label": label, **extraction}
                self.write_eval_payload(label, payload)
                print(f"[eval] {label}: error {payload['error']}", flush=True)
                return payload
            payload = self.evaluate_solution(extracted, label)

        payload.update({
            "solution": OFFICIAL_AC_PATH,
            "path_in_package": OFFICIAL_AC_PATH,
            "source_package": str(package),
        })
        self.write_eval_payload(label, payload)
        return payload

    def write_eval_payload(self, label: str, payload: dict[str, Any]) -> None:
        out = self.out_dir / "eval"
        out.mkdir(exist_ok=True)
        payload.setdefault("label", label)
        (out / f"{label}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_code(md: str) -> str:
    matches = re.findall(r"```(?:cpp|c\+\+|cxx|cc)?\s*\n(.*?)```", md or "", re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return (md or "").strip()


def extract_official_ac_source(package: Path, destination: Path) -> dict[str, Any]:
    if not package.is_file():
        return {
            "error": f"Polygon package not found: {package}",
            "path_in_package": OFFICIAL_AC_PATH,
            "source_package": str(package),
        }
    try:
        with zipfile.ZipFile(package) as zf:
            data = zf.read(OFFICIAL_AC_PATH)
    except KeyError:
        return {
            "error": f"{OFFICIAL_AC_PATH} not found in Polygon package",
            "path_in_package": OFFICIAL_AC_PATH,
            "source_package": str(package),
        }
    except zipfile.BadZipFile as e:
        return {
            "error": f"invalid Polygon package: {e}",
            "path_in_package": OFFICIAL_AC_PATH,
            "source_package": str(package),
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return {
        "path": str(destination),
        "path_in_package": OFFICIAL_AC_PATH,
        "source_package": str(package),
    }


def summarize_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if selected := evaluation.get("selected"):
        summary["selected"] = compact_eval_result(selected)
    if gen0 := evaluation.get("gen0"):
        summary["gen0"] = summarize_eval_population(gen0)
    if official := evaluation.get("official_ac"):
        summary["official_ac_sanity"] = compact_eval_result(official)
    return summary


def summarize_eval_population(results: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(evaluation_status(r) for r in results)
    return {
        "total": len(results),
        "passed": counts.get("passed", 0),
        "failed": counts.get("failed", 0),
        "statuses": dict(sorted(counts.items())),
    }


def compact_eval_result(result: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {"status": evaluation_status(result)}
    for key in ("tests_passed", "tests_total", "returncode", "path_in_package"):
        if key in result:
            compact[key] = result[key]
    if "error" in result:
        compact["error"] = result["error"]
    return compact


def evaluation_status(result: dict[str, Any]) -> str:
    if status := result.get("status"):
        return str(status)
    if result.get("error"):
        return "error"
    if result.get("returncode") not in (None, 0):
        return "error"
    return "unknown"


def parse_json_object(text: str) -> dict[str, Any]:
    clean = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", (text or "").strip(), flags=re.MULTILINE)
    for candidate in (clean,):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}


def collect_feedback(sol_idx: int, comparisons: list[dict[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    wins: list[str] = []
    ties: list[str] = []
    losses: list[str] = []
    for c in comparisons:
        ia, ib, winner = int(c["idx_a"]), int(c["idx_b"]), str(c.get("winner", "TIE"))
        if ia == sol_idx:
            feedback = rewrite_feedback(str(c.get("feedback_a", "")), mine="A")
            if winner == "A":
                wins.append(feedback)
            elif winner == "B":
                losses.append(feedback)
            else:
                ties.append(feedback)
        elif ib == sol_idx:
            feedback = rewrite_feedback(str(c.get("feedback_b", "")), mine="B")
            if winner == "B":
                wins.append(feedback)
            elif winner == "A":
                losses.append(feedback)
            else:
                ties.append(feedback)
    return wins, ties, losses


def rewrite_feedback(text: str, *, mine: str) -> str:
    other = "B" if mine == "A" else "A"
    text = re.sub(fr"\bSolution {mine}\b", "this solution", text)
    text = re.sub(fr"\bSolution {other}\b", "the other solution", text)
    return text


def build_feedback_sections(wins: list[str], ties: list[str], losses: list[str]) -> str:
    parts = []
    if wins:
        parts.append("### Wins (this solution was judged better):\n" + "\n".join(f"- {fb}" for fb in wins))
    if ties:
        parts.append("### Ties (judged equally likely to be Accepted):\n" + "\n".join(f"- {fb}" for fb in ties))
    if losses:
        parts.append("### Losses (this solution was judged worse):\n" + "\n".join(f"- {fb}" for fb in losses))
    return "\n\n".join(parts) + "\n\n"


def compute_bt_scores(n: int, comparisons: list[dict[str, Any]], lam: float = 0.01) -> tuple[list[float], list[int]]:
    wins = [[0.0 for _ in range(n)] for _ in range(n)]
    for c in comparisons:
        ia, ib = int(c["idx_a"]), int(c["idx_b"])
        winner = str(c.get("winner", "TIE")).upper()
        if ia == ib:
            continue
        if winner == "A":
            wins[ia][ib] += 1.0
        elif winner == "B":
            wins[ib][ia] += 1.0
        else:
            wins[ia][ib] += 0.5
            wins[ib][ia] += 0.5
    strengths = [1.0 for _ in range(n)]
    floor = 1e-12
    for _ in range(1000):
        total_wins = [sum(row) for row in wins]
        updated = []
        for i in range(n):
            denom = lam
            for j in range(n):
                if i == j:
                    continue
                comps = wins[i][j] + wins[j][i]
                if comps:
                    denom += comps / max(strengths[i] + strengths[j], floor)
            value = floor if total_wins[i] <= 0 else total_wins[i] / denom
            updated.append(max(value, floor))
        scale = sum(updated) / n
        if scale > 0:
            updated = [x / scale for x in updated]
        delta = max(abs(updated[i] - strengths[i]) for i in range(n))
        strengths = updated
        if delta < 1e-9:
            break
    mean = sum(strengths) / n
    var = sum((x - mean) ** 2 for x in strengths) / n
    std = math.sqrt(var)
    scores = [(x - mean) / std if std > 1e-9 else x - mean for x in strengths]
    ranking = sorted(range(n), key=lambda i: (-scores[i], i))
    return scores, ranking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed OpenDeepThink flow on one CF-73 problem")
    parser.add_argument("--pid", default="2161F")
    parser.add_argument("--problem-dir", required=True)
    parser.add_argument("--private-judge-dir")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default=os.environ.get("NANOMA_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--elite", type=int, default=5)
    parser.add_argument("--eliminate", type=int, default=5)
    parser.add_argument("--k-rounds", type=int, default=4)
    parser.add_argument("--m-rounds", type=int, default=10)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens-generate", type=int, default=12000)
    parser.add_argument("--max-tokens-judge", type=int, default=2048)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--http-timeout", type=float, default=300.0)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--evaluate-gen0", action="store_true")
    parser.add_argument("--evaluate-official-ac", action="store_true")
    parser.add_argument("--evaluate-gen0-limit", type=int, default=0)
    parser.add_argument("--eval-max-tests", type=int, default=0)
    parser.add_argument("--eval-timeout", type=float, default=15.0)
    parser.add_argument("--eval-process-timeout", type=float, default=1800.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Path(args.out_dir).exists() and any(Path(args.out_dir).iterdir()):
        raise SystemExit(f"out dir is not empty: {args.out_dir}")
    asyncio.run(Runner(args).run())


if __name__ == "__main__":
    main()
