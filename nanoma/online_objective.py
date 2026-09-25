"""Opt-in feedback for tasks that permit online objective measurements.

No judge calls, hidden scores, or task-name rules live here. Unconfigured
runtimes keep their existing acceptance and delivery paths.
"""
from __future__ import annotations

import math
import re
from typing import Any


def contract(value: dict | None) -> dict | None:
    if value is None:
        return None
    if value.get("visibility") != "agent_submission":
        raise ValueError("online objective requires agent_submission visibility")
    if value.get("direction") not in {"maximize", "minimize"}:
        raise ValueError("online objective requires an explicit direction")
    if value.get("selection") not in {"score_first", "valid_then_score", "pass_rate_first"}:
        raise ValueError("online objective requires an explicit selection policy")
    if not value.get("metric_id"):
        raise ValueError("online objective requires a metric identity")
    return dict(value)


def finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def parse_feedback(text: str) -> dict | None:
    """Parse only the result of an intentional submission, never list/history."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    if "Results" not in text:
        return None
    score_match = re.search(r"\bScore:\s*([-+\d.eE]+)", text)
    score = finite(score_match.group(1)) if score_match else None
    passed = re.search(r"Passed:\s*(\d+)\s*/\s*(\d+)", text)
    rate = re.search(r"Pass rate:\s*([\d.]+)%", text)
    pass_rate = finite(rate.group(1)) / 100 if rate and finite(rate.group(1)) is not None else None
    if pass_rate is None and passed and int(passed.group(2)):
        pass_rate = int(passed.group(1)) / int(passed.group(2))
    if pass_rate is None and "All tests passed!" in text:
        pass_rate = 1.0
    if score is None or pass_rate is None or not 0 <= pass_rate <= 1:
        return None
    round_match = re.search(r"(?:^|\n)\s*(\S+) Results\s*(?:\n|$)", text)
    failed = re.findall(r"^\s+- (.+)$", text, re.MULTILINE)
    return {
        "score": score, "pass_rate": pass_rate,
        "valid": not bool(re.search(r"Valid:\s*(?:no|false)\b", text, re.I)),
        "round": round_match.group(1) if round_match else None,
        "failed": failed[:20], "failed_total": len(failed),
    }


def from_result(result: dict) -> dict | None:
    """Use the final response, not earlier failed attempts or a history listing."""
    observation = result.get("objective_observation")
    if isinstance(observation, dict):
        score, rate = finite(observation.get("score")), finite(observation.get("pass_rate"))
        if score is not None and rate is not None and 0 <= rate <= 1:
            return {**observation, "score": score, "pass_rate": rate}
    nested = result.get("submission")
    if isinstance(nested, dict):
        return from_result(nested)
    return parse_feedback("\n".join(str(result.get(k) or "") for k in ("stdout", "output", "stderr")))


def rank(entry: dict, spec: dict) -> tuple | None:
    """Comparable official objectives only; local check scales stay separate."""
    if (entry.get("source") != "official" or not entry.get("counts", True)
            or entry.get("metric_id") != spec["metric_id"]
            or entry.get("direction") != spec["direction"]
            or entry.get("selection") != spec["selection"]):
        return None
    score = finite(entry.get("metric"))
    if score is None:
        return None
    oriented = score if spec["direction"] == "maximize" else -score
    if spec["selection"] == "valid_then_score" and not entry.get("valid", True):
        return None
    if spec["selection"] == "pass_rate_first":
        rate = finite(entry.get("pass_rate"))
        if rate is None or rate <= 0:
            return None
        # SForge only breaks ties by score after complete pass rate.
        return (rate, oriented if rate >= 1 else 0.0)
    return (oriented,)


def summarize(entries: list[dict], spec: dict) -> str:
    comparable = [(entry, rank(entry, spec)) for entry in entries]
    comparable = [(entry, key) for entry, key in comparable if key is not None]
    if not comparable:
        return ""
    best, best_key = max(comparable, key=lambda item: item[1])
    latest, _ = comparable[-1]
    goal = finite(spec.get("goal"))
    met = None if goal is None else (
        float(best["metric"]) >= goal if spec["direction"] == "maximize"
        else float(best["metric"]) <= goal
    )
    text = (
        f"[Online objective] {spec['metric_id']} ({spec['direction']}): "
        f"latest={latest['metric']}; best={best['metric']} "
        f"on artifact {best.get('state') or 'unbound'}, round {best.get('round') or '?'}. "
        f"Submission validity={latest.get('valid')}; pass_rate={latest.get('pass_rate')}. "
        + ("No explicit goal threshold; validity is not an optimality claim. "
           if met is None else f"Explicit goal satisfied={met and bool(best.get('valid'))}. ")
    )
    failures = latest.get("failed") or []
    if failures:
        text += "Reported failures: " + "; ".join(str(x)[:100] for x in failures[:3]) + ". "
    window = max(0, int(spec.get("plateau_window", 3)))
    if window:
        # Repeated measurements and byte-identical candidates are not new routes.
        since_best: set[str] = set()
        running_best = None
        for entry, key in comparable:
            if running_best is None or key > running_best:
                running_best = key
                since_best.clear()
            elif entry.get("state") and entry["state"] != best.get("state"):
                since_best.add(entry["state"])
        if len(since_best) >= window and not met:
            text += (
                f"{len(since_best)} distinct measured artifacts have not improved the best. "
                "Reconsider the underlying hypothesis and the cheapest discriminating check; "
                "if useful, explore a materially different method rather than another parameter tweak. "
                "Choose whether and how to delegate; no additional agents are required."
            )
    return text[:1600]


def runtime_time_limit(requested: float, deadline: float, wall_seconds: float,
                       safety: float, *, now: float, elapsed: float = 0.0) -> float:
    """An inner deadline leaves a drain interval before the outer hard stop."""
    if wall_seconds <= 0:
        return requested
    remaining = max(0.001, deadline - now - 2 * max(0.0, safety))
    if requested > 0:
        remaining = min(remaining, requested)
    return max(0.0, elapsed) + remaining
