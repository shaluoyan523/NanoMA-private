"""Deterministic checkpoints for tool-mediated agent orchestration."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FixedAgentSpec:
    key: str
    bio: str
    task: str
    assignment: str = ""
    delivery_answer_regex: str = ""
    allowed_paths: tuple[str, ...] = ()
    read_only: bool = False
    end_elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class FixedCheckpoint:
    key: str
    parent_key: str
    agents: tuple[FixedAgentSpec, ...]
    mode: str = "progress"
    progress_logic: str = "any"
    depends_on: tuple[str, ...] = ()
    min_parent_turns: int = 0
    min_parent_tool_calls: int = 0
    min_parent_age_seconds: float = 0.0
    min_stall_turns: int = 0
    min_parent_web_no_gain_calls: int = 0
    min_parent_unavailable_tool_calls: int = 0
    require_parent_without_candidate: bool = False
    not_before_elapsed_seconds: float = 0.0
    fallback_elapsed_seconds: float = 0.0
    fallback_elapsed_fraction: float = 0.0


@dataclass(frozen=True)
class FixedVerificationGate:
    """Runtime-owned command used to accept or reject a writer candidate."""

    argv: tuple[str, ...]
    cwd: str = ""
    timeout_seconds: float = 900.0
    acceptance: str = "exit-code"
    score_direction: str = "maximize"
    always_run: bool = False


@dataclass(frozen=True)
class FixedOrchestrationPlan:
    name: str
    root_instructions: str
    checkpoints: tuple[FixedCheckpoint, ...]
    max_live_agents: int = 8
    max_concurrent_llm: int = 8
    root_parks_after_spawn: bool = True
    completion_checkpoint: str = ""
    task_repo: str = ""
    root_source_writes: bool = True
    direct_build_gate: bool = False
    verification_gate: FixedVerificationGate | None = None
    snapshot_paths: tuple[str, ...] = ()
    snapshot_enabled: bool = True


_TOPOLOGY_COMMON = """
You are part of a runtime-managed agent topology. Work only on your assigned
workstream. You may create children autonomously whenever NanoMA's normal spawn
policy allows. Runtime checkpoints may additionally require one specific parent
to create listed roles with NanoMA's spawn or spawn_many tool; that requirement
supplements and never replaces autonomous orchestration. Runtime never creates
an agent on your behalf. Send concrete findings, changed files, validation
commands, and remaining risks to your parent. Never submit unless your role
explicitly says you are the root submitter. Keep changes attributable and leave
the shared task in a valid state.
""".strip()


def _required_string(data: dict[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def load_topology_orchestration_plan(path: str | Path) -> FixedOrchestrationPlan:
    """Load an executable topology exported by the EdgeBench simulator."""
    source = Path(path).expanduser()
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("schema_version") not in {1, 2, 3}:
        raise ValueError("topology schema_version must be 1, 2, or 3")

    task = data.get("task")
    runtime = data.get("runtime")
    if not isinstance(task, dict) or not isinstance(runtime, dict):
        raise ValueError("topology requires task and runtime objects")
    spawn_execution = str(
        runtime.get("spawn_execution", "parent_agent_via_nanoma_tool")
    ).strip()
    if spawn_execution != "parent_agent_via_nanoma_tool":
        raise ValueError(
            "runtime.spawn_execution must be parent_agent_via_nanoma_tool; "
            "runtime checkpoints may steer spawn behavior but may not create agents"
        )
    if bool(runtime.get("runtime_creates_agents", False)):
        raise ValueError("runtime.runtime_creates_agents must be false")
    spawn_policy = str(
        runtime.get("spawn_policy", "checkpoint_supplements_autonomous")
    ).strip()
    if spawn_policy != "checkpoint_supplements_autonomous":
        raise ValueError(
            "runtime.spawn_policy must be checkpoint_supplements_autonomous; "
            "fixed checkpoints may not disable NanoMA autonomous spawning"
        )
    if not bool(runtime.get("autonomous_spawn_enabled", True)):
        raise ValueError("runtime.autonomous_spawn_enabled must be true")
    task_id = _required_string(task, "id", "task")
    objective = _required_string(task, "objective", "task")
    gates = [str(item) for item in task.get("gates", []) if str(item).strip()]
    budget_minutes = max(1.0, float(runtime.get("budget_minutes", 120)))
    nodes = runtime.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("runtime.nodes must be a non-empty array")

    node_ids = {_required_string(node, "id", "runtime.nodes[]") for node in nodes}
    grouped_nodes: dict[str, dict[str, Any]] = {}
    group_order: list[str] = []
    node_checkpoint_keys: dict[str, str] = {}
    for node in nodes:
        node_id = _required_string(node, "id", "runtime.nodes[]")
        parent_key = _required_string(node, "spawn_parent", f"node {node_id}")
        spawn_executor = str(node.get("spawn_executor", "parent_agent")).strip()
        if spawn_executor != "parent_agent":
            raise ValueError(f"node {node_id}.spawn_executor must be parent_agent")
        spawn_tool = str(node.get("spawn_tool", "nanoma")).strip()
        if spawn_tool != "nanoma":
            raise ValueError(f"node {node_id}.spawn_tool must be nanoma")
        if parent_key != "root" and parent_key not in node_ids:
            raise ValueError(f"node {node_id} has unknown spawn_parent {parent_key}")
        role = _required_string(node, "role", f"node {node_id}")
        assignment = _required_string(node, "assignment", f"node {node_id}")
        delivery_answer_regex = str(node.get("delivery_answer_regex", "")).strip()
        if delivery_answer_regex:
            try:
                re.compile(delivery_answer_regex)
            except re.error as exc:
                raise ValueError(
                    f"node {node_id}.delivery_answer_regex is invalid: {exc}"
                ) from exc
        guide = str(node.get("guide", "Own this workstream and return validated evidence.")).strip()
        read_only = bool(node.get("read_only", False))
        raw_paths = node.get("allowed_paths", ["*"] if not read_only else [])
        if not isinstance(raw_paths, list):
            raise ValueError(f"node {node_id}.allowed_paths must be an array")
        allowed_paths = tuple(str(item).strip() for item in raw_paths if str(item).strip())
        dependencies = tuple(str(item) for item in node.get("depends_on", []))
        start_minute = max(0.0, float(node.get("start_minute", 0)))
        end_minute = max(0.0, float(node.get("end_minute", 0)))
        if end_minute and end_minute <= start_minute:
            raise ValueError(
                f"node {node_id}.end_minute must be greater than start_minute"
            )
        mode = str(node.get("mode", "progress"))
        if mode not in {"progress", "settled", "stalled"}:
            raise ValueError(f"node {node_id} has unsupported mode {mode}")
        progress_logic = str(node.get("progress_logic", "all"))
        if progress_logic not in {"any", "all"}:
            raise ValueError(f"node {node_id} has unsupported progress_logic {progress_logic}")
        spawn_group = str(node.get("spawn_group") or node_id).strip()
        if not spawn_group:
            raise ValueError(f"node {node_id}.spawn_group must not be empty")
        checkpoint_key = str(
            node.get("checkpoint_key") or f"spawn-{spawn_group}"
        ).strip()
        if not checkpoint_key:
            raise ValueError(f"node {node_id}.checkpoint_key must not be empty")
        task_prompt = f"""{_TOPOLOGY_COMMON}

ROLE: {role}
ASSIGNMENT: {assignment}
ROLE CONTRACT: {guide}
TASK OBJECTIVE: {objective}
ACCEPTANCE GATES: {'; '.join(gates) if gates else 'Return reproducible validation evidence.'}
"""
        group_contract = {
            "key": checkpoint_key,
            "parent_key": parent_key,
            "mode": mode,
            "progress_logic": progress_logic,
            "depends_on": dependencies,
            "min_parent_turns": max(0, int(node.get("min_parent_turns", 1))),
            "min_parent_tool_calls": max(0, int(node.get("min_parent_tool_calls", 1))),
            "min_parent_age_seconds": max(
                0.0, float(node.get("min_parent_age_seconds", 0))
            ),
            "min_stall_turns": max(0, int(node.get("min_stall_turns", 0))),
            "min_parent_web_no_gain_calls": max(
                0, int(node.get("min_parent_web_no_gain_calls", 0))
            ),
            "min_parent_unavailable_tool_calls": max(
                0, int(node.get("min_parent_unavailable_tool_calls", 0))
            ),
            "require_parent_without_candidate": bool(
                node.get("require_parent_without_candidate", False)
            ),
            "not_before_elapsed_seconds": start_minute * 60,
            "fallback_elapsed_seconds": start_minute * 60,
            "fallback_elapsed_fraction": min(1.0, start_minute / budget_minutes),
        }
        existing = grouped_nodes.get(spawn_group)
        if existing is None:
            existing = {**group_contract, "agents": []}
            grouped_nodes[spawn_group] = existing
            group_order.append(spawn_group)
        else:
            conflicting = [
                key
                for key, value in group_contract.items()
                if existing.get(key) != value
            ]
            if conflicting:
                raise ValueError(
                    f"spawn_group {spawn_group!r} has inconsistent checkpoint fields: "
                    f"{', '.join(conflicting)}"
                )
        existing["agents"].append(FixedAgentSpec(
            key=node_id,
            bio=role,
            task=task_prompt,
            assignment=assignment,
            delivery_answer_regex=delivery_answer_regex,
            allowed_paths=allowed_paths,
            read_only=read_only,
            end_elapsed_seconds=end_minute * 60,
        ))
        node_checkpoint_keys[node_id] = checkpoint_key

    checkpoints: list[FixedCheckpoint] = []
    for spawn_group in group_order:
        group = grouped_nodes[spawn_group]
        normalized_dependencies: list[str] = []
        for dependency in group["depends_on"]:
            normalized = node_checkpoint_keys.get(dependency, dependency)
            if not normalized.startswith("spawn-") and normalized in grouped_nodes:
                normalized = grouped_nodes[normalized]["key"]
            normalized_dependencies.append(normalized)
        checkpoints.append(FixedCheckpoint(
            key=group["key"],
            parent_key=group["parent_key"],
            agents=tuple(group["agents"]),
            mode=group["mode"],
            progress_logic=group["progress_logic"],
            depends_on=tuple(normalized_dependencies),
            min_parent_turns=group["min_parent_turns"],
            min_parent_tool_calls=group["min_parent_tool_calls"],
            min_parent_age_seconds=group["min_parent_age_seconds"],
            min_stall_turns=group["min_stall_turns"],
            min_parent_web_no_gain_calls=group["min_parent_web_no_gain_calls"],
            min_parent_unavailable_tool_calls=group[
                "min_parent_unavailable_tool_calls"
            ],
            require_parent_without_candidate=group[
                "require_parent_without_candidate"
            ],
            not_before_elapsed_seconds=group["not_before_elapsed_seconds"],
            fallback_elapsed_seconds=group["fallback_elapsed_seconds"],
            fallback_elapsed_fraction=group["fallback_elapsed_fraction"],
        ))

    checkpoint_keys = {item.key for item in checkpoints}
    unknown_dependencies = {
        dependency
        for checkpoint in checkpoints
        for dependency in checkpoint.depends_on
        if dependency not in checkpoint_keys
    }
    if unknown_dependencies:
        raise ValueError(
            f"topology has unknown checkpoint dependencies "
            f"{sorted(unknown_dependencies)}"
        )
    for checkpoint in checkpoints:
        if checkpoint.parent_key != "root" and f"spawn-{checkpoint.parent_key}" not in checkpoint_keys:
            if checkpoint.parent_key not in node_checkpoint_keys:
                raise ValueError(f"checkpoint {checkpoint.key} has unavailable parent")

    root_role = str(runtime.get("root_role", "Root / Integrator")).strip()
    services = runtime.get("services", {})
    services = services if isinstance(services, dict) else {}
    raw_gate = services.get("verification_gate", services.get("build_gate", {}))
    verification_gate = None
    if isinstance(raw_gate, dict) and raw_gate.get("owner") == "runtime":
        raw_argv = raw_gate.get("argv")
        if not isinstance(raw_argv, list) or not raw_argv:
            raise ValueError("runtime.services verification gate requires non-empty argv")
        argv = tuple(str(item).strip() for item in raw_argv)
        if any(not item for item in argv):
            raise ValueError("runtime.services verification gate argv entries must be non-empty")
        if raw_gate.get("shell") is not False:
            raise ValueError("runtime.services verification gate must set shell=false")
        acceptance = str(raw_gate.get("acceptance", "exit-code")).strip()
        if acceptance not in {"exit-code", "judge-valid", "judge-valid-nondecreasing"}:
            raise ValueError(f"unsupported verification gate acceptance {acceptance}")
        score_direction = str(raw_gate.get("score_direction", "maximize")).strip()
        if score_direction not in {"maximize", "minimize"}:
            raise ValueError(f"unsupported verification gate score_direction {score_direction}")
        verification_gate = FixedVerificationGate(
            argv=argv,
            cwd=str(raw_gate.get("cwd", "")).strip(),
            timeout_seconds=max(1.0, float(raw_gate.get("timeout_seconds", 900))),
            acceptance=acceptance,
            score_direction=score_direction,
            always_run=bool(raw_gate.get("always_run", False)),
        )
    raw_snapshot_paths = runtime.get("snapshot_paths", [])
    if not isinstance(raw_snapshot_paths, list):
        raise ValueError("runtime.snapshot_paths must be an array")
    snapshot_paths = tuple(
        str(item).strip().rstrip("/")
        for item in raw_snapshot_paths
        if str(item).strip().rstrip("/")
    )
    direct_build_gate = verification_gate is not None
    runtime_owns_submission = bool(
        verification_gate and verification_gate.argv == ("sforge-submit",)
    )
    completion = str(runtime.get("completion_checkpoint", "")).strip()
    completion = node_checkpoint_keys.get(completion, completion)
    if completion and not completion.startswith("spawn-") and completion in grouped_nodes:
        completion = grouped_nodes[completion]["key"]
    if completion and completion not in checkpoint_keys:
        raise ValueError(f"unknown completion_checkpoint {completion}")
    return FixedOrchestrationPlan(
        name=f"topology:{task_id}",
        root_instructions=f"""[Runtime topology: {task_id}]
ROLE: {root_role}
You are the unique submitter and best-version owner. Objective: {objective}
At each exported checkpoint, Runtime will constrain the designated parent to a
NanoMA spawn or spawn_many tool call. Runtime itself never creates children.
Outside those checkpoint turns, use NanoMA's spawn and spawn_many tools normally
whenever adaptive delegation is useful. Coordinate all planned and autonomous
children and enforce these gates: {'; '.join(gates) if gates else 'validate the final deliverable'}.
{'Runtime owns all official sforge-submit calls; do not submit manually.' if runtime_owns_submission else ''}
""".strip(),
        checkpoints=tuple(checkpoints),
        max_live_agents=max(1, int(runtime.get("max_live_agents", 8))),
        max_concurrent_llm=max(1, int(runtime.get("max_concurrent_llm", 8))),
        root_parks_after_spawn=bool(runtime.get("root_parks_after_spawn", True)),
        completion_checkpoint=completion,
        task_repo=str(runtime.get("task_repo", "")).strip(),
        root_source_writes=bool(runtime.get("root_source_writes", True)),
        direct_build_gate=direct_build_gate,
        verification_gate=verification_gate,
        snapshot_paths=snapshot_paths,
        snapshot_enabled=bool(runtime.get("snapshot_enabled", True)),
    )


_COMMON = """
You are part of a runtime-managed Lean proof tree. Work only on the declarations
and files assigned to your subtree. Do not change public theorem signatures and
do not introduce sorry, admit, or new axioms. Use local source search before
inventing lemmas. Keep the shared benchmark tree buildable: make small atomic
edits, run a focused Lean build after each proof batch, and revert your own
broken edit before continuing. Never run sforge-submit; only the root agent may
submit. Report completed declarations, touched files, exact build command, and
remaining errors to your parent with send(). Do not silently finish.
For source edits, use ws_replace_string/ws_multi_replace/ws_apply_patch with
absolute paths under /home/workspace/se-bmk-intern/combinatorial-games. Shell is
for search and builds only; Runtime rejects shell-based writes and out-of-scope
file edits.
""".strip()


def _lead(
    key: str,
    bio: str,
    scope: str,
    worker_scope: str,
    allowed_paths: tuple[str, ...],
) -> FixedAgentSpec:
    return FixedAgentSpec(
        key=key,
        bio=bio,
        task=f"""{_COMMON}

ROLE: domain lead and local integrator.
SCOPE: {scope}

Inventory the sorries and dependency clusters in your scope. Personally solve
the easiest non-overlapping declarations while preserving file ownership. A
runtime checkpoint will ask you to create a proof worker with NanoMA's spawn
tool for: {worker_scope}
When that child appears, coordinate through query/wait/send, inspect its edits,
run the relevant focused build, and send the root a structured handoff. Do not
spawn broad extra agents yourself; only execute checkpoint-authorized spawn
tool calls.
""",
        allowed_paths=allowed_paths,
    )


def _worker(
    key: str,
    bio: str,
    scope: str,
    allowed_paths: tuple[str, ...],
) -> FixedAgentSpec:
    return FixedAgentSpec(
        key=key,
        bio=bio,
        task=f"""{_COMMON}

ROLE: proof worker.
SCOPE: {scope}

Start with a precise declaration list and solve a small adjacent batch. Prefer
existing local lemmas, simp/rfl/exact, then short rw/calc proofs. Compile after
each batch and immediately send a structured handoff to your parent. If one
hard theorem blocks progress, leave the shared tree green and state the exact
goal and compiler error; the runtime may attach a narrow alternative explorer.
""",
        allowed_paths=allowed_paths,
    )


def combinatorial_games_plan() -> FixedOrchestrationPlan:
    leads = (
        _lead(
            "lead-dyadic",
            "Dyadic and order domain lead",
            "Dyadic coercions, casts, order lemmas, powers, and ArchimedeanClass bridges.",
            "coercion/cast/simp declarations such as coe_eq_*, coe_le_*, coeRingHom, and coe_half.",
            (
                "CombinatorialGames/Game/Player.lean",
                "CombinatorialGames/Game/Functor.lean",
            ),
        ),
        _lead(
            "lead-igame",
            "IGame and LGame domain lead",
            "IGame/LGame instances, order/equivalence lemmas, game operations, and Domineering.",
            "recursive IGame operations and equivalence/order proofs, excluding Surreal files.",
            (
                "CombinatorialGames/Game/Classes.lean",
                "CombinatorialGames/Game/Order.lean",
            ),
        ),
        _lead(
            "lead-surreal",
            "Surreal domain lead",
            "Surreal core, numeric interfaces, and easier Pow/Real bridge declarations.",
            "Surreal.Real declarations and their dyadic/rational bridge lemmas.",
            (
                "CombinatorialGames/Surreal/Pow.lean",
                "CombinatorialGames/Surreal/Ordinal.lean",
            ),
        ),
    )
    workers = (
        FixedCheckpoint(
            key="dyadic-worker",
            parent_key="lead-dyadic",
            agents=(_worker(
                "worker-dyadic-casts",
                "Dyadic cast proof worker",
                "Dyadic coe/cast/ring-hom and short order/simp declarations only.",
                ("CombinatorialGames/Mathlib/Dyadic.lean",),
            ),),
            min_parent_turns=2,
            min_parent_tool_calls=3,
            min_parent_age_seconds=180,
        ),
        FixedCheckpoint(
            key="igame-worker",
            parent_key="lead-igame",
            agents=(_worker(
                "worker-igame-recursive",
                "IGame recursive proof worker",
                "IGame/LGame recursive operations, equivalence, order, and nearby instances.",
                (
                    "CombinatorialGames/Game/IGame.lean",
                    "CombinatorialGames/Game/Specific/Domineering.lean",
                ),
            ),),
            min_parent_turns=2,
            min_parent_tool_calls=3,
            min_parent_age_seconds=180,
        ),
        FixedCheckpoint(
            key="surreal-worker",
            parent_key="lead-surreal",
            agents=(_worker(
                "worker-surreal-real",
                "Surreal Real proof worker",
                "Surreal.Real numeric, dyadic/rational, multiplication, and toIGame bridge declarations.",
                ("CombinatorialGames/Surreal/Real.lean",),
            ),),
            min_parent_turns=2,
            min_parent_tool_calls=3,
            min_parent_age_seconds=180,
        ),
    )
    alternatives = (
        FixedCheckpoint(
            key="igame-alternative",
            parent_key="worker-igame-recursive",
            mode="stalled",
            agents=(FixedAgentSpec(
                key="explorer-igame-alternative",
                bio="IGame alternative proof explorer",
                task=f"""{_COMMON}

ROLE: narrow alternative-route explorer. Read the parent worker's latest state
and choose exactly one blocked IGame theorem. Independently search nearby Lean
sources for a different proof route. Do not edit the same file concurrently;
return a candidate proof and required imports/lemmas to the parent with send().
""",
                read_only=True,
            ),),
            min_parent_turns=8,
            min_stall_turns=6,
            min_parent_age_seconds=900,
        ),
        FixedCheckpoint(
            key="surreal-alternative",
            parent_key="worker-surreal-real",
            mode="stalled",
            agents=(FixedAgentSpec(
                key="explorer-surreal-alternative",
                bio="Surreal alternative proof explorer",
                task=f"""{_COMMON}

ROLE: narrow alternative-route explorer. Read the parent worker's latest state
and choose exactly one blocked Surreal.Real or Surreal.Pow theorem. Search for
an independent proof route without editing the parent's active file. Send the
candidate proof, supporting lemmas, and expected compile command to the parent.
""",
                read_only=True,
            ),),
            min_parent_turns=8,
            min_stall_turns=6,
            min_parent_age_seconds=900,
        ),
    )
    final = FixedCheckpoint(
        key="final-verification",
        parent_key="root",
        mode="settled",
        depends_on=("dyadic-worker", "igame-worker", "surreal-worker"),
        agents=(FixedAgentSpec(
            key="final-verifier",
            bio="Lean final verification and regression agent",
            task=f"""{_COMMON}

ROLE: read-only final verifier. Inspect all current changes, identify newly
broken declarations or overlapping edits, run focused checks and then lake
build. Do not modify source. Send the root a final report with build status,
suspicious new bad declarations, and the exact files safe to submit.
""",
            read_only=True,
        ),),
        fallback_elapsed_seconds=4200,
        fallback_elapsed_fraction=0.60,
    )
    return FixedOrchestrationPlan(
        name="combinatorial_games",
        root_instructions="""
[Runtime fixed orchestration: combinatorial_games]
You are the root governor. Keep the benchmark submission tree green, maintain
the declaration inventory, integrate domain reports, and be the only agent that
runs sforge-submit. Runtime checkpoints will direct designated parents to create
three domain leads and their fixed descendants through NanoMA spawn tools.
Runtime itself never creates agents. Do not duplicate their owned files. Query children, score
meaningful green batches one at a time, and record score deltas. Two consecutive
no-gain submissions require switching to an uncovered cluster. Do not call done
until the final-verification checkpoint has completed or the runtime releases
the safety gate near the deadline.
""".strip(),
        checkpoints=(
            FixedCheckpoint(
                key="domain-leads",
                parent_key="root",
                agents=leads,
                min_parent_turns=2,
                min_parent_tool_calls=4,
                min_parent_age_seconds=120,
            ),
            *workers,
            *alternatives,
            final,
        ),
        max_live_agents=8,
        max_concurrent_llm=8,
        completion_checkpoint="final-verification",
        task_repo="/home/workspace/se-bmk-intern/combinatorial-games",
    )


def load_fixed_orchestration_plan(
    name: str,
    config_path: str | Path | None = None,
) -> FixedOrchestrationPlan | None:
    if config_path:
        return load_topology_orchestration_plan(config_path)
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"combinatorial_games", "combinatorial_games_formalization"}:
        return combinatorial_games_plan()
    return None
