"""Deterministic tests for runtime-owned final artifact delivery."""

import asyncio
import json
from pathlib import Path

import pytest

from nanoma import DeliveryContract, DeliveryTree, Runtime, RuntimeConfig
from nanoma.cost import UsageRecord
from nanoma.delivery import delivery_tree_digest
from nanoma.llm import LLMResponse, ToolCall, estimate_tokens
from nanoma.meta import (
    meta_deliver_to_parent,
    meta_get_task_context,
    meta_set_status,
    meta_submit,
)
from optimizations.todo_tools.todo_tools import meta_task_create


def _runtime(tmp_path: Path, *, contract: DeliveryContract | None = None) -> Runtime:
    return Runtime(
        config=RuntimeConfig(
            workspace_root=tmp_path / "workspace",
            delivery_contract=contract,
            log_dir=None,
        )
    )


def _contract(tmp_path: Path) -> DeliveryContract:
    return DeliveryContract(
        target_root=tmp_path / "task",
        trees=(
            DeliveryTree(
                target="output",
                candidates=("output", "Py2JS/output"),
                required=("test.mjs",),
            ),
        ),
    )


def test_delivery_contract_leaves_verification_properties_to_agents(tmp_path):
    runtime = _runtime(tmp_path, contract=_contract(tmp_path))
    for child in (False, True):
        instructions = runtime._delivery_contract_instructions(child=child)
        assert "Derive any verification strategy from the public task contract" in instructions
        assert "non_monotonic" not in instructions
        assert "repeated" not in instructions
        assert "boundary values" not in instructions


@pytest.mark.asyncio
async def test_root_parks_inside_done_until_exact_review_accepts(tmp_path):
    runtime = Runtime(
        config=RuntimeConfig(
            workspace_root=tmp_path / "workspace",
            final_candidate_review_enabled=True,
            final_candidate_review_require_executed_evidence=False,
            log_dir=None,
        )
    )
    root = runtime.create_agent("return the exact result")
    prior_child = runtime.create_agent("explore an alternative", parent=root.id, depth=1)
    prior_child.status = "done"
    reviewer_ids = []

    def _start_reviewer(reviewer):
        reviewer_ids.append(reviewer.id)

        async def _accept():
            await asyncio.sleep(0.05)
            runtime._record_candidate_delivery(
                reviewer,
                parent_id=root.id,
                answer="ACCEPT",
                evidence="the exact pending candidate satisfies the contract",
                method="independent exact-candidate review",
                source="deliver_to_parent",
            )
            reviewer.status = "done"

        reviewer._task = asyncio.create_task(_accept())

    runtime.start_agent = _start_reviewer
    result = await meta_set_status(
        {"status": "done", "result": "final answer"}, root, runtime
    )

    assert result["status"] == "done"
    assert root.status == "done"
    assert len(reviewer_ids) == 1
    assert root._turns == 0
    assert any(
        event["event"] == "final_candidate_review_parent_park_start"
        for event in runtime._events
    )
    assert any(
        event["event"] == "final_candidate_review_parent_park_end"
        and event["data"]["ready"] is True
        for event in runtime._events
    )


@pytest.mark.asyncio
async def test_changed_bytes_are_reviewed_serially_without_waking_root(tmp_path):
    contract = _contract(tmp_path)
    runtime = Runtime(
        config=RuntimeConfig(
            workspace_root=tmp_path / "workspace",
            delivery_contract=contract,
            final_candidate_review_enabled=True,
            final_candidate_review_require_executed_evidence=False,
            log_dir=None,
        )
    )
    root = runtime.create_agent("publish a runnable artifact")
    prior_child = runtime.create_agent("explore an alternative", parent=root.id, depth=1)
    prior_child.status = "done"
    candidate = root.workspace / "output"
    candidate.mkdir(parents=True)
    (candidate / "test.mjs").write_text("export const version = 1;\n", encoding="utf-8")
    reviewer_ids = []
    active_reviewers = 0
    max_active_reviewers = 0

    def _start_reviewer(reviewer):
        reviewer_ids.append(reviewer.id)
        review_number = len(reviewer_ids)

        async def _accept_snapshot():
            nonlocal active_reviewers, max_active_reviewers
            active_reviewers += 1
            max_active_reviewers = max(max_active_reviewers, active_reviewers)
            try:
                await asyncio.sleep(0.03)
                state = runtime._final_candidate_reviews[root.id]
                reviewed_digest = state["artifact_sha256"]
                if review_number == 1:
                    published = contract.target_root / "output/test.mjs"
                    published.write_text(
                        "export const version = 2;\n", encoding="utf-8"
                    )
                runtime._record_candidate_delivery(
                    reviewer,
                    parent_id=root.id,
                    answer="ACCEPT",
                    evidence=f"accepted frozen candidate {review_number}",
                    method="independent exact-candidate review",
                    source="deliver_to_parent",
                    reviewed_artifact_sha256=reviewed_digest,
                )
                reviewer.status = "done"
            finally:
                active_reviewers -= 1

        reviewer._task = asyncio.create_task(_accept_snapshot())

    runtime.start_agent = _start_reviewer
    result = await meta_set_status(
        {"status": "done", "result": "artifact ready"}, root, runtime
    )

    assert result["status"] == "done"
    assert len(reviewer_ids) == 2
    assert max_active_reviewers == 1
    assert root._turns == 0
    assert runtime._final_candidate_reviews[root.id]["round"] == 2
    accepted_events = [
        event for event in runtime._events
        if event["event"] == "final_candidate_review_accepted"
    ]
    assert len(accepted_events) == 1
    assert accepted_events[0]["data"]["round"] == 2


@pytest.mark.asyncio
async def test_revise_wakes_root_once_with_actionable_evidence(tmp_path):
    runtime = Runtime(
        config=RuntimeConfig(
            workspace_root=tmp_path / "workspace",
            final_candidate_review_enabled=True,
            final_candidate_review_require_executed_evidence=False,
            log_dir=None,
        )
    )
    root = runtime.create_agent("return the exact result")
    prior_child = runtime.create_agent("explore an alternative", parent=root.id, depth=1)
    prior_child.status = "done"

    def _start_reviewer(reviewer):
        async def _revise():
            await asyncio.sleep(0.03)
            runtime._record_candidate_delivery(
                reviewer,
                parent_id=root.id,
                answer="REVISE",
                evidence="the public return shape is wrong",
                method="independent exact-candidate review",
                source="deliver_to_parent",
            )
            reviewer.status = "done"

        reviewer._task = asyncio.create_task(_revise())

    runtime.start_agent = _start_reviewer
    result = await meta_set_status(
        {"status": "done", "result": "candidate"}, root, runtime
    )

    assert result["status"] == "running"
    assert result["review"]["reason"] == "revision_requested"
    assert result["review"]["evidence"] == "the public return shape is wrong"
    assert root.status == "running"
    assert root._turns == 0


@pytest.mark.asyncio
async def test_review_only_node_has_one_completed_execution_phase(tmp_path):
    runtime = Runtime(
        config=RuntimeConfig(
            workspace_root=tmp_path / "workspace",
            final_candidate_review_enabled=True,
            log_dir=None,
        )
    )
    root = runtime.create_agent("publish a candidate")
    reviewer = runtime.create_agent(
        "review the frozen candidate",
        parent=root.id,
        depth=1,
        review_only=True,
    )
    first = await runtime._execute_tool(
        ToolCall(id="check-1", name="shell", arguments={"command": "true"}),
        reviewer,
        runtime._all_tools(),
    )
    second = await runtime._execute_tool(
        ToolCall(id="check-2", name="shell", arguments={"command": "true"}),
        reviewer,
        runtime._all_tools(),
    )

    assert first["exit_code"] == 0
    assert second["error"] == (
        "the bounded final-review execution phase is already complete"
    )

    retry_reviewer = runtime.create_agent(
        "review another frozen candidate",
        parent=root.id,
        depth=1,
        review_only=True,
    )
    malformed = await runtime._execute_tool(
        ToolCall(id="bad-check", name="shell", arguments={"command": "("}),
        retry_reviewer,
        runtime._all_tools(),
    )
    recovered = await runtime._execute_tool(
        ToolCall(id="good-check", name="shell", arguments={"command": "true"}),
        retry_reviewer,
        runtime._all_tools(),
    )
    assert "syntax error" in malformed["stderr"].lower()
    assert recovered["exit_code"] == 0


@pytest.mark.asyncio
async def test_done_atomically_publishes_complete_alternative_tree(tmp_path):
    contract = _contract(tmp_path)
    runtime = _runtime(tmp_path, contract=contract)
    root = runtime.create_agent("write the final output")

    candidate = root.workspace / "Py2JS/output"
    (candidate / "nanoma/packages/example").mkdir(parents=True)
    (candidate / "test.mjs").write_text(
        "import './nanoma/packages/example/index.mjs';\n",
        encoding="utf-8",
    )
    (candidate / "nanoma/packages/example/index.mjs").write_text(
        "export const ready = true;\n",
        encoding="utf-8",
    )
    official = contract.target_root / "output"
    official.mkdir(parents=True)
    (official / "keep.txt").write_text("existing", encoding="utf-8")

    result = await meta_set_status(
        {"status": "done", "result": "complete"}, root, runtime
    )

    assert result["status"] == "done"
    assert result["delivery"]["ready"] is True
    assert root.status == "done"
    assert (official / "test.mjs").is_file()
    assert (official / "nanoma/packages/example/index.mjs").is_file()
    assert (official / "keep.txt").read_text(encoding="utf-8") == "existing"


@pytest.mark.asyncio
async def test_done_is_blocked_when_required_output_has_no_candidate(tmp_path):
    runtime = _runtime(tmp_path, contract=_contract(tmp_path))
    root = runtime.create_agent("write the final output")

    result = await meta_set_status(
        {"status": "done", "result": "claimed complete"}, root, runtime
    )

    assert "error" in result
    assert result["delivery"]["ready"] is False
    assert root.status == "running"
    assert root.result is None


@pytest.mark.asyncio
async def test_directory_submit_preserves_tree_and_publishes_contract(tmp_path):
    contract = _contract(tmp_path)
    runtime = _runtime(tmp_path, contract=contract)
    root = runtime.create_agent("write the final output")
    candidate = root.workspace / "Py2JS/output"
    (candidate / "nanoma/packages/example").mkdir(parents=True)
    (candidate / "test.mjs").write_text("console.log('ok');\n", encoding="utf-8")
    (candidate / "nanoma/packages/example/index.mjs").write_text(
        "export {};\n", encoding="utf-8"
    )

    result = await meta_submit(
        {"path": "Py2JS/output", "description": "complete output tree"},
        root,
        runtime,
    )

    assert result["kind"] == "directory"
    assert result["delivery"]["ready"] is True
    assert (runtime._tool_context.shared_dir / "output/test.mjs").is_file()
    assert (contract.target_root / "output/test.mjs").is_file()
    assert (
        contract.target_root / "output/nanoma/packages/example/index.mjs"
    ).is_file()


@pytest.mark.asyncio
async def test_plain_directory_submit_is_supported_without_contract(tmp_path):
    runtime = _runtime(tmp_path)
    root = runtime.create_agent("submit a directory")
    bundle = root.workspace / "bundle"
    (bundle / "nested").mkdir(parents=True)
    (bundle / "nested/result.txt").write_text("answer", encoding="utf-8")

    result = await meta_submit({"path": "bundle"}, root, runtime)

    assert result["kind"] == "directory"
    assert (runtime._tool_context.shared_dir / "bundle/nested/result.txt").is_file()


def test_incomplete_candidate_never_replaces_existing_target(tmp_path):
    contract = _contract(tmp_path)
    runtime = _runtime(tmp_path, contract=contract)
    root = runtime.create_agent("write the final output")
    official = contract.target_root / "output"
    official.mkdir(parents=True)
    (official / "keep.txt").write_text("unchanged", encoding="utf-8")
    candidate = root.workspace / "Py2JS/output"
    candidate.mkdir(parents=True)
    (candidate / "not-the-entry.mjs").write_text("export {};\n", encoding="utf-8")

    report = runtime.finalize_delivery(root, trigger="test")

    assert report["ready"] is False
    assert (official / "keep.txt").read_text(encoding="utf-8") == "unchanged"
    assert not (official / "not-the-entry.mjs").exists()


def test_workspace_root_itself_can_be_a_complete_candidate(tmp_path):
    contract = _contract(tmp_path)
    runtime = _runtime(tmp_path, contract=contract)
    root = runtime.create_agent("write the required entry directly in the workspace")
    (root.workspace / "test.mjs").write_text("export const ok = true;\n", encoding="utf-8")

    report = runtime.finalize_delivery(root, trigger="test")

    assert report["ready"] is True
    assert (contract.target_root / "output/test.mjs").read_text(encoding="utf-8") == (
        "export const ok = true;\n"
    )


@pytest.mark.asyncio
async def test_child_delivery_auto_discovers_validates_and_freezes_artifact(tmp_path):
    contract = _contract(tmp_path)
    runtime = _runtime(tmp_path, contract=contract)
    root = runtime.create_agent("assemble the best child artifact")
    child = runtime.create_agent("implement a complete candidate", parent=root.id, depth=1)
    source = child.workspace / "test.mjs"
    source.write_text("export const version = 1;\n", encoding="utf-8")

    delivered = await meta_deliver_to_parent(
        {
            "answer": "candidate implemented",
            "evidence": "the required entry is complete",
            "confidence": 0.9,
            "method": "direct implementation",
        },
        child,
        runtime,
    )

    assert delivered["artifact_sha256"]
    assert delivered["artifacts"][0]["validated"] is True
    snapshot = Path(delivered["artifacts"][0]["snapshot"])
    assert snapshot != child.workspace
    assert (snapshot / "test.mjs").read_text(encoding="utf-8") == (
        "export const version = 1;\n"
    )
    source.write_text("export const version = 2;\n", encoding="utf-8")
    assert (snapshot / "test.mjs").read_text(encoding="utf-8") == (
        "export const version = 1;\n"
    )
    record = runtime._candidate_deliveries[-1]
    assert record["artifact_validated"] is True
    assert record["artifact_sha256"] == delivered["artifact_sha256"]
    assert child.artifacts[0].absolute_path == snapshot


@pytest.mark.asyncio
async def test_frozen_child_checkpoint_can_be_replayed_without_root_overwrite(tmp_path):
    """Reproduce the historical "good child, wrong official entry" tail only.

    The model's implementation turns are intentionally absent.  The child has
    already delivered an immutable candidate, while the root still owns a
    different valid-looking tree.  Replaying the child checkpoint must publish
    those exact bytes, and later implicit finalizers must preserve that choice.
    """

    contract = _contract(tmp_path)
    runtime = _runtime(tmp_path, contract=contract)
    root = runtime.create_agent("select and publish the best frozen candidate")
    child = runtime.create_agent(
        "produce the implementation candidate",
        parent=root.id,
        depth=1,
    )
    (root.workspace / "output").mkdir(parents=True)
    (root.workspace / "output/test.mjs").write_text(
        "export const selected = 'wrong-root-candidate';\n",
        encoding="utf-8",
    )
    (child.workspace / "test.mjs").write_text(
        "export const selected = 'correct-child-checkpoint';\n",
        encoding="utf-8",
    )

    delivered = await meta_deliver_to_parent(
        {
            "answer": "candidate ready",
            "evidence": "entry was executed before handoff",
            "confidence": 0.95,
            "method": "independent implementation",
        },
        child,
        runtime,
    )
    checkpoint = Path(delivered["artifacts"][0]["snapshot"])
    checkpoint_digest = delivery_tree_digest(checkpoint)

    replay = await meta_submit(
        {
            "path": str(checkpoint),
            "description": "resume from the frozen pre-delivery checkpoint",
        },
        root,
        runtime,
    )

    official = contract.target_root / "output"
    assert replay["delivery"]["ready"] is True
    assert replay["delivery"]["published"][0]["source"] == str(checkpoint)
    assert delivery_tree_digest(official) == checkpoint_digest
    assert "correct-child-checkpoint" in (
        official / "test.mjs"
    ).read_text(encoding="utf-8")

    # These are the two implicit calls that historically selected another
    # valid-looking tree after the intended candidate had already been chosen.
    after_status = runtime.finalize_delivery(root, trigger="set_status")
    after_exit = runtime.finalize_delivery(root, trigger="runtime_exit")
    assert after_status["satisfied"][0]["source"] == str(official)
    assert after_exit["satisfied"][0]["source"] == str(official)
    assert delivery_tree_digest(official) == checkpoint_digest


@pytest.mark.asyncio
async def test_final_artifact_reviewer_must_deliver_self_derived_executed_checks(tmp_path):
    runtime = Runtime(
        config=RuntimeConfig(
            workspace_root=tmp_path / "workspace",
            delivery_contract=_contract(tmp_path),
            log_dir=None,
        )
    )
    root = runtime.create_agent("publish a runnable artifact")
    reviewer = runtime.create_agent(
        "review the frozen candidate",
        parent=root.id,
        depth=1,
        review_only=True,
    )
    setattr(reviewer, "_final_candidate_reviewer_for", root.id)
    runtime._final_candidate_reviews[root.id] = {
        "artifact_sha256": "frozen-digest",
        "artifact_snapshots": [{"snapshot": "/frozen/candidate"}],
    }

    def check(property_name: str, command: str, passed: bool = True) -> dict:
        return {
            "property": property_name,
            "rationale": f"{property_name} is material to the public interface",
            "command": command,
            "expected_observation": f"expected:{property_name}",
            "expected_basis": "public task contract",
            "actual_observation": f"actual:{property_name}",
            "passed": passed,
        }

    checks = [
        check("interface_shape", "check exported interface"),
        check("source_invariant", "check source-derived invariant", passed=False),
    ]
    for item in checks[:1]:
        runtime._emit(reviewer.id, "tool_call", {
            "tool": "shell",
            "args": {"command": item["command"]},
            "result": '{"exit_code": 0, "stdout": "checked", "stderr": ""}',
        })

    blocked = await meta_deliver_to_parent(
        {
            "answer": "REVISE",
            "evidence": "candidate needs more verification",
        },
        reviewer,
        runtime,
    )
    assert blocked["error"] == (
        "final reviewer verdict requires self-derived executed evidence"
    )
    assert reviewer.status == "running"
    assert runtime._candidate_deliveries == []

    unexecuted = await meta_deliver_to_parent(
        {
            "answer": "REVISE",
            "evidence": "source invariant exposes a semantic mismatch",
            "coverage_summary": (
                "Selected the exported interface and a source-derived invariant "
                "from the public contract."
            ),
            "verification_checks": checks,
        },
        reviewer,
        runtime,
    )
    assert unexecuted["error"] == (
        "final reviewer self-derived verification evidence is invalid"
    )
    assert unexecuted["invalid_checks"][0]["reason"] == (
        "command was not executed by this reviewer"
    )
    assert unexecuted["executed_commands"] == [checks[0]["command"]]

    runtime._emit(reviewer.id, "tool_call", {
        "tool": "shell",
        "args": {"command": checks[1]["command"]},
        "result": '{"exit_code": 0, "stdout": "mismatch", "stderr": ""}',
    })
    contradictory = await meta_deliver_to_parent(
        {
            "answer": "ACCEPT",
            "evidence": "incorrectly claims the candidate is ready",
            "coverage_summary": "Interface and semantic invariant checked.",
            "verification_checks": checks,
        },
        reviewer,
        runtime,
    )
    assert contradictory["error"] == (
        "final reviewer cannot ACCEPT failed self-derived checks"
    )
    assert contradictory["failed_properties"] == ["source_invariant"]
    assert runtime._candidate_deliveries == []

    delivered = await meta_deliver_to_parent(
        {
            "answer": "REVISE",
            "evidence": "source invariant exposes a semantic mismatch",
            "coverage_summary": "Interface and semantic invariant checked.",
            "verification_checks": checks,
        },
        reviewer,
        runtime,
    )
    assert delivered["delivered"] is True
    assert {item["property"] for item in delivered["verification_checks"]} == {
        "interface_shape",
        "source_invariant",
    }
    assert runtime._candidate_deliveries[-1]["verification_checks"] == (
        delivered["verification_checks"]
    )


@pytest.mark.asyncio
async def test_final_reviewer_cannot_mark_failed_command_as_passed(tmp_path):
    runtime = Runtime(
        config=RuntimeConfig(
            workspace_root=tmp_path / "workspace",
            delivery_contract=_contract(tmp_path),
            log_dir=None,
        )
    )
    root = runtime.create_agent("publish a candidate")
    reviewer = runtime.create_agent(
        "review the candidate", parent=root.id, depth=1, review_only=True
    )
    setattr(reviewer, "_final_candidate_reviewer_for", root.id)
    runtime._final_candidate_reviews[root.id] = {
        "artifact_sha256": "frozen-digest",
        "artifact_snapshots": [{"snapshot": "/frozen/candidate"}],
    }
    command = "run self-selected public check"
    runtime._emit(reviewer.id, "tool_call", {
        "tool": "shell",
        "args": {"command": command},
        "result": '{"exit_code": 1, "stdout": "MISMATCH", "stderr": ""}',
    })

    blocked = await meta_deliver_to_parent(
        {
            "answer": "ACCEPT",
            "coverage_summary": "Selected the core semantic invariant.",
            "verification_checks": [{
                "property": "core_semantics",
                "rationale": "The output semantics are the primary task requirement.",
                "command": command,
                "expected_observation": "public invariant holds",
                "expected_basis": "public task contract",
                "actual_observation": "claimed to hold",
                "passed": True,
            }],
        },
        reviewer,
        runtime,
    )

    assert blocked["error"] == (
        "final reviewer self-derived verification evidence is invalid"
    )
    assert blocked["invalid_checks"][0]["reason"] == (
        "passed check command did not complete successfully"
    )
    assert runtime._candidate_deliveries == []


def test_invalid_candidate_is_rejected_before_atomic_publication(tmp_path):
    def require_support_file(tree_root: Path) -> str | None:
        support = tree_root / "nanoma/packages/example/index.mjs"
        return None if support.is_file() else f"missing dependency: {support}"

    contract = DeliveryContract(
        target_root=tmp_path / "task",
        trees=(
            DeliveryTree(
                target="output",
                candidates=("output", "Py2JS/output"),
                required=("test.mjs",),
                validator=require_support_file,
            ),
        ),
    )
    runtime = _runtime(tmp_path, contract=contract)
    root = runtime.create_agent("write a complete output tree")
    candidate = root.workspace / "Py2JS/output"
    candidate.mkdir(parents=True)
    (candidate / "test.mjs").write_text(
        "import './nanoma/packages/example/index.mjs';\n",
        encoding="utf-8",
    )

    blocked = runtime.finalize_delivery(root, trigger="test")

    assert blocked["ready"] is False
    assert blocked["rejected"][0]["candidate"] == str(candidate)
    assert not (contract.target_root / "output/test.mjs").exists()

    support = candidate / "nanoma/packages/example/index.mjs"
    support.parent.mkdir(parents=True)
    support.write_text("export const ready = true;\n", encoding="utf-8")

    published = runtime.finalize_delivery(root, trigger="test")

    assert published["ready"] is True
    assert (contract.target_root / "output/test.mjs").is_file()
    assert (contract.target_root / "output/nanoma/packages/example/index.mjs").is_file()


def test_invalid_explicit_submit_is_not_hidden_by_an_existing_valid_target(tmp_path):
    contract = _contract(tmp_path)
    runtime = _runtime(tmp_path, contract=contract)
    root = runtime.create_agent("replace the published artifact")
    official = contract.target_root / "output"
    official.mkdir(parents=True)
    (official / "test.mjs").write_text("export const version = 1;\n", encoding="utf-8")
    invalid = root.workspace / "replacement"
    invalid.mkdir(parents=True)
    (invalid / "wrong-name.mjs").write_text("export const version = 2;\n", encoding="utf-8")

    report = runtime.finalize_delivery(
        root,
        trigger="submit",
        explicit_paths=[invalid],
    )

    assert report["ready"] is False
    assert "previously published target was preserved" in report["missing"][0]["reason"]
    assert (official / "test.mjs").read_text(encoding="utf-8") == (
        "export const version = 1;\n"
    )


@pytest.mark.asyncio
async def test_runtime_exit_publishes_when_turn_limit_bypasses_done(tmp_path):
    async def write_then_expire(messages, model, tools=None, **kwargs):
        return LLMResponse(
            tool_calls=[ToolCall(
                id="write",
                name="ws_create_file",
                arguments={
                    "path": "Py2JS/output/test.mjs",
                    "content": "console.log('ready');\n",
                    "overwrite": True,
                },
            )],
            usage=UsageRecord(input_tokens=10, output_tokens=5, model=model),
        )

    contract = _contract(tmp_path)
    runtime = Runtime(
        config=RuntimeConfig(
            workspace_root=tmp_path / "workspace",
            delivery_contract=contract,
            max_turns=1,
            log_dir=None,
        ),
        llm_call=write_then_expire,
    )

    result = await runtime.run("write the entry")

    assert result == "[Max turns reached]"
    assert (contract.target_root / "output/test.mjs").is_file()
    assert runtime.stats()["delivery_contract"]["ready"] is True
    assert any(
        item["trigger"] == "runtime_exit"
        for item in runtime._delivery_contract_history
    )


@pytest.mark.parametrize("unsafe", ("../output", "/absolute/output"))
def test_delivery_tree_rejects_unsafe_paths(unsafe):
    with pytest.raises(ValueError):
        DeliveryTree(target=unsafe, candidates=("output",), required=("test.mjs",))


@pytest.mark.asyncio
async def test_child_capsule_is_bounded_and_full_task_is_paged(tmp_path):
    runtime = Runtime(
        config=RuntimeConfig(
            workspace_root=tmp_path / "workspace",
            task_capsule_max_tokens=180,
            task_capsule_inline_root_max_tokens=90,
            task_context_chunk_max_tokens=120,
            final_candidate_review_enabled=False,
            log_dir=None,
        )
    )
    root_task = (
        "Solve the numerical routine.\n"
        + "Background material that should not be copied into every child.\n" * 400
        + "def evolve(state: np.ndarray, steps: int) -> np.ndarray:\n"
        + "Return shape must be exactly (steps, state.size); indexing is one-based.\n"
    )
    root = runtime.create_agent(root_task)
    child = runtime.create_agent("Audit the output contract.", parent=root.id, depth=1)

    system = str(child.history[0]["content"])
    capsule = system.split("## Task Capsule\n", 1)[1].split("\n## Tool Philosophy", 1)[0].strip()
    assert estimate_tokens(capsule) <= 180
    assert "def evolve" in capsule
    assert root_task not in system

    first = await meta_get_task_context(
        {"section": "root", "offset": 0, "max_tokens": 10_000}, child, runtime
    )
    assert first["chunk_tokens"] <= 120
    assert first["has_more"] is True
    second = await meta_get_task_context(
        {"section": "root", "offset": first["next_offset"]}, child, runtime
    )
    assert second["offset"] == first["next_offset"]
    assert first["content"] + second["content"] == root_task[:second["next_offset"]]


@pytest.mark.asyncio
async def test_tagged_task_payload_does_not_leak_root_protocol_to_children(tmp_path):
    runtime = Runtime(
        config=RuntimeConfig(
            workspace_root=tmp_path / "workspace",
            final_candidate_review_enabled=False,
            log_dir=None,
        )
    )
    official = (
        "NEXT STEP\n"
        "def lanczos(A, b, m):\n"
        "Return Q with shape exactly (M, m + 1).\n"
    )
    wrapped = (
        "Solve this step.\n\n"
        "Protocol:\n"
        "- At your first planning boundary call task_create.\n"
        "- Every child node may call task_create.\n\n"
        f"<official_scicode_prompt>\n{official}</official_scicode_prompt>"
    )
    root = runtime.create_agent(wrapped)
    child = runtime.create_agent("Audit the Lanczos return shape.", parent=root.id, depth=1)

    system = str(child.history[0]["content"])
    assert "def lanczos" in system
    assert "At your first planning boundary" not in system
    assert "Every child node may call task_create" not in system

    root_context = await meta_get_task_context({"section": "root"}, child, runtime)
    parent_context = await meta_get_task_context({"section": "parent"}, child, runtime)
    contract = await meta_get_task_context({"section": "contract"}, child, runtime)
    assert root_context["content"] == official.strip()
    assert parent_context["content"] == official.strip()
    assert "task_create" not in contract["content"]


def test_task_contract_excerpt_fills_single_line_semantic_gap(tmp_path):
    runtime = Runtime(
        config=RuntimeConfig(
            workspace_root=tmp_path / "workspace",
            task_capsule_inline_root_max_tokens=256,
            log_dir=None,
        )
    )
    source = "\n".join([
        "Source program:",
        "print(nx.clustering(G, args.c))",
        "print(nx.clustering(G))",
        "print(nx.average_clustering(G))",
        "print(nx.transitivity(G))",
        "print(nx.triangles(G, args.c))",
        *(f"background prose without interface signal {i}" for i in range(300)),
    ])

    excerpt = runtime._task_contract_excerpt(source)

    assert "print(nx.average_clustering(G))" in excerpt


@pytest.mark.asyncio
async def test_spawn_judge_uses_custom_policy_and_filters_lineage_duplicate(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("NANOMA_SPAWN_TODOLIST_JUDGE", "1")
    monkeypatch.setenv(
        "NANOMA_SPAWN_JUDGE_INSTRUCTION",
        "CUSTOM_POLICY_SENTINEL: spawn only for a genuinely independent scientific subproblem.",
    )
    runtime = Runtime(
        config=RuntimeConfig(
            workspace_root=tmp_path / "workspace",
            max_agents=20,
            max_depth=20,
            default_model="worker-model",
            final_candidate_review_enabled=False,
            log_dir=None,
        )
    )
    runtime.start_agent = lambda _agent: None
    root = runtime.create_agent("Solve the official numerical task.")
    assignment = (
        "Derive the Lanczos recurrence, normalization, alpha beta indexing, "
        "orthogonality, output shape, and breakdown behavior."
    )
    child = runtime.create_agent(assignment, parent=root.id, depth=1)

    class _Response:
        content = json.dumps({
            "spawn": True,
            "reasoning": "repeat the same audit",
            "subagents": [{
                "subject": "Repeat Lanczos derivation",
                "role": "verifier",
                "task": assignment,
            }],
        })

    seen: dict[str, str] = {}

    async def _fake_llm(messages, model, tools=None, **kwargs):
        seen["system"] = messages[0]["content"]
        seen["user"] = messages[-1]["content"]
        return _Response()

    runtime.llm_call = _fake_llm
    before = len(runtime.agents)
    spawned = await runtime._spawn_judge_at_plan(child, "audit Lanczos")

    assert spawned is False
    assert len(runtime.agents) == before
    assert "CUSTOM_POLICY_SENTINEL" in seen["system"]
    assert "Assignments already represented" in seen["user"]
    assert any(
        event["event"] == "spawn_judge_duplicate_filtered"
        for event in runtime._events
    )


@pytest.mark.asyncio
async def test_spawn_judge_allows_reexpand_after_children_finish(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("NANOMA_SPAWN_TODOLIST_JUDGE", "1")
    runtime = Runtime(
        config=RuntimeConfig(
            workspace_root=tmp_path / "workspace",
            max_agents=20,
            max_depth=20,
            default_model="worker-model",
            final_candidate_review_enabled=False,
            log_dir=None,
        )
    )
    runtime.start_agent = lambda _agent: None
    root = runtime.create_agent("Solve a multi-part numerical task.")

    class _Response:
        content = json.dumps({
            "spawn": True,
            "reasoning": "independent parts",
            "subagents": [{
                "subject": "Derive boundary conditions",
                "role": "solver",
                "task": "Derive boundary conditions and indexing for the numerical recurrence.",
            }],
        })

    calls = 0

    async def _fake_llm(messages, model, tools=None, **kwargs):
        nonlocal calls
        calls += 1
        return _Response()

    runtime.llm_call = _fake_llm
    assert await runtime._spawn_judge_at_plan(root, "split the recurrence") is True
    for child_id in root.children:
        runtime.agents[child_id].status = "done"

    # Second wave: children are done, so the in-flight short-circuit is off.
    # A later planning moment may fan out again — that is how hill-climbing
    # tasks pick up a new approach after the first wave lands.
    assert await runtime._spawn_judge_at_plan(root, "split it again") is True
    assert calls == 2


@pytest.mark.asyncio
async def test_spawn_judge_allows_child_to_expand_a_distinct_subproblem(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("NANOMA_SPAWN_TODOLIST_JUDGE", "1")
    runtime = Runtime(
        config=RuntimeConfig(
            workspace_root=tmp_path / "workspace",
            max_agents=20,
            max_depth=20,
            default_model="worker-model",
            final_candidate_review_enabled=False,
            log_dir=None,
        )
    )
    runtime.start_agent = lambda _agent: None
    root = runtime.create_agent("Solve the official numerical task.")
    child = runtime.create_agent(
        "Implement the core search algorithm and report the best permutation.",
        parent=root.id,
        depth=1,
    )

    class _Response:
        content = json.dumps({
            "spawn": True,
            "reasoning": "local search can run beside the constructive heuristic",
            "subagents": [{
                "subject": "Iterated local search",
                "role": "explorer",
                "task": (
                    "Run iterated local search with swap and reverse moves on the "
                    "current best permutation and return only the improved sequence."
                ),
            }],
        })

    async def _fake_llm(messages, model, tools=None, **kwargs):
        return _Response()

    runtime.llm_call = _fake_llm
    before = len(runtime.agents)
    assert await runtime._spawn_judge_at_plan(child, "refine in parallel") is True
    assert len(runtime.agents) == before + 1
    grandchild = next(
        a for a in runtime.agents.values() if a.parent == child.id
    )
    assert grandchild.depth >= 1


@pytest.mark.asyncio
async def test_root_completion_reviews_the_exact_candidate(tmp_path):
    runtime = Runtime(
        config=RuntimeConfig(
            workspace_root=tmp_path / "workspace",
            final_candidate_review_enabled=True,
            log_dir=None,
        )
    )
    root = runtime.create_agent("Return code with the exact required shape.")
    prior_child = runtime.create_agent("derive a candidate", parent=root.id, depth=1)
    prior_child.status = "done"
    runtime.start_agent = lambda _agent: None

    candidate = "def solve(x):\n    return x[:, None]"
    first = await meta_set_status(
        {"status": "done", "result": candidate}, root, runtime
    )
    assert first["review"]["reason"] == "review_started"
    assert root.status == "running"

    reviewer_id = first["review"]["reviewer_id"]
    reviewer = runtime.agents[reviewer_id]
    assert runtime._is_review_only(reviewer)
    assert "bounded finish gate" in reviewer.task
    assert "one bounded shell call" in reviewer.task
    assert "## Task Capsule" not in str(reviewer.history[0]["content"])
    assert "task_create and all spawn paths are disabled" in str(
        reviewer.history[0]["content"]
    )
    blocked_spawn = await runtime._invoke_meta_spawn(
        {"task": "delegate this review"}, reviewer
    )
    blocked_plan = await meta_task_create(
        {"subject": "delegate this review"}, reviewer, runtime
    )
    assert "review-only" in blocked_spawn["error"]
    assert "review-only" in blocked_plan["error"]
    exact = await meta_get_task_context(
        {"section": "final_candidate", "offset": 0}, reviewer, runtime
    )
    assert exact["content"] == candidate

    runtime._record_candidate_delivery(
        reviewer,
        parent_id=root.id,
        answer="ACCEPT",
        evidence="Signature and (n, 1) output shape match the root contract.",
        confidence=0.9,
        method="exact final candidate review",
        source="deliver_to_parent",
    )
    reviewer.status = "done"
    second = await meta_set_status(
        {"status": "done", "result": candidate}, root, runtime
    )
    assert second["status"] == "done"
    assert root.result == candidate
    assert any(event["event"] == "final_candidate_review_accepted" for event in runtime._events)


@pytest.mark.asyncio
async def test_final_review_is_bound_to_an_immutable_artifact_digest(tmp_path):
    contract = _contract(tmp_path)
    runtime = Runtime(
        config=RuntimeConfig(
            workspace_root=tmp_path / "workspace",
            delivery_contract=contract,
            final_candidate_review_enabled=True,
            log_dir=None,
        )
    )
    root = runtime.create_agent("publish and review the exact runnable artifact")
    prior_child = runtime.create_agent("produce independent evidence", parent=root.id, depth=1)
    (prior_child.workspace / "test.mjs").write_text(
        "export const reviewed = 'alternative';\n", encoding="utf-8"
    )
    child_delivery = await meta_deliver_to_parent(
        {
            "answer": "alternative artifact",
            "evidence": "independent implementation",
            "confidence": 0.8,
            "method": "independent candidate",
        },
        prior_child,
        runtime,
    )
    assert child_delivery["artifact_sha256"]
    runtime.start_agent = lambda _agent: None
    candidate = root.workspace / "output"
    candidate.mkdir(parents=True)
    (candidate / "test.mjs").write_text("export const reviewed = true;\n", encoding="utf-8")

    first = await meta_set_status(
        {"status": "done", "result": "artifact complete"}, root, runtime
    )
    assert first["review"]["reason"] == "review_started"
    artifact_sha256 = first["review"]["artifact_sha256"]
    assert artifact_sha256
    reviewer = runtime.agents[first["review"]["reviewer_id"]]
    exact = await meta_get_task_context(
        {"section": "final_candidate", "offset": 0}, reviewer, runtime
    )
    payload = json.loads(exact["content"])
    assert payload["artifact_sha256"] == artifact_sha256
    assert payload["validated_child_alternatives"][0]["artifact_sha256"] == (
        child_delivery["artifact_sha256"]
    )
    snapshot = Path(payload["artifacts"][0]["snapshot"])
    assert snapshot.is_dir()
    review_command = f"test -f {snapshot / 'test.mjs'}"
    runtime._emit(reviewer.id, "tool_call", {
        "tool": "shell",
        "args": {"command": review_command},
        "result": '{"exit_code": 0, "stdout": "", "stderr": ""}',
    })

    verdict = await meta_deliver_to_parent(
        {
            "answer": "ACCEPT",
            "evidence": "inspected the frozen entry and its exact exported value",
            "confidence": 0.95,
            "method": "immutable artifact review",
            "coverage_summary": (
                "Derived the required entry-file property from the delivery contract "
                "and checked it on the exact frozen snapshot."
            ),
            "verification_checks": [{
                "property": "required_entry_exists",
                "rationale": "The public delivery contract requires test.mjs.",
                "command": "the single bounded verification command above",
                "expected_observation": "test.mjs exists in the frozen artifact",
                "expected_basis": "public delivery contract",
                "actual_observation": "test -f exited 0",
                "passed": True,
            }],
        },
        reviewer,
        runtime,
    )
    assert verdict["delivered"] is True
    assert runtime._candidate_deliveries[-1]["reviewed_artifact_sha256"] == artifact_sha256
    assert runtime._candidate_deliveries[-1]["verification_checks"][0]["command"] == (
        review_command
    )

    agent_count = len(runtime.agents)
    second = await meta_set_status(
        {"status": "done", "result": "artifact complete with reviewer approval"},
        root,
        runtime,
    )
    assert second["status"] == "done"
    assert root.status == "done"
    assert len(runtime.agents) == agent_count


@pytest.mark.asyncio
async def test_external_worktree_uses_generic_immutable_final_review(tmp_path):
    task_root = tmp_path / "external-task"
    task_root.mkdir()
    (task_root / "public_contract.txt").write_text(
        "Produce a finite JSON estimate.\n", encoding="utf-8"
    )
    (task_root / "answer.json").write_text(
        '{"estimate": 4.0}\n', encoding="utf-8"
    )
    runtime = Runtime(
        config=RuntimeConfig(
            workspace_root=tmp_path / "runtime-workspace",
            final_candidate_review_snapshot_roots=(task_root,),
            final_candidate_review_enabled=True,
            log_dir=None,
        )
    )
    root = runtime.create_agent("Solve the public task in its external worktree.")
    runtime.start_agent = lambda _agent: None

    pending = await meta_set_status(
        {"status": "done", "result": "answer.json is ready"}, root, runtime
    )

    assert pending["review"]["reason"] == "review_started"
    assert pending["review"]["artifact_sha256"]
    reviewer = runtime.agents[pending["review"]["reviewer_id"]]
    context = await meta_get_task_context(
        {"section": "final_candidate", "offset": 0}, reviewer, runtime
    )
    payload = json.loads(context["content"])
    snapshot = Path(payload["artifacts"][0]["snapshot"])
    assert (snapshot / "answer.json").read_text(encoding="utf-8") == (
        '{"estimate": 4.0}\n'
    )
    assert payload["artifacts"][0]["target"] == "review-root-00"

    # A later mutable-worktree edit cannot change what the reviewer saw.
    (task_root / "answer.json").write_text(
        '{"estimate": 999.0}\n', encoding="utf-8"
    )
    assert (snapshot / "answer.json").read_text(encoding="utf-8") == (
        '{"estimate": 4.0}\n'
    )
    review_command = f"python -m json.tool {snapshot / 'answer.json'}"
    runtime._emit(reviewer.id, "tool_call", {
        "tool": "shell",
        "args": {"command": review_command},
        "result": '{"exit_code": 0, "stdout": "{\\\"estimate\\\": 4.0}", "stderr": ""}',
    })
    delivered = await meta_deliver_to_parent(
        {
            "answer": "ACCEPT",
            "evidence": "The frozen JSON is parseable and contains the public output.",
            "confidence": 0.9,
            "method": "public artifact verification",
            "coverage_summary": "Checked the public JSON interface on the frozen worktree.",
            "verification_checks": [{
                "property": "json_output_is_parseable",
                "rationale": "The public contract requires a JSON result.",
                "command": review_command,
                "expected_observation": "The parser exits successfully.",
                "expected_basis": "public task contract",
                "actual_observation": "The parser exited 0.",
                "passed": True,
            }],
        },
        reviewer,
        runtime,
    )
    assert delivered["delivered"] is True

    # This review-only integration never publishes or rolls back the external
    # worktree; the benchmark harness remains its sole owner.  Restore the
    # reviewed bytes ourselves before completing this deterministic test.
    assert (task_root / "answer.json").read_text(encoding="utf-8") == (
        '{"estimate": 999.0}\n'
    )
    (task_root / "answer.json").write_text(
        '{"estimate": 4.0}\n', encoding="utf-8"
    )
    accepted = await meta_set_status(
        {"status": "done", "result": "answer.json is ready"}, root, runtime
    )
    assert accepted["status"] == "done"
    assert (task_root / "answer.json").read_text(encoding="utf-8") == (
        '{"estimate": 4.0}\n'
    )


@pytest.mark.asyncio
async def test_every_revised_artifact_gets_a_fresh_review_by_default(tmp_path):
    runtime = Runtime(
        config=RuntimeConfig(
            workspace_root=tmp_path / "workspace",
            delivery_contract=_contract(tmp_path),
            final_candidate_review_enabled=True,
            final_candidate_review_require_executed_evidence=False,
            log_dir=None,
        )
    )
    root = runtime.create_agent("publish a runnable artifact")
    runtime.create_agent("produce an independent alternative", parent=root.id, depth=1)
    runtime.start_agent = lambda _agent: None
    candidate = root.workspace / "output"
    candidate.mkdir(parents=True)

    for revision in range(3):
        (candidate / "test.mjs").write_text(
            f"export const revision = {revision};\n", encoding="utf-8"
        )
        submitted = await meta_submit({"path": "output"}, root, runtime)
        assert submitted["delivery"]["ready"] is True
        pending = await meta_set_status(
            {"status": "done", "result": f"artifact revision {revision}"},
            root,
            runtime,
        )
        assert pending["review"]["reason"] == "review_started"
        assert pending["review"]["round"] == revision + 1
        reviewer = runtime.agents[pending["review"]["reviewer_id"]]
        delivered = await meta_deliver_to_parent(
            {
                "answer": "REVISE",
                "evidence": f"revision {revision} still has a public defect",
                "confidence": 0.9,
                "method": "independent artifact check",
            },
            reviewer,
            runtime,
        )
        assert delivered["delivered"] is True

    assert runtime._final_candidate_reviews[root.id]["round"] == 3


@pytest.mark.asyncio
async def test_formal_revise_cannot_be_bypassed_with_root_override(tmp_path):
    runtime = Runtime(
        config=RuntimeConfig(
            workspace_root=tmp_path / "workspace",
            delivery_contract=_contract(tmp_path),
            final_candidate_review_enabled=True,
            final_candidate_review_require_executed_evidence=False,
            log_dir=None,
        )
    )
    root = runtime.create_agent("publish a runnable artifact")
    runtime.create_agent("produce an independent alternative", parent=root.id, depth=1)
    runtime.start_agent = lambda _agent: None
    candidate = root.workspace / "output"
    candidate.mkdir(parents=True)
    (candidate / "test.mjs").write_text(
        "export const fittedToSamples = true;\n", encoding="utf-8"
    )

    pending = await meta_set_status(
        {"status": "done", "result": "candidate complete"}, root, runtime
    )
    reviewer = runtime.agents[pending["review"]["reviewer_id"]]
    await meta_deliver_to_parent(
        {
            "answer": "REVISE",
            "evidence": "literal lookup table is keyed to observed test inputs",
            "confidence": 0.99,
            "method": "independent artifact review",
        },
        reviewer,
        runtime,
    )

    bypass = await meta_set_status(
        {
            "status": "done",
            "result": "candidate complete",
            "review_override_reason": "I disagree with the reviewer",
        },
        root,
        runtime,
    )

    assert bypass["status"] == "running"
    assert bypass["review"]["reason"] == "formal_review_cannot_be_overridden"
    assert bypass["review"]["verdict_label"] == "REVISE"
    assert root.status == "running"


def test_repozero_validator_rejects_process_bridge_and_external_packages(tmp_path):
    from benchmarks.repozero.run_nanoma_repozero_hard_small import (
        validate_py2js_delivery_tree,
    )

    entry = tmp_path / "test.mjs"
    entry.write_text(
        "import { spawnSync } from 'node:child_process';\n"
        "export const answer = spawnSync('python3', ['solver.py']);\n",
        encoding="utf-8",
    )
    rejected_bridge = validate_py2js_delivery_tree(tmp_path, "test.mjs")
    assert rejected_bridge is not None
    assert "process spawning is forbidden" in rejected_bridge

    entry.write_text(
        "import helper from 'unlisted-package';\nexport default helper;\n",
        encoding="utf-8",
    )
    rejected_package = validate_py2js_delivery_tree(tmp_path, "test.mjs")
    assert rejected_package is not None
    assert "external package import is forbidden" in rejected_package
