"""Deterministic tests for runtime-owned final artifact delivery."""

from pathlib import Path

import pytest

from nanoma import DeliveryContract, DeliveryTree, Runtime, RuntimeConfig
from nanoma.cost import UsageRecord
from nanoma.llm import LLMResponse, ToolCall, estimate_tokens
from nanoma.meta import meta_get_task_context, meta_set_status, meta_submit


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
