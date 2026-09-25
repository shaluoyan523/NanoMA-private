"""Unit tests for the experiment ledger.

The keep-best ratchet in `merge_submit` is driven entirely by the agent's own
`verify` check, so a run whose check never passes the submission path keeps no
record of what it measured. One run failed its check on all five merges and then
narrowed the scored config to fit a local shell timeout, deleting the parameters
behind the best result the judge had already recorded for it.

Nothing extra has to be measured to prevent that: the runtime already parses
every judge verdict and already fingerprints the state each verdict applies to.
It kept one boolean from that pair. These tests are about keeping the pair, and
about refusing to spend a submission on a state that is worse than one on record.

Run:  python3 -m optimizations.experiment_ledger.test_experiment_ledger
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

# What a judge verdict looks like coming back through the submit tool: the
# harness prints it for the agent to read, and the runtime parses that print.
VERDICT = """\
========================================
  {round_id} Results
========================================
  Valid:       {valid}
  Score:       {score}
  Pass rate:   {pass_pct}%
  Passed:      {passed}/72
"""


def _mk_runtime(tmp: Path):
    """Runtime wired to temp dirs, with a submit tool that records its calls."""
    from nanoma.core import Runtime, RuntimeConfig

    workspace_root = tmp / "task_cwd" / ".nanoma-task-work"
    submit_path = tmp / "task_cwd"
    submit_path.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    submits: list[str] = []

    async def fake_submit(args, agent, runtime):
        submits.append(agent.id)
        return {"submitted": True}

    config = RuntimeConfig(
        max_agents=32, max_depth=4, default_model="m", allowed_models=["m"],
        workspace_root=workspace_root, workspace_extra_roots=[submit_path],
        extra_tools={"submit": {"handler": fake_submit, "is_meta": True, "schema": {}}},
    )
    rt = Runtime(config=config)
    rt.start_agent = lambda child: None
    return rt, submit_path, submits


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _enable():
    os.environ["NANOMA_MERGE_SUBMIT_PATH"] = "1"


def _disable():
    os.environ.pop("NANOMA_MERGE_SUBMIT_PATH", None)
    os.environ.pop("NANOMA_SUBMIT_REQUIRE_MEASURED", None)
    os.environ.pop("NANOMA_OFFICIAL_LOWER_IS_BETTER", None)


def _verdict(rt, agent, score, *, valid=True, pass_rate=1.0, round_id="agent-1"):
    """Feed a judge verdict in the way a real one arrives: a submit tool result."""
    text = VERDICT.format(
        score=score, pass_pct=round(pass_rate * 100, 1), passed=round(pass_rate * 72),
        valid="yes" if valid else "no", round_id=round_id,
    )
    return _run(
        rt._calibrate_from_tool_result(agent, "submit", {}, {"submission": text})
    )


def _experiments(rt, agent, **args):
    from optimizations.experiment_ledger import meta_experiments
    return _run(meta_experiments(args, agent, rt))


# ── recording what the run has measured ──────────────────────────────────────

def test_a_verdict_is_recorded_against_the_state_it_scored():
    """The runtime computed both halves of this pair already and kept neither."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "M: 16\nef: 60\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)

        _verdict(rt, agent, 4068.0)

        entries = rt._ledger_entries()
        assert len(entries) == 1, entries
        assert entries[0]["metric"] == 4068.0, entries
        assert entries[0]["source"] == "official", entries
        assert entries[0]["state"], entries
        # and it names the state that was actually on disk at the time
        live = rt._ledger_digest(rt._merge_scope_signature(submit_path))
        assert entries[0]["state"] == live, (entries, live)
    _disable()
    print("ok: a judge verdict is recorded against the state it scored")


def test_a_verdict_names_the_state_that_was_sent_not_the_one_that_replaced_it():
    """Observed 2026-07-31: an official 0.0 and an official 4468.0 landed on the
    same digest 31 seconds apart, and the 4468 state was snapshotted as somebody
    else's contents.

    The submit call blocks for the minutes the judge takes, and other agents keep
    editing the shared submit path while it does. Reading the workspace when the
    verdict returns therefore files the measurement against whatever arrived in
    the meantime — and every promise the ledger makes rests on those pairs being
    true.
    """
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        agent = rt.create_agent(task="root", parent=None, depth=0)

        _write(submit_path, "config.yml", "name: custom\nM: 16\n")
        sent = rt._ledger_digest(rt._merge_scope_signature(submit_path))

        # what a blocking submit looks like from the ledger's side: the state is
        # read as it is sent, then somebody else changes it before the verdict.
        rt._note_state_being_submitted(agent, "submit", {})
        _write(submit_path, "config.yml", "name: custom-ivf-flat\nM: 32\n")
        replaced = rt._ledger_digest(rt._merge_scope_signature(submit_path))
        assert sent != replaced, "the edit has to be visible, or this proves nothing"

        _verdict(rt, agent, 4468.0)

        entry = rt._ledger_entries()[-1]
        assert entry["metric"] == 4468.0
        assert entry["state"] == sent, (
            "the verdict belongs to the state that earned it"
        )
        assert entry["state"] != replaced
    _disable()
    print("ok: a verdict names the state that was sent, not the one that replaced it")


def test_two_verdicts_landing_together_stay_apart():
    """The symptom that exposed it: contradictory measurements under one digest,
    which makes the best state, the snapshot and the proxy comparison meaningless."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        first = rt.create_agent(task="root", parent=None, depth=0)
        second = rt.create_agent(task="other", parent=None, depth=0)

        _write(submit_path, "config.yml", "name: custom\n")
        good = rt._ledger_digest(rt._merge_scope_signature(submit_path))
        rt._note_state_being_submitted(first, "submit", {})

        _write(submit_path, "config.yml", "name: custom-ivf-flat\n")
        broken = rt._ledger_digest(rt._merge_scope_signature(submit_path))
        rt._note_state_being_submitted(second, "submit", {})

        # both verdicts come back while the workspace holds the broken state
        _verdict(rt, first, 4468.0, pass_rate=0.74, round_id="agent-1")
        _verdict(rt, second, 0.0, pass_rate=0.0, round_id="agent-2")

        by_state = {e["state"]: e["metric"] for e in rt._ledger_entries()}
        assert by_state[good] == 4468.0
        assert by_state[broken] == 0.0
        assert rt._ledger_best() and rt._ledger_best()["state"] == good, (
            "and the state worth returning to is the one that scored"
        )
    _disable()
    print("ok: two verdicts landing together stay apart")


def test_a_verdict_with_no_recorded_send_still_gets_filed():
    """A verdict the runtime did not see leave — a resumed run, a submission made
    outside the tool path — is worth more against the live scope than dropped."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        agent = rt.create_agent(task="root", parent=None, depth=0)
        _write(submit_path, "config.yml", "name: custom\n")

        _verdict(rt, agent, 3200.0)  # no _note_state_being_submitted beforehand

        entry = rt._ledger_entries()[-1]
        assert entry["metric"] == 3200.0
        assert entry["state"] == rt._ledger_digest(rt._merge_scope_signature(submit_path))
    _disable()
    print("ok: a verdict with no recorded send is still filed against the live scope")


def test_the_record_survives_in_a_file_not_in_memory():
    """A run that loses its process must not lose what it measured."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "M: 16\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)
        _verdict(rt, agent, 3760.0)

        path = rt._ledger_path()
        assert path.is_file(), path
        line = json.loads(path.read_text().splitlines()[0])
        assert line["metric"] == 3760.0, line

        # a second runtime over the same workspace reads the same history
        rt2, _, _ = _mk_runtime(Path(d))
        assert rt2._ledger_best()["metric"] == 3760.0, rt2._ledger_entries()
    _disable()
    print("ok: the record is a file, so a new runtime reads the same history")


def test_a_truncated_final_line_does_not_hide_the_rest():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "M: 16\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)
        _verdict(rt, agent, 100.0)
        with rt._ledger_path().open("a") as handle:
            handle.write('{"metric": 999, "sta')  # killed mid-write

        assert len(rt._ledger_entries()) == 1
        assert rt._ledger_best()["metric"] == 100.0
    _disable()
    print("ok: a torn final line does not hide the entries before it")


def test_the_best_state_is_kept_so_it_can_be_returned_to():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "M: 16\nef: 60\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)
        _verdict(rt, agent, 4068.0)

        best = rt._ledger_best()
        kept = rt._ledger_snapshot_dir(best["state"])
        assert kept.is_dir(), kept
        assert (kept / "config.yml").read_text() == "M: 16\nef: 60\n"
    _disable()
    print("ok: the best-measured state is snapshotted, not just named")


def test_an_invalid_verdict_is_recorded_but_is_not_somewhere_to_go_back_to():
    """A rejected submission still says something true about the state.

    It does not say the state is worth returning to, so it is on the record but
    never the best. This is the six-zeros case: the algorithm had been renamed,
    the judge reported "Nothing to run", and the score field was still present.
    """
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "name: custom\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)
        _verdict(rt, agent, 3000.0, round_id="agent-1")

        _write(submit_path, "config.yml", "name: custom-ivf\n")
        _verdict(rt, agent, 5000.0, valid=False, pass_rate=0.0, round_id="agent-2")

        assert len(rt._ledger_entries()) == 2
        best = rt._ledger_best()
        assert best["metric"] == 3000.0, best
        assert best["round"] == "agent-1", best
    _disable()
    print("ok: an invalid verdict is recorded but never becomes the best")


def test_lower_is_better_is_honoured():
    _enable()
    os.environ["NANOMA_OFFICIAL_LOWER_IS_BETTER"] = "1"
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "a.txt", "one\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)
        _verdict(rt, agent, 40.0, round_id="agent-1")
        _write(submit_path, "a.txt", "two\n")
        _verdict(rt, agent, 10.0, round_id="agent-2")

        assert rt._ledger_best()["metric"] == 10.0, rt._ledger_entries()
    _disable()
    print("ok: a minimised score picks the smallest as best")


def test_nothing_is_recorded_without_a_submission_path():
    """No target means no state to attach a measurement to."""
    _disable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "M: 16\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)
        _verdict(rt, agent, 4068.0)

        assert rt._ledger_entries() == []
    print("ok: with no merge target there is nothing to record against")


# ── refusing to ship a state worse than one on record ────────────────────────

def test_a_submission_is_refused_on_an_unmeasured_state_when_a_better_one_is_known():
    """The run that shrank its config to fit a local timeout, in miniature.

    The narrowed state has never been measured, so it cannot be claimed to be
    better, and a measured state that scored well is sitting on record.
    """
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, submits = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "M: 16\nef: 60\nM: 32\nef: 500\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)
        _verdict(rt, agent, 4068.0, round_id="auto-21")

        # narrowed so a local run finishes inside the agent's shell timeout
        _write(submit_path, "config.yml", "M: 32\nef: 500\n")

        blocked = _run(rt._submit_preflight(agent))
        assert blocked is not None, "an unmeasured narrowing was allowed through"
        assert blocked["blocked"] == "worse_than_measured", blocked
        assert "4068" in blocked["reason"], blocked
        assert "never been measured" in blocked["reason"], blocked
        assert submits == [], submits
    _disable()
    print("ok: a submission on an unmeasured state is refused when a better one is known")


def test_a_measured_worse_state_is_refused_too():
    """Refusing needs a scatter to measure the gap against.

    The wide config is measured twice, which is what lets the run see how much
    re-measuring moves a number; 1996 against 4068 then clears that. Without the
    repeat the run declines to judge, which the next test covers.
    """
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "wide\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)
        _verdict(rt, agent, 4068.0, round_id="auto-21")
        _verdict(rt, agent, 3900.0, round_id="auto-22")
        _write(submit_path, "config.yml", "narrow\n")
        _verdict(rt, agent, 1996.0, round_id="auto-23")

        blocked = _run(rt._submit_preflight(agent))
        assert blocked is not None and blocked["blocked"] == "worse_than_measured", blocked
        assert "1996" in blocked["reason"], blocked
    _disable()
    print("ok: a state measured worse than the record is refused as well")


def test_the_best_state_itself_submits():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "wide\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)
        _verdict(rt, agent, 4068.0)

        assert _run(rt._submit_preflight(agent)) is None
    _disable()
    print("ok: the best-measured state submits without objection")


def test_an_improvement_submits():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "wide\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)
        _verdict(rt, agent, 4068.0, round_id="auto-21")
        _write(submit_path, "config.yml", "wider\n")
        _verdict(rt, agent, 4927.0, round_id="agent-2")

        assert _run(rt._submit_preflight(agent)) is None
    _disable()
    print("ok: a state that beat the record submits")


def test_the_first_submission_is_never_refused():
    """Without a measurement there is nothing to be worse than.

    A gate that blocks before any evidence exists turns straight into a zero.
    """
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "M: 16\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)

        assert _run(rt._submit_preflight(agent)) is None
    _disable()
    print("ok: the first submission is never refused")


def test_the_gate_can_be_turned_off():
    _enable()
    os.environ["NANOMA_SUBMIT_REQUIRE_MEASURED"] = "0"
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "wide\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)
        _verdict(rt, agent, 4068.0)
        _write(submit_path, "config.yml", "narrow\n")

        assert _run(rt._submit_preflight(agent)) is None
    _disable()
    print("ok: the gate can be turned off")


def test_a_run_without_a_merge_target_keeps_its_submit_path_unwrapped():
    """The gate must not appear where it could never have evidence."""
    _disable()
    with tempfile.TemporaryDirectory() as d:
        rt, _, _ = _mk_runtime(Path(d))
        assert rt._submit_requires_measured_state() is False
        assert rt._submit_gate_enabled() is False
    print("ok: without a merge target the gate stays out of the way")


# ── a proxy metric has to agree with the authority ───────────────────────────
# A local check's number is a stand-in for the score. One that ranks states
# differently from the judge is worse than having none: the ratchet then keeps and
# protects whichever state the proxy prefers. One run's check reported 18138 for a
# state on a scale where the judge's best was 3454, and nothing could notice.

def _pair(rt, agent, submit_path, content, local, official, round_id):
    """Measure one state on both channels: the local check, then the judge."""
    _write(submit_path, "config.yml", content)
    sig = rt._merge_scope_signature(rt._merge_target())
    rt._ledger_note("verify", local, signature=sig, agent_id=agent.id, counts=False)
    _verdict(rt, agent, official, round_id=round_id)


def test_a_proxy_that_ranks_like_the_judge_keeps_its_authority():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        agent = rt.create_agent(task="root", parent=None, depth=0)
        _pair(rt, agent, submit_path, "a\n", local=100.0, official=1000.0, round_id="agent-1")
        _pair(rt, agent, submit_path, "b\n", local=200.0, official=2000.0, round_id="agent-2")

        assert rt._verify_metric_tracks_authority() is True
        rt._verify_metric_seen["chk"] = [1.0, 2.0]
        assert rt._verify_metric_is_proven({"command": "chk"}) is True
    _disable()
    print("ok: a proxy that ranks like the judge keeps its ratchet authority")


def test_a_proxy_that_ranks_against_the_judge_loses_it():
    """The state of the 2026-07-30 run: local said better, the judge said worse."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        agent = rt.create_agent(task="root", parent=None, depth=0)
        _pair(rt, agent, submit_path, "a\n", local=100.0, official=2000.0, round_id="agent-1")
        _pair(rt, agent, submit_path, "b\n", local=200.0, official=1000.0, round_id="agent-2")

        assert rt._verify_metric_tracks_authority() is False

        seen: list[str] = []
        original = rt._emit
        rt._emit = lambda a, kind, data=None: (seen.append(kind), original(a, kind, data))[1]
        rt._verify_metric_seen["chk"] = [1.0, 2.0]
        assert rt._verify_metric_is_proven({"command": "chk"}) is False, (
            "a metric that disagrees with the judge must not be a ratchet floor"
        )
        assert "verify_metric_withdrawn" in seen, seen
    _disable()
    print("ok: a proxy that ranks against the judge loses its ratchet authority")


def test_one_paired_state_is_not_enough_to_judge_the_proxy():
    """Agreement needs two states to compare; one pair ranks nothing."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        agent = rt.create_agent(task="root", parent=None, depth=0)
        _pair(rt, agent, submit_path, "a\n", local=100.0, official=2000.0, round_id="agent-1")

        assert rt._verify_metric_tracks_authority() is None
        rt._verify_metric_seen["chk"] = [1.0, 2.0]
        assert rt._verify_metric_is_proven({"command": "chk"}) is True, (
            "no evidence against the proxy is not evidence against it"
        )
    _disable()
    print("ok: one paired state is not enough to disqualify a proxy")


def test_a_scale_difference_alone_is_not_disagreement():
    """The two channels need not be on one scale, only to agree on which is better.

    18138 local against 3454 official is a 5x scale gap, and on its own says
    nothing: what matters is whether the proxy puts the same state first.
    """
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        agent = rt.create_agent(task="root", parent=None, depth=0)
        _pair(rt, agent, submit_path, "a\n", local=15556.0, official=2564.0, round_id="agent-1")
        _pair(rt, agent, submit_path, "b\n", local=18138.0, official=3454.0, round_id="agent-2")

        assert rt._verify_metric_tracks_authority() is True
    _disable()
    print("ok: a scale difference alone is not disagreement")


# ── submissions the runtime makes, not the model ─────────────────────────────
# The gates are installed on the tool table handed to the model each turn, so
# every submission that does not come from a model tool call used to skip them.
# The EdgeBench adapter's closing submission of each iteration is one of those,
# and it is the one the round is scored on.

def test_a_verdict_where_nothing_passed_is_not_somewhere_to_go_back_to():
    """Observed on the 2026-07-30 run, and it cost that run two submissions.

    Its only official verdict was a broken state scoring 0 with a pass rate of 0.
    That was snapshotted as the best state on record, and then used to refuse two
    later submissions of a state the agent's own check had just verified at 18000
    — on the grounds that they were "unmeasured while a better one is known".
    Nothing is worse than a state where nothing passed.
    """
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, submits = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "broken\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)
        _verdict(rt, agent, 0.0, pass_rate=0.0, round_id="agent-1")

        # Still on the record — that it scored nothing is worth knowing.
        assert [e["metric"] for e in rt._ledger_entries()] == [0.0]
        assert rt._ledger_best() is None, "but it is not a state to fall back to"

        _write(submit_path, "config.yml", "verified but unsubmitted\n")
        assert _run(rt._submit_preflight(agent)) is None
        _run(rt.submit_official())
        assert submits == [agent.id], "the submission must not be refused"
    _disable()
    print("ok: a verdict where nothing passed never becomes the best")


def test_a_runtime_submission_answers_to_the_gate():
    """The bypass: this submission is the decisive one and was ungated."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, submits = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "wide\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)
        _verdict(rt, agent, 4068.0, round_id="auto-21")
        _verdict(rt, agent, 3900.0, round_id="auto-22")
        _write(submit_path, "config.yml", "narrow\n")
        _verdict(rt, agent, 1996.0, round_id="auto-23")
        submits.clear()

        out = _run(rt.submit_official(reason="iteration final submit"))
        assert out is not None and out.get("blocked") == "worse_than_measured", out
        assert submits == [], "the refused state must not reach the judge"
    _disable()
    print("ok: a runtime-initiated submission answers to the gate")


def test_a_runtime_submission_records_its_verdict():
    """The reason the ledger was missing its most recent entry."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "wide\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)

        async def judging_submit(args, ag, runtime):
            return {"submission": VERDICT.format(
                score=3074.0, pass_pct=100.0, passed=72, valid="yes",
                round_id="agent-2",
            )}

        rt.config.extra_tools["submit"]["handler"] = judging_submit
        _run(rt.submit_official(reason="iteration final submit"))

        entries = rt._ledger_entries()
        assert [e["round"] for e in entries] == ["agent-2"], entries
        assert entries[0]["metric"] == 3074.0, entries
    _disable()
    print("ok: a runtime-initiated submission's verdict reaches the ledger")


def test_a_runtime_submission_finds_the_root_agent_itself():
    """The caller is outside the runtime and has no agent handle to pass."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, submits = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "wide\n")
        root = rt.create_agent(task="root", parent=None, depth=0)
        rt.create_agent(task="child", parent=root.id, depth=1)

        _run(rt.submit_official())
        assert submits == [root.id], submits
    _disable()
    print("ok: a runtime-initiated submission attributes itself to the root")


def test_a_run_with_no_agents_declines_so_the_caller_can_submit():
    """A gate with nothing to judge on must not turn into a zero.

    Declining rather than submitting with a stand-in agent: handlers are written
    against a real one, and the caller already falls back to submitting directly.
    """
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, submits = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "wide\n")

        assert _run(rt.submit_official()) is None
        assert submits == [], submits
    _disable()
    print("ok: a run that produced no agents leaves the submission to its caller")


# ── telling a difference from measurement scatter ────────────────────────────
# Numbers below are the ones this task actually produced: on the 2026-07-29 run,
# 17 rounds submitted a byte-identical archive and scored 2564 to 4160 QPS.

def test_the_run_estimates_its_own_measurement_noise():
    """Nothing can be assumed about the scatter, so it is measured."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "wide\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)

        assert rt._measurement_noise("official") is None, "no repeat yet"
        for i, score in enumerate((2564.0, 3406.0, 4160.0)):
            _verdict(rt, agent, score, round_id=f"auto-{i + 1}")

        noise = rt._measurement_noise("official")
        assert noise is not None and 0.15 < noise < 0.30, noise
        # The margin is in the metric's own units and is sized at the magnitude
        # being compared, so it grows with the value it has to protect.
        near = rt._noise_margin("official", at=4160.0)
        far = rt._noise_margin("official", at=100.0)
        assert near > far > 0, (near, far)
    _disable()
    print("ok: the run estimates the scatter of its own measurements")


def test_a_gap_inside_the_scatter_is_not_a_regression():
    """The same archive scoring 2564 once and 4160 another time is one state.

    Treating that as a regression is what made the ratchet a coin flip.
    """
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "wide\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)
        for i, score in enumerate((2564.0, 3406.0, 4160.0)):
            _verdict(rt, agent, score, round_id=f"auto-{i + 1}")

        # 3074 against a best of 4160: a 26% gap, inside a 2-sigma margin here.
        assert rt._metric_is_worse([3074.0], [4160.0], "official") is False
    _disable()
    print("ok: a gap inside the scatter is not called a regression")


def test_a_gap_outside_the_scatter_still_is_one():
    """The mechanism must not become inert — a real collapse still counts."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "wide\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)
        for i, score in enumerate((2564.0, 3406.0, 4160.0)):
            _verdict(rt, agent, score, round_id=f"auto-{i + 1}")

        # The renamed-algorithm zeros, which are a real regression, not scatter.
        assert rt._metric_is_worse([0.0], [4160.0], "official") is True
    _disable()
    print("ok: a gap outside the scatter is still a regression")


def test_without_a_repeat_the_run_declines_to_judge():
    """One measurement per state says nothing about how much re-measuring moves.

    Returning False here rather than True is deliberate: the cost of refusing a
    good state is the whole score, and the cost of allowing a worse one is one
    submission.
    """
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "wide\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)
        _verdict(rt, agent, 4160.0, round_id="auto-1")

        assert rt._metric_is_worse([3074.0], [4160.0], "official") is None
    _disable()
    print("ok: with no repeat measurement the run declines to judge")


def _spread(vals: list[float]):
    """A runtime whose verify channel has measured one state repeatedly."""
    import tempfile as _tf
    rt, submit_path, _ = _mk_runtime(Path(_tf.mkdtemp()))
    _write(submit_path, "config.yml", "wide\n")
    sig = rt._merge_scope_signature(rt._merge_target())
    for v in vals:
        rt._ledger_note("verify", v, signature=sig, agent_id="a", counts=False)
    return rt


def test_a_deterministic_check_gets_exact_comparison_back():
    """Not every task measures something noisy.

    A check that counts passing tests returns the same number every time, so its
    scatter is zero and the margin is zero — the comparison is exact again. This
    has to hold or the change would loosen every deterministic gate in exchange
    for fixing one noisy benchmark.
    """
    _enable()
    rt = _spread([47.0, 47.0])
    assert rt._noise_margin("verify", at=47.0) == 0.0
    assert rt._metric_is_worse([46.0], [47.0], "verify") is True
    _disable()
    print("ok: a deterministic check is compared exactly, as before")


def test_a_metric_that_crosses_zero_does_not_break_the_margin():
    """A ratio is meaningless when the mean is smaller than the scatter.

    Repeats of -0.01 and 0.02 have a 424% coefficient of variation, purely from
    dividing by nearly nothing. Taken at face value it sets a margin no
    regression could ever clear, which would silently disable the rollback for
    any task whose metric is a delta or a signed error.
    """
    _enable()
    rt = _spread([-0.01, 0.02])
    assert rt._metric_is_worse([0.001], [0.005], "verify") is False, "inside scatter"
    assert rt._metric_is_worse([-5.0], [0.005], "verify") is True, "a real collapse"
    _disable()
    print("ok: a metric that crosses zero still gets a usable margin")


def test_a_minimised_metric_is_judged_in_the_right_direction():
    """Latency, cost, error rate: worse is larger."""
    _enable()
    os.environ["NANOMA_OFFICIAL_LOWER_IS_BETTER"] = "1"
    rt = _spread([100.0, 120.0])
    assert rt._metric_is_worse([108.0], [100.0], "verify") is False, "inside scatter"
    assert rt._metric_is_worse([900.0], [100.0], "verify") is True, "a real regression"
    _disable()
    print("ok: a minimised metric is judged in the right direction")


def test_the_two_channels_treat_an_unknown_scatter_differently():
    """Asymmetric on purpose, because the two mistakes do not cost the same.

    At the submission gate an unknown scatter allows the submission: refusing
    wrongly costs the round's score, and submissions are finite. At the merge
    rollback it reverts: each promotion verifies a different state, so a margin
    there may never become estimable, and being inert would mean keeping every
    broken merge — while a wrong rollback loses one delivery that `prev` holds.
    """
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "wide\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)
        _verdict(rt, agent, 4160.0, round_id="auto-1")
        _write(submit_path, "config.yml", "narrow\n")
        _verdict(rt, agent, 3074.0, round_id="auto-2")

        assert rt._metric_is_worse([3074.0], [4160.0], "official") is None
        # Submission gate: None lets it through.
        assert _run(rt._submit_preflight(agent)) is None
        # Merge rollback reads the same None as "revert", tested against the real
        # promotion path in optimizations/merge_submit/test_merge_submit.py.
    _disable()
    print("ok: an unknown scatter is read differently by gate and by rollback")


def test_channels_are_not_pooled():
    """A local check's number and an official score are not one scale."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "wide\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)
        _verdict(rt, agent, 4160.0, round_id="auto-1")
        sig = rt._merge_scope_signature(rt._merge_target())
        rt._ledger_note("verify", 0.42, signature=sig, agent_id=agent.id, counts=False)
        rt._ledger_note("verify", 0.44, signature=sig, agent_id=agent.id, counts=False)

        digest = rt._ledger_digest(sig)
        assert rt._ledger_state_metrics(digest, "official") == [4160.0]
        assert rt._ledger_state_metrics(digest, "verify") == [0.42, 0.44]
        assert rt._measurement_noise("official") is None, "one official reading"
        assert rt._measurement_noise("verify") is not None
        assert rt._ledger_best()["metric"] == 4160.0, "verify never becomes best"
    _disable()
    print("ok: measurement channels are kept apart")


def test_a_submission_inside_the_scatter_is_allowed():
    """End to end: the gate stops refusing what it cannot actually distinguish."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "wide\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)
        for i, score in enumerate((2564.0, 3406.0, 4160.0)):
            _verdict(rt, agent, score, round_id=f"auto-{i + 1}")
        # A different state, measured once, inside the scatter of the best.
        _write(submit_path, "config.yml", "narrow\n")
        _verdict(rt, agent, 3074.0, round_id="agent-1")

        assert _run(rt._submit_preflight(agent)) is None
    _disable()
    print("ok: a submission inside the scatter is allowed through")


# ── reading the record back, and returning to a state ────────────────────────

def test_the_agent_can_read_what_it_measured():
    """The run that never learned its own configuration's score."""
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "wide\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)
        _verdict(rt, agent, 4068.0, round_id="auto-21")
        _write(submit_path, "config.yml", "narrow\n")

        out = _experiments(rt, agent)
        assert out["best"]["metric"] == 4068.0, out
        assert out["best"]["round"] == "auto-21", out
        assert out["best"]["is_current"] is False, out
        assert out["best"]["restorable"] is True, out
        assert out["current_state_measured"] is False, out
        assert out["measurements"][0]["metric"] == 4068.0, out
    _disable()
    print("ok: the agent can read the scores its own run produced")


def test_an_empty_record_says_so_rather_than_looking_broken():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "M: 16\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)

        out = _experiments(rt, agent)
        assert out["measurements"] == [], out
        assert "nothing measured yet" in out["note"], out
    _disable()
    print("ok: an empty record explains itself")


def test_restoring_a_state_puts_it_back_and_unblocks_the_submission():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "M: 16\nef: 60\n")
        _write(submit_path, "notes.md", "keep\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)
        _verdict(rt, agent, 4068.0, round_id="auto-21")
        best_state = rt._ledger_best()["state"]

        _write(submit_path, "config.yml", "M: 32\n")
        assert _run(rt._submit_preflight(agent)) is not None

        out = _experiments(rt, agent, restore=best_state)
        assert out["restored"] is True, out
        assert (submit_path / "config.yml").read_text() == "M: 16\nef: 60\n"
        assert _run(rt._submit_preflight(agent)) is None
    _disable()
    print("ok: restoring a measured state puts it back and the submission proceeds")


def test_restoring_an_unknown_state_says_which_ones_exist():
    _enable()
    with tempfile.TemporaryDirectory() as d:
        rt, submit_path, _ = _mk_runtime(Path(d))
        _write(submit_path, "config.yml", "wide\n")
        agent = rt.create_agent(task="root", parent=None, depth=0)
        _verdict(rt, agent, 4068.0)

        out = _experiments(rt, agent, restore="deadbeef")
        assert out["restored"] is False, out
        assert out["states"], out
    _disable()
    print("ok: restoring an unknown state reports the ones on record")


def main() -> None:
    tests = [
        test_a_verdict_is_recorded_against_the_state_it_scored,
        test_the_record_survives_in_a_file_not_in_memory,
        test_a_truncated_final_line_does_not_hide_the_rest,
        test_the_best_state_is_kept_so_it_can_be_returned_to,
        test_an_invalid_verdict_is_recorded_but_is_not_somewhere_to_go_back_to,
        test_lower_is_better_is_honoured,
        test_nothing_is_recorded_without_a_submission_path,
        test_a_submission_is_refused_on_an_unmeasured_state_when_a_better_one_is_known,
        test_a_measured_worse_state_is_refused_too,
        test_the_best_state_itself_submits,
        test_an_improvement_submits,
        test_the_first_submission_is_never_refused,
        test_the_gate_can_be_turned_off,
        test_a_run_without_a_merge_target_keeps_its_submit_path_unwrapped,
        test_a_proxy_that_ranks_like_the_judge_keeps_its_authority,
        test_a_proxy_that_ranks_against_the_judge_loses_it,
        test_one_paired_state_is_not_enough_to_judge_the_proxy,
        test_a_scale_difference_alone_is_not_disagreement,
        test_a_verdict_where_nothing_passed_is_not_somewhere_to_go_back_to,
        test_a_runtime_submission_answers_to_the_gate,
        test_a_runtime_submission_records_its_verdict,
        test_a_runtime_submission_finds_the_root_agent_itself,
        test_a_run_with_no_agents_declines_so_the_caller_can_submit,
        test_the_run_estimates_its_own_measurement_noise,
        test_a_gap_inside_the_scatter_is_not_a_regression,
        test_a_gap_outside_the_scatter_still_is_one,
        test_without_a_repeat_the_run_declines_to_judge,
        test_a_deterministic_check_gets_exact_comparison_back,
        test_a_metric_that_crosses_zero_does_not_break_the_margin,
        test_a_minimised_metric_is_judged_in_the_right_direction,
        test_the_two_channels_treat_an_unknown_scatter_differently,
        test_channels_are_not_pooled,
        test_a_submission_inside_the_scatter_is_allowed,
        test_the_agent_can_read_what_it_measured,
        test_an_empty_record_says_so_rather_than_looking_broken,
        test_restoring_a_state_puts_it_back_and_unblocks_the_submission,
        test_restoring_an_unknown_state_says_which_ones_exist,
        test_a_verdict_names_the_state_that_was_sent_not_the_one_that_replaced_it,
        test_two_verdicts_landing_together_stay_apart,
        test_a_verdict_with_no_recorded_send_still_gets_filed,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} experiment ledger tests passed.")


if __name__ == "__main__":
    main()
