# NanoMA Fast Full Batch Fanout Task

你是 root coordinator。跑一次能收敛的全量高并发工作流：24 个 scout、12 个 reviewer、6 个 integrator、4 个 redteam，总计 46 个子 agent，加 root 为 47 个 agent。

目标是验证工作流结构、batch 拉起、query 主动查询、memory tags、可视化查询边，不是完整源码审计。不要使用 `shell`。

## Root Rules

1. 立即写 `scout_batch.json` 并调用 `batch`，不要预读源码。
2. 每一波都必须一次性用 `batch` 批量拉起：24 scout、12 reviewer、6 integrator、4 redteam。
3. 每波 `batch` 后检查返回结果数量；成功后 `wait(mode="all")`。
4. 每波 `wait` 后必须 `query(filter={"group_id":"...","status":"done"}, tags=["phase:..."], include_memory=true)`。
5. 不要要求子 agent 给 root 发完成消息；父子关系只表示 lineage。
6. 所有产物写 `shared/`。

## Child Completion Protocol

每个子 agent 最后只调用：

```json
compact(summary="...", tags=[...], files=[...], experience="...", stop_after=true, result="<ROLE>-XX done")
```

不要在 compact 后再调用 `set_status`。不要 `send`。

## Evidence Rules

每个 scout 只做快速采样：

1. `query(filter={"group_id":"scout_wave"}, tags=["phase:scout"], include_memory=true, limit=6)`。
2. 最多一次 `grep`。
3. 最多一次 `file_read(path="...", offset=<grep附近行号或1>, limit=30)`。
4. 立即写短 summary 和 notes。

如果 grep 没结果，直接基于任务给定的源码区域和 query 快照写“未找到证据”的风险说明，不要继续搜索。

## Scout Wave

一次性批量拉起 24 个 scout，`group_id="scout_wave"`。每个 scout 写：

- `shared/scouts/scout_XX_summary.md`：60-90 words
- `shared/scouts/scout_XX_notes.md`：90-140 words

每个 scout 的任务必须包含：

- role: `scout`
- create_type: `peer_agent`
- relationship: `peer`
- group_id: `scout_wave`
- workflow_prior: `research_loop`
- tags: `["batch","phase:scout","scout:XX","topic:<slug>"]`
- 必须先 query 同波 agent。
- 最多一次 grep、最多一次 file_read。
- 最后 submit summary，再 compact stop。

主题表：

1. scout 01: runtime architecture, grep `class Runtime`, read `shared/source/nanoma/core.py`
2. scout 02: agent lifecycle, grep `async def _agent_loop`, read `shared/source/nanoma/core.py`
3. scout 03: spawn behavior, grep `async def meta_spawn`, read `shared/source/nanoma/meta.py`
4. scout 04: batch behavior, grep `async def meta_batch`, read `shared/source/nanoma/meta.py`
5. scout 05: wait behavior, grep `async def meta_wait`, read `shared/source/nanoma/meta.py`
6. scout 06: query behavior, grep `async def meta_query`, read `shared/source/nanoma/meta.py`
7. scout 07: compact memory, grep `async def meta_compact`, read `shared/source/nanoma/meta.py`
8. scout 08: submit artifacts, grep `async def meta_submit`, read `shared/source/nanoma/meta.py`
9. scout 09: shared paths, grep `_resolve_workspace_path`, read `shared/source/nanoma/tools.py`
10. scout 10: tool boundaries, grep `enabled_work_tools`, read `shared/source/nanoma/core.py`
11. scout 11: sandbox behavior, grep `class SandboxSession`, read `shared/source/nanoma/sandbox.py`
12. scout 12: LLM calls, grep `async def call_openai`, read `shared/source/nanoma/llm.py`
13. scout 13: budget ledger, grep `class CostLedger`, read `shared/source/nanoma/cost.py`
14. scout 14: context compression, grep `def _compress`, read `shared/source/nanoma/core.py`
15. scout 15: scheduling, grep `class Scheduler`, read `shared/source/nanoma/scheduler.py`
16. scout 16: error handling, grep `except Exception`, read `shared/source/nanoma/core.py`
17. scout 17: workflow prior, grep `workflow_prior`, read `shared/source/nanoma/meta.py`
18. scout 18: group metadata, grep `group_id`, read `shared/source/nanoma/core.py`
19. scout 19: observability, grep `def _emit`, read `shared/source/nanoma/core.py`
20. scout 20: runner CLI, grep `def parse_args`, read `shared/source/examples/high_concurrency_runner.py`
21. scout 21: viewer logging, grep `query`, read `shared/source/nanoma/viewer.html`
22. scout 22: tests, grep `test_meta_batch`, read `shared/source/tests/test_nanoma.py`
23. scout 23: security, grep `Access denied`, read `shared/source/nanoma/tools.py`
24. scout 24: improvements, grep `TODO|FIXME|error`, read `shared/source/nanoma`

After scout wait, root query must see 24 done.

## Reviewer Wave

一次性批量拉起 12 个 reviewer，`group_id="reviewer_wave"`。每个 reviewer 负责两个 scout：

1. reviewer 01: scouts 01-02
2. reviewer 02: scouts 03-04
3. reviewer 03: scouts 05-06
4. reviewer 04: scouts 07-08
5. reviewer 05: scouts 09-10
6. reviewer 06: scouts 11-12
7. reviewer 07: scouts 13-14
8. reviewer 08: scouts 15-16
9. reviewer 09: scouts 17-18
10. reviewer 10: scouts 19-20
11. reviewer 11: scouts 21-22
12. reviewer 12: scouts 23-24

Reviewer steps:

1. `query(tags=["phase:scout"], include_memory=true)` and use responsible `scout:XX` tags.
2. Read only the two assigned summary files.
3. Write `shared/reviews/reviewer_XX.md` in 100-150 words with risks and next action.
4. submit review and compact stop with tags `phase:reviewer`, `reviewer:XX`, assigned scout tags.

## Integrator Wave

一次性批量拉起 6 个 integrator，`group_id="integrator_wave"`。

1. chapter 01: scouts 01-04, reviewers 01-02
2. chapter 02: scouts 05-08, reviewers 03-04
3. chapter 03: scouts 09-12, reviewers 05-06
4. chapter 04: scouts 13-16, reviewers 07-08
5. chapter 05: scouts 17-20, reviewers 09-10
6. chapter 06: scouts 21-24, reviewers 11-12

Integrator steps:

1. Query `phase:scout` and `phase:reviewer`.
2. Read only assigned reviewer files and up to four assigned scout summaries.
3. Write `shared/integration/chapter_XX.md` in 140-220 words with implementation recommendations.
4. submit chapter and compact stop with tags `phase:integrator`, `chapter:XX`.

## Redteam Wave

一次性批量拉起 4 个 redteam，`group_id="redteam_wave"`。

1. redteam 01: correctness and source grounding
2. redteam 02: runtime reliability and completion risk
3. redteam 03: security and permission boundaries
4. redteam 04: workflow scalability and cost control

Redteam steps:

1. Query `phase:integrator` and `phase:reviewer`.
2. Read all six chapter files only.
3. Write `shared/redteam/redteam_XX.md` in 140-220 words with prioritized risks.
4. submit report and compact stop with tags `phase:redteam`, `redteam:XX`.

## Final

Root must query all four phases, then read:

- 24 scout summaries
- 12 reviewer files
- 6 chapter files
- 4 redteam files

Write `shared/final_batch_report.md` in 700-1000 words. Include:

1. Actual agent count and phase completion.
2. Whether batch reached 24+12+6+4.
3. Query-edge behavior and whether children queried peers/upstream phases.
4. Remaining workflow gaps exposed by the run.
5. Recommendations for batch/spawn/wait/query/stats/visualization.

Then submit final report and call:

```json
set_status(status="done", result="Fast full batch high-concurrency workflow completed: 24 scouts + 12 reviewers + 6 integrators + 4 redteam.")
```
