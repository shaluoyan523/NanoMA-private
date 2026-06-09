# NanoMA Full Batch Fanout Task

你是 root coordinator。完成一次全量高并发工作流压力测试：24 个 scout、12 个 reviewer、6 个 integrator、4 个 redteam，总计 46 个子 agent，加 root 为 47 个 agent。

源码快照在 `shared/source/`。不要使用 `shell`。

Root 的职责是编排，不是先研究源码：启动后不要预读 `shared/source` 中的大文件；立即创建目录、写 batch JSON、调用 `batch` 拉起 scout wave。源码阅读交给 scout。

## 硬性规则

1. 子 agent 必须通过 `batch` 一波一波批量拉起，不要逐个手动 `spawn`。
2. 每一波 batch JSON 写在 root 私有工作区根目录，例如 `scout_batch.json`，不要写到 `shared/`。
3. batch 文件优先使用 `{ "tool": "...", "args": {...} }`；如果使用 `{ "name": "...", "params": {...} }` 也可以。
4. 每次 `batch` 后检查结果：目标数量必须全部执行成功。如果有 error，补齐缺失 agent 后再进入 `wait`。
5. 每一波全部创建完成后再 `wait(mode="all")`；随后必须用 `query(filter={...}, tags=[...])` 主动盘点状态、public_memory、artifact_index。
6. 所有协作文件都写到 `shared/` 下。
7. 子 agent 不要默认向 root 汇报完成；父子关系只表示 lineage。每个 agent 通过细粒度 tags、`query`、`compact`、`submit` 让其他 agent 主动发现信息。
8. 控制输出长度。每个子 agent 只读必要文件，禁止复制长段源码或长段报告。
9. 阅读源码时先 `grep`，再用 `file_read(path=..., offset=<grep 行号附近>, limit=40)` 读取小片段。不要整文件读取 `core.py`、`meta.py`、`tools.py`、`viewer.html`、测试文件或 runner。
10. 这次目标是工作流压力测试，不是完整源码审计。子 agent 看够证据后立刻写短产物并停止，不要继续深挖。

## Batch 文件格式

```json
[
  {
    "tool": "spawn",
    "args": {
      "task": "子 agent 的完整任务说明",
      "role": "scout",
      "create_type": "peer_agent",
      "relationship": "peer",
      "group_id": "scout_wave",
      "workflow_prior": "research_loop",
      "current_task_tags": ["batch", "phase:scout", "scout:01", "topic:runtime"]
    }
  }
]
```

写完文件后调用：

```json
{"path": "scout_batch.json"}
```

## 通用子 Agent 完成协议

每个子 agent 的最后一个工具调用应是：

```json
{
  "summary": "一句到三句公开摘要，说明发现、产物和风险",
  "tags": ["phase:<phase>", "<role>:XX", "...细粒度主题 tag..."],
  "files": ["shared/.../产物.md"],
  "experience": "可被后续 agent 查询复用的一句话经验",
  "stop_after": true,
  "result": "<ROLE>-XX done"
}
```

也就是用 `compact(..., stop_after=true, result=...)` 完成最终 memory 整理和停止；不要再额外调用 `set_status`，也不要给 root 发完成消息。

## 建议源码定位

每个 scout 先 `grep` 一次，再最多读取 1-2 个相关文件的片段。每次 `file_read` 必须带 `offset` 和 `limit`，建议 `limit<=40`。可优先参考：

- runtime/lifecycle/spawn/wait/query/compact/stats：`shared/source/nanoma/core.py`、`shared/source/nanoma/meta.py`、`shared/source/nanoma/memory.py`
- file/shared/permission/sandbox/tools：`shared/source/nanoma/tools.py`、`shared/source/nanoma/sandbox.py`、`shared/source/examples/high_concurrency_runner.py`
- LLM/router/budget/context：`shared/source/nanoma/llm.py`、`shared/source/nanoma/models.py`、`shared/source/nanoma/cost.py`
- viewer/logging/observability/tests：`shared/source/nanoma/viewer.html`、`shared/source/tests/test_nanoma.py`

## Phase 1: Scout Wave

一次性通过 `batch` 拉起 24 个 scout agent，`group_id="scout_wave"`。

每个 scout 研究一个主题，写两个文件：

- `shared/scouts/scout_XX_summary.md`：80-120 words
- `shared/scouts/scout_XX_notes.md`：最多 180 words

24 个主题：

1. `runtime architecture`
2. `agent lifecycle`
3. `spawn and create_agent behavior`
4. `batch workflow behavior`
5. `wait and message interruption`
6. `query and state board`
7. `memory and compact behavior`
8. `artifact submit behavior`
9. `workspace and shared path behavior`
10. `tool permission boundaries`
11. `sandbox behavior`
12. `LLM routing and model selection`
13. `budget and ledger behavior`
14. `context growth risks`
15. `parallel scheduling`
16. `error handling`
17. `workflow_prior limitations`
18. `agent group metadata`
19. `high-concurrency observability`
20. `runner CLI ergonomics`
21. `viewer/logging behavior`
22. `test coverage gaps`
23. `security risks`
24. `recommended runtime improvements`

每个 scout 任务必须包含：

- 只使用 `file_read`、`file_list`、`grep`、`file_write`、`query`、`compact`、`submit`；只有明确协作需要时才使用 `send`。
- 先 `query(filter={"group_id":"scout_wave"}, tags=["phase:scout"], include_memory=true, limit=8)`，了解同波 agent 的 tags/状态，避免重复角度。
- 先 `grep` 相关关键词，再用 `file_read(..., offset=<命中行附近>, limit=40)` 读取片段。
- 只读相关源码片段，不超过 2 个文件、总共不超过 3 个片段。不要因为证据不完美而继续搜索。
- 完成后 `submit` summary 文件。
- 最后 `compact(..., stop_after=true, result="SCOUT-XX done")`，tags 必须包含 `phase:scout`、`scout:XX`、主题 tag 和至少一个源码区域 tag。

Phase 1 完成标准：root 调用 `query(filter={"group_id":"scout_wave","status":"done"}, tags=["phase:scout"])` 能看到 24 个 scout 完成，且 `shared/scouts/` 下有 24 个 summary 文件。

## Phase 2: Reviewer Wave

一次性通过 `batch` 拉起 12 个 reviewer agent，`group_id="reviewer_wave"`。

每个 reviewer 审查两个 scout summary/notes，写：

- `shared/reviews/reviewer_XX.md`：120-180 words

分配：

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

每个 reviewer 必须：

- 先调用 `query(tags=["phase:scout"], include_memory=true)`，并用负责的 `scout:XX` tags 定位 public_memory/artifact_index。
- 只读自己负责的 scout 文件。
- 指出事实风险、遗漏、重复和可执行改进。
- 完成后 `submit` review 文件。
- 最后 `compact(..., stop_after=true, result="REVIEWER-XX done")`，tags 必须包含 `phase:reviewer`、`reviewer:XX`、负责的 `scout:XX`。

Phase 2 完成标准：root 调用 `query(filter={"group_id":"reviewer_wave","status":"done"}, tags=["phase:reviewer"])` 能看到 12 个 reviewer 完成，且 `shared/reviews/` 下有 12 个 review 文件。

## Phase 3: Integrator Wave

一次性通过 `batch` 拉起 6 个 integrator agent，`group_id="integrator_wave"`。

每个 integrator 合成四个 scout 主题和两个 reviewer 文件，写：

- `shared/integration/chapter_XX.md`：180-260 words

分配：

1. chapter 01: scouts 01-04, reviewers 01-02
2. chapter 02: scouts 05-08, reviewers 03-04
3. chapter 03: scouts 09-12, reviewers 05-06
4. chapter 04: scouts 13-16, reviewers 07-08
5. chapter 05: scouts 17-20, reviewers 09-10
6. chapter 06: scouts 21-24, reviewers 11-12

每个 integrator 必须：

- 先通过 `query(tags=["phase:scout"], include_memory=true)` 和 `query(tags=["phase:reviewer"], include_memory=true)` 主动发现相关 agent 状态、public_memory、artifact_index。
- 只读负责范围内的 scout summary 和 reviewer 文件。
- 输出清晰的 implementation recommendations。
- 完成后 `submit` chapter 文件。
- 最后 `compact(..., stop_after=true, result="INTEGRATOR-XX done")`，tags 必须包含 `phase:integrator`、`chapter:XX`、覆盖的 scout/reviewer tags。

Phase 3 完成标准：root 调用 `query(filter={"group_id":"integrator_wave","status":"done"}, tags=["phase:integrator"])` 能看到 6 个 integrator 完成，且 `shared/integration/` 下有 6 个 chapter 文件。

## Phase 4: Redteam Wave

一次性通过 `batch` 拉起 4 个 redteam agent，`group_id="redteam_wave"`。

每个 redteam 读 6 个 chapter 和必要 reviewer 摘要，写：

- `shared/redteam/redteam_XX.md`：160-240 words

四个角度：

1. `correctness and source-grounding`
2. `runtime reliability and completion risk`
3. `security and permission boundaries`
4. `workflow scalability and cost control`

每个 redteam 必须：

- 先通过 `query(tags=["phase:integrator"], include_memory=true)` 和 `query(tags=["phase:reviewer"], include_memory=true)` 主动发现相关 agent 状态和 public_memory。
- 找出最终报告里最容易误判的地方。
- 给出优先级排序的修正建议。
- 完成后 `submit` redteam 文件。
- 最后 `compact(..., stop_after=true, result="REDTEAM-XX done")`，tags 必须包含 `phase:redteam`、`redteam:XX`、审查角度 tag。

Phase 4 完成标准：root 调用 `query(filter={"group_id":"redteam_wave","status":"done"}, tags=["phase:redteam"])` 能看到 4 个 redteam 完成，且 `shared/redteam/` 下有 4 个 redteam 文件。

## Final Synthesis

root 最后必须先 query 四个 phase：

- `query(filter={"group_id":"scout_wave","status":"done"}, tags=["phase:scout"], include_memory=true)`
- `query(filter={"group_id":"reviewer_wave","status":"done"}, tags=["phase:reviewer"], include_memory=true)`
- `query(filter={"group_id":"integrator_wave","status":"done"}, tags=["phase:integrator"], include_memory=true)`
- `query(filter={"group_id":"redteam_wave","status":"done"}, tags=["phase:redteam"], include_memory=true)`

然后读取：

- 24 个 scout summary
- 12 个 reviewer 文件
- 6 个 integration chapter
- 4 个 redteam 文件

写 `shared/final_batch_report.md`，控制在 800-1200 words，必须包含：

1. 实际拉起 agent 数量与各 phase 完成情况
2. batch 拉起能力是否达标
3. 查询边是否足够活跃，父子 agent 是否通过 query 获取彼此信息
4. 数量不达标或统计误判的原因
5. 对 `batch`、`spawn`、`wait`、`query`、`stats` 的修改建议
6. 高并发任务的权限边界建议
7. 后续实现优先级

最后：

1. `submit(path="shared/final_batch_report.md")`
2. `set_status(status="done", result="Full batch high-concurrency workflow completed: 24 scouts + 12 reviewers + 6 integrators + 4 redteam.")`
