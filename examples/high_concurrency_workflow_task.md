# NanoMA 高并发工作流压力测试任务

你是 root orchestrator。目标是构建一份“2026 年 AI 多智能体研发平台技术评估报告”，但这个任务的真正目的不是报告本身，而是压力测试 NanoMA 的高并发、多轮协作、共享文件、消息和状态公告板。

必须遵守：

1. 你只负责调度，不要自己完成子任务。
2. 每一波 worker 必须先全部 `spawn()` 完，再 `wait()`，不要 spawn 一个等一个。
3. 每个子 agent 都要写文件到 `shared/`，并在结束前 `send()` 简短结果给父 agent，然后 `set_status(status="done", result=...)`。
4. 每个子 agent 开始时调用 `set_status(action="work", current_task_tags=[...], work_outline="...")`。
5. 每一波结束后你都要调用一次 `query()`，检查 agent 状态、state_board 和 public_memory。
6. 最终必须 `submit("shared/final_report.md")`，然后 `set_status(status="done", result="...")`。
7. 不要用 shell 做大规模真实下载或联网请求；这个测试靠并发 agent 和文件协作施压。

## Phase 0: 初始化

先写 `shared/README.md`，说明本次测试的阶段、预期 agent 数量、输出目录：

- `shared/scouts/`
- `shared/reviews/`
- `shared/integration/`
- `shared/redteam/`
- `shared/final_report.md`

然后设置自己的状态：

`set_status(action="work", current_task_tags=["orchestration", "stress-test"], work_outline="fanout scouts -> review mesh -> integration -> redteam -> final")`

## Phase 1: 高并发侦察波，24 个并行 agent

一次性 spawn 24 个 SCOUT agent。每个 agent 研究一个不同主题，并写两份文件：

- `shared/scouts/scout_##.md`
- `shared/scouts/scout_##_signals.md`

主题列表：

1. runtime architecture
2. agent lifecycle
3. tool calling model
4. message routing
5. state board design
6. shared filesystem coordination
7. memory compaction
8. rebirth/context reset
9. cost ledger
10. scheduler/concurrency
11. sandbox security
12. trace viewer/events
13. preset prompt patterns
14. hierarchical delegation
15. swarm coordination
16. map-reduce coordination
17. debate/critic workflows
18. human-in-the-loop gates
19. failure modes
20. observability
21. scalability limits
22. model routing
23. reproducible evaluation
24. product positioning

每个 SCOUT 的任务模板：

你是 SCOUT-##，负责主题：<topic>。

流程：

1. `set_status(action="work", current_task_tags=["scout", "<topic-slug>"], work_outline="analyze topic -> write findings -> write signals")`
2. 写 `shared/scouts/scout_##.md`，内容包括：
   - 主题概述
   - 关键发现 6-10 条
   - 对 NanoMA 工作流测试的意义
   - 潜在风险
   - 建议指标
3. 写 `shared/scouts/scout_##_signals.md`，内容包括：
   - 3 个成功信号
   - 3 个失败信号
   - 3 个需要后续 reviewer 检查的问题
4. `send(to=<parent>, message="SCOUT-## done: <one paragraph summary>")`
5. `set_status(status="done", result="SCOUT-## completed <topic>")`

spawn 参数建议：

- `role="scout"`
- `create_type="peer_agent"`
- `relationship="parallel_worker"`
- `group_id="phase1_scouts"`
- `workflow_prior="research_loop"`
- `current_task_tags=["scout", "<topic-slug>"]`

全部 spawn 完以后，`wait(mode="all", timeout=300)`。
然后 `query()`。

## Phase 2: 交叉评审波，12 个并行 agent

一次性 spawn 12 个 REVIEWER agent。每个 reviewer 负责审查两个 scout 的输出：

1. reviewer_01: scouts 01-02
2. reviewer_02: scouts 03-04
3. reviewer_03: scouts 05-06
4. reviewer_04: scouts 07-08
5. reviewer_05: scouts 09-10
6. reviewer_06: scouts 11-12
7. reviewer_07: scouts 13-14
8. reviewer_08: scouts 15-16
9. reviewer_09: scouts 17-18
10. reviewer_10: scouts 19-20
11. reviewer_11: scouts 21-22
12. reviewer_12: scouts 23-24

每个 REVIEWER 的任务模板：

你是 REVIEWER-##。读取你负责的两个 scout 文件和 signals 文件。

流程：

1. `set_status(action="work", current_task_tags=["review", "cross-check"], work_outline="read scout files -> find gaps -> write review")`
2. 读取对应的 `shared/scouts/scout_*.md` 和 `shared/scouts/scout_*_signals.md`
3. 写 `shared/reviews/review_##.md`，内容包括：
   - 哪些发现可靠
   - 哪些发现重复、空泛或冲突
   - 至少 5 条改进建议
   - 建议进入最终报告的 top insights
   - 对工作流测试本身的观察
4. `send(to=<parent>, message="REVIEWER-## done: <summary>")`
5. `set_status(status="done", result="REVIEWER-## completed")`

spawn 参数建议：

- `role="reviewer"`
- `create_type="peer_agent"`
- `relationship="parallel_reviewer"`
- `group_id="phase2_reviewers"`
- `workflow_prior="critic_review_loop"`
- `current_task_tags=["review", "cross-check"]`

全部 spawn 完以后，`wait(mode="all", timeout=300)`。
然后 `query()`。

## Phase 3: 分区整合波，6 个并行 agent

一次性 spawn 6 个 INTEGRATOR agent。每个 integrator 负责一个报告章节：

1. architecture_and_runtime
2. concurrency_and_scheduler
3. memory_and_context
4. communication_and_coordination
5. safety_and_observability
6. evaluation_and_product_readiness

每个 INTEGRATOR 的任务模板：

你是 INTEGRATOR-##，负责章节：<section>。

流程：

1. `set_status(action="work", current_task_tags=["integration", "<section>"], work_outline="read scouts/reviews -> synthesize section")`
2. 读取所有相关 `shared/scouts/` 和 `shared/reviews/` 文件
3. 写 `shared/integration/<section>.md`，内容包括：
   - 章节摘要
   - 8-12 条综合洞察
   - 关键风险
   - 推荐改进
   - 可量化评估指标
4. `send(to=<parent>, message="INTEGRATOR-## done: <summary>")`
5. `set_status(status="done", result="INTEGRATOR-## completed <section>")`

spawn 参数建议：

- `role="integrator"`
- `create_type="peer_agent"`
- `relationship="parallel_synthesizer"`
- `group_id="phase3_integrators"`
- `workflow_prior="synthesis_loop"`
- `current_task_tags=["integration", "<section>"]`

全部 spawn 完以后，`wait(mode="all", timeout=300)`。
然后 `query()`。

## Phase 4: 红队压力评估波，4 个并行 agent

一次性 spawn 4 个 REDTEAM agent：

1. concurrency_failure_redteam
2. memory_consistency_redteam
3. artifact_quality_redteam
4. orchestration_failure_redteam

每个 REDTEAM 的任务模板：

你是 REDTEAM-##，负责：<redteam_focus>。

流程：

1. `set_status(action="work", current_task_tags=["redteam", "<focus>"], work_outline="stress critique -> failure hypotheses -> mitigations")`
2. 读取 `shared/integration/*.md`，必要时抽查 `shared/scouts/` 和 `shared/reviews/`
3. 写 `shared/redteam/redteam_##.md`，内容包括：
   - 可能失败的地方
   - 哪些文件或结论最脆弱
   - 并发/记忆/消息/文件协作方面的异常信号
   - 最少 8 条修复建议
4. `send(to=<parent>, message="REDTEAM-## done: <summary>")`
5. `set_status(status="done", result="REDTEAM-## completed")`

spawn 参数建议：

- `role="redteam"`
- `create_type="peer_agent"`
- `relationship="parallel_critic"`
- `group_id="phase4_redteam"`
- `workflow_prior="critic_review_loop"`
- `current_task_tags=["redteam", "<focus>"]`

全部 spawn 完以后，`wait(mode="all", timeout=240)`。
然后 `query()`。

## Phase 5: 最终汇总

你自己读取：

- `shared/integration/*.md`
- `shared/redteam/*.md`
- 关键 `shared/reviews/*.md`

写 `shared/final_report.md`，结构必须包括：

1. Executive Summary
2. What This Stress Test Exercised
3. Agent Concurrency Timeline
4. Workflow Primitive Coverage
5. Architecture Findings
6. Memory and Context Findings
7. Messaging and Coordination Findings
8. Failure Modes
9. Red Team Findings
10. Recommendations
11. Appendix: Files Produced

最后：

1. `submit("shared/final_report.md")`
2. `set_status(status="done", result="High-concurrency workflow stress test completed with 24 scouts + 12 reviewers + 6 integrators + 4 redteam agents.")`

## 规模预期

总 agent 数：root + 24 + 12 + 6 + 4 = 47。

推荐运行参数：

```bash
nanoma "$(cat examples/high_concurrency_workflow_task.md)" \
  --model deepseek-v4-flash \
  --budget 20 \
  --max-agents 60 \
  --time-limit 1200 \
  --workspace ./workspace-high-concurrency \
  --log-dir ./logs-high-concurrency
```

更激进的并发设置可以在程序化 API 里把 `max_concurrent_llm` 提到 32 或 48；CLI 当前没有暴露这个参数。
