# 自然并发全流程测试任务

你是 root orchestrator。请为一个真实感较强的产品做一份完整方案：

**产品主题：给 20-80 人远程团队使用的 AI 会议纪要与知识库助手。**

目标是跑完整个多 agent 工作流，最终必须产出并提交 `shared/final_plan.md`。

## 边界和约束

1. 不要调用 `shell`。只使用 `file_write`、`file_read`、`file_list`、`grep`、`spawn`、`wait`、`query`、`send`、`set_status`、`submit`。
2. 所有共享产物都写到 `shared/...`。这个路径就是全局共享目录，不要写绝对路径。
3. 每个子 agent 的输出文件控制在 600-900 字，重点写可执行结论，不要写长篇报告。
4. 每个子 agent 开始时调用 `set_status(action="work", ...)`。
5. 每个子 agent 结束前都要 `send()` 简短汇报给 root，再 `set_status(status="done", result="...")`。
6. 每一波必须一次性 spawn 完全部 agent，再 wait，不要串行 spawn/wait。

## Phase 0: Brief

root 先写 `shared/brief.md`，内容包括：产品背景、目标用户、协作计划、最终报告结构。

然后调用：

`set_status(action="work", current_task_tags=["orchestration", "bounded-product-plan"], work_outline="parallel research -> compact reviews -> final synthesis")`

## Phase 1: 6 个并发 researcher

一次性 spawn 6 个 researcher：

1. `USER_NEEDS`：用户、痛点、核心场景。写 `shared/research/user_needs.md`。
2. `MVP_SCOPE`：MVP 范围、非目标、验收标准。写 `shared/research/mvp_scope.md`。
3. `WORKFLOW`：会议录入、摘要、行动项、归档、检索流程。写 `shared/research/workflow.md`。
4. `DATA_AI`：数据模型和 AI 能力设计。写 `shared/research/data_ai.md`。
5. `SECURITY`：权限、隐私、审计、数据保留。写 `shared/research/security.md`。
6. `METRICS_STACK`：技术栈和评估指标。写 `shared/research/metrics_stack.md`。

每个 researcher 的任务：

- `set_status(action="work", current_task_tags=["research", "<role>"], work_outline="focused research -> concise shared file -> report")`
- 写对应文件，600-900 字，必须包括：
  - 5 条关键结论
  - 3 条具体建议
  - 2 个主要风险
  - 和其他模块的接口点
- `send(to="alpha", message="<ROLE> completed: <2 sentence summary>")`
- `set_status(status="done", result="<ROLE> completed")`

spawn 参数：

- `role="researcher"`
- `create_type="peer_agent"`
- `relationship="parallel_researcher"`
- `group_id="bounded_phase1"`
- `workflow_prior="research_loop"`

spawn 完 6 个后，root 调用 `wait(mode="all", timeout=360)`。如果被消息打断，就再次 wait 未完成的 agent，直到 6 个 researcher 都完成。随后调用一次 `query()`。

## Phase 2: 3 个并发 reviewer

一次性 spawn 3 个 reviewer：

1. `PRODUCT_REVIEWER`：检查产品范围、用户价值和 MVP 取舍。写 `shared/reviews/product_review.md`。
2. `TECH_REVIEWER`：检查架构、数据流、技术实现风险。写 `shared/reviews/tech_review.md`。
3. `DELIVERY_REVIEWER`：制定 6 周实施计划和上线指标。写 `shared/reviews/delivery_review.md`。

每个 reviewer 的任务：

- `set_status(action="work", current_task_tags=["review", "<role>"], work_outline="read concise research -> synthesize review -> report")`
- 读取 `shared/research/*.md`
- 写对应文件，600-900 字，必须包括：
  - 5 条评审结论
  - 5 条可执行建议
  - 3 个需要 root 在最终方案里处理的问题
- `send(to="alpha", message="<ROLE> completed: <2 sentence summary>")`
- `set_status(status="done", result="<ROLE> completed")`

spawn 参数：

- `role="reviewer"`
- `create_type="peer_agent"`
- `relationship="parallel_reviewer"`
- `group_id="bounded_phase2"`
- `workflow_prior="critic_review_loop"`

spawn 完 3 个后，root 调用 `wait(mode="all", timeout=360)`。如果被消息打断，就再次 wait 未完成的 agent，直到 3 个 reviewer 都完成。随后调用一次 `query()`。

## Phase 3: Final Plan

root 读取：

- `shared/brief.md`
- 所有 `shared/research/*.md`
- 所有 `shared/reviews/*.md`

写 `shared/final_plan.md`，1200-1800 字，结构：

1. Executive Summary
2. Target Users And Pain Points
3. MVP Scope
4. End-To-End Workflow
5. AI And Data Design
6. Security And Privacy
7. Architecture And Tech Stack
8. Six-Week Delivery Plan
9. Evaluation Metrics
10. Top Risks And Mitigations

最后执行：

1. `submit("shared/final_plan.md")`
2. `set_status(status="done", result="Bounded natural concurrency workflow completed with final_plan.md submitted.")`
