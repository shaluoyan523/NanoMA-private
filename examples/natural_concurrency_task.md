# 自然并发工作流测试任务

你是 root orchestrator。请为一个真实感较强的产品做一份完整方案：

**产品主题：给 20-80 人远程团队使用的 AI 会议纪要与知识库助手。**

目标不是写营销文案，而是产出一份可执行的产品与技术方案，最终文件为 `shared/final_plan.md`。你应通过并发子 agent 协作完成，不要自己包办所有分析。

## 工作要求

1. 先写 `shared/brief.md`，简要说明产品背景、输出结构、分工方式。
2. 调用 `set_status(action="work", current_task_tags=["orchestration", "product-planning"], work_outline="parallel research -> review -> architecture -> final plan")`。
3. 第一波一次性并发 spawn 8 个 researcher，不要 spawn 一个等一个。
4. researcher 全部完成后调用 `wait(mode="all", timeout=240)`，再调用一次 `query()`。
5. 第二波一次性并发 spawn 4 个 reviewer / architect，不要串行。
6. reviewer / architect 全部完成后调用 `wait(mode="all", timeout=240)`，再调用一次 `query()`。
7. 你最后读取所有共享文件，写 `shared/final_plan.md`，调用 `submit("shared/final_plan.md")`，再 `set_status(status="done", result="...")`。

每个子 agent 开始时都要调用 `set_status(action="work", current_task_tags=[...], work_outline="...")`；结束前都要写文件、`send()` 简短汇报给你，并 `set_status(status="done", result="...")`。

## 第一波：8 个并发 researcher

一次性 spawn 以下 8 个 agent：

1. `USER_RESEARCH`：分析远程团队会议纪要和知识沉淀的真实用户需求、痛点、使用场景。输出 `shared/research/user_research.md`。
2. `PRODUCT_SCOPE`：定义 MVP 范围、非目标、用户故事、验收标准。输出 `shared/research/product_scope.md`。
3. `WORKFLOW_DESIGN`：设计会议录入、摘要生成、任务提取、知识库归档、检索复用的端到端流程。输出 `shared/research/workflow_design.md`。
4. `DATA_MODEL`：设计核心数据模型，包括 meeting、transcript、summary、action item、knowledge card、source citation、permission。输出 `shared/research/data_model.md`。
5. `AI_FEATURES`：设计 AI 能力，包括摘要、行动项、决策提取、主题聚类、语义检索、冲突提醒。输出 `shared/research/ai_features.md`。
6. `SECURITY_PRIVACY`：分析权限、隐私、审计、数据保留、企业部署风险。输出 `shared/research/security_privacy.md`。
7. `TECH_STACK`：提出后端、前端、队列、搜索、向量库、存储、部署的技术选择。输出 `shared/research/tech_stack.md`。
8. `EVALUATION_METRICS`：设计上线前和上线后的评估指标，包括摘要质量、召回率、延迟、用户采纳、错误率。输出 `shared/research/evaluation_metrics.md`。

每个 researcher 的任务模板：

你是 `<ROLE>`，负责 `<主题>`。请完成：

- `set_status(action="work", current_task_tags=["research", "<role-slug>"], work_outline="analyze assigned area -> write shared file -> report")`
- 写对应的 `shared/research/*.md`
- 文件内容要包括：关键结论、具体建议、风险、需要其他角色注意的接口点
- `send(to=<parent>, message="<ROLE> completed: <3 sentence summary>")`
- `set_status(status="done", result="<ROLE> completed")`

spawn 参数建议：

- `role="researcher"`
- `create_type="peer_agent"`
- `relationship="parallel_researcher"`
- `group_id="natural_phase1"`
- `workflow_prior="research_loop"`

## 第二波：4 个并发 reviewer / architect

第一波完成后，一次性 spawn 以下 4 个 agent：

1. `PRODUCT_REVIEWER`：读取所有 `shared/research/*.md`，检查产品范围是否一致、是否有明显缺口。输出 `shared/reviews/product_review.md`。
2. `ARCHITECTURE_REVIEWER`：读取所有研究文件，提出系统架构、服务边界、数据流、异步任务、检索链路。输出 `shared/reviews/architecture_review.md`。
3. `RISK_REVIEWER`：读取所有研究文件，挑出隐私、合规、权限、幻觉、误提取任务等风险。输出 `shared/reviews/risk_review.md`。
4. `IMPLEMENTATION_PLANNER`：读取所有研究文件，制定 6 周 MVP 实施计划，分阶段列交付物。输出 `shared/reviews/implementation_plan.md`。

每个 reviewer / architect 的任务模板：

你是 `<ROLE>`。请完成：

- `set_status(action="work", current_task_tags=["review", "<role-slug>"], work_outline="read research outputs -> critique/synthesize -> write review file")`
- 读取所有 `shared/research/*.md`
- 写对应的 `shared/reviews/*.md`
- 文件内容要具体，避免空泛；至少给出 8 条可执行建议
- `send(to=<parent>, message="<ROLE> completed: <3 sentence summary>")`
- `set_status(status="done", result="<ROLE> completed")`

spawn 参数建议：

- `role="reviewer"`
- `create_type="peer_agent"`
- `relationship="parallel_reviewer"`
- `group_id="natural_phase2"`
- `workflow_prior="critic_review_loop"`

## 最终汇总

你最后读取：

- `shared/brief.md`
- `shared/research/*.md`
- `shared/reviews/*.md`

写 `shared/final_plan.md`，结构包括：

1. Executive Summary
2. Target Users And Pain Points
3. MVP Scope
4. End-To-End Workflow
5. AI Capabilities
6. Data Model
7. System Architecture
8. Security And Privacy
9. Evaluation Metrics
10. Six-Week Implementation Plan
11. Top Risks And Mitigations

最后执行：

- `submit("shared/final_plan.md")`
- `set_status(status="done", result="Natural concurrency product planning workflow completed.")`
