# NanoMA

[English README](README.md)

## 节点自主的动态多智能体编排

> 本仓库是论文双盲审稿阶段使用的匿名材料。作者姓名、所属机构、联系方式、仓库地址和引用信息均暂时隐藏。

长时程任务很少会始终遵循运行开始时设想的任务分解。执行过程中会出现新的依赖关系，一个分支发现的证据可能改变另一个分支的价值，部分分支也可能变得冗余。NanoMA 将不断变化的协作结构视为求解状态的一部分，而不是在执行前固定好的工作流。

NanoMA 是一个多智能体 ReAct 框架，其中**每个智能体同时承担规划者和执行者的角色**。在每一步中，一个节点可以通过执行动作推进自己的局部任务，也可以通过以下四种拓扑动作改变当前的协作结构：

- `spawn`：为一个有明确边界的子任务插入新的智能体节点；
- `kill`：终止剩余价值已不足以支持继续执行的分支；
- `query`：按需读取非本地的执行状态或证据；
- `send`：向指定智能体传递证据、约束、进展或更新后的指令。

NanoMA 的核心并不只是递归委派，而是将固定的任务来源关系与执行过程中逐渐形成的协作关系分离。子节点会保留其任务来自哪个父节点的信息，但这一来源关系并不限制它在执行过程中可以查看、通知或协调哪些节点。

## 从固定工作流到动态演化的执行图

论文区分了三种组织方式：

| 组织方式 | 规划能力位于何处 | 协作结构如何变化 |
| --- | --- | --- |
| 中心规划 | 由一个控制器分解任务、安排依赖并汇总结果 | 所有变化都需要经过中心控制器 |
| 节点递归 | 每个节点可以继续分解自己的任务并汇总后代结果 | 扩展能力被分散，但交互主要局限于父子关系 |
| **NanoMA** | **每个活跃节点都能规划和执行** | **节点可以在运行时扩展、观察、更新并收缩协作结构** |

在时刻 \(t\)，NanoMA 将执行过程表示为一个带类型的动态图：

$$
G_t = (V_t, E_t^{\mathrm{prov}}, E_t^{\mathrm{comm}}),
$$

其中，\(V_t\) 是已经实例化的智能体集合，\(E_t^{\mathrm{prov}}\) 保存不可变的创建边，\(E_t^{\mathrm{comm}}\) 保存执行过程中动态形成的有向信息边。来源边只记录任务的起点，并不限制之后的通信关系。

每个节点 \(v_i\) 都维护独立状态：

$$
s_i^t = (\tau_i^t, h_i^t, Q_i^t, W_i^t, \ell_i^t, b_i^t, O_i^t, I_i^t),
$$

分别表示当前任务、有效交互历史、动态局部任务队列、私有工作区、生命周期状态、资源状态、结果与产物，以及按优先级组织的消息收件箱。新创建的子节点拥有独立上下文、工作区和任务队列，并自行决定如何执行或进一步分解自己的任务。

## 将拓扑动作视为图操作

四种拓扑动作构成了操作执行图的最小数据结构接口。

| 数据结构操作 | 拓扑动作 | 直接作用 |
| --- | --- | --- |
| 插入 | `spawn` | 向 \(V_t\) 添加节点，并向 \(E_t^{\mathrm{prov}}\) 添加来源边 |
| 删除 | `kill` | 将指定分支移出活跃执行集合，同时保留其历史和已经交付的证据 |
| 查询 | `query` | 读取可见节点的状态，不修改目标节点或执行图 |
| 更新 | `send` | 更新接收者的收件箱，并在 \(E_t^{\mathrm{comm}}\) 中形成一条有向信息边 |

这些动作可以组合成更大的组织结构。`spawn + wait` 可以表达递归委派；创建子节点后不立即等待，则允许父子节点并行工作；`query + send` 可以在不同分支和层级之间连接证据；`kill` 则在有用信息得到保留后收缩活跃前沿。因此，系统无需预先指定拓扑模板，就可以形成星形、链式、深层递归路径、树、稠密子图、轮图或有向环等结构。

### 当前实现如何落实 `spawn`

`spawn` 是概念层面的节点插入动作。在当前的同模型配置中，它并不是一个可以由模型无限制直接调用的工具。智能体首先通过 `task_create` 创建一个边界明确的局部工作项；在这个新的规划节点上，**SpawnJudge 使用该节点自身的模型和上下文**，判断该工作项应当留在本地执行，还是转换为一个或多个子节点。随后，Runtime 再检查模型可用性、内存、深度、节点数量和策略等可执行约束。

节点插入能力仍然存在于每一个符合条件的节点上，因此，尽管节点创建受到守卫机制约束，规划能力依然被下放到了各个节点。

这一脚手架针对的是当前模型的一项现实局限：LLM 容易生成宽泛的角色名称、创建语义重复的分支，或者在没有明确输出契约的情况下持续递归。守卫机制只约束单次扩展的质量和可接受性，并不会预先生成全局计划，也不会提前决定完整执行图。

## 每个智能体的执行循环

每个活跃节点都执行相同的循环：

1. 观察本地工具结果、收到的消息，以及当前决策所需的可见节点。
2. 通过 `task_create`、`task_update` 和 `task_list` 维护节点自己的局部计划。
3. 在本地执行任务；或者在 SpawnJudge 与 Runtime 准入同时通过时，创建拥有独立局部循环的子节点。
4. 当非本地证据会改变当前计划时，使用 `query` 和 `send` 获取或传递信息。
5. 继续独立工作，通过 `wait` 等待依赖，或者通过 `kill` 剪除冗余后代节点。
6. 只有在满足局部输出契约后，才交付有证据支持的结果并结束运行。

四种拓扑动作始终是方法层面的核心抽象。规划、同步、交付、验证、生命周期和工作区接口是使这些动作能够可靠执行并形成持久状态的工程适配层。

## 可靠协作与最终发布

NanoMA 将协作过程与最终发布过程分开：

- **定向通信。** `query` 按需读取状态，`send` 改变指定接收者的信息状态。消息可以排队到下一轮、插入两组工具调用之间，或者立即送达。
- **持久化答案。** `deliver_to_parent` 将答案、证据、置信度、方法和来源写入候选账本，而不是只依赖可能被压缩或中断的对话历史。
- **私有工作区。** 并行节点在隔离的任务树中工作。`transfer` 和交付差分负责跨工作区移动修改，而不是让所有子节点直接操作最终目标。
- **经验证的状态晋升。** 可复现检查可以验证合并候选、回滚明确的退化，并保留由 Runtime 持有的最佳状态快照。
- **由根节点负责最终发布。** 根节点的交付契约负责暂存完整候选、检查必需文件与验证器，并以原子方式替换正式目标。因此，子节点完成任务并不等同于最终发布。

`kill` 还包含交付宽限机制：如果一个后代节点已经产生实质性内容但尚未正式交付，Runtime 会先要求它交付，并推迟终止操作。这样既能保留有用工作，也允许活跃执行图继续收缩。

## 核心贡献

论文提出三个核心贡献：

1. **每个节点都具备自主规划和执行能力。** NanoMA 将编排决策分散到各个节点，并将不可变的任务来源关系与动态形成的协作关系分离。
2. **统一的拓扑动作接口。** `spawn`、`kill`、`query` 和 `send` 在与任务执行相同的 ReAct 循环中，为运行时执行图提供插入、删除、查询和更新操作。
3. **对性能与涌现编排结构进行实证研究。** 实验在相同模型骨干下比较不同组织方式，分析完整轨迹中形成的协作结构，并消融信息可见范围和消息接收范围。

## 实验结果

### RepoZero-Py2JS Hard

| Agent 框架 | Harness 类型 | GPT-5.6 Luna High All-pass | GPT-5.6 Luna High Micro | Claude Sonnet 5 All-pass | Claude Sonnet 5 Micro | Qwen3.7-Plus All-pass | Qwen3.7-Plus Micro |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenHands-bash | 单智能体 | 44.83% | 80.03% | 43.52% | 79.82% | 25.899% | 57.097% |
| CrewAI | 顺序执行 | 53.24% | 75.02% | 36.69% | 77.11% | 28.777% | 67.102% |
| CC-Workflow | 中心规划 | 48.20% | 74.69% | 24.46% | 52.89% | 24.460% | 54.347% |
| Omnigent | 节点递归 | 47.48% | 79.20% | 48.20% | 87.17% | **39.568%** | 72.260% |
| **NanoMA** | **节点自主** | **66.91%** | **93.32%** | **51.80%** | **87.30%** | 35.971% | **76.561%** |

### EdgeBench

| 模型 | Agent 框架 | Harness 类型 | Overall score |
| --- | --- | --- | ---: |
| GPT-5.6 Luna | CC-Workflow | 中心规划 | 16.60 |
| GPT-5.6 Luna | CrewAI | 顺序执行 | 19.36 |
| GPT-5.6 Luna | Codex baseline | 单智能体 | 20.04 |
| GPT-5.6 Luna | Omnigent | 节点递归 | 14.31 |
| GPT-5.6 Luna | **NanoMA** | **节点自主** | **22.10** |
| Qwen3.7-Plus | CC-Workflow | 中心规划 | 10.95 |
| Qwen3.7-Plus | CrewAI | 顺序执行 | 13.09 |
| Qwen3.7-Plus | Codex baseline | 单智能体 | 17.52 |
| Qwen3.7-Plus | Omnigent | 节点递归 | 13.06 |
| Qwen3.7-Plus | **NanoMA** | **节点自主** | **18.58** |

## 局限性

当前 LLM 仍然无法可靠预判拓扑动作所产生的延迟影响和跨智能体影响。模型可能低估协调成本、破坏有价值的并行过程、过早终止分支，或者创建范围不清晰且语义重复的工作。仅靠静态提示无法使这些决策达到最优。因此，论文将基于 Runtime 反馈学习动作选择策略视为后续的重要方向。

结构收敛论证也依赖明确假设：每个节点只创建有限个更小的任务义务；每次被接受的进展都会减少未完成工作；停滞节点最终会重新进入有效执行或被终止。这些条件能够保证活跃前沿最终终止，但不能保证答案正确。

## 仓库结构

| 路径 | 在方法中的作用 |
| --- | --- |
| `nanoma/core.py` | 节点状态、逐节点 ReAct 循环、Runtime 状态、规划守卫、生命周期转移和检查点 |
| `nanoma/meta.py` | 拓扑动作，以及通信、同步和交付适配器 |
| `optimizations/todo_tools/` | 节点局部任务队列，以及 SpawnJudge 使用的规划节点 |
| `nanoma/delivery.py` | 交付契约、结构化候选和冻结的产物树 |
| `nanoma/merge_submit.py` | 候选账本、差分合并、验证、最佳状态保留和最终发布 |
| `nanoma/online_objective.py` | 解析并排序 benchmark 可见的目标测量结果 |
| `nanoma/plugins/workspace_tools/` | 结构化文件创建、读取、编辑和代码检索工具 |
| `benchmarks/edgebench/` | EdgeBench 评测适配器 |
| `benchmarks/repozero/` | RepoZero-Py2JS Hard 评测适配器 |

## 安装

NanoMA 需要 Python 3.11 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

通过环境变量配置兼容 Chat Completions 的模型接口。请始终将凭据保存在仓库之外。

```bash
export NANOMA_API_KEY="<provider-key>"
export NANOMA_LLM_BASE_URL="https://provider.example/v1"
```

最小化调用示例：

```python
import asyncio
from pathlib import Path

from nanoma import Runtime, RuntimeConfig


async def main() -> None:
    runtime = Runtime(
        config=RuntimeConfig(
            default_model="<model-id>",
            workspace_root=Path("./workspace"),
            log_dir=Path("./logs"),
        )
    )
    print(await runtime.run("Your task"))


asyncio.run(main())
```

## 评测适配器

评测适配器需要另外准备相应 benchmark 的安装环境和官方任务资源。

EdgeBench：

```bash
python benchmarks/edgebench/run_nanoma_edgebench.py \
  --prompt-file /path/to/task_prompt.txt \
  --task-cwd /path/to/edgebench/task \
  --workspace /path/to/workspace \
  --model <model-id> \
  --wall-seconds 7200
```

RepoZero-Py2JS Hard：

```bash
python benchmarks/repozero/run_nanoma_repozero_hard_small.py \
  --run-dir /path/to/output \
  --repozero-root /path/to/repozero \
  --model <model-id> \
  --tasks all
```

两个适配器均不包含模型凭据、私有接口地址、benchmark 数据或评分器输出。

## 验证

```bash
pytest -q \
  tests/test_delivery_contract.py \
  tests/test_online_objective.py \
  tests/test_shell_memory_admission.py
```

该匿名快照在这组回归测试中的结果为 `57 passed, 1 skipped`。

## 匿名审稿说明

`anonymous` 分支由一个无父提交组成，提交作者统一为 `Anonymous Authors`。其中不包含作者身份、引用信息、凭据、私有基础设施地址或机器专用启动脚本。为了维持双盲审稿，请分发该分支的压缩包，或将其镜像到匿名仓库服务；即使分支内容和提交历史已经匿名化，直接共享个人 GitHub 仓库地址仍可能暴露账号所有者。

## 许可证

本项目采用 MIT License。引用信息将在匿名审稿结束后恢复。
