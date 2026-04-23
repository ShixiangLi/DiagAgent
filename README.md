# DiagAgent — 基于大语言模型的工业故障诊断智能体

> **Diagnostic Agent for Industrial HVAC Fault Detection & Diagnosis**
>
> 利用 Qwen2.5-7B-Instruct + SFT + GRPO 强化学习，构建具备多步推理和工具调用能力的端到端故障诊断智能体。

---

## 1. 项目背景

现代建筑 HVAC（暖通空调）系统由多个高度耦合的子系统组成——冷水机组、锅炉、空气处理机组、风机盘管等。当故障发生时，其影响往往跨越多个子系统传播：一台冷水机组的性能退化可能导致下游空气处理机组的温度异常，再进一步影响末端风机盘管的舒适度。

传统的故障检测与诊断（FDD）方法面临以下挑战：
- **基于规则的专家系统**：需要大量人工编写规则，维护成本高，无法处理复杂的跨系统故障传播
- **单点预测模型**：只能判断单个设备是否异常，无法自主追踪故障的因果链
- **人工逐一排查**：效率低下，依赖维护人员的经验和对系统拓扑的理解

本项目构建一个**基于 Qwen2.5-7B 大语言模型的诊断智能体（Diagnostic Agent）**，使其能够理解系统拓扑、调用诊断工具、追踪故障因果链，并最终定位根因输出结构化诊断结论。

## 2. 数据集

### 2.1 LBNL-FDD 数据集

项目使用 **LBNL-FDD（Lawrence Berkeley National Laboratory Fault Detection and Diagnostics）** 数据集，包含 8 个 HVAC 子系统的完整模拟数据：

| 子系统 | 简称 | 组件数 | 传感器数 | 故障类型 | CSV 文件数 |
|:---|:---:|:---:|:---:|:---:|:---:|
| 冷水机组 (Chiller Plant) | `chiller_plant` | 18 | 79 | 23 | 24 |
| 锅炉 (Boiler Plant) | `boiler_plant` | 5 | 22 | 16 | 17 |
| 单风道 AHU | `sdahu` | 13 | 30 | 20 | 21 |
| 双风道 AHU | `ddahu` | 29 | 114 | 55 | 56 |
| 屋顶式空调 (RTU) | `rtu` | 5 | 24 | 24 | 25 |
| 风机盘管 (FCU) | `fcu` | 1 | 5 | 48 | 49 |
| 主风量末端 (PFPU) | `pfpu` | 29 | 109 | 30 | 31 |
| 从风量末端 (SFPU) | `sfpu` | 29 | 109 | 30 | 31 |
| **合计** | — | **129** | **492** | **246** | **254** |

每个 CSV 文件包含 50,000 行时间步的传感器数据，故障类型涵盖传感器偏差、阀门卡死、盘管结垢、制冷剂泄漏等。

### 2.2 TTL 语义文件

每个子系统附带 Brick Schema TTL 文件，定义了设备之间的层次结构（`brick:hasPart`）和传感器关联（`brick:hasPoint`），用于自动构建系统拓扑图。

## 3. 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                       DiagAgent 系统                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐   ┌───────────────┐   ┌────────────────┐  │
│  │  Topology     │   │  Node Models  │   │   Data Gen     │  │
│  │  Builder      │   │  (Oracle)     │   │   Pipeline     │  │
│  │  • TTL 解析   │   │  • 85 XGBoost │   │  • 场景采样    │  │
│  │  • 4层拓扑图  │   │  • 节点级预测 │   │  • 轨迹生成    │  │
│  │  • 跨系统链接 │   │  • 置信度输出 │   │  • SFT/RL 格式 │  │
│  └──────┬───────┘   └──────┬────────┘   └──────┬─────────┘  │
│         │                  │                    │             │
│         ▼                  ▼                    ▼             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              Oracle Environment Layer                   │  │
│  │  • UnifiedToolExecutor (8 tools)                        │  │
│  │  • FaultScenarioState (dynamic sensor data loading)     │  │
│  │  • PredictionToolExecutor (real XGBoost inference)       │  │
│  └────────────────────────┬───────────────────────────────┘  │
│                           │                                   │
│       ┌───────────────────┼────────────────────┐              │
│       ▼                   ▼                    ▼              │
│  ┌──────────┐     ┌──────────────┐     ┌────────────────┐   │
│  │   SFT    │     │     GRPO     │     │   Evaluation   │   │
│  │ Trainer  │     │   RL Trainer │     │   Framework    │   │
│  │          │     │              │     │                │   │
│  │ • LoRA   │     │ • Oracle     │     │ • 6 项指标     │   │
│  │   r=64   │ ──▶ │   rollouts   │     │ • 多模型对比   │   │
│  │ • 3000条 │     │ • 多维 reward │     │ • Markdown 报告│   │
│  └──────────┘     └──────────────┘     └────────────────┘   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## 4. 数据处理流程

### 4.1 拓扑构建

从 Brick Schema TTL 文件自动构建 4 层 NetworkX 有向图：

```
Level 0: Building (1 个根节点)
  └── Level 1: Systems (8 个子系统)
        └── Level 2: Components (129 个设备/部件)
              └── Level 3: Sensors (486 个传感器)
```

**拓扑规模**：624 节点，671 条边

**跨系统依赖关系**（10 条跨系统边）：
- `chiller_plant` → `sdahu`, `ddahu`, `fcu`（供冷水）
- `boiler_plant` → `sdahu`, `ddahu`, `fcu`（供热水）
- `sdahu`/`ddahu` → `pfpu`, `sfpu`（供空气）

### 4.2 节点模型（Oracle）构建

为拓扑中每个 Level-2 组件节点训练独立的 XGBoost 分类器：

| 项目 | 说明 |
|:---|:---|
| **输入特征** | 节点关联传感器的时序特征（当前值 + 滑动窗口统计量） |
| **输出** | 多分类：Normal / Fault_Type_1 / Fault_Type_2 / ... |
| **模型数** | 85 个（含 8 个 oracle 系统级模型） |
| **训练方法** | SMOTE + 5-fold 交叉验证 |
| **作用** | `diagnose_node` 工具的后端，为 Agent 提供真实的传感器数据驱动诊断结果 |

### 4.3 场景生成

从 246 种故障类型生成 5 类诊断场景：

| 场景类型 | 数量 | 说明 |
|:---|:---:|:---|
| `single_system` | 984 | 单系统故障，直接诊断 |
| `cross_system` | 160 | 跨系统故障传播，需要追踪因果链 |
| `no_fault` | 19 | 正常运行，需要排除故障 |
| `low_confidence` | 1,600 | 低置信度异常，需要多步验证 |
| `ambiguous` | 984 | 模糊症状，多个可能根因 |
| **合计** | **4,684** | — |

### 4.4 SFT 数据生成

对每个场景，通过路径引导的轨迹生成器构建多轮诊断对话：

1. **路径规划**：根据 `configs/diagnostic_paths.yaml` 预设的诊断路径确定最优工具调用序列
2. **轨迹模拟**：按路径依次生成 Agent 的工具调用和推理文本
3. **推理组合**：使用 `reasoning_composer.py` 生成高多样性的推理文本，引用真实传感器证据
4. **格式化**：转为 ShareGPT 多轮对话格式（system → user → assistant → tool → assistant → ...）

**最终数据规模**：
- SFT 训练集：3,000 条（2,700 train + 300 eval）
- RL 训练集：3,747 条（+ 468 val）

### 4.5 RL 数据格式

RL 数据仅包含 prompt（system + user），不包含 assistant 回复：

```json
{
  "id": "scenario_123",
  "messages": [
    {"role": "system", "content": "You are a diagnostic agent..."},
    {"role": "user", "content": "检测到 HVAC 系统异常..."}
  ],
  "ground_truth": {
    "root_cause_system": "rtu",
    "root_cause_node": "rtu::RTU",
    "fault_type": "undercharge_10",
    "optimal_path_length": 3
  }
}
```

## 5. 诊断工具体系

Agent 可调用 8 种工具与 Oracle 环境交互：

| 工具名称 | 输入 | 输出 | 用途 |
|:---|:---|:---|:---|
| `get_system_overview` | — | 系统列表 + 状态 | 全局视图 |
| `get_node_children` | `node_id` | 子节点列表 | 层次钻取 |
| `get_downstream_nodes` | `node_id` | 下游节点 | 影响分析 |
| `get_upstream_nodes` | `node_id` | 上游节点 | 根因追溯 |
| `get_node_sensors` | `node_id` | 传感器列表 + 值 | 数据观察 |
| `get_related_systems` | `system_id` | 关联系统 | 跨系统分析 |
| `diagnose_node` | `node_id` | 预测 + 置信度 | **核心诊断** |
| `get_node_status_summary` | `node_id` | 状态汇总 | 快速概览 |

工具调用格式：
```xml
<tool_call>
{"name": "diagnose_node", "arguments": {"node_id": "rtu::RTU"}}
</tool_call>
```

## 6. 训练方法

### 6.1 SFT（监督微调）

| 项目 | 配置 |
|:---|:---|
| 基座模型 | Qwen2.5-7B-Instruct |
| 微调方法 | LoRA (r=64, α=128, dropout=0.05) |
| 可训练参数 | 161M / 7.8B (2.08%) |
| 训练集 | 2,700 条多轮对话 |
| 验证集 | 300 条 |
| 训练轮次 | 3 epochs (255 steps) |
| 批大小 | 2 × 16 gradient accumulation |
| 学习率 | 2e-5, cosine scheduler, warmup 100 steps |
| 最大序列长度 | 6,144 tokens |
| 精度 | BF16 |
| 评估 | 每 50 步进行 Oracle 诊断评估（10 episodes） |

### 6.2 GRPO（强化学习）

| 项目 | 配置 |
|:---|:---|
| 起点模型 | SFT best checkpoint |
| 算法 | Group Relative Policy Optimization (GRPO) |
| 组大小 | 4 rollouts per prompt |
| KL 系数 | 0.05 |
| 内存优化 | 共享 base model + adapter toggling（省 ~14GB） |
| 训练步数 | 500 steps |
| 学习率 | 5e-6, cosine scheduler |
| 最大 rollout 步数 | 15 tool calls |
| 生成温度 | 0.8 |
| 评估 | 每 50 步进行 Oracle 诊断评估（30 episodes） |

**奖励函数**（加权组合）：

| 维度 | 权重 | 说明 |
|:---|:---:|:---|
| 诊断准确率 (DA) | 0.35 | 根因节点 + 故障类型正确 |
| 搜索效率 (SE) | 0.20 | 实际步数 / 最优步数 |
| 工具格式 (TFV) | 0.15 | JSON 格式合规 |
| 推理真实性 (RA) | 0.15 | 推理依据与工具结果一致 |
| 诊断完整度 (DC) | 0.15 | 关键诊断步骤覆盖 |

## 7. 评估指标

6 项指标综合评估智能体性能：

| 指标 | 缩写 | 定义 |
|:---|:---:|:---|
| 诊断准确率 | DA | 根因节点 + 故障类型均正确的比例 |
| 工具格式合规率 | TFV | 工具调用 JSON 格式正确的比例 |
| 诊断完整度 | DC | 关键诊断步骤的覆盖率 |
| 搜索效率 | SE | 最优路径长度 / 实际路径长度 |
| 推理真实性 | RA | 推理依据与工具返回结果一致的比例 |
| 工具调用合理性 | TIR | 合理工具调用占总调用的比例 |
| **综合得分** | **Agg** | **加权平均**（DA=0.35, SE=0.20, 其余各 0.15） |

## 8. 训练结果

### 8.1 SFT 训练曲线

| Step | DA | TFV | DC | SE | RA | TIR | Agg |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 50 | 0.0% | 100.0% | 0.0% | 19.0% | 83.6% | 88.4% | 0.437 |
| 100 | 50.0% | 100.0% | 66.7% | 25.7% | 79.5% | 95.9% | 0.677 |
| 150 | 60.0% | 100.0% | 66.7% | 24.6% | 87.0% | 94.6% | 0.709 |
| 200 | **80.0%** | 100.0% | 83.3% | 28.4% | 82.6% | 100.0% | **0.791** |
| 250 | **80.0%** | 100.0% | 83.3% | 28.2% | 81.6% | 100.0% | **0.790** |

- **最佳 checkpoint**: Step 200 (aggregate_score = 0.7914)
- **训练时长**: 4.1 小时（单卡 RTX PRO 6000 96GB）
- **最终 eval_loss**: 0.0799

### 8.2 RL 训练结果

> RL 训练进行中，结果待更新。

## 9. 项目结构

```
DiagAgent/
├── configs/                          # 配置文件
│   ├── topology_config.yaml          # 拓扑构建配置（系统、TTL路径、跨系统链接）
│   ├── diagnostic_paths.yaml         # 预设诊断路径
│   ├── sft_config.yaml               # SFT 训练配置
│   └── rl_config.yaml                # RL 训练配置
├── data/
│   └── lbnl/                         # LBNL-FDD 原始数据集（8 个子系统）
├── src/
│   ├── topology/                     # 拓扑模块
│   │   ├── ttl_parser.py             # Brick Schema TTL 解析器
│   │   ├── topology_builder.py       # 4 层 NetworkX 有向图构建
│   │   └── topology_tools.py         # 拓扑查询工具 API
│   ├── node_models/                  # Oracle 模型模块
│   │   ├── model_trainer.py          # XGBoost 节点模型训练
│   │   ├── model_registry.py         # 模型注册与管理
│   │   ├── prediction_tools.py       # diagnose_node 工具封装
│   │   └── data_loader.py            # LBNL CSV 数据加载
│   ├── environment/                  # 环境层
│   │   ├── tool_executor.py          # 统一工具执行器
│   │   ├── fault_scenario.py         # 故障场景定义与状态管理
│   │   └── diagnostic_path.py        # 诊断路径定义
│   ├── data_gen/                     # 数据生成管线
│   │   ├── trajectory_generator.py   # 路径引导轨迹生成
│   │   ├── reasoning_composer.py     # 推理文本组合生成
│   │   ├── scenario_sampler.py       # 分层场景采样
│   │   └── sft_formatter.py          # ShareGPT 格式化
│   ├── training/                     # 训练模块
│   │   ├── sft_trainer.py            # LoRA SFT 训练器
│   │   ├── rl_trainer.py             # GRPO RL 训练器
│   │   └── reward_functions.py       # 多维奖励函数
│   └── evaluation/                   # 评估模块
│       ├── metrics.py                # 6 项评估指标计算
│       ├── evaluator.py              # 端到端评估流程
│       └── report_generator.py       # Markdown 报告生成
├── scripts/                          # 执行脚本（按序号排列）
│   ├── 01_build_topology.py          # Step 1: 构建拓扑
│   ├── 02_train_node_models.py       # Step 2: 训练 Oracle 模型
│   ├── 03_generate_sft_data.py       # Step 3: 生成 SFT 数据
│   ├── 04_generate_rl_data.py        # Step 4: 生成 RL 数据
│   ├── 05_train_sft.py               # Step 5: SFT 训练
│   ├── 06_train_rl.py                # Step 6: RL 训练
│   └── 07_evaluate.py                # Step 7: 多模型评估
├── outputs/                          # 输出目录
│   ├── topology/                     # 拓扑图和元数据
│   ├── models/                       # 85 个 XGBoost 模型
│   ├── data/                         # SFT/RL 训练数据
│   ├── sft/                          # SFT checkpoints + eval history
│   ├── rl/                           # RL checkpoints + eval history
│   └── evaluation/                   # 评估报告
├── docs/                             # 文档
├── requirements.txt                  # 依赖列表
└── .gitignore
```

## 10. 快速开始

### 10.1 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 下载 LBNL-FDD 数据集到 data/lbnl/
```

### 10.2 全流程执行

```bash
# Step 1: 构建系统拓扑（约 10 秒）
python scripts/01_build_topology.py

# Step 2: 训练 Oracle 节点模型（约 30 分钟）
python scripts/02_train_node_models.py

# Step 3: 生成 SFT 训练数据（约 15 分钟）
python scripts/03_generate_sft_data.py

# Step 4: 生成 RL 训练数据（约 5 分钟）
python scripts/04_generate_rl_data.py

# Step 5: SFT 训练（约 4 小时，需 GPU）
python scripts/05_train_sft.py --config configs/sft_config.yaml

# Step 6: RL 训练（约 20+ 小时，需 GPU）
python scripts/06_train_rl.py --config configs/rl_config.yaml

# Step 7: 多模型评估
python scripts/07_evaluate.py --models base,sft,rl
```

### 10.3 硬件要求

| 阶段 | 最低 GPU | 推荐 GPU | 显存 |
|:---|:---|:---|:---:|
| 数据处理 / Oracle 训练 | CPU only | — | — |
| SFT 训练 | RTX 3090 24GB | A100 40GB | 24GB+ |
| RL 训练 | A100 80GB | RTX PRO 6000 96GB | 80GB+ |
| 评估 | RTX 3090 24GB | — | 24GB+ |

## 11. 技术亮点

1. **端到端 Agent 范式**：不同于传统 FDD 方法，本项目让 LLM 自主决定诊断策略，通过多轮工具调用追踪故障因果链
2. **真实 Oracle 环境**：RL 训练和评估中使用真实的 XGBoost 模型作为工具后端，而非模拟响应
3. **共享 Base Model 优化**：通过 adapter toggling 实现 policy 和 reference model 共享权重，节省 ~14GB 显存
4. **分层场景设计**：5 种场景类型覆盖从简单单系统故障到复杂跨系统传播的全部诊断难度
5. **多维奖励信号**：GRPO 使用 5 维加权奖励，同时优化准确率、效率、格式、推理和完整度

## License

MIT License
