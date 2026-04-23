# DiagAgent — 基于大语言模型的工业故障诊断智能体

## 1. 项目背景

### 1.1 问题描述

现代建筑 HVAC（暖通空调）系统由多个高度耦合的子系统组成——冷水机组、锅炉、空气处理机组、风机盘管等。当故障发生时，其影响往往跨越多个子系统传播：一台冷水机组的性能退化可能导致下游空气处理机组的温度异常，再进一步影响末端风机盘管的舒适度。

传统的故障检测与诊断（FDD）方法依赖于：
- **基于规则的专家系统**：需要大量人工编写规则，维护成本高，无法处理复杂的跨系统故障传播
- **单点预测模型**：只能判断单个设备是否异常，无法自主追踪故障的因果链
- **人工逐一排查**：效率低下，依赖维护人员的经验和对系统拓扑的理解

### 1.2 解决思路

本项目构建一个**基于 Qwen2.5-7B 大语言模型的诊断智能体（Diagnostic Agent）**，使其具备：
1. 理解建筑 HVAC 系统拓扑结构的能力
2. 调用诊断工具（Oracle 预测模型）获取设备状态的能力
3. 基于工具返回结果进行多步推理、追踪故障因果链的能力
4. 最终定位根因并输出结构化诊断结论的能力

### 1.3 数据来源

项目使用 **LBNL-FDD（Lawrence Berkeley National Laboratory Fault Detection and Diagnostics）数据集**，包含 8 个 HVAC 子系统的完整模拟数据：

| 子系统 | 简称 | 故障类型数 | 数据文件数 |
|:---|:---:|:---:|:---:|
| 冷水机组 (Chiller Plant) | chiller_plant | 23 | 24 |
| 锅炉 (Boiler Plant) | boiler_plant | 16 | 17 |
| 单风道空气处理机组 | sdahu | 20 | 21 |
| 双风道空气处理机组 | ddahu | 55 | 56 |
| 屋顶式空调 (RTU) | rtu | 24 | 25 |
| 风机盘管 (FCU) | fcu | 48 | 49 |
| 主风量末端 (PFPU) | pfpu | 30 | 31 |
| 从风量末端 (SFPU) | sfpu | 30 | 31 |

每个子系统包含正常运行数据和多种故障注入数据（传感器偏差、阀门卡死、盘管结垢、泄漏等），每个文件 50,000 行时间步。

---

## 2. 项目目标

### 2.1 核心目标

训练一个端到端的故障诊断智能体，能够：

```
用户: "检测到楼宇 HVAC 系统异常，请诊断故障原因"
Agent: [调用拓扑工具] → [选择疑似系统] → [调用诊断工具] → 
       [分析结果] → [追踪因果链] → [定位根因] → 输出诊断结论
```

### 2.2 量化指标

通过 6 项指标评估智能体性能：

| 指标 | 缩写 | 定义 |
|:---|:---:|:---|
| 诊断准确率 | DA | 根因节点 + 故障类型均正确的比例 |
| 工具格式合规率 | TFV | 工具调用 JSON 格式正确的比例 |
| 诊断完整度 | DC | 关键诊断步骤的覆盖率 |
| 搜索效率 | SE | 实际路径长度 / 最优路径长度 |
| 推理真实性 | RA | 推理依据与工具返回结果一致的比例 |
| 工具调用合理性 | TIR | 合理工具调用占总调用的比例 |

### 2.3 训练策略

采用两阶段训练：

1. **SFT（监督微调）**：教会模型正确的工具调用格式和基本诊断流程
2. **RL（强化学习）**：在真实 Oracle 环境中优化诊断准确率和搜索效率

---

## 3. 整体架构

### 3.1 系统架构图

```
┌──────────────────────────────────────────────────────────┐
│                      DiagAgent 系统                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────┐  │
│  │  Topology    │    │  Node Models │    │   Data Gen  │  │
│  │  Builder     │    │  (Oracle)    │    │  Pipeline   │  │
│  │             │    │              │    │             │  │
│  │ • TTL解析    │    │ • 84个XGBoost│    │ • 场景采样  │  │
│  │ • 4层拓扑图  │    │ • 节点级预测  │    │ • 轨迹生成  │  │
│  │ • 跨系统链接 │    │ • 置信度输出  │    │ • SFT格式化 │  │
│  └──────┬──────┘    └──────┬───────┘    └──────┬──────┘  │
│         │                  │                    │         │
│         ▼                  ▼                    ▼         │
│  ┌──────────────────────────────────────────────────┐    │
│  │              Environment Layer                    │    │
│  │  ┌────────────────┐  ┌─────────────────────────┐ │    │
│  │  │ UnifiedTool     │  │ PredictionToolExecutor  │ │    │
│  │  │ Executor        │  │ (Real Oracle)           │ │    │
│  │  └────────────────┘  └─────────────────────────┘ │    │
│  └──────────────────────────┬───────────────────────┘    │
│                             │                            │
│         ┌───────────────────┼───────────────────┐        │
│         ▼                   ▼                   ▼        │
│  ┌────────────┐    ┌──────────────┐    ┌────────────┐   │
│  │  SFT       │    │  RL          │    │ Evaluation  │   │
│  │  Trainer   │    │  Trainer     │    │ Framework   │   │
│  │            │    │              │    │             │   │
│  │ • LoRA 64  │    │ • GRPO       │    │ • 6项指标   │   │
│  │ • 3000条   │    │ • Oracle环境 │    │ • 按类型/   │   │
│  │ • 步级评估 │    │ • 多维奖励   │    │   难度分析  │   │
│  └────────────┘    └──────────────┘    └────────────┘   │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 3.2 模块说明

#### 拓扑模块 (`src/topology/`)

| 文件 | 功能 |
|:---|:---|
| `ttl_parser.py` | 解析 Brick Schema TTL 文件，提取设备、传感器、关系 |
| `topology_builder.py` | 构建 4 层 NetworkX 有向图（Building → System → Node → Sensor） |
| `topology_tools.py` | 供 Agent 调用的拓扑查询 API（get_system_overview, get_node_children 等） |

#### Oracle 模型 (`src/node_models/`)

| 文件 | 功能 |
|:---|:---|
| `model_registry.py` | 管理 84 个 XGBoost 节点模型的注册、加载和查询 |
| `prediction_tools.py` | 封装诊断工具（diagnose_node），接收 node_id，返回状态+置信度+传感器数据 |
| `data_loader.py` | 加载 LBNL-FDD CSV 数据，提取特征 |

#### 环境层 (`src/environment/`)

| 文件 | 功能 |
|:---|:---|
| `tool_executor.py` | 统一工具执行器，路由 6 种工具调用到对应处理器 |
| `fault_scenario.py` | 故障场景定义和管理（含跨系统场景、无故障场景） |
| `diagnostic_path.py` | 预设诊断路径（从症状到根因的最优路径） |

#### 数据生成 (`src/data_gen/`)

| 文件 | 功能 |
|:---|:---|
| `trajectory_generator.py` | 路径引导的多轮诊断轨迹生成（核心） |
| `reasoning_composer.py` | 组合式高多样性推理文本生成，含传感器证据引用 |
| `scenario_sampler.py` | 分层场景采样（按类型、系统、难度均衡） |
| `sft_formatter.py` | 轨迹 → ShareGPT 格式转换 |

#### 训练 (`src/training/`)

| 文件 | 功能 |
|:---|:---|
| `sft_trainer.py` | LoRA SFT 训练器，含 DiagnosticEvalCallback（步级 Oracle 评估） |

#### 评估 (`src/evaluation/`)

| 文件 | 功能 |
|:---|:---|
| `metrics.py` | 6 项量化评估指标计算 |
| `evaluator.py` | 端到端评估流程（加载模型 → Oracle 环境 → 多轮对话 → 指标计算） |
| `report_generator.py` | 生成 Markdown 评估报告 |

---

## 4. 技术方案细节

### 4.1 拓扑构建

**4 层层次结构**：
```
Level 0: Building (根节点)
  └── Level 1: Systems (8个子系统)
        └── Level 2: Nodes/Components (设备/部件)
              └── Level 3: Sensors (传感器数据点)
```

**跨系统依赖关系**：
- Chiller Plant → AHU / FCU（供冷水）
- Boiler Plant → AHU / FCU（供热水）
- AHU → PFPU / SFPU（供空气）

这些跨系统边使 Agent 能够追踪故障在系统间的传播路径。

### 4.2 Oracle 模型

为拓扑中每个 Level-2 节点训练独立的 XGBoost 分类器：

- **输入**：该节点关联传感器的时序特征（当前值 + 滑动窗口统计量）
- **输出**：多分类结果（Normal / Fault_Type_1 / Fault_Type_2 / ...）
- **总计**：84 个模型，覆盖所有可诊断节点
- **作用**：作为 `diagnose_node` 工具的后端，为 Agent 提供真实的基于传感器数据的诊断结果

### 4.3 诊断工具体系

Agent 可调用 6 种工具：

| 工具名称 | 输入 | 输出 | 用途 |
|:---|:---|:---|:---|
| `get_system_overview` | 无 | 所有子系统列表 | 了解全局拓扑 |
| `get_node_children` | node_id | 子节点列表 | 展开系统/节点 |
| `get_upstream_nodes` | node_id | 上游节点列表 | 追踪故障来源 |
| `get_downstream_nodes` | node_id | 下游节点列表 | 验证故障影响范围 |
| `diagnose_node` | node_id | 状态+置信度+传感器 | **核心诊断** |
| `get_node_info` | node_id | 节点详细属性 | 获取设备信息 |

### 4.4 SFT 数据生成

#### 场景类型

| 类型 | 数量 | 说明 |
|:---|:---:|:---|
| `single_system` | 1,200 | 单系统内故障，2-3 步诊断路径 |
| `cross_system` | 750 | 跨系统故障传播，4-6 步路径 |
| `no_fault` | 300 | 无故障（所有节点 Normal），排除误报 |
| `ambiguous` | 300 | 模糊场景，多个节点异常 |
| `low_confidence` | 450 | 低置信度场景，需更多证据 |
| **总计** | **3,000** | |

#### 轨迹生成流程

```
1. 故障场景选择 → 确定根因节点和诊断路径
2. Phase 1: 系统提示 → Agent 角色定义
3. Phase 2: 全局拓扑查询 → get_system_overview
4. Phase 3: 选择起始系统 → 基于症状描述
5. Phase 4: 展开系统组件 → get_node_children
6. Phase 5: 逐节点诊断 → diagnose_node（沿路径）
7. Phase 6: 跨系统追踪 → get_upstream_nodes（如需）
8. Phase 7: 排除法 → 诊断非路径节点（增加真实性）
9. Phase 8: 下游验证 → get_downstream_nodes
10. Phase 9: 输出结论 → <diagnosis> 结构化 JSON
```

#### 推理多样性设计

采用**组合式推理生成**（`reasoning_composer.py`），避免模板过拟合：
- 10 种初始推理开头 × 8 种规划表述 = 80 种首轮组合
- 每种诊断状态（Normal / Abnormal / Fault）8+ 种观察描述 × 8 种推理逻辑 × 6 种下一步计划
- **60-70% 概率**注入真实传感器数据作为证据引用
- 最终唯一率 **33.8%**（11,642/34,485 唯一推理块）

#### 数据质量保障

| 检查维度 | 结果 |
|:---|:---:|
| 幻觉（编造传感器/数值） | 0 条 |
| 工具输出与推理矛盾 | 0 条 |
| 推理引用证据 | 90.6% |
| 推理唯一率 | 33.8% |
| 信息泄漏（提前暴露答案） | 0 条 |
| 偷懒跳步（跳过诊断直接猜测） | 0 条 |

### 4.5 SFT 训练配置

| 参数 | 值 | 说明 |
|:---|:---:|:---|
| 基座模型 | Qwen2.5-7B-Instruct | |
| 微调方式 | LoRA (rank=64, alpha=128) | 仅调 ~2% 参数 |
| 目标模块 | q/k/v/o/gate/up/down_proj | 全注意力+FFN |
| 数据格式 | ShareGPT | Qwen 原生兼容 |
| 训练数据 | 3,000 条 (2,700 训 / 300 验) | |
| 最大序列长度 | 6,144 tokens | 覆盖长诊断轨迹 |
| Batch Size | 2 × 16 (accum) = 32 | 适配 95GB VRAM |
| 学习率 | 2e-5, cosine schedule | |
| 训练轮次 | 3 epochs (~255 步) | |
| 评估策略 | 每 50 步 eval_loss + 6 项诊断指标 | |

### 4.6 步级诊断评估

训练过程中每 50 步自动：
1. 计算验证集 `eval_loss`（HF Trainer 内置）
2. 加载当前模型到真实 Oracle 环境
3. 运行 50 个诊断 episode
4. 输出全部 6 项诊断指标
5. 保存到 `diag_eval_history.json`

这确保了不仅监控损失下降，还能实时追踪诊断能力的真实变化。

---

## 5. 项目结构

```
DiagAgent/
├── configs/
│   ├── sft_config.yaml         # SFT 训练配置
│   ├── topology_config.yaml    # 拓扑构建配置
│   └── path_config.yaml        # 诊断路径配置
├── data/
│   └── lbnl_fdd/               # LBNL-FDD 原始数据
│       ├── chiller_plant/
│       ├── boiler_plant/
│       └── ...
├── docs/
│   ├── project_documentation.md  # 本文档
│   ├── topology_diagrams.md      # 拓扑关系图
│   └── diagnostic_flow_examples.md # 诊断流程示例
├── outputs/
│   ├── topology/               # 构建的拓扑图
│   ├── node_models/            # 84个Oracle模型
│   ├── data/
│   │   ├── sft_train.jsonl     # SFT训练数据 (3,000条)
│   │   └── eval_test.jsonl     # 评估测试数据
│   ├── sft/                    # SFT训练输出
│   │   ├── best/               # 最佳checkpoint
│   │   ├── diag_eval_history.json  # 步级评估记录
│   │   └── training_summary.json
│   └── evaluation/             # 评估结果
├── scripts/
│   ├── 01_build_topology.py    # 步骤1: 构建拓扑
│   ├── 02_train_node_models.py # 步骤2: 训练Oracle
│   ├── 03_generate_sft_data.py # 步骤3: 生成SFT数据
│   ├── 05_train_sft.py         # 步骤5: SFT训练
│   ├── 06_train_rl.py          # 步骤6: RL训练
│   └── 07_evaluate.py          # 步骤7: 统一评估
└── src/
    ├── topology/               # 拓扑构建与查询
    ├── node_models/            # Oracle预测模型
    ├── environment/            # 工具执行环境
    ├── data_gen/               # 数据生成管线
    ├── training/               # SFT/RL训练器
    └── evaluation/             # 评估框架
```

---

## 6. 执行流程

### 完整管线

```bash
# 1. 构建HVAC拓扑图
python scripts/01_build_topology.py

# 2. 训练84个节点级Oracle模型
python scripts/02_train_node_models.py

# 3. 生成3000条SFT训练数据
python scripts/03_generate_sft_data.py --n-total 3000

# 4. SFT微调 (GPU服务器)
python scripts/05_train_sft.py --config configs/sft_config.yaml

# 5. 统一评估
python scripts/07_evaluate.py --models base,sft --prepare-test-data
```

---

## 7. 当前进展

### 已完成 ✅

| 阶段 | 状态 | 说明 |
|:---|:---:|:---|
| 拓扑构建 | ✅ | 8 系统、84 节点、跨系统依赖关系完整 |
| Oracle 训练 | ✅ | 84 个 XGBoost 模型全部训练完成并验证 |
| SFT 数据生成 | ✅ | 3,000 条高质量训练数据，7 维审计通过 |
| 数据质量审计 | ✅ | 幻觉/泄漏/偷懒 = 0，唯一率 33.8%，证据引用 90.6% |
| 训练管线 | ✅ | SFT Trainer + DiagnosticEvalCallback 实现完毕 |
| 评估框架 | ✅ | 6 项指标 + Oracle 环境评估完整实现 |

### 进行中 🔄

| 阶段 | 状态 | 说明 |
|:---|:---:|:---|
| SFT 训练 | 🔄 | GPU 服务器上执行中（~255 步，3 epochs） |

### 待完成 📋

| 阶段 | 依赖 | 说明 |
|:---|:---|:---|
| SFT 模型评估 | SFT 训练完成 | Base vs SFT 对比评估 |
| RL 数据准备 | SFT 评估通过 | 基于 SFT 模型生成 RL 探索数据 |
| RL 训练 (GRPO) | RL 数据 | 在 Oracle 环境中强化学习优化 |
| 最终评估 | RL 训练完成 | Base vs SFT vs RL 三方对比 |

### 关键设计决策

1. **3,000 vs 15,000 训练数据**：7B LoRA SFT 的最佳数据区间为 3K-5K，过多数据会导致对模板格式过拟合
2. **步级诊断评估**：不仅监控 eval_loss，每 50 步运行完整 Oracle 环境评估，确保诊断能力实时可追踪
3. **组合式推理生成**：用 `reasoning_composer.py` 替代固定模板，唯一率从 1.8% 提升到 33.8%
4. **传感器证据注入**：最终诊断中 90.6% 的推理引用了具体传感器数据，避免"知道答案就直接说"的模式
