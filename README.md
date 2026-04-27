# DiagAgent

基于 Qwen2.5-7B-Instruct 的工业 HVAC 故障定位与诊断智能体。项目面向 LBNL-FDD 数据集，结合系统拓扑、Oracle 预测模型、SFT 轨迹学习和 GRPO 强化学习，使智能体能够根据用户故障描述自主选择诊断路径、调用工具、追踪跨系统传播并输出结构化根因诊断。

## 当前状态

本地环境负责代码、数据、拓扑、Oracle 和训练前校验；全量 SFT 与 RL 训练在云端 GPU 环境执行。

当前已完成：

- 拓扑构建：624 nodes，671 edges，8 个系统，129 个组件，486 个传感器，10 条跨系统边。
- Oracle 模型：85 个 LightGBM 模型，其中 8 个系统级 Oracle；训练和评估优先使用系统级 Oracle 路由。
- SFT 数据：3000 条 ShareGPT 多轮工具轨迹，审计结果 3000/3000 clean。
- RL 数据：train/val/test = 5259/657/658。
- 工具覆盖：`get_node_status_summary`、`get_related_systems`、`get_node_sensors` 已进入 SFT 轨迹。
- 中间评估：SFT/RL eval step 使用按场景比例的 stratified sample，不做全量评估以节约时间。
- RL preflight：SFT checkpoint、RL 数据、课程学习、奖励函数、Oracle 环境检查均通过。

仍需注意：

- per-node fallback 模型较弱，训练和评估应继续优先使用 system Oracle。
- RL 历史结果需要在新一轮云端训练后更新。

## 数据与场景

项目使用 LBNL-FDD HVAC 数据集，覆盖以下 8 个子系统：

| 系统 | ID |
|---|---|
| Chiller Plant | `chiller_plant` |
| Boiler Plant | `boiler_plant` |
| Single-Duct AHU | `sdahu` |
| Dual-Duct AHU | `ddahu` |
| Rooftop Unit | `rtu` |
| Fan Coil Unit | `fcu` |
| Parallel Fan-Powered Unit | `pfpu` |
| Series Fan-Powered Unit | `sfpu` |

场景库规模为 6574，包含单系统故障、跨系统故障、低置信故障、无故障和 ambiguous prompt 等类型。SFT 数据由真实拓扑和真实 Oracle 输出驱动生成；评估和 RL rollout 均通过 Oracle 环境真实调用工具。

## Agent 工具

智能体通过统一工具执行器调用 8 类工具：

| 工具 | 用途 |
|---|---|
| `get_system_overview` | 查看系统级健康状态和异常分数 |
| `get_node_children` | 展开系统或组件的层级结构 |
| `get_downstream_nodes` | 追踪下游影响 |
| `get_upstream_nodes` | 追踪上游根因 |
| `get_node_sensors` | 查看组件关联传感器 |
| `get_related_systems` | 查询跨系统连接 |
| `diagnose_node` | 调用 Oracle 判断节点状态 |
| `get_node_status_summary` | 快速汇总系统内组件状态 |

工具调用格式：

```xml
<tool_call>{"name": "diagnose_node", "arguments": {"node_id": "rtu::RTU"}}</tool_call>
```

最终诊断格式：

```xml
<diagnosis>{"root_cause_node": "rtu::RTU", "fault_type": "undercharge_20", "confidence": 0.66, "affected_systems": ["rtu"]}</diagnosis>
```

## 本地准备与校验

本地不跑全量 LLM 训练，只做数据、拓扑、Oracle 和脚本校验。

```bash
pip install -r requirements.txt

python scripts/01_build_topology.py
python scripts/02_train_node_models.py
python scripts/03_generate_sft_data.py --validate
python scripts/04_generate_rl_data.py

python scripts/local_training_readiness.py --require-oracle-runtime
python scripts/_rl_preflight.py
```

训练脚本语法检查：

```bash
python -m py_compile \
  scripts/05_train_sft.py scripts/06_train_rl.py scripts/07_evaluate.py \
  src/training/sft_trainer.py src/training/rl_trainer.py \
  src/evaluation/evaluator.py src/evaluation/metrics.py \
  src/environment/oracle_env.py src/environment/tool_executor.py
```

## 云端训练流程

建议云端先重新 SFT，再使用新的 `outputs/sft/best` 启动 RL。Oracle 当前不需要重训。

```bash
cd ~/autodl-tmp/workspace
mkdir -p logs

python scripts/local_training_readiness.py --require-oracle-runtime \
  2>&1 | tee logs/00_readiness_before.log

if [ -d outputs/sft ]; then
  mv outputs/sft outputs/sft_backup_$(date +%Y%m%d_%H%M%S)
fi
mkdir -p outputs/sft

python scripts/05_train_sft.py \
  --config configs/sft_config.yaml \
  --data-path outputs/data/sft_train.jsonl \
  --output-dir outputs/sft \
  2>&1 | tee logs/01_sft_train.log

python scripts/local_training_readiness.py --require-oracle-runtime \
  2>&1 | tee logs/02_readiness_after_sft.log

python scripts/_rl_preflight.py \
  2>&1 | tee logs/03_rl_preflight.log

python scripts/06_train_rl.py \
  --config configs/rl_config.yaml \
  --sft-checkpoint outputs/sft/best \
  --output-dir outputs/rl \
  2>&1 | tee logs/04_rl_train.log
```

最终离线评估：

```bash
python scripts/07_evaluate.py \
  --models base,sft,rl \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --sft-model outputs/sft/best \
  --rl-model outputs/rl/best \
  --test-data outputs/data/rl_test.jsonl \
  --output-dir outputs/eval \
  --test-size 200 \
  --sampling-strategy stratified \
  --max-steps 15 \
  2>&1 | tee logs/05_final_eval.log
```

## 中间评估策略

为节约云端训练时间，每个 eval step 不进行全量诊断评估，而是按 `scenario_type` 比例分层采样。

SFT 配置：

```yaml
diag_eval_episodes: 64
diag_eval_sampling: "stratified"
diag_eval_seed: 42
diag_eval_max_steps: 15
```

RL 配置：

```yaml
diag_eval_episodes: 30
diag_eval_sampling: "stratified"
diag_eval_seed: 42
max_rollout_steps: 15
```

实现要点：

- 采样按场景类型比例分配，避免只取文件前 N 条造成偏置。
- 每个 eval step 使用 `diag_eval_seed + step`，使样本随训练进度轮换。
- 最终论文/报告用评估可通过 `scripts/07_evaluate.py --test-size` 指定更大样本。

## 主要目录

```text
DiagAgent/
  configs/                    # 拓扑、SFT、RL 配置
  data/lbnl/                   # LBNL-FDD 原始数据，本地下载，不提交
  docs/                        # 项目说明与训练准备文档
  outputs/
    data/                      # SFT/RL 训练数据
    models/                    # LightGBM Oracle 模型
    sft/                       # SFT checkpoint 和评估记录
    rl/                        # RL checkpoint、评估记录和 episodes
  scripts/                     # 构建、生成、训练、评估脚本
  src/
    topology/                  # TTL 解析和拓扑工具
    node_models/               # Oracle 模型和 diagnose_node 封装
    environment/               # 场景状态和统一工具执行器
    data_gen/                  # 场景采样、轨迹生成、格式化
    training/                  # SFT 和 GRPO 训练
    evaluation/                # Oracle 评估和指标
```

`web-viz/` 是本地可视化/临时 Web 工程目录，不作为训练与论文复现实验的必要工件提交。

## 评估指标

| 指标 | 含义 |
|---|---|
| Diagnostic Accuracy | 根因节点和故障类型是否正确 |
| Tool Format Validity | 工具调用 JSON/XML 格式是否有效 |
| Diagnosis Completeness | 是否输出完整结构化诊断 |
| Search Efficiency | 实际工具步数相对最优路径的效率 |
| Reasoning Alignment | 推理文本是否与工具结果一致 |
| Tool Invocation Rationality | 工具选择是否符合拓扑与诊断流程 |

## License

MIT License
