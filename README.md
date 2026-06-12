# DiagAgent — 拓扑感知与证据增强的工业故障诊断智能体

DiagAgent 将复杂工业系统(以 LBNL HVAC 暖通数据集为载体)的故障诊断建模为**部分可观测的多步交互式决策过程**。智能体不能直接读取完整拓扑或真实根因,只能通过拓扑查询、上下游追踪、节点诊断、传感器读取等工具逐步获取局部观测,最终定位根因节点及故障类型。

方法包含两个阶段:

- **TopoITL**(Topology-Internalized Trajectory Learning):从专家诊断轨迹中,以「可见诊断状态 → 结构化动作 → 后继状态」三项联合损失监督微调策略,学习局部拓扑探索与工具调用。
- **EPO**(Evidence-Potential Optimization):在 TopoITL 策略上构建候选根因信念、证据状态机与证据势能函数,以**步级势能回报 + 终止正确性**做强化学习,学习在低置信/冲突证据下的验证与终止决策。

> 论文原稿见 `paper/`(中文)与 `paper_en/`(英文)。

---

## 目录结构

```
DiagAgent/
├── configs/                 # 拓扑/诊断路径/SFT/RL 配置
│   ├── topology_config.yaml
│   ├── diagnostic_paths.yaml
│   ├── sft_config.yaml
│   └── rl_config.yaml
├── data/lbnl/               # LBNL FDD 原始 CSV + TTL 拓扑(需自行放置)
├── scripts/                 # 端到端流水线(00–07)
├── src/
│   ├── topology/            # TTL 解析 + 拓扑图构建
│   ├── node_models/         # 节点级 LightGBM oracle + 温度校准
│   ├── data_gen/            # 场景采样 / 轨迹生成 / TopoITL 标注 / SFT·RL 格式化
│   ├── environment/         # 工具执行器 / 场景状态 / 诊断路径
│   ├── training/            # SFT 三项加权损失 / EPO 步级 RL / 奖励函数
│   ├── evaluation/          # 评估器 / 指标(DA·TR·ECR·SE)/ 证据融合
│   └── utils/
├── outputs/                 # 产物:topology / models / data / sft / rl / evaluation
├── paper/  paper_en/        # 论文源文件
└── requirements.txt
```

---

## 环境安装

```bash
pip install -r requirements.txt
```

- **数据生成 / 节点模型训练 / 评估**:仅需 CPU 依赖(numpy、pandas、scikit-learn、lightgbm、networkx、rdflib 等)。
- **SFT / RL 训练**:需 GPU 依赖(torch、transformers、peft、flash-attn),在 GPU 服务器安装。

将 LBNL FDD 数据集(8 类子系统的 CSV + TTL)放入 `data/lbnl/`。

---

## 端到端流水线

| 脚本 | 作用 | 运行环境 |
|---|---|---|
| `00_cache_lbnl_parquet.py` | 把原始 CSV 缓存为 Parquet(无损 IO 加速) | CPU |
| `01_build_topology.py` | 解析 9 个 TTL → 624 节点异构拓扑图 | CPU |
| `02_train_node_models.py` | 训练 8 个系统级 oracle(LightGBM)+ 验证集温度校准 | CPU |
| `03_generate_sft_data.py` | 生成专家轨迹级 SFT 数据(ShareGPT 格式) | CPU |
| `04_generate_rl_data.py` | 生成 RL prompt(仅起点 + GT,无轨迹) | CPU |
| `05_train_sft.py` | TopoITL 监督微调(LoRA,三项加权损失) | GPU |
| `06_train_rl.py` | EPO 强化学习(步级势能 + KL 约束) | GPU |
| `07_evaluate.py` | 统一评估 base / sft / rl(DA·TR·ECR·SE) | GPU |

### 数据生成(本地 CPU)

```bash
python scripts/00_cache_lbnl_parquet.py            # 一次性,加速后续 CSV 读取
python scripts/01_build_topology.py
python scripts/02_train_node_models.py --oracle-only
python scripts/03_generate_sft_data.py --n-total 3000
python scripts/04_generate_rl_data.py
# 质量校验
python scripts/validate_sft_data.py
python scripts/validate_rl_data.py
```

数据生成**纯规则/确定性**:工具观测来自真实 oracle 模型 + 真实 CSV,推理文本由模板库组合生成,**不依赖任何大模型**,因此零幻觉、可复现。

### 训练(GPU 服务器)

```bash
# 阶段一:SFT(多卡 DDP,flash-attn)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29501 \
  scripts/05_train_sft.py --config configs/sft_config.yaml

# 阶段二:EPO RL(单卡,初始化自 SFT best)
CUDA_VISIBLE_DEVICES=0 python scripts/06_train_rl.py \
  --config configs/rl_config.yaml --sft-checkpoint outputs/sft/best

# 阶段三:统一评估
CUDA_VISIBLE_DEVICES=0 python scripts/07_evaluate.py \
  --models vanilla,sft,rl --test-size 200 --eval-csv-nrows 50000
```

---

## 评估指标

| 指标 | 含义 |
|---|---|
| **DA** | 诊断准确率(根因节点 ∧ 故障类型均正确) |
| **TR** | 拓扑合理性(参考传播路径**边**的覆盖率) |
| **ECR** | 证据闭环性(结论有可靠支持证据且无冲突) |
| **SE** | 诊断效率(参考路径长度 / 实际工具调用数) |

任务分四类:单系统、跨系统、低置信、无故障。

---

## 关键配置

- **基础模型**:Qwen2.5-7B-Instruct + LoRA(rank 64 / alpha 128)
- **TopoITL**:三项加权损失 `topoitl_loss_weights`(state=λ_z / action=λ_act / transition=λ_tr / base),`configs/sft_config.yaml`
- **EPO**:`use_step_level_epo: true`、`lambda_epo: 0.75`(势能强度,敏感性主参)、`epo_discount`,`configs/rl_config.yaml`
- **TopoITL 标签布局**:action-first —— 可执行块在前,`<topoitl_state>`/`<topoitl_action>` 作为尾随的训练监督 token,推理时不输出(保证多轮 rollout 短而稳定)。

显存不足时,优先调小 `configs/sft_config.yaml` 的 `max_seq_length`(16384→8192),再降 `per_device_train_batch_size`(并相应提高 `gradient_accumulation_steps`)。

---

## 产物说明(`outputs/`)

- `topology/topology.json` — 拓扑图
- `models/<system>/<system>__oracle/` — 节点级 oracle + 校准温度
- `data/sft_train.jsonl`、`data/rl_{train,val,test}.jsonl`、`data/all_scenarios.json`
- `sft/best/` — SFT 最优 LoRA adapter(RL 初始化与 KL 参考)
- `rl/` — RL 检查点、rollout 日志、诊断评估历史
- `evaluation/` — 各模型评估结果 JSON + 逐 episode 轨迹

---

## 复现要点

1. 数据格式为 **action-first**:可执行块在前、TopoITL 标签尾随作监督。SFT 与 RL 共用同一场景池与真实 oracle 工具环境,保证训练-推理一致、无分布偏移。
2. 跨系统场景采用 oracle **软门控**(同系统非根因节点返回 `Abnormal` + `within_system` 方向提示,精确责任节点内部保留不泄漏)+ **下钻纠正**专家段,提升跨系统根因定位。
3. 节点模型输出经**温度校准**(验证集 NLL 拟合),为 EPO 证据势能提供可靠置信度。
