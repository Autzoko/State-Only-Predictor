# State-Only Pretraining (SOPT): 研究设计

> 只用机械臂本体状态序列（joint / EE pose / gripper），不用视觉和语言，通过自监督预训练获得**动作/运动先验**（motion prior），再迁移到下游视觉运动策略。

## 1. 核心问题与假设

**问题**：大规模机器人数据集（DROID 等）里，本体状态序列只占数据量的约 0.5%（DROID：state 列约 1.4 GB，视频约 400 GB），却记录了全部"动作是怎么做出来的"信息。只用这部分数据能学到什么？学到的东西对下游策略有没有用？

**假设**
- **H1（可学性）**：state-only 模型在长时域（≥1 s）上的未来轨迹预测，明显好于运动学外推（保持静止 / 匀速外推），说明它学到的不只是平滑性，还有操作的时序结构（reach → grasp → lift → transport → release）。
- **H2（多模态）**：没有视觉时，未来本质上是多模态的（不知道目标在哪），所以生成式预测头（flow matching）在 minADE@K / NLL 上应明显好于 L2 回归，而 L2 回归会退化成"平均轨迹"。
- **H3（迁移）**：预训练出的轨迹 Transformer 拿来初始化策略的动作解码器 / 本体编码器，能在少量示范（LIBERO / RoboMimic，每任务 5–50 条）下提高成功率和样本效率。
- **H4（规模）**：数据量和模型大小的 scaling 曲线在 state-only 场景下有可测量的趋势，而且极其便宜（单卡几小时）。

**能学 / 不能学**
- 能学：运动学可行性（关节限位、可操作度）、速度/加速度剖面、操作阶段的时序结构、夹爪与手臂的协同（接近时减速 → 闭合 → 抬起）、遥操作风格。
- 不能学：目标的空间位置、任务语义。所以先验的定位是 **p(未来动作 | 本体历史)**，下游再用视觉/语言去 condition。

## 2. 主要风险与对策

| 风险 | 表现 | 对策 |
|---|---|---|
| 预测任务过于平凡 | 短 horizon 下匀速外推已经很准 | horizon 设为 16 步（约 1.07 s）及以上；**每个指标都和 zero-velocity / constant-velocity 基线对比**；按阶段（gripper 事件附近）分别报告 |
| 多模态被平均掉 | L2 回归的预测迟疑、走中间 | flow-matching 头；评估时报告 minADE@K、样本多样性、gripper 事件时刻误差 |
| Copycat / 因果混淆（Wen et al., 2020） | 下游策略过度依赖本体历史，闭环时出问题 | 下游消融"历史长度 × 是否预训练"；在先验和视觉之间加 history dropout；**一定要用闭环成功率评估，不能只看离线 loss** |
| 数据泄露 | 同一场景/同一操作员同时出现在 train 和 test | 按 **site** 整体切分：把规范化后的 building 和 collector 连成二部图，取连通分量作为 site（`building` 是自由文本，同一地点有多种拼写）。val/test 都是整个 OOD 实验室，另设 1% 的 id_val 作为分布内参考 |
| 遥操作偏置 | DROID 是 VR 遥操作、15 Hz | 跨数据集零样本评估（LIBERO、RoboMimic 为 Franka 脚本/人类示范）；时间缩放增广 |
| 跨本体 | 不同机器人的关节空间不通用 | v1 只做 Franka（DROID）。v2 再用 EE-only 表示（pos + rot6d + gripper）接入 Bridge/Fractal 等，关节作为可选 token |

## 3. 数据：DROID（已核实）

来源：`cadene/droid_1.0.1_v30`（LeRobot v3.0，固定在 421fe53；共 95,584 条 episode、2048 个数据文件）。**只读取 state 列和元数据列**：通过 parquet 列投影（HTTP range 请求）只拉取每个文件约 21% 的字节，总共约 1.4 GB，视频和其他列都不下载，原始文件也不落盘。下面的 schema 是在 v2.1 版（`cadene/droid_1.0.1`）上核实的，两版 episode 0 的数值完全一致。

- 95,600 条 episode，27,612,581 帧，15 Hz；长度均值 289，中位数 222，p5/p95 = 89/737。
- 状态字段：`observation.state.joint_position` (7)、`observation.state.cartesian_position` (6, xyz + 欧拉角)、`observation.state.gripper_position` (1, 0–1)。
- **已核实**：`cartesian_position` 的欧拉角是 **extrinsic XYZ**，即 `R = Rz(yaw)·Ry(pitch)·Rx(roll)`，与 scipy `"xyz"` 一致。它**精确等于** Franka FK 算出的 flange 位姿（d=0.107，误差 0）。所以 EE 位姿是关节的确定性函数，roll 在 ±π 附近会跳变，必须转成 rot6d。
- `action.joint_position` 是指令目标，不是下一帧状态（`|q_{t+1} - a_t|` ≈ 4× 单步状态变化），state-only 主线不使用 action 列，只存下来用于后续分析。
- **元数据列有错位**：`task_category` 与 `building` 相同、`date` 与 `collector_id` 相同；约 22% 的 `language_instruction` 为空。`is_episode_successful` 可用于失败检测实验，成功率为 82.4%。
- v3.0 版的 `episode_index` 88905 同时标记了两段数据，其中一段是 59 帧的 BAIR 残片。去掉这段之后，帧数正好等于 `info.json` 中的 27,607,757。
- **切分（固定协议）**：74 个 `building` 字符串合并成 15 个 site。最大的一个 site（Stanford/Berkeley/TRI 集群，共享操作员）约占 73%，只用于训练。
  - val = UW（CSE2*、Gates G60*、Smith Hall），约 4%；
  - test = UT Austin AHG* + Edinburgh Bayes* + Penn，约 7.5%；
  - id_val = 训练 site 中按 hash 取 1%。
  - 具体数字见 `SOPT_DATA/processed/droid/summary.json`。

## 4. 表示

每帧 17 维特征：`[q(7), gripper(1), ee_pos(3), ee_rot6d(6)]`（`src/sopt/data/features.py`）。

- **归一化**放在模型内部（buffer），统计量只用训练集计算，所以 checkpoint 是自包含的。
- **预测目标**：未来 H 步相对锚点（当前帧）的增量 `x_{t+k} - x_t`，在归一化空间中计算。
- **增广**
  - **Yaw 旋转（精确）**：Franka 的 joint 1 轴就是基座 z 轴，所以 `q1 += θ` 与 `ee ← Rz(θ)·ee` 在物理上完全一致（除了关节限位）。这是一个"免费"的、符合运动学的数据增广。
  - **时间缩放**：0.8–1.25× 重采样（线性插值），模拟不同操作速度。
  - 小幅高斯噪声（仅加在输入上，不加在目标上）。

## 5. 模型

```
state window (T_ctx + H steps, 17-d)
   │ normalize (in-model buffers)
   ▼
PatchEmbed: 每 P=4 步 → 1 token        (≈0.27 s/token)
   ▼
Transformer (pre-LN, RoPE, causal 或 bidirectional)
   ▼
每个 token 的输出 h_i ──► Head: 预测锚点之后 H 步的增量 chunk
                          ├─ RegressionHead (L1/L2)
                          └─ FlowHead (rectified flow, 条件 MLP, 10 步 Euler 采样)
```

**目标函数（`model.objective`）**
- `ar_regression`：causal，所有 token 并行预测各自的未来 chunk（GPT 式稠密监督），用于基线。
- `ar_flow`（**主方法**）：同上，但预测头是 flow matching，可以采样多条未来。
- `masked`（MTM 式）：bidirectional，随机 span mask + "未来 mask"（以概率 p 只 mask 最后若干 patch，使其同时具备预测能力），重建被 mask 的 patch（目标同样是相对"其前最近可见帧"的增量，与 AR 目标尺度一致，保证对比公平）。

**模型规模**（训练集 23.1M 帧 / 4 ≈ 5.8M token/epoch，过拟合是主要风险）

| 名字 | d_model | layers | heads | 参数量 |
|---|---|---|---|---|
| S | 256 | 6 | 4 | 9.5M（含 flow 头） |
| B | 512 | 12 | 8 | 56M（含 flow 头） |
| L | 768 | 16 | 12 | ~115M |

**后续可做（v2）**：离散 token（FAST DCT+BPE / VQ）+ CE 损失；JEPA 式隐空间预测；跨本体 EE-only stem。

## 6. 实验计划（逐步进行，每一步根据结果决定下一步）

| ID | 内容 | 关键指标 | 机器 |
|---|---|---|---|
| E0 | 数据管线：下载 parquet → `prepare_droid.py` → 统计量；按 building 切分 | 帧数、切分比例、特征分布 | 127 |
| E1 | 目标函数对比：zero-vel / const-vel / `ar_regression` / `ar_flow` / `masked`，S 和 B 两种规模 | held-out building 上的 ADE/FDE（EE 位置单位 m，关节单位 rad）、minADE@10、gripper MAE | 127 |
| E2 | Scaling：数据 {1, 10, 100}% × 模型 {S, B, L} | val loss / minADE 曲线 | Jubail（≤2 个作业并发） |
| E3 | 表征探测（冻结 + 线性 probe）：成功/失败（`is_episode_successful`）、gripper 事件预测、操作阶段、building/操作员 ID（作为负面对照：不应主导） | AUROC / acc | 127 |
| E4 | **下游迁移（核心）**：LIBERO / RoboMimic 少样本视觉运动策略，用预训练 Transformer 初始化 action decoder，对比 scratch / frozen / finetune；示范数 {5, 10, 50}/任务 | **闭环成功率**、学习曲线 | Jubail（A100，EGL rollout） |
| E5 | 先验作为 flow 策略的噪声源 / 引导：从先验样本出发去噪，代替高斯噪声 | 成功率 vs 去噪步数 | Jubail |
| E6 | 零样本跨数据集预测：在 LIBERO / RoboMimic 的状态轨迹上做预测 | ADE vs 基线 | 127 |

**判停标准**：如果 E1 中 `ar_flow` 在 1 s horizon 上相比 const-vel 的 minADE 降幅小于 30%，说明先验太弱，先改表示/horizon，不进入 E4。

## 7. 相关工作（需要逐一对比定位）
- Masked Trajectory Models (MTM, Wu et al. 2023)：bidirectional 状态-动作 mask 建模（含 RL）。
- HPT (Wang et al. 2024)：异构本体 stem + 共享 trunk，但用了视觉。
- FAST (Pertsch et al. 2025)、PRISE、QueST、VQ-BeT：动作 token 化 / 技能抽象。
- Diffusion Policy、ACT、π0：action chunk + 生成式头。
- 人体运动先验：HuMoR、MDM、MotionGPT，它们是"只有运动、没有场景"预训练的类比。
- Copycat problem（Wen et al. 2020）：本体历史导致的因果混淆。

**定位**：已有工作都把本体状态当成视觉策略的附属输入。SOPT 主张"**运动先验可以和感知解耦、单独大规模预训练**"，而且代价极低。

## 8. 最终目标与路线图（2026-10-09）

最终目标：state-only 预训练能**提升 VLM / 世界模型（WM）**的能力，并考察是否可能**涌现 in-context 能力**。

| 阶段 | 问题 | 实验 | 状态 |
|---|---|---|---|
| 1 | 先验本身是否有效、能否泛化 | E1 / E1b / E2（DROID 内，OOD 实验室） | 完成 |
| 2 | 给定"往哪去"，能否补全"怎么去"（VLM 接口） | E3a 目标条件后训练 | 完成：endpoint 比插值低 35% |
| 3 | 能否迁移到新数据域 | E4a DROID → LIBERO 预测（零样本 / 少样本微调 / 从零训练） | 进行中 |
| 4 | 能否提升视觉策略（VLA） | E4b：LIBERO 少样本 BC，用先验初始化动作头 / 作为噪声源，闭环成功率 | 待做（Jubail） |
| 5 | 能否提升 WM 规划 | E5：先验作为 CEM / MPPI 的动作提议分布，接入动作条件 WM（如 V-JEPA 2-AC，同为 DROID Franka） | 待做 |
| 6 | VLM + 先验 | E6：VLM 给出关键点 / 子目标 → 目标条件先验生成轨迹，对比 VLM 直接输出动作 | 待做 |
| 7 | in-context 能力 | E7：多 episode 上下文（同场景 + 同操作员 + 同指令的 DROID 示范拼接），比较 0/1/3 条示范与不相关示范；随模型规模和上下文长度观察 | 待做 |

**关于 in-context 的判断**：当前设置（6.4 s 上下文、单 episode 窗口、约 10M 参数）只有 episode 内的隐变量推断（阶段、方向、速度），不会出现跨示范的 ICL。ICL 被认为依赖"成组且重复出现"的数据分布和足够长的上下文，因此 E7 需要专门构造训练分布，而不是期待它自发涌现。
