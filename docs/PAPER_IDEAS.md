# 论文构思：如何让 state-only 预训练发挥最大作用（2026-10-10）

## 0. 定位：一个被明确指出、但尚无人填补的空白
- 2608.03052（*How Should VLA Models Use Proprioceptive State?*，2026-08）系统比较了五种状态接口和 1–96 帧的历史长度，并明确写道：视觉和语言在 VLM 中经过了大规模预训练，**proprioceptive state 却没有任何预训练**。它研究的是"状态怎么接入"，没有研究"状态是否应该预训练"。
- 相关但不同的工作：HPT（proprio + 视觉 stem 联合预训练，不是纯 proprio）；ReViP、Think Proprioceptively、GeoProp（重新平衡视觉与 proprio）；copycat / causal confusion（Wen et al. 2020，de Haan et al. 2019：历史输入导致抄袭先前动作）；latent action 方法（LAPA、Motion-Focused Latent Action，从视频中学动作，没有本体接地）；Blind Dexterity（人形机器人纯 proprio 控制，不涉及预训练）。
- **SOPT 的位置**：第一个"**只用本体轨迹、不用视觉和语言**"的大规模自监督预训练，并给出它何时、以何种接口、在什么数据规模下帮助 VLA 的系统性证据。

## 1. 推荐的核心论点（主线 A）
> **Proprioception deserves its own pretraining**：本体历史在从零训练时是"毒药"（copycat / 过拟合），在 vision-free 的大规模轨迹上预训练之后变成"良药"，并且这种预训练几乎是免费的。

我们已有的证据恰好构成一个干净的"翻转"：
| | 从零训练的历史 trunk | 预训练的历史 trunk |
|---|---|---|
| vs BC（k5，3 个 seed） | 59.0% < 63.6% | **70.0%** > 63.6%（p = 0.012） |
| vs BC（全部，3 个 seed） | 89.3% < 92.4% | **96.4%** > 92.4%（p = 0.0095） |
| SmolVLA（k5，3 个 seed） | 运动 token 57.4% < 无历史 60.5% | 61.2%（vs scratch p = 0.11，**不显著**；seed 0 的 65.0% 没有复现） |

即"历史本身无益，有益的是**预训练过的**历史表示"。这正好回答了 2608.03052 提出的问题之一：历史的收益来自真实的时间信息，还是只是多了条件容量？我们的答案是：容量（scratch）无益，**预训练**才有益。

### 主张与证据状态
| 主张 | 已有证据 | 缺口 |
|---|---|---|
| C1 只用 vision-free 轨迹就能学到可迁移的运动先验 | DROID OOD 实验室（ES −55%），零样本迁移到 LIBERO（ES −60%） | 跨本体（EE 表示 + OXE），规模曲线 |
| C2 预训练的本体历史能翻转 copycat，提升闭环策略 | 小策略 3 个 seed 显著；SmolVLA 3 个 seed 只有弱趋势、不显著 | **VLA 接口需要改进**（实验 7 提前）；第二个 VLA；**扰动 / 鲁棒性分析**（直接展示 copycat 被缓解） |
| C3 收益集中在低数据区间，数据充足时收敛 | E4b、E5、E4a 一致 | 更细的 k 曲线（1 / 2 / 5 / 10 / 20 / 全部） |
| C4 代价极低 | 1.4 GB 数据，单卡数小时 | 与 VLA 预训练成本的对照表 |
| C5（方法）接口选择很关键：表示 / trunk 有效，逐步状态跟踪无效；必须按步归一化 | E4b v1–v3 和诊断 | 作为分析小节即可 |

## 2. 让它"大"的三个扩展方向（按性价比排序）
### A+. Vision-free 的"暗数据"规模化（最具影响力的卖点）
- 工业和部署中存在海量**没有摄像头记录**的机器人日志（关节、末端、夹爪），VLA 用不上，SOPT 可以用。
- 论文中用可获得的数据模拟这一点：DROID + OXE 中所有带本体状态的子集（以 EE 为中心的表示，跨本体）+ 仿真数据（MimicGen / RoboCasa 轨迹，不渲染图像，近乎无限）。
- 关键图：**下游闭环成功率 vs 纯状态预训练数据量**（log 轴）。如果曲线单调，就是一个非常有说服力的"本体数据的 scaling law"。
- 前置工作：实现 EE-only 表示的先验（pos + rot6d + gripper，去掉关节）。这样也能直接用于 LeRobot 标准的 LIBERO（只有 EE 状态）和其他机器人。

### B. 显式的 copycat 分析（让主线论点"可见"）
- 扰动测试：初始状态偏移、中途推动 / 抓取失败后的恢复、图像退化。看从零训练的历史 trunk 是否因为抄历史而失败，预训练的是否更依赖视觉、更能恢复。
- 机制分析：用对视觉 / 历史的注意力和梯度归因衡量依赖程度；用"previous-action predictability"衡量 copycat（Wen et al. 的指标）。
- 这会让论文从"又一个预训练涨点"变成"**解释了本体状态为什么一直难用、并给出解法**"。

### C. 两个"新能力"作为加分项（视进展决定是否放入）
- **State-only in-context 目标推断**（`IDEAS_WM_ICL.md` B）：从上下文示范推断"往哪去"。如果成功，就是一个全新的能力演示，novelty 很高。
- **WM 规划的提议分布**（`IDEAS_WM_ICL.md` A1）：先验作为 CEM 的初始分布，以样本效率为指标。

## 3. 建议的论文结构（主线 A + A+ + B）
1. **Intro**：vision 和 language 都有预训练，proprio 没有；本体历史在 VLA 中常常有害（copycat）。我们提出 SOPT，用 vision-free 轨迹预训练本体表示。
2. **Method**：patch-token Transformer；masked / flow 目标；按步归一化；运动学精确的 yaw 增广；site-level 防泄露协议；即插即用的 motion-token 接口（替换 `state_proj`）。
3. **What does SOPT learn?**：OOD 预测、多模态（ES）、目标补全、probe（成功 / 失败、阶段）。
4. **Transfer**：零样本 / 少样本跨域（DROID → LIBERO）；数据效率（1 条 demo ≈ 全部 demo 的 scratch）。
5. **Policies**：小 BC trunk + SmolVLA（+ 第二个 VLA）；k 曲线；多 seed。
6. **Why it works: copycat analysis**：扰动、归因。
7. **Scaling vision-free data**：DROID → +OXE → +sim，以及模型规模。
8. **Interfaces that fail**：先验 + IDM 的负面结果和原因（对社区有价值）。

## 4. 缺口实验（优先级、预计成本）
| # | 实验 | 目的 | 成本 |
|---|---|---|---|
| 1 | SmolVLA k5 多 seed | C2 显著性 | 已完成：不显著（61.2 vs 57.4 vs 60.5） |
| 2 | 细粒度 k 曲线：k ∈ {1, 2, 5, 10, 20, 全部}，3 个 seed（小 BC trunk） | C3 | 127 上约 1 天 |
| 3 | 扰动 / copycat 分析（复用 E4b 策略和仿真） | B | 127 上 1–2 天 |
| 4 | EE-only 先验 + 标准 LIBERO 四个 suite（LeRobot 数据），SmolVLA | 标准 benchmark，便于对比文献 | Jubail 2–3 天 |
| 5 | 第二个 VLA（π0 / π0.5 via openpi，或 GR00T）+ motion token | 通用性 | Jubail 3–5 天 |
| 6 | 预训练数据规模：DROID 1/10/100% + OXE + sim → 下游 | A+ scaling | 数据工程 3–5 天，训练便宜 |
| 7 | **（优先级提升）** 接口消融：motion token 进入 VLM prefix vs 只给 action expert；冻结 vs 微调 SOPT | 回应 2608.03052 | Jubail 2 天 |
| 8 | 真机（实验室 Franka，如果有）：少样本，2–3 个任务 | 可信度 | 视硬件而定 |

## 5. 审稿人可能的质疑与对策
- **"收益只是因为多了历史 / 参数"** → scratch 历史 trunk / scratch motion token 对照已经覆盖（容量相同，没有收益）。
- **"只在低数据下有效"** → 正面承认并作为主张 C3（低数据正是真实部署的常态），再用 A+ 的 scaling 结果说明更多 vision-free 数据会继续抬高上限。
- **"任务太少、单一仿真"** → 补实验 4、5、8。
- **"和 HPT / latent action 的区别"** → 我们完全不用视觉，数据来源不同（暗数据），并直接与之对比（如果可行，复现 HPT stem 作为基线）。
- **"Franka-only"** → EE-only 表示 + OXE 跨本体（实验 4、6）。

## 6. 标题备选
- *Proprioception Deserves Pretraining: Vision-Free Motion Priors for Robot Policies*
- *State-Only Pretraining Turns Proprioceptive History from Liability into Asset*
- *Learning to Move Before Learning to See: Scalable Proprioceptive Pretraining for VLAs*
