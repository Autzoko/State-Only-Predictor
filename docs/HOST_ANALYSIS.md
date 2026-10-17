# 为什么 SOPT 在 GR00T / π0.5 / X-VLA 上无效或有害（2026-10-17）

本文汇总截至 2026-10-17 的证据、相关文献和待验证的假设，并给出后续实验（E26–E28）的预注册判据。原始数字见 `EXPERIMENTS.md`。

## 1. 现象（LIBERO-90 17 任务，k5，每臂 17 × 50 init × seed 数，配对 McNemar）

| 宿主 | seeds | none | +SOPT scratch | +SOPT DROID | DROID vs none | 宿主预训练是否含 DROID / 大规模机械臂动作数据 |
|---|---|---|---|---|---|---|
| DP | 3 | 72.9 | 71.2 | **77.5** | +4.6，p=3e-7 | 无（从头训练） |
| FastWAM | 2 | 42.9 | 49.2 | **54.8** | +11.9，p=8e-20 | 视频世界模型，无大规模动作预训练 |
| SmolVLA | 2 | 60.4 | 58.6 | **65.7** | +5.3，p=5e-6 | 社区 SO-100 数据，无 DROID |
| GR00T N1.7 | 3 | 81.1 | 79.3 | 81.1 | 0，p=1.0 | 大规模跨本体机器人数据（N1.5 起含 Open X-Embodiment） |
| π0.5 (`pi05_droid`) | 1 | 57.6 | 56.5 | 56.6 | n.s. | **有**：就是在 DROID 上微调过的权重 |
| X-VLA (`xvla-base`) | 3 | 71.7 | 71.8 | 69.2 | **−2.5，p=0.006** | **有**：DROID / RoboMind / AgiBot 共 290K 条 |

规律：宿主预训练里没有大规模机器人动作数据 → SOPT 有效；有 → 无效或有害。只有 6 个宿主，且"预训练数据"与架构、冻结策略、基线水平混在一起，这只是相关，不是因果。

## 2. 已排除的解释

- **实现错误**（E20d）：零初始化成立（新建的 expert_droid 与 none 输出逐元素相同），hook 生效，SOPT 编码器漂移 12%，与 DP / GR00T 同量级。
- **对 5 条示范过拟合**（E20d）：X-VLA + DROID SOPT 在**留出**示范上的开环误差反而更低（−3% / −10%），夹爪一致率更高。
- **只是提升空间小**：X-VLA 基线（71.7）与 DP（72.9）几乎相同，DP 提升 4.6，X-VLA 下降 2.5。GR00T（81%）的天花板效应可能存在，但解释不了 X-VLA。

## 3. 用 scratch 臂把"多一个组件的代价"和"DROID 信息的作用"分开

scratch 臂的结构、注入位置、训练方式与 DROID 臂完全相同，只是编码器是随机初始化的。

| 宿主 | scratch − none（组件本身） | DROID − scratch（DROID 信息） | 解读 |
|---|---|---|---|
| DP | −1.7（p=0.05） | **+6.3**（p=1e-12） | 组件有小代价，DROID 信息收益远大于代价 |
| GR00T | **−1.8**（p=0.03） | **+1.8**（p=0.04） | DROID 信息有用，但刚好抵消组件代价 |
| X-VLA | +0.1（p=0.96） | **−2.6**（p=0.005） | 组件本身无害，**DROID 特征本身**有害 |

所以"宿主已有 DROID 信息 → SOPT 冗余"最多只能解释"没有增益"，解释不了 X-VLA 的"DROID 特征有害"：冗余的信息应当被忽略，而不是让策略变差。

## 4. 假设

### H1 冗余：宿主已经具备 SOPT 提供的信息
- 支持：上表的相关性；π0.5-DROID、X-VLA 都在 DROID 上训练过。
- 需要澄清的是哪种信息。所有宿主在我们的接法下都只看**当前一帧**图像和状态（预训练时也一样），所以 DROID 预训练给它们的是"看到什么画面就做什么动作"的映射，**不包含历史动态**。E24 显示当前帧对速度、加速度、未来 0.5–1 s 运动的 R² 只有 0.15–0.27。
- 因此 H1 的准确形式是：**强视觉宿主能从图像推断出任务阶段和进度（物体被移动后画面就变了），历史中真正独有的只剩局部动力学（速度、加速度），而它对 LIBERO 的成功率不重要。** E27 直接检验这一点。

### H2 接口 / 耦合：加组件的方式与宿主冲突
- GR00T：主干冻结、梯度裁剪 1.0、新本体的动作头从头训练；光加组件就掉 1.8。
- X-VLA：SOPT 特征相对动作 token 的范数比 0.48（GR00T 0.20）。X-VLA 的动作 token 来自单个 DomainAwareLinear（拼接带噪动作 + 当前本体状态 + 时间），所以 SOPT 在 token 里占比很大。X-VLA 用按数据源区分的 soft prompt 来切换本体行为（Zheng et al., 2025, ICLR 2026），DROID 正是它的预训练数据源之一。推测：DROID 风格的运动特征可能把策略推向"DROID 模式"，而 DROID 与 LIBERO 的控制频率和动作空间都不同。**这一点是推测，尚未验证。**

### H3 抄捷径 / 因果混淆（copycat）：最符合"开环更好、闭环更差"这一特征
- 文献：
  - de Haan et al., 2019 指出，模仿学习里**给策略更多信息，闭环反而可能更差**（因果误判 + 分布偏移）。
  - Wen et al., 2020（copycat）：观测历史与专家动作高度相关时，策略学会"延续过去的动作"，而不是根据当前情境决策。
  - Chuang et al., 2022：带历史的模仿常常比只用当前帧更差。
  - Torne et al., 2025：长上下文扩散策略会因虚假相关而退化，主流方法因此丢弃历史。他们用"预测过去动作"作为辅助任务来缓解。
  - "Do You Need Proprioceptive States in Visuomotor Policies?"（2025）：本体状态本身就是低维、无噪声、极易拟合的捷径特征，会损害空间泛化。
- 与我们的数据吻合之处：
  - E20d：X-VLA 加 DROID SOPT 后开环误差更低，但闭环更差，这正是因果混淆的典型表现：拟合示范更准，自己执行时更差。
  - E20d：加 DROID SOPT 后，三个宿主对"换图"的敏感度都大幅下降（X-VLA 1.55→0.95，DP 1.13→0.64，GR00T 1.50→0.57），即策略把一部分依赖从视觉转移到历史。
  - E9a（TrunkBC）：从零训练的历史 trunk 在"去哪"类失败上更差（37 vs 17），已经出现过 copycat 迹象。
- 能统一所有宿主的解释：**SOPT 用视觉依赖换历史依赖，净收益 = 历史信息的价值 − 视觉对齐下降在闭环中的代价。** 视觉通路弱的宿主（DP 从头训练、FastWAM、k5 下的 SmolVLA）用历史换来的收益大于代价；视觉通路强的预训练宿主本来就靠视觉做得好，用历史替换视觉得不偿失。X-VLA 全部参数可训练（VLM 0.1× 学习率），而 GR00T 主干冻结、π0.5 只训动作专家，所以 X-VLA 的视觉通路最容易在微调中被"让位"。这是解释性假设，需要 E26 / E28 检验。

## 5. 这些宿主缺什么，SOPT 能补什么

| 信息 | 宿主为何缺 | 现有 SOPT | 改进后的 SOPT |
|---|---|---|---|
| 速度、加速度 | 只看单帧 | 能（E24 R² 0.99 / 0.90） | — |
| 任务阶段、进度 | 单帧有时有歧义，但强视觉宿主多半能从画面推断 | 能（t_grip 0.80，prog 0.94） | — |
| 下一事件时间 | 同上 | 部分（0.53） | 多 token / 注意力读取整段特征 |
| 动作块之间的衔接 | 每 10 步重新规划，看不到上一刻的运动惯性 | 间接（提供速度） | 显式输入上一块的剩余动作 |
| 执行反馈（是否卡住、抓空、打滑） | 需要对比"发出的指令"和"实际状态" | **不能**（只看状态） | 输入宿主自己的历史动作（微调阶段可得；DROID 侧只用状态时可用状态差分近似） |
| 6.4 s 以前的事 | 无记忆 | **不能** | 更长上下文（E25 ctx192）/ 事件记忆 |
| 接触力 | 视觉和位置都看不到 | **不能** | 需要力传感器数据，代价大 |

相关工作：HAMLET（2025）、MemoryVLA、ContextVLA 等用压缩的历史 token 给预训练 VLA 加记忆，收益主要在长时程、非马尔可夫任务上。这与我们的判断一致：在 LIBERO-90 这类短任务上，历史的边际价值有限。

## 6. 后续实验（预注册）

### E26 闭环失败归类（`scripts/slurm/e26_trace.sbatch`，`scripts/e26_taxonomy.py`）
宿主 X-VLA / GR00T / π0.5 / DP（正对照），三个臂，seed 0，17 任务 × 50 init，重新评测并记录仿真轨迹和每步动作。分类规则（目标物体、抓取距离）由全部 run 的成功 episode 统一标定，不针对单个策略调参。
- 失败类别：wrong_target / no_reach（"去哪"）、grasp_miss / place_fail（"怎么做"）、artic_fail。
- 闭环行为：frozen（最后 5 s 内 EE 移动 < 2 cm）、grip_cycles（重复抓取次数）、jerk_ratio（动作块边界处 / 块内的动作跳变比）。
- 配对分析：none 成功而 SOPT 失败的 episode，看 SOPT 臂的失败类别（SOPT 新造成了哪些失败）；以及反向。
- 预测：
  - H3 成立 → X-VLA 中 SOPT 新造成的失败集中在"去哪"类和 frozen；DP 中 SOPT 修复的失败集中在 grasp_miss（与 E9a 一致）。
  - H2 成立 → X-VLA 新造成的失败集中在 grasp_miss / place_fail，且 jerk_ratio 上升。

### E27 宿主表示探针（`scripts/e27_host_probe.py`）
在宿主动作解码器的输入上，对 E24 的时间目标做线性探针（在 k5 训练示范上拟合，在留出示范上评分）。
- 冗余检验：比较 host(none) 与 host(none)+SOPT 的 R² 差。差 ≈ 0 → 宿主已经知道（H1）。差大 → 信息可用但没有转化为闭环收益（指向 H2 / H3）。
- 使用检验：比较 host(droid) 与 host(none)。前者高，说明 SOPT 信息确实进入了宿主的决策表示。

### E28 干预（等 E26 / E27 结果后决定；先做 X-VLA，若有效再做 GR00T）
- 若 E26 指向 H3（视觉对齐下降 / 冻结）：
  - **E28a SOPT dropout**（`--cond-drop 0.5`，现有代码已支持，E19 在 SmolVLA 上测过）：强迫策略在没有历史时也能只靠视觉行动。
  - **E28b 历史扰动**：训练时对 SOPT 的输入历史加随机游走噪声（类似 DART），切断"延续过去"的捷径。
- 若 E26 指向 H2（执行精度 / 跳变）：
  - **E28c 门控注入**：可学习的标量门（tanh，初始化为 0，类似 Flamingo 的 gated cross-attention），并对 SOPT 特征做 LayerNorm 和尺度约束，把范数比压到 GR00T 的水平。
- 若 E27 显示强宿主已有全部 SOPT 信息：干预意义不大，转向宿主确实缺的信息，即执行反馈（动作-状态历史）和更长记忆（E25 ctx192 的结果决定是否推进）。
- 判据：与对应 seed 的 none 配对，总体 p<0.05 且 dev / new 同向，才算修复；3 seed 确认后才写入主表。

## 7. 对论文叙事的影响（暂定）
- 主张从"SOPT 提升所有宿主"改为：**SOPT 为缺乏大规模机器人动作预训练的策略提供可迁移的运动先验（DP、SmolVLA、FastWAM、TrunkBC 均显著）；对已在 DROID 级数据上预训练的 VLA，增益消失。** 这也说明 SOPT 学到的正是大规模动作预训练所提供的那类信息，而 SOPT 只需状态数据就能得到它。
- X-VLA 的负结果如实报告，并配合 E26 / E28 的机制分析。若 E28 能修复它，则作为"如何把历史接入强 VLA"的方法贡献。

## 参考
- de Haan, Jayaraman, Levine. Causal Confusion in Imitation Learning. NeurIPS 2019. https://arxiv.org/abs/1905.11979
- Wen, Lin, Darrell, Jayaraman, Gao. Fighting Copycat Agents in Behavioral Cloning from Observation Histories. NeurIPS 2020. https://arxiv.org/abs/2010.14876
- Chuang, Yang, Wen, Gao. Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction. ECCV 2022. https://arxiv.org/abs/2207.09705
- Torne et al. Learning Long-Context Diffusion Policies via Past-Token Prediction. 2025. https://arxiv.org/abs/2505.09561
- Do You Need Proprioceptive States in Visuomotor Policies? 2025. https://arxiv.org/abs/2509.18644
- HAMLET: Switch your Vision-Language-Action Model into a History-Aware Policy. 2025. https://arxiv.org/abs/2510.00695
- X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment VLA. ICLR 2026. https://arxiv.org/abs/2510.10274
- GR00T N1.5 / N1.7 model cards（NVIDIA）。N1.5 预训练含 Open X-Embodiment；N1.7 另有 DROID 后训练版本。
