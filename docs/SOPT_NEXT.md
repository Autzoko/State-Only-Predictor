# SOPT 的新用途与无损提升：分析与实验计划（2026-10-23）

依据：本项目 E18–E35 的结果，加上下列已发表的工作，链接见文末。本文的方案都要求预先写明判据，并说明对已有表现的保护方式。

## 0. 从现有结果得到的出发点
- SOPT 表示里真正有、宿主没有的信息：局部动力学（速度 / 加速度 R² 0.9 以上）、夹爪事件时间、阶段，以及只看运动就能识别任务（0.87）。见 E24、E27。
- 这些信息在强宿主的**短任务**上转化不成成功率，因为瓶颈在视觉抓取精度（E26 / E34）。在**需要记住进度的长任务**上能转化（E35 摩卡壶，p=2e-4）。
- 加入方式本身有代价：加法和 AdaLN 会诱发"抓空照常搬运"，即 copycat / 惯性。交叉注意力加零初始化门控可以消除代价，但带 dropout 的版本学会了不用 SOPT。
- 不同先验实例之间的方差可能比预想的大（E35 中 ctx96 的 e13 先验掉分），需要量化（E33 ctx96 复制、E25 base）。
- 除了"作为策略的输入"，A1 门槛已经通过：只看本体感觉信号就能预测失败（AUC 0.73–0.89），但我们还没有用过。

## 1. 之前没有考虑到的收益（不需要改宿主策略，或者只是测试时外挂）

### E36 运行时失败预测与恢复（宿主无关，只用本体感觉）
- 依据：
  - Sentinel（Agia et al., CoRL 2024）用动作的时间一致性检测"抖动型"失败，用 VLM 检测"进度型"失败。
  - FAIL-Detect（RSS 2025）只用成功数据学习失败分数，再用保形预测给出误报率保证。
  - 两者都依赖宿主自身或视觉。SOPT 提供的是**与宿主无关**的运动"惊讶度"：先验对未来 1 s 的预测与实际执行结果之间的偏差。这一点没有人做过。
- E36a（只做分析，CPU）。数据是 E26 的轨迹：4 个宿主 × 3 个臂，17 任务 × 50 init。比较以下特征：
  - SOPT 惊讶度（预测似然 / ADE）；
  - SOPT 表示 + 线性分类器；
  - 手工特征 blocked；
  - 原始本体感觉统计。
  - 指标：
    - 各时间点的 AUROC；
    - 在保形 5% 误报率下的提前检测时间；
    - **跨宿主迁移**：在 A 宿主上拟合，在 B 宿主上测试。
  - 判据：SOPT 惊讶度 AUROC ≥ 0.75，且比原始本体感觉基线高 ≥ 0.05；跨宿主迁移时下降 ≤ 0.05。
- E36b（只评测）：检测器触发后执行恢复，即回到闭合夹爪前的位姿重新抓取，或者重新采样动作块。宿主是 GR00T none / π0.5 none，不重新训练。
  - 判据：17 任务成功率配对 p<0.05。
  - 无损保证：不触发时与原策略逐步完全相同。

### E37 视觉分布偏移下的鲁棒性（只评测已有 checkpoint）
- 依据：LIBERO-Plus（CVPR 2026）显示，相机视角和机器人初始状态的轻微扰动就能让 VLA 从 95% 掉到 30% 以下。本体感觉历史不受相机、光照、纹理的影响。
- 假设：视觉扰动下，DROID SOPT 的收益会**变大**；机器人初始状态扰动（这是本体感觉的分布偏移）作为反向对照。
- 设置：
  - checkpoint：SmolVLA none / droid、DP none / droid（17 任务），GR00T none / xattn_droid ctx384（libero_10）；
  - 扰动：相机小 / 中位移、光照、背景纹理、图像噪声、机器人初始关节扰动；
  - 优先使用 LIBERO-Plus 的扰动定义，装不上时在评测封装里自己实现。
- 判据：预先指定的交互检验，即（扰动下 droid−none）−（干净时 droid−none）> 0，配对自助法 p<0.05。

### E38 数据效率曲线
- 设置：SmolVLA、DP，每任务示范数 k ∈ {1, 2, 5, 10, all}，17 任务。
- 假设：示范越少，DROID 先验的增益越大，到 all 时收敛。E4b 中 TrunkBC 在 k5 / all 上都有增益，但没测过宿主。

### E39 示范质量打分与数据筛选（离线）
- 依据：DemInf（RSS 2025）用互信息估计示范质量；CUPID（CoRL 2025）用影响函数筛选数据，不到 33% 的数据即可达到最好效果。
- SOPT 是 DROID 上的运动似然模型，可以直接给每条示范打"运动自然度"分。
- 设置：在 LIBERO k10 中注入 20% 劣化示范（停顿、抖动、错误速度回放）。
  - 指标：SOPT 分数识别劣化示范的 AUROC；
  - 闭环：按分数筛掉之后，DP / SmolVLA 的成功率能否恢复。

### 后续（成本高，放在后面）
- 残差 RL 精调（Policy Decorator，ICLR 2025），用 SOPT 作为探索先验 / 动作平滑约束。
- 跨本体：EE-only 先验迁到非 Franka 机械臂。LIBERO 与 DROID 都是 Franka，跨本体目前完全没有测过。

## 2. 在不破坏当前表现的前提下提升 SOPT

所有改动都要求**构造上可退化为原模型**，并先通过**非劣效检验**：LIBERO-90 17 任务上相对 none / 现有 SOPT 臂，预先设定 −1 个点的界值，单侧 α=0.05。然后才看长时程任务上的优效。

### E40 权重插值（WiSE-FT）：零训练成本的保底
- 依据：WiSE-FT（Wortsman et al., CVPR 2022）在微调模型与原模型之间做线性插值，分布内精度不降，分布外 +8.7 点。
- 做法：
  - 我们的 xattn 臂在初始化时与 none 逐元素相同（E31 已验证 max|Δ|=0），两者来自同一个 GR00T 预训练权重。因此可以在 none 与 xattn_droid 之间插值共享权重，同时把 SOPT 支路缩放 α。
  - α=0 时严格等于 none。α 在不参与测试的 init（50–59）上选。
- 判据：选出的 α 在 libero_10 上不低于 xattn，在 LIBERO-90 上对 none 非劣效。

### E41 测试时用 SOPT 给动作块打分（BID 式）
- 依据：Bidirectional Decoding（ICLR 2025）在测试时采样多个动作块，按后向一致性和前向对比选择，对生成式策略即插即用。
- 做法：GR00T / SmolVLA 每步采样 N=8 个动作块，用 SOPT 计算每个块在当前历史下的运动似然，选最高的。N=1 时与原策略完全相同。
- 风险：强宿主的失败主要是抓取精度，似然打分未必能分辨。所以先在 DP / SmolVLA 上验证。

### E42 保护预训练特征
- 依据：LP-FT（Kumar et al., ICLR 2022）指出，全量微调会扭曲好的预训练特征；先线性探针，再微调，在分布内和分布外都更好。我们的 E24 也看到，在 k5 上微调会让编码器丢失通用运动信息。
- 设置：GR00T xattn_droid ctx384，libero_10，对比四种做法：
  1. 冻结 SOPT 编码器；
  2. 先冻结 5k 步再解冻；
  3. 先验学习率 ×0.1，但不加 dropout；
  4. 现在的全量微调。
- 附带检验：冻结能否降低先验实例之间的方差。

### E43 抗 copycat 的辅助目标
- 依据：
  - Copycat（Wen et al., NeurIPS 2020）：带历史的模仿学习容易学成"抄上一步动作"；
  - Past-Token Prediction（Torne et al., CoRL 2025）：让长上下文策略同时预测过去的动作，成功率提高 3 倍，训练快 10 倍以上；
  - Octo 也报告过本体感觉输入导致因果混淆、效果变差。
- 这与我们的"抓空照常搬运"和"长历史更好"两个现象直接对应。
- 做法：在 GR00T xattn 上加过去动作预测头（权重 0.1–0.5），只在训练时使用，推理路径不变。

### E44 先验集成 / 多实例
- 依据：E35 中不同先验实例的结果差了约 5 个点。
- 做法：
  - 三个先验实例的特征取平均，或者训练时每个 batch 随机选一个实例；
  - 论文中报告跨先验实例的方差。
- 成本低：每个先验训练约 22 min。

### E45 扩大状态预训练数据
- 依据：E18 表明收益来自数据规模和多样性，同域少量状态没用；1% → 10% → 100% DROID 的预测误差仍在下降（5.21 → 4.40 → 3.98）。
- 做法：加入其他公开数据集的**本体感觉**部分（只下载状态），用 EE-only 先验跨本体合并。
- 判据：先在 TrunkBC 上预先指定，再上宿主。

## 3. 执行顺序（Jubail 最多 2 个并发；DGX 的任务已全部迁到 Jubail）
1. 已排队：
   - E35 ctx96 复制先验 3 seed；
   - E25j 的 ref / ctx48 / ctx192 / ctx384 / base / speed / scratch；
   - SmolVLA 17 任务 s2 的三个臂。
2. P0，不需要训练：
   - E36a 失败预测，CPU 分析，用已有轨迹；
   - E40 权重插值，只评测；
   - E37 视觉扰动鲁棒性，只评测。
3. P1，训练成本小：E42、E43、E44，以及 E38 中 SmolVLA 的部分。
4. P2：E36b 恢复、E39 数据筛选、E41 动作块打分、E45 扩数据，以及残差 RL 和跨本体。

## 参考
- Agia et al., Unpacking Failure Modes of Generative Policies: Runtime Monitoring of Consistency and Progress (Sentinel), CoRL 2024. https://arxiv.org/abs/2410.04640
- Xu et al., Can We Detect Failures Without Failure Data? (FAIL-Detect), RSS 2025. https://arxiv.org/abs/2503.08558
- Fei et al., LIBERO-Plus: In-depth Robustness Analysis of VLA Models, CVPR 2026. https://arxiv.org/abs/2510.13626
- Hejna et al., Robot Data Curation with Mutual Information Estimators (DemInf), RSS 2025. https://arxiv.org/abs/2502.08623
- Agia et al., CUPID: Curating Data your Robot Loves with Influence Functions, CoRL 2025. https://arxiv.org/abs/2506.19121
- Yuan et al., Policy Decorator: Model-Agnostic Online Refinement for Large Policy Model, ICLR 2025. https://arxiv.org/abs/2412.13630
- Wortsman et al., Robust Fine-Tuning of Zero-Shot Models (WiSE-FT), CVPR 2022. https://arxiv.org/abs/2109.01903
- Liu et al., Bidirectional Decoding: Improving Action Chunking via Closed-Loop Resampling, ICLR 2025. https://arxiv.org/abs/2408.17355
- Kumar et al., Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution (LP-FT), ICLR 2022. https://arxiv.org/abs/2202.10054
- Wen et al., Fighting Copycat Agents in Behavioral Cloning from Observation Histories, NeurIPS 2020. https://proceedings.neurips.cc/paper/2020/hash/1b113258af3968aaf3969ca67e744ff8-Abstract.html
- Torne et al., Learning Long-Context Diffusion Policies via Past-Token Prediction, CoRL 2025. https://arxiv.org/abs/2505.09561
- Octo Model Team, Octo: An Open-Source Generalist Robot Policy, RSS 2024. https://arxiv.org/abs/2405.12213
- Wang et al., Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers (HPT), NeurIPS 2024. https://arxiv.org/abs/2409.20537
- Shi et al., MemoryVLA, ICLR 2026. https://arxiv.org/abs/2508.19236
