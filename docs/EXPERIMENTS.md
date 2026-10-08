# 实验计划与日志

计划和判停标准见 `DESIGN.md` §6。每次运行记录：日期、机器、git rev、命令、关键指标、结论。
规则：checkpoint 选择和调参只用 `val`；`test` 只在论文最终数字时使用一次。

## 待办（按顺序，每一步以上一步结果为前提）
- [x] E0 在 127 上构建 state-only 数据，检查 `summary.json`
- [x] E0 吞吐测试
- [ ] dataloader 优化（S 模型可能受 CPU 限制）
- [ ] E1 `ar_regression` / `ar_flow` / `masked` × S（然后是 B），与 zero-vel / const-vel 对比
- [ ] 判停检查（DESIGN §6）：`ar_flow` 的 min10_pos_ade 相比 const_vel 下降 ≥30%？
- [ ] E3 线性 probe（成功/失败、gripper 事件），需要 `sopt/eval/probe.py`
- [ ] E4 下游：LIBERO 少样本策略，需要 `sopt/policy/`（视觉条件 + 预训练 trunk）

## 日志

### 2026-10-07 搭建
- 已核实 DROID schema（`cadene/droid_1.0.1` @ 56b622a）：欧拉角为 extrinsic XYZ，`cartesian_position` 与 Franka FK 的 flange 位姿完全一致；元数据列错位（见 DESIGN §3）。
- 本地 CPU 调试（30 条 episode，40 步）：三种目标函数从训练、评估到保存 checkpoint 全流程跑通。指标没有意义。

### 2026-10-07 E0 数据 + 吞吐（127, 1a176fb）
- 数据来源改为 `cadene/droid_1.0.1_v30` @421fe53，只用列投影读取 state，总传输约 1.4 GB，耗时约 30 分钟。产出 `SOPT_DATA/processed/droid`：states 1.5 GB，features 1.9 GB。
- 95,584 条 episode，27,607,757 帧（与 info.json 一致）。去掉了 episode 88905 的 59 帧 BAIR 残片（`episode_index` 重复）。
- 切分改为 site 级：74 个 building 合并为 15 个 site。
  - train：83,606 条 / 23.1M 帧
  - id_val：853 条
  - val（UW）：3,961 条 / 1.55M 帧
  - test（UT Austin + Edinburgh + Penn）：7,164 条 / 2.69M 帧
- 吞吐（A6000，与 XICM 共用 GPU，batch 256，ar_flow）：
  - S（9.5M）约 20 it/s，100k 步约 1.4 小时；
  - B（56M）约 9 it/s，100k 步约 3 小时。
  - 300 步时的 val pos_ADE：S 为 5.99 cm（min10：3.31），const-vel 为 3.08 cm，还没训练充分。
- 数据已复制到 3090（`~/Desktop/langtian/SOPT_DATA`）。

### 2026-10-07 17:00 E1 启动（127, 4ea4625）
- `e1_{ar_regression,ar_flow,masked}_s`：S 模型，100k 步，batch 256，三个任务共用 A6000 同时运行，每个 6 个 dataloader worker（XICM 也在占用 CPU）。速度 10–14 it/s，预计 2.5–3 小时。
- `masked` 第一次启动时只有约 1 it/s（span mask 在 Python 循环里，每步都和 GPU 同步）。改成向量化实现后重启（4ea4625）。
- 早期 val（UW）pos_ADE：ar_regression 在 4k 步时为 2.36 cm；ar_flow 在 2k 步时 min10 为 2.30 cm（单样本 4.39 cm）；const-vel 为 3.20 cm。

### 2026-10-07 19:00 E1 v1 结果：训练不稳定（127，归档在 `outputs/e1v1/`）
- **现象**：两个 AR 模型的**训练 loss** 在约 28k 步（约 1 个 epoch）后回升（regression 从 0.313 升到 0.50，flow 从 0.150 升到 0.25）。grad norm 从约 25k 步开始指数增长（0.5 → 1e8）；masked 增长较慢。
- **原因**：第一层 attention 的 qkv 权重范数涨到约 590（其他层约 18），ln1 的 gain 也在变大，属于 attention logit growth（Dehghani et al. 2023）。
- **修复**：对每个 head 的 q/k 加 RMSNorm（`model.qk_norm: true`，87deea0）。另外新增 `samplemean_*` 指标：K 个样本取均值后的点估计，和确定性模型对比更公平。
- best.pt（不稳定之前的 checkpoint；masked 为 96k 步，其余为 28k 步）的完整评估，单位 cm：

| 模型 | split | ADE | min10 | 样本均值 | FDE | const-vel ADE |
|---|---|---|---|---|---|---|
| ar_regression | val | 2.25 | – | – | 4.57 | 3.21 |
| ar_flow | val | 3.05 | **1.59** | 2.36 | 6.23 | 3.21 |
| masked | val | 2.27 | – | – | 4.58 | 3.21 |
| ar_regression | id_val | 2.37 | – | – | 4.84 | 3.86 |
| ar_flow | id_val | 3.25 | 1.71 | 2.50 | 6.65 | 3.86 |
| masked | id_val | 2.51 | – | – | 5.08 | 3.86 |

- zero-vel 的 val ADE 为 3.51。**判停检查通过**：ar_flow 的 min10 比 const-vel 低 50%（val）和 56%（id_val）。确定性模型比 const-vel 低约 30%。
- 多模态明显：flow 的 min10（1.59）远好于样本均值（2.36），而样本均值和回归（2.25）相当，说明一次 forecast 内的多条样本确实分布在不同的走法上。
- 以上结论只基于单个 seed，并且是不稳定之前的 checkpoint。需要用 v2 的数字确认。

### 2026-10-07 19:05 E1 v2 启动（127, 87deea0）：QK-norm，其余设置与 v1 相同
- 20:07 检查：三个 run 都已越过 v1 出问题的步数（45–58k）。grad norm 稳定在 0.2–0.5，训练 loss 单调下降，**QK-norm 修复有效**。val 基本已饱和：regression 最佳 2.22（26k 步），flow min10 最佳 1.57（40k 步），masked 2.25（仍在缓慢下降）。

### 2026-10-07 E1 v2 最终结果（127, 87deea0；best.pt 按 val 选择；单 seed）
位置单位 cm，角度单位 deg；min10 = 10 个样本中最好的一个；smean = 10 个样本取均值。

| 模型 | split | step | ADE | min10 | smean | FDE | rot | joint | gripper |
|---|---|---|---|---|---|---|---|---|---|
| zero-vel | val | – | 3.51 | – | – | 6.18 | 6.50 | 2.95 | 0.056 |
| const-vel | val | – | 3.21 | – | – | 6.94 | 5.52 | 2.53 | 0.056 |
| ar_regression | val | 26k | 2.23 | – | – | 4.56 | 4.28 | 1.85 | 0.035 |
| masked | val | 90k | **2.19** | – | – | **4.47** | **4.21** | **1.82** | 0.035 |
| ar_flow | val | 90k | 2.91 | **1.53** | 2.30 | 6.01 | 5.66 | 2.43 | 0.051 |
| zero-vel | id_val | – | 4.20 | – | – | 7.52 | 6.87 | 3.18 | 0.080 |
| const-vel | id_val | – | 3.86 | – | – | 8.39 | 6.16 | 2.90 | 0.080 |
| ar_regression | id_val | 26k | **2.34** | – | – | **4.80** | **4.26** | **1.86** | 0.045 |
| masked | id_val | 90k | 2.39 | – | – | 4.91 | 4.30 | 1.88 | 0.045 |
| ar_flow | id_val | 90k | 3.00 | **1.59** | 2.37 | 6.17 | 5.60 | 2.42 | 0.061 |

**结论**
1. 先验有效：确定性模型比 const-vel 低 31–32%（val）和 38–39%（id_val），FDE 低约 35–43%，gripper 误差低约 40%。
2. 跨实验室泛化好：在 OOD val（UW）上的相对提升与 id_val 接近，没有明显的 OOD 掉点。
3. 多模态：flow 的 min10 比 const-vel 低 52%（val），但样本均值（2.30）略差于确定性模型，单个样本更差（2.91）。**min-of-K 对生成模型偏乐观**，需要加 proper scoring rule（energy score）才能公平比较。
4. ar_regression 在 26k 步之后 val 就不再提升（训练 loss 仍在下降）。flow 和 masked 到 90k 步还在缓慢改善。
5. 三种目标的确定性误差相差在 0.1 cm 以内。单 seed 下不能区分 masked 和 regression。

**下一步候选**
- (a) 评估：加 energy score 和按预测步长分开的误差曲线（便宜，先做）。
- (b) E1b：更长的 horizon（32/48 步，约 2–3 s）和 context 消融。1 s 内运动学外推还较强，先验的价值应该在更长的时域上更明显。
- (c) E2：数据量 × 模型规模（B/L 放 Jubail 或 3090）。
- (d) E3：线性 probe（成功/失败、gripper 事件）。

### 2026-10-08 补充评估：energy score 与逐步误差（8bbde14，E1 v2 best.pt）
energy score（ES，EE 位置轨迹，cm，越低越好；对确定性模型等于轨迹 RMSE）和第 k 步的位置误差（cm）：

| 模型 | val ES | 第 1 步 | 第 4 步 | 第 8 步 | 第 16 步 | id_val ES |
|---|---|---|---|---|---|---|
| zero-vel | 3.99 | 0.46 | 1.79 | 3.43 | 6.18 | 4.80 |
| const-vel | 3.91 | 0.16 | 1.03 | 2.77 | 6.94 | 4.71 |
| ar_regression | 2.69 | 0.14 | 0.78 | 2.02 | 4.56 | 2.83 |
| masked | 2.65 | 0.13 | 0.76 | 1.98 | 4.47 | 2.90 |
| ar_flow | **1.96** | 0.16 | 0.99（单样本） | 2.62 | 6.01（样本均值 4.78；min10 2.90） | **2.01** |

- 用 proper scoring rule 衡量，**ar_flow 作为概率预测比确定性模型好约 27%**（ES 1.96 vs 2.65–2.69）。min-of-K 上的优势并不只是样本多样性带来的假象。
- 第 1 步时所有方法和 const-vel 持平（短期运动学外推就足够了），之后差距随步长增大，第 16 步时为 4.5 vs 6.9 cm。这支持做更长 horizon 的消融。

### 2026-10-08 10:35 E1b 启动：horizon 消融（127, 8bbde14）
- `e1b_h{32,48}_{ar_regression,ar_flow,masked}_s`：S 模型，ctx 96，horizon 32（2.1 s）和 48（3.2 s），100k 步。6 个 run 共用 A6000，每个 7–9 it/s，预计 3.5–4 小时。
- 对比方式：逐步误差曲线的前 16 步可以和 E1 的 H=16 模型直接比较，看预测更远的未来是否会损害近期精度；ES 只在同一 horizon 内比较。

### 2026-10-08 E1b 结果：horizon 消融（127, 8bbde14；best.pt 按 val 选择；单 seed）
val（UW）：ES 为 energy score；ADE 对 flow 用样本均值；"第 16 步" 是点估计在第 16 步（约 1 s）的误差，用来检查预测更远的未来是否损害近期精度。单位 cm。

| H | 模型 | best step | ES（cv） | ADE（cv，相对） | 第 16 步 | 第 32 步 | 第 48 步 |
|---|---|---|---|---|---|---|---|
| 16 | regression | 26k | 2.69（3.91） | 2.23（3.21，−30%） | 4.56 | – | – |
| 16 | masked | 90k | 2.65 | 2.19（−32%） | 4.47 | – | – |
| 16 | flow | 90k | **1.96** | 2.30（−28%） | 4.78 | – | – |
| 32 | regression | 22k | 5.45（8.83） | 4.60（7.30，−37%） | 4.66 | 8.62 | – |
| 32 | masked | 76k | 5.24 | **4.41（−40%）** | 4.47 | 8.29 | – |
| 32 | flow | 54k | **3.99** | 4.92（−33%） | 4.94 | 9.22 | – |
| 48 | regression | 10k | 7.58（13.63） | 6.48（11.38，−43%） | 4.71 | 8.58 | 11.45 |
| 48 | masked | 34k | 7.34 | **6.26（−45%）** | 4.58 | 8.30 | 11.04 |
| 48 | flow | 92k | **6.02** | 7.64（−33%） | 5.65 | 9.66 | 12.86 |

const-vel 在第 16/32/48 步的误差为 6.9 / 15.4 / 23.6 cm。id_val 上的趋势相同（相对提升更大：H48 确定性模型 −50%）。

**结论**
1. **先验的价值随 horizon 增大**：确定性 ADE 相比 const-vel 的降幅从 −30%（H16）到 −40%（H32）再到 −45%（H48）；flow 的 ES 降幅从 −50% 到 −55% 再到 −56%。
2. **预测更远的未来基本不损害近期精度**：masked 第 16 步的误差为 4.47（H16）、4.47（H32）、4.58（H48）。flow 的样本均值退化较多（4.78 → 5.65）。
3. **确定性目标中 masked 稳定最好**，flow 在 ES 上一直最好。
4. **对 OOD 实验室过拟合，horizon 越长越严重**：H48 regression 的 val ES 在 10k 步达到最优 7.59，到 100k 步升至 8.39，而训练 loss 仍在下降。masked 和 flow 的 val 曲线基本平稳。说明目前的瓶颈是**泛化而非容量**：S 模型 10–40k 步就能吃完能泛化的信号。

**对后续实验的含义**
- 默认 horizon 改为 32（约 2 s）：收益明显，近期精度不受损害，flow/masked 也不过拟合。
- E2 要重点看**数据量**这条轴；直接加大模型很可能加重过拟合，需要配合正则（dropout、weight decay、增广强度），或引入更多元的数据（其他 Franka 数据集）。
- 训练步数可降到约 40k（val 在 20–40k 步就已饱和），节省一半算力。
