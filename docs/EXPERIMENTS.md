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

### 2026-10-08 14:52 E2 启动：数据规模与正则（127, fa42c66）
- 问题：先验质量是否随数据量提升？OOD 过拟合能否靠正则缓解？
- 设置：S 模型，H=32，40k 步（`configs/experiment/e2.yaml`），目标为 masked 和 ar_flow。
- 数据比例：1% / 10% / 100%（嵌套子集，分别为 864 / 8,288 / 83,606 条 episode）。
- 正则组（只在 100% 上做，`e2_strongreg.yaml`）：dropout 0.2、weight decay 0.1、yaw ±60°、输入噪声 0.02。
- 共 8 个 run 同时运行，每个约 5.3 it/s，预计约 2 小时。

### 2026-10-08 E2 结果：数据规模与正则（127, fa42c66；best.pt 按 val 选择；单 seed）
val（UW），H=32，40k 步。ES 为 energy score；ADE 对 flow 用样本均值；单位 cm。const-vel：ES 8.83，ADE 7.30。

| 目标 | 数据 | best step | val ES | val ADE | 第 32 步 | gripper | val ES 最后一次评估 | id_val ES |
|---|---|---|---|---|---|---|---|---|
| masked | 1% | 2k | 6.59 | 5.72 | 10.03 | 0.110 | 7.37 | 7.60 |
| masked | 10% | 8k | 5.52 | 4.69 | 8.62 | 0.086 | 6.17 | 6.38 |
| masked | 100% | 38k | **5.28** | **4.45** | 8.34 | 0.079 | 5.28 | 5.85 |
| masked | 100% + 强正则 | 36k | 5.40 | 4.58 | 8.45 | 0.080 | 5.41 | 6.02 |
| flow | 1% | 2k | 5.21 | 6.49 | 11.38 | 0.166 | 7.38 | 5.55 |
| flow | 10% | 8k | 4.40 | 5.45 | 9.94 | 0.130 | 5.86 | 4.72 |
| flow | 100% | 38k | **3.98** | **4.91** | 9.19 | 0.111 | 4.01 | 4.24 |
| flow | 100% + 强正则 | 38k | 4.03 | 4.98 | 9.15 | 0.113 | 4.05 | 4.35 |

**结论**
1. **数据越多越好，但收益递减**：ES 从 1% 到 10% 降约 16%，从 10% 到 100% 再降 4%（masked）和 10%（flow）。flow 从数据中获益更多。
2. **即使只有 1% 的数据（864 条）也明显好于 const-vel**（flow ES 5.21 vs 8.83），说明运动先验中相当一部分很容易学到。
3. **小数据过拟合非常快**：1% 在 2k 步、10% 在 8k 步就达到最优，之后 val ES 变差 12–36%。用 100% 数据时 40k 步内不过拟合。
4. **强正则没有帮助**，反而略差（ES +0.05 到 +0.12）。E1b 中的过拟合主要来自长时间训练的 regression，不是 masked/flow 的问题。
5. **40k 步已经足够**：100% 数据下的结果与 E1b 的 100k 步一致（masked 5.28 vs 5.24，flow 3.98 vs 3.99）。

**含义与下一步**
- 用 100% 数据时 S 模型不再过拟合，而从 10% 到 100% 的收益在递减。下一步的问题变成"**更大的模型能否从现有数据中挖出更多**"（E2b：B/L × 100%）。如果不能，就需要扩充数据的多样性（引入其他 Franka state 数据集）。
- 关键对比需要多 seed（至少 3 个），才能写进论文。

### 2026-10-09 12:05 E3a 启动：目标条件先验（后训练；127, c54ee25）
- 问题：如果有人告诉先验"往哪去"（例如 VLM 给出抓取位姿），它能否把"怎么去"做好？预训练对这种后训练有没有帮助？
- 目标类型（都由数据自监督生成）：
  - **endpoint** = horizon 末端（锚点 + 32 步）的状态；
  - **keyframe** = 下一次夹爪开合事件时的状态，时间未知。中位数在 52 步之后，35% 落在 32 步以内，26% 回退为 episode 末端。
  - 训练时按 20% / 40% / 40% 混合无目标 / endpoint / keyframe。
- 注入方式：目标（相对锚点的 EE 位姿 + gripper + 类型 one-hot）经过 MLP 加到 head 的输入上。最后一层零初始化，所以后训练从预训练模型完全等价的状态开始（有测试验证）。backbone 不感知目标。
- 4 个 run，S 模型，H=32，20k 步：
  - `gc_{ar_flow,masked}_ft`：从 `e2_*_f100_s/best.pt` 初始化，lr 1e-4；
  - `gc_*_scratch`：从零训练，lr 3e-4（对照，衡量预训练的价值）。
  - masked 后训练只用 future mask。
- 新基线 `interp@endpoint`：从当前状态直线插值到 endpoint，是已知终点时很强的运动学基线。
- 2k 步时的 val ES：
  - flow_ft：endpoint 2.23（interp 2.89），keyframe 3.67（无目标 4.12）；
  - flow_scratch：endpoint 2.75，keyframe 4.18。

### 2026-10-09 E3a 结果：目标条件先验（127, c54ee25；best.pt 按 val keyframe ES 选择；单 seed）
val（UW），H=32（2.1 s），单位 cm。ADE 对 flow 用样本均值；"中段" 为第 16 步（1.07 s）的误差。

| 模型 | 无目标 ES | endpoint ES | endpoint ADE | endpoint 中段 | keyframe ES | keyframe ADE |
|---|---|---|---|---|---|---|
| const-vel | 8.83 | – | – | – | – | – |
| interp@endpoint | – | 2.89 | 2.54 | 3.81 | – | – |
| 预训练 flow（无后训练） | 3.98 | – | – | – | – | 4.91（无目标） |
| 预训练 masked（无后训练） | 5.28 | – | – | – | – | 4.45（无目标） |
| gc_ar_flow_ft | 3.97 | **1.88** | 2.37 | 3.34 | **3.51** | 4.35 |
| gc_ar_flow_scratch | 4.09 | 1.91 | 2.41 | 3.41 | 3.52 | 4.38 |
| gc_masked_ft | 5.25 | 2.38 | 2.06 | 3.09 | 4.62 | 3.91 |
| gc_masked_scratch | 5.38 | 2.32 | **2.00** | **3.06** | 4.50 | **3.82** |

id_val 上的趋势相同（flow_ft：endpoint ES 1.97 vs interp 3.59，keyframe 3.69 vs 无目标 4.30）。

**结论**
1. **知道"往哪去"后，先验能把"怎么去"做得比直线好**：endpoint 条件下，flow 的 ES 比插值低 35%，masked 的 ADE 低 19%，轨迹中段误差从 3.8 降到 3.1 cm。gripper 误差与插值持平或更好。
2. **只给 keyframe（时间未知）也有帮助，但有限**：ES 和 ADE 都降约 12%。keyframe 中位数在约 3.5 s 之后，且不知道何时到达，所以 2 s 内的约束较弱。
3. **后训练不损害无目标能力**：ft 模型的无目标 ES 为 3.97，预训练模型为 3.98（20% 的无目标采样起了作用）。
4. **在全量 DROID 上，预训练的优势到 20k 步时基本消失**：2k 步时 ft 明显领先（endpoint ES 2.23 vs 2.75），到 20k 步时 scratch 追平，masked scratch 甚至略好。原因是后训练用的是同一份全量数据，scratch 本身就相当于在做预训练。**这个设置测不出预训练的价值**，只能说明预训练能加快收敛。

**下一步**
- 正确衡量预训练价值的方式是**少量目标标注**或**换数据域**：
  - (a) 只用 1% / 10% 的 episode 做目标条件后训练，比较 ft 和 scratch；
  - (b) LIBERO 迁移（E4）。
- 即使已知终点，中段误差仍有约 3 cm：更强的条件注入（让 backbone 也看到目标）和更大的模型可能还有提升空间。

### 2026-10-09 E4a：DROID → LIBERO-90 迁移（预测任务；127, 549ec85）
- **数据**：`IPEC-COMMUNITY/libero_90_no_noops_lerobot` @70696ae（唯一保留关节状态的 LeRobot 版本）。
  - 校验：FK + 基座偏移 (-0.75, 0, 0.912) + 工具偏移 9.65 cm 重建 `ee_state`，残差 0.2 mm。
  - 处理：20 Hz 重采样到 15 Hz；夹爪 = (0.08 − 两指间距) / 0.08；按任务切分。
  - 规模：train 52 个任务 / 2,725 条 demo，val 11 / 551，test 10 / 645；平均 109 帧（7.3 s）。
  - 窗口：左侧 padding 80 帧（用第一帧填充，视为静止）。
- **已知域差异**：
  - LIBERO 去掉了空闲帧（运动更快、更不连贯）；
  - q7 的均值不同（1.25 vs 0.21），夹爪语义不同（夹住物体时约 0.6，DROID 接近 1）；
  - 机器人基座安装方式不同。由于用的是基座坐标系下的 FK，这一项已经消除。
- **零样本**（DROID 预训练的 `e2_*_f100_s`，不做任何后训练）：

| | val ES | val ADE | test ES | test ADE |
|---|---|---|---|---|
| zero-vel | 13.08 | 11.55 | 14.48 | 12.79 |
| const-vel | 17.47 | 14.22 | 17.79 | 14.41 |
| flow 零样本 | **6.91** | 8.06 | **7.63** | 8.84 |
| masked 零样本 | 9.86 | 8.19 | 10.67 | 8.85 |
| （参考）interp@endpoint，已知终点 | 6.08 | – | 6.17 | – |

  → 虽然存在域差异，**DROID 先验零样本迁移到 LIBERO**，ES 比 const-vel 低 60%，ADE 比 zero-vel 低约 30%。
- **后训练**（运行中）：{flow, masked} × {从 DROID ft（lr 1e-4），从零训练（lr 3e-4）} × 每任务 {1, 5, 全部} 条 demo，各 10k 步，按 val ES 选 checkpoint。
- **E4a v1 结果**（10k 步，每 500 步评估一次，ft lr 1e-4；归档在 `outputs/e4a_v1/`）。LIBERO val ES：

| | k=1 | k=5 | 全部 |
|---|---|---|---|
| flow ft | 7.58 | 5.93 | 5.32 |
| flow scratch | 7.50 | 6.61 | 5.71 |
| masked ft | 9.64 | 7.90 | 7.30 |
| masked scratch | 11.80 | 9.39 | 7.91 |
| 零样本（flow / masked） | 6.91 / 9.86 | | |

  - **协议问题**：大多数 run 的最优点出现在第一次评估（500 步），说明小数据下几百步内就开始过拟合。k=1 时 flow ft 反而差于零样本，原因是错过了最优停止点。结论暂不采用。
  - 另外发现每次评估抽到的 val 子集不同（打乱时没有重新设种子）。DROID 实验的 eval_batches 覆盖了 val 的约 90%，影响很小。
- **E4a v2**（4b17929）：在 step 0 也做评估和选择（`eval_at_start`，保证后训练结果不劣于零样本），每 100 步评估一次，固定 val 子集；ft lr 3e-5，scratch lr 3e-4，3k 步。评估 val 和 test。

### 2026-10-09 E4a v2 结果：DROID → LIBERO-90（127, 4b17929；按 val ES 选 checkpoint；单 seed）
ES（cm）；括号中为样本均值 ADE（cm）。test = 10 个没见过的 LIBERO 任务，只评估一次。

| | val k=1 | val k=5 | val 全部 | test k=1 | test k=5 | test 全部 |
|---|---|---|---|---|---|---|
| const-vel | 17.47 | | | 17.79 | | |
| zero-vel | 13.08 | | | 14.48 | | |
| flow 零样本（无后训练） | 6.91（8.06） | | | 7.63（8.84） | | |
| flow scratch（无预训练） | 7.36（9.26） | 6.29 | 5.73（6.94） | 8.76（10.79） | 7.93 | 7.38（8.67） |
| **flow 预训练 + ft** | **5.67（6.91）** | **5.35** | **5.13（6.35）** | **6.57（7.93）** | **6.34** | **6.14（7.47）** |
| masked 零样本 | 9.86（8.19） | | | 10.67（8.85） | | |
| masked scratch | 11.39（10.05） | 9.34 | 8.08（6.88） | 12.91（11.36） | 11.64 | 11.28（9.67） |
| **masked 预训练 + ft** | **8.37（6.97）** | **7.72** | **7.32（6.08）** | **8.77（7.34）** | **8.72** | **8.94（7.45）** |

**结论**
1. **预训练 + 后训练在所有设置下都最好**：两种目标、三种数据量、val 和 test 都成立。
2. **数据效率约 50×**：每任务只用 1 条 demo 的 ft，优于或持平使用全部约 52 条 demo 的 scratch。
   - flow：val 5.67 vs 5.73，test 6.57 vs 7.38；
   - masked：test 8.77 vs 11.28。
3. **数据充足时优势仍在**：全部数据下 ft 比 scratch 低 10–21%（flow test 6.14 vs 7.38，masked test 8.94 vs 11.28）。
4. **少量后训练就超过零样本**：flow 在 k=1 时 val 从 6.91 降到 5.67（−18%）。最优点在 100–200 步，说明主要是对新域（速度、夹爪语义）的快速适配，而不是重新学习运动。
5. 在 held-out 任务（test）上，预训练的相对优势比 val 更大，说明预训练提升的是对新任务的泛化。

**局限**：单 seed，few-shot 子集只抽了一次；评估的是预测能力，不是闭环控制。下一步是 E4b（视觉策略的闭环成功率）和多 seed。

### 2026-10-09 E4b 启动：LIBERO 闭环少样本策略（127, 17c5cc8）
- **仿真**：LIBERO 部署在 127（hf-libero 0.1.4、robosuite 1.4.0、mujoco 3.8.1，EGL 正常）。使用独立的 `LIBERO_CONFIG_PATH`，不改动共享的 `~/.libero`。
  - 已验证图像方向：IPEC 数据相对原始渲染旋转了 180°，因此仿真图像两轴都翻转。
  - 已验证夹爪约定：数据中 1 = 开 / 0 = 关，仿真中 −1 = 开 / +1 = 关。
- **任务**：LIBERO-90 中有 90 个任务，但只有 74 种不同的指令；IPEC 把同一指令的不同场景合并了。因此只用 test split 中**指令唯一的 7 个任务**（E4a 从未见过），每个任务 39–49 条 demo。
- **策略**（共享 ResNet18 + 任务嵌入编码器，各 arm 只在动作产生方式上不同）：
  - `prior_*`：图像和任务作为条件（零初始化，加在 head 输入上）→ ar_flow 先验预测 15 Hz 下 2.1 s 的未来状态；每 8 个控制步重新规划一次；IDM 根据实时状态和计划中接下来 4 步的目标状态输出 20 Hz 的动作。trunk 有三种：scratch / DROID 预训练 / DROID→LIBERO 后训练（E4a ft kall，只见过 train 任务）。
  - `bc`：常规基线，图像 + 任务 + 当前状态 → 16 步动作块（不用先验、不用 IDM）。
- **IDM**：在 LIBERO-90 train-split 任务上训练（只用 state + action）。在 held-out 的 val 任务上，OSC L1 为 0.019（零动作为 0.14），**夹爪准确率 98.4%**，不会是瓶颈。
- **训练与评估**：每任务 5 条 / 全部 demo；15k 步，batch 64，lr 1e-4，所有 arm 设置相同，使用最终 checkpoint（不做闭环选模）。评估为 7 个任务 × 20 个官方初始状态，最多 400 步，所有 arm 使用相同的种子。

### 2026-10-09 E4b v1 结果：先验 + IDM 路线失败；BC 基线很强（127, 17c5cc8）
闭环成功率（7 个任务 × 20 个初始状态）：

| 策略 | 每任务 5 条 | 全部 |
|---|---|---|
| prior scratch + IDM | 10.7% | 12.1% |
| prior DROID + IDM | 5.7% | 7.9% |
| prior DROID→LIBERO + IDM | 9.3% | 10.0% |
| **BC（直接输出动作）** | **65.0%** | **91.4%** |

先验类策略几乎只在"close the bottom drawer"（不需要抓取）上成功。

**离线诊断**（`scripts/e4b_diagnose.py`，k=5 策略在 held-out demo 上）：
- 真实未来状态 + IDM：OSC L1 0.033，夹爪准确率 96.9% → **IDM 没有问题**。
- 先验计划 + IDM：L1 0.126，夹爪准确率 82.8%；BC：0.072，92.9%。
- 0.2 s 时的位置误差：先验单样本 1.93 cm，8 样本均值 1.12 cm，**const-vel 0.75 cm**，zero-vel 2.42 cm。

**原因**：先验的预测目标在整个 2.1 s 内只用一个归一化尺度（delta_std，在 k ∈ [1, 32] 上合并统计），所以近端（0.05–0.2 s）的位移相对于这个尺度极小，被 flow 采样噪声淹没。IDM 恰好依赖这一段 → 动作接近噪声。

**方法层面的教训**（写入设计）：
1. 先验应该按 horizon 分步归一化，或预测相对 const-vel 的残差，否则近端精度不足；
2. 用于控制时，应该把先验当作表示/trunk，或者只在远端提供引导，而不是逐步跟踪它的样本。

**E4b v2**（ae12c7d，运行中）：
- `trunkbc_*`：先验 Transformer 作为 BC trunk（输入 15 Hz 状态历史，加图像和任务条件，输出动作块，与 BC 相同的头和损失）；trunk 分 scratch / DROID / DROID→LIBERO 三种；
- 先验 + IDM 改用 16 样本均值作为计划，重新评估。

### 2026-10-09 E1c：按步归一化（per-step normalization）修复近端精度（127, 1c7e631）
`e1c_ar_flow_perstep_s` = `e2_ar_flow_f100_s` 的设置 + `model.per_step_norm=true`（第 k 步的增量用第 k 步自己的 std 归一化）。DROID val，第 k 步的位置误差（cm）：

| | 第 1 步 | 第 3 步（0.2 s） | 第 8 步 | 第 16 步 | 第 32 步 | ES |
|---|---|---|---|---|---|---|
| 原先验，单样本 | 0.33 | 1.34 | 4.13 | 7.04 | 12.91 | 3.98 |
| **按步归一化，单样本** | **0.22** | **0.89** | **3.04** | 6.99 | 13.79 | **3.96** |
| 原先验，样本均值 | 0.18 | 0.69 | 2.43 | 4.97 | 9.19 | |
| **按步归一化，样本均值** | **0.13** | **0.56** | **2.13** | 4.94 | 9.26 | |
| const-vel | 0.16 | 0.68 | 2.75 | 6.89 | 15.44 | |

→ 近端误差降低 25–34%，样本均值在近端也优于 const-vel；远端误差和 ES 基本不变。**今后的预训练默认应开启按步归一化。**
E4b v3（374d20c，运行中）：用按步归一化的先验做先验 + IDM（scratch 与 DROID 对比，每任务 5 条 / 全部 demo），计划取 16 样本均值。

### 2026-10-09 E4b v2 结果：先验 Transformer 作为 BC trunk（127, ae12c7d；单个训练 seed，140 个配对 episode）
闭环成功率（7 个任务 × 20 个官方初始状态，各 arm 使用相同的种子；最终 checkpoint）：

| 策略 | 每任务 5 条 | 全部（约 44 条） |
|---|---|---|
| BC（MLP，只用当前状态） | 65.0% | 91.4% |
| trunk-BC，scratch | 57.1% | 87.9% |
| **trunk-BC，DROID 预训练** | **71.4%** | **95.7%** |
| trunk-BC，DROID→LIBERO 后训练 | 67.1% | 95.0% |

各任务（k5，DROID trunk vs scratch trunk）：drawer 0.90 vs 0.55，cream cheese 0.70 vs 0.70，tomato sauce 0.75 vs 0.80，ketchup→drawer 0.45 vs 0.30，white bowl 0.85 vs 0.75，wine bottle 0.55 vs 0.20，mug 0.80 vs 0.70。

配对 McNemar 精确检验（只在一方成功的 episode 数）：
- DROID trunk vs scratch trunk：k5 27 vs 7，**p = 0.0008**；全部 15 vs 4，**p = 0.019**。
- DROID→LIBERO trunk vs scratch trunk：k5 p = 0.016；全部 p = 0.041。
- DROID trunk vs BC：k5 21 vs 12，p = 0.16；全部 10 vs 4，p = 0.18（方向占优但不显著）。
- scratch trunk vs BC：k5 p = 0.07（scratch trunk 更差）；全部 p = 0.33。

**结论**
1. **DROID 的 state-only 预训练显著提升同一架构的闭环成功率**（+14.3 / +7.8 个百分点）。它把一个从零训练时比 BC 还差的历史 Transformer trunk（scratch 57% < BC 65%）变成最好的策略（71%），说明预训练缓解了长状态历史带来的过拟合 / copycat 问题。
2. 先在其他 LIBERO 任务上后训练没有额外收益（67% / 95% vs 71% / 96%）：DROID 预训练已经足够，LIBERO 后训练可能对 train 任务过于特化。
3. 相对 BC 的优势方向一致但还不显著；需要更多 seed 或更多初始状态。

### 2026-10-09 E4b v3 结果：按步归一化先验 + IDM（127, 374d20c / 951c42e；计划取 K=16 样本均值）

| 策略 | 每任务 5 条 | 全部 |
|---|---|---|
| prior DROID（旧归一化）+ IDM，K=1 | 5.7% | 7.9% |
| prior DROID（旧归一化）+ IDM，K=16 | 14.3% | 10.0% |
| prior 按步归一化 scratch + IDM，K=16 | 17.1% | 17.1% |
| **prior 按步归一化 DROID + IDM，K=16** | 16.4% | **32.9%** |
| （参考）BC / trunk-BC DROID | 65.0% / 71.4% | 91.4% / 95.7% |

- 按步归一化 + 多样本均值让这条路线从 8% 提升到 33%（全部 demo）。全部 demo 时 DROID 预训练显著优于 scratch（29 vs 7，p = 0.0003）；k5 时没有差异。
- 但先验 + IDM 仍然**远不如**直接输出动作的策略。成功几乎集中在不需要精细抓取的任务上（drawer）。按状态逐步跟踪对精度要求太高，在这个数据规模下不适合作为控制接口。
- **结论**：state-only 先验对控制最有效的接口是**表示 / trunk 初始化**（E4b v2），而不是"预测状态 → IDM"。

### 2026-10-09 E4b 多 seed 启动（127, fb03c93）
- BC / trunk-BC scratch / trunk-BC DROID × {k5, 全部} × 训练 seed {1, 2}（seed 0 即 v1/v2 的结果）。评估与之前相同（7 个任务 × 20 个初始状态，评估种子相同），用于跨 seed 的配对检验。

### 2026-10-09 E5 启动：SmolVLA + SOPT 运动 token（Jubail job 18731435, a49f62d）
- **接入方式**：替换 SmolVLA 的 `state_proj`。SmolVLA 的 prefix 构造本来就支持 (B, N, d) 形式的多 token 状态，这些 token 放在图像和语言之后，action expert 通过 KV cache 读取。
  - 新的状态输入是 6.4 s（15 Hz）的 SOPT 状态特征历史，展平传入（因为 SmolVLA 的 `prepare_state` 对 3-D 状态只保留最后一帧；在加载权重后把 `max_state_dim` 设为 96×17，使 padding 成为 no-op）；
  - 经过 SOPT backbone 得到 24 个运动 token，再加 1 个当前状态 token，都投影到 VLM 宽度。
- **arms**：
  - `none`：只有当前状态 token，相当于原版 SmolVLA（状态改用 SOPT 的 17 维特征）；
  - `scratch`：运动编码器随机初始化（控制 token 数和历史信息带来的影响）；
  - `droid`：DROID 预训练的 SOPT backbone。
  - 图像、语言、VLM（冻结）、action expert（可训练）在各 arm 之间完全相同。
- **数据**：与 E4b 相同的 7 个任务和 demo 选择，图像为原生 256 分辨率，chunk = 50，每次执行 10 步。20k 步，batch 32，lr 1e-4（SmolVLA 默认的 AdamW / cosine 设置）。
- **算力**：127 上与 E4b 共享 GPU 时只有约 0.2 it/s，因此改在 Jubail 上训练，一个 job 用 3 张 A100，复用 RoboticsNAS 的 python 环境（只读）。评估时 7 个任务并行推进，只保留一份策略副本。
- 设计构思：`docs/IDEAS_WM_ICL.md`（WM：先验作为 CEM / MPPI 的提议分布，先用仿真器作为 oracle WM 验证；ICL：同族多 episode 上下文，在 LIBERO 上先验证"示范携带目标位置"）。

### 2026-10-09 E4b 多 seed 结果（3 个训练 seed × 140 个配对 episode；127）
闭环成功率（seed 0 / 1 / 2，均值 ± 标准差）：

| 策略 | k5 | 全部 |
|---|---|---|
| BC | 0.650 / 0.614 / 0.643 → **63.6 ± 1.9%** | 0.914 / 0.907 / 0.950 → **92.4 ± 2.3%** |
| trunk-BC scratch | 0.571 / 0.607 / 0.593 → **59.0 ± 1.8%** | 0.879 / 0.907 / 0.893 → **89.3 ± 1.4%** |
| **trunk-BC DROID** | 0.714 / 0.721 / 0.664 → **70.0 ± 3.1%** | 0.957 / 0.964 / 0.971 → **96.4 ± 0.7%** |

配对比较（各 seed 的差值；把 3 个 seed 的 McNemar 不一致对合并后做精确检验，近似把各 seed 的 episode 视为独立配对）：
- DROID trunk vs scratch trunk：k5 [+14.3, +11.4, +7.1] 个百分点，70 vs 24，**p = 2.2e-6**；全部 [+7.9, +5.7, +7.9]，41 vs 11，**p = 3.6e-5**。
- DROID trunk vs BC：k5 [+6.4, +10.7, +2.1]，68 vs 41，**p = 0.012**；全部 [+4.3, +5.7, +2.1]，28 vs 11，**p = 0.0095**。
- scratch trunk vs BC：k5 p = 0.056，全部 p = 0.066（scratch trunk 略差）。

**结论**（3 个 seed，6 组对比方向全部一致）：DROID state-only 预训练让历史 trunk 从"不如 BC"变为"显著优于 BC"。k5 时 +6.4 个百分点（相对 scratch trunk +11.0），全部 demo 时 +4.0 个百分点（失败率从 7.6% 降到 3.6%，约减半）。

### 2026-10-10 E5 结果：SmolVLA + SOPT 运动 token（Jubail job 18731435，3×A100，3 h 48 min；单 seed）
闭环成功率（7 个任务 × 20 个初始状态，与 E4b 的设置相同）：

| SmolVLA | k5 | 全部 |
|---|---|---|
| none（只有当前状态 token） | 56.4% | **85.0%** |
| scratch（运动 token，随机初始化） | 55.7% | 82.1% |
| **droid（运动 token，DROID 预训练）** | **65.0%** | 80.7% |

各任务（k5，none / scratch / droid）：drawer 0.60 / 0.95 / 0.95，cream cheese 0.65 / 0.65 / 0.70，tomato sauce 0.40 / 0.50 / 0.80，ketchup 0.40 / 0.15 / 0.20，white bowl 0.80 / 0.85 / 0.90，wine 0.65 / 0.45 / 0.65，mug 0.45 / 0.35 / 0.35。

配对 McNemar 检验：
- k5：droid vs none 24 vs 12（p = 0.065）；droid vs scratch 20 vs 7（**p = 0.019**）；scratch vs none p = 1.0。
- 全部：droid vs none 14 vs 20（p = 0.39）；其余比较也都不显著。

**初步解读**：少数据时提升来自**预训练本身**（随机初始化的运动 token 没有收益）；数据充足时没有收益，甚至略有下降（不显著）。这和 E4b 一致：低数据下收益最大。不过 SmolVLA 本身在全部数据时（85%）弱于我们的小模型 BC（92%），说明 20k 步、冻结 VLM 的配置对这个任务集并不是最优。

**下一步**：k5 补跑 seed 1 和 seed 2（`scripts/slurm/e5_vla_seeds.sbatch`），确认显著性。
