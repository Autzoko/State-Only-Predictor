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

### 2026-10-10 E5 k5 多 seed 结果（Jubail job 18734693，cn016，aad555f；3 个训练 seed × 140 个配对 episode）
| SmolVLA k5 | seed 0 | seed 1 | seed 2 | 均值 ± 标准差 |
|---|---|---|---|---|
| none | 56.4 | 59.3 | 65.7 | 60.5 ± 4.8% |
| scratch 运动 token | 55.7 | 57.9 | 58.6 | 57.4 ± 1.5% |
| droid 运动 token | 65.0 | 57.9 | 60.7 | 61.2 ± 3.6% |

合并配对 McNemar 检验（420 对）：droid vs scratch 52 vs 36（p = 0.11）；droid vs none 55 vs 52（p = 0.85）；scratch vs none 45 vs 58（p = 0.24）。

各任务合并（none / scratch / droid）：drawer 77 / 92 / 90，cream cheese 65 / 67 / 72，tomato sauce 50 / 50 / 68，ketchup 38 / 20 / 23，white bowl 87 / 87 / 93，wine 57 / 55 / 60，mug 50 / 32 / 22。

**结论**：seed 0 的 +8.6 个百分点**没有复现**。在 SmolVLA 上，DROID 运动 token 相对无历史基线**没有显著收益**。方向上仍是 droid ≥ none ≥ scratch，"scratch 历史有害、预训练历史可以把损害补回来"这一模式还在，但只是弱趋势，不显著。seed 间方差（none 的标准差 4.8 pp）和单 seed 的效应大小相当，单 seed 的 VLA 结论不可靠。与 E4b（小策略 trunk，3 seed 都显著）相比，当前的接入方式（运动 token 进入 VLM 前缀，VLM 冻结、20k 步）没能把先验的作用传递到 SmolVLA。

可能原因（待验证）：
1. 运动 token 只是 prefix 中的 25 个 token，action expert 通过交叉注意力读取，信号可能被约 200 个视觉 token 稀释。
2. 在 E4b 中先验作为 trunk 端到端微调；这里 SOPT 是端到端可训的，但学习率和调度跟随 SmolVLA（2.5e-6 余弦衰减），对 SOPT 部分可能不合适。
3. 两个任务（ketchup、mug）上带历史的两个 arm 都明显更差，提示存在 copycat 类失败，值得做扰动分析。

### 2026-10-11 E6 启动：预训练先验作为动作 decoder（MotionDecoder；127, 8d5f180）
动机：E5 中 SOPT 只是 SmolVLA 前缀里的输入 token，经过冻结的 VLM 后才被动作模块读取，3 个 seed 不显著。TrunkBC（先验直接连动作头）有效。E6 让先验本身成为动作 decoder：
- 视觉：ResNet18 的空间特征图（2 个相机 × 4×4 = 32 个 token）+ 1 个任务 token。
- 在先验的每一层前插入交叉注意力 + MLP，输出投影零初始化，所以初始化时 decoder 与预训练先验完全相同（单测验证）。
- 最后一个 token 同时驱动两个头：新的动作块头（L1 + 夹爪 BCE，与 TrunkBC 相同）和预训练的 flow 头，后者预测未来 32 帧状态（"本体预见"，权重 1.0）。两个头共享特征，动作直接受监督，不经过状态到动作的换算。
- 训练设置与 E4b 相同：15k 步，bs 64，lr 1e-4（先验部分同样 1e-4），7 个任务 × 20 个初始状态评估。

组别（3 个 seed）：k5 下有 droid、scratch、droid_noaux（去掉预见损失）；全部示范下有 droid、scratch。对照直接使用 E4b 的 BC 和 TrunkBC 多 seed 结果。队列为 `scripts/remote/e6_queue.sh`，tmux 会话 `sopt_e6`。冒烟测试：12 it/s（单独跑），评估流程正常。

### 2026-10-11 E6 k5 结果：MotionDecoder（127, 8d5f180；3 个训练 seed × 140 个配对 episode；汇总脚本 `scripts/agg_e6.py`）
| k5 | seed 0 | seed 1 | seed 2 | 均值 ± 标准差 |
|---|---|---|---|---|
| BC（E4b） | 65.0 | 61.4 | 64.3 | 63.6 ± 1.9% |
| TrunkBC scratch（E4b） | 57.1 | 60.7 | 59.3 | 59.0 ± 1.8% |
| TrunkBC DROID（E4b） | 71.4 | 72.1 | 66.4 | 70.0 ± 3.1% |
| MotionDecoder scratch | 67.9 | 56.4 | 69.3 | 64.5 ± 7.0% |
| **MotionDecoder DROID** | 73.6 | 68.6 | 71.4 | **71.2 ± 2.5%** |
| MotionDecoder DROID，去掉预见损失 | 67.9 | 69.3 | 70.7 | 69.3 ± 1.4% |

合并配对 McNemar 检验（420 对）：
- MD DROID vs MD scratch：67 vs 39，**p = 0.008**
- MD DROID vs BC：82 vs 50，**p = 0.007**
- MD DROID vs TrunkBC DROID：57 vs 52，p = 0.70
- MD DROID vs 去掉预见损失：52 vs 44，p = 0.48
- MD scratch vs TrunkBC scratch：75 vs 52，p = 0.05
- MD scratch vs BC：p = 0.78

各任务（BC / TrunkBC DROID / MD scratch / MD DROID）：cream cheese 53 / 68 / 58 / **93**，wine 37 / 50 / 43 / **62**，drawer 83 / 78 / 93 / 85，tomato 78 / 80 / 68 / 70，ketchup 33 / **52** / 42 / 33，bowl 87 / 82 / 85 / 87，mug 73 / 80 / 62 / 68。

**解读**：
1. 在 decoder 形式下，预训练的收益依然稳定：比同结构 scratch 高 6.7 个百分点（p = 0.008），比 BC 高 7.6 个百分点（p = 0.007）。
2. MotionDecoder DROID 是目前 k5 下最高的结果，但相对 TrunkBC DROID 只高 1.2 个百分点，不显著。在 k5 下，结构上的改动（逐层交叉注意力 + 空间视觉 token）没有带来显著的额外收益。它主要帮了 scratch（59.0 → 64.5，p = 0.05，但 seed 间方差很大）。
3. 预见损失 +1.9 个百分点，不显著，seed 间方差变大。目前不能声称它有效。
4. 各任务差异大：cream cheese 和 wine 大幅提升，ketchup 下降。这提示两种结构学到的是不同的东西，值得后续分析。

全部示范的组正在训练。

### 2026-10-11 E6 全部示范结果（127；3 个训练 seed × 140 个配对 episode）
| 全部示范 | seed 0 | seed 1 | seed 2 | 均值 ± 标准差 |
|---|---|---|---|---|
| BC（E4b） | 91.4 | 90.7 | 95.0 | 92.4 ± 2.3% |
| TrunkBC scratch（E4b） | 87.9 | 90.7 | 89.3 | 89.3 ± 1.4% |
| TrunkBC DROID（E4b） | 95.7 | 96.4 | 97.1 | 96.4 ± 0.7% |
| MotionDecoder scratch | 94.3 | 85.7 | 90.7 | 90.2 ± 4.3% |
| **MotionDecoder DROID** | 96.4 | 96.4 | 95.7 | **96.2 ± 0.4%** |

配对检验：MD DROID vs MD scratch 33 vs 8，**p = 1e-4**；vs BC 27 vs 11，**p = 0.014**；vs TrunkBC DROID 13 vs 14，p = 1.0。MD scratch vs BC p = 0.24。

**E6 总结**：预训练的收益在两种数据量下都稳定复现（k5 p = 0.008，全部 p = 1e-4）。但在 ResNet 视觉条件下，"先验作为 decoder"这个结构与 TrunkBC 打平（71.2 vs 70.0，96.2 vs 96.4）。收益来自预训练本身，而不是结构。scratch 版本的 seed 间方差明显更大（7.0 / 4.3），预训练版本更稳定（2.5 / 0.4）。

### 2026-10-12 防泄露审计：LIBERO 评估初始状态 vs 训练示范的初始场景（127, 53c19d6）
问题：如果评估用的官方初始状态和某条训练示范的初始场景相同，策略就可能只是在背轨迹。
方法（`scripts/check_init_leakage.py`）：对 7 个任务各渲染 50 个官方初始状态（与评估相同的 settle 步数），和该任务所有示范的第一帧比较 64 px RMS 距离，统计最近示范距离、示范之间的距离分布，以及最近 / 次近距离之比。如果某个场景是从示范复制的，比值会接近 0。

结果：每个任务中，初始状态到最近示范的距离中位数为 0.027–0.048，和示范之间距离的 5% 分位数（0.027–0.050）相当，没有接近 0 的匹配。最近 / 次近距离之比最小 0.52，中位数 0.92–0.95。**结论：评估场景不是训练示范场景的复制，不存在这种泄露。**其他防泄露设置：
- SOPT 只在 DROID 训练 split 上预训练，不含任何 LIBERO 数据。
- 7 个评估任务来自 LIBERO-90 的 test 任务 split，E4a 的后训练从未见过。
- 策略只使用这 7 个任务的示范（k5 按 salted hash 选取，嵌套）。
- 归一化统计来自 DROID（SOPT）或训练示范（动作）。
- smolvla_base 的预训练数据不含 LIBERO。

### 2026-10-12 E7 启动：SOPT 调制 SmolVLA 的动作专家（Jubail job 18769062，3×A100，8fc2e4c）
E5 中运动 token 放在 VLM 前缀里，结果不显著。E7 改为：前缀与 none 组完全相同（只有当前状态 token）；SOPT 最后一个 token 的特征经过一个最后一层零初始化的 MLP，加到动作专家的每个动作 token 上（位置在 SmolVLA 的 action-time MLP 之后）。初始化时策略与 none 完全一致。这对应小策略中有效的接法：运动特征紧挨动作输出。
- 组别：expert_droid、expert_scratch，k5，seed 0–2。训练设置与 E5 相同（20k 步，bs 32，峰值 lr 1e-4，SOPT 同 lr）。
- 评估：每任务 50 个初始状态（每个 seed 350 个配对 episode，比之前 140 个更有统计功效），E5 的 none checkpoint 也用 50 个初始状态重新评估（`rollouts_n50.json`）。
- 冒烟测试（job 18769001）：2.7 it/s，评估正常。

### 2026-10-12 E8 启动：SOPT 与强预训练视觉编码器（冻结 DINOv2-S）组合（127, 62bfd8f）
问题：SOPT 的收益是否只是弥补了较弱的 ResNet18？如果换成冻结的 DINOv2-S/14（CLS + patch 均值，224 px），预训练的本体表示是否仍有增益？
组别：BC、TrunkBC scratch、TrunkBC DROID，视觉统一用 `--vision dinov2`；k5 和全部示范，各 3 个 seed，50 个初始状态。队列为 `scripts/remote/e8_queue.sh`，tmux 会话 `sopt_e8`。

### 2026-10-12 E8 k5 结果（冻结 DINOv2-S，50 个初始状态 × 7 任务 × 3 seed，按 seed 池化的配对 McNemar 检验，`scripts/agg_e8.py`）
| 组别 | s0 / s1 / s2 | 均值 ± SD |
|---|---|---|
| BC（DINOv2） | 55.1 / 51.1 / 64.6 | 57.0 ± 6.9 |
| TrunkBC scratch | 51.4 / 54.9 / 57.7 | 54.7 ± 3.1 |
| TrunkBC DROID | 58.6 / 58.3 / 62.6 | **59.8 ± 2.4** |

- DROID 对 scratch：194 对 140，p = 0.0037。"翻转"在冻结 DINOv2 上复现。
- DROID 对 BC：174 对 144，p = 0.10，方向一致但**不显著**。
- scratch 对 BC：168 对 192，p = 0.23（从零训练的历史没有收益）。
- 观察：冻结 DINOv2 的 BC（57.0）比可微调的 ResNet18 BC（E4b 是 63.6，但用的是 20 个初始状态）更弱。k5 下冻结特征 + MLP 头的容量不足，可能压低了所有组别。
- 全部示范阶段仍在运行。

### 2026-10-12 E9a 失败归类（127）：主流小策略主要败在"怎么做"，而不是"去哪"
目的：确认"着重优化动作"是否对症。我们给 rollout 增加了可选的仿真状态记录（`--trace 4`：EE 位置、手指开度、夹爪命令、物体位置；只用于诊断，不输入策略），然后按以下规则分类（`scripts/failure_taxonomy.py`）：
- 目标物体被抬起但任务未完成 → place_fail；
- EE 曾接近目标物体（距离 < 该物体成功抓取时的中位距离 + 6 cm）但未抬起 → grasp_miss；
- 从未接近目标，而其他物体被移动 → wrong_target，否则 → no_reach。

e4b_bc_k5（seed 0，每任务 50 个初始状态，成功率 64%）：
- 126 次失败中，grasp_miss 81 次，place_fail 22 次，即 **82% 是执行问题**；
- wrong_target 11 次、no_reach 6 次（13%），抽屉任务失败 6 次。
- ketchup 的失败 EE 都停在约 8.7 cm 处（成功时为 3.3 cm），即"到了附近，但没有对准或下降"。

完整结果（seed 0，每任务 50 个初始状态，共 350 个 episode；数字为 episode 数）：

| 策略 | 成功 | 错误目标 + 未接近（"去哪"） | grasp_miss | place_fail | 抽屉失败 |
|---|---|---|---|---|---|
| BC k5 | 224 | 17 | 81 | 22 | 6 |
| TrunkBC scratch k5 | 214 | 37 | 63 | 19 | 17 |
| TrunkBC DROID k5 | **248** | 25 | **53** | 18 | 6 |
| BC 全部示范 | 325 | 0 | 18 | 5 | 2 |
| TrunkBC DROID 全部示范 | **335** | 0 | **9** | 6 | 0 |

- **SOPT 的增益几乎全部来自执行**：k5 下 grasp_miss 从 81 降到 53（−35%），全部示范下从 18 降到 9（−50%）；"去哪"类失败没有减少（17 → 25）。这与"SOPT 是运动先验"的定位一致。
- 从零训练的历史 trunk 在"去哪"上更差（37），抽屉任务也更差（17）。这符合 copycat：模型依赖历史而不是视觉。
- 所以下一步应该同时改进"去哪"（视觉/关键帧）和"怎么做"（SOPT），E10 正是这样分工。

### 2026-10-12 E10 启动：关键帧（"去哪"）+ 预训练的目标条件 SOPT（"怎么做"）（127）
- 动机：E9a 显示失败以执行为主；E3a 显示给定关键帧后，目标条件先验的运动明显优于直线插值。
- `KFMotor`：
  - 策略隐状态（SOPT 历史 trunk + 视觉/任务，同 TrunkBC）预测下一个关键帧，即下一次夹爪事件时的 EE 位姿和夹爪状态，时间未知。关键帧直接来自示范，没有额外标注，与 DROID 的定义一致。
  - 关键帧通过 SOPT **预训练好的目标通路**（E3a 在 DROID 上用事后关键帧后训练）同时条件化动作头和 flow 头的 foresight。
  - 训练时以 0.5 的概率使用真实关键帧，否则使用预测值（teacher forcing）；测试时用预测值。也可以由外部模块（如 VLM 指点）提供关键帧，这是与主流模型互补的接口。
- 组别：k5，seed 0–2，15k 步，每任务 50 个初始状态。
  - `e10_kf_gcdroid`：先验 = `gc_ar_flow_ft`（只用 DROID）；
  - `e10_kf_droid`：先验 = `e2_ar_flow_f100_s`，目标通路新建并零初始化；
  - `e10_kf_scratch`：从零训练。
  - 同时把 BC / TrunkBC scratch / TrunkBC DROID 的 k5 三个 seed 在 50 个初始状态上重新评估，用于配对比较。
- 防泄露：所有先验只在 DROID 训练 split 上训练；LIBERO 只用 7 个 test 任务的 k5 示范；关键帧来自这些示范本身。
- 冒烟测试：训练 8 it/s，闭环评估正常。

### 2026-10-12 E7 结果：SOPT 调制 SmolVLA 动作专家，**显著提升**（Jubail 18769062；k5，3 个 seed，每任务 50 个初始状态，`scripts/agg_e7.py`）
| 组别 | s0 / s1 / s2 | 均值 ± SD |
|---|---|---|
| SmolVLA（none，只有当前状态） | 58.9 / 58.3 / 66.6 | 61.2 ± 4.6 |
| + SOPT 动作专家条件（scratch） | 62.0 / 56.9 / 64.9 | 61.2 ± 4.1 |
| + SOPT 动作专家条件（DROID 预训练） | 68.6 / 64.3 / 65.1 | **66.0 ± 2.3** |

按 seed 池化的配对 McNemar 检验（1050 个配对 episode）：
- DROID 对 none：132 对 82，**p = 0.0008**；
- DROID 对 scratch：123 对 73，**p = 0.0004**；
- scratch 对 none：72 对 72，p = 1。

结论：
- **接口是关键**。E5 把运动 token 放进 VLM 前缀，结果不显著；E7 把 SOPT 特征经零初始化 MLP 加到动作专家的每个动作 token 上，增益显著，而且 3 个 seed 都高于 none。这和小策略中的发现一致：运动特征应该紧挨动作输出。
- **"翻转"在 VLA 上复现**：从零训练的 SOPT 与 none 完全持平；只有预训练的 SOPT 带来增益。
- 这是"SOPT 可以与预训练 VLA 组合、带来增益"的第一个显著证据。SmolVLA 本身完全不用重新预训练。

### 2026-10-12 E7b 提交（Jubail）：SmolVLA 失败归类 + 全部示范
- 用 `vla_eval.py --trace 4` 记录 e5_vla_none_k5 与 e7_vla_expert_droid_k5_s0 的仿真状态，做失败归类。
- 同样的三组在全部示范下各跑 3 个 seed，检验增益在数据充足时是否保持。

### 2026-10-12 E10 seed 0（早期信号，未下结论）
k5，每任务 50 个初始状态：
- KFMotor gcdroid 66.0；
- KFMotor scratch 61.4；
- 对照：TrunkBC DROID 70.9（同一组初始状态，来自 E9a trace 评估），BC 64.0。

关键帧瓶颈目前不如直接的 TrunkBC，主要输在抽屉任务（68 vs 88）和 tomato sauce 任务（58 vs 76）。预训练相对 scratch 仍有 +4.6 的增益。等 3 个 seed 跑完再判断。

### 2026-10-12 E11 排队（127，E8 完成后自动开始）：预训练规模 → 下游
- TrunkBC k5，先验分别为 DROID 1% / 10% / 100%（现有 S checkpoint）以及 B 模型（d512 × 12 层，在 100% DROID 上按同一配方预训练，lr 2e-4）。
- 3 个 seed，每任务 50 个初始状态。
- 目的是验证"更多、更大的纯状态预训练 → 更好的下游策略"（暗数据扩大规模的论点）。

### 2026-10-12 E12 启动：SOPT + FastWAM（世界动作模型）（Jubail）
- **X = FastWAM**（LeRobot 0.6.2）：Wan2.2-TI2V-5B 视频 DiT + 1B 动作 DiT（Mixture-of-Transformers）。
  - 基座 `lerobot/fastwam_base`：Wan2.2 权重，动作专家从 Wan DiT 插值得到；没有机器人数据，也没有数据统计。动作 encoder/head 不在 checkpoint 中，需要新建。因此不存在 LIBERO 泄露。
- **微调**：所有组完全相同。
  - 冻结视频专家、VAE 和文本编码器；训练动作专家（10.3 亿参数）和 proprio encoder。
  - 损失为 FastWAM 自己的 flow-matching 动作损失，走推理路径（当前帧 → 32 步动作块），不训练视频损失（显存所限）。
  - proprio token 为当前 17 维 SOPT 特征。
  - 10k 步，bs 16，lr 1e-4（warmup 500，余弦退火）。
- **SOPT 接入**：与 E7 完全相同的思路。SOPT 历史的最后一个 token 经零初始化 MLP 加到每个动作 token 上（动作 encoder 之后），所以初始化时等于 none。
- **组别**：none / expert_scratch / expert_droid，k5，3 个 seed（每个 seed 一个 job，3 卡并行），每任务 50 个初始状态。
- **冒烟测试**（job 18771945）：1.19 it/s，46.6 GB，闭环评估正常（200 步时 1 个初始状态为 1/7）。
- 依赖：diffusers 0.39 和 ftfy 装在 `/scratch/ll5582/SOPT/pydeps`（不改动 RoboticsNAS 环境）。权重缓存在 `SOPT_DATA/hf_cache`。

### 2026-10-12 E7b 第一部分：SmolVLA 失败归类（seed 0，每任务 50 个初始状态，`--trace 4`）
| 策略 | 成功 | 错误目标 + 未接近 | grasp_miss | place_fail | 抽屉失败 |
|---|---|---|---|---|---|
| SmolVLA none | 205 | 15 | 97 | 20 | 13 |
| SmolVLA + DROID SOPT（动作专家侧） | **238** | 12 | **84** | **12** | **4** |

- **VLA 的失败同样以执行为主**：none 的 145 次失败中，grasp_miss + place_fail + 抽屉失败共 130 次（90%），"去哪"类只有 15 次。这说明"主流模型在基础任务上败在执行"不只是小策略的现象。
- SOPT 减少的仍然是执行类失败：grasp_miss −13，place_fail −8，抽屉失败 −9；"去哪"类基本不变（15 → 12）。
- 与小策略（E9a）的机制一致：SOPT 是动作侧的运动先验。

### 2026-10-12 E13 准备：标准 LIBERO 四个 suite（Spatial / Object / Goal / Long），可与文献数字对照
- **数据**：IPEC `libero_{spatial,object,goal,10}_no_noops_1.0.0_lerobot`（OpenVLA 的去空闲帧版本，20 Hz，256 px；版本已固定）。共 1,693 条 demo、273k 帧、40 个任务。
  - 这些数据**只有 8 维 EE 状态**（世界坐标系位置 + 轴角 + 两指），没有关节角。
- **EE-only SOPT**：新增 `model.ee_only`。归一化后的关节输入和关节目标都置零，模型只看到夹爪 + 法兰位姿，17 维布局保持不变。
  - 在 DROID 上按 e2 的 S 配方重新预训练：`e13_ar_flow_f100_s_ee`（127；只用 DROID）。
- **EE → DROID 法兰坐标的标定**（在同时有关节和 EE 的 IPEC libero_90 上完成）：
  - `R_ee = R_flange · Rz(−45°)`；`p_ee = p_flange + R_flange·[0,0,0.0965] + base(task)`；位置残差 0.27 mm，旋转偏差 ≤ 0.05°。
  - **基座位置随场景不同**：有 (−0.75, 0, 0.912)、(−0.66, 0, 0.912)、(−0.51, 0, 0.42) 三种。每个任务的 base 在仿真中由机器人自身关节测得（`scripts/libero_std_base.py` → `configs/libero_std_base.json`），只涉及机器人几何，不用任何示范。
  - 数据集中的任务语言与 LIBERO benchmark 完全一致（有断言检查）。
- **闭环评估**：特征由仿真关节经 FK 得到后，同样把关节维置零（`hide_joints`），与训练完全一致。步数上限按 OpenVLA 协议：220 / 280 / 300 / 520。每任务 50 个官方初始状态，即每个 suite 500 个 episode。
- **组别**：SmolVLA none / expert_scratch / expert_droid，全部示范，seed 0（之后视情况加 seed），20k 步，bs 32（与 E7 相同）。每个 suite 一个 job（`SUITE=... sbatch scripts/slurm/e13_libero_std.sbatch`）。
- **防泄露**：SOPT 只用 DROID；`smolvla_base` 不含 LIBERO；下游使用 benchmark 的标准训练示范，评估使用官方初始状态（这是该 benchmark 的标准协议）。

### 2026-10-12 E10 结果：关键帧分解（KFMotor）**不如**直接用 SOPT 历史主干（127；k5，3 个 seed × 350 个配对 episode，`scripts/agg_e10.py`）
| 方法 | seed 0 / 1 / 2 | 均值 ± std |
|---|---|---|
| BC（无历史） | 64.0 / 62.9 / 65.7 | 64.2 ± 1.4 |
| TrunkBC 从零训练 | 61.1 / 62.9 / 59.4 | 61.1 ± 1.7 |
| **TrunkBC + DROID SOPT** | 70.9 / 71.7 / 66.3 | **69.6 ± 2.9** |
| KFMotor 从零训练 | 61.4 / 62.3 / 60.3 | 61.3 ± 1.0 |
| KFMotor + 目标条件 DROID SOPT | 66.0 / 58.3 / 65.7 | 63.3 ± 4.4 |
| KFMotor + DROID SOPT（目标通路零初始化新加） | 59.7 / 69.1 / 69.7 | 66.2 ± 5.6 |

池化配对 McNemar（只统计结果不同的 episode）：TrunkBC DROID 对 BC 155:98，p=4e-4；对 TrunkBC 从零 160:71，p=5e-9。KFMotor gcdroid 对 KFMotor 从零 155:134，p=0.24；对 TrunkBC DROID 114:180，p=1.4e-4。KFMotor droid 对 KFMotor 从零 159:108，p=0.002。

结论：
- 显式地"先预测关键帧，再生成动作"没有带来收益，反而显著低于直接把 SOPT 当历史主干（负结果，写进论文分析）。关键帧预测的误差会传给动作，抵消了结构带来的好处。
- 在 KFMotor 这个架构里，"预训练优于从零"依然成立（droid 对 scratch，p=0.002），和 E4b/E7/E8 一致。
- DROID 上的 hindsight 目标条件微调（gc_ar_flow_ft）对下游没有额外好处，而且方差更大。
- 主线仍然是：**预训练 SOPT 作为历史/运动表征接入**（TrunkBC、SmolVLA expert 调制）。

### 2026-10-12 E11 第一部分：下游效果随预训练数据量单调提升（127；TrunkBC k5，3 个 seed × 350 个配对 episode，`scripts/agg_e11.py`）
| SOPT 预训练（S 模型） | seed 0 / 1 / 2 | 均值 ± std |
|---|---|---|
| 无（从零训练） | 61.1 / 62.9 / 59.4 | 61.1 ± 1.7 |
| DROID 1% | 64.6 / 64.0 / 68.3 | 65.6 ± 2.3 |
| DROID 10% | 68.3 / 66.3 / 66.9 | 67.1 ± 1.0 |
| DROID 100% | 70.9 / 71.7 / 66.3 | 69.6 ± 2.9 |

池化配对 McNemar：1% 对从零 121:74，p=9e-4；10% 对从零 140:77，p=2e-5；100% 对从零 160:71，p=5e-9；100% 对 1% 133:91，p=0.006；10% 对 1% p=0.28；100% 对 10% p=0.09。
- 只要有 1% 的 DROID 状态预训练就能显著超过从零训练；数据越多越好（100% 显著优于 1%）。说明增益来自预训练本身，并且可以随数据扩展。
- 第二部分（B 模型，d512×12）第一次失败：pretrain 的 data.root 没设置，3 个下游任务也跟着失败。已修复（93d88ba，`sopt_e11b`），会在 E8 重跑完成后重新运行。

### 2026-10-12 E8 全部示范结果（冻结 DINOv2-S；约 50 个示范/任务；3 个 seed × 350 个配对 episode，`scripts/agg_e8.py all`）
| 方法 | seed 0 / 1 / 2 | 均值 ± std |
|---|---|---|
| BC（DINOv2，无历史） | 81.7 / 83.7 / 85.7 | 83.7 ± 2.0 |
| TrunkBC 从零训练 | 80.0 / 79.7 / 81.1 | 80.3 ± 0.8 |
| **TrunkBC + DROID SOPT** | 88.0 / 89.7 / 89.1 | **89.0 ± 0.9** |

池化配对 McNemar：DROID 对从零 148:57，p=2e-10；DROID 对 BC 142:87，p=3e-4；从零对 BC 100:136，p=0.02（从零训练的历史主干反而有害）。
- 换成强预训练视觉编码器、并用全部示范后，SOPT 的增益仍然存在而且更稳（+5.3 个百分点，seed 间 std < 1）。
- "从零训练时历史有害、预训练后历史有益"这个反转在高数据量下再次出现，而且是显著的。
- （s1 从零训练那一次曾因 GPU 争用 OOM，在 sopt_e8b 中重跑。）

### 2026-10-12 E12 seed 0 结果：SOPT + FastWAM（世界动作模型）**显著提升**（Jubail 18771995，cn265，79389f2；k5，每任务 50 个初始状态，`scripts/agg_e12.py 0`）
| 方法 | 成功率 |
|---|---|
| FastWAM（无 SOPT） | 49.7 |
| FastWAM + SOPT 从零训练 | 56.0 |
| **FastWAM + DROID SOPT** | **66.9** |

配对 McNemar（350 个 episode）：DROID 对无 SOPT 87:27，p=2e-8；DROID 对从零 73:35，p=3e-4；从零对无 SOPT 56:34，p=0.03。
- FastWAM 的动作专家没有在机器人数据上预训练过（由 Wan 权重插值而来），所以从零训练的历史编码器也有一点帮助。DROID 预训练的 SOPT 在此之上再多 +10.9 个百分点。
- 这是继 SmolVLA（E7）之后第二种预训练机器人基础模型被 SOPT 显著提升，而且这次是视频生成式的世界动作模型。
- seed 1、2 已排队（每次提交一个，在 2 个并发作业的限制内）。

### 2026-10-13 作业状态（Jubail）
- E7b（18771468）：9 个全部示范的 SmolVLA 模型都训练完了，但作业在 8 h 14 min 时被取消（CANCELLED，sacct 里没有原因，没到 48 h 的时间上限），评估还没跑。原 sbatch 会跳过已完成的步骤，所以按原样重新提交、只跑评估。
- E13 libero_spatial（18773286，cn008）：数据准备通过（432 个 episode，52,970 帧，10 个任务）。三个分支训练正常（约 2.8 it/s）。另外三个 suite 要等 spatial 的第一份评估结果出来才提交，避免评估流程有错时白白浪费算力。
- E12 seed 1 已提交（18773470）。seed 2 在串行队列里，每次只提交一个作业，同时最多 2 个。

### 2026-10-13 E11 第二部分：模型规模（127；TrunkBC k5，3 个 seed × 350 个配对 episode）
- B 模型（d512×12，56.6M 参数，100% DROID，40k 步，lr 2e-4，93d88ba）的预训练验证指标：selection 3.37。S 模型（约 5M 参数）是 4.2 左右。B 的位置 ADE 是 6.04 cm，S 是 7.37 cm。也就是说，B 在预训练任务上明显更好。
- 下游成功率：B_f100 为 68.0 / 70.3 / 72.6，平均 70.3 ± 2.3；S_f100 为 69.6 ± 2.9。B 对 S 的配对结果 128:121，p=0.70，没有显著差别。B 对从零训练 192:96，p=2e-8。
- 结论：下游增益随**预训练数据量**单调上升（E11 第一部分），但把模型从 5M 扩大到 57M，在 LIBERO-90 k5 上几乎没有额外收益。预训练指标更好，没有转化成下游成功率的提升。目前的瓶颈更可能在下游（只有 5 个示范、策略容量、视觉），而不是先验本身的容量。论文中主模型继续用 S，B 作为规模消融。

### 2026-10-13 E13 libero_spatial 全部示范（seed 0，Jubail 18773286，cn008，03a5941）：**无提升**
| 方法（SmolVLA，EE-only SOPT） | 成功率 | 各任务 |
|---|---|---|
| 无 SOPT | 78.2 | 90 92 84 82 72 88 96 18 86 74 |
| + SOPT 从零训练 | 79.0 | 88 96 76 92 80 80 96 24 84 74 |
| + DROID SOPT | 77.6 | 82 86 90 94 80 88 86 16 82 72 |

配对 McNemar（500 个 episode）：DROID 对无 SOPT 44:47，p=0.83；从零对无 SOPT 43:39，p=0.74。
- 每任务约 43 个示范，10 个任务的动作几乎一样（拿碗放到盘子上），难点主要在"去哪"。SmolVLA 可能已经能从下游示范里学到这个动作。
- 有一个任务三种方法都只有 16–24%：「pick up the black bowl on the ramekin」（碗放在小烤碗上面）。已核查：spatial 的 10 个任务 base 标定完全相同，而且没有 SOPT 的分支也一样低，所以不是标定问题，是任务本身难。
- 待检验的假设：SOPT 的增益随"模型已经掌握的机器人动作知识"增加而减小。这里的动作知识既来自基础模型的预训练，也来自下游示范的数量。对应两组新实验：
  1. **E13 k5**：在 4 个标准 suite 上补跑每任务 5 个示范（`K=5`，dcb77cc），检验是不是示范数量的原因；
  2. **E14**：π0.5-DROID + SOPT。

### 2026-10-13 E14 准备：π0.5-DROID + SOPT（e406355）
- 基础模型 `lerobot/pi05_droid`（revision 72824c0）：Physical Intelligence 在 π0.5 base 上用 DROID **动作**微调得到的模型，LeRobot 移植版。它已经见过 DROID 的图像、语言和动作，是检验"VLA 已经学到了 SOPT 的运动先验"最强的情况。
- 各分支用同一套配方（π0.5/openpi 默认）：冻结 VLM（`train_expert_only`），训练 Gemma-300M 动作专家；lr 2.5e-5，wd 0.01，梯度裁剪 1.0；状态和动作都用 QUANTILES 归一化（训练帧的 q01/q99）；当前状态离散化后写进 prompt（π0.5 原生方式）；开 gradient checkpointing。
- 相机：agent view → base_0_rgb，wrist → left_wrist_0_rgb，right_wrist 为空（mask 掉，和 DROID 一致）。
- SOPT 接入方式和 E7/E12 相同：历史 96 帧的最后一个 token 特征，经零初始化 MLP 加到每个动作 token 上。
- 协议与 E7/E12 相同：LIBERO-90 E4b 的 7 个任务，k5，20k 步，bs 32，每任务 50 个初始状态。
- Tokenizer：HF 上的 PaliGemma tokenizer 需要申请权限，所以改用 openpi 用的公开 SentencePiece 模型（gs://big_vision/paligemma_tokenizer.model，sha256 8986bb4f…）转换成 HF 格式，并在 LIBERO 提示词上逐 token 和 SentencePiece（带 BOS）比对，结果一致。
- `scripts/vla_train.py` / `vla_eval.py` 新增 `--vla smolvla|pi05`；SmolVLA 的行为不变。
- sbatch 先跑 30 步 + 1 个初始状态的冒烟测试，失败就整个作业退出。
- Jubail 修复：之前手动 scp 的 agg_e12.py 挡住了 `git pull`，导致 E12 seed 1 用的是不带 80g 约束的旧脚本。已取消并重新提交（18774817）。提交队列的作业数判断 bug 也已修复。

### 2026-10-13 E7b 结果：LIBERO-90 全部示范时，SOPT 仍**显著**提升 SmolVLA（Jubail 18771468 训练 + 18774788 评估；3 个 seed × 350 个配对 episode，`scripts/agg_e13.py e7b`）
| 方法 | seed 0 / 1 / 2 | 均值 ± std |
|---|---|---|
| SmolVLA（无 SOPT） | 85.7 / 86.9 / 84.9 | 85.8 ± 1.0 |
| + SOPT 从零训练 | 86.3 / 81.1 / 82.9 | 83.4 ± 2.6 |
| **+ DROID SOPT** | 88.6 / 88.0 / 89.1 | **88.6 ± 0.6** |

池化配对 McNemar：DROID 对无 SOPT 100:71，p=0.03；DROID 对从零 107:53，p=2e-5；从零对无 SOPT 68:93，p=0.06（从零训练反而有害的趋势）。
- 示范从 5 个增加到约 50 个时，SOPT 的增益从 +4.8 缩小到 +2.8，但仍然显著；"从零训练有害、预训练有益"的反转也还在。所以"示范多了 VLA 就自己学会了"**不能**解释 E13 为什么没有提升。

### 2026-10-13 E12 seed 1（Jubail 18774817，80 GB 节点）：none 55.4 / scratch 52.9 / droid 65.1
两个 seed 池化（700 个配对 episode，`scripts/agg_e12.py 0 1`）：none 52.6，scratch 54.4，**droid 66.0**。DROID 对无 SOPT 151:57，p=5e-11；DROID 对从零 141:60，p=1e-8；从零对无 SOPT p=0.37。seed 2 正在跑（18776675）。

### 2026-10-13 E13 更多 suite（seed 0，每个 500 个配对 episode，`scripts/agg_e13.py`）
| suite | 无 SOPT | + 从零 | + DROID | DROID 对无 SOPT |
|---|---|---|---|---|
| spatial 全部示范 | 78.2 | 79.0 | 77.6 | 44:47，p=0.83 |
| spatial k5（18775456） | **83.0** | 79.4 | 79.2 | 25:44，**p=0.03（有害）** |
| object 全部示范（18775765） | 92.6 | 92.8 | 91.8 | 32:36，p=0.72 |

- 在标准 suite 上，EE-only SOPT 不仅没有提升，在 spatial k5 上还显著有害；从零训练的版本也一样有害（19:37，p=0.02）。
- E7b 已经排除了"示范多"这个解释，剩下的差别在于 E13 用的是 **EE-only SOPT 先验 + EE→法兰坐标换算**。E15 用来把这两个因素分开。
- 另一个奇怪的现象：spatial 上无 SOPT 的 k5（83.0）比全部示范（78.2）还高，说明全部示范、20k 步的配方在这里可能训练不足，或者有过拟合问题。还需要后续分析。

### 2026-10-13 E15 诊断（提交中，54b114d）：把 EE-only 先验放到 E7 的 LIBERO-90 协议上
- 只把 E7 的 expert_droid 先验换成 `e13_ar_flow_f100_s_ee`（SOPT 内部隐藏关节，归一化统计量与原先验逐元素相同），其余设置完全一致。3 个 seed 和 E7 的 none、expert_droid 做配对比较。
- 如果增益消失，问题出在 EE-only 表示（需要保留关节信息，或者改进 EE-only 预训练）；如果增益保留，问题出在标准 suite 的数据或坐标换算。

### 2026-10-13 E14 冒烟测试失败（18774962）与修复（59179b1）
- 失败原因：π0.5 的预处理流水线在构建时就按名字去拉取需要权限的 HF tokenizer（Jubail 计算节点离线），我原来的替换代码在它之后才执行。改为在构建时直接替换。权重加载正常（"All keys loaded successfully"）。已重新提交（18777256）。
- E14 seed 0（18777256，cn268）：冒烟测试通过（训练 30 步，1 个初始状态评估跑通；成功率 0%，30 步时符合预期）。三个分支已开始训练：可训练参数 693M（expert 分支 705M），速度 0.28 it/s（bs 32，开 gradient checkpointing），20k 步约 20 h，加评估约 21 h，在 30 h 时限内。为了和 E7 可比，保持 20k 步不变。

### 2026-10-13 E15b 结果：EE-only 先验在 TrunkBC 上**不变差**（127；k5，3 个 seed × 350 个配对 episode，`scripts/agg_e15b.py`）
| 方法 | 各 seed | 均值 |
|---|---|---|
| BC（无历史） | 64.0 / 62.9 / 65.7 | 64.2 ± 1.4 |
| 从零训练主干 | 61.1 / 62.9 / 59.4 | 61.1 ± 1.7 |
| DROID 先验（含关节，e2） | 70.9 / 71.7 / 66.3 | 69.6 ± 2.9 |
| DROID 先验（EE-only，e13） | 75.7 / 69.1 / 68.0 | **71.0 ± 4.2** |
- ee_prior 对比 joint_prior：109 vs 95，p=0.36（无显著差异）；对比 scratch：p=1e-11；对比 BC：p=2e-6。
- 结论：在 LIBERO-90 上，去掉关节信息的 SOPT 先验依然有效。所以标准 suite（E13）上没有增益，**不太可能是 EE-only 表示本身造成的**，更可能是标准 suite 的数据特点（任务短、初始状态分布窄、基线已经很高）或坐标换算的问题。E15（SmolVLA + EE-only，Jubail）会在 VLA 上再验证一次。

### 2026-10-14 E12 三个 seed 汇总：SOPT + FastWAM **显著提升**（Jubail 18771995 / 18774817 / 18776675；k5，3 个 seed × 350 个配对 episode，`scripts/agg_e12.py 0 1 2`）
| 分支 | 各 seed | 均值 |
|---|---|---|
| none | 49.7 / 55.4 / 55.1 | 53.4 ± 3.2 |
| expert_scratch | 56.0 / 52.9 / 56.9 | 55.2 ± 2.1 |
| expert_droid | 66.9 / 65.1 / 63.1 | **65.0 ± 1.9** |
- droid vs none：223 vs 101，p=1e-11；droid vs scratch：p=5e-9；scratch vs none：p=0.26（不显著）。增益来自 DROID 预训练，而不是额外的历史输入或参数量。

### 2026-10-14 E15 结果：在 SmolVLA 上，EE-only 先验与含关节先验**效果相同**（Jubail 18787774，cn016，9589990；`scripts/agg_e15.py`）
| 分支 | 各 seed | 均值 |
|---|---|---|
| none（E5） | 58.9 / 58.3 / 66.6 | 61.2 ± 4.6 |
| expert_scratch（E7） | 62.0 / 56.9 / 64.9 | 61.2 ± 4.1 |
| expert_droid 含关节（E7） | 68.6 / 64.3 / 65.1 | 66.0 ± 2.3 |
| expert_droid EE-only（E15） | 69.1 / 63.7 / 63.7 | **65.5 ± 3.1** |
- ee vs joint：44 vs 49，p=0.68；ee vs none：p=0.003；ee vs scratch：p=0.002。
- 结论（与 E15b 一致）：EE-only 先验本身没有问题。所以 E13 libero_spatial 上没有增益，原因在 suite 本身，而不在先验的表示方式。

### 2026-10-14 E13 更多 suite（seed 0，每个 500 个配对 episode）
| suite | 设置 | none | expert_scratch | expert_droid | droid vs none |
|---|---|---|---|---|---|
| libero_object | k5（18794424） | 77.6 | 78.6 | **84.2** | 78 vs 45，p=0.004（vs scratch p=0.009） |
| libero_10 | 全部示范（18799492） | 65.6 | 70.2 | 63.2 | 77 vs 89，p=0.39（droid 比 scratch 低，p=0.005） |
- libero_object k5：SOPT 显著提升（+6.6）；全部示范时不显著（91.8 vs 92.6，接近上限）。
- libero_10 全部示范：DROID 先验没有带来提升，反而不如从零训练的 SOPT。libero_10 是长时程任务（最多 520 步），96 帧历史（6.4 s）只能看到其中一小段，从零训练的历史编码器可能更适合学到这个 suite 特有的阶段信息。只有一个 seed，需要补 seed。
- libero_goal（18799465 / 18799491）在数据准备阶段失败：HF 缓存里的 `wrist_image/episode_000082.mp4` 损坏（InvalidDataError）。逐个解码检查了已缓存的 262 个视频，只有这一个坏了。进一步核查发现，用 curl 重新下载后的文件 sha256 = b3788859…，和 HF 上记录的 LFS 哈希一致，说明**上游数据本身已损坏**。处理方式：在 `sopt/data/libero_std.py` 中设 `CORRUPT = {"libero_goal": {82}}`，把这条示范从可用列表中排除（goal 剩 427 条，10 个任务都在；其他 suite 不变，spatial 仍是 432 条，已有结果可复现）。这条只是训练示范，评测用的是仿真器的初始状态，所以不影响评测。另外，e13 sbatch 改为数据准备失败就直接退出。已重新排队：goal、goal k5、spatial k5 seed 1、object k5 seed 1。

### 2026-10-14 E16：SOPT 先验在各评测域上的零样本预测质量（Jubail 登录节点 CPU，`scripts/e16_prior_fit.py`，EE-only 先验 e13，K=8，stride 10）
窗口和策略输入的构造方式完全相同（96 帧 15 Hz 历史，预测 16 步），只用训练示范，不用任何 rollout。

| 域 | 示范数 | 窗口数 | 先验 ES（cm） | 匀速外推 ES（cm） | 比值 | SOPT 下游增益（SmolVLA） |
|---|---|---|---|---|---|---|
| LIBERO-90（E4b 7 个任务） | 311 | 3500 | 7.14 | 14.99 | 0.476 | k5 +4.8*，全部示范 +2.8* |
| libero_spatial | 432 | 3631 | 10.32 | 23.23 | 0.444 | k5 −3.8（单 seed），全部示范 −0.6 |
| libero_object | 454 | 4950 | 9.24 | 20.13 | 0.459 | k5 +6.6*（单 seed），全部示范 −0.8 |
| libero_goal | 427 | 3550 | 8.46 | 19.40 | 0.436 | 待测 |
| libero_10 | 379 | 8689 | 7.15 | 16.65 | 0.429 | 全部示范 −2.4（单 seed） |
- 先验在所有域上都把匀速外推的误差降低到 0.43–0.48 倍，各 suite 和 LIBERO-90 相当，甚至略好。所以**坐标换算没有问题，也不存在明显的域失配**：先验对标准 suite 的运动同样"懂"。下游增益随 suite 变化，不能用先验拟合得好不好来解释。
- 另一个噪声参考：spatial 的 none 分支在 k5 时 83.0，比全部示范时的 78.2 还高。数据更多反而更差，说明单 seed 的 suite 级差异有 ±4 点左右的噪声，E13 的单 seed 结论都需要补 seed（已在排队）。

### 2026-10-14 E17 启动：语言模型式的模块是否更好（127 tmux `sopt_e17`，fb3d1bf，`scripts/remote/e17_queue.sh`）
- 原来的 S 模块：pre-LN，LayerNorm，GELU MLP，RoPE，QK-norm（RMSNorm），PyTorch 默认初始化。
- E17 的改法：GELU 换成 SwiGLU（隐藏层乘 2/3，参数量对齐：977 万 vs 978 万），LayerNorm 换成 RMSNorm（含最终的 norm），初始化换成 xavier-normal、偏置置零、残差输出投影按 1/sqrt(2L) 缩放。RoPE 和 QK-norm 原来就有，保持不变。
- 改动是配置选项（`model.mlp / norm / init`），默认值等于原来的模块，旧 checkpoint 照常加载（已验证）。
- 训练配方和 `e2_ar_flow_f100_s` 完全一致（S，ar_flow，100% DROID，40k 步，相同的训练/验证划分），然后在 TrunkBC 上每个任务 5 条示范、3 个 seed、每任务 50 个初始状态评估，用 `scripts/agg_e17.py` 和 E4b 的 droid 先验配对比较。预训练只做 1 个 seed。
- 预期：E11 中模型放大约 10 倍下游都没变，所以这次大概率没有显著差异；如果确实没有，就和 E11 一起说明"结构不敏感，数据才是关键"。

### 2026-10-14 E17 结果：语言模型式模块**没有显著差别**（127，预训练 39 min；TrunkBC k5，3 个 seed × 350 个配对 episode，`scripts/agg_e17.py`）
| 预训练（DROID val，40k 步） | ES（cm） | ADE（cm） | FDE（cm） |
|---|---|---|---|
| 原 S 模块（e2） | 3.99（最佳）/ 4.01（最终） | 6.96 | 12.97 |
| LM 模块（e17） | **3.97** | **6.90** | 12.89 |

| 下游 TrunkBC k5 | 各 seed | 均值 |
|---|---|---|
| BC | 64.0 / 62.9 / 65.7 | 64.2 ± 1.4 |
| 从零训练主干 | 61.1 / 62.9 / 59.4 | 61.1 ± 1.7 |
| 原 S 先验 | 70.9 / 71.7 / 66.3 | 69.6 ± 2.9 |
| LM 模块先验 | 75.1 / 68.6 / 69.4 | 71.0 ± 3.6 |
- 预训练指标好了约 0.5–1%（训练曲线几乎重合）；下游 lm vs 原 S：112 vs 97，p=0.33，不显著；lm vs scratch p=2e-11。
- 结论：和 E11 的模型规模结果一致，**骨干结构不是瓶颈**。保持原模块不变（已有的 E7/E12/E13/E14 结果都基于原模块，不需要重跑）；新选项留作可选配置。论文里可以作为"结构消融"一行：SwiGLU / RMSNorm / xavier 初始化 ≈ 原模块，增益来自状态预训练本身。

### 2026-10-14 E13 更多 suite（seed 0，每组 500 个配对 episode）
| suite | 设置 | none | expert_scratch | expert_droid | droid vs none | droid vs scratch |
|---|---|---|---|---|---|---|
| libero_10 | k5（18799808） | 44.6 | 50.8 | 50.4 | 84 vs 55，p=0.017 | 64 vs 66，p=0.93 |
| libero_goal | 全部示范（18801445，排除 ep82 后 427 条示范） | 89.8 | 92.0 | **94.4** | 45 vs 22，p=0.007 | 35 vs 23，p=0.15 |
- libero_10 k5：加入历史（无论是否预训练）提升约 6 点（scratch vs none p=0.009），这里的增益来自**历史本身**，而不是 DROID 预训练。这和长时程任务需要知道"进行到哪一步"一致。
- libero_goal 全部示范：DROID 版显著优于 none（+4.6），对 scratch 的优势不显著。
- 标准 suite 汇总（seed 0）：droid vs none 显著为正的有 object k5、goal 全部示范、libero_10 k5；不显著的有 spatial 全部示范、object 全部示范、libero_10 全部示范；显著为负的只有 spatial k5。补 seed 后再下结论。

### 2026-10-14 E13 libero_goal k5（18801771，seed 0，500 个配对 episode）：**显著提升**
- none 75.0 / expert_scratch 76.8 / expert_droid **80.6**；droid vs none 62 vs 34，p=0.006；droid vs scratch 50 vs 31，p=0.045；scratch vs none p=0.40。
- 这是标准 suite 中第二个预训练本身（droid vs scratch）也显著的设置（另一个是 object k5）。

### 2026-10-14 E14 seed 0（18777256，cn268，59179b1）：π0.5-DROID 整体成功率异常偏低，**结果暂不解读**
- none 17.1 / expert_scratch 17.7 / expert_droid 19.1（每组 350 个 episode）。同样协议下 SmolVLA 的 none 是 61.2，π0.5 却低得多，而且几乎所有任务都很低（SmolVLA 的 cream cheese 任务 74%，π0.5 是 0%）。
- 训练 loss 在 2 万步时为 0.004（SmolVLA 是 0.006），说明能拟合训练数据，问题出在闭环阶段：要么训练和推理不一致，要么在 35 条示范上严重过拟合（π0.5 只训练 action expert，VLM 冻结）。
- 诊断（18802574，`scripts/vla_openloop.py`）：在训练示范和同任务的另外 5 条未训练示范上，比较开环动作误差，SmolVLA 作对照。在诊断出结果之前，不提交 seed 1、2。
- 诊断 1 结果（18802574，a169502；每个数据集 256 帧；误差 = 前 10 个动作的 MSE，以动作 std 为单位）：

  | 策略 | 训练示范 MSE | 未训练示范 MSE | 未训练示范夹爪一致率 |
  |---|---|---|---|
  | SmolVLA none k5（E5） | 0.002 | 0.348 | 0.909 |
  | π0.5 none k5 | 0.074 | 0.363 | 0.908 |
  | π0.5 expert_droid k5 | 0.090 | 0.338 | 0.929 |
  - 推理流程没有问题：在未训练示范上，π0.5 和 SmolVLA 一样好。差别在于 π0.5 对训练示范的拟合精度低约 30 倍（lr 2.5e-5、VLM 冻结；也可能是采样噪声）。k5 时 LIBERO 闭环很依赖精确复现示范，这可能是闭环成功率低的原因。
  - 诊断 2（18802627）：每块执行的动作数 n_exec ∈ {5, 25, 50} 时的闭环敏感性，以及 10 个初始状态的失败归类。
- 诊断 2 结果（18802627，π0.5 none s0，10 个初始状态 = 70 个 episode）：
  - 每块执行的动作数：n_exec 5 → 17.1%，10 → 18.6%（追踪运行），25 → **30.0%**，50 → 24.3%。重新规划太频繁会加重抖动或模式切换。
  - 失败归类（n_exec 10）：成功 13，找错物体 8，没接近目标 2，**抓取失误 28**，放置失败 17，抽屉任务失败 2。主要是精度问题，和诊断 1 中"训练示范拟合不精确"一致。
- 处理方案：
  1. 用现有的 seed 0 checkpoint 在 n_exec 25 下重新评测三个分支（50 个初始状态，18802803）。三个分支用同样的设置，所以配对比较仍然有效。n_exec 是根据 none 分支的 10 个初始状态选出来的，同一个设置用于所有分支，报告时两个 n_exec 都列出。
  2. **E14b**：lr 1e-4（与 SmolVLA 相同），其他设置不变，评测时 n_exec 取 10 和 25。三个分支用同样的配方，SOPT 的对比依然公平。

### 2026-10-15 E14 seed 0 在 n_exec 25 下重新评测（18802803，cn010，同一批 checkpoint，`scripts/agg_e14.py --x 25 0`）
| n_exec | none | expert_scratch | expert_droid | droid vs none | droid vs scratch |
|---|---|---|---|---|---|
| 10 | 17.1 | 17.7 | 19.1 | 36 vs 29，p=0.46 | p=0.58 |
| 25 | 26.3 | 25.4 | **30.0** | 45 vs 32，p=0.17 | 49 vs 33，p=0.10 |
- 方向和 SmolVLA、FastWAM 一致（droid > none ≈ scratch），但单个 seed 不显著，而且整体水平偏低，因为 π0.5 欠拟合（诊断 1）。最终结论看 E14b（lr 1e-4，18802970，已开始运行）。

### 2026-10-15 E13 libero_spatial k5 补充 seed 1（18802280）：负结果**复现了**
- 2 个 seed（1000 个配对 episode）：none 81.8 / expert_scratch 79.8 / expert_droid 79.2；droid vs none 54 vs 80，p=0.03；droid vs scratch p=0.65；scratch vs none p=0.057。
- 两个带历史的分支都比 none 低约 2–2.6 点，DROID 预训练本身没有带来额外损失，问题出在**加入运动历史**。libero_spatial 的 10 个任务动作几乎相同（都是拿起黑碗放到盘子上），区别只在于是哪个碗（只能靠视觉区分）。推测：历史条件让策略更多依赖"运动惯性"而不是视觉，导致在选哪个碗上更容易出错。可以用失败归类（trace）来检验：找错物体的比例是否上升。

### 2026-10-15 E13 libero_object k5 补充 seed 1（18810466）：seed 0 的显著增益**没有复现**
- seed 1：none 83.6 / scratch 82.8 / droid 81.6。合并 2 个 seed（1000 个配对 episode）：none 80.6 / scratch 80.7 / droid 82.9；droid vs none 126 vs 103，p=0.15；droid vs scratch p=0.14。方向为正，但不显著。
- 这说明单 seed 的 suite 级结论很不可靠：seed 0 的 none 只有 77.6，偏低。spatial、object 的 seed 2 已在排队或运行中（18812643 spatial s2 正在运行）。

### 2026-10-15 扩展宿主和"在更多场景都变好"（E18–E22 启动）
目标：(1) 更多主流宿主，覆盖完整的"动作知识"谱系；(2) 让 SOPT 在标准 suite 上至少不变差。

| 编号 | 内容 | 位置 | 对照 |
|---|---|---|---|
| E18 | 预训练数据来源：只用 LIBERO-90 训练任务的状态（e4a scratch_kall，2,725 条示范）vs DROID+LIBERO（e4a ft_kall）vs 只用 DROID | 127 TrunkBC k5，3 个 seed | E4b |
| E19 | SOPT dropout：训练时以概率 0.5 置零 SOPT 特征（迫使策略同时依赖视觉），评估时始终开启。SmolVLA expert_droid | LIBERO-90（127，`SOPT_VLA_ENV`）+ libero_spatial（Jubail），各 3 个 seed | E7 / E13 |
| E20 | **X-VLA-0.9B**（Florence-2 + soft-prompt flow transformer；预训练数据 = DROID/RoboMind/AgiBot，**不含 LIBERO**，arXiv 2510.10274）。按 LeRobot 推荐配方全参数微调（VLM 用 0.1× lr），domain 0（**不用** LIBERO 的 soft prompt 3），chunk 30 | Jubail | none / scratch / droid |
| E21 | **Diffusion Policy**（ResNet-18 ImageNet + 1-D U-Net，无机器人预训练；任务用 one-hot 环境状态表示），SOPT 特征加到 U-Net 的全局条件上（零初始化），30k 步，bs 64，DDIM 100 训练 / 10 推理步 | Jubail | 同上 |
| E22 | **GR00T N1.7-3B**（Cosmos-Reason2 / Qwen3-VL + DiT 动作头），用 `new_embodiment` tag（不用可能带 LIBERO 信息的 `libero_sim` 投影） | 权重下载中 | 同上 |
- 每个宿主的 SOPT 接入方式相同：96 帧历史 → SOPT 最后一个 token 的特征 → 零初始化 MLP → 加到宿主的动作 token（或全局条件）上。所有分支除 SOPT 外完全相同。
- 工程：`vla_train.py` 支持各 VLA 自己的 chunk 长度（`CHUNK`）、betas、分层学习率（`param_groups`）、任务列表（`NEEDS_TASKS`）和 `--cond-drop`；SmolVLA / π0.5 的行为不变。127 的 `SOPT_VLA_ENV`（LeRobot 0.6.2）补装了 diffusers 0.41；Jubail 的 pydeps 补装了 dm-tree，diffusers 用已有的 0.39（同一宿主的所有分支都在同一环境中运行）。
- Jubail 提交改成读队列文件的方式（每次名额空出来就提交队首）。

### 2026-10-15 统一统计（`scripts/stats_all.py` + `configs/paper_manifest.json` → `docs/PAPER_STATS.md`）
- 每组比较同时报告配对 McNemar（以 episode 为单位）和 **(seed, task) 簇 bootstrap 95% 置信区间**（10k 次），主要比较做 Holm 校正。
- **重要发现**：LIBERO-90 协议只有 7 个任务，按任务分簇后置信区间明显变宽。以 McNemar 判断显著的比较中，部分在任务层面并不稳健：
  - 稳健（CI 不含 0）：所有"droid vs scratch"（TrunkBC +8.5 [4.4, 12.7]、DINOv2 全部示范 +8.7、SmolVLA k5 +4.8 [1.1, 8.6]、SmolVLA 全部示范 +5.1 [2.5, 7.8]、FastWAM +9.8 [5.2, 14.8]）；FastWAM droid vs none +11.6 [7.9, 15.3]；SmolVLA k5 droid vs none +4.8 [0.2, 9.2]（勉强）；DINOv2 全部示范 droid vs bc +5.2 [1.1, 9.8]。
  - 不稳健（CI 含 0）：TrunkBC droid vs bc +5.4 [−0.8, 11.9]（McNemar p=4e-4）；SmolVLA 全部示范 droid vs none +2.8 [−0.5, 6.0]（Holm p=0.064）；DINOv2 k5 droid vs bc。
  - 解读："预训练 vs 同结构从零训练"这一结论在任务层面稳健；"加 SOPT vs 不加历史"的增益在不同任务之间差异较大，7 个任务不足以给出窄的置信区间。
- 改进方向（写入 PAPER_PLAN）：扩大评测任务数。LIBERO-90 的 val split 还有指令唯一的任务，SOPT 从未见过（主先验只用 DROID），可以和标准 suite 的 40 个任务一起，把簇数从 7 提到 15+，使任务层面的结论站得住。

### 2026-10-15 E23：主表改用 17 个任务的 LIBERO-90 协议（`configs/libero90_17_tasks.json`）
- 任务 = 原来的 7 个 test-split 任务 + 10 个 val-split 任务（指令在 LIBERO-90 的 90 个场景中唯一，可以直接在仿真中评估）。val split 只在 E4a 的 LIBERO 后训练中用来做模型选择；主先验只用 DROID，下游从未在这些任务上做过任何选择，所以对主先验来说完全没见过。
- **定位**：原来的 7 任务协议在开发过程中被反复使用（包括选择 n_exec、诊断等），因此作为**开发集**（论文附录）；17 任务协议作为**主表**，所有宿主（TrunkBC / SmolVLA / FastWAM / X-VLA / Diffusion Policy / π0.5 / GR00T）都在上面用 3 个 seed 重跑。这样同时解决两个问题：簇数从 7 增加到 17（任务层面的置信区间更窄），并避免"在开发集上反复调整"的质疑。
- 代码：`vla_train.py --suite libero90_17`、`wam_train.py --suite libero90_17`、`e4b_train.py --tasks configs/libero90_17_tasks.json`；episode 从 test+val split 中选取，任务内的选择规则不变（7 任务的 run 可以完全复现）。Jubail 上已预先缓存 17 个任务 k5（85 条示范）的 256 px 帧。运行名后缀为 `_t17`。
- 排队：Jubail 按 X-VLA → SmolVLA → DP → E19 spatial → FastWAM → … 的顺序，每个宿主 3 个 seed；127 上 E19 完成后跑 TrunkBC 17（`sopt_e23t`）。
- GR00T N1.7：权重已下载（6.5 GB），但分词器和图像预处理器在需要授权的 `nvidia/Cosmos-Reason2-2B` 仓库中（401）。需要账号接受许可并提供 HF token；在此之前暂停，不用公开的 Qwen3-VL 处理器替代（无法验证分词完全一致）。
- `configs/paper_manifest.json`：17 任务的 MAIN 表在前（Holm 只对 MAIN 的主要比较做校正），7 任务结果标为 DEV。

## E18 结果（2026-10-15，127，TrunkBC k5，7 任务 × 50 init × 3 seeds）：收益来自 DROID，不是"任何状态预训练"

| trunk 初始化 | per seed | mean ± sd |
|---|---|---|
| BC (MLP) | 64.0 / 62.9 / 65.7 | 64.2 ± 1.4 |
| scratch | 61.1 / 62.9 / 59.4 | 61.1 ± 1.7 |
| LIBERO-90 状态预训练（e4a_ar_flow_scratch_kall，仅 LIBERO train 任务） | 62.9 / 61.7 / 59.7 | 61.4 ± 1.6 |
| DROID 预训练（E4b trunk_droid） | 70.9 / 71.7 / 66.3 | 69.6 ± 2.9 |
| DROID → LIBERO 微调（e4a_ar_flow_ft_kall） | 72.6 / 72.0 / 71.1 | 71.9 ± 0.7 |

Pooled McNemar：DROID vs LIBERO-only 166 vs 80，p=4e-8；LIBERO-only vs scratch p=0.89；DROID→LIBERO vs DROID p=0.088（n.s.，方差更小）；DROID→LIBERO vs scratch p=4e-13。
结论：同域少量状态预训练 = 没预训练；增益来自大规模跨场景的 DROID 运动数据。在目标域状态上再微调只带来小幅、不显著的提升。
基础设施：此后 127 不再跑 SOPT 计算（让给他人）；小模型实验迁到实验室 DGX Spark（`scripts/dgx/`，冒烟测试 158374 通过，GB10 TrunkBC ~19 it/s），E23 TrunkBC 17 任务 = DGX 作业 158381；E19 LIBERO-90 改到 Jubail 队列（127 上的 seed0 中途杀掉，无 checkpoint）。

## E23 TrunkBC 17 任务结果（2026-10-15，DGX Spark 作业 158381，k5 × 50 init × 3 seeds，850 ep/run）

| arm | per seed | mean ± sd |
|---|---|---|
| BC (MLP) | 72.2 / 71.5 / 66.8 | 70.2 ± 2.9 |
| scratch trunk | 69.2 / 65.9 / 69.2 | 68.1 ± 1.9 |
| DROID trunk | 72.8 / 70.9 / 72.9 | 72.2 ± 1.1 |

stats_all（cluster bootstrap over (seed, task)，Holm）：DROID vs scratch +4.2 [+1.2, +7.2]，p=1.7e-5（Holm 3.5e-5）——稳健；DROID vs BC +2.0 [−1.8, +6.0]，McNemar p=0.04，CI 含 0——不稳健。
按子集：7 个 dev 任务 BC 58.0 / scratch 58.0 / DROID 64.7（+6.7，与 DEV 结论一致）；10 个新任务 78.7 / 75.1 / 77.5（DROID ≈ BC，> scratch +2.4）。
解读：新 10 任务基线更高（~79%，天花板更近），且 7 个 dev 任务曾被用于开发期的选择，"DROID trunk > BC" 在干净的新任务上没有复现；"DROID 预训练 > 同结构随机初始化" 在两个子集上都成立。论文主张应以 pretrain-vs-scratch 与 host+SOPT-vs-host 为主，TrunkBC-vs-BC 只作为辅助。

### E23 分析：增益集中在难任务（headroom），交叉 seed 验证
按任务计算 DROID trunk 相对 BC 的增益与 BC 基线成功率的相关：全 seed r=−0.62（BC<70% 的 6 个任务 +11.3，70–90% 的 9 个 −1.9）。
为排除回归均值（同一基线出现在两侧），用一个 seed 定义难度、另外两个 seed 测增益：
| 难度来自 | r | 难任务 DROID−BC | 易任务 DROID−BC | 难任务 DROID−scratch | 易任务 DROID−scratch |
|---|---|---|---|---|---|
| s0（增益用 s1,s2） | −0.40 | +9.6 (n=7) | −2.0 | +4.9 | +4.1 |
| s1（增益用 s0,s2） | −0.54 | +10.4 (n=5) | +0.4 | +2.8 | +4.1 |
| s2（增益用 s0,s1） | −0.55 | +7.4 (n=7) | −5.2 | +8.0 | +1.8 |
结论：SOPT 相对普通 BC 的收益集中在策略本来容易失败的任务（基线 <70%），简单任务上无收益；预训练相对随机初始化在难易两类任务上都为正。这解释了 E23 dev（偏难，基线 58%）与 new（偏易，基线 79%）子集的差异。论文中按预先定义的难度分层（用独立 seed 定难度）报告，避免事后挑选。之后对各 host 用同样的交叉 seed 分析（难度来自 none arm）。
- 推广到各 host（`scripts/headroom.py`，同样交叉 seed）：TrunkBC 在 DEV 7 任务上同样是 headroom 型（r −0.61…−0.80，难任务 DROID−BC +9…+19，易任务 ≈0）；但**作为插件加到 host 上时增益不依赖难度**——SmolVLA k5 难/易任务 DROID−none ≈ +1.5…+7 / +3…+9（r 为正或≈0），FastWAM k5 难 +10.6…+14.6 / 易 +4.5…+13。即：替换动作骨干（TrunkBC vs BC）的收益只在难任务出现；把 SOPT 作为额外条件加到现成 host 上是全面的提升。仅 7 任务，样本小；17 任务 host 结果出来后复查。

## E22 GR00T N1.7 17 任务 seed 0（2026-10-15，Jubail 18815823，cn017，7c80813；k5，20k steps，n_exec 10，850 ep/run）
| arm | all 17 | 7 dev | 10 new |
|---|---|---|---|
| none | 81.9 | 79.1 | 83.8 |
| expert_scratch | 77.2 | 72.6 | 80.4 |
| expert_droid | 79.2 | 76.9 | 80.8 |
McNemar：droid vs none 78 vs 101 p=0.10；droid vs scratch 95 vs 78 p=0.22；scratch vs none 60 vs 100 **p=0.002**。
- 单 seed：GR00T 上 SOPT 插件**没有带来提升**（−2.7，不显著）；随机初始化的同结构插件显著变差（−4.7），DROID 先验部分抵消了这一损失。逐任务差异方向不一致（例如 book/back compartment +20，cream cheese box/basket −20），单 seed 噪声大。
- 可能原因（待 seed 1/2 确认后再诊断）：GR00T 基线已经较高（82%，所有 host 中最高），且其 DiT 动作头已经以 proprio 为条件；把额外特征加到动作 token 上对其是一种扰动（scratch 显著下降说明接口本身有成本）。
- 处理：按预先规定照常跑完 seed 1/2 并纳入主表，不因单 seed 结果更改协议或剔除该 host。若三 seed 仍为负，论文中如实报告为"高基线 host 上无收益"，并与 headroom 分析一起讨论。

## E14b π0.5-DROID lr 1e-4（2026-10-16，Jubail 18802970，cn010，d2bd367；7 dev 任务，k5，seed 0，350 ep/arm）
| n_exec | none | expert_scratch | expert_droid | droid vs none | droid vs scratch |
|---|---|---|---|---|---|
| 10 | 49.4 | 49.1 | **56.6** | 52 vs 27，**p=0.007** | 55 vs 29，**p=0.006** |
| 25 | 43.7 | 47.4 | 49.1 | 46 vs 27，p=0.034 | p=0.57 |
- lr 从 2.5e-5 提到 1e-4 后，π0.5 基线从 17.1 升到 49.4（欠拟合问题基本解决）；n_exec 10 优于 25，与诊断 2 在低 lr 下的结论相反，说明那时的抖动来自欠拟合。
- SOPT（DROID）对 none 和对 scratch 都显著，且 scratch ≈ none：增益来自 DROID 先验而非额外参数。
- 决定：π0.5 以 lr 1e-4、n_exec 10（与其他 host 相同的预设值）进入 17 任务主表，seed 0/1/2 已排入 Jubail 队列（每个 ~24h）。

## E21 Diffusion Policy 17 任务 seed 0（2026-10-15，Jubail 18818112；k5，30k steps，n_exec 10，850 ep/run）
| arm | all 17 | 7 dev | 10 new |
|---|---|---|---|
| none | 73.6 | 64.6 | 80.0 |
| expert_scratch | 69.5 | 62.9 | 74.2 |
| expert_droid | **76.1** | 64.3 | **84.4** |
McNemar：droid vs none 90 vs 69 p=0.11；droid vs scratch 103 vs 47 **p=6e-6**；scratch vs none 59 vs 94 **p=0.006**。
- 与 GR00T 相同的模式：随机初始化插件显著变差，DROID 先验把它拉回并超过基线（+2.5，单 seed 不显著）。增益集中在 10 个新任务（+4.4）。

## E20 X-VLA 17 任务 seed 0 冒烟失败（18818090）与修复（f42fb42）
- `ValueError: Sequence length 1204 exceeds max_len_seq=512`：`lerobot/xvla-base` 的 config 把指令补齐到 1024 个 BART token（30 动作 + 3×50 图像 + 1024 文本）。改为 LeRobot 默认的 64（17 个任务指令最长 17 个 token），对三个 arm 相同。已重新排在队首。

## E22 GR00T 17 任务 seed 1（2026-10-16，Jubail 18819658）及 seed 0+1 汇总
| arm | s0 | s1 | 均值 | 7 dev（2 seed） | 10 new（2 seed） |
|---|---|---|---|---|---|
| none | 81.9 | 79.6 | 80.8 | 79.3 | 81.8 |
| expert_scratch | 77.2 | 79.9 | 78.5 | 74.1 | 81.6 |
| expert_droid | 79.2 | **84.4** | 81.8 | 78.1 | 84.3 |
合并 McNemar（1700 配对 episode）：droid vs none 190 vs 173 p=0.40；droid vs scratch 198 vs 143 **p=0.003**；scratch vs none 135 vs 173 **p=0.035**。
- seed 1 方向与 seed 0 相反（droid +4.8），说明 GR00T 上 seed 间差异大，s0 的负结果不稳定。两 seed 合并：DROID 先验相对同结构随机初始化显著更好，相对 none 持平略高。等 seed 2。
- X-VLA 修复后冒烟通过（18819790，82a9891），训练 1.62 it/s（20k 步约 3.4 h）。

## E12 FastWAM 17 任务 seed 0（2026-10-16，Jubail 18818286；k5，n_exec 10，850 ep/run）
| arm | all 17 | 7 dev | 10 new |
|---|---|---|---|
| none | 41.6 | 36.3 | 45.4 |
| expert_scratch | 50.2 | 40.0 | 57.4 |
| expert_droid | **54.4** | **50.3** | 57.2 |
McNemar：droid vs none 182 vs 74 **p=1e-11**；droid vs scratch 119 vs 84 **p=0.017**；scratch vs none 155 vs 82 **p=2e-6**。
- 与 7 任务 E12 一致：SOPT 对 FastWAM 提升最大（+12.8）。但 17 任务上随机初始化的插件也显著提升（+8.6，7 任务上只有 +1.8），说明 FastWAM 的部分收益来自"额外的本体历史输入"本身；DROID 先验相对 scratch 的额外增益集中在 dev 任务（+10.3），new 任务上两者持平。

## E20 X-VLA 17 任务 seed 0（2026-10-16，Jubail 18819790，f42fb42 修复后；k5，20k steps，n_exec 10，850 ep/run）
| arm | all 17 | 7 dev | 10 new |
|---|---|---|---|
| none | 72.2 | 68.3 | 75.0 |
| expert_scratch | 71.3 | 62.9 | 77.2 |
| expert_droid | 69.9 | 64.3 | 73.8 |
McNemar：droid vs none 68 vs 88 p=0.13；droid vs scratch p=0.40；scratch vs none p=0.57。
- 单 seed 无收益（−2.3，不显著）。X-VLA 的预训练数据本身包含 DROID（290K episode 中的一部分），且使用 proprio 输入；SOPT 提供的 DROID 运动知识对它可能是冗余的。π0.5-DROID 同样在 DROID 上训练过却有显著增益（E14b），二者差异待 seed 1/2 后再分析。seed 1 已提交（18820646）。

## E24 探测：SOPT 交给宿主的特征里有什么（2026-10-16，037fd1f，`scripts/e24_probe.py`，outputs/e24_probe.json）
LIBERO-90 17 个主表任务的全部示范（771 条，13130 个窗口，stride 8），窗口构造与策略输入相同；按 episode 哈希 70/30 划分，岭回归线性探针（alpha 由训练集上按 episode 分组的 5 折 CV 选），报告测试集 R²（任务为准确率，chance 0.06）。
注：此次在 Jubail 登录节点 CPU 上运行（约 6 分钟），用户要求今后不在登录节点运行计算，之后的分析改用计算节点 / DGX。

| 特征 | vel | acc | grip_vel | t_grip | prog | fut 0.5s | fut 1.0s | t_next | task |
|---|---|---|---|---|---|---|---|---|---|
| cur（宿主已有的当前帧） | 0.27 | 0.15 | 0.13 | 0.29 | 0.67 | 0.22 | 0.24 | 0.15 | 0.63 |
| hand（手工历史特征） | 1.00 | 1.00 | 1.00 | 0.39 | 0.76 | 0.79 | 0.60 | 0.27 | 0.70 |
| raw（96×17 全历史，线性） | 1.00 | 1.00 | 1.00 | 0.54 | 0.86 | 0.84 | 0.67 | 0.37 | 0.76 |
| rand（随机初始化 SOPT，3 个种子均值） | 0.89 | 0.50 | 0.93 | 0.72 | 0.94 | 0.71 | 0.64 | 0.47 | 0.83 |
| **droid（预训练，冻结）** | **0.99** | **0.90** | **0.97** | **0.80** | 0.94 | **0.85** | **0.75** | **0.53** | **0.87** |
| droid_ee（EE-only 先验） | 0.99 | 0.90 | 0.98 | 0.83 | 0.95 | 0.84 | 0.75 | 0.51 | 0.87 |
| 微调后 droid 初始化（SmolVLA/GR00T/DP/FastWAM） | 0.96–0.97 | 0.76–0.79 | 0.91–0.95 | 0.76–0.78 | 0.93–0.94 | 0.82–0.86 | 0.72–0.79 | 0.50–0.54 | 0.85–0.87 |
| 微调后 scratch 初始化（同上 4 个宿主） | 0.76–0.84 | 0.40–0.47 | 0.69–0.93 | 0.73–0.77 | 0.94–0.95 | 0.64–0.69 | 0.61–0.67 | 0.43–0.50 | 0.82 |
（t_still 所有特征 R² ≤ 0.07，SmolVLA droid 微调后 0.23；该标签分布极偏，结论不依赖它。SmolVLA 为 7 任务 E7 checkpoint，其余为 17 任务 s0。）

- **宿主缺的是动态信息**：当前帧对速度、加速度、未来 0.5–1 s 运动的 R² 只有 0.15–0.27。
- **DROID 预训练让动态信息线性可读**：相对随机初始化，加速度 0.50→0.90、未来 0.5 s 位移 0.71→0.85、未来 1 s 0.64→0.75、距上次夹爪事件时间 0.72→0.80、到下次夹爪事件时间 0.47→0.53。在非线性量（t_grip、fut 1 s、t_next、task）上还超过对整段原始历史做线性回归。信息本身都在原始历史里，预训练的作用是把它整理成 256 维、线性可用的形式。
- **微调后仍保留大部分**：droid 初始化的编码器在 4 个宿主内微调后，加速度 R² 0.76–0.79、未来运动 0.72–0.86，仍远高于 scratch 初始化微调后（加速度 0.40–0.47）；scratch 编码器微调后甚至比随机初始化时更差（加速度 0.50→0.41，速度 0.89→0.79），即在 5 条示范上训练会让它变得更专门化、丢掉通用的运动信息。与"scratch 历史支路有害"的下游结果一致。
- **任务可识别性**：任何历史编码器都能从运动中以 0.82–0.87 准确率识别任务（当前帧 0.63），说明历史本身带有"抄捷径"的风险，SOPT 并不比随机编码器更少；SOPT 的区别在于额外提供了准确的动态/预测信息。
- 局限：线性探针（非线性探针可能缩小 raw 与 droid 的差距）；微调编码器的 5 条训练示范可能落在探针测试集中（约 85/771 条，影响小，待排除后复核）；没有"是否抓住物体"的仿真标签。

## E25 计划：四个预训练改进逐一验证（2026-10-16，1049961；DGX，`scripts/dgx/e25_pretrain.sbatch`，`scripts/agg_e25.py`）
每个变体 = e2_ar_flow_f100_s 的配方（S，ar_flow，100% DROID，40k 步）+ 一处改动；然后用 E23 的 TrunkBC 协议（17 任务，k5，3 seed，50 init）和 E24 探针评估，按 (seed, task, init) 与 e23_trunk_droid 配对。
| 变体 | 改动 | 动机（来自 E9a / E24） |
|---|---|---|
| event | `model.aux_event=0.5`：每个 token 额外预测到下一次夹爪开合的时间（未来 H 帧内）和开合后 0.27 s 的夹爪开度（夹到物体时手指提前停住） | 增益集中在抓取；探针中"下次夹爪"最弱（0.53） |
| jepa | `model.jepa=1.0`：每个 token 预测 EMA 目标编码器未来 H/P 个 token 的特征（隐空间预测，layer-norm 目标，smooth L1） | 把容量从精确坐标回归转向阶段/意图 |
| speed | `data.speed_aug=[0.5,2.0] speed_aug_prob=0.8`（原默认 [0.8,1.25]、0.5，即原先已有较弱的速度增强） | DROID 15 Hz 真机 vs LIBERO 20 Hz 仿真控制器的速度差异 |
| ctx192 | `model.ctx_len=192`（12.8 s 历史；下游按先验的 ctx_len 构造历史） | 长时程任务中"有历史"本身有用 |
| base | 同配方、预训练 seed 1 | 预训练本身的随机波动（判断变体差异是否超出噪声） |
- 预先规定的判定：在 7 个 dev 任务上相对 e23_trunk_droid 提升 ≥ 2 点且配对 McNemar p < 0.05，且超过 base 对照的差异；10 个 new 任务上方向一致才算通过。通过的变体再在宿主（DP / SmolVLA）上确认。
- 调度：DGX 一次一个作业，与 SmolVLA s1/s2 交替（event → SmolVLA s1 → jepa → base → SmolVLA s2 → speed → ctx192）。每个约 2–3 h，主表的关键路径是 Jubail 上的 π0.5（3 × ~24 h），所以不推迟主表完成时间。127 恢复前无法提交。
- 2026-10-16 更新：127 恢复后发现 DGX 共享盘上没有 DROID 状态数据（`/media/cvpr/langtian/SOPT/data/processed/droid` 不存在），已从 127 的 `~/langtian/SOPT_DATA/processed/droid` rsync 过去（3.4 GB，仅文件传输）。为不让 E25 因缺数据空转，队列改为 SmolVLA s1 → event → jepa → base → SmolVLA s2 → speed → ctx192。

## E23 SmolVLA 17 任务 seed 0（2026-10-16，DGX Spark 作业 158432；k5，n_exec 10，850 ep/run）
| arm | all 17 | 7 dev | 10 new |
|---|---|---|---|
| none | 60.6 | 54.6 | 64.8 |
| expert_scratch | 58.7 | 49.7 | 65.0 |
| expert_droid | **64.0** | **58.3** | **68.0** |
McNemar：droid vs none 110 vs 81 **p=0.042**；droid vs scratch 121 vs 76 **p=0.0017**。
- 与 7 任务 E7（66.0 vs 61.2，p=0.0008）一致；10 个 new 任务上方向相同（+3.2），scratch 插件不带来收益（与 FastWAM 不同）。seed 1/2 在 DGX 队列中。

## E20 X-VLA 17 任务 seed 1（2026-10-16，Jubail 18820646，cb81c9f）及 seed 0+1 汇总
| arm | s0 | s1 | 均值 | 7 dev | 10 new |
|---|---|---|---|---|---|
| none | 72.2 | 70.5 | 71.4 | 65.0 | 75.8 |
| expert_scratch | 71.3 | 71.2 | 71.2 | 64.4 | 76.0 |
| expert_droid | 69.9 | 67.6 | **68.8** | 62.3 | 73.3 |
合并 McNemar（1700 ep/arm）：droid vs none 137 vs 181 **p=0.016（droid 更差）**；droid vs scratch 146 vs 188 **p=0.025（droid 更差）**；scratch vs none p=0.95。
- 两个 seed 方向一致、dev/new 都是负的：X-VLA 是目前唯一 SOPT 显著有害的宿主。scratch 插件与 none 持平，说明问题不在"额外输入"本身，而在 DROID 先验的特征与 X-VLA 的表示冲突。
- 待查假设：(1) X-VLA 预训练已包含 DROID 及其 proprio 输入，SOPT 信息冗余且以加法注入干扰了已对齐的动作 token；(2) SOPT 以宿主 lr 微调对这个宿主过大；(3) X-VLA 的动作空间/归一化与 SOPT 特征尺度不匹配。seed 2 仍在 Jubail 队列中；论文中如实报告为负例。

### E20 分析：X-VLA 负结果是实现问题还是算法问题（2026-10-16，代码审查 + 已有日志；E20d 诊断作业排在 Jubail 队首）
代码审查（对照 LeRobot 0.6.2 `policies/xvla/{modeling_xvla,soft_transformer,action_hub}.py`）未发现实现错误：
- 注入点：`SoftPromptedTransformer.forward` 中 `action_encoder`（DomainAwareLinear，输入 = [带噪动作 20, proprio 20, 时间 32] 拼接）每次前向只调用一次，在拼接视觉/文本 token 和 pos_emb 之前；包装函数签名 `(x, domain_id)` 一致，批大小断言在训练和 10 步去噪中都成立。
- 零初始化：MLP 最后一层权重/偏置为 0，初始时三组完全等价（E20d `--init-check` 再做数值验证）。
- 保存/加载：`save_trainable` 保存所有 requires_grad 参数（含 sopt.prior、sopt.mlp）和 normalizer buffer；`load_trainable` 断言训练参数全部在 ckpt 中且无多余键。评估前 `policy.eval()` 覆盖 sopt 子模块。
- 三组训练用的 85 条示范完全相同（meta.json），日志中除 robosuite 宏文件提示外无 warning / missing key。
- 优化：SOPT 与 X-VLA 策略 transformer 同组（lr 1e-4，VLM 0.1×），与 SmolVLA / DP / GR00T 的 SOPT 学习率相同；clip 10 与 SmolVLA / DP 相同。
- X-VLA 损失对 chunk 末尾的 padding 步也计算（LeRobot 原实现忽略 `action_is_pad`），三组一致，不是 SOPT 特有。

X-VLA 与其他宿主的结构差异（可能的算法原因）：
1. **全量微调**：X-VLA 是唯一连 VLM（Florence-2）一起训练的宿主（0.1× lr，可训练 891M）；SmolVLA 冻结 VLM，π0.5 只训 action expert，GR00T 冻结 backbone，FastWAM 冻结视频专家。
2. **严重过拟合**：训练损失（x0 预测的 MSE，归一化单位）最后 2k 步 X-VLA 0.0017 vs DP 0.0066 vs GR00T 0.013；85 条示范训练 ~48 个 epoch 后几乎背下。droid 组训练损失始终最低（1–2k 步 0.059/0.063 vs none 0.075/0.069），但闭环最差 → 典型的"训练更好、泛化更差"。
3. **本体输入已在同一位置**：X-VLA 的 none 组本身就把当前 proprio 拼进每个动作 token 的输入投影；SOPT 加在同一处，等于在已有本体通道上再加一份历史。
4. **x0 预测 + 单一 transformer**：动作 token 与视觉/文本 token 在同一个 24 层 transformer 中双向注意，没有独立的 action expert / AdaLN；其他 flow 宿主预测速度场。
- 逐任务（2 seeds）：droid 的损失集中在需要精确视觉定位的抓放任务（黄白杯 −13、奶油奶酪→托盘 −10、右侧摩卡壶 −10），抽屉/书本类持平或略升；成功 episode 步数不变（141 vs 144）。
- 主假设（待 E20d 验证）：在全量微调、强记忆的宿主上，SOPT 历史提供了"识别是哪条示范、走到哪一步"的捷径，模型更依赖历史回放、更少依赖视觉 → 新初始位置下抓取失误。E20d 预测：droid 组训练集误差最低但留出示范误差更高；对图像互换的敏感度低于 none；对 SOPT 置零/互换的敏感度高。DP / GR00T 作为对照。

## E21 DP 17 任务 seed 1（2026-10-16，Jubail 18821704）及 seed 0+1 汇总
| arm | s0 | s1 | 均值 | 7 dev | 10 new |
|---|---|---|---|---|---|
| none | 73.6 | 71.4 | 72.5 | 64.6 | 78.1 |
| expert_scratch | 69.5 | 67.9 | 68.7 | 60.0 | 74.8 |
| expert_droid | 76.1 | 77.3 | **76.7** | **68.0** | **82.8** |
合并 McNemar（1700 ep/arm）：droid vs none 206 vs 135 **p=1.4e-4**；droid vs scratch 235 vs 99 **p=7e-14**；scratch vs none 126 vs 191 p=3e-4（scratch 更差）。
- seed 0 时 droid vs none 不显著（p=0.11），两个 seed 合并后显著；dev 与 new 方向一致（+3.4 / +4.7）。

## E20d 结果：X-VLA 诊断（2026-10-16，Jubail 18825586，94d13c1；`outputs/e20d_diag.json`）
开环误差 = 前 10 步动作 MSE（动作 std 单位，dims 0–5），256 帧/划分，配对噪声；留出 = 7 个 dev 任务的第 6–10 条示范（35 条，从未训练；10 个 new 任务的留出画面在离线节点上不可用）。
| run | SOPT 漂移 | 特征/token 范数比 | 训练 normal | 留出 normal | 留出 换图 | 留出 SOPT 置零 | 留出 SOPT 互换 | 留出夹爪一致率 |
|---|---|---|---|---|---|---|---|---|
| X-VLA none s0 / s1 | – | – | 0.001 / 0.002 | 0.446 / 0.469 | 1.548 / 1.570 | – | – | 0.929 / 0.932 |
| X-VLA scratch s0 / s1 | 0.13 | 0.14 / 0.16 | 0.002 / 0.001 | 0.460 / 0.460 | 1.462 / 1.456 | 0.469 / 0.468 | 0.464 / 0.467 | 0.930 / 0.928 |
| X-VLA droid s0 / s1 | 0.12 | 0.48 / 0.49 | 0.001 / 0.001 | **0.432 / 0.421** | 0.975 / 0.919 | 0.528 / 0.520 | 0.664 / 0.671 | **0.950 / 0.950** |
| DP none s0 | – | – | 0.008 | 0.387 | 1.131 | – | – | 0.924 |
| DP scratch s0 | 0.16 | – | 0.007 | 0.387 | 1.110 | 0.427 | 0.391 | 0.926 |
| DP droid s0 | 0.18 | – | 0.006 | **0.320** | 0.637 | 0.523 | 0.595 | 0.945 |
| GR00T none s0 | – | – | 0.014 | 0.444 | 1.499 | – | – | 0.921 |
| GR00T scratch s0 | 0.14 | 0.15 | 0.013 | 0.423 | 0.991 | 0.509 | 0.582 | 0.930 |
| GR00T droid s0 | 0.13 | 0.20 | 0.010 | **0.326** | 0.567 | 0.629 | 0.856 | 0.945 |
- **实现正确**：新建的 expert_droid 与 none 输出逐元素完全相同（max |diff| = 0.0，零初始化成立）；置零 SOPT 特征会改变动作（hook 生效且被使用）；SOPT 编码器漂移 12%，与 DP（18%）/ GR00T（13%）同量级，DROID 先验未被冲掉。
- **过拟合假设被否定**：X-VLA droid 在留出示范上的开环误差反而低于 none（两个 seed −3% / −10%），夹爪一致率更高（0.950 vs 0.93）。开环上 SOPT 对 X-VLA 是有益的，只是幅度最小（DP −17%，GR00T −27%）。
- **"视觉依赖下降"不是 X-VLA 特有**：加 DROID 先验后，换图带来的误差在三个宿主上都大幅下降（X-VLA 1.55→0.95，DP 1.13→0.64，GR00T 1.50→0.57）；对 SOPT 互换的敏感度 X-VLA（+0.24）也不高于 DP（+0.28）、GR00T（+0.53）。
- 唯一突出的差异：X-VLA 中 SOPT 特征相对动作 token 的范数比 0.48，是 GR00T（0.20）的 2.4 倍——X-VLA 的动作 token 来自单个 DomainAwareLinear 投影（范数小），SOPT 在其中占比更大。
- 结论：负结果不是实现错误，也不是开环可见的过拟合，而是**闭环特有**的问题——开环更准的策略在自身 rollout 产生的历史（偏离示范分布）下表现更差。GR00T 开环收益最大但闭环无收益，同样说明开环指标不能预测闭环收益。下一步需要闭环层面的诊断（失败分类 trace，以及降低 SOPT 依赖的干预：SOPT dropout / 冻结 VLM）。

## E14 π0.5-DROID 17 任务 seed 0（2026-10-17，Jubail 18820621，cb81c9f；lr 1e-4，k5，n_exec 10，850 ep/run）
| arm | all 17 | 7 dev | 10 new |
|---|---|---|---|
| none | 57.6 | 51.1 | 62.2 |
| expert_scratch | 56.5 | 45.7 | 64.0 |
| expert_droid | 56.6 | 48.0 | 62.6 |
McNemar：droid vs none 91 vs 100 p=0.56；droid vs scratch p=1.0；scratch vs none p=0.49。
- **7 任务 E14b 的显著收益（56.6 vs 49.4，p=0.007）在 17 任务上没有复现**：同样的 7 个 dev 任务上 droid 48.0 vs none 51.1。差别在于训练任务数（7 → 17，示范 35 → 85）：none 在 dev 任务上基本不变（49.4 → 51.1），droid 下降（56.6 → 48.0）。单 seed，待 s1/s2；与 E5 SmolVLA（seed 0 收益不复现）类似，提示单 seed 的宿主结论不可靠。

## E23 SmolVLA 17 任务 seed 1（2026-10-17，DGX 158465）及 seed 0+1 汇总
| arm | s0 | s1 | 均值 | 7 dev | 10 new |
|---|---|---|---|---|---|
| none | 60.6 | 60.1 | 60.4 | 54.6 | 64.4 |
| expert_scratch | 58.7 | 58.5 | 58.6 | 49.7 | 64.8 |
| expert_droid | 64.0 | 67.4 | **65.7** | **60.4** | **69.4** |
合并 McNemar（1700 ep/arm）：droid vs none 241 vs 150 **p=4.9e-6**；droid vs scratch 260 vs 139 **p=1.4e-9**；scratch vs none p=0.096。
- 与 7 任务 E7 一致且更强；dev / new 方向一致（+5.8 / +5.0）。E25 event 已在 DGX 开始（158598）。

## E25 event 结果（2026-10-17，DGX 158598，7b19dac；预训练 32 min + TrunkBC 3 seeds，整作业 1.6 h）
| | all 17 | 7 dev | 10 new |
|---|---|---|---|
| e23_trunk_droid（参照） | 72.2 | 64.7 | 77.5 |
| e25_trunk_event | **74.0**（+1.8，269:223 p=0.042） | 67.0（+2.4，114:89 p=0.092） | 78.9（+1.4，p=0.24） |
探针 R²（droid → event）：t_grip 0.80 → **0.86**，t_next 0.53 → 0.56，t_still 0.07 → 0.14，prog 0.94 → 0.95；acc 0.90 → 0.81，fut 0.5 s 0.85 → 0.83。
- 辅助目标按预期把"夹爪事件时间"写进了表示，闭环方向为正、dev / new 一致。
- **按预注册标准未通过**：dev +2.4 达到幅度门槛，但 p=0.09 > 0.05；且需与 base 对照（预训练换 seed 的波动）比较后才能判断。先不推进到宿主，等 base 结果。

## E25 jepa 结果（2026-10-17，DGX 158599，整作业 1.7 h）
| | all 17 | 7 dev | 10 new |
|---|---|---|---|
| e23_trunk_droid（参照） | 72.2 | 64.7 | 77.5 |
| e25_trunk_event | 74.0（+1.8，p=0.042） | 67.0（+2.4，p=0.092） | 78.9（+1.4，p=0.24） |
| e25_trunk_jepa | **74.4**（+2.1，284:230 p=0.019） | **69.9**（+5.2，133:78 **p=1.9e-4**） | 77.5（−0.1，151:152 p=1.0） |
探针 R²（droid → jepa）：prog 0.94 → 0.95，task 0.871 → 0.877，其余持平或下降（acc 0.90 → 0.69，grip_vel 0.97 → 0.89，t_next 0.53 → 0.51）。
- dev 上收益大且显著，但 **10 个 new 任务完全没有收益（−0.1）**：按预注册标准（new 上同向）**未通过**。dev 与 new 不一致本身提示 dev 收益可能部分来自预训练随机性与任务的交互——必须由 base 对照（同配方换预训练 seed）判断：若 base 在 dev 上也有 ±5 点的波动，则 jepa 的 dev 收益不可信。
- 探针上 JEPA 没有让局部动力学更可读（acc 反而大降），与"隐空间预测未来 → 更抽象的表示"一致。

## E25 base 对照（2026-10-17，DGX 158607；158606 因节点 git clone 中断失败，ac43d95 加重试）
| | all 17 | 7 dev | 10 new |
|---|---|---|---|
| e23_trunk_droid（参照，预训练 seed 0） | 72.2 | 64.7 | 77.5 |
| base（同配方，预训练 seed 1） | 72.9（+0.6，p=0.49） | 65.1（+0.5，p=0.79） | 78.3（+0.7，p=0.54） |
| event | 74.0（+1.8，p=0.042） | 67.0（+2.4，p=0.092） | 78.9（+1.4，p=0.24） |
| jepa | 74.4（+2.1，p=0.019） | 69.9（+5.2，p=1.9e-4） | 77.5（−0.1，p=1.0） |
探针：base 与参照几乎相同（acc 0.88 vs 0.90，t_next 0.54 vs 0.53）。
- **预训练本身很稳定**：换预训练 seed，总体只差 +0.6，探针几乎不变；但**单任务**波动可达 ±8 点（base 逐任务 −8.0 … +8.0）。
- **jepa 的 dev 收益集中在一个任务**：wine rack +20.7（31.3 → 52.0），一个任务贡献了 dev +5.2 中的约 3 点；其余 dev 任务在 base 的单任务波动范围内，new 上无收益。→ **未通过**（new 不同向；收益不具普遍性）。
- **event** 在 dev / new 上都同向为正，相对 base 净增约 +1.2（总体），但 dev p=0.09 → **未通过**（证据不足，而非反向）。
- 小结：两个辅助目标都没有达到预注册门槛；E25 继续跑 speed、ctx192。若最终无变体通过，论文中保持原配方，E25 作为"预训练配方的消融：原配方已接近局部最优、预训练 seed 稳健"报告。

## 17 任务宿主补齐 seed（2026-10-17，Jubail：GR00T s2 18825596，X-VLA s2 18826423，DP s2 18826936，FastWAM s1 18827085）
配对 McNemar 汇总全部 seed（每臂 17 任务 × 50 init × seed 数）。

| 宿主 | seeds | none | +SOPT scratch | +SOPT DROID | DROID vs none | DROID vs scratch |
|---|---|---|---|---|---|---|
| DP | 0,1,2 | 72.9 ± 1.3 | 71.2 ± 4.4 | **77.5 ± 1.5** | 321:203 **p=2.9e-7** | 337:176 p=1.1e-12 |
| FastWAM | 0,1 | 42.9 ± 1.7 | 49.2 ± 1.4 | **54.8 ± 0.6** | 351:149 **p=7.8e-20** | 275:181 p=1.2e-5 |
| GR00T N1.5 | 0,1,2 | 81.1 ± 1.3 | 79.3 ± 1.8 | 81.1 ± 2.8 | 271:271 p=1.0 | 284:236 p=0.039 |
| X-VLA | 0,1,2 | 71.7 ± 1.1 | 71.8 ± 0.9 | 69.2 ± 1.4 | 228:291 **p=0.0064（变差）** | 230:295 p=0.0052 |

dev / new 拆分（agg_split）：
- DP：dev 64.9 → 69.0，new 78.5 → 83.5（两边同向 +4~5）。
- FastWAM：dev 37.0 → 51.0，new 47.0 → 57.4。
- GR00T：dev 79.0 → 76.9，new 82.7 → 84.1（互相抵消）。
- X-VLA：dev 67.0 → 63.2，new 75.0 → 73.4（两边都变差）。

- DP、FastWAM 的收益随 seed 增加更显著；GR00T 3 seed 完全打平；X-VLA 的负效应在 3 seed 下稳定（−2.5，两边同向），与 E20d 诊断"实现正确、闭环特有"一致。
- 剩余：FastWAM s2（18827134 运行中）、π0.5 lr1e-4 s1（18828640 运行中）/ s2（排队），SmolVLA s2（DGX 158608 运行中）。

## E26 / E27 计划：强宿主为什么不受益（2026-10-17；完整分析与预注册判据见 `docs/HOST_ANALYSIS.md`）
- 三个候选解释：H1 冗余（强视觉宿主能从画面推断阶段 / 进度）、H2 接口耦合（GR00T 组件代价 −1.8；X-VLA 范数比 0.48）、H3 copycat / 因果混淆（开环更好、闭环更差；加 SOPT 后对视觉的依赖下降）。
- **E26**（`scripts/slurm/e26_trace.sbatch`）：X-VLA / GR00T / π0.5 / DP × none / scratch / droid，seed 0，17 任务 × 50 init。
  - 重新评测，记录仿真轨迹（`--trace 4`）和每步动作（`--act-log`，本次新增）。
  - 用 `scripts/e26_taxonomy.py` 做 17 任务通用的失败归类：目标物体和抓取距离由全部 run 的成功 episode 统一标定；另算 frozen / grip_cycles / jerk_ratio。
  - 做配对分析：SOPT 新造成了哪些失败、修复了哪些失败。
  - 同时检查重新评测的成功率能否复现原始 rollouts。
- **E27**（同一作业，`scripts/e27_host_probe.py`）：在宿主动作解码器的输入上，对 E24 的时间目标做线性探针（k5 训练示范上拟合，留出示范上评分）。
  - host(none)+SOPT 相对 host(none) 的 R² 增量 = SOPT 带来的、宿主原本没有的信息。
- 按用户要求排在 Jubail 现有队列之后。E28 干预（SOPT dropout / 历史扰动 / 门控注入）根据 E26 / E27 结果再定。

## E12 FastWAM 17 任务 seed 2（2026-10-17，Jubail 18827134）及 3 seed 汇总
| | s0 / s1 / s2 | 均值 ± SD | dev 7 | new 10 |
|---|---|---|---|---|
| none | 41.6 / 44.1 / 44.8 | 43.5 ± 1.7 | 37.0 | 48.1 |
| +SOPT scratch | 50.2 / 48.2 / 50.8 | 49.8 ± 1.4 | 42.1 | 55.1 |
| +SOPT DROID | 54.4 / 55.2 / 54.0 | **54.5 ± 0.6** | 50.6 | 57.3 |
- DROID vs none：525:245，**p=2.7e-24**；DROID vs scratch：403:282，p=4.3e-6；scratch vs none：411:252，p=7.0e-10。
- FastWAM 是唯一一个**光加历史（scratch）就显著提升**的宿主（+6.3）。这符合 `HOST_ANALYSIS.md` 的统一解释：基线视觉通路越弱，历史的价值越大。DROID 预训练在此基础上再加 +4.7。

## E14 π0.5-DROID 17 任务 seed 1（2026-10-18，Jubail 18828640；lr 1e-4）及 seed 0+1 汇总
| | s0 / s1 | 均值 | dev 7 | new 10 |
|---|---|---|---|---|
| none | 57.6 / 59.6 | 58.6 | 51.9 | 63.4 |
| +SOPT scratch | 56.5 / 54.8 | 55.6 | 46.6 | 62.0 |
| +SOPT DROID | 56.6 / 59.8 | 58.2 | 51.9 | 62.6 |
- DROID vs none：184:192，p=0.72；DROID vs scratch：208:165，**p=0.030**；scratch vs none：150:201，**p=0.0075**。
- π0.5 与 GR00T 的模式相同：光加组件就有代价（−3.0），DROID 信息补回来（+2.6），净效应为 0。在三个强宿主中，只有 X-VLA 是"组件无害、DROID 特征有害"。这支持 `HOST_ANALYSIS.md` 中 GR00T / π0.5 属于 H2（接口代价）、X-VLA 另有原因的划分。E26 会直接比较三者的失败类型。

## E29 设计：改进后的 SOPT 能补上哪些宿主没有的信息（2026-10-18；详见 `docs/HOST_ANALYSIS.md` §8）
- 阶段 A（分析，挂在已排队作业上，不新增作业）：
  - A1 执行反馈：`e26_taxonomy.py` 新增 blocked 指标和失败预测 AUC，随 E26 运行。
  - A2 多 token 读出：`e24_probe.py --multi`，E25 的 DGX 探针步骤已改为带 `--multi`，speed / ctx192 自动包含。
  - A3 长记忆：`e24_probe.py` 新增 stage 目标（已发生的夹爪事件数），ctx192 对比 ctx96。
- 阶段 B（B1 执行反馈分支 / B2 多 token 读出 / B3 长记忆）只对通过阶段 A 预注册门槛的方向执行：先 TrunkBC 筛选，再上 GR00T / π0.5。

## E13 libero_spatial k5 seed 2 补记（18812643）：3 seed 合并后负效应消失
| | s0 / s1 / s2 | 均值 |
|---|---|---|
| none | 83.0 / 80.6 / 77.0 | 80.2 |
| +SOPT scratch | 79.4 / 80.2 / 81.0 | 80.2 |
| +SOPT DROID | 79.2 / 79.2 / 80.2 | 79.5 |
- DROID vs none：100:110，p=0.53；scratch vs none：87:87，p=1.0。两个 seed 时的"spatial 上变差"（p=0.03）**没有复现**，是 none 在 s0 偏高造成的。此前基于它的"历史导致选错碗"推测失去依据。

## 掉分情况的共同模式（2026-10-18，本地分析已取回的 rollouts，`scripts/analysis/drop_pattern{,2}.py`）
对象：6 个"变差或零收益"的情况（X-VLA droid、GR00T scratch / droid、π0.5 scratch / droid、DP scratch），以及 DP / FastWAM 的正收益作对照。逐任务配对（seed × 50 init）。
- **没有跨宿主的任务级模式**：各情况逐任务变化之间的 Spearman 相关大多 |ρ| < 0.4，符号不一致。不同宿主掉分的任务基本不同。
- **"靠视觉区分是哪一个物体"的任务（4 个：右边的书、中间的黑碗、右边的摩卡壶、左碗叠右碗）并没有掉得更多**：X-VLA droid −3.5 vs 其余 −2.2；GR00T、π0.5 在这些任务上反而略好。任务层面不支持"视觉区分能力下降 → 抓错物体"。
- **对视觉依赖下降本身不是掉分的标志**：E20d 中，提升的 DP 和不变的 GR00T 加 SOPT 后对视觉的依赖下降得同样多甚至更多。
- 反复出现的个别任务：
  - "把巧克力布丁放进上层抽屉**并关上**"（唯一的两阶段任务之一）：GR00T droid 3:17（p=0.003）、π0.5 droid 2:17（p=0.001），两个强宿主 + DROID SOPT 都显著变差；DP、FastWAM 在同一任务上提升。
  - "奶油奶酪放进托盘"：X-VLA droid 7:31、GR00T scratch 2:10、DP scratch 4:16 变差；但 π0.5 droid 14:4、DP droid 提升。
  - 逐任务共做了约 100 次检验，单个 p≈0.003 不能单独作为结论。布丁任务在两个独立宿主上同向（Fisher 合并 p≈5e-5）是最值得跟进的线索。
- 对 E26 的影响：重点看 GR00T / π0.5 droid 在布丁两阶段任务上的失败类型（是否在第一阶段完成后停住或重复，即 frozen / grip_cycles），以及 X-VLA 在奶油奶酪 / 杯子两个任务上的失败类型。

## E31 / E32 设计（2026-10-18，详见 `docs/HOST_ANALYSIS.md` §10）
- 方向由用户提出：用交叉注意力让模型自己学会何时依赖视觉语言、何时依赖运动先验；阶段任务不靠文本，靠轨迹中的隐含语义。
- E31：双路交叉注意力 + 学习门控（以动作 token 隐状态和去噪时间为条件）+ 模态 dropout + SOPT 低学习率；对照组为加法注入、AdaLN 注入（10-16 提出，未跑过）、固定门控、无 dropout。先在 GR00T 上跑，1 seed 筛选。
- E32：以夹爪事件分段的阶段记忆，用 DROID 状态做段级预训练（段级 JEPA + 下一关键帧）；评测看两阶段任务和 libero_10。
- DROID 统计（本地，元数据 + next_event）：轨迹中位 14.7 s，41% 超过 17 s；22% 有 ≥2 次抓放循环；40% 的松开之后还有后续抓取；明确"放进抽屉再关上"的 572 条。现有预训练窗口只有 8.5 s，学不到阶段结构。

## E31 / E33 实现并排队（2026-10-18，按用户要求：AdaLN、交叉注意力、更长历史消融）
- **E31（Jubail，`scripts/slurm/e31_groot.sbatch`，GR00T，17 任务，k5，seed 0，与 e22_groot_{none,expert_droid}_s0 配对）**
  - `src/sopt/vla/groot.py` 新增两种接法：
    - `adaln_*`：SOPT 特征经零初始化 MLP 加到 DiT 的时间步嵌入上，通过 AdaLayerNorm 调制每一层和输出层；
    - `xattn_*`：每隔 4 个 block（视觉语言交叉注意力层）后加一路运动交叉注意力，读取全部 SOPT 历史 token。这一路有独立的 softmax，输出投影零初始化，由门控 σ(W[h, temb]) 加权，门以 token 的隐状态和去噪时间为条件；`--cond-drop` 按样本整路关闭（模态 dropout）。
  - `vla_train.py` 新增 `--prior-lr-mult`（SOPT 编码器学习率倍数）；历史长度改为跟随先验的 ctx_len，写入 meta，`vla_eval.py` 读取。
  - SET=screen 三组：adaln_droid、xattn_droid + dropout 0.3 + SOPT lr ×0.1、xattn_droid 不加正则。
  - 作业开头做两项检查，任一失败即中止：
    - 零初始化等价检查：新建的 adaln / xattn 策略与 none 逐元素相同，`vla_diag.py --init-arms`；
    - 每种接法 30 步冒烟训练 + 1 个初始状态的评测。
  - 排在 Jubail 队列 E26 之后。scratch 对照（SET=scratch）和补 seed 在筛选后决定。
- **E33 历史长度消融**
  - DGX 预训练：ctx48（3.2 s）/ ctx96（现有）/ ctx192（12.8 s）/ ctx384（25.6 s），每个接 TrunkBC 17 任务 × 3 seed，加探针（含新的 stage 目标）。
  - 长上下文变体用 `data.left_pad = ctx − 96`：每个窗口仍有 ≥96 帧真实历史，episode 和锚点与 ctx96 完全相同，只是多出更早的历史。这与 LIBERO rollout 早期用第一帧填充历史的方式一致。ctx192 尚未提交，已改为同样设置。
  - DGX 队列：speed → ctx192 → ctx48 → ctx384。
  - 宿主层面（SET=history）：GR00T expert_droid / xattn_droid 换用长上下文先验，等 DGX 产出先验后排队。

## E19 libero_spatial 结果（2026-10-18，Jubail 18831408；SmolVLA expert_droid + SOPT dropout 0.5，k5，3 seed，`scripts/agg_e19.py libero_spatial`）
| | s0 / s1 / s2 | 均值 |
|---|---|---|
| none | 83.0 / 80.6 / 77.0 | 80.2 ± 3.0 |
| +SOPT DROID | 79.2 / 79.2 / 80.2 | 79.5 ± 0.6 |
| +SOPT DROID，dropout 0.5 | 78.8 / 78.4 / 78.6 | 78.6 ± 0.2 |
- dropout vs none：84:108，p=0.097；dropout vs droid：82:96，p=0.33。
- SOPT dropout **没有**带来改善。而 spatial 上 3 seed 合并后本来就没有负效应（droid vs none p=0.54），所以 E19 原本要修复的问题已不存在；这一组结果只说明 dropout 0.5 在 spatial 上不提升。
- 对 E31 的含义：单纯按样本丢弃 SOPT 不是有效的正则。E31 完整版仍保留 dropout 0.3 作为一个组件，但"不加正则"的对照组会直接给出它是否有用。
