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

## E19 LIBERO-90 结果（2026-10-18，Jubail 18831646；SmolVLA expert_droid + SOPT dropout 0.5，k5，3 seed，`scripts/agg_e19.py libero90`）
| | s0 / s1 / s2 | 均值 |
|---|---|---|
| none | 58.9 / 58.3 / 66.6 | 61.2 ± 4.6 |
| +SOPT DROID | 68.6 / 64.3 / 65.1 | 66.0 ± 2.3 |
| +SOPT DROID，dropout 0.5 | 66.6 / 63.7 / 61.7 | 64.0 ± 2.4 |
- droid vs none：132:82，p=7.7e-4（E7 结论复现）；dropout vs none：123:94，p=0.057；dropout vs droid：75:96，p=0.13。
- dropout 0.5 把 SOPT 收益从 +4.8 缩到 +2.8（方向一致但不显著）。连同 spatial 的结果，E19 结论：**按样本丢弃 SOPT 在两个 suite 上都不提升，LIBERO-90 上还削弱收益**。SmolVLA 不存在需要靠 dropout 修复的"视觉依赖丧失"问题。
- 对 E31 的含义：xattn 的 cd0.3 + 先验 lr×0.1 组合可能同样削弱收益；screen 组里的 xattn_droid 无正则对照正好用来检验。若无正则版更好，后续 seed 与 SET=history 改用无正则版本。

## π0.5 17 任务 3 seed 汇总（2026-10-18，Jubail 18829161 = s2；lr 1e-4，k5，`agg_host.py --prefix e14_pi05 --tag _lr1e-4_t17 0 1 2`）
| | s0 / s1 / s2 | 均值 | 7 dev | 10 new |
|---|---|---|---|---|
| none | 57.6 / 59.6 / 56.9 | 58.1 ± 1.4 | 50.6 | 63.3 |
| +SOPT scratch | 56.5 / 54.8 / 56.8 | 56.0 ± 1.1 | 47.3 | 62.1 |
| +SOPT DROID | 56.6 / 59.8 / 59.2 | 58.5 ± 1.7 | 52.2 | 62.9 |
- droid vs none：288:277，p=0.67（无增益）；scratch vs none：246:298，p=0.029（接口代价）；droid vs scratch：317:254，p=0.009（DROID 预训练把代价补回）。和 GR00T 一样是 H2 模式（组件有代价，DROID 先验刚好抵消），3 seed 后结论不变。
- 两阶段布丁任务（放进抽屉并关上）：droid vs none 2:18，p=4e-4（s2 又多 0:1）；scratch 6:16，p=0.053。同一物体的单阶段任务（布丁放到盘子右边）droid 8:7，无差异 → 掉分特定于阶段结构，不是物体或场景。E26 的 trace 会给出失败类型。

## E13 libero_object k5 3 seed 汇总（2026-10-18，Jubail 18831951 = s2；SmolVLA，`agg_e13.py libero_object 5`）
| | s0 / s1 / s2 | 均值 |
|---|---|---|
| none | 77.6 / 83.6 / 83.0 | 81.4 ± 3.3 |
| +SOPT scratch | 78.6 / 82.8 / 87.6 | 83.0 ± 4.5 |
| +SOPT DROID | 84.2 / 81.6 / 83.6 | 83.1 ± 1.4 |
- droid vs none：175:149，p=0.17；scratch vs none：126:102，p=0.13；droid vs scratch：140:138，p=0.95。
- 结论：libero_object 上 SOPT 两个臂都为正（约 +1.6），但都不显著，DROID 预训练相对 scratch 没有额外收益（object 任务的 pick-place 结构简单、上下文短，先验内容用不上）。seed 0 的 +6.6 是 none 偏低造成的。标准 suite 目前：spatial 持平（p=0.53）、object 小正（p=0.17），goal / libero_10 补 seed 中。

## E13 libero_goal k5 2 seed 汇总（2026-10-18，Jubail 18834313 = s1；SmolVLA，`agg_e13.py libero_goal 5`）
| | s0 / s1 | 均值 |
|---|---|---|
| none | 75.0 / 80.4 | 77.7 |
| +SOPT scratch | 76.8 / 79.2 | 78.0 |
| +SOPT DROID | 80.6 / 81.4 | 81.0 |
- droid vs none：101:68，p=0.014；droid vs scratch：86:56，p=0.015；scratch vs none：83:80，p=0.88。
- seed 1 的增益（+1.0）小于 seed 0（+5.6），但方向一致；2 seed 合并后仍显著，而且收益来自 DROID 先验本身（scratch 臂无效果）。libero_goal 是标准 suite 里唯一"同一场景、多种目标"的 suite，策略必须从语言 + 当前进展区分任务，与 SOPT 的运动阶段信息互补。
- 为凑齐 3 seed，goal s2、libero_10 s2 追加到 Jubail 队列末尾（E31 screen 之后）。

## E13 libero_10 k5 2 seed 汇总（2026-10-18，Jubail 18834462 = s1；SmolVLA，`agg_e13.py libero_10 5`）
| | s0 / s1 | 均值 |
|---|---|---|
| none | 44.6 / 45.0 | 44.8 |
| +SOPT scratch | 50.8 / 49.6 | 50.2 |
| +SOPT DROID | 50.4 / 47.0 | 48.7 |
- scratch vs none：149:95，p=6.6e-4；droid vs none：156:117，p=0.021；droid vs scratch：117:132，p=0.38。
- libero_10（长时程、多数为两阶段任务）上，**SOPT 组件本身**（6.4 s 本体感觉历史 + 因果编码器，从头训练）就带来 +5.4，两个 seed 都复现；DROID 预训练没有额外收益。说明长时程任务缺的主要是"历史/进度"这个输入通道，而当前 DROID 先验的内容并不比在 k5 示范上从头学到的更好。
- 与 17 任务结论对照：π0.5 / GR00T 在两阶段布丁任务上被 DROID SOPT 拉低，而 SmolVLA 在 libero_10 的两阶段任务上靠历史明显受益 → 阶段信息有用，问题在于强宿主的接法（E31）和先验内容（E32 段级阶段记忆、E33 更长历史）。
- 标准 suite 当前（SmolVLA k5）：spatial 持平（3 seed，p=0.53）、object +1.7（3 seed，p=0.17）、goal +3.3（2 seed，p=0.014）、libero_10 +3.9（2 seed，p=0.021；scratch +5.4）。goal / libero_10 s2 已排队。

## E26 / E27 结果（2026-10-19，Jubail 18834721；详见 HOST_ANALYSIS.md §11）
- 4 宿主 × 3 臂 seed 0 重评测加轨迹：与原评测逐 episode 一致率 96–99%。分类标定修正：位移从场景稳定后的第一帧开始算（ketchup_1 在 reset 时下落 0.55 m，导致 3 个任务的目标物体被标错）。作业里的 `outputs/e26_taxonomy.json` 是修正前的版本，修正后在本地重算，结果以本条和 HOST_ANALYSIS §11 为准。
- 强宿主（GR00T / X-VLA / π0.5）中 SOPT 新增的失败集中在 grasp_miss（droid 臂 +28 / +25 / +13），"去哪"类没有增加；DP 的 DROID SOPT 修复了 grasp_miss（−19）。
- 机制是**抓空后照常搬运**：grasp_miss 中闭合夹爪后 2 s 内 EE 上升超过 5 cm 的比例，GR00T 从 none 的 15.7% 到 scratch 44.9% 再到 droid 81.9%；X-VLA 51.5→68.4%；π0.5 48.0→64.4%。重试次数同时下降，jerk_ratio 也下降（动作更平滑）。即 copycat / 惯性，表现在抓取结果核对上。
- 两阶段布丁任务的掉分几乎全部是第一阶段的 grasp_miss（布丁抓取距离只有 1.1 cm），不是阶段混淆。"阶段 / 记忆"这条线索被削弱。
- A1 执行反馈门槛通过（AUC 0.73–0.89，高 blocked 失败占 40–76%）。
- E27：宿主没有 SOPT 的局部动力学和阶段信息（H1 冗余被否定）。droid 臂把速度信息传进了解码器，但没有传入阶段信息。
- 新增 E34（只评测）：闭合夹爪后 2 s 内关掉 SOPT 输入，用来检验这种惯性是否来自 SOPT 历史；排在 E31 之后。

## E13 libero_goal k5 3 seed 汇总（2026-10-19，Jubail 18837969 = s2；SmolVLA）
| | s0 / s1 / s2 | 均值 |
|---|---|---|
| none | 75.0 / 80.4 / 79.2 | 78.2 ± 2.8 |
| +SOPT scratch | 76.8 / 79.2 / 80.8 | 78.9 ± 2.0 |
| +SOPT DROID | 80.6 / 81.4 / 80.2 | 80.7 ± 0.6 |
- droid vs none：139:101，p=0.017（3 seed 确认，+2.5）；droid vs scratch：120:93，p=0.075；scratch vs none：p=0.51。
- 3 个 seed 中 droid 都 ≥ none（+5.6 / +1.0 / +1.0），seed 方差也最小。libero_goal 是标准 suite 里第一个 3 seed 确认的显著提升。

## E31 screen seed 0 结果（2026-10-19，Jubail 18837197；GR00T，17 任务 × 50 init，与 E22 s0 配对，`scripts/agg_e31.py 0`）
- 启动检查通过：3 个新接口初始化时与 none 完全相等（max |Δ| = 0.0），冒烟训练和评测通过。20k 步约 3.7 h（1.4–1.5 it/s）。

| 臂 | 全部 | 7 dev | 10 new | 布丁两阶段 |
|---|---|---|---|---|
| none | 81.9 | 79.1 | 83.8 | 92 |
| expert_droid（加法，E22） | 79.2 | 76.9 | 80.8 | 76 |
| expert_scratch（E22） | 77.2 | 72.6 | 80.4 | 88 |
| adaln_droid | 79.9 | 75.1 | 83.2 | 84 |
| xattn_droid（无正则） | 80.4 | **80.3** | 80.4 | 88 |
| xattn_droid + dropout 0.3 + 先验 lr×0.1 | **80.5** | 77.4 | 82.6 | 90 |

- 对 none：adaln 80:97（p=0.23），xattn 73:86（p=0.34），xattn+reg 79:91（p=0.40）。所有接口都没有超过 none，也都没有显著变差。加法 expert_droid 这一 seed 是 78:101（p=0.10）。
- 对加法 expert_droid：三种新接口都略好（85:79 / 92:82 / 87:76，都不显著）。两个 xattn 在布丁两阶段任务上把掉分补回来了（vs expert_droid 6:0，p=0.031；8:1，p=0.039）。
- 初步解读：交叉注意力接口消除了加法注入在布丁任务上的损失（与 E26 的抓空照常搬运机制一致，待 E34 的 trace 验证），但整体仍然无法超过 GR00T none。强宿主上，接口只能做到"不伤"，做不到"提升"。
- 下一步：E34（已解除依赖，自动提交）给出 xattn+reg 的 phantom 比例，以及接触窗口内关掉 SOPT 的效果；E31 SET=screen SEED=1 排队，用来确认这些单 seed 差异。

## E13 libero_10 k5 3 seed 汇总（2026-10-19，Jubail 18839255 = s2；SmolVLA）
| | s0 / s1 / s2 | 均值 |
|---|---|---|
| none | 44.6 / 45.0 / 54.0 | 47.9 ± 5.3 |
| +SOPT scratch | 50.8 / 49.6 / 53.0 | 51.1 ± 1.7 |
| +SOPT DROID | 50.4 / 47.0 / 46.2 | 47.9 ± 2.2 |
- scratch vs none：201:152，p=0.011（3 seed 仍显著，+3.2）；droid vs none：205:205，p=1；droid vs scratch：168:217，p=0.014。
- s2 的 none 偏高（54.0），droid 偏低（46.2），2 seed 时 droid 的 +3.9 消失了。3 seed 结论：libero_10 上有用的是 SOPT 组件，也就是因果编码的 6.4 s 本体感觉历史；**DROID 预训练的先验反而显著差于从头训练**。
- 解读：DROID 先验在短时程运动统计上很强（E24：速度、加速度 R² 约 0.9 以上），但 libero_10 这类长任务需要的是任务内的进度和阶段表示，这类表示用 k5 示范从头学更合适。这与 E27 一致：droid 臂并没有把阶段信息传进宿主。E32（段级阶段记忆预训练）和 E33（更长历史）正是针对这一点。
- 标准 suite 3 seed 汇总（SmolVLA k5，droid vs none）：spatial −0.7（p=0.53），object +1.7（p=0.17），goal +2.5（p=0.017），libero_10 0.0（p=1；scratch +3.2，p=0.011）。

## E34 结果（2026-10-19，Jubail 18844862；GR00T，只评测，17 任务 × 50 init 加轨迹，固定 E26 标定）
| checkpoint / 评测方式 | 成功率 | grasp_miss | 其中抓空照常搬运 | 布丁两阶段 |
|---|---|---|---|---|
| e22 none | 81.2 | 83 | 15.7%（13） | 46/50 |
| e22 expert_droid | 78.8 | 116 | 81.9%（95） | 38/50 |
| e22 expert_droid，闭合后 2 s 关 SOPT | **49.8** | 171 | 2.3%（4） | 24/50 |
| e31 xattn_droid+reg | 80.0 | 88 | 43.2%（38） | 45/50 |
| e31 xattn_droid+reg，闭合后 2 s 关 SOPT | 80.0 | 88 | **23.9%（21）** | 47/50 |
| e31 xattn_droid+reg，全程关 SOPT | 80.9 | 84 | 17.9%（15） | 46/50 |
- 配对：xattn 2 s 关闭 vs 不关 29:29（p=1）；全程关 vs 不关 61:53（p=0.51）；全程关 vs e22 none 79:81（p=0.94）；e22 expert_droid 2 s 关闭 vs 不关 18:265（p=2e-57）。
- **因果确认**：关掉 SOPT 输入会让抓空照常搬运显著减少（xattn 43%→24%，全程关为 18%，接近 none 的 16%）。这一行为确实来自 SOPT 历史。
- **但它不决定成败**：关闭后 grasp_miss 数量不变（88→88），只是失败的样子从"抬起来搬空气"变成了"停在原地或重试"。成功率不变。抓空照常搬运是抓空之后的症状，不是失败的原因。
- **加法接口高度依赖 SOPT**：e22 expert_droid 在闭合后关 SOPT 会崩溃（78.8→49.8，place_fail 从 49 增至 244，grip_cycles 11.5）。抓住之后的搬运阶段完全靠 SOPT 特征驱动。训练时 SOPT 一直开着，所以这一结果属于分布外，只说明依赖程度，不说明机制。
- **xattn + dropout 0.3 学会了不用 SOPT**：全程关掉 SOPT 与 none 无差别（80.9 vs 81.2），开着也没有收益。它对 GR00T "不伤"的原因就是基本不用。所以后续 xattn 的 seed 和新实验都改用无正则版（E31 s0 中它的 dev 是 80.3，高于 none 的 79.1）。
- **对强宿主的总结（LIBERO-90 17 任务）**：GR00T 的失败以视觉精度决定的 grasp_miss 为主。SOPT 提供的局部动力学和阶段信息（E27 证明宿主原本没有）在这类短任务上转化不成成功率。加法注入的代价是更多 grasp_miss，交叉注意力去掉了这一代价，但没有带来收益。
- **下一步 E35**：在长时程 suite 上测强宿主（GR00T libero_10 k5）。SmolVLA 在这里从头训练的历史组件显著有用（+3.2，p=0.011）。SET=a 为 none / expert_scratch / expert_droid，SET=b 为 xattn_scratch / xattn_droid（无正则），使用 EE-only 先验。脚本 `scripts/slurm/e35_groot_std.sbatch`，排在 E31 s1 之后。

## E31 screen seed 1 + 2 seed 汇总（2026-10-20，Jubail 18845047；GR00T，与 E22 同 seed 配对，`agg_e31.py 0 1`）
| 臂 | s0 | s1 | 2 seed | 7 dev | 10 new | 布丁两阶段 |
|---|---|---|---|---|---|---|
| none | 81.9 | 79.6 | 80.8 | 79.3 | 81.8 | 92 |
| expert_droid（加法） | 79.2 | **84.4** | 81.8 | 78.1 | 84.3 | 83 |
| expert_scratch | 77.2 | 79.9 | 78.5 | 74.1 | 81.6 | 89 |
| adaln_droid | 79.9 | 80.6 | 80.2 | 76.9 | 82.6 | 81 |
| xattn_droid | 80.4 | 83.9 | **82.1** | **80.0** | 83.6 | 89 |
| xattn_droid + reg | 80.5 | 82.2 | 81.4 | 78.6 | 83.3 | 92 |
- s1 上 none 偏低（79.6），expert_droid 对 none 112:72，p=0.004（这一 seed 新任务 +8.0）。2 seed 合并后：expert_droid vs none 190:173（p=0.40），xattn_droid vs none 166:143（p=0.21），xattn+reg 166:156（p=0.62），adaln 174:183（p=0.67）。
- xattn_droid（无正则）是唯一 dev 和 new 都不低于 none 的接口，2 seed 均值最高；布丁两阶段任务上，加法 droid 的掉分（3:12，p=0.035）在 xattn 上消失（3:6，p=0.51）。adaln 在布丁任务上同样掉分（2:13，p=0.007），且整体最弱。
- GR00T 的 seed 间波动（none 79.6–81.9、expert_droid 79.2–84.4）与效应量相当，需要第 3 个 seed：E31 SET=screen SEED=2 已排队（E22 s2 已有）。

### 2026-10-20 E35 SET=b（18851299）被系统取消
- cn264 上运行到约 10.7k / 20k 步时，于 04:00 被 root（uid 0）取消，不是我们的操作，日志里没有错误。没有中间 checkpoint，已重新放到 Jubail 队首，从头重跑。

## E35 SET=a seed 0：GR00T libero_10（2026-10-20，Jubail 18845212，cn262，commit b954db7；k5，EE-only 先验，10 任务 × 50 init）
| 臂 | 成功率 |
|---|---|
| none | 61.8 |
| expert_scratch | **63.8** |
| expert_droid | 59.2 |
- 配对 McNemar：scratch vs none 74:64（p=0.44）；droid vs none 68:81（p=0.33）；droid vs scratch 57:80（p=0.060）。
- 与 SmolVLA libero_10（3 seed：none 47.9 / scratch 51.1 / droid 47.9；scratch vs none p=0.011，droid vs scratch p=0.014）的排序相同：从头训练的历史组件 ≥ none > DROID 预训练。目前只有 1 个 seed，不显著。
- 分任务（none / scratch / droid）：
  - "put both moka pots on the stove"：26 / 56 / 62。两个 SOPT 臂都 +30 以上，是唯一的大幅收益。这是双物体、重复同一动作的任务，需要记住做到第几步。
  - 掉分集中在 "alphabet soup + cream cheese"（70 / 64 / 50）、"mug + pudding"（48 / 54 / 38）、"mug in microwave and close"（46 / 32 / 40）。
- 初步判断：强宿主在长时程任务上也能从历史获益，但这一收益来自"历史"本身（scratch 同样有），DROID 预训练没有额外贡献，反而略差。SET=b（xattn_scratch / xattn_droid，18855804）正在运行，再决定是否补 seed 1/2。

### 2026-10-20 E35 SET=b 第二次被取消（18855804）：原因是闲置 GPU
- 同样在 cn264，于 08:00:20 被取消，当时训练到 18.3k / 20k 步。两次取消都恰好落在整 4 小时边界（04:00、08:00）。`nvidia` 分区的 PreemptMode=OFF，所以不是抢占。
- SET=b 只有 2 个臂，但申请了 3 块 A100，有 1 块一直闲置。推断原因是集群定期清理占着闲置 GPU 的作业（推断，管理员没有给出原因）。
- 修复：2 个臂的 SET 改为 `SBATCH_GRES=gpu:a100:2` 提交（这个环境变量优先于脚本里的 #SBATCH 指令）。说明已写进 e35 和 e31 两个 sbatch 的头部。SET=b 已重新排队，排在 E35 SET=a s1（18861939）之后。

## E31 screen seed 2 + 3 seed 汇总（2026-10-20，Jubail 18857850；GR00T，17 任务，与 E22 同 seed 配对，`agg_e31.py 0 1 2`）
| 臂 | s2 | 3 seed | 7 dev | 10 new | 布丁两阶段 |
|---|---|---|---|---|---|
| none | 81.9 | 81.1 | 79.0 | 82.7 | 94.0 |
| expert_droid（加法） | 79.9 | 81.1 | 76.9 | 84.1 | 84.7 |
| expert_scratch | 80.7 | 79.3 | 75.5 | 81.9 | 89.3 |
| adaln_droid | 77.4 | 79.3 | 75.6 | 81.9 | 81.3 |
| xattn_droid | 81.2 | **81.8** | **79.4** | 83.5 | 90.7 |
| xattn_droid + reg | 78.7 | 80.5 | 77.2 | 82.7 | 94.0 |
- 3 seed 配对（各 2550 个 episode）：
  - xattn_droid vs none 246:229（p=0.46）；
  - expert_droid vs none 271:271（p=1）；
  - expert_scratch vs none 204:252（**p=0.028**，有害）；
  - adaln_droid vs none 264:311（p=0.055）。
  - expert_droid vs expert_scratch 284:236（p=0.039）：在 GR00T 上，DROID 预训练相对从头训的同一组件有显著好处。但两者都没有超过 none。
- 布丁两阶段任务：
  - 加法 droid 3:17（p=0.0026）、adaln 2:21（p=6.6e-5）显著掉分；
  - xattn_droid 3:8（p=0.23），对加法 droid 11:2（p=0.022），这项代价基本消除。
- **结论**：在 LIBERO-90 17 任务这类短任务上，三种接口都没能让 GR00T 超过 none。
  - xattn_droid（无正则）是唯一完全无代价的接口：dev 和 new 都 ≥ none，布丁掉分消失。后续统一用它作为 SOPT 接入强宿主的方式。
  - AdaLN 最差，不再追加。
  - SET=scratch 对照不再需要（screen 没有收益可以归因），取消。
- 下一步：重点转到长时程（E35 libero_10，seed 0 的摩卡壶任务 +30）以及更长历史（E33）。

## E35 SET=a seed 1 + 2 seed 汇总（2026-10-20，Jubail 18861939，cn264，commit 533f785；GR00T libero_10，k5，EE-only 先验）
| 臂 | s0 | s1 | 2 seed |
|---|---|---|---|
| none | 61.8 | 67.4 | 64.6 |
| expert_scratch | 63.8 | 66.4 | 65.1 |
| expert_droid | 59.2 | 63.6 | 61.4 |
- 配对（各 1000 个 episode）：
  - scratch vs none 137:132（p=0.81）；
  - droid vs none 131:163（p=0.070）；
  - **droid vs scratch 126:163（p=0.034）**。
- **跨宿主一致**：在长时程的 libero_10 上，DROID 先验不如同结构的从头训历史组件。SmolVLA 是 p=0.014（3 seed），GR00T 是 p=0.034（2 seed）。这与 LIBERO-90 17 任务正好相反：那里 GR00T 上 droid 优于 scratch，p=0.039（E31）。
- **摩卡壶任务**（"put both moka pots on the stove"，同一动作做两遍）：
  - none / scratch / droid = 41 / 60 / 68（2 seed）；
  - droid vs none 38:11（p=1e-4，任务数 Bonferroni ×10 后仍 < 0.01）；scratch vs none 35:16（p=0.011）；
  - 这个任务是 s0 事后挑出来的，s1 独立复现：56 / 64 / 74。
- 但 droid 在其余多物体任务上掉分（白杯 + 黄白杯 74→62，cream cheese + butter 58→48，alphabet + tomato 32→26），总分因此被拉低。
- 解读：在需要"记住做到第几步"的重复动作任务上，DROID 先验的历史表示确实有用，而且是全部 10 个任务里最大、最稳定的收益。在其他多物体任务上，它反而比 scratch 更差。下一步看 SET=b（xattn，18869802 运行中）能否保住摩卡壶收益、同时去掉掉分；E33 的长历史先验也会在这里测。

## E33 先验预训练（2026-10-20，Jubail 18874812，cn006，commit 472dc96；DGX 自 10-17 起不可达，改在 Jubail 跑）
- 数据：127 上的 DROID 状态数据（3.4 GB，只含状态）经本机中转传到 `/scratch/ll5582/SOPT_DATA/processed/droid`。各文件大小一致，parquet 和 summary 的 md5 一致。
- 配方与 `e13_ar_flow_f100_s_ee` 完全相同（S，ar_flow，100% DROID，ee_only，40k 步），只改 ctx_len。长上下文变体加 left_pad = ctx − 96。3 个变体各占 1 块 GPU，并行跑，30–36 分钟完成。

| 先验 | 历史 | val pos ADE（cm） | const-vel | selection（越低越好） |
|---|---|---|---|---|
| ctx48 | 3.2 s | 7.80 | 7.50 | 4.846 |
| e13（ctx96） | 6.4 s | 7.37 | 7.30 | 4.672 |
| ctx192 | 12.8 s | 7.25 | 7.43 | 4.663 |
| ctx384 | 25.6 s | **7.13** | 7.52 | **4.660** |
- 历史越长，DROID 上的预测越好，但从 96 帧往上增益很小。注意 const-vel 基线随 ctx 变化，说明各变体的验证窗口并不完全相同（left_pad 改变了锚点的可用范围），所以绝对值只能粗略比较。
- 宿主实验：`e35_groot_std.sbatch` 新增 SET=hist（CTX=…），三个臂为 expert_scratch / expert_droid / xattn_droid，换用长上下文先验，与 E35 ctx96 的同 seed 结果配对。先跑 CTX=384，再跑 CTX=192，都是 seed 0。

## E35 SET=b seed 0：GR00T libero_10 交叉注意力接口（2026-10-20，Jubail 18869802，cn013，2 块 GPU，commit 244df2f）
| 臂（s0） | 全部 | 摩卡壶 | 其余 9 个任务 |
|---|---|---|---|
| none | 61.8 | 26 | 65.8 |
| expert_scratch | 63.8 | 56 | — |
| expert_droid | 59.2 | 62 | 59.6 |
| xattn_scratch | 62.8 | 56 | — |
| xattn_droid | **63.0** | **66** | 62.7 |
- 配对：
  - xattn_droid vs none 88:82（p=0.70）；vs expert_droid 85:66（p=0.14）；vs xattn_scratch 83:82（p=1）。
  - xattn_scratch vs none 68:63（p=0.73）。
- 摩卡壶任务：四个 SOPT 臂都显著高于 none（xattn_droid 23:3，p=1e-4；xattn_scratch 15:0，p=1e-4；expert_droid 21:3，p=3e-4）。
- 其余 9 个任务：
  - expert_droid 对 none 47:78（**p=0.007**，加法接口的代价）；
  - xattn_droid 65:79（p=0.28）；xattn_scratch 53:63（p=0.40）。
  - 交叉注意力缩小了加法接口在其他任务上的代价，但没有完全消除。
- 与 LIBERO-90 17 任务（E31）一致：xattn 是代价最小的接入方式，同时保住了历史带来的收益。
- 下一步：SET=b seed 1 插到队首（2 块 GPU），用于把 xattn 与 SET=a 的 2 seed 配对；hist384（18876338）正在运行，之后是 hist192。

## E35 SET=hist CTX=384 seed 0：25.6 s 历史（2026-10-20，Jubail 18876338，cn262，commit 42f12be；先验 e33_ar_flow_f100_s_ee_ctx384）
| 臂（s0） | ctx96（6.4 s） | ctx384（25.6 s） | 384 vs 96 | 384 vs none（61.8） |
|---|---|---|---|---|
| expert_scratch | 63.8 | 64.6 | 73:69，p=0.80 | 82:68，p=0.29 |
| expert_droid | 59.2 | 61.2 | 78:68，p=0.46 | 71:74，p=0.87 |
| xattn_droid | 63.0 | **68.8** | **83:54，p=0.016** | **90:55，p=0.005** |
- xattn_droid + 25.6 s：
  - 摩卡壶仍是 66（对 none 21:1）；
  - 其余 9 个任务对 ctx96 74:45（p=0.010），对 none 69:54（p=0.21）；
  - 长历史主要改善了 xattn 在其他任务上的代价，例如 "turn on the stove + moka" 74→96、"mug in microwave + close" 38→58、"black bowl drawer + close" 84→96。
- 两个加法臂对历史长度不敏感。
- **注意**：这只是 1 个 seed，而且是 3 个臂 × 2 个历史长度中最好的一格，有挑选的成分。在复现之前，不能作为结论。
- 复现与对照（已排队，优先于 hist192）：
  1. SET=hist CTX=384 SEED=1：3 个臂的第 2 个 seed，与 SET=a / b 的 s1 配对；
  2. SET=histx CTX=384 SEED=0：xattn_scratch + 25.6 s（1 块 GPU），用来区分收益来自"长历史"还是"DROID 预训练 + 长历史"。

## E35 SET=b seed 1 + 2 seed 汇总（2026-10-20，Jubail 18877325，cn267，commit dfdf991；ctx96）
| 臂 | s0 | s1 | 2 seed | 对 none（配对，1000 个 episode） |
|---|---|---|---|---|
| none | 61.8 | 67.4 | 64.6 | — |
| expert_scratch | 63.8 | 66.4 | 65.1 | 137:132，p=0.81 |
| expert_droid | 59.2 | 63.6 | 61.4 | 131:163，p=0.070 |
| xattn_scratch | 62.8 | 59.8 | 61.3 | 120:153，p=0.053 |
| xattn_droid | 63.0 | 61.8 | 62.4 | 155:177，p=0.25 |
- s1 上 xattn_droid 没有复现 s0 的小幅领先（61.8 vs none 67.4）。2 seed 下，ctx96 的四个 SOPT 臂都没有超过 none。只有 expert_scratch 与 none 持平。
- 摩卡壶 vs 其余 9 个任务（2 seed，对 none）：

  | 臂 | 摩卡壶 | 其余 9 个任务 |
  |---|---|---|
  | xattn_droid | 37:6（p<1e-5） | 118:171（p=0.002） |
  | expert_droid | 38:11（p=1e-4） | 93:152（p=2e-4） |
  | xattn_scratch | 21:11（p=0.11） | 99:142（p=0.007） |
  | expert_scratch | 35:16（p=0.011） | 102:116（p=0.38） |
- 结论（ctx96）：DROID 先验在"重复动作、需要记住进度"的任务上收益很大，而且稳定。但在其他多物体任务上，它造成显著代价，并且 xattn 在 2 seed 下也没有消除这一代价（s0 的 p=0.28 是偶然偏好）。
- 这让 hist384 s0 的 68.8 更需要复现：hist384 SEED=1（18885772）和 histx（18887043）已提交，正在排队等资源。

## E35 SET=hist CTX=384 seed 1 + 2 seed 汇总（2026-10-21，Jubail 18885772；GR00T libero_10）
| 臂 | ctx384 s0 / s1 | ctx384 2 seed | ctx96 2 seed | 384 vs 96 | 384 vs none（64.6） |
|---|---|---|---|---|---|
| expert_scratch | 64.6 / 62.0 | 63.3 | 65.1 | 128:146，p=0.30 | 141:154，p=0.49 |
| expert_droid | 61.2 / 64.0 | 62.6 | 61.4 | 147:135，p=0.51 | 139:159，p=0.27 |
| xattn_droid | 68.8 / 66.6 | **67.7** | 62.4 | **176:123，p=0.003** | 161:130，p=0.078 |
- s1 没有复现 s0 对 none 的优势：s1 上 none 是 67.4，xattn_droid ctx384 是 66.6（71:75，p=0.80）。
- 但 ctx384 对 ctx96 的提升在两个 seed 上方向一致（s0 83:54，s1 93:69），合并 p=0.003。提升集中在其余 9 个任务：161:92（p<1e-4）。
  - 也就是说，长历史消除了 xattn 在多物体任务上的代价：其余 9 个任务对 none 从 118:171（ctx96）变为 130:114（ctx384）。
  - 摩卡壶收益随之缩小：72→56（none 41）。
- 加法接口对历史长度不敏感；expert_scratch 的 ctx384 甚至略差。所以"长历史 + 交叉注意力"是一个特定组合的效应，不是"历史越长越好"。
- 现状：xattn_droid ctx384 对 none +3.1，2 seed p=0.078。需要第 3 个 seed。
  - 新增 SET=key：只跑 none / xattn_droid ctx96 / xattn_droid ctx384 三个关键臂，CTX=384 SEED=2 已插到队首。
  - histx（xattn_scratch ctx384，18887043）正在运行。

## E35 SET=histx CTX=384 seed 0：xattn_scratch + 25.6 s 历史（2026-10-21，Jubail 18887043，cn254，1 块 GPU）
| 臂（s0） | ctx96 | ctx384 |
|---|---|---|
| none | 61.8 | — |
| xattn_scratch | 62.8 | 62.4（对 ctx96 71:73，p=0.93；对 none 67:64，p=0.86） |
| xattn_droid | 63.0 | **68.8** |
- **xattn_droid ctx384 vs xattn_scratch ctx384：91:59（p=0.011）**。两者结构相同、历史长度相同，唯一区别是 SOPT 编码器是否用 DROID 预训练。
- 从头训的 xattn 对历史长度完全不敏感。长历史带来的提升只在用 DROID 预训练的编码器上出现，也就是说，它来自预训练学到的长时程运动表示，而不是单纯多看了一段历史。
- 这是目前强宿主上 DROID 预训练价值最直接的证据（1 个 seed）。histx SEED=1 已排队，用于把 droid vs scratch 的对比扩到 2 seed。

## E35 SET=histx CTX=384 seed 1 + 2 seed 汇总（2026-10-21，Jubail 18890761，cn254）
| 臂 | s0 | s1 | 2 seed |
|---|---|---|---|
| none | 61.8 | 67.4 | 64.6 |
| xattn_scratch ctx96 | 62.8 | 59.8 | 61.3 |
| xattn_scratch ctx384 | 62.4 | 59.4 | 60.9 |
| xattn_droid ctx96 | 63.0 | 61.8 | 62.4 |
| xattn_droid ctx384 | 68.8 | 66.6 | **67.7** |
- **DROID 预训练 vs 从头训（同结构、同 25.6 s 历史）：2 seed 191:123，p=1.4e-4**。两个 seed 分别是 91:59（p=0.011）和 100:64（p=0.006），均显著。
  - 收益来自其余 9 个任务：171:109（p=2e-4）。摩卡壶上两者相当：20:14（p=0.39）。
- 从头训的 xattn 对历史长度不敏感（152:156，p=0.86），而且 ctx384 时显著低于 none（121:158，p=0.031）。
- **结论（2 seed）**：在强宿主 GR00T、长时程 libero_10 上，交叉注意力接口加 25.6 s 历史时，DROID 预训练的 SOPT 编码器显著优于同结构的从头训编码器，差 6.8 个点。只有预训练表示能把长历史转化成收益。
  - 对 none 仍是 +3.1（p=0.078）。决定性的第 3 个 seed 仍在等资源。
- 调度：3 块 GPU 的 SET=key（18889504）排了 10 小时以上，预计明天才开始；1 块 GPU 的作业通常几小时内就开始。新增 SET=one（单臂，1 块 GPU）。取消 18889504（尚未开始），改为三个 1 块 GPU 的 s2 作业，顺序为：
  1. xattn_droid ctx384；
  2. none；
  3. xattn_droid ctx96。
  
  hist192 也改为单臂 xattn_droid ctx192 s0。

## E35 seed 2：none 与 xattn_droid ctx384（2026-10-21，Jubail 18896070 / 18896058，SET=one，各 1 块 GPU）
| | s0 | s1 | s2 | 3 seed |
|---|---|---|---|---|
| none | 61.8 | 67.4 | 62.8 | 64.0 |
| xattn_droid ctx384 | 68.8 | 66.6 | 67.2 | **67.5** |
| 配对 | 90:55 | 71:75 | 90:68 | **251:198，p=0.014** |
- 3 seed 合并（1500 个配对 episode）：xattn_droid ctx384 对 none +3.5，p=0.014。
  - 摩卡壶 40→62（55:22，p=2e-4）；其余 9 个任务 196:176（p=0.32），即不再有代价。
  - 分任务：微波炉 + 关门 40.7→51.3，杯子 + 布丁 50.7→58.7；书本 96.0→89.3 下降。
- **统计上的保留意见**：ctx384 + xattn 这一格是看了 s0 之后选出来的（当时是 3 个臂 × 2 个历史长度中最好的）。只算选定之后的验证 seed（s1 + s2）是 161:143，p=0.33，方向一致但不显著。所以对 none 的 p=0.014 含有挑选偏差，不能作为预先注册的结果报告。
- 更稳健的结论是预先设计的配对对照：
  - DROID vs scratch（同为 xattn ctx384，2 seed）191:123，p=1.4e-4，两个 seed 各自显著；
  - ctx384 vs ctx96（同为 xattn_droid，2 seed）176:123，p=0.003。
  - xattn_droid ctx96 s2（18899088）补齐后，这两个对比可以扩到 3 seed。
- 若要对 none 给出干净的证据，需要预先注册的新 seed（s3、s4），只比较 none vs xattn_droid ctx384。已加入队列，排在 ctx192 之后。

## E35 xattn_droid ctx96 seed 2 → ctx384 vs ctx96 扩到 3 seed（2026-10-22，Jubail 18899088，SET=one）
| xattn_droid | s0 | s1 | s2 | 3 seed |
|---|---|---|---|---|
| ctx96（6.4 s） | 63.0 | 61.8 | 60.2 | 61.7 |
| ctx384（25.6 s） | 68.8 | 66.6 | 67.2 | 67.5 |
| 配对 | 83:54 | 93:69 | 78:43 | **254:166，p=2.0e-5** |
- 三个 seed 各自方向一致，s2 单独 p=0.002。这是预先设计的对照（同接口、同编码器，只改历史长度），s1 和 s2 不受挑选偏差影响。
- 分解：
  - 其余 9 个任务 231:130（p<1e-6）；摩卡壶 23:36（p=0.12），即 ctx96 在摩卡壶上更强（71 vs 62）。
  - ctx96 对 none：摩卡壶 59:13（p<1e-6），其余 9 个任务 162:243（p=7e-5）。
- 机制图景（3 seed）：
  - 6.4 s 的 DROID 历史表示给出强烈的"进度"信号，摩卡壶 +31，但在其他多物体任务上误导宿主，代价显著。
  - 25.6 s 的表示保留了大部分进度收益（+22），同时去掉了这一代价。
  - 从头训的同结构编码器两种长度都没有收益（histx，2 seed）。

## E35 xattn_droid ctx192 seed 0：12.8 s 历史（2026-10-22，Jubail 18899136，SET=one）
| xattn_droid（s0） | 全部 | 摩卡壶 | 其余 9 个任务 |
|---|---|---|---|
| none | 61.8 | 26 | 65.8 |
| ctx96（6.4 s） | 63.0 | 66 | 62.7 |
| ctx192（12.8 s） | **70.0** | **78** | 69.1 |
| ctx384（25.6 s） | 68.8 | 66 | 69.1 |
- 配对：ctx192 vs ctx96 90:55（p=0.005）；vs ctx384 59:53（p=0.64）；vs none 83:42（p=3e-4）。
- 剂量曲线（1 个 seed）：从 6.4 s 到 12.8 s，提升明显；12.8 s 与 25.6 s 相当。ctx192 在这个 seed 上同时保住了摩卡壶收益（78）和其余任务（69.1）。
- 预先注册不变：s3 / s4 只比较 none vs xattn_droid ctx384，这是看到 ctx192 之前就定好的，不因此改成 ctx192。
- 为描述完整的剂量曲线，追加以下 seed（排在 s3 / s4 之后）：
  - ctx192 s1 / s2；
  - ctx48 s0（3.2 s，短端）。

## E35 预注册 seed 3：none vs xattn_droid ctx384（2026-10-22，Jubail 18903043 / 18903172，SET=one）
- s3：none 62.0，xattn_droid ctx384 64.6，配对 78:65（p=0.32）。摩卡壶 36→76（23:3，p=1e-4）；其余 9 个任务 55:62（p=0.58）。
- 4 seed 合计（含挑选用的 s0）：63.5 vs 66.8，329:263（p=0.0075）。
- 预注册检验只用 s3 + s4，要等 s4（18908057 / 18908292 已提交）出来再下结论。s3 的方向与 s1 / s2 / s0 一致，均为正。

## E35 预注册检验结果：none vs xattn_droid ctx384，s3 + s4（2026-10-22，Jubail 18908057 / 18908292）
| seed | none | xattn_droid ctx384 | 配对 |
|---|---|---|---|
| s3 | 62.0 | 64.6 | 78:65 |
| s4 | 61.6 | 64.8 | 81:65 |
| **s3 + s4（预注册）** | 61.8 | 64.7 | **159:130，p=0.099** |
| s1–s4（排除挑选用的 s0） | 63.5 | 65.8 | 320:273，p=0.059 |
| s0–s4（全部，含挑选偏差） | 63.1 | 66.4 | 410:328，p=0.0028 |
- **预注册检验没有达到 p<0.05**：+2.9 个点，p=0.099。5 个 seed 中有 4 个方向为正（s1 为 71:75）。
- 在预注册 seed 上，摩卡壶收益稳健：42→68，37:11，p=2e-4。这个任务在 s3 / s4 之前就已确定是重点任务，不受挑选影响。其余 9 个任务 122:119（p=0.90），没有代价。
  - s3 / s4 上较明显的掉分任务："turn on stove + moka" 92→78，"cream cheese + butter" 64→53。
- 功效估计：按每个 episode 约 29% 的不一致率和 +10% 的净差比例，要在 α=0.05、功效 0.8 下检出约 +3 个点，需要约 780 个不一致对，约 2700 个 episode，即 5–6 个预注册 seed。现在只有 2 个。
- **对论文表述的含义（GR00T，libero_10）**：
  - 可以作为确认性结论的：
    1. 同一接口和长历史下，DROID 预训练显著优于从头训（2 seed，p=1.4e-4）；
    2. 同一预训练编码器下，长历史显著优于 6.4 s（3 seed，p=2e-5）；
    3. 在需要记住进度的重复动作任务上，收益大且稳健（预注册 seed 上 p=2e-4）。
  - 不能作为确认性结论的：总成功率对 none 的提升（+2.9，p=0.099）。
- 是否追加 s5–s8（预先声明以 s5–s8 合并作为新的独立确认检验，约 8 个单 GPU 作业、40 GPU 小时）需要权衡，先完成已排队的剂量曲线（ctx192 s1 / s2、ctx48 s0）。

## E35 历史长度剂量曲线：xattn_droid ctx192 补齐 3 seed（2026-10-23，Jubail 18909907 / 18910213，SET=one）
（更正：本条首次提交 52c33af 时，"摩卡壶" 一列误把 "put both moka pots" 和 "turn on the stove and put the moka pot on it" 两个任务合并统计。下表已改为只统计 "put both moka pots on the stove"，与此前各条口径一致。）
| xattn_droid，s0–s2 合计（1500 episode / 臂） | 全部 | 摩卡壶（both moka pots） | 其余 9 个任务 |
|---|---|---|---|
| none | 64.0 | 40.0 | 66.7 |
| ctx96（6.4 s） | 61.7 | 70.7 | 60.7 |
| ctx192（12.8 s） | 64.7 | **75.3** | 63.6 |
| ctx384（25.6 s） | **67.5** | 62.0 | **68.1** |
- 分 seed（ctx192）：s0 70.0、s1 65.8、s2 58.4。
- 配对（3 seed）：
  - ctx192 vs ctx96：全部 239:193（p=0.030），其余 9 个任务 212:173（p=0.053）。
  - ctx384 vs ctx192：全部 228:186（p=0.044）。
    - 其余 9 个任务 212:150（p=0.0013），ctx384 更好；
    - 摩卡壶 16:36（p=0.008），ctx192 更好。
  - 对 none：
    - 摩卡壶三档都显著高于 none（ctx96 59:13，ctx192 60:7，ctx384 55:22，各 p≤2e-4）；
    - 其余 9 个任务：ctx96 162:243（p=1e-4），ctx192 165:207（p=0.033），ctx384 196:176（p=0.32）。
- **结论：**
  - 总成功率随历史变长单调上升（61.7 → 64.7 → 67.5），相邻两档都显著。
  - 这主要来自其余 9 个任务上代价的消失：60.7 → 63.6 → 68.1。只有 ctx384 不再低于 none。
  - 摩卡壶收益在所有长度都存在；12.8 s 最高，25.6 s 回落但仍高于 none。存在折中：最长历史对常规任务最好，中等历史对进度记忆任务最好。
- 注意：ctx384 vs none 的 3 seed 结果（p=0.014）包含挑选用的 s0，不作确认性结论；确认性检验仍以预注册 s3 + s4（p=0.099）为准。
- 下一步：补 ctx48（3.2 s）s1 / s2，与已完成的 s0（18912745）一起完成短端，构成 4 档 × 3 seed 的完整曲线。

## E35 ctx48 seed 0：3.2 s 历史（2026-10-23，Jubail 18912745，SET=one）
- s0：全部 63.2（ctx96 63.0，ctx192 70.0，ctx384 68.8，none 61.8）；摩卡壶 72（none 26，27:4）；其余 9 个任务 62.2（none 65.8）。
- 单 seed 上 ctx48 与 ctx96 无差别（72:73）；ctx192 / ctx384 都显著优于 ctx48（87:53 p=0.005；79:51 p=0.018），差别来自其余 9 个任务。
- 短历史（≤ 6.4 s）已足以给出摩卡壶收益；要消除常规任务代价需要 ≥ 12.8 s。s1（18914505）在跑，s2 已排队。

## E35 历史长度曲线完整版：4 档 × 3 seed（2026-10-23，ctx48 s1 / s2：Jubail 18914505 / 18915193）
| xattn_droid，s0–s2 合计（1500 episode / 臂） | 全部 | 摩卡壶（both moka pots） | 其余 9 个任务 |
|---|---|---|---|
| none | 64.0 | 40.0 | 66.7 |
| ctx48（3.2 s） | **67.0** | 68.7 | 66.8 |
| ctx96（6.4 s，e13 先验） | 61.7 | 70.7 | 60.7 |
| ctx192（12.8 s） | 64.7 | 75.3 | 63.6 |
| ctx384（25.6 s） | **67.5** | 62.0 | **68.1** |
- 分 seed（ctx48）：s0 63.2、s1 67.8、s2 70.0。
- 配对（3 seed）：
  - ctx48 vs none：228:183（p=0.030）。摩卡壶 60:17（p<1e-4），其余 9 个任务 168:166（p=0.96），没有代价。
  - ctx48 vs ctx96：250:170（p=1e-4），差别全部在其余 9 个任务（230:147）。
  - ctx384 vs ctx48：202:194（p=0.73），两者无差别。
- **更正上两条的结论："总成功率随历史变长单调上升" 不成立。** 曲线不单调：只有 ctx96 一档明显偏低，并带有常规任务代价（其余 9 个任务 60.7）；ctx48 和 ctx384 都没有这个代价。
- **主要混杂：先验的训练批次。** ctx96 用的是 e13 先验（git 402e3b6，在 127 上训练）。ctx48 / 192 / 384 都来自同一个 E33 作业（git 472dc96，Jubail）。
  - 两批的数据（episodes.parquet md5 相同）和配方相同，只有 ctx_len / left_pad 不同；但每个长度都只训练了一个先验实例，先验之间的训练波动完全没有被抽样。
  - 因此 "ctx96 有代价" 可能是 e13 这一个先验实例的问题，而不是 6.4 s 本身的问题。
  - 之前 "长历史 vs 6.4 s，p=2e-5" 也受这个混杂影响：它比较的是两个不同的先验实例。
- **对照实验（已排队）：** 用 E33 同一脚本 / 环境重训一个 ctx96 先验（e33_ar_flow_f100_s_ee_ctx96，`ONE=96`，1 GPU），再跑 xattn_droid × 3 seed（SET=one CTX=96，输出名带 _ctx96）。判读：
  - 若结果接近 ctx48 / ctx384（约 67）：说明 ctx96 的低分是 e13 先验实例的问题。此时历史长度在 3.2–25.6 s 内影响不大，先验实例之间的波动是需要报告的方差来源。
  - 若仍约 62：说明 6.4 s 确有问题，需要另找解释。
- 不受此影响、仍成立的结论：
  - 摩卡壶收益在所有先验 / 长度上都显著（各 p≤2e-4）；
  - 预注册 s3 + s4 的结果（p=0.099）。
- 受影响、需要重新验证的结论："DROID vs scratch，p=1.4e-4"。它在 ctx384 上比较，只用了一个 DROID 先验实例；scratch 不依赖先验实例。
- ctx96 复制先验已训练（2026-10-23，Jubail 18917745，22 min，git b41c4cc）。DROID val selection 4.711，e13 为 4.672；E33 其余各档：ctx48 4.846、ctx192 4.663、ctx384 4.660。已排入 xattn_droid × s0–s2（输出 e35_groot_libero_10_xattn_droid_k5_ctx96_s*）。

## DGX 任务迁到 Jubail（2026-10-23，按用户要求）
- DGX 自 10-17 起不可达。本地的 DGX 排队器和等待 SmolVLA s2 的监视已停止，DGX 上不再提交任何作业。
- DGX 上的 e23 TrunkBC / SmolVLA s0–s1 rollouts 目前取不到，本地缓存已被 /tmp 清理。因此迁移时所有对照臂都在 Jubail 上重跑，保证同集群、同代码配对。
- 新脚本 `scripts/slurm/e25_trunk.sbatch`（1 GPU）：VARIANT = ref | scratch | base | speed | ctx48 | ctx192 | ctx384。
  - 先验为全状态 e2 配方，输出 `e25j_<V>`；TrunkBC 17 任务 × k5 × 3 seed × 50 init，输出 `e25j_trunk_<V>_k5_t17_s*`；最后跑 E24 探针。
  - ref 使用现有的 `outputs/e2_ar_flow_f100_s`。base 为 seed=1 的同配方复制，用于量化先验实例方差。
  - IDM `e4b_idm` 从 127 拷到 Jubail outputs/，md5 一致：36daaddc…。
- SmolVLA 17 任务 s2：`e23_smolvla.sbatch` 新增 ARM=（单臂、单 GPU）模式，三臂各 1 个作业。
  - s0 / s1 在 DGX 上已完成，逐 episode 数据暂时取不到；3 seed 合并时，s0 + s1 用已记录的不一致对计数（droid vs none 241:150，droid vs scratch 260:139）加上 s2 的计数。McNemar 的合并统计量就是不一致对之和，因此这样合并是等价的。
- 排队顺序（Jubail 最多 2 个并发）：
  1. E35 ctx96 复制先验 s2
  2. ref
  3. SmolVLA s2 none
  4. ctx192
  5. SmolVLA s2 droid
  6. ctx384
  7. SmolVLA s2 scratch
  8. ctx48
  9. base
  10. speed
  11. scratch

## E36 / E37 设计与实现（2026-10-23，按用户要求，判据在看到任何结果前写定；计划见 `docs/SOPT_NEXT.md`）

### 共用基础设施（9301499 及之后）
- `src/sopt/sim/perturb.py`：评测时的场景扰动，维度参照 LIBERO-Plus。
  - 视觉类：camera（agentview 绕它在桌面上的注视点做水平旋转，正负号随机）、light（场景灯变暗并偏色）、texture（只改桌面 / 地面 / 墙面材质的颜色，不碰任务物体和机器人）、noise（两路相机的像素高斯噪声）。
  - 本体感觉对照：robot（初始关节加均匀偏移）。
  - 每种 3 档，强度事先固定。所有随机数只由 (kind, level, task, init) 决定，noise 另加 step，所以被比较的各臂看到的是完全相同的扰动 episode。
  - 改动在 set_init_state 之后、settle 之前施加，物体在扰动后的场景里落稳；hard reset 会重建场景。
- `vla_eval.py`：
  - `--perturb kind:level`；
  - `--log-npz`：逐步记录原始关节 + 手指、实际执行的动作、每次预测的完整动作块，以及 max_steps；
  - 不加这两个参数时，输出与原来逐字段相同，只多 perturb / ctx 两个汇总字段。
- 本地单元检查（通过）：
  - 相机旋转保持注视点、高度和距离，R' = Rz·R；四元数乘法与矩阵乘法一致；
  - 随机种子稳定；noise 只在 noise 档生效。
- 集群预检 `scripts/slurm/e37_probe.sbatch`（已排在队首）。17 个 LIBERO-90 任务加 libero_goal / libero_10 场景，逐项检查：
  1. perturb=None 时与原环境的图像和关节完全相同；
  2. 每一档都可复现；
  3. 普通 reset 后相机、灯光、材质恢复为 XML 原值；
  4. 每个场景实际匹配到的材质名、相机父刚体、灯数；
  5. 输出图像，人工查看；
  6. 另外：DP none s0 的 2 个 init 重评测与原 rollouts 一致，npz 偏移与步数一致；DP 和 GR00T 各做一次扰动冒烟。

### E36a 只用本体感觉的运行时失败预测（`scripts/e36_failure.py`，`scripts/slurm/e36_log.sbatch`）
- 数据：重新做干净评测并加 `--log-npz`，s0、LIBERO-90 17 任务 × 50 init：
  - GR00T none / xattn_droid、DP none / expert_droid、π0.5 none、X-VLA none；
  - 长时程：libero_10 s3 的 GR00T none / xattn_droid ctx384。
  - 先检查与原 rollouts 的逐 episode 一致率。
- 分数，全部严格因果：t 时刻只用 ≤ t 的数据，已用"改写 t 之后的轨迹，t 之前的分数不变"测试过：
  - sopt_ee：冻结的 DROID EE-only 先验在 t−1 s 发出的预测，对 (t−1 s, t] 实际 EE 位姿 + 夹爪的能量分数；K=16，每个宿主都用同一个先验；
  - sopt_full：同上，全状态先验，仅 LIBERO-90；
  - rand_ee：同结构、同归一化，随机初始化；
  - constvel：恒速外推；
  - still / gripcyc / blocked：E26 的定义，在线版本；
  - chunkcons：宿主自己相邻动作块重叠部分的不一致度，Sentinel 式，宿主专用。
  - 聚合：截至目前的最大值，或当前值（_last）。基线取两者中较好的一个，对 SOPT 更保守。
  - 检查时刻从第 2 s 开始，每 0.5 s 一次。
- 划分：init < 25 只用于校准，只用成功轨迹，与 FAIL-Detect 一致；init ≥ 25 用于测试。
- 指标：
  - 在步数上限 25 / 50 / 75 % 时刻，对仍在运行的测试 episode 计算 AUROC；
  - 随时间变化的保形阈值（α=0.1）下的 TPR / FPR / 中位检测时间；
  - 跨 run 的阈值迁移。
- **判据**：在 6 个干净 LIBERO-90 run 上，50% 时刻 sopt_ee 的平均 AUROC 须同时满足：
  - ≥ 0.75；
  - 比最好的非 SOPT 分数高 ≥ 0.05；
  - 合并配对自助法的 95% CI > 0；
  - 跨 run 迁移时 TPR 平均下降 ≤ 0.05，且 FPR ≤ 0.15。
- 合成数据端到端测试通过：AUROC 与暴力枚举一致；因果性检查通过；恒速轨迹的恒速误差为 0。

### E36b（有条件执行）
- 只有 E36a 通过才设计和实施检测后的恢复，阈值沿用 E36a 的校准结果。

### E37 视觉分布偏移下的 SOPT 收益（`scripts/slurm/e37_robust.sbatch`，`scripts/e37_analyze.py`）
- 只做评测，比较已报告过的干净对照：
  - DP none vs expert_droid（17 任务）；
  - SmolVLA libero_goal none vs expert_droid；
  - GR00T none（E22）vs xattn_droid（E31）。
  - seeds 0–1。
- **档位标定**：MODE=pilot，用三个宿主 seed 2 的 none checkpoint。这些 checkpoint 不进入检验，也不看任何 SOPT 臂；每种 3 档 × 10 init。
  - 规则事先写定：每种扰动取"三宿主平均的扰动 / 干净成功率 ≤ 0.85"的最低一档，都不满足则取第 3 档。
- **主检验**：每对 (task, init) 的交互项 = (SOPT_扰动 − none_扰动) − (SOPT_干净 − none_干净)。
  - 在 4 种视觉扰动 × 3 宿主 × 2 seed 上合并取平均，按 宿主 × seed × 扰动 分层做自助法得 95% CI，CI > 0 才认为"视觉偏移下 SOPT 收益更大"。
  - robot 单独报告，不并入主检验。
  - 次要：分宿主、分扰动的交互项；各条件下 SOPT vs none 的配对 McNemar；两臂的成功率保持比例。
- 合成数据测试：真值为视觉 +0.15、robot −0.05，分析恢复出 +0.162 [0.145, 0.179] 和 −0.049。
- 执行顺序：probe 通过 → e36_log 和 e37 pilot → 我检查档位 → e37 main（DP / SmolVLA 每 seed 一个作业；GR00T 每臂一个作业）。

## E35 ctx96 复制先验 seed 0（2026-10-23，Jubail 18918290）
- xattn_droid + e33 ctx96 复制先验，s0：
  - 全部 63.2，与 e13 ctx96 的 63.0 相同，配对 66:67；与 ctx48 的 63.2 相同，70:70。
  - 摩卡壶 68，其余 9 个任务 62.7。
- s0 上短历史（3.2 / 6.4 s）三者一致，都低于 ctx192（70.0，85:51，p=0.005）和 ctx384（68.8，p=0.02）。
- 之前 ctx96 的掉分主要出现在 s1 / s2（例如 s2：ctx48 70.0 vs ctx96 60.2），要等复制先验的 s1 / s2 才能判断。

## E37 probe v1 未通过 → 诊断与修复（2026-10-23/24，Jubail 18920763 / 18924405 / 18925205 / 18925501）
- **probe v1（18920763）**：17 个 LIBERO-90 场景加 libero_goal 和 libero_10 的部分场景中，有 10 个"同 init 两次 reset 完全相同"检查失败。
  - 失败的全是厨房类场景，带柜子、炉灶或酒架。
  - 不加扰动时同样不一致，所以不是扰动代码造成的。
- **诊断 1（e37_diag.py）**：
  - 同一环境第二次 reset 时，固定件（柜子、酒架）的位置会变 0.4–1.6 cm，但全局 RNG 状态完全相同。原因是 LIBERO 在每次 hard reset 时都把固定件位置重新采样后写进 `model.body_pos`，而 `set_init_state` 只恢复 qpos，并且采样器本身带状态。
  - 两个全新环境的第一次 reset 完全一致。
  - 炉灶场景的手腕相机有 2–3 个像素的渲染抖动（像素占比 ≤6e-5）。
- **诊断 2（e37_diag2.py）**：每个 worker 按 init 0..49 顺序 reset。
  - 固定件抖动 ≤1.9 cm，不累积。
  - reset 之间有没有执行 60 步随机动作，固定件序列完全相同（差 0）。
  - 所以抖动只由 reset 序列决定，与策略行为无关：各臂之间、干净与扰动评测之间同一 (task, init) 的场景一致，配对比较成立。
  - 这也解释了 DP none s0 重跑 2 init 时成功率 100% 一致、但 5/34 个 episode 步数略有差异。另有 1 例差异出现在客厅场景，可能来自 GPU 推理的非确定性。
- **诊断 3（e37_diag3.py）**：厨房场景的纹理 1 档确实改了模型里的 `mat_rgba`（0.65–0.98），但图像逐像素不变；2 / 3 档的像素比只在单个通道偏离 1。
  - 原因：MuJoCo 先把"光照 × rgba"截断到 [0, 1] 再乘以纹理，厨房光照强、处于饱和区，所以温和的 rgba 混合不可见。
  - 客厅场景较暗，不受影响。
- **修复（ce5ff6d）**：
  - 纹理扰动改为直接把场地纹理图像（`tex_data`）向随机颜色混合，并上传到渲染上下文。
  - 与物体或机器人共用的纹理一律跳过并记入日志；无纹理的场地材质沿用 rgba 混合。
  - robosuite 在每次 hard reset 时重建 sim 和渲染上下文，所以修改不会残留。
  - 在本地 MuJoCo 3.8.1 玩具场景（饱和光照）上验证：1 档改变 43% 的像素，共享纹理未动。
  - 纹理颜色的随机抽取方式改变了；当时还没有任何扰动结果，不影响预注册。
- **probe v2**：
  - 确定性：两个全新环境的第一次 reset，q 必须完全一致，图像允许 ≤0.1% 的渲染抖动。
  - 每种视觉扰动在每一档都必须改变 ≥5% 的 agent 像素。
  - 恢复检查加入 `tex_data`。
  - 新增：各扰动类型下，init 0–4 的固定件序列必须与不加扰动时完全一致。
  - 已排队；通过后才恢复队列。

## E35 ctx96 复制先验 seed 1（2026-10-24，Jubail 18918295）
- 复制先验 = e33 ctx96，在 Jubail 上用与 ctx48 / 192 / 384 相同的代码重训。
- xattn_droid + 复制先验，s1：全部 62.2，摩卡壶 68，其余 9 个任务 61.6。
  - 对比 e13 ctx96 s1 的 61.8：配对 67:69，p=0.93，几乎完全一致。
  - 低于 ctx48 的 67.8（55:83，p=0.021），也低于 none 的 67.4（61:87，p=0.040）。
- s0 + s1 合并：

| 臂 | 全部 | 摩卡壶 | 其余 9 个 |
|---|---|---|---|
| none | 64.6 | 41.0 | 67.2 |
| ctx48 | 65.5 | 72.0 | 64.8 |
| ctx96（e13） | 62.4 | 72.0 | 61.3 |
| ctx96r | 62.7 | 68.0 | 62.1 |
| ctx192 | 67.9 | 80.0 | 66.6 |
| ctx384 | 67.7 | 56.0 | 69.0 |

  - ctx96 vs ctx96r：133:136，p=0.90。
  - ctx384 vs ctx96r：169:119，p=0.004。
- **结论**：6.4 s 的掉分在独立重训的先验实例和平台上都复现，不是先验实例 / 训练平台造成的混淆。
  - 历史长度效应确实非单调：3.2 s ≈ none，6.4 s 低，12.8 / 25.6 s 高。
  - 作为"更长历史更好"的证据，只能说 12.8 / 25.6 s 优于 6.4 s，不能说单调。
  - 复制先验的 s2 在挂起队列里，跑完后做 3 seed 汇总。

## E37 probe v2 通过（2026-10-24，Jubail 18925528，commit 73714a5）
- ALL CHECKS PASSED：17 个 LIBERO-90 场景 + 2 个 libero_goal + 3 个 libero_10 场景。
  - 不加扰动时与原环境一致：q 完全相同，图像差 ≤9e-5，只有炉灶场景手腕相机的闪烁。
  - 两个全新环境之间完全确定；hard reset 能完整恢复；每种扰动、每一档都改变 ≥5% 的像素。
  - 在所有带固定件的场景里，5 种扰动下 init 0–4 的固定件序列都与不加扰动时完全一致。
- 纹理只作用于场地纹理，没有出现共享而被跳过的纹理，也没有纯 rgba 的材质。
- 人工看图：厨房纹理 1 档现在明显变色，物体、柜子和机器人不变。libero_goal 的炉灶面板反光，会随墙色变化，是倒影而不是纹理被修改。
- 各场景强度（agent 视角平均 |像素差|，1 / 2 / 3 档）：
  - camera 15–38 / 20–53 / 25–63。
  - light：客厅 / 书房 11–20 / 15–17 / 18–23；厨房 2–3 / 11–16 / 24–40（厨房光照饱和，变暗不明显）。
  - texture：客厅 / 厨房 10–27 / 15–51 / 17–67；书房 1.5 / 5.9 / 7.2（场地可见面积小）。
  - noise 6.3 / 12.5 / 24.0。
  - 场景之间强度不一致，由 pilot 的跨任务平均来标定档位；各臂看到的扰动完全相同，交互项分析不受影响。
- 重跑 DP none s0（2 init）：与原始结果成功率 100% 一致，步数相同 0.824（来自 LIBERO / GPU 的固有抖动）；npz 检查全部为 True。
- 下一步：E37 pilot（seed-2 none 模型标定档位）与 E36 数据采集一起排在队首。

## E37 pilot 档位标定（2026-10-24，Jubail 18927630）
- 三个宿主 seed-2 的 none 模型，10 init。干净成功率（同样 10 个 init）：DP 74.1，SmolVLA-goal 79.0，GR00T 79.4。
- 相对成功率（扰动 / 干净），依次为 DP / SmolVLA-goal / GR00T → 三宿主平均：

| 扰动 | 1 档 | 2 档 | 3 档 |
|---|---|---|---|
| camera | .73/.92/.99 → .88 | .59/.82/.56 → **.66** | .60/.68/.37 → .55 |
| light | .94/.90/.99 → .94 | .94/.96/.96 → .95 | .87/.96/1.01 → **.95** |
| texture | .84/.97/1.02 → .95 | .60/.91/.93 → **.81** | .44/.80/.94 → .73 |
| noise | .90/1.10/1.03 → 1.01 | .52/1.01/.87 → **.80** | .10/.80/.63 → .51 |
| robot | .94/.94/.93 → .94 | .83/.76/.83 → **.80** | .51/.48/.64 → .54 |

- 按预注册规则（最低满足平均 ≤ 0.85 的档位，否则取 3 档），得到 **camera 2、light 3、texture 2、noise 2、robot 2**（`outputs/e37_levels.json`）。
  - 光照即使到 3 档平均也只到 0.95，没达到标准，按规则取 3 档。这一项的偏移很弱，检验功效低，结果照实报告，不追加档位。
- 宿主之间的鲁棒性差异明显：DP 对纹理和噪声最敏感（noise 3 档只剩 0.10），GR00T 对相机最敏感。
- 已排队 8 个主实验作业：
  - DP、SmolVLA 每个 seed 一个作业，两臂都跑；
  - GR00T 每臂每 seed 一个作业。
  - 排在 E25j 和 SmolVLA s2 之前。

## E35 ctx96 复制先验 seed 2 + 3 seed 汇总（2026-10-24，Jubail 18926672）
- s2：ctx96r 65.0，e13 ctx96 60.2（48:72，p=0.035）；ctx48 70.0；ctx384 67.2；ctx192 58.4；none 62.8。
  - 同一配置换一个先验实例，单个 seed 上就能差 4.8 个点，而且显著。先验实例之间的方差不可忽略。
- **3 seed 汇总（libero_10 GR00T xattn_droid，3 × 500 episode）**：

| 臂 | 全部 | 摩卡壶 | 其余 9 个 |
|---|---|---|---|
| none | 64.0 | 40.0 | 66.7 |
| ctx48 | 67.0 | 68.7 | 66.8 |
| ctx96（e13） | 61.7 | 70.7 | 60.7 |
| ctx96r | 63.5 | 68.7 | 62.9 |
| ctx192 | 64.7 | 75.3 | 63.6 |
| ctx384 | 67.5 | 62.0 | 68.1 |

  - ctx96r vs ctx96：181:208，p=0.19。
  - ctx96r vs ctx48：171:224，p=0.009。
  - ctx384 vs ctx96r：237:176，p=0.003。
  - ctx384 vs ctx48：202:194，p=0.73。
- **结论**：
  1. 换了先验实例和训练平台之后，ctx96 仍然是曲线的低点（63.5 / 61.7）。所以"ctx96 低"不是 e13 先验实例的偶然结果，但幅度会随实例浮动。
  2. 历史长度没有单调的剂量效应：3.2 s ≈ 25.6 s ≈ 67，6.4 s 最低，12.8 s 居中。**不能**声称"更长历史更好"。
  3. 稳健的结论只有一个：所有带先验的臂在摩卡壶上都明显高于 none（62–75 vs 40），这就是预注册的进度记忆收益；与历史长度无关。
  4. 跨先验实例的方差较大。论文中所有 SOPT 对比都应注明"单个先验实例"，或者在主要结果上补 2 个先验实例。
- 已排队的 E25j（TrunkBC × 先验变体）会用另一条路线检验同一个问题。

## E36a 运行时失败预测：预注册检验未通过（2026-10-24，Jubail 18929085，commit 3c0ab28）
- **复现**：8 个 run 都用 `--log-npz` 重跑干净评测，与原 `rollouts_n50.json` 对比：
  - 成功率差在 ±1 以内（例如 GR00T none 81.3 / 81.9，DP none 73.5 / 73.6，pi0.5 58.5 / 57.6）；
  - 逐 episode 一致率 0.936–0.993，来自 LIBERO / GPU 的固有非确定性，见 E37 诊断。
- **主检验（预注册，6 个 LIBERO-90 run 的平均 AUROC@50%）**：
  - sopt_ee **0.456**；最好的基线是 still_last（末端最近 2 s 的移动量取负），**0.755**。
  - sopt_ee − best 为 −0.30，95% CI [−0.37, −0.22]。
  - sopt_ee − rand_ee 为 +0.107，CI [0.05, 0.16]：DROID 预训练比随机初始化多带一点信息，但整体仍在随机水平以下。
  - 迁移 FPR 0.23，超过 0.15。
  - **→ 未通过（pass=false）。按预注册，E36b（恢复）不实施。**
- 分 run、分时间点：
  - 25% 时 sopt_ee 为 0.48–0.68，和 rand_ee（0.44–0.66）差不多；50% 时降到 0.41–0.51。
  - 宿主自身的动作块一致性（chunkcons）在 25% 时 0.63–0.82，是早期最强的信号。
  - 75% 时仍在运行的成功 episode 很少，各分数在 0.1–0.99 之间剧烈波动，不作解读。
  - sopt_full（全状态先验，仅 LIBERO-90）也只有 0.37–0.59。
- **机制核查（`scripts/e36_sanity.py`，本地 CPU，DP none s0，60 个 episode / 277–318 个检查点）**：
  - 1 s 预测误差（尺度归一化单位）：sopt_ee 6.31，rand 15.52，恒速 3.93，保持不动 5.39。
  - 按步长的位置误差（cm），先验 / 恒速 / 保持不动：
    - 0.07 s：0.44 / 0.13 / 0.53；
    - 0.2 s：1.89 / 0.55 / 1.56；
    - 0.53 s：4.82 / 2.09 / 3.96；
    - 1 s：7.26 / 5.48 / 6.87；
    - 2.1 s：14.10 / 16.01 / 11.51。
  - 0.2 s 的 1.89 cm 与 E4b 诊断的 1.93 cm 基本一致；训练好的先验远好于随机先验，2.1 s 时优于恒速。这说明输入的构造和对齐是对的，**不是代码错误**。
- **结论**：
  1. 零样本的 DROID 先验在 ≤1 s 尺度上的预测精度低于恒速外推。这就是 E4b 已发现的问题：整个 horizon 共用一个归一化尺度，近端精度不足。
  2. 闭环策略 rollout 里有大量停顿和慢速段，2.1 s 时连"保持不动"都优于先验。所以"预测意外度"主要反映的是先验自身的噪声和速度失配，而不是失败。
  3. LIBERO 中的失败以停滞、犹豫为主，简单的静止启发式就能抓到（AUROC 0.75）。
  4. 论文中 **不** 把失败预测作为 SOPT 的收益；作为负结果和局限写入，并注明近端精度这一根因。
- 如果以后要再做（**探索性，非预注册**）：先验改为预测相对恒速的残差，或者按步长归一化（E4b 的教训 1）之后再测；或者用 LIBERO 后训练过的先验（E4a ft）。这两项都要重新预注册。

## E37 main 进度：DP seed 0（2026-10-24，Jubail 18929253，commit bcce74c，5.5 h）
- 10 个评测全部 exit 0，json/npz 齐全；17 任务 × 50 init。原始干净 rollout：none 73.6 / expert_droid 76.1（+2.5）。
- 扰动下成功率 none → droid（差值）：camera:2 32.1 → 45.4（+13.3）；light:3 69.2 → 69.6（+0.4）；texture:2 41.5 → 45.6（+4.1）；noise:2 31.6 → 41.3（+9.7）；robot:2 56.7 → 65.3（+8.6）。
- 方向上 camera / noise / robot 的 SOPT 增益大于干净条件下的 +2.5（交互为正），light 档位偏弱（pilot 已知）。**这是单宿主单 seed 的中间读数，不做显著性判断**；预注册主检验（视觉三项合并的交互，宿主分层 bootstrap）等 8 个作业全部完成后由 `scripts/e37_analyze.py` 统一计算。

## E38 先验实例复制（预注册 2026-10-24，任何结果出来之前写定；按用户要求排队）
- **动机**：E35 显示同一配方换一个先验实例，单 seed 上下游就能差 4.8 点（p=0.035）。论文主表全部只用了一个先验实例 `e2_ar_flow_f100_s`（预训练 seed 0，记作 p0）。需要确认"SOPT 有收益"不依赖这一个实例。
- **先验**：同一配方（S，ar_flow，100% DROID，全状态，ctx96，40k 步），只改预训练 seed：pB1 = seed 1（`outputs/e25j_base`，E25j 原已排队），pB2 = seed 2（`outputs/e25j_base2`，新增 `VARIANT=base2`）。都在 Jubail 训练，所以与 p0 的差异包含 seed 和平台两部分，即完整的"实例方差"。
- **宿主 1：TrunkBC**（E25j 协议，17 任务 k5，seed 0–2，50 init）：ref_p0 / base_pB1 / base2_pB2 / scratch 全部在 Jubail 同一代码重跑。
- **宿主 2：Diffusion Policy**（主表中 SOPT 增益显著的宿主，E21 配方 30k 步）：只训练 expert_droid 臂，换 pB1 / pB2，seed 0–2，单卡单臂（`scripts/slurm/e21_dp.sbatch` 新增 `ARM` / `PRIOR` / `PTAG`，run 名 `e21_dp_expert_droid_k5_t17_pB{1,2}_s{seed}`）。与已有的 none 臂和 p0 臂按 (seed, task, init) 配对。评测从 checkpoint 的 `meta.prior` 重建先验（已核对 `vla_eval.py`），不会回落到默认先验。
- **分析**：`python scripts/stats_all.py --root outputs --manifest configs/e38_manifest.json`（与主表同口径：配对差、(seed, task) 聚类 bootstrap 95% CI、合并 McNemar；主检验 Holm 校正）。已用合成数据跑通。
- **主检验（4 个，Holm）**：DP droid_pB1 vs none、DP droid_pB2 vs none、TrunkBC base_pB1 vs scratch、TrunkBC base2_pB2 vs scratch。
- **判据**：4 个主检验 Holm p < 0.05 且差值为正 → "SOPT 收益在先验实例间可复制"。同时报告 3 个实例增益的均值 ± sd，以及 pB vs p0 的直接差（次要）。若有主检验未通过，论文中主表的 SOPT 结论需注明实例依赖。
- **队列顺序**（E37 剩余 4 个作业之后）：E25j ref → base → base2 → scratch → SmolVLA s2 none / droid → DP pB1/pB2 s0 → SmolVLA s2 scratch → DP s1、s2 → E25j ctx192 / ctx384 / ctx48 / speed。DP 作业开头检查先验文件存在，缺失则立刻退出（不占用 GPU）。

## E37 main 进度：SmolVLA（libero_goal）seed 0（2026-10-24，Jubail 18931487，3.0 h）
- 10 个评测全部 exit 0，20 个 json/npz 齐全。干净：none 75.0 / expert_droid 80.6（+5.6）。
- 扰动下 none → droid：camera:2 54.8 → 66.6（+11.8）；light:3 74.8 → 83.2（+8.4）；texture:2 65.2 → 73.8（+8.6）；noise:2 70.0 → 82.8（+12.8）；robot:2 63.2 → 64.8（+1.6）。
- 单 seed 中间读数，不做显著性判断；视觉四项的增益都大于干净条件，机器人初始位姿一项反而小于干净条件。

## E37 main 进度：GR00T seed 0 两臂（2026-10-24，Jubail 18930617 none / 18931751 xattn_droid）
- 两个作业各 5 个评测全部 exit 0，json/npz 齐全。干净：none 81.9 / xattn_droid 80.4（−1.5）。
- 扰动下 none → droid：camera:2 42.1 → 70.5（+28.4）；light:3 80.2 → 79.1（−1.1）；texture:2 78.6 → 80.6（+2.0）；noise:2 71.9 → 73.6（+1.7）；robot:2 69.3 → 72.0（+2.7）。
- **camera +28.4 的核查（本地，两臂 camera2 json + 干净 rollout）**：
  - 两臂 850 个 episode 的扰动参数（yaw ±20°、pivot）逐条完全一致；正负偏转各 425 个。
  - 增益不集中：17 个任务里 16 个 droid ≥ none（唯一例外 close bottom drawer 98 vs 96），例如 white bowl on cabinet 12 → 68、yellow-white mug 10 → 56、wine rack 6 → 54、ketchup in drawer 2 → 38。
  - none 臂对偏转方向敏感（−20°：36.5，+20°：47.8），droid 臂不敏感（70.4 / 70.6）。
  - 结论：不是实现或配对伪影；在 GR00T 上，SOPT 历史分支在相机视角偏移时大幅减小性能损失（干净 → camera：none −39.8，droid −9.9）。仍为单 seed 读数，正式检验等 E37 全部完成。

## E37 main 进度：DP seed 1（2026-10-25，Jubail 18932425，4.6 h）
- 10 个评测全部 exit 0，20 个 json/npz 齐全。干净：none 71.4 / expert_droid 77.3（+5.9）。
- 扰动下 none → droid：camera:2 42.6 → 56.7（+14.1）；light:3 65.4 → 73.6（+8.2）；texture:2 37.4 → 47.6（+10.2）；noise:2 30.0 → 44.2（+14.2）；robot:2 54.7 → 64.7（+10.0）。
- 与 seed 0 同向：所有扰动下的 SOPT 增益都 ≥ 干净条件下的增益（light 例外于 s0）。GR00T seed 1 none 臂（18932782）也已完成（camera 51.9 / light 78.8 / texture 78.6 / noise 74.6 / robot 69.6），等 droid 臂配对。

## E37 main 进度：GR00T seed 1 两臂（2026-10-25，Jubail 18932782 none / 18937181 xattn_droid）
- 10 个评测全部 exit 0，json/npz 齐全。干净：none 79.6 / xattn_droid 83.9（+4.2）。
- 扰动下 none → droid：camera:2 51.9 → 68.9（+17.1）；light:3 78.8 → 80.7（+1.9）；texture:2 78.6 → 83.6（+5.1）；noise:2 74.6 → 72.8（−1.8）；robot:2 69.6 → 70.2（+0.6）。
- 与 seed 0 一致：GR00T 上只有相机视角偏移的增益明显大于干净条件，其余扰动与干净增益相当或更小。

## E37 最终结果：视觉分布偏移下 SOPT 收益更大——预注册主检验通过（2026-10-25，8 个作业全部完成）
- 数据：3 宿主 × seed 0–1 × 5 种扰动，全部评测 exit 0；分析在本地 Mac 上对拉回的 json 运行（`scripts/e37_analyze.py`，未在登录节点计算），共 22 000 个配对 episode。档位：camera 2 / light 3 / texture 2 / noise 2 / robot 2（pilot 事先定档）。
- **主检验（预注册）**：视觉四项 × 3 宿主 × 2 seed 合并的交互项 = **+4.6 点，95% CI [+3.7, +5.6] → pass = true**。
- 分宿主（视觉四项）：DP +5.1 [3.6, 6.6]；SmolVLA-goal +2.7 [0.8, 4.7]；GR00T +5.3 [3.8, 6.7]。
- 分扰动：camera **+13.0** [11.0, 15.1]；texture +2.6 [0.9, 4.3]；noise +2.8 [0.9, 4.7]；light +0.2 [−1.5, 1.8]（档位偏弱，pilot 已知）；robot（本体偏移，不进主检验）+2.0 [0.2, 3.9]。
- 各单元格成功率（两 seed 合并，%）与扰动下 SOPT vs none 配对 McNemar：

| 宿主 | 干净 none / SOPT | camera | light | texture | noise | robot |
|---|---|---|---|---|---|---|
| DP | 72.5 / 76.7 | 37.4 / 51.1（p=1e-25） | 67.3 / 71.6（2e-4） | 39.5 / 46.6（2e-9） | 30.8 / 42.8（1e-21） | 55.7 / 65.0（3e-15） |
| SmolVLA-goal | 77.7 / 81.0 | 60.2 / 68.4（9e-6） | 77.6 / 83.0（5e-5） | 67.7 / 73.6（9e-5） | 76.8 / 81.4（0.004） | 63.4 / 66.4（0.05） |
| GR00T | 80.8 / 82.1 | 47.0 / 69.7（4e-57） | 79.5 / 79.9（0.79） | 78.6 / 82.1（8e-4） | 73.2 / 73.2（1.0） | 69.5 / 71.1（0.17） |

  - 成功率保持比例（扰动 / 干净，none → SOPT）：camera DP 0.52 → 0.67、SmolVLA 0.77 → 0.84、GR00T 0.58 → 0.85；noise DP 0.42 → 0.56。
- **稳健性（事后、更保守，`scripts/e37_cluster_boot.py`）**：预注册的 bootstrap 在 episode 层面重抽样；改为按 (宿主, seed, 任务) 整簇重抽样后：
  - 主效应仍成立：+4.6 [+2.5, +6.9]；DP +5.1 [1.5, 9.0]，GR00T +5.3 [2.5, 8.0]，SmolVLA +2.7 [−2.5, 8.7]（单独不显著，只有 20 个簇）。
  - camera +13.1 [8.4, 17.9] 稳健；texture +2.6 [−0.1, 5.2]、noise +2.8 [−1.5, 7.3]、robot +2.0 [−0.1, 4.2] 在任务层面不再显著；**去掉 camera 后视觉三项合计 +1.8 [−0.3, 4.0]**。
- **结论**：
  1. 预注册结论成立：视觉分布偏移下，SOPT 带来的收益显著大于干净条件下的收益，三个宿主方向一致。
  2. 效应主要来自相机视角偏移（三个宿主一致、任务层面稳健，GR00T 的保持比例 0.58 → 0.85）。论文中应如实表述为"SOPT 尤其在相机视角变化下提升稳健性"，纹理 / 噪声的额外收益较小、在任务层面不稳健，光照档位太弱无法判断。
  3. 解释（假设，待验证）：视角变化破坏了图像到动作的映射，而 SOPT 的本体运动历史与相机无关，宿主在视觉不可靠时可以更多依赖它。GR00T 在干净条件下几乎不用 SOPT（+1.3），但在 camera 下 +22.7，与此一致。
- 可选的后续（**未排队**，需要用户决定）：camera 档位剂量曲线（10°/20°/30°）检验效应是否随偏移增大；用 E34 的 SOPT 遮蔽在 camera 扰动下做因果检验（遮蔽后增益应消失）。

## E25j / E38：TrunkBC ref（p0）在 Jubail 重跑（2026-10-25，Jubail 18939254，1.2 h）
- 3 个 seed 训练 + 评测全部 exit 0（TrunkBC 很小：15k 步约 12 min，850 episode 评测约 10 min），E24 探针 exit 0。
- 成功率：74.6 / 71.5 / 74.2，均值 73.4。DGX 上的 E23 原始结果为 72.8 / 70.9 / 72.9（72.2），同配方、同先验、跨平台重跑差 +1.2，逐 seed 方向一致——作为 E38 中 p0 的同平台参照。
- 等待 base（pB1，18939367）、base2、scratch 完成后按 `configs/e38_manifest.json` 做分析。

## E25j / E38：TrunkBC base（pB1，预训练 seed 1）（2026-10-25，Jubail 18939367，1.6 h）
- 预训练 40k 步 30 min，exit 0；DROID val selection 4.216（pos_ade 7.02 cm，const-vel 7.30）。base2（pB2，18939543）的预训练也已完成：selection 4.200（pos_ade 6.91）。
- E24 探针与 ref 几乎相同（例如 prog 0.942 vs 0.936，task 0.871 vs 0.871，stage 0.795 vs 0.771），说明新实例的表征质量与 p0 相当。
- TrunkBC 成功率：73.0 / 73.6 / 72.4，均值 73.0（ref p0 同平台 73.4）。base2 s0 = 72.7。
- 正式比较等 scratch 臂完成后用 `configs/e38_manifest.json` 计算。

## E25j / E38：TrunkBC base2（pB2）完成；scratch 臂首跑失败与修复（2026-10-25）
- base2（Jubail 18939543，1.7 h）：3 seed 全部 exit 0，成功率 72.7 / 72.1 / 72.6，均值 72.5。至此三个先验实例的 TrunkBC 均值：p0 73.4 / pB1 73.0 / pB2 72.5（极差 0.9 点）。
- scratch（Jubail 18939762）46 s 内失败：随机初始化的 trunk 用 LIBERO-90 train 划分的状态统计做归一化（`e4b_train.py::libero_state_stats`），而 Jubail 的 `processed/libero90/` 只有 `episodes.parquet`，缺 `states.npy`（以前 scratch 臂都在 127 / DGX 上跑）。
  - 修复：从 127 经本机中转 `states.npy`（md5 72422ef2…，与 127 一致）和 `summary.json`；两边 `episodes.parquet` md5 相同（9684b361…）。这只是 LIBERO-90 的本体状态数据，与 E23 scratch 臂所用完全相同，不涉及新数据或泄漏。
  - `e25_trunk.sbatch` 增加前置检查：scratch 缺该文件时立即退出。
  - scratch 已放回队首重新排队（不影响其他作业）。

## E38 TrunkBC 部分：SOPT 预训练收益在 3 个先验实例上都成立（2026-10-25，scratch 重跑 Jubail 18940106）
- scratch 重跑：3 seed 全部 exit 0，68.7 / 67.6 / 67.8（均值 68.0；DGX 上 E23 scratch 为 68.1）。
- `stats_all.py --manifest configs/e38_manifest.json`（本地，12 个 rollout json；DP 部分尚未完成，表中标为 missing）：

| 臂 | 每 seed | 均值 ± sd |
|---|---|---|
| scratch | 68.7 / 67.6 / 67.8 | 68.0 ± 0.6 |
| ref_p0 | 74.6 / 71.5 / 74.2 | 73.5 ± 1.7 |
| base_pB1 | 72.9 / 73.6 / 72.4 | 73.0 ± 0.6 |
| base2_pB2 | 72.7 / 72.1 / 72.6 | 72.5 ± 0.3 |

| 比较 | 差值 | 95% 聚类 CI | 不一致对 | McNemar p |
|---|---|---|---|---|
| ref_p0 vs scratch | +5.4 | [+2.2, +8.8] | 368:230 | 1.8e-8 |
| **base_pB1 vs scratch**（主检验） | +4.9 | [+1.8, +8.1] | 366:240 | 3.5e-7 |
| **base2_pB2 vs scratch**（主检验） | +4.4 | [+0.9, +7.8] | 367:254 | 6.6e-6 |
| base_pB1 vs ref_p0 | −0.5 | [−2.9, +2.0] | 235:247 | 0.62 |
| base2_pB2 vs ref_p0 | −1.0 | [−3.5, +1.5] | 234:259 | 0.28 |

- 预注册的 Holm 校正针对 4 个主检验；即使按 4 个校正（p × 4），两个 TrunkBC 主检验的调整后 p 仍 ≤ 3e-5 → **TrunkBC 部分通过**。
- 三个实例的预训练增益：+5.4 / +4.9 / +4.4，均值 +4.9 ± 0.5；实例之间两两差异都不显著。与 E35 中 GR00T × 长时程上较大的实例方差相比，TrunkBC 上"DROID 预训练 > 随机初始化"对先验实例很稳健。
- DP 部分（6 个作业）正在排队，完成后补全 4 个主检验。

## 进度（2026-10-25 晚）：SmolVLA 17 任务 s2 droid；E38 DP pB1 s0
- SmolVLA s2 expert_droid（Jubail 18940450）：训练 / 评测 exit 0，67.8（s2 none 64.2，+3.5）。expert_scratch s2（18941753）排队中，完成后合并三 seed 更新主表。
- E38 DP pB1 s0（Jubail 18940875，1.3 h）：exit 0，**76.6**（同 seed：none 73.6、p0 76.1）。
  - 核对：meta 记录 prior = `outputs/e25j_base/best.pt`；与 none / p0 两臂的 85 条示范 episode 完全相同，steps / bs / lr / chunk / cond_drop 一致（旧 meta 无 ctx 字段，默认 96）→ 配对有效。
- E38 DP pB2 s0（Jubail 18941750，1.3 h）：exit 0，**75.9**（prior = `outputs/e25j_base2/best.pt`，示范 episode 与 none 臂一致）。seed 0 三个实例：p0 76.1 / pB1 76.6 / pB2 75.9，none 73.6。
- E38 DP pB1 s1（Jubail 18942601，1.3 h）：exit 0，**74.0**（prior = `outputs/e25j_base/best.pt`，示范 episode 与 none 臂一致）。seed 1：none 71.4 / p0 77.3 / pB1 74.0。

## SmolVLA 17 任务 3 seed 主表（2026-10-26；s2 三臂均在 Jubail：none / droid 已记，expert_scratch 18941753，2.6 h，exit 0）
| arm | s0 | s1 | s2 | 均值 |
|---|---|---|---|---|
| none | 60.6 | 60.1 | 64.2 | 61.6 |
| expert_scratch | 58.7 | 58.5 | 62.2 | 59.8 |
| expert_droid | 64.0 | 67.4 | 67.8 | **66.4** |
- 核对：s2 三臂 850 ep / n_exec 10；步数 20000、lr 1e-4、k5 一致；三臂示范 episode 完全相同。
- s2 配对 McNemar：droid vs none 100:70 p=0.026；droid vs scratch 108:61 p=3.7e-4；scratch vs none 65:82 p=0.19。
- **3 seed 合并 McNemar**（s0+s1 的不一致对计数取自上方 E23 汇总，加上 s2）：droid vs none 341:220 **p=3.7e-7**；droid vs scratch 368:200 **p=1.7e-12**。三个 seed 方向都一致（+3.4 / +7.3 / +3.5）。
- 簇 bootstrap（stats_all + paper_manifest）需要 s0/s1 逐回合数据，这些数据在 DGX 上；目前 DGX 和 127 都连不上，等可访问时补算。

- E38 DP pB2 s1（Jubail 18942994，1.3 h）：exit 0，**75.8**（prior = `outputs/e25j_base2/best.pt`，示范 episode 与 none 臂一致）。seed 1 对 none（71.4）：p0 77.3（116:66 p=2.6e-4）/ pB1 74.0（100:78 p=0.12）/ pB2 75.8（106:69 p=0.006）。剩 pB1 s2（18943514 运行中）、pB2 s2。

## E38 最终结果：先验实例复制 —— **按预注册判据通过**（2026-10-26）
- 最后两个 DP 作业：pB1 s2（Jubail 18943514，1.3 h）78.8；pB2 s2（18943535，1.3 h）76.7，均 exit 0。
- 12 个 DP 运行逐一核对：
  - `meta.prior` 与臂一致（p0 = `e2_ar_flow_f100_s`，pB1 = `e25j_base`，pB2 = `e25j_base2`）。
  - 示范 episode 与同 seed 的 none 臂完全相同；(task, init) 键集合相同；每个运行 850 ep。
  - steps / bs / lr / k / chunk / cond_drop 与 none 一致。新运行 meta 中多出的 ctx = 96、prior_lr_mult = 1.0 是后来加入的字段，取值等于旧代码的默认值。
- 分析：`stats_all.py --manifest configs/e38_manifest.json`，本地运行，Jubail 结果用单个 tar 拉回。

| DP | s0 / s1 / s2 | 均值 | vs none | 95% CI（(seed,task) 聚类） | 不一致对 | McNemar p |
|---|---|---|---|---|---|---|
| none | 73.6 / 71.4 / 73.5 | 72.9 | | | | |
| droid_p0 | 76.1 / 77.3 / 79.1 | 77.5 | +4.6 | [+1.8, +7.4] | 321:203 | 2.9e-7 |
| droid_pB1 | 76.6 / 74.0 / 78.8 | 76.5 | +3.6 | [+0.9, +6.5] | 311:219 | 7.4e-5 |
| droid_pB2 | 75.9 / 75.8 / 76.7 | 76.1 | +3.3 | [+0.2, +6.2] | 306:223 | 3.5e-4 |

| TrunkBC | s0 / s1 / s2 | 均值 | vs scratch | 95% CI | 不一致对 | McNemar p |
|---|---|---|---|---|---|---|
| scratch | 68.7 / 67.6 / 67.8 | 68.0 | | | | |
| ref_p0 | 74.6 / 71.5 / 74.2 | 73.5 | +5.4 | [+2.1, +8.7] | 368:230 | 1.8e-8 |
| base_pB1 | 72.9 / 73.6 / 72.4 | 73.0 | +4.9 | [+1.8, +8.1] | 366:240 | 3.5e-7 |
| base2_pB2 | 72.7 / 72.1 / 72.6 | 72.5 | +4.4 | [+1.0, +7.8] | 367:254 | 6.6e-6 |

- **4 个主检验**：按预注册做 Holm 校正，跨两个宿主 4 个一起校正。（stats_all 本身就对 manifest 中全部主检验一起做 Holm，表内 Holm 列与此处数值一致；2026-10-26 更正：此前误写为"块内校正"。）
  - TrunkBC pB1：1.4e-6
  - TrunkBC pB2：2.0e-5
  - DP pB1：1.5e-4
  - DP pB2：3.5e-4
  - 全部 < 0.05 且差值为正 → **"SOPT 收益在先验实例间可复制"成立**。
- 实例间增益（均值 ± sd，3 个实例）：DP +3.8 ± 0.7，TrunkBC +4.9 ± 0.5。
- 次要比较（pB vs p0，直接差）：
  - DP：pB1 −1.0 [−2.6, +0.6] p=0.15；pB2 −1.4 [−3.1, +0.2] p=0.058。
  - TrunkBC：pB1 −0.5 p=0.62；pB2 −1.0 p=0.28。
  - 均不显著，但两个宿主上新实例都略低于 p0，方向一致。主表用 p0 的点估计可能略偏乐观，约 1 点；论文中应报告 3 个实例的均值作为更保守的效应量（DP +3.8，TrunkBC +4.9）。
- E35 中"换实例单 seed 差 4.8 点"的现象，在 3 seed 合并后缩小到约 1 点：主要是单 seed 噪声，不是实例本身的系统差异。

## E25j / E33：ctx192（12.8 s 历史）TrunkBC（2026-10-26，Jubail 18944391，1.6 h，76d6b67）
- 预训练：`model.ctx_len=192 data.left_pad=96`（`outputs/e25j_ctx192/config.yaml` 已核对），40k 步，selection 4.190（base 4.216 / base2 4.200）。
- TrunkBC 训练和评测都从先验的 `ctx_len` 读取历史长度（`e4b_train.py:115`，`e4b_eval.py:29`），所以策略确实使用 192 帧历史。3 seed 训练 / 评测和探针全部 exit 0。
- 分析：`stats_all.py --manifest configs/e25j_manifest.json`（新增；探索性，不设 Holm 主检验）。按 E38 的实例波动带（base / base2 vs ref：−0.5 / −1.0）来读各变体相对 ref 的差。

| | s0 / s1 / s2 | 均值 | vs ref | 95% CI | 不一致对 | p |
|---|---|---|---|---|---|---|
| scratch | 68.7 / 67.6 / 67.8 | 68.0 | | | | |
| ref（ctx96） | 74.6 / 71.5 / 74.2 | 73.5 | | | | |
| ctx192 | 71.2 / 75.4 / 72.9 | 73.2 | −0.3 | [−3.2, +2.5] | 254:261 | 0.79 |

- ctx192 vs scratch：+5.1，CI [+2.0, +8.2]，p=9.2e-8。
- 探针 R²（ctx96 droid_ee → ctx192）：
  - 两个目标略升：t_next 0.51 → 0.54，fut05 0.84 → 0.85。
  - 其余持平或略降：acc 0.90 → 0.87，grip_vel 0.98 → 0.95，stage 0.82 → 0.80。
  - multi 版：stage 0.92 → 0.94，t_next 0.57 → 0.58。
- **结论**：把历史加长到 12.8 s，下游 TrunkBC 没有收益，与 ctx96 的差落在先验实例波动范围内。探针中阶段 / 下一事件信息只小幅增加。剩 ctx384（18944415 运行中）、ctx48、speed。

## E25j / E33：ctx384（25.6 s 历史）TrunkBC（2026-10-26，Jubail 18944415，1.8 h，7692b6d）
- 预训练：`model.ctx_len=384 data.left_pad=288`（config 已核对），40k 步，selection 4.245（ctx96 系列 4.19–4.22）。3 seed 训练 / 评测和探针全部 exit 0。

| | s0 / s1 / s2 | 均值 | vs ref | 95% CI | 不一致对 | p |
|---|---|---|---|---|---|---|
| ref（ctx96） | 74.6 / 71.5 / 74.2 | 73.5 | | | | |
| ctx192 | 71.2 / 75.4 / 72.9 | 73.2 | −0.3 | [−3.2, +2.5] | 254:261 | 0.79 |
| ctx384 | 72.8 / 71.5 / 70.4 | 71.6 | −1.9 | [−5.0, +1.1] | 226:274 | 0.035 |

- ctx384 vs scratch：+3.5，CI [−0.1, +7.1]，p=3.8e-4。
- 探针（droid_ee → ctx384）：局部动力学下降（acc 0.90 → 0.83，grip_vel 0.98 → 0.93，t_grip 0.83 → 0.79，stage 0.82 → 0.76）；t_next 0.51 → 0.54 略升；multi 版基本持平。
- **结论**：历史再加长到 25.6 s 反而略差。
  - 合并 McNemar p=0.035，但 (seed, task) 聚类 CI 包含 0，幅度只略超先验实例波动（−0.5 / −1.0）：记为"无收益、可能有害"，不作强结论。
  - 合理解释：LIBERO-90 示范平均 108.8 帧（15 Hz，约 7.3 s；`processed/libero90/summary.json`），只比 ctx96 的 6.4 s 窗口略长；ctx192（12.8 s）和 ctx384（25.6 s）多出的历史在 LIBERO 上大部分是首帧填充，因此长窗口在下游无从发挥；DROID 上更长窗口也稀释了对局部动力学的建模（探针 acc / grip_vel 下降）。
  - 结合 ctx192：ctx96 已在平台区，论文保持 ctx96。剩 ctx48（18945277）、speed。

## E25j / E33：ctx48（3.2 s 历史）TrunkBC（2026-10-26，Jubail 18945277，1.6 h，7692b6d）
- 预训练：`model.ctx_len=48`，left_pad 0（config 已核对）。窗口只需 ≥48 帧真实历史，所以 DROID 锚点比 ctx96 略多。40k 步，selection 4.364（ctx96 系列 4.19–4.22）；DROID val ADE 7.35 cm，接近匀速基线 7.50，说明短历史对 DROID 预测明显更弱。3 seed 训练 / 评测和探针全部 exit 0。

| 历史 | s0 / s1 / s2 | 均值 | vs ref | 95% CI | 不一致对 | p |
|---|---|---|---|---|---|---|
| 3.2 s（ctx48） | 71.9 / 71.3 / 74.8 | 72.7 | −0.8 | [−3.5, +2.0] | 229:249 | 0.38 |
| 6.4 s（ref，ctx96） | 74.6 / 71.5 / 74.2 | 73.5 | | | | |
| 12.8 s（ctx192） | 71.2 / 75.4 / 72.9 | 73.2 | −0.3 | [−3.2, +2.5] | 254:261 | 0.79 |
| 25.6 s（ctx384） | 72.8 / 71.5 / 70.4 | 71.6 | −1.9 | [−5.0, +1.1] | 226:274 | 0.035 |

- ctx48 vs scratch：+4.6，CI [+1.8, +7.5]，p=3.3e-6。
- 探针（droid_ee → ctx48）：stage 0.82 → 0.74、prog 0.95 → 0.92、t_next 0.51 → 0.49 下降（长程阶段信息变少）；acc 0.90 → 0.91 持平。
- **历史长度消融小结**：
  - 下游 TrunkBC 对历史长度在 3.2–12.8 s 内不敏感，差值都在先验实例波动（≈1 点）之内；25.6 s 略差。
  - 每个长度相对 scratch 都显著为正（+3.5 至 +5.1）。
  - 探针中的阶段信息从 3.2 s 到 6.4 s 明显增加，更长后持平或回落（stage 0.74 / 0.82 / 0.80 / 0.76，multi 版 0.85 / 0.92 / 0.94 / 0.93），但这类信息在 LIBERO 短任务上没有转化为下游收益。
  - 论文保持 ctx96。只剩 speed（18945933）。

## E39 / E40 / E41 / E38b 预注册 + SmolVLA 主表 Jubail 重跑（2026-10-26，任何结果出来之前写定；按用户要求排队 P0 与统计缺口）
统一分析：`scripts/contrast_boot.py`（新增）。
- 每个对比是"各宿主组的加权和"；组内是若干 run 的带权和，按 (seed, task) 聚类重采样，同组各项共用同一组簇，以保持配对；各宿主组独立重采样。
- 报告：估计值（百分点）、95% 百分位 CI、双侧 bootstrap p；主检验做 Holm 校正。
- 定义由 `scripts/make_contrast_specs.py` 生成（`configs/e39/e40/e41_contrasts.json`，`configs/e38b_manifest.json`，`configs/smolvla_jubail_manifest.json`），并已用植入已知效应的合成数据检验：
  - 低数据增益：植入 +10 → 估计 +10.0；
  - 剂量：植入 +10 → 估计 +7.8，CI [5.2, 10.5]；
  - 遮蔽：植入 0 → 估计 +0.8，CI 含 0。

**E39 示范数量曲线（P0-1）**
- 宿主：Diffusion Policy（E21 配方，30k 步）和 SmolVLA（E23 配方，20k 步），17 任务。
- k ∈ {1, 2, 10, 25, 全部}，加上已有的 k5；seed 0–2。
  - 臂：none / expert_droid；k1 和全部再加 expert_scratch。
  - 每个 (宿主, k, seed) 一个作业，各臂各占一块 GPU。
- 示范子集是嵌套的（`select_episodes` 按加盐哈希排序取前 k 个）。步数在各 k 下固定为宿主配方值，不随 k 调。
- 主检验（2 个，Holm）：
  - **P1** k1 和 k2 的平均增益（droid − none，两宿主平均）> 0；
  - **P2** 增益随数据量下降：k1、k2、k5 的平均增益 − k25、全部的平均增益 > 0（两宿主平均）。
- 次要：每个 (宿主, k) 的增益；k1 和全部下 droid − scratch（区分预训练与组件）。
- 判据：P1、P2 的 Holm p < 0.05 且方向为正 → "SOPT 是数据效率收益，随示范增多而减小"。若 P2 不显著，则论文写"在所测数据量范围内收益稳定"。

**E40 宿主自身的动作预训练（P0-3，Diffusion Policy）**
- 阶段 1：none 臂在 LIBERO-90 **train split 的 52 个任务的全部示范**上预训练，100k 步，batch 64。
  - 17 个评测任务不在其中；`vla_train.py` 会断言评测任务、以及微调用的 episode 都不在宿主预训练数据中。
  - 任务 one-hot 覆盖全部 LIBERO-90 任务（`--task-vocab all`），所以两阶段的权重形状一致、可完整加载。
- 阶段 2：从这个宿主出发，17 任务 k5 微调 none / expert_droid（30k 步，seed 0–2；`--init-host`）。
  - 动作 / 状态归一化沿用宿主预训练时的统计量，日志会记录微调动作超出该范围的比例。
- 主检验（1 个）：交互 = 增益（从头训练的宿主，E21 主表）− 增益（预训练过的宿主）> 0。
- 次要：预训练宿主上的增益；宿主预训练本身对 none 的效果（检查天花板）。
- 局限：只有 1 个宿主预训练实例；"从头训练的宿主"一侧用 E21 主表的 run（同配方、同 seed、同示范，one-hot 维度不同：多出的维度输入恒为 0，不影响输出）。
- 判读：交互显著为正，说明"互补"的因果证据成立；不显著，则论文中强宿主无收益只作相关性描述。

**E41 相机分析补充（P0-4，只评测，E37 的 checkpoint，seed 0–1，50 init）**
- 剂量：none 和 SOPT 臂在 camera:1（10°）和 camera:3（30°）下评测；20° 与干净条件已有。
- 遮蔽：SOPT 臂整个 episode 关闭 SOPT 输入（`--sopt-mask-grasp -1`，同 E34），在干净和 20° 下各测一次。只做 DP 和 GR00T：SmolVLA 的 SOPT 特征在 forward 内部计算，没有可用的遮蔽钩子。
- 主检验（2 个，Holm）：
  - **P1** 交互（30°）− 交互（10°）> 0（3 个宿主平均；交互 = (sopt−none)_扰动 − (sopt−none)_干净）；
  - **P2** 关闭 SOPT 的代价在 20° 下大于干净条件下（DP 和 GR00T 平均）。
- 注意：训练时 SOPT 一直开着，关闭属于分布外（E34），所以 P2 测的是"偏移下对 SOPT 的依赖是否增加"，不是干净的因果消融。

**统计缺口**
- SmolVLA 主表 s0 / s1 在 Jubail 重跑（`TAG=_jr`，三臂）。之后 SmolVLA 主表统一用 Jubail 的 3 个 seed（`configs/smolvla_jubail_manifest.json`），DGX 上的 s0 / s1 作为跨平台复制报告。
- **E38b**：把 E38 的先验实例复制扩到 SmolVLA 和 FastWAM。
  - expert_droid 换 pB1 / pB2，seed 0–2，同一作业里两个先验各占一块 GPU（`PSET`）。
  - 主检验：每个宿主的 pB1 vs none、pB2 vs none，共 4 个，Holm。
  - `wam_train.py` / `wam_eval.py` 新增 `--prior`，评测时从 meta 读取；旧 run 没有这个字段，回落到默认先验。

**实现**
- `e21_dp` / `e23_smolvla` 新增 `K`、`ARMS`、`PSET`（逗号分隔，不含空格，避免队列脚本的引号问题）；默认值下 run 名不变，命名逻辑已在本地用桩函数逐一核对。
- DP 新增 `INIT_HOST`。
- 新文件：`e40_host.sbatch`（宿主预训练前先跑一遍 30 步的全链路冒烟）、`e41_camera.sbatch`、`prep_frames.{py,sbatch}`。
  - `prep_frames`：在 CPU 节点联网解码 17 任务全部示范和 train split 全部示范的视频，避免离线训练作业缺帧。
- 评测与诊断脚本从 checkpoint 的 `vocab` 读取 one-hot 词表，评测任务仍是 `tasks`。

**P0-2 跨机器人（E42）调研结论，尚未实现、未排队**
- 候选：MimicGen robot-transfer 数据（HF `amandlek/mimicgen_datasets/robot`，CC-BY-4.0）。Square / Threading D0 在 Panda / Sawyer / IIWA / UR5e 上各有 1000 条示范，栈是 robosuite 1.4.1 + MuJoCo 2.3.2，与 LIBERO 同类。
- 需要单独的 pydeps 环境、数据转换（基座坐标系下的 EE 位姿和夹爪标量，20 → 15 Hz）和闭环评测循环。
- 局限：非 Panda 示范由 Panda 示范重放生成，运动风格接近。
- 下一步：先实现并做冒烟，再写预注册。

**排队顺序**（最多 2 个并发）：
1. prep_frames
2. SmolVLA _jr s0、s1
3. E39 DP k1、k2（s0–2）
4. E40 宿主预训练
5. E39 DP k10、k25、全部
6. E41 × 6
7. E40 微调 s0–2
8. E39 SmolVLA（15 个）
9. E38b SmolVLA × 3
10. E38b FastWAM × 3
- 帧缓存准备（Jubail 18946665，compute 分区，27 min，exit 0）：
  - 17 任务全部示范：771 / 771 个 episode（之前已有 361 个）。
  - train split 52 个任务全部示范：2725 / 2725 个 episode。
  - `libero90_frames256` 共 186 GB。
  - E39（k = 全部）和 E40 宿主预训练所需的帧均已就绪，训练作业保持离线运行。

## E25j / E33：speed（速度增强）TrunkBC（2026-10-26，Jubail 18945933，1.6 h，3daf60e）
- 预训练：`data.speed_aug=[0.5,2.0] data.speed_aug_prob=0.8`，其余同 ctx96 ref 配方，40k 步，selection 4.194（ctx96 系列 4.19–4.22）；DROID val ADE 6.84 cm（匀速 7.30）。预训练、3 seed 训练 / 评测、探针全部 exit 0；每 seed 850 回合、17 任务（本地核对）。

| 变体 | s0 / s1 / s2 | 均值 | vs ref | 95% CI | 不一致对 | p |
|---|---|---|---|---|---|---|
| speed | 73.9 / 71.4 / 72.9 | 72.7 | −0.7 | [−3.0, +1.6] | 209:227 | 0.42 |
| ref（ctx96） | 74.6 / 71.5 / 74.2 | 73.5 | | | | |

- speed vs scratch：+4.7，CI [+1.6, +7.8]，p=1.1e-6。
- 探针（droid_ee → speed）：acc 0.90 → 0.87 略降，其余基本持平（stage 0.82 → 0.81，fut10 0.75 → 0.76，task 0.87 → 0.87）；multi 版 stage 0.92 → 0.94。
- **结论**：速度增强对下游 TrunkBC 无收益，差值在先验实例波动（≈1 点）内。E25j 预训练配方消融（ctx48 / ctx192 / ctx384 / speed）全部完成：没有任何变体超过 ref，各变体相对 scratch 都显著为正（+3.5 至 +5.1）。论文保持 ref 配方，并以"增益对预训练配方细节稳健"呈现这组结果（探索性，不做 Holm）。

## E37s / E18r 预注册（2026-10-26，用户批准；任何结果出来之前写定）
动机（论文审稿风险）：
- 视角鲁棒（E37）只比了 none 和 DROID，没有 scratch 臂，分不清鲁棒性来自"预训练"还是只来自"有本体历史输入"。
- E18（同域状态预训练无效）和 E11（1% DROID 就有效）只在 DEV7 上测过，DEV7 在开发中被反复使用。

**E37s：scratch 臂在视觉偏移下（只评测）**
- 臂：同结构随机初始化的 SOPT（expert_scratch）。
  - DP：`e21_dp_expert_scratch_k5_t17_s{0,1}`，17 任务；
  - SmolVLA-goal：`e13_libero_goal_expert_scratch_k5_s{0,1}`。
  - 17 任务的 GR00T xattn_scratch checkpoint 不存在，本次不含 GR00T。
- 扰动、档位、初始状态与 E37 完全相同（camera 2 / light 3 / texture 2 / noise 2 / robot 2，50 init，`outputs/e37_levels.json`）。干净条件用已有的 `rollouts_n50.json`。
- 运行：`MODE=main HOST=dp|smolvla_goal SEED=0|1 ARM=scratch sbatch scripts/slurm/e37_robust.sbatch`，4 个单卡作业。
- 分析：`python scripts/contrast_boot.py --root outputs --spec configs/e37s_contrasts.json`，(seed, task) 簇 bootstrap，宿主分层。
- 主检验（2 个，Holm）：交互 D = (droid − scratch)_扰动 − (droid − scratch)_干净。
  - **P1**：camera，DP 与 SmolVLA-goal 平均；
  - **P2**：视觉 4 项平均，同样两宿主平均。
- 次要：
  - scratch vs none 的交互（camera、视觉 4 项），即"只有历史"的鲁棒性；
  - E37 的 droid vs none 交互限定在这两个宿主上的参照值（现有数据：camera +7.2 [0.1, 14.2]，视觉 4 项 +3.9 [0.6, 7.4]）；
  - 每个宿主 × 扰动的两种交互。
- 判读（写作措辞）：
  1. P1 的 Holm p < 0.05 且为正：写"预训练表示带来视角鲁棒性"。
  2. 否则，若 scratch vs none 的 camera 交互 CI > 0：写"本体历史带来视角鲁棒性，预训练的额外贡献未被证实"，论点 B 按此改写。
  3. 否则：写"鲁棒性来源不能确定"，只报告 SOPT（DROID）vs none 的结果并注明这一局限。
- 功效说明：两宿主的 droid 交互本身 CI 较宽（camera +7.2 [0.1, 14.2]），P1 不显著不能解读为"预训练与 scratch 相同"。
- 代码与验证：
  - `e37_robust.sbatch` 新增 `ARM=scratch`；`make_contrast_specs.py` 新增 `e37s`。
  - 用真实 E37 文件植入已知效应检验：scratch := droid 时 P1 = 0；scratch := none 时 P1 = droid 交互（+7.2）。
  - `contrast_boot.py` 的 bootstrap p 上限改为 1（估计恰为 0 时原先显示 2）。

**E18r / E11r：预训练数据来源与数量在 17 任务上复现（TrunkBC，Jubail）**
- 变体（`e25_trunk.sbatch` 新增 VARIANT，TrunkBC 17 任务 × k5 × seed 0–2 × 50 init，并跑 E24 探针）：
  - `f1` / `f10`：DROID 1% / 10%（e2 配方，嵌套的 `data.train_fraction`；按 DROID val 选模，与 E11 原先验相同）。
  - `libero`：只用 LIBERO-90 train split 的状态，用 E4a scratch 配方（s + e2 + libero90 + e4a，lr 3e-4，warmup 300，3k 步）。
- **防泄漏修正**：原 E18 的 LIBERO-only 先验（e4a_ar_flow_scratch_kall）按 LIBERO val split 选模，而 val split 含 17 任务协议中的 10 个评测任务。复现中改为在 train split 内部按加盐哈希留出 10% 的任务做选模（新增 `data.holdout_train_tasks`）：
  - 训练 48 个任务 / 2401 条示范，选模 4 个任务 / 324 条；
  - 与 17 个评测任务交集为 0（本地用真实 episodes.parquet 验证）。
  - 本地 CPU 30 步冒烟通过（9.78M 参数）。
- 对照：已有的 `e25j_trunk_scratch` 与 `e25j_trunk_ref`（DROID 100%，p0），同 seed 配对。
- 分析：`python scripts/stats_all.py --root outputs --manifest configs/e18r_manifest.json`。
- 主检验（2 个，Holm）：**ref vs libero > 0**（DROID 优于同域状态预训练）；**f1 vs scratch > 0**（1% DROID 就有增益）。
- 次要：libero vs scratch（报告点估计与 CI；只有当 CI 很窄、接近 0 时，才写"同域预训练无增益"，否则写"增益显著小于 DROID"）；f10 vs scratch；ref vs f1、ref vs f10、f10 vs f1（数据量趋势）。
- 判读：两个主检验都通过时，这两条才能进摘要和图 1；未通过的一条只放附录，并注明 DEV7 与 17 任务的差异。
- 局限：每个变体只有一个先验实例（E38：实例间约 ±1 点）。

**排队**：插到 E41 之前，顺序为 E37s DP s0、s1 → SmolVLA-goal s0、s1 → E18r f1 → f10 → libero，均为单卡作业。

## E37 补充分解：宿主 × 扰动的任务簇 CI（2026-10-26，事后、本地分析）
- 数据：`e37final`（与 Jubail `outputs/` 逐文件 md5 一致；档位 camera 2 / light 3 / texture 2 / noise 2 / robot 2）。方法同 `scripts/e37_cluster_boot.py`（宿主 × seed 分层，按任务整簇重抽样，5k）。
- camera 交互项分宿主：DP **+9.5 [3.0, 16.8]**，GR00T-xattn **+21.4 [14.5, 28.4]**，SmolVLA-goal **+4.9 [−6.6, 17.6]**（20 簇，CI 跨 0）。三宿主合并 +13.0 [8.5, 17.8]。
  - 6 个（宿主, seed）的 camera 交互全部为正：DP 10.8 / 8.2，SmolVLA 6.2 / 3.6，GR00T 29.9 / 12.8。
- DP + SmolVLA 按 episode 加权为 +7.8 [1.6, 14.3]；E37s 预注册参照值 +7.2 是宿主等权平均，两者口径不同。
- 论文措辞：相机鲁棒"三宿主方向一致、合并显著"，不写"每个宿主都显著"。只测了 DP / SmolVLA-goal / GR00T-xattn，FastWAM、π0.5、X-VLA、TrunkBC 没有视觉偏移评测。
- 完整表：`paper/SOPT_results.xlsx` T2（分 seed 成功率、交互分解、保持比例）。

## SmolVLA 主表 Jubail 重跑（`_jr`）改为单卡作业（2026-10-26 17:45，用户批准）
- 原来的两个 3 卡作业 18946722（s0）、18948300（s1）从 11:08 / 13:22 起一直 PENDING，Slurm 预计开始时间为 10-27 11:11 和 11-01 10:22，会把后面整个队列卡住；两个作业都未启动，没有任何输出。
- 已取消，改为 6 个单卡作业：`SEED ∈ {0,1}` × `ARM ∈ {none, expert_scratch, expert_droid}`，`TAG=_jr`，排在队首。run 名不变（`e23_smolvla_<arm>_k5_jr_t17_s<S>`），训练与评测配方不变。

## SmolVLA `_jr` 重跑进度：s0 none（2026-10-27，Jubail 18951191，cn011，2.4 h）
- 训练 20k 步 exit 0（1.8 h），评测 n_exec 10 exit 0（34 min）。核验：run 名 `e23_smolvla_none_k5_jr_t17_s0`，commit 0130b58，k=5，suite libero90_17，seed 0，prior p0，samples 13186；850 episode = 17 任务 × 50。
- 成功率 **60.1**（DGX 原 s0 为 60.6，同平台差 −0.5）。
- s0 expert_scratch（18951197）训练已完成、评测中；s0 droid 与 s1 三臂由队列自动提交。6 臂齐后按 `configs/smolvla_jubail_manifest.json` 计算。
- s0 expert_scratch（Jubail 18951197，cn017，2.8 h）：训练、评测均 exit 0；run 名、commit 0130b58、k=5、seed 0、samples 13186 与 none 臂一致；850 episode。成功率 **61.9**（DGX 原 s0 为 58.7，跨平台 +3.2）。与 Jubail s0 none 配对：82:67。
  - 注意：DGX 上 scratch < none（58.7 vs 60.6），Jubail 上 scratch > none（61.9 vs 60.1）。单 seed 跨平台波动约 3 点，与之前 none 的 seed 间标准差（约 4.6）同量级；结论等 Jubail 3 seed 齐后再下。

## 【数据验证】DV1–DV3 预注册（2026-10-27，用户批准；任何结果出来之前写定）
目的：为论文中"历史提供了当前状态之外的未来运动信息"这一论证提供 DROID 上的直接数据证据。全部只用 DROID 本体状态。
**评测集**：DROID test 站点（UT Austin + Edinburgh + Penn，7,164 条），此前从未用于任何选择或报告。检查点按原配方在 val（UW）上选。

**DV1 历史长度**（`scripts/dv_anchor_eval.py`，`scripts/slurm/dv_forecast.sbatch`）
- 先验：上下文 4 / 16 / 48 / 96 帧（1 / 4 / 12 / 24 个 token；0.27 / 1.07 / 3.2 / 6.4 s）。
  - 都用主先验配方（S、ar_flow、100% DROID、40k 步、H=32 = 2.1 s）且 seed 0，只改 `model.ctx_len`。
  - ctx4 / ctx16 新训（`dv1_ctx4` / `dv1_ctx16`）；ctx48 = `e25j_ctx48`；ctx96 = p0（`e2_ar_flow_f100_s`）。
  - 另两个 ctx96 实例（pB1 / pB2，seed 1 / 2）用来给出先验实例间的噪声尺度。
- **固定锚点**：`evaluate.py` 按各模型自己的 ctx_len 取窗口，不同 ctx 的模型会在不同窗口上评分。
  - 这里每个窗口都要求 96 帧真实历史 + 32 帧真实未来，锚点间隔 32 帧；ctx=C 的模型读最后 C 帧。
  - 所有模型预测同一批未来，每个 batch 使用相同的 flow 采样噪声，K = 10。
- **主检验**：test 上 ΔES = ES(ctx4) − ES(ctx96, p0)，95% CI 为按 episode 整簇的 bootstrap，CI > 0 即通过。
  - 报告相对增幅 X% = ES(ctx4) / ES(ctx96) − 1。
- **次要**：
  - 相邻长度差（4→16→48→96）；
  - 分预测步长的 ES 比值曲线；
  - 分站点结果；
  - val 上的复制；
  - 384 帧历史的锚点集上加入 ctx192 / ctx384（`e25j_ctx192` / `e25j_ctx384`，训练时 left_pad，这里历史全部真实）。
- **措辞**：
  - 通过 → "历史从 0.27 s 增加到 6.4 s，留出实验室上未来 2.1 s 末端轨迹的能量分数下降 X%"。
  - ctx4 已含约 0.27 s 的运动（速度），不是严格的"只看当前状态"。patch = 4 时无法训练 1 帧上下文的同配方模型，论文里如实写成"近乎瞬时的状态"。
  - 若 ctx48 与 ctx96 的差小于实例间差，则写"信息主要集中在最近约 3 s"。
- 已在本地核对：在同一窗口上，基线 ES 与 `evaluate_forecast` 完全一致，模型 ES 只差采样噪声；逐步 ES 与暴力计算一致。

**DV2 相邻帧冗余度**（`scripts/dv_redundancy.py`，CPU 作业）
- 15 Hz，episode 内统计，train 与 test 分别给出，按 episode bootstrap 得 95% CI，纯描述、不做检验：
  - 恒速外推的 R²：目标 p_{t+h} − p_t，预测 h·v_t，v_t 取 1 帧差分或 3 帧平均（后者即 const-vel 基线）；h = 1, 2, 4, 8, 16, 32, 64；h = 1 即下一帧冗余度。末端位置与关节角各算一次。
  - 速度自相关：滞后 1–96 帧。
  - 夹爪状态保持概率。
- 所有 h / 滞后共用同一锚点集，曲线之间可比。汇总公式已在调试数据上与直接计算逐位核对。

**DV3 "怎么动 / 去哪"分解**（只评测，E3a 检查点）
- 检查点：`gc_ar_flow_ft`（主）和 `gc_ar_flow_scratch`（复制）。从 127 经本机中转到 Jubail，md5 一致。
- 在 test 的固定锚点上，比较三种条件下的逐步 ES：不给目标、给 keyframe（下一次夹爪事件，时间未知）、给 endpoint（第 32 步状态）。
- 描述量："去哪"份额(h) = 1 − ES_h(给目标) / ES_h(不给目标)。endpoint 在最后一步给出答案，末端份额趋近 1 是平凡的，关键看曲线中段；keyframe 是非平凡版本。
- 检查项：endpoint 的 ES 应明显低于不给目标（E3a 在 val 上为 1.88 vs 3.97），否则说明目标通路加载有误。
- 定位：Stage 2 的一张图，Stage 1 不需要。

**排队**：DV2（CPU）与 DV1+DV3（1 个 GPU 作业，约 2 h）放到队首。为腾出名额，取消尚未开始的 SmolVLA `_jr` s1 none（18955245），重新排在 DV 之后。

## 【数据验证】DV2 结果：相邻帧冗余度（2026-10-27，Jubail 18955433，CPU，5 min，exit 0，commit ca0eb4c）
- 覆盖：train 78,018 / 83,606 条，test 6,930 / 7,164 条；不足 98 帧的短 episode 不计。95% CI 为 episode bootstrap（1000 次），train 上 CI 宽度都在 ±0.01 以内。
- **恒速外推的 R²**（末端位置，v 取 1 帧差分；train / test）：

| 预测步长 | 1 帧（67 ms） | 2 帧 | 4 帧（0.27 s） | 8 帧（0.53 s） | 16 帧（1.07 s） | 32 帧（2.1 s） | 64 帧（4.3 s） |
|---|---|---|---|---|---|---|---|
| R² train | 0.957 | 0.915 | 0.798 | 0.510 | −0.03 | −1.03 | −3.78 |
| R² test | 0.967 | 0.935 | 0.847 | 0.623 | 0.18 | −0.60 | −2.44 |

  - 关节角几乎相同（train：0.944 / 0.779 / 0.504 / −0.03 / −1.03）。3 帧平均速度在每个步长上都略差（下一帧 0.864），因为平滑引入了滞后。
- **速度自相关**（末端位置，train / test）：
  - 滞后 1 帧 0.978 / 0.984，4 帧 0.763 / 0.819，8 帧 0.432 / 0.542，16 帧 0.164 / 0.247，32 帧约 0，48–96 帧弱负（−0.04 至 −0.09）。
- **夹爪状态保持概率**（train）：1 帧 0.992，16 帧 0.882，32 帧 0.778，96 帧 0.561。
- 静止帧（与上一帧完全相同）只占 0.5% / 0.1%，所以下一帧的高 R² 不是重复帧造成的。
- **结论**：
  1. 相邻帧高度冗余：当前速度能解释下一帧位移的 96%。
  2. 这种冗余衰减得很快：0.5 s 时剩约一半；到 1 s，恒速外推已不比"预测平均位移"好（R² ≈ 0）；2 s 以后明显更差。速度自相关在约 2 s 处降到 0。
  3. 因此 1–2 s 尺度的运动信息不在"当前状态 + 速度"里。若学到的先验在这个尺度上仍能预测（DV1），信息只能来自更长的时间结构。这正好给"信息在长程时间结构中"提供了与模型无关的统计依据。
  4. test 站点的运动比 train 更平滑（各项 R² 更高），正文用 train 的数字，test 列为复制。
- s0 expert_droid（Jubail 18955126，cn011，2.5 h）：训练、评测均 exit 0；commit 5e3acf3（与 0130b58 只差日志文本）；k=5、seed 0、prior p0、samples 13186 与另两臂一致；850 episode。成功率 **64.6**（DGX 原 s0 为 64.0）。
  - Jubail s0 三臂：none 60.1 / scratch 61.9 / droid 64.6。droid vs none 115:77（p=0.0074），droid vs scratch 111:88（p=0.12）。
  - s1 三臂排在 DV 作业之后；3 seed 齐后按 `configs/smolvla_jubail_manifest.json` 计算。

## 【数据验证】DV1 / DV3 结果（2026-10-27，Jubail 18955489，cn004，1.0 h，全部 exit 0，commit 5c048fa）
- **核验**：
  - 训练：dv1_ctx4 / dv1_ctx16 训满 40k 步，seed 0，配方与主先验相同（S、ar_flow、H=32、100% DROID），只改 ctx_len；train 83,606 条，val 3,961 条。
  - 评测集：test 固定锚点 59,540 个窗口，来自 6,518 条 episode（共 7,164 条，其余太短）；val 35,389 / 3,201；384 帧锚点集 21,067 / 1,974。
  - 各次运行的基线完全一致（const-vel ES 8.582）。
  - DV3 的检查项通过：endpoint ES 1.64，远低于不给目标的 3.77。
- 分析：`scripts/dv_analyze.py`，在本地运行，episode 整簇 bootstrap 2000 次；结果文件 `dv_summary.json`。
  - 站点按 site 连通分量标注。直接用楼名正则匹配会漏掉同一站点下的其他楼名，已改正。

**DV1 历史长度 —— 预注册主检验通过**
- test（96 帧锚点）未来 32 帧（2.1 s）末端轨迹的 ES（cm）：

| ctx | 4（0.27 s） | 16（1.07 s） | 48（3.2 s） | 96（6.4 s，p0） | 96（pB1） | 96（pB2） | const-vel |
|---|---|---|---|---|---|---|---|
| ES | 4.327 | 4.040 | 3.827 | **3.737** | 3.756 | 3.736 | 8.582 |
| 相对 ctx96 | **+15.8% [15.4, 16.2]** | +8.1% [7.8, 8.4] | +2.4% [2.2, 2.6] | — | +0.5% | 0.0% | |

- **主检验**：ES(ctx4) − ES(ctx96) = **+0.590 [0.577, 0.604] cm → 通过**。
- **相邻长度差**都显著且递减：4→16 −0.287，16→48 −0.213，48→96 −0.090。
  - 三个 ctx96 实例之间的极差只有 0.021 [0.017, 0.025]，所以 48→96 的 0.09 也大于实例噪声。
- **分站点**（ctx4 相对 ctx96）：UT Austin +20.1%，Edinburgh +13.0%，Penn +12.8%，三个站点都显著。
- **val 复制**：+12.7% [12.2, 13.2]，单调性相同。
- **384 帧锚点集**：
  - ctx4 +16.7%，ctx16 +8.8%，ctx48 +2.6%；
  - ctx192 −1.6%，ctx384 −2.0%（相对 ctx96，CI 都不含 0）。
  - 结论：历史越长越好，但收益到 6.4 s 后已很小。
- **分步长**（ES 比值 ctx4 / ctx96）：第 1 步 1.40，第 4 步 1.31，第 8 步 1.24，第 16 步 1.18，第 32 步 1.14。
  - 近端相对差最大（绝对值小）；整个 2.1 s 窗口内都显著大于 1。
  - 次要发现：ctx48 在第 1 步反而比 ctx96 好 8%，第 8 步以后差 2.5%。
- **结论与措辞**：在 DROID 留出实验室上，只看近乎瞬时的状态（0.27 s，含速度），未来 2.1 s 末端轨迹的能量分数比用 6.4 s 历史高 15.8%（val 12.7%，三个站点 12.8–20.1%）。
  - 信息主要来自最近约 3 s（ctx48 只差 2.4%），更长的历史仍有小而稳定的增益。
  - 与 DV2 合起来：当前速度在约 1 s 后已不含信息，但历史仍能显著改善 1–2 s 的预测，说明信息在更长的时间结构里。

**DV3 "怎么动 / 去哪"分解（描述性，test）**
- gc_ar_flow_ft 逐步 ES（cm），依次为第 1 / 4 / 8 / 16 / 24 / 32 步：

| 条件 | 第 1 步 | 第 4 步 | 第 8 步 | 第 16 步 | 第 24 步 | 第 32 步 |
|---|---|---|---|---|---|---|
| 不给目标 | 0.10 | 0.68 | 1.54 | 3.08 | 4.70 | 6.27 |
| keyframe | 0.10 | 0.66 | 1.47 | 2.77 | 4.02 | 5.22 |
| endpoint | 0.10 | 0.65 | 1.38 | 2.02 | 1.80 | 1.58 |
| 直线插值到终点 | 0.45 | 1.67 | 2.89 | 3.86 | 2.87 | 0 |

- **"去哪"份额**（1 − ES(给目标) / ES(不给)）：
  - endpoint：第 4 步 4%，第 8 步 11%，**中段第 16 步 34%**，第 24 步 62%，第 32 步 75%（末步是平凡情形）；
  - keyframe：2% / 4% / 10% / 15% / 17%。
- **复制**：gc_ar_flow_scratch 几乎相同（endpoint 中段 35%，keyframe 第 32 步 18%）。ft 的无目标 ES 略好（3.77 vs 3.86）。
- **解读**：
  - 0.5 s 以内的不确定性几乎全部是"怎么动"，知道终点也帮不上忙。
  - 到 1 s，知道终点能消除约三分之一；即使告诉模型 2.1 s 时的位置，中段仍有约 2/3 的不确定性来自运动方式。
  - 只知道下一个夹爪事件的位置（时间未知），在整个窗口内最多能消除 17%。
  - 这与 E9a / E7b 的失败归类一致：SOPT 改善的是执行（"怎么动"），不是"去哪"。

## SmolVLA `_jr` 重跑进度：s1 none（2026-10-27，Jubail 18955945，cn268，2.1 h）
- 训练 20k 步 exit 0（1.5 h），评测 n_exec 10 exit 0（31 min）。核验：run 名 `e23_smolvla_none_k5_jr_t17_s1`，commit fe35765（与 s0 的 0130b58 相比只多了 DV 脚本和 `forecast.py` 的评测曲线，不涉及 VLA 训练 / 评测代码），k=5，suite libero90_17，seed 1，prior p0，samples 13186，85 条示范；850 episode = 17 任务 × 50。
- 成功率 **57.8**（DGX 原 s1 为 60.1，跨平台 −2.3）。s1 expert_scratch（18955948）运行中，s1 expert_droid（18960640）排队。

## 剩余多卡排队作业全部改为单卡（2026-10-27 16:10，用户批准）
- 队列里 39 个多卡作业（2–3 卡，DP / SmolVLA 的 E39 示范数扫描 k1 / k2 / k10 / k25 / kall、E40 `_hpre`、E38 SmolVLA / FastWAM 的 pB1 / pB2）均未提交，拆成 90 个单卡作业（`SBATCH_GRES=gpu:a100:1 SBATCH_MEM_PER_NODE=120G`），队列共 104 行，原有顺序不变（E40 主机预训练仍在 `_hpre` 之前）。
- 拆分方式：`ARMS=a,b,c` → 每臂一行 `ARM=a`；SmolVLA 的 `PSET` → `ARM=expert_droid PRIOR=<ckpt> PTAG=_pB1/_pB2`（脚本已支持）；FastWAM 的 `PSET` → 每行只含一个先验实例。run 名、训练与评测配方都不变（run 名由 arm / K / TAG / suite / PTAG / seed 决定，与是否同作业无关）。
- 代价：最多 2 个并行作业的限制下，每个作业只占 1 卡，总 GPU 时不变但墙钟更长；好处是不再出现多卡作业排队数天的情况（SmolVLA `_jr` 拆分后均在数小时内开始）。
- 16:15 更新：用户允许单卡作业同时最多 4 个（多卡仍为 2 个）。队列 runner 上限改为 4，第一个新作业 E39 DP k1 s0 none 已提交（18961064）。

## SmolVLA `_jr` 重跑进度：s1 expert_scratch（2026-10-27，Jubail 18955948，cn259，2.6 h）
- 训练 exit 0（2.0 h），评测 n_exec 10 exit 0（39 min）。核验：run 名、commit fe35765、k=5、seed 1、prior p0、samples 13186 与 s1 none 一致，85 条示范与 none 臂完全相同；850 episode，与 none 臂的 (task, init) 一一对应。
- 成功率 **58.9**（DGX 原 s1 为 58.5，跨平台 +0.4）。与 Jubail s1 none 配对：83:73，p=0.47。
- s1 expert_droid（18960640）16:39 开始运行。

## E39 进度：DP k1 s0 none（2026-10-27，Jubail 18961064，cn259，1.3 h）
- 训练 30k 步 exit 0（51 min），评测 n_exec 10 exit 0（27 min）。核验：run 名 `e21_dp_none_k1_t17_s0`，commit b2b2169，k=1（17 条示范，samples 2697），suite libero90_17，seed 0；850 episode。
- 成功率 **48.4**（k5 none 3 seed 均值 72.9；k1 的示范只有 k5 的 1/5）。同 seed 的 scratch / droid 臂（18961094 / 18961518）运行中；单臂结果不做比较，等 E39 预注册的整组。
- DP k1 s0 expert_scratch（Jubail 18961094，cn268，1.3 h）：训练、评测 exit 0；run 名、commit b2b2169、k=1、seed 0、samples 2697 与 none 臂一致，17 条示范完全相同；850 episode，(task, init) 与 none 一一对应。成功率 **47.2**；对 none 配对 77:87（p=0.48）。droid 臂（18961518）运行中。
- DP k1 s0 expert_droid（Jubail 18961518，cn006，1.4 h）：训练、评测 exit 0；run 名、commit、k=1、seed 0、samples 2697 与另两臂一致，17 条示范相同；850 episode 一一对应。成功率 **46.8**。
  - k1 s0 三臂：none 48.4 / scratch 47.2 / droid 46.8。droid vs none 85:98（p=0.38），droid vs scratch 89:92（p=0.88）。
  - 单 seed 下 k1 没有看到增益（对照：k5 时 DP 3 seed 均值为 none 72.9、droid 77.5）。按预注册不做中途判断，等 s1 / s2 与 k2 后计算 P1 / P2。

## SmolVLA `_jr` 重跑完成：s1 expert_droid + 主表 3 seed 统计（2026-10-27，Jubail 18960640，cn256，2.6 h）
- s1 expert_droid：训练、评测 exit 0；run 名、commit b2b2169（相对 fe35765 只多文档提交）、k=5、seed 1、prior p0、samples 13186，85 条示范与 none / scratch 相同；850 episode 一一对应。成功率 **66.9**（DGX 原 s1 为 67.4）。s1 配对：droid vs none 147:69（p=1.2e-7），droid vs scratch 132:64（p=1.3e-6）。
- s2 三臂（原 Jubail 运行，commit 479534a / e76fb6e / 18ebc60）核验：meta 的 k / steps / bs / lr / prior / suite / samples 与 `_jr` s0 完全一致，85 条示范、850 episode。
- `stats_all.py --manifest configs/smolvla_jubail_manifest.json`（3 seed，Jubail）：

| arm | s0 / s1 / s2 | 均值 ± sd |
|---|---|---|
| none | 60.1 / 57.8 / 64.2 | 60.7 ± 3.3 |
| expert_scratch | 61.9 / 58.9 / 62.2 | 61.0 ± 1.8 |
| expert_droid | 64.6 / 66.9 / 67.8 | 66.4 ± 1.6 |

- 主表口径（`configs/paper_main_manifest.json`：原主表 manifest 末尾加入 SmolVLA Jubail，Holm 对 13 个主比较；SmolVLA 放在末尾，使其它宿主的 bootstrap CI 与原表逐位一致）：
  - droid − none **+5.7 [+3.1, +8.2]**，362:216，McNemar 1.3e-9，Holm 1.5e-8；
  - droid − scratch **+5.4 [+2.8, +8.0]**，351:213，6.8e-9，Holm 6.8e-8；
  - scratch − none +0.3 [−1.8, +2.6]，p=0.74。
  - 并入后其它宿主 Holm 只有两处变化：FastWAM droid−none 3e-23 → 3.5e-23，DP droid−scratch 1.1e-11 → 1.3e-11。
- 跨平台复制（原 DGX s0 / s1 + s2）：+4.8（p=3.7e-7）、droid − scratch +6.6。逐 seed 跨平台差：none −0.5 / −2.3，scratch +3.2 / +0.4，droid +0.6 / −0.5。结论不变：SmolVLA 上 SOPT 显著提升，同结构 scratch 无增益。
- 已更新：表 T1 SmolVLA 行与 Holm 注、T10，大纲 5.1 与摘要数字（+4.8 → +5.7）。
- DP k1 s1 none（Jubail 18962935，1.3 h）：训练、评测 exit 0；run 名、k=1、seed 1、samples 2697，commit cbbcb2b（相对 b2b2169 只多文档提交）；850 episode。成功率 **46.0**（s0 为 48.4）。
  - 核对：示范集与 s0 相同、且是 k5 示范集的子集（`select_episodes` 按加盐哈希取前 k 条，不随 seed 变；seed 只改初始化和训练随机性），与主表 k5 的协议一致，符合预注册"嵌套子集"。
- DP k1 s1 expert_scratch（Jubail 18963109，1.4 h）：训练、评测 exit 0；run 名、k=1、seed 1、samples 2697 与 none 一致，示范相同；850 episode 一一对应；commit b4ff68a（只多文档提交）。成功率 **40.7**；对 none（46.0）配对 56:101（p=4e-4）。s1 droid（18963657）运行中。
- DP k1 s1 expert_droid（Jubail 18963657，1.4 h）：训练、评测 exit 0；run 名、k=1、seed 1、samples 2697 与另两臂一致，示范相同；850 episode 一一对应；commit 065db0b（只多文档提交）。成功率 **46.1**。
  - k1 s1 三臂：none 46.0 / scratch 40.7 / droid 46.1。droid vs none 84:83（p=1），droid vs scratch 101:55（p=3e-4）。
  - k1 s0+s1 合并（描述性，非预注册检验）：droid vs none 169:181（p=0.56），droid vs scratch 190:147（p=0.022）。即 k1 下 SOPT 相对 none 无增益，但抵消了 scratch 组件的代价。主检验等 s2 与 k2 后按 `configs/e39_contrasts.json` 计算。
- DP k1 s2 none（Jubail 18964007，1.3 h）：训练、评测 exit 0；run 名、k=1、seed 2、samples 2697，示范与 s0 / s1 相同；850 episode；commit 6159595（只多文档提交）。成功率 **48.0**（k1 none 3 seed：48.4 / 46.0 / 48.0）。s2 scratch / droid（18964093 / 18964305）排队中。
- DP k1 s2 expert_scratch（Jubail 18964093，1.4 h）：训练、评测 exit 0；run 名、k=1、seed 2、samples 2697，示范与 none 相同；850 episode 一一对应。成功率 **49.6**；对 none（48.0）配对 88:74（p=0.31）。k1 scratch 3 seed：47.2 / 40.7 / 49.6。s2 droid（18964305）排队中。
- 运维：22:29 起 Jubail 登录节点连不上 github.com（runner 的 `git pull` 失败，sbatch 照常提交）。Jubail 仓库停在 6159595；此后的提交只有文档，作业所用代码不受影响。若要改代码，需等 GitHub 连通后再提交依赖新代码的作业。
- DP k2 s0 none（Jubail 18964641，1.3 h）：训练、评测 exit 0；run 名 `e21_dp_none_k2_t17_s0`，k=2（34 条示范，samples 5367，k1 的 17 条为其子集），seed 0；850 episode。成功率 **64.7**（k1 48.4，k5 73.6，同 seed）。
- DP k1 s2 expert_droid（Jubail 18964305，1.4 h；23:33 结束时 watcher 未及时重启，00:20 检查补处理）：训练、评测 exit 0；k=1、seed 2、samples 2697、prior p0，示范与另两臂相同；850 episode 一一对应。成功率 **48.2**；对 none（48.0）76:74（p=0.94），对 scratch（49.6）79:91（p=0.40）。
  - **DP k1 三臂 3 seed 已齐**：none 48.4 / 46.0 / 48.0（47.5），scratch 47.2 / 40.7 / 49.6（45.8），droid 46.8 / 46.1 / 48.2（47.0）。合并配对（描述性）：droid vs none 245:255（p=0.69），droid vs scratch 269:238（p=0.18）。
- DP k2 s0 expert_droid（Jubail 18968787，1.4 h）：训练、评测 exit 0；k=2、seed 0、samples 5367、prior p0，示范与 none 相同；850 episode 一一对应。成功率 **58.4**；对 none（64.7）82:136（**p=3e-4，SOPT 更差**）。
  - 排查：两臂训练正常（loss 曲线平滑、无发散；droid 前期 loss 下降更快：10k 步 0.0128 vs 0.0212）。逐任务看 17 个任务中 droid 更好 7 个、更差 10 个，最大降幅在个别任务（−32、−46 点），不是全局性故障。暂按真实结果记录；s1 / s2 出来后再看是否稳定。
  - 当前 DP 曲线（droid − none，点）：k1 −0.5（3 seed）、k2 −6.3（s0）、k5 +4.6（3 seed）。与预注册假设"低数据收益最大"方向相反；主检验 P1 / P2 等 k2 3 seed 与 SmolVLA 部分后按 `configs/e39_contrasts.json` 计算。
- DP k2 s1 none（Jubail 18970115，1.3 h）：训练、评测 exit 0；k=2、seed 1、samples 5367，示范与 s0 相同；850 episode。成功率 **57.9**（s0 64.7）。s1 droid（18970628）、s2 none / droid（18970747 / 18971599）运行中。
- E40 宿主预训练（Jubail 18972066，cn002，00:58 开始，hsteps 100k）：smoke exit 0；init-host 冒烟检查加载 398 个张量、新建 127 个（sopt.*），正常。
- DP k2 s1 expert_droid（Jubail 18970628，1.4 h）：训练、评测 exit 0；k=2、seed 1、samples 5367、prior p0，示范与 none 相同；850 episode 一一对应。成功率 **60.4**；对 none（57.9）106:85（p=0.15）。即 k2 上 s0 为 −6.3、s1 为 +2.5，seed 间方向不一致。
- DP k2 s2 none（Jubail 18970747，1.3 h）：训练、评测 exit 0；k=2、seed 2、samples 5367，示范相同；850 episode。成功率 **59.2**（k2 none 3 seed：64.7 / 57.9 / 59.2）。s2 droid（18971599）运行中。
- DP k2 s2 expert_droid（Jubail 18971599，1.4 h）：训练、评测 exit 0；k=2、seed 2、samples 5367、prior p0，示范相同；850 episode 一一对应。成功率 **59.2**；对 none（59.2）95:95（p=1）。

### E39 中期：DP k1 / k2 已齐（2026-10-28 02:05；`contrast_boot.py --spec configs/e39_contrasts.json`，(seed, task) 簇 bootstrap）
| k | none（3 seed） | droid（3 seed） | 增益 droid − none | 95% CI | p |
|---|---|---|---|---|---|
| 1 | 48.4 / 46.0 / 48.0 | 46.8 / 46.1 / 48.2 | −0.4 | [−4.0, +3.3] | 0.84 |
| 2 | 64.7 / 57.9 / 59.2 | 58.4 / 60.4 / 59.2 | −1.3 | [−5.5, +2.7] | 0.55 |
| 5（主表） | 73.6 / 71.4 / 73.5 | 76.1 / 77.3 / 79.1 | +4.6 | [+1.8, +7.5] | 0.0018 |
- 次要：DP 低数据增益（k1、k2 平均）−0.8 [−4.0, +2.2]（p=0.59）；k1 预训练效应（droid − scratch）+1.2 [−3.0, +5.5]（p=0.56）。
- 自检：同一脚本给出的 DP k5 增益与主表 stats_all 完全一致（+4.6 [+1.8, +7.5]）。
- 主检验 P1（DP 与 SmolVLA 平均）和 P2 需要 SmolVLA k1 / k2 与 k25 / 全部，尚未计算，不做 Holm。
- 解读（暂定）：DP 上 SOPT 的增益在每任务 1–2 条示范时消失，5 条时出现。若 SmolVLA 同样如此，P1 将不通过，按预注册改写为"增益需要最低数据量"，论点 A 以宿主类型为主轴。
- DP k10 s0 none（Jubail 18972375，1.2 h）：训练、评测 exit 0；k=10（170 条示范，samples 26462，k2 的示范为其子集），seed 0；850 episode。成功率 **82.0**（同 seed：k1 48.4 / k2 64.7 / k5 73.6）。s0 droid（18972439）、s1 none（18972546）运行中。
- DP k10 s0 expert_droid（Jubail 18972439，1.3 h）：训练、评测 exit 0；k=10、seed 0、samples 26462、prior p0，示范与 none 相同；850 episode 一一对应。成功率 **87.3**；对 none（82.0）93:48（p=2e-4，+5.3）。s0 上 DP 增益：k1 −1.6 / k2 −6.3 / k5 +2.5 / k10 +5.3。
- DP k10 s1 none（Jubail 18972546，1.3 h）：训练、评测 exit 0；k=10、seed 1、samples 26462，示范与 s0 相同；850 episode。成功率 **79.9**。s1 droid（18973032）、s2 none / droid（18973120 / 18973265）运行中。

## E40 阶段 1 完成：DP 宿主动作预训练（2026-10-28，Jubail 18972066，cn002，3.0 h）
- smoke（预训练 → 从宿主初始化接 expert_droid → 1 init 评测）exit 0；init-host 加载 398 个张量，新建 127 个（sopt.*）。
- 宿主预训练 exit 0：`e40_dp_host_train`，suite libero90_train（LIBERO-90 train 划分 52 个任务的全部示范，2725 条，samples 388710），task_vocab all，100k 步，bs 64，seed 0，commit 6159595。
- 防泄漏：`vla_train.py` 断言 train 划分任务与 17 个评测任务及 E4b 任务无交集，断言通过（作业未报错）。
- loss：0.97（第 100 步）→ 0.031（50k）→ 0.026（100k），平稳收敛。
- 阶段 2（`_hpre`，k5，none / expert_droid × seed 0–2）在队列中按原顺序排在 E39 DP k25 / 全部、E37s、E18r、E41 之后。
- DP k10 s1 expert_droid（Jubail 18973032，1.3 h）：训练、评测 exit 0；k=10、seed 1、samples 26462、prior p0，示范相同；850 episode 一一对应。成功率 **87.4**；对 none（79.9）104:40（p<1e-6，+7.5）。
- DP k10 s2 none（Jubail 18973120，1.3 h）：训练、评测 exit 0；k=10、seed 2，示范相同；850 episode。成功率 **82.6**（k10 none 3 seed：82.0 / 79.9 / 82.6）。s2 droid（18973265）运行中。
- DP k10 s2 expert_droid（Jubail 18973265，1.3 h）：训练、评测 exit 0；k=10、seed 2、samples 26462、prior p0，示范相同；850 episode 一一对应。成功率 **86.0**；对 none（82.6）79:50（p=0.013）。
  - **DP k10 3 seed 已齐**：none 82.0 / 79.9 / 82.6，droid 87.3 / 87.4 / 86.0。`contrast_boot.py`：增益 **+5.4 [+3.0, +7.8]**（p=1e-4）。
  - DP 增益曲线（3 seed，簇 bootstrap）：k1 −0.4 [−4.0, +3.3]，k2 −1.3 [−5.5, +2.7]，k5 +4.6 [+1.8, +7.5]，k10 +5.4 [+3.0, +7.8]。增益在 k ≥ 5 出现，k ≤ 2 时为 0。（加入 k10 后 bootstrap 的随机流变化，k1 预训练效应 CI 末位变为 [−3.1, +5.5]，低数据增益 [−3.9, +2.2]。）
- DP k25 s0 expert_droid（Jubail 18973867，1.3 h）：训练、评测 exit 0；k=25（425 条示范，samples 65725，k10 示范为其子集），seed 0、prior p0；850 episode。成功率 **94.6**。同 seed none（18973789）仍在训练，配对待其完成。
- DP k25 s1 none（Jubail 18973872，1.2 h）：训练、评测 exit 0；k=25、seed 1，示范与 s0 相同；850 episode。成功率 **89.8**。s1 droid（18974001）评测中。
- DP k25 s1 expert_droid（Jubail 18974001，cn258，1.2 h）：训练、评测 exit 0；k=25、seed 1、samples 65725、prior p0，示范与 none 相同；850 episode 一一对应。成功率 **94.1**；对 none（89.8）72:35（p=4.5e-4，+4.3）。
- **运维：取消并重提 DP k25 s0 none**（18973789，cn256）。该作业 03:57 开始，1h53m 只训练到 10k / 30k 步：日志呈"几百步 → 停 5–13 分钟"的阵发模式（瞬时速度 10 it/s 正常），同时段其它 k25 作业在别的节点上 55 min 跑完 30k 步，判断为 cn256 的 I/O 问题。按当前速度还需约 10 h。
  - 05:50 scancel；cancelled 作业的 `metrics.jsonl` 与训练日志改名为 `*.cancelled_18973789.*`（JsonlLogger 是追加写，避免与重跑混在一起）；输出目录无 final.pt，重跑从头训练，配方与 run 名不变。
  - 重提行放在队首，加 `SBATCH_EXCLUDE=cn014,cn015,cn270,cn256`。
  - 更正：`SBATCH_EXCLUDE` 环境变量未覆盖脚本内的 `#SBATCH --exclude`，第一次重提（18974314）仍落在 cn256，12 s 后取消（未写出 metrics；日志会被重跑覆盖）。改为在 sbatch 命令行传 `--exclude=cn014,cn015,cn270,cn256`，重提为 **18974318**（cn008，ExcNodeList 已含 cn256）。
- DP k25 s2 none（Jubail 18974264，1.2 h）/ expert_droid（18974267，1.2 h）：训练、评测均 exit 0；k=25、seed 2、samples 65725、prior p0，示范与 s1 相同；850 episode 一一对应。none **91.2**，droid **93.6**；配对 58:37（p=0.040，+2.4）。k25 只差 s0 none（18974318 重跑中）。
- DP k25 s0 none 重跑（Jubail 18974318，cn008，1.2 h）：训练、评测 exit 0；k=25、seed 0、samples 65725，示范与 s1 / s2 相同；850 episode；`metrics.jsonl` 300 行（第 100 → 30000 步，未与取消的那次混合）。成功率 **92.7**；droid（94.6）对 none 45:29（p=0.081，+1.9）。
  - **DP k25 3 seed 已齐**：none 92.7 / 89.8 / 91.2，droid 94.6 / 94.1 / 93.6。`contrast_boot.py`：增益 **+2.9 [+1.1, +4.7]**（p=2e-4）。
  - DP 增益曲线（3 seed）：k1 −0.4，k2 −1.3，k5 +4.6，k10 +5.4，k25 +2.9（k25 接近天花板，none 已 91%）。
- DP kall s0 none（Jubail 18974313，cn005，1.2 h）：训练、评测 exit 0；k=全部（771 条示范，samples 117739，k25 示范为其子集），seed 0；850 episode。成功率 **93.1**。
- DP kall s0 expert_scratch（Jubail 18974482，cn259，1.2 h）：训练、评测 exit 0；k=全部、seed 0、samples 117739，示范与 none 相同；850 episode 一一对应。成功率 **95.2**；对 none（93.1）50:32（p=0.060）。
- DP kall s0 expert_droid（Jubail 18974502，cn258，1.2 h）：训练、评测 exit 0；k=全部、seed 0、samples 117739、prior p0，示范相同；850 episode 一一对应。成功率 **94.6**。kall s0 三臂：none 93.1 / scratch 95.2 / droid 94.6；droid vs none 53:40（p=0.21），droid vs scratch 33:38（p=0.64）。
- DP kall s1 none（Jubail 18974530，cn003，1.2 h）：训练、评测 exit 0；k=全部、seed 1、samples 117739，示范与 s0 相同；850 episode。成功率 **93.9**。
- DP kall s1 expert_scratch（Jubail 18974535，cn004，1.3 h）：训练、评测 exit 0；k=全部、seed 1、samples 117739，示范与 none 相同；850 episode 一一对应。成功率 **94.8**；对 none（93.9）46:38（p=0.45）。
- DP kall s1 expert_droid（Jubail 18974709，cn004，1.3 h）：训练、评测 exit 0；k=全部、seed 1、samples 117739、prior p0，示范相同；850 episode 一一对应。成功率 **94.8**；对 none（93.9）42:34（p=0.42），对 scratch（94.8）36:36（p=1）。
- 队列推进到 E37s：DP s0 scratch（`MODE=main HOST=dp ARM=scratch`，18975053）09:29 开始。
- DP kall s2 none（18974816）/ expert_scratch（18974819）：训练、评测 exit 0；k=全部、seed 2、samples 117739，示范相同；850 episode 一一对应。none **94.4**，scratch **94.5**（35:34，p=1）。s2 droid（18974849）运行中，完成后 E39 DP 部分齐。
- DP kall s2 expert_droid（Jubail 18974849，cn004，1.3 h）：训练、评测 exit 0；k=全部、seed 2、samples 117739、prior p0，示范相同；850 episode 一一对应。成功率 **96.4**；对 none（94.4）42:25（p=0.050），对 scratch（94.5）41:25（p=0.064）。

### E39 DP 部分完成（2026-10-28 09:51；`contrast_boot.py --spec configs/e39_contrasts.json`，(seed, task) 簇 bootstrap，3 seed）
| k（每任务示范） | none（s0 / s1 / s2） | droid（s0 / s1 / s2） | 增益 droid − none | 95% CI | p |
|---|---|---|---|---|---|
| 1 | 48.4 / 46.0 / 48.0 | 46.8 / 46.1 / 48.2 | −0.4 | [−4.0, +3.3] | 0.85 |
| 2 | 64.7 / 57.9 / 59.2 | 58.4 / 60.4 / 59.2 | −1.3 | [−5.5, +2.7] | 0.54 |
| 5 | 73.6 / 71.4 / 73.5 | 76.1 / 77.3 / 79.1 | **+4.6** | [+1.7, +7.4] | 0.0018 |
| 10 | 82.0 / 79.9 / 82.6 | 87.3 / 87.4 / 86.0 | **+5.4** | [+3.0, +7.8] | 1e-4 |
| 25 | 92.7 / 89.8 / 91.2 | 94.6 / 94.1 / 93.6 | **+2.9** | [+1.1, +4.8] | 0.002 |
| 全部（约 45） | 93.1 / 93.9 / 94.4 | 94.6 / 94.8 / 96.4 | +1.5 | [−0.2, +3.2] | 0.081 |
- 预训练效应（droid − scratch）：k1 +1.2 [−3.0, +5.6]（p=0.56），全部 +0.4 [−0.7, +1.6]（p=0.47）。scratch 在 k1 为 45.8，全部为 94.8 / 94.8 / 94.5。
- DP 的两个预注册分量：低数据增益（k1、k2 平均）**−0.8 [−3.9, +2.3]**（p=0.59）；增益下降（k1、k2、k5 平均 − k25、全部平均）**−1.2 [−3.5, +1.1]**（p=0.31）。DP 上两者方向都与假设相反或为 0。
- 注：k5 的 CI 与主表（stats_all，[+1.8, +7.5]）末位不同，来自两个脚本不同的 bootstrap 随机流；点估计一致。论文 k5 引主表口径。
- **解读**：DP 上 SOPT 增益对示范数呈倒 U 形——k ≤ 2 时为 0，k5–k10 最大（+4.6 / +5.4），k ≥ 25 随天花板回落。"低数据收益最大"在 DP 上不成立；更准确的说法是"SOPT 需要每任务约 5 条以上示范才能被宿主利用，收益在中等数据量最大"。P1 / P2 的最终判定（两宿主平均，Holm）待 SmolVLA 部分（队列中 36 个作业）。

## E37s 进度（2026-10-28）
- SmolVLA-goal s1 scratch（Jubail 18975172，cn257，2.0 h）：5 个扰动评测均 exit 0，各 50 init × 10 任务 = 500 episode。成功率 camera 2 **79.6** / light 3 83.6 / texture 2 72.8 / noise 2 87.2 / robot 2 69.0。其余 3 个作业（DP s0 / s1、SmolVLA-goal s0）各完成 3 项，运行中。4 个齐后按预注册跑 `contrast_boot.py --spec configs/e37s_contrasts.json`。
- E18r f1（`e25_trunk.sbatch VARIANT=f1`，18977035）11:58 开始。
- SmolVLA-goal s0 scratch（Jubail 18975131，cn256，2.5 h；cn256 上评测作业未受 I/O 问题影响）：5 个扰动均 exit 0，各 500 episode。camera 2 **76.2** / light 3 83.0 / texture 2 71.0 / noise 2 82.6 / robot 2 67.8。
- DP s0 scratch（Jubail 18975053，cn253，2.9 h）：5 个扰动均 exit 0，各 850 episode。camera 2 **31.3** / light 3 66.0 / texture 2 41.6 / noise 2 29.3 / robot 2 56.1。只剩 DP s1（18975056，最后一项 robot 进行中）。
- DP s1 scratch（Jubail 18975056，cn255，2.9 h）：5 个扰动均 exit 0，各 850 episode。camera 2 **37.4** / light 3 65.5 / texture 2 39.1 / noise 2 23.9 / robot 2 52.8。

### E37s 结果（2026-10-28 12:45；`contrast_boot.py --spec configs/e37s_contrasts.json`，(seed, task) 簇 bootstrap，宿主分层，seed 0–1）
输入：12 个 run × (干净 `rollouts_n50.json` + 5 个 `e37_main_*.json`)，共 72 个文件；DP 每文件 850 episode，SmolVLA-goal 500 episode，核对 perturb 字段与档位（camera 2 / light 3 / texture 2 / noise 2 / robot 2）。

成功率（2 seed 平均）：
| 宿主 | 臂 | 干净 | camera | light | texture | noise | robot |
|---|---|---|---|---|---|---|---|
| DP | none | 72.5 | 37.4 | 67.3 | 39.5 | 30.8 | 55.7 |
| DP | scratch | 68.7 | 34.4 | 65.8 | 40.4 | 26.6 | 54.5 |
| DP | droid | 76.7 | 51.1 | 71.6 | 46.6 | 42.8 | 65.0 |
| SmolVLA-goal | none | 77.7 | 60.2 | 77.6 | 67.7 | 76.8 | 63.4 |
| SmolVLA-goal | scratch | 78.0 | 77.9 | 83.3 | 71.9 | 84.9 | 68.4 |
| SmolVLA-goal | droid | 81.0 | 68.4 | 83.0 | 73.6 | 81.4 | 66.4 |

主检验（Holm，2 个）：
- **P1** camera 交互（droid − scratch）两宿主平均：**−1.9 [−7.8, +3.9]**，p=0.52，Holm 0.81 → 不通过。
- **P2** 视觉 4 项平均：**−1.3 [−4.5, +1.8]**，p=0.41，Holm 0.81 → 不通过。
次要：
- scratch vs none 交互（"只有历史"）：camera **+9.1 [+3.1, +15.4]**（p=0.003），视觉 4 项 +5.2 [+2.0, +8.6]。
- droid vs none 交互（参照，限定两宿主）：camera +7.2 [+0.5, +14.5]，视觉 4 项 +3.9 [+0.6, +7.5]，与预注册时的参照值一致。
- **宿主间方向相反**（camera）：
  - DP：droid − scratch **+8.7 [+1.6, +15.7]**，scratch − none +0.8 [−4.2, +6.0] → DP 上相机鲁棒性来自预训练，scratch 历史组件没有带来鲁棒性。noise 同样（+8.2 [+0.1, +17.1]）。
  - SmolVLA-goal：droid − scratch **−12.5 [−22.2, −3.3]**，scratch − none **+17.4 [+6.6, +28.7]** → 鲁棒性来自历史组件本身，scratch 甚至比 DROID 先验更鲁棒（camera 77.9 vs 68.4）；light / robot 同方向。
**按预注册判读**：P1 不通过，scratch vs none 的 camera 交互 CI > 0 → 规则 2："本体历史带来视角鲁棒性，预训练的额外贡献未被证实"，论点 B 按此改写。
- 必须同时报告宿主异质性：DP 上预训练是必要的（scratch 无鲁棒性），SmolVLA-goal 上历史组件本身就够。合并估计接近 0 是两个方向相反的宿主效应相抵，不是"两宿主上都没有预训练效应"。
- 功效与口径：SmolVLA-goal 只有 10 任务 × 2 seed = 20 簇，CI 宽；它用的是 libero_goal，不是 LIBERO-90-17。GR00T 没有 scratch 臂（无 17 任务 xattn_scratch checkpoint），不在本检验中。

## E18r 进度：libero 变体（2026-10-28，Jubail 18977241，cn009，1.3 h）
- 先验 `outputs/e25j_libero`：LIBERO-90 train 划分状态，E4a scratch 配方（lr 3e-4，warmup 300，3k 步），`data.holdout_train_tasks=0.1`（按 train 划分内部留出任务选模，不触碰 17 个评测任务；config.yaml 已核对）。预训练 exit 0；第 3000 步留出任务上 pos ADE 10.68 cm（常速 16.17）。
- TrunkBC 17 任务 × k5 × seed 0–2：训练、评测均 exit 0；init = `outputs/e25j_libero/best.pt`，示范与 `e25j_trunk_ref` 相同，各 850 episode。成功率 **70.1 / 73.3 / 70.4**（均值 71.3）。对照：scratch 68.7 / 67.6 / 67.8（68.0），ref（DROID 100%）74.6 / 71.5 / 74.2（73.4）。
- E24 探针 exit 0（droid_ee / libero / droid_mt / libero_mt 四组线性探针，结果在作业日志）。
- commit ed6512c：Jubail 已能连上 GitHub（12:33 的作业 `git pull` 成功）。
- f1 / f10 运行中；3 个变体齐后按 `configs/e18r_manifest.json` 跑 `stats_all.py`。
- E18r f1（Jubail 18977035，cn002，1.9 h）：先验 `e25j_f1`（e2 配方，`data.train_fraction=0.01`，40k 步，按 DROID val 选模）。预训练 exit 0；最终步 val pos ADE 8.89 cm 高于常速 7.30，选中的 best.pt 为 7.25（1% 数据 40k 步明显过拟合，按 val 选早期 checkpoint）。TrunkBC 3 seed 训练、评测 exit 0，init = `outputs/e25j_f1/best.pt`，示范与 ref 相同，各 850 episode。成功率 **70.9 / 70.4 / 67.3**（均值 69.5；scratch 68.0，ref 73.4）。探针 exit 0。f10（18977169）运行中。
- E18r f10（Jubail 18977169，cn008，2.4 h）：先验 `e25j_f10`（`data.train_fraction=0.1`，40k 步；最终步 val pos ADE 7.87，选中 best.pt 5.74）。TrunkBC 3 seed 训练、评测 exit 0，init = `outputs/e25j_f10/best.pt`，示范与 ref 相同，各 850 episode。成功率 **74.9 / 70.2 / 70.8**（均值 72.0）。s0 评测耗时 57 min（其余约 10 min），疑似 cn008 短时 I/O 慢，结果完整。

### E18r / E11r 结果（2026-10-28 14:55；`stats_all.py --manifest configs/e18r_manifest.json`，(seed, task) 簇 bootstrap，合并 McNemar，Holm 对 2 个主检验）
| 先验 | s0 / s1 / s2 | 均值 ± sd |
|---|---|---|
| scratch | 68.7 / 67.6 / 67.8 | 68.0 ± 0.6 |
| LIBERO-only | 70.1 / 73.3 / 70.4 | 71.3 ± 1.8 |
| DROID 1% | 70.9 / 70.4 / 67.3 | 69.5 ± 2.0 |
| DROID 10% | 74.9 / 70.2 / 70.8 | 72.0 ± 2.6 |
| DROID 100%（ref） | 74.6 / 71.5 / 74.2 | 73.5 ± 1.7 |

| 比较 | 差 | 95% CI（任务簇） | 不一致对 | McNemar p | Holm |
|---|---|---|---|---|---|
| **主** ref vs LIBERO-only | +2.2 | [−1.2, +5.6] | 295:239 | 0.017 | **0.034** |
| **主** DROID 1% vs scratch | +1.5 | [−1.5, +4.5] | 274:236 | 0.10 | 0.10 |
| LIBERO-only vs scratch | +3.2 | [−0.1, +6.4] | 333:251 | 7.9e-4 | – |
| DROID 10% vs scratch | +4.0 | [+0.4, +7.4] | 345:244 | 3.6e-5 | – |
| ref vs scratch | +5.4 | [+2.0, +8.7] | 368:230 | 1.8e-8 | – |
| ref vs 1% | +3.9 | [−0.0, +7.8] | 329:229 | 2.7e-5 | – |
| ref vs 10% | +1.5 | [−1.3, +4.3] | 246:209 | 0.091 | – |
| 10% vs 1% | +2.5 | [−1.4, +6.3] | 289:226 | 0.0062 | – |

**按预注册判读**：
- 主检验 1（DROID 优于同域 LIBERO 状态预训练）：Holm 0.034 通过（回合级 McNemar），但任务簇 CI [−1.2, +5.6] 含 0——与本项目惯例一致，记为"方向一致、回合级显著、任务级不稳健"。
- 主检验 2（1% DROID 相对 scratch 有增益）：**不通过**（+1.5，Holm 0.10）。DEV7 上的 +4.5 未在 17 任务复现。
- 两个主检验未同时稳健通过 → 按预注册，这两条**不进摘要和图 1**，放附录并注明 DEV7 与 17 任务的差异。
- 次要（描述）：
  - 同域 LIBERO-only 预训练在 17 任务上**有**增益（+3.2 [−0.1, +6.4]，McNemar 8e-4），DEV7 上的"无增益"（+0.3）不复现；CI 下界贴近 0，按预注册措辞写"增益小于 DROID 100%"，不写"同域无增益"。
  - 数据量趋势单调：scratch 68.0 → 1% 69.5 → 10% 72.0 → 100% 73.5；10% 已显著（+4.0 [+0.4, +7.4]），10% 与 100% 差 1.5（n.s.）。
  - 1% 先验在 DROID val 上已过拟合（最终步 ADE 高于常速外推），可能是 1% 增益偏小的原因之一。
- 局限：每个变体一个先验实例（E38：实例间约 ±1 点）。

## 运维：GPU 利用率低（2026-10-28 16:30，Jubail 提醒）
- 原因：E37s / E41 是纯闭环评测，时间花在 CPU 上的 MuJoCo 仿真与渲染（每次评测 17 个仿真进程），GPU 每 10 步做一次小 batch 推理。实测运行中的 E41 DP s0（18977270）GPU 利用率 15%、显存 4 GB；seff 显示近期作业 CPU 效率 9–26%（申请 16–30 核）。
- 处理：`e41_camera.sbatch` 改为两次评测同时跑（`EVAL_PAR=2`，981abaa），在本地用假评测函数验证了配对执行和参数（dose 不带 mask，mask 带 `--sopt-mask-grasp -1`，clean 不带 `--perturb`）。每次评测有独立仿真进程、按 init 设种子，结果与串行相同；历史峰值内存约 70 GB / 评测，两个在 160 GB 内。
- 已提交但未开始的 3 个 E41 作业（dp s1 18977716、groot s0 18977737、groot s1 18979728）取消，按原顺序放回队首，用新脚本重提。正在运行的 dp s0（18977270）保持串行，不中断。

## E41 进度：DP s0（2026-10-28，Jubail 18977270，cn253，3.1 h，串行旧脚本）
- 6 次评测均 exit 0，各 850 episode；perturb / sopt_mask_grasp 字段与设计一致。
- 剂量（成功率，s0；干净与 20° 来自主表 / E37）：

| 条件 | none | SOPT | 差 |
|---|---|---|---|
| 干净 | 73.6 | 76.1 | +2.5 |
| camera 10° | 52.4 | 61.6 | +9.2 |
| camera 20° | 32.1 | 45.4 | +13.3 |
| camera 30° | 28.8 | 40.2 | +11.4 |

- 遮蔽（SOPT 臂整个 episode 关闭 SOPT 输入）：干净 76.1 → **29.3**（−46.8），20° 45.4 → **22.7**（−22.7）。绝对代价在干净条件更大，与 P2 的预期方向相反；关闭 SOPT 属于分布外（E34），DP 对 SOPT 输入的依赖本身很强。单 seed，按预注册等 DP s1 / GR00T s0–1 / SmolVLA-goal s0–1 齐后计算 P1 / P2。
- 运维：17:30–18:15 本机 VPN 断开，Jubail 不可达；后台 runner / watcher 只做查询，无误操作。当前 4 个 E41 作业（dp s1、groot s0 / s1、smolvla_goal s0）因集群拥挤（nvidia 分区 360 个排队作业）PENDING，Slurm 估计 10-30 16:38 开始。

## E41 进度：SmolVLA-goal s0、GR00T s0（2026-10-29，新的两两并行脚本 c9ad6c9）
- 作业 01:49–02:33 开始，比 Slurm 估计（10-30）早。
- SmolVLA-goal s0（Jubail 18981828，cn016，54 min；串行约需 1.8 h）：4 次剂量评测均 exit 0，各 500 episode。
- GR00T s0（none = e22，SOPT = e31 xattn；18980890，cn264，1.6 h）：剂量 4 次 + 遮蔽 2 次均 exit 0，各 850 episode。
- 剂量（s0；干净与 20° 来自 E37 / 主表）：

| 宿主 | 条件 | none | SOPT | 差 |
|---|---|---|---|---|
| SmolVLA-goal | 干净 | 75.0 | 80.6 | +5.6 |
| SmolVLA-goal | 10° | 67.4 | 74.4 | +7.0 |
| SmolVLA-goal | 20° | 54.8 | 66.6 | +11.8 |
| SmolVLA-goal | 30° | 48.4 | 70.2 | **+21.8** |
| GR00T xattn | 干净 | 81.9 | 80.4 | −1.5 |
| GR00T xattn | 10° | 73.5 | 80.0 | +6.5 |
| GR00T xattn | 20° | 42.1 | 70.5 | +28.4 |
| GR00T xattn | 30° | 24.8 | 55.5 | **+30.7** |

- 遮蔽（GR00T s0 SOPT 臂关闭 SOPT 输入）：干净 80.4 → 24.7（−55.7），20° 70.5 → 11.5（−59.0）。
- s0 小结（描述性）：SmolVLA-goal 与 GR00T 的增益随偏转角单调增大；DP s0 在 20° 最大（+13.3）、30° 略降（+11.4）。三宿主 30° 交互 − 10° 交互（P1 的单 seed 版本）：DP +2.2，SmolVLA-goal +14.8，GR00T +24.2。遮蔽代价 GR00T 在 20° 略大于干净，DP 相反。等 s1（dp 18980876、groot 18980898 运行中，smolvla_goal s1 待提交）。
- DP s1（Jubail 18980876，cn013，1.8 h，并行脚本）：6 次评测均 exit 0，各 850 episode。剂量 none / SOPT：干净 71.4 / 77.3（+5.9），10° 54.9 / 71.2（+16.3），20° 42.6 / 56.7（+14.1），30° 36.1 / 53.8（+17.7）。遮蔽：干净 77.3 → 16.6（−60.7），20° 56.7 → 14.4（−42.3）。DP 两个 seed 的遮蔽代价都是干净条件更大。
- 队列推进：smolvla_goal s1（18986691）与 E40 阶段 2 `_hpre` s0 none / droid（18986694 / 18986735）已提交。
- GR00T s1（Jubail 18980898，cn264，1.9 h，并行脚本）：6 次评测均 exit 0，各 850 episode。剂量 none / SOPT（xattn）：干净 79.6 / 83.9（+4.3），10° 72.5 / 79.2（+6.7），20° 51.9 / 68.9（+17.0），30° 26.9 / 53.6（+26.7）。遮蔽：干净 83.9 → 23.2（−60.7），20° 68.9 → 9.9（−59.0）。只剩 smolvla_goal s1（18986691，排队）。

## E40 阶段 2 进度：DP `_hpre` s0 none（2026-10-29，Jubail 18986694，cn008，1.2 h）
- 训练、评测 exit 0；run 名 `e21_dp_none_k5_hpre_t17_s0`，init_host = `outputs/e40_dp_host_train/final.pt`（加载 398 个张量，新建 0 个——none 臂无 SOPT 模块，符合预期），task_vocab all，k=5，seed 0，示范与主表 `e21_dp_none_k5_t17_s0` 完全相同（85 条，samples 13186）；850 episode。
- 成功率 **82.2**（从头训练的同臂主表为 73.6）：宿主在 52 个 train 任务上的动作预训练使 none 提升 +8.6。droid 臂（18986735）排队中。
- SmolVLA-goal s1（Jubail 18986691，cn267，41 min，并行脚本）：4 次剂量评测均 exit 0，各 500 episode。none / SOPT：10° 72.4 / 76.2，30° 59.2 / 70.6。

### E41 结果（2026-10-29 08:30；`contrast_boot.py --spec configs/e41_contrasts.json`，(seed, task) 簇 bootstrap，宿主分层，seed 0–1；输入 56 个 json，已核对 perturb / mask 字段与 episode 数）
主检验（Holm，2 个）：
- **P1 剂量**（交互(30°) − 交互(10°)，三宿主平均）：**+11.7 [+6.4, +17.3]**，p=1e-4，Holm 2e-4 → **通过**。
- **P2 遮蔽**（关闭 SOPT 的代价：20° − 干净，DP 与 GR00T 平均）：**−10.2 [−15.7, −4.8]**，Holm 2e-4 → **方向相反，不通过**（偏移下的代价反而更小）。

次要（交互 = (SOPT − none)_扰动 − (SOPT − none)_干净）：
| 宿主 | 10° | 20° | 30° | 30° − 10° |
|---|---|---|---|---|
| DP | +8.6 [+4.1, +13.0] | +9.5 [+3.1, +16.8] | +10.4 [+2.6, +18.9] | +1.8 [−5.8, +9.9] |
| SmolVLA-goal | +2.1 [−4.6, +9.7] | +4.9 [−6.9, +17.8] | +13.3 [+3.7, +24.7] | +11.2 [+0.4, +22.8] |
| GR00T xattn | +5.2 [+1.5, +9.2] | +21.4 [+13.9, +28.6] | +27.4 [+18.0, +36.5] | +22.1 [+13.3, +30.9] |
| 三宿主合并 | +5.3 [+2.3, +8.5] | — | +17.0 [+11.7, +22.7] | — |

遮蔽代价（SOPT − 关闭 SOPT）：
| 宿主 | 干净 | 20° | 关闭后 − none（20°） |
|---|---|---|---|
| DP | +53.8 [+43.1, +63.9] | +32.5 [+20.9, +43.8] | −18.8 [−29.3, −8.4] |
| GR00T xattn | +58.2 [+48.8, +66.8] | +59.0 [+50.4, +67.3] | −36.3 [−43.9, −28.9] |

**判读**：
- P1 通过：SOPT 在相机偏移下的额外收益随偏转角增大（三宿主合并 10° +5.3 → 30° +17.0）。GR00T 与 SmolVLA-goal 单调上升；DP 在 10° 已达约 +9 后基本持平（30° − 10° = +1.8，n.s.）。按预注册，5.3 节加剂量曲线。
- P2 不通过且方向相反：关闭 SOPT 后两宿主都远低于 none（DP −18.8、GR00T −36.3 @20°）——训练时 SOPT 一直开着，关闭是分布外（E34），宿主整体依赖这路输入，代价主要反映 OOD 崩溃而不是"偏移下的额外依赖"。DP 干净条件代价更大（成功率基数高，可掉的更多）；GR00T 两条件几乎相同（+0.8）。论文只报告为"宿主强依赖 SOPT 输入（关闭后低于不用 SOPT 的同结构基线）"，不写"偏移下依赖增加"。
- DP `_hpre` s0 expert_droid（Jubail 18986735，cn011，1.2 h）：训练、评测 exit 0；init_host 加载 398 个张量、新建 127 个（sopt.*，与 E40 smoke 一致），prior p0，task_vocab all，示范与 `_hpre` none 相同；850 episode 一一对应。成功率 **85.9**；对 `_hpre` none（82.2）74:43（p=0.005，+3.7）。
  - s0 对照：从头训练的宿主上 droid − none = 76.1 − 73.6 = +2.5；预训练宿主上 +3.7。单 seed 下 SOPT 增益没有因宿主动作预训练而缩小。E40 主检验（交互 = 增益(从头) − 增益(预训练) > 0）等 s1 / s2。
- DP `_hpre` s1 none（Jubail 18986804，cn004，1.2 h）：训练、评测 exit 0；init_host 加载 398 / 新建 0，task_vocab all，k=5、seed 1，示范相同；850 episode。成功率 **81.5**（从头训练的同臂 71.4）。
- 队列推进到 E39 SmolVLA：k1 s0 none（18988181）已提交。
- DP `_hpre` s1 expert_droid（Jubail 18987274，cn264，1.2 h）：训练、评测 exit 0；init_host 加载 398 / 新建 127（sopt.*），prior p0，示范与 `_hpre` none 相同；850 episode 一一对应。成功率 **83.6**；对 `_hpre` none（81.5）69:51（p=0.12，+2.1）。s1 对照（从头训练宿主）：77.3 − 71.4 = +5.9。
  - 两 seed 增益：从头宿主 +2.5 / +5.9，预训练宿主 +3.7 / +2.1。s2 none（18987453）运行中，s2 droid（18988110）排队；齐后按 `configs/e40_contrasts.json` 计算主检验。
- DP `_hpre` s2 none（Jubail 18987453，cn004，1.2 h）：训练、评测 exit 0；init_host 加载 398 / 新建 0，task_vocab all，seed 2，示范相同；850 episode。成功率 **81.8**（从头训练的同臂 73.5）。`_hpre` none 3 seed：82.2 / 81.5 / 81.8。只剩 s2 droid（18988110，排队）。
- DP `_hpre` s2 expert_droid（Jubail 18988110，cn266，1.2 h）：训练、评测 exit 0；init_host 加载 398 / 新建 127，prior p0，示范相同；850 episode 一一对应。成功率 **83.4**；对 `_hpre` none（81.8）58:44（p=0.20，+1.6）。

### E40 结果（2026-10-29 15:25；`contrast_boot.py --spec configs/e40_contrasts.json`，(seed, task) 簇 bootstrap，seed 0–2；输入 12 个 run 的 rollouts 与 meta，示范一致性已核对）
| 宿主 | none（s0 / s1 / s2） | droid（s0 / s1 / s2） | 增益 |
|---|---|---|---|
| 从头训练（E21 主表） | 73.6 / 71.4 / 73.5（72.9） | 76.1 / 77.3 / 79.1（77.5） | +4.6 [+1.8, +7.4] |
| 在 52 个 train 任务上动作预训练（`_hpre`） | 82.2 / 81.5 / 81.8（81.8） | 85.9 / 83.6 / 83.4（84.3） | +2.5 [+0.5, +4.5]，p=0.015 |

- **主检验**（交互 = 增益(从头) − 增益(预训练) > 0）：**+2.2 [−1.2, +5.5]**，p=0.21 → **不显著**。
- 次要：宿主预训练对 none +9.0 [+6.1, +12.0]（无天花板问题，none 仍在 82%）；droid_hpre − droid（从头）+6.8 [+3.0, +10.8]。
- **按预注册判读**：交互不显著 → 论文中"强宿主无收益"只作相关性描述，不写因果"互补"。
- 补充说明（描述性）：增益点估计从 +4.6 降到 +2.5（方向与互补一致），但宿主有了同域动作预训练后 SOPT **仍显著有益**（+2.5，CI > 0），两者大致可叠加（81.8 → 84.3）。这里的宿主预训练是 52 个 LIBERO 任务、2725 条示范，规模远小于 π0.5 / GR00T / X-VLA 的大规模机器人预训练，不能直接外推到后者。
- 局限：1 个宿主预训练实例；从头一侧沿用 E21 主表 run（one-hot 维度不同，多出维度恒为 0）。

## E39 SmolVLA 部分开始（2026-10-29）
- SmolVLA k1 s0 none（Jubail 18988181，cn010，2.5 h）：训练 20k 步、评测均 exit 0；run 名 `e23_smolvla_none_k1_t17_s0`，k=1（17 条示范，samples 2697，与 DP k1 示范相同），seed 0；850 episode。成功率 **45.3**（SmolVLA k5 none 3 seed 均值 60.7）。
- 运行中：k1 s0 scratch（18988861）、k1 s0 droid（18989081）、k1 s1 none（18989650）。
- SmolVLA k1 s0 expert_scratch（Jubail 18988861，cn016，2.7 h）：训练、评测 exit 0；k=1、seed 0、samples 2697，示范与 none 相同；850 episode 一一对应。成功率 **47.9**；对 none（45.3）80:58（p=0.074）。
- SmolVLA k1 s0 expert_droid（Jubail 18989081，cn255，2.7 h）：训练、评测 exit 0；k=1、seed 0、samples 2697、prior p0，示范相同；850 episode 一一对应。成功率 **53.4**。
  - k1 s0 三臂：none 45.3 / scratch 47.9 / droid 53.4。droid vs none 134:65（**+8.1**，p=1.1e-6），droid vs scratch 103:56（p=2e-4）。
  - 与 DP 相反：DP 在 k1 无增益（−0.4），SmolVLA 在 k1 s0 有大增益。单 seed，等 s1 / s2。
- SmolVLA k1 s1 none（Jubail 18989650，cn254，2.6 h）：训练、评测 exit 0；k=1、seed 1、samples 2697，示范与 s0 相同；850 episode。成功率 **44.1**（s0 45.3）。
- SmolVLA k1 s1 expert_scratch（Jubail 18990106，cn262，2.6 h）：训练、评测 exit 0；k=1、seed 1、samples 2697，示范相同；850 episode 一一对应。成功率 **47.4**；对 none（44.1）70:42（p=0.010）。
- SmolVLA k1 s1 expert_droid（Jubail 18990411，cn016，2.7 h）：训练、评测 exit 0；k=1、seed 1、samples 2697、prior p0，示范相同；850 episode 一一对应。成功率 **52.0**；对 none（44.1）122:55（**+7.9**，p=5.2e-7），对 scratch（47.4）97:58（p=0.0022）。SmolVLA k1 两 seed 增益 +8.1 / +7.9，与 DP k1（−0.4）方向相反。
- SmolVLA k1 s2 none（Jubail 18990538，cn255，2.6 h）：训练、评测 exit 0；k=1、seed 2，示范相同；850 episode。成功率 **43.9**（k1 none 3 seed：45.3 / 44.1 / 43.9）。
- 运维：剩余 37 个排队作业的 sbatch 命令行加 `--exclude=cn014,cn015,cn270,cn256`（cn256 曾出现阵发 I/O 停顿；只改资源申请，不影响结果）。已在 cn256 上运行的 SmolVLA k2 s0 droid（18994898）保留，监视其训练速度。
- SmolVLA k1 s2 expert_scratch（Jubail 18990630，cn254，2.7 h）：训练、评测 exit 0；k=1、seed 2，示范相同；850 episode 一一对应。成功率 **49.4**；对 none（43.9）86:39（p=3e-5）。SmolVLA k1 scratch 3 seed：47.9 / 47.4 / 49.4，三个 seed 都高于 none（与 DP 相反：DP k1 scratch 低于 none）。
- 新提交作业的 ExcNodeList 已含 cn256（18995301 核对）。
- SmolVLA k1 s2 expert_droid（Jubail 18992016，cn262，2.6 h）：训练、评测 exit 0；k=1、seed 2、samples 2697、prior p0，示范相同；850 episode 一一对应。成功率 **50.8**；对 none（43.9）99:40（p=6e-7），对 scratch（49.4）73:61（p=0.34）。

### E39 中期：SmolVLA k1 已齐（2026-10-29 22:55；`contrast_boot.py`，3 seed）
| 宿主 | none | scratch | droid | 增益 droid − none | 预训练 droid − scratch |
|---|---|---|---|---|---|
| SmolVLA k1 | 45.3 / 44.1 / 43.9（44.4） | 47.9 / 47.4 / 49.4（48.2） | 53.4 / 52.0 / 50.8（52.1） | **+7.6 [+4.1, +11.5]**（p=1e-4） | **+3.8 [+1.3, +6.5]**（p=0.003） |
| DP k1（对照） | 47.5 | 45.8 | 47.0 | −0.4 [−4.0, +3.3] | +1.2 [−3.0, +5.6] |
- SmolVLA 的 k1 增益（+7.6）大于其 k5 增益（+5.7 [+3.2, +8.3]），符合"低数据收益更大"；DP 相反。两宿主在低数据端方向相反，P1（两宿主平均）的结果取决于 SmolVLA k2；按预注册等齐后计算。
- SmolVLA k2 s0 none（Jubail 18994196，cn016，2.5 h）：训练、评测 exit 0；k=2（34 条示范，samples 5367，k1 示范为其子集），seed 0；850 episode。成功率 **55.9**。
- SmolVLA k2 s0 expert_droid（Jubail 18994898，cn256，2.6 h；cn256 上训练速度正常）：训练、评测 exit 0；k=2、seed 0、samples 5367、prior p0，示范相同；850 episode 一一对应。成功率 **61.5**；对 none（55.9）123:75（**+5.6**，p=8e-4）。
- SmolVLA k2 s1 none（Jubail 18995301，cn254，2.5 h）：训练、评测 exit 0；k=2、seed 1、samples 5367，示范与 s0 相同；850 episode。成功率 **54.5**（s0 55.9）。
- SmolVLA k2 s1 expert_droid（Jubail 18996264，cn255，2.7 h）：训练、评测 exit 0；k=2、seed 1、samples 5367、prior p0，示范相同；850 episode 一一对应。成功率 **64.8**；对 none（54.5）150:62（**+10.3**，p=1.3e-9）。SmolVLA k2 两 seed 增益 +5.6 / +10.3。
- SmolVLA k2 s2 none（Jubail 18996762，cn016，2.5 h）：训练、评测 exit 0；k=2、seed 2、samples 5367，示范相同；850 episode。成功率 **57.1**（k2 none 3 seed：55.9 / 54.5 / 57.1）。s2 droid（18997147）运行中。
- SmolVLA k2 s2 expert_droid（Jubail 18997147，cn268，2.2 h）：训练、评测 exit 0；k=2、seed 2、samples 5367、prior p0，示范相同；850 episode 一一对应。成功率 **59.8**；对 none（57.1）97:74（p=0.092）。

### E39 主检验 P1 可计算（2026-10-30 02:45；DP 与 SmolVLA 的 k1、k2 均已 3 seed 齐）
| | k1 | k2 | k5（主表） | 低数据增益（k1、k2 平均） |
|---|---|---|---|---|
| DP | −0.4 [−4.0, +3.3] | −1.3 [−5.5, +2.8] | +4.6 [+1.8, +7.5] | −0.8 [−3.9, +2.2] |
| SmolVLA | **+7.6 [+4.0, +11.5]** | **+6.2 [+2.7, +10.0]** | +5.7 [+3.0, +8.3] | **+6.9 [+3.8, +10.3]** |
- SmolVLA k2：none 55.9 / 54.5 / 57.1（55.8），droid 61.5 / 64.8 / 59.8（62.0）。
- **P1**（低数据增益，两宿主平均）：**+3.0 [+0.9, +5.3]**，bootstrap p=0.007。P2 尚缺（需 SmolVLA k25 / 全部），Holm 按 2 个主检验校正后 P1 的 p ≤ 2 × 0.007 = 0.014 < 0.05，**无论 P2 结果如何 P1 都通过**。
- 但 P1 的通过完全由 SmolVLA 驱动：DP 低数据增益 −0.8（CI 含 0），SmolVLA +6.9。写作必须报告宿主异质性，不能写成"SOPT 在低数据时普遍收益最大"。
- SmolVLA k10 s0 none（Jubail 18997450，cn263，2.0 h）：训练、评测 exit 0；k=10（170 条示范，samples 26462，k2 示范为其子集），seed 0；850 episode。成功率 **67.3**（同 seed k1 45.3 / k2 55.9 / k5 60.1）。
- SmolVLA k10 s0 expert_droid（Jubail 18999047，cn016，2.6 h）/ k10 s1 expert_droid（18999929，cn268，2.1 h）：训练、评测均 exit 0；k=10、samples 26462、prior p0，示范与 none 相同；各 850 episode。成功率 **75.6** / **74.8**。s0 对 none（67.3）133:62（**+8.3**，p=4e-7）。s1 none（18999515）运行中。
- SmolVLA k10 s1 none（Jubail 18999515，cn255，2.5 h）/ k10 s2 none（19000123，cn263，2.0 h）：训练、评测均 exit 0；k=10、samples 26462，示范与 s0 相同；各 850 episode。成功率 **69.1** / **71.6**（k10 none 3 seed：67.3 / 69.1 / 71.6）。s1 droid（74.8）对 none 138:89（**+5.7**，p=0.0014）。只剩 k10 s2 droid。
