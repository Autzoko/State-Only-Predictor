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
