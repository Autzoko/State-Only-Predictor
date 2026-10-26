# SOPT 论文大纲（初版，2026-10-26）

配套表格：`paper/SOPT_results.xlsx`，由 `scripts/paper/build_results_xlsx.py` 生成。下文的"表 T1…T10"指工作簿中的同名 sheet。
数字以 `docs/EXPERIMENTS.md` 为准。新结果出来后，先改脚本、重新生成表格，再改本文。

---

## 0. 投稿目标与时间表

下表为 2026-10-26 查阅官方 CFP 的结果。投稿前请再到官网核对一次。

| 会议 | 截止（AoE） | 篇幅 | 模板 | 适配度 |
|---|---|---|---|---|
| **RSS 2027 Stage 1（推荐）** | **2026-12-04**：6 页扩展摘要 | Stage 1：正文 5 页 + 参考文献 1 页；Stage 2（2027-04-16）：正文 8 页，不含参考文献 | RSS LaTeX：roboticsconference.org/docs/paper-template-latex.tar.gz（不得改模板） | 高。机器人学习核心会议；Stage 1 明确允许实验不完整或为初步结果；双盲 |
| L4DC 2027 | 2026-11-13 | 10 页 PMLR 正文，不含参考文献 | `l4dc2027.cls` | 中高。可按"本体动力学先验"讲；纯仿真常见；时间紧（18 天） |
| CVPR 2027 | 摘要 2026-11-10（必填），正文 2026-11-16，补充材料 11-23 | 8 页，不含参考文献 | CVPR 2027 author kit | 中。只适合以"视角鲁棒性"为卖点；方法刻意不用视觉，评审口味有风险 |
| TMLR / RA-L | 滚动投稿 | — | — | 备选 |

- **建议**：按 RSS 模板写。
  - 12 月 4 日交 Stage 1 的 5 页版本，见 §6"Stage 1 压缩方案"。
  - 第一版全文按 Stage 2 的 8 页写，之后再压缩，比从 5 页往上扩省事。
  - 如果进入 Stage 2，到 4 月 16 日前可以补进 E39、E40、E41、E38b、E42，以及可能的真机实验。
- **一稿多投限制**：CVPR 评审期（11 月 16 日至次年 2 月 22 日）和 L4DC 决定日（1 月 25 日）都与 RSS Stage 1 重叠，同一工作只能选其一。
- **双盲**：正文不写机构名，不放 GitHub 链接。代码在 Stage 2 时以匿名补充材料提交（RSS Stage 2 允许 100 MB zip）。

---

## 1. 标题与一句话论点

**标题候选**（先选定一个，摘要和引言都围绕它写）
1. *State-Only Pretraining: Proprioceptive Motion Priors from Robot Logs Make Imitation Policies More Data-Efficient and Viewpoint-Robust*
2. *Robots Already Know How to Move: Pretraining on Proprioception Alone Improves Visuomotor Policies*
3. *SOPT: A Plug-in Motion Prior Pretrained Only on Robot States*

**一句话论点**：只用大规模机器人日志里的本体状态序列（DROID，2760 万帧，不含图像和动作），就能预训练出一个不到 1000 万参数的运动先验。把它以零初始化的方式接入现成策略后：
- 没有经过大规模机器人动作预训练的宿主（Diffusion Policy、SmolVLA、视频世界动作模型、历史主干 BC）都显著提升；
- 这一收益来自预训练本身；
- 在相机视角偏移下，收益放大数倍。

**两条主线**（用户确认的方向）
- **论点 A（数据效率 / 宿主互补）**：SOPT 的收益集中在宿主缺少机器人运动知识的情形，即低示范量、宿主没有在机器人动作上做过大规模预训练。宿主已具备这类知识时（π0.5-DROID、GR00T、X-VLA），收益消失。这一点作为适用边界如实报告。
- **论点 B（视角鲁棒）**：本体运动历史与相机无关。视觉输入不可靠时，宿主会更依赖 SOPT，所以视角偏移下 SOPT 的收益远大于干净条件。GR00T 干净条件下 +1.3，20° 相机偏移下 +22.7。

---

## 2. 摘要骨架（约 200 词，英文写作）

每句一个功能，数字取自 T1 和 T2：
1. **问题**：大规模机器人日志里最便宜、最通用的部分是本体状态，现有预训练却几乎只用图像和动作。
2. **方法**：SOPT，一个只在 DROID 状态序列上用 flow-matching 自回归预训练的因果 Transformer；以零初始化的方式接到任意策略的动作 token 或全局条件上。
3. **主结果**：LIBERO-90 的 17 个未见任务、每任务 5 条示范、3 个 seed：
   - Diffusion Policy +4.6；
   - SmolVLA +4.8；
   - 视频世界动作模型 FastWAM +11.0（相对提升 25%）；
   - 同结构随机初始化的组件没有增益。
4. **鲁棒性**：视觉偏移下的预注册检验通过，交互项 +4.6 [3.7, 5.6]。主要来自相机视角：相对干净条件额外 +13 点；GR00T 的成功率保持比例从 0.58 提到 0.85。
5. **分析**：
   - 预训练把加速度、未来 1 s 运动等动态量变得线性可读（加速度 R² 0.50→0.90）；
   - 减少的主要是抓取等执行类失败（grasp_miss −35%）；
   - 只用 1% 的 DROID 就有显著增益；
   - 同域的少量状态预训练没有增益。
6. **边界**：已在机器人动作上大规模预训练的 VLA 不受益，诚实报告。
7. **意义**：本体日志是一个被忽视、几乎免费的预训练来源。

---

## 3. 贡献（引言末尾列 3–4 条）

1. **新的预训练来源与方法**：首次系统地证明，**只用本体状态**（无图像、无动作、无任务标签）的大规模预训练，可以作为通用插件提升视觉运动策略。设计上的教训（§4.3）也算方法贡献：
   - 接口必须紧挨动作输出；
   - 先验应作为表示使用，而不是作为规划器。
2. **跨 6 类宿主的受控评测**：BC 主干、Diffusion Policy、SmolVLA、FastWAM、π0.5、GR00T、X-VLA。
   - 每个宿主都设 none / 同结构 scratch / DROID 三臂；
   - 3 个 seed，配对设计，任务层簇 bootstrap，Holm 校正，关键结论均预注册。
   - 结果给出"什么样的宿主受益"的清晰图景。
3. **视角鲁棒性**：预注册检验显示，SOPT 在视觉偏移、尤其是相机视角变化下收益放大。待补：E41 剂量与遮蔽检验。
4. **机制分析**：
   - 线性探针：SOPT 补上了宿主缺失的动态信息；
   - 失败归类：减少的是执行类失败；
   - 数据规模与来源：多样性和规模比同域更重要；
   - 先验实例复制：结论不依赖某一次预训练。

---

## 4. 正文各节写什么（按 RSS Stage 2 的 8 页估算篇幅）

### 4.1 Introduction（约 1 页）
- **第 1 段**：机器人基础模型的数据瓶颈。图像 + 语言 + 动作的配对数据昂贵；跨机器人的动作空间难以统一。
- **第 2 段**：被忽略的信号是**本体状态**。每台机器人都会记录它，格式在同类机械臂之间统一（EE 位姿 + 夹爪），不涉及相机标定与隐私。DROID 的状态部分只有约 1.4 GB 下载量。从动作的角度看，"机器人怎么动"是一种可迁移的知识：加速、对准、抓取前减速、夹爪事件的节奏。
- **第 3 段**：现有做法与缺口：
  - 视觉预训练（R3M、MVP、DINOv2）只管"看"；
  - 动作预训练（Octo、OpenVLA、π0）需要动作标签；
  - 隐动作（LAPA）从视频里反推动作；
  - 带本体的跨机器人预训练（HPT、RPT）都和视觉 / 动作绑定。
  - 尚无人回答：**只用状态能学到对控制有用的东西吗？**
- **第 4 段**：本文做法（SOPT）与核心发现，用 2–3 个最亮的数字：FastWAM +11、GR00T 相机偏移 +22.7、1% DROID 就够。
- **第 5 段**：边界。强 VLA 不受益，这本身说明 SOPT 提供的是"宿主缺的运动知识"。
- 贡献列表（§3）。
- **图 1（首页大图）**：左为 DROID 状态序列 → SOPT 预训练；中为零初始化接入任意宿主的动作模块；右为两个小柱状图：主表的增益条，以及相机偏移下 GR00T 的 47.0 vs 69.7。

### 4.2 Related Work（约 0.6 页，三段）
- **机器人表征 / 预训练**：视觉（R3M、VIP、MVP、DINOv2，以及 Burns et al. 关于视觉表征鲁棒性的研究）；动作与 VLA（RT-1/2、Octo、OpenVLA、π0/π0.5、GR00T、SmolVLA、X-VLA）；隐动作与视频（LAPA、GR-1、UniPi、VPT）；带本体的跨机器人模型（HPT、CrossFormer、RPT）。**区分点**：我们的预训练数据完全不含图像和动作。
- **本体历史与运动先验**：RMA（本体历史做自适应）；动画和腿足中的运动先验（AMP）；轨迹预测（MotionLM）；只用状态的模仿（BCO）。**区分点**：我们研究可插拔的、在大规模异构数据上预训练的操作运动先验。
- **历史条件策略的 copycat / 因果混淆**（de Haan et al. 2019；Wen et al. 2020）：解释"scratch 历史有害、预训练历史有益"的反转。**鲁棒性评测**：Xie et al. 2023 的泛化差距分解；LIBERO-Plus 类扰动评测。

### 4.3 Method（约 1.3 页）
- **4.3.1 数据与状态表示**：
  - DROID 1.0.1 只取状态列，共 95,584 条 episode、2760 万帧，按 site 划分 train / val / test（val 为 OOD 实验室）；
  - 17 维特征：法兰位姿（基座系）+ 关节 + 夹爪；重采样到 15 Hz；
  - EE-only 变体把关节置零，用于没有关节信息的数据（E13 / E15）。
  - 写清楚：**只下载状态列**，不含图像、动作或语言。
- **4.3.2 预训练目标**：
  - 因果 Transformer（S：d256 × 6 层，4 头，约 980 万参数（E17 计数）；RoPE；QK-norm）；
  - 输入 96 帧（6.4 s）历史，flow-matching 自回归预测未来 32 帧（2.1 s）的增量；
  - 40k 步，单张 A100 约 30 分钟。
  - 交代 QK-norm 是稳定训练的必要条件（E1 v1 发散）。
  - 先验质量放附录 T8。正文只用一句话："在 OOD 实验室上 ES 比恒速外推低 50%，零样本迁移到 LIBERO 低 57%。"
- **4.3.3 接入宿主（plug-in）**：
  - 取历史最后一个 token 的特征，经零初始化 MLP 加到宿主每个动作 token 上（DP 加在 U-Net 的全局条件上）；
  - 初始化时策略与原宿主**逐元素相同**（已验证）；
  - SOPT 与宿主联合微调，学习率与宿主相同；
  - 每个宿主各接在哪里，用一个小表或图 2 说明。
- **4.3.4 设计教训（短段，引用附录负结果）**：
  - 作为 VLM 前缀 token 无效（E5）；接在动作侧才有效（E7）。
  - "先验规划 + IDM 跟踪"失败（E4b：33% vs 91%），所以把 SOPT 当表示而不是规划器。
  - 这两点给出"运动先验应该放在哪里"的结论，是对后来者有用的方法学贡献。

### 4.4 Experimental Setup（约 0.8 页）
- **基准**：LIBERO-90 中 17 个指令唯一的任务（7 个 test-split + 10 个 val-split），从未用于任何选择；每任务 k=5 条示范（嵌套、按加盐哈希选取）；每任务 50 个官方初始状态。
  - 7 任务开发集（DEV7）在开发中反复使用，只放附录。这一点要主动交代，体现严谨性。
- **宿主**：每个宿主一句话，再加一个小表（宿主、参数量、动作预训练数据、训练哪些部分、步数）：
  - TrunkBC（ResNet-18）；
  - Diffusion Policy；
  - SmolVLA（冻结 VLM）；
  - FastWAM（Wan2.2-5B 视频 DiT + 1B 动作 DiT）；
  - π0.5-DROID；
  - GR00T N1.7；
  - X-VLA-0.9B。
  - 投稿前核对一处版本号：日志 E22 及之后均为 N1.7，但 3 seed 汇总处有一处写成 N1.5。
- **三臂**：none / +SOPT scratch（同结构随机初始化，控制"多了历史输入和参数"）/ +SOPT DROID。除 SOPT 外所有设置完全相同。
- **统计**：
  - 配对设计，(seed, task, init) 对齐；
  - 合并 McNemar；
  - (seed, task) 簇 bootstrap 95% CI；
  - Holm 校正；
  - 关键检验事先写定判据（E37、E38，以及 E39–E41）。
  - 这是本文相对多数机器人论文的优势，正文要写明。
- **防泄漏**：SOPT 只见过 DROID；评测初始状态与训练示范无重复（已检查）；评测任务未参与任何选择。

### 4.5 Results（约 3.3 页，正文核心）

**5.1 主结果：SOPT 是通用插件（表 1 = T1，正文最重要的表）**
- 行按"宿主的机器人动作预训练程度"排序：无 → 少 → 大规模。这样表格本身就讲出论点 A。
- 列：none | +scratch | +DROID | Δ（CI）| Δ vs scratch（CI）。
- 文字要点：
  1. 无大规模机器人动作预训练的 4 个宿主全部显著提升：
     - TrunkBC 相对 scratch +5.4；
     - DP +4.6；
     - SmolVLA +4.8；
     - FastWAM +11.0。
  2. 增益来自**预训练**。scratch 组件在 5/6 个宿主上没有增益或有代价（−1.7 至 −2.0），FastWAM 例外（视觉通路最弱，历史本身就有用）。
  3. 已大规模预训练的宿主上增益消失：π0.5 +0.4，GR00T 0.0，X-VLA −2.5。这里 DROID vs scratch 仍为正（π0.5 +2.5，GR00T +1.9），说明预训练表示本身是有用的，只是宿主已有同类知识。X-VLA 的预训练本身就含 DROID。
- 待补：SmolVLA 的簇 CI，等 _jr 重跑。

**5.2 低数据下收益最大（论点 A 的第二根支柱）**
- 现有证据，都在 DEV7 上：
  - SmolVLA k5 +4.8 → 全部示范 +2.8；
  - TrunkBC k5 +11.0 → 全部示范 +7.1（相对 scratch）；
  - E4a 预测任务：k=1 的预训练模型优于全部示范的 scratch，约 50 倍数据效率。
- **E39 完成后替换为正文图 3**：横轴 k ∈ {1, 2, 5, 10, 25, 全部}，纵轴增益，DP 和 SmolVLA 两条线；预注册主检验为 P1（低数据增益 > 0）和 P2（增益随数据量下降）。
- **E40 完成后加一段**：同一个 DP 宿主先在 52 个其他任务上做动作预训练，SOPT 增益是否缩小。这把 5.1 中"强宿主无收益"从相关关系变成因果证据。如果交互显著，就是全文第二亮的结果。

**5.3 视角鲁棒性（论点 B，表 2 = T2 精简版 + 图 4）**
- 图 4：三宿主 × 5 种扰动的增益条（干净 vs 扰动），突出 camera。
- 要点：
  - 预注册交互 +4.6 [3.7, 5.6]，通过；
  - camera +13.0（三宿主一致，任务层面稳健）；
  - GR00T 在干净条件几乎不用 SOPT（+1.3），相机偏移下 +22.7（47.0 → 69.7），而且对偏转方向不敏感（70.4 / 70.6 vs none 36.5 / 47.8）。
- 如实写：
  - texture / noise 的额外收益在任务层面不稳健；
  - light 的档位太弱，无法判断；
  - 档位由 pilot 事先标定，标定用的模型不进入检验。
- E41 完成后补：剂量（10° / 20° / 30°）随偏移增大，以及遮蔽 SOPT 后增益消失。

**5.4 增益从哪来、对什么稳健（消融表 = T3 精简版，正文一张合并表或图）**
- 正文只放最有冲击力的三组：
  1. **数据来源（E18）**：只用 LIBERO 同域状态预训练 = 无增益（61.4 vs 61.1），DROID +8.2。说明关键是**规模和多样性**，而不是同域。这是很反直觉的一条，值得放正文。
  2. **数据量（E11）**：1% DROID（864 条）就显著（+4.5），100% 最好（+8.5），呈单调趋势。模型放大 6 倍和换 LM 式模块都无增益，即数据比容量重要。
  3. **先验实例（E38，预注册）**：3 次独立预训练，DP 增益为 +3.8 ± 0.7，TrunkBC 为 +4.9 ± 0.5，Holm 后 4 个检验全部通过。
- 一句话带过、放附录的：历史长度（3.2–25.6 s 无差别，E25j）；速度增强；事件 / JEPA 辅助目标；EE-only 先验与含关节先验等效。

**5.5 SOPT 提供了什么（图 5：探针 + 失败归类，二合一）**
- 左图：线性探针 R²（T4）。宿主当前帧、随机初始化 SOPT、DROID SOPT 三组柱：
  - 加速度：0.15 / 0.50 / 0.90；
  - 未来 1 s：0.24 / 0.64 / 0.75；
  - 下一夹爪事件：0.15 / 0.47 / 0.53。
- 右图：失败归类（E9a BC→TrunkBC，E7b SmolVLA）。SOPT 减少的是 grasp_miss / place_fail，"去哪"类不变。
- 结论：SOPT 补的是"怎么动"，不是"去哪"。这与它的数据来源一致，也解释了为什么它在视觉受损时最有用。

### 4.6 Discussion & Limitations（约 0.5 页）
- **强宿主的边界**：π0.5、GR00T、X-VLA 无收益。机制证据（E26 / E34，附录）：加法注入在强宿主上导致抓空后照常搬运；交叉注意力能去掉代价，但带不来收益。
- **长时程任务**：libero_10 上历史组件本身有用，但 DROID 先验没有额外增益。GR00T 上，只有"重复动作、需要记住进度"的任务稳健受益（附录 T6）。
- **纯仿真**：真机实验是 Stage 2 的目标（PAPER_PLAN §5 已有完整方案）。
- **单一具身**：DROID 和 LIBERO 都是 Franka。EE-only 先验为跨机器人迁移留了接口（E42 待做）。
- **失败预测这一应用未成功**（E36a）。
- 写法：每条局限给出已有证据或计划，不要只列不答。

### 4.7 Conclusion（约 0.2 页）
- 重申：本体日志是一个被低估的预训练来源；
- 小而便宜的先验可以插入任意策略；
- 在低数据和视角偏移下价值最大。

---

## 5. 图表清单（正文 vs 附录）

| 编号 | 内容 | 数据来源（xlsx / 日志） | 位置 | 状态 |
|---|---|---|---|---|
| 图 1 | 方法总览 + 两个关键数字 | — | 正文首页 | 待画 |
| 表 1 | 主表：6+1 个宿主 × 三臂 | T1 | 正文 | SmolVLA CI 待补 |
| 图 2 | 各宿主的接入位置示意 | 4.3.3 | 正文（可与图 1 合并） | 待画 |
| 图 3 | 示范数量曲线 | E39 | 正文 | **待跑** |
| 表 2 / 图 4 | 视觉扰动 × 宿主 | T2 | 正文 | 已完成；E41 待补 |
| 表 3 | 消融：数据来源 / 数据量 / 先验实例 | T3 | 正文 | 已完成；E38b 待补 |
| 图 5 | 探针 + 失败归类 | T4 | 正文 | 已完成 |
| 表 A1 | DEV7 全部宿主 | T7 | 附录 | 已完成 |
| 表 A2 | 标准 LIBERO 四个 suite | T5 | 附录 | 已完成 |
| 表 A3 | GR00T 长时程 + 历史长度 | T6 | 附录 | 已完成 |
| 表 A4 | 预训练配方消融（E25j、E25） | T3 下半 | 附录 | 已完成 |
| 表 A5 | 先验质量（DROID 预测 / LIBERO 迁移） | T8 | 附录 | 已完成 |
| 表 A6 | 负结果汇总 | T9 | 附录 | 已完成 |
| 表 A7 | 逐任务成功率（17 任务 × 宿主） | 需从 rollouts 生成 | 附录 | 待生成 |

**"眼前一亮"的数字（放图 1、摘要、引言第 4 段）**
- FastWAM +11.0 点（43.5 → 54.5，相对 +25%），p=3e-24；
- GR00T 相机偏移：none 47.0，加 SOPT 69.7（干净条件只差 1.3）；
- 只用 1% DROID（864 条状态轨迹）就显著提升；同域 LIBERO 状态预训练无效；
- 先验不到 1000 万参数，单卡（A100）预训练约 30 分钟，DROID 状态数据下载约 1.4 GB；
- grasp_miss 减少 35%，"去哪"类失败不变。

---

## 6. RSS Stage 1 压缩方案（5 页正文 + 1 页参考文献，12 月 4 日）

- 保留：引言（0.8 页）、方法（1 页）、设置（0.5 页）、表 1（主表）、表 2 / 图 4（视角鲁棒）、表 3 的精简版（数据来源 + 先验实例两行即可）、一段边界与局限。
- 删到附录或删掉：相关工作压缩成 0.3 页；图 5 只留探针或只留失败归类之一；5.2 低数据部分如果 E39 还没完成，只写一句 DEV7 的证据，并注明"完整曲线见 Stage 2"。
- Stage 1 不允许补充材料，所以附录内容留到 Stage 2。

---

## 7. 附录页规划（Stage 2 补充材料）

- **A. 实现细节**：SOPT 结构与超参；每个宿主的训练配方（步数、批大小、学习率、冻结哪些部分、n_exec、chunk）；接入代码位置；零初始化等价性检查。
- **B. 数据**：DROID 只取状态的处理流程、site 划分、坐标系校验（FK 残差 0.2 mm）；LIBERO 预处理（20→15 Hz、夹爪语义、IPEC 图像翻转）；EE → 法兰标定（残差 0.27 mm）。
- **C. 评测协议与统计**：17 任务清单；示范选取规则；McNemar 与簇 bootstrap 的定义；所有预注册文本的原文及时间（E37、E38、E39–E41）；Holm 的范围。
- **D. 防泄漏审计**：初始状态与示范场景的距离分析（2026-10-12 泄漏审计，`check_init_leakage.py`）；DROID 与 LIBERO 无交集；宿主预训练数据不含 LIBERO（X-VLA 的 domain 选择、GR00T 的 embodiment tag）。
- **E. DEV7 全部结果**（表 A1），以及 7 任务 → 17 任务的协议变更理由。
- **F. 标准 LIBERO suite**（表 A2）与 SOPT dropout（E19）。
- **G. 强宿主分析**：E20d 开环诊断、E26 失败归类、E27 宿主探针、E31 接口对比（加法 / AdaLN / 交叉注意力）、E34 遮蔽。
- **H. 长时程 GR00T**（表 A3）：历史长度非单调、先验实例对照、预注册 s3/s4 检验未达 0.05（如实写）。
- **I. 预训练配方消融**（表 A4）与先验本身的预测质量（表 A5、E1–E4a、E16）。
- **J. 负结果**（表 A6）：先验 + IDM、KFMotor、失败预测、辅助目标。
- **K. 逐任务结果**（表 A7）与难度分层（headroom）分析。
- **L. 计算开销**：各宿主的 GPU 小时；SOPT 预训练 30 分钟；推理额外开销（待实测，PAPER_PLAN 中 < 2 ms 只是估计）。
- **M. 真机实验**（若 Stage 2 前完成）：任务、协议、盲测。

---

## 8. 参考文献初稿

所有条目都请用 arXiv / DBLP 导出 BibTeX 核对。标 [待核实] 的条目，我对编号或作者没有把握。

**数据集与基准**
- Khazatsky et al. *DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset.* RSS 2024. arXiv:2403.12945
- Liu et al. *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning.* NeurIPS 2023 D&B. arXiv:2306.03310
- Open X-Embodiment Collaboration. *Open X-Embodiment: Robotic Learning Datasets and RT-X Models.* ICRA 2024. arXiv:2310.08864
- Mandlekar et al. *MimicGen.* CoRL 2023. arXiv:2310.17596（E42 用）
- LIBERO-Plus / LIBERO-PRO 等扰动评测 [待核实：标题与编号]

**宿主策略 / 基础模型**
- Chi et al. *Diffusion Policy.* RSS 2023. arXiv:2303.04137
- Shukor et al. *SmolVLA.* 2025. arXiv:2506.01844
- Black et al. *π0.* 2024. arXiv:2410.24164
- Physical Intelligence. *π0.5.* 2025. arXiv:2504.16054
- NVIDIA. *GR00T N1.* 2025. arXiv:2503.14734（N1.7 另引模型卡）
- Zheng et al. *X-VLA.* 2025. arXiv:2510.10274
- FastWAM [待查：LeRobot `fastwam_base` 对应的论文或技术报告]
- Wan Team. *Wan: Open and Advanced Large-Scale Video Generative Models.* 2025. arXiv:2503.20314
- Kim et al. *OpenVLA.* CoRL 2024. arXiv:2406.09246
- Octo Model Team. *Octo.* RSS 2024. arXiv:2405.12213
- Brohan et al. *RT-1* arXiv:2212.06817；*RT-2* arXiv:2307.15818
- Zhao et al. *ACT / ALOHA.* RSS 2023. arXiv:2304.13705

**机器人预训练与表征**
- Nair et al. *R3M.* CoRL 2022. arXiv:2203.12601
- Ma et al. *VIP.* ICLR 2023. arXiv:2210.00030
- Xiao et al. *Masked Visual Pre-training for Motor Control.* 2022. arXiv:2203.06173
- Oquab et al. *DINOv2.* 2023. arXiv:2304.07193
- Radosavovic et al. *Robot Learning with Sensorimotor Pre-training (RPT).* CoRL 2023. arXiv:2306.10007（最接近的工作之一，必须讨论差异）
- Wang et al. *HPT: Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers.* NeurIPS 2024. arXiv:2409.20537（必须讨论）
- Doshi et al. *CrossFormer.* CoRL 2024. arXiv:2408.11812
- Ye et al. *LAPA: Latent Action Pretraining from Videos.* 2024. arXiv:2410.11758
- Wu et al. *GR-1.* ICLR 2024. arXiv:2312.13139
- Du et al. *UniPi.* NeurIPS 2023. arXiv:2302.00111
- Baker et al. *VPT.* NeurIPS 2022. arXiv:2206.11795
- Lin et al. *Data Scaling Laws in Imitation Learning for Robotic Manipulation.* 2024. arXiv:2410.18647

**运动先验、本体历史与只用状态的学习**
- Kumar et al. *RMA: Rapid Motor Adaptation.* RSS 2021. arXiv:2107.04034
- Peng et al. *AMP: Adversarial Motion Priors.* SIGGRAPH 2021. arXiv:2104.02180
- Seff et al. *MotionLM.* ICCV 2023. arXiv:2309.16534
- Torabi et al. *Behavioral Cloning from Observation.* IJCAI 2018. arXiv:1805.01954

**历史条件策略的失败模式**
- de Haan et al. *Causal Confusion in Imitation Learning.* NeurIPS 2019. arXiv:1905.11979
- Wen et al. *Fighting Copycat Agents in Behavioral Cloning from Observation Histories.* NeurIPS 2020 [待核实 arXiv 编号]

**鲁棒性与失败检测**
- Xie et al. *Decomposing the Generalization Gap in Imitation Learning for Visual Robotic Manipulation.* ICRA 2024. arXiv:2307.03659
- Burns et al. *What Makes Pre-Trained Visual Representations Successful for Robust Manipulation?* CoRL 2024. arXiv:2312.12444 [待核实会议年份]
- Agia et al. *Unpacking Failure Modes of Generative Policies (Sentinel).* CoRL 2024. arXiv:2410.04640 [待核实]
- FAIL-Detect [待核实]

**模型组件与统计**
- Lipman et al. *Flow Matching for Generative Modeling.* ICLR 2023. arXiv:2210.02747
- Dehghani et al. *Scaling Vision Transformers to 22B Parameters*（QK-norm）. ICML 2023. arXiv:2302.05442
- Su et al. *RoFormer (RoPE).* arXiv:2104.09864
- Peebles & Xie. *DiT（AdaLN）.* ICCV 2023. arXiv:2212.09748
- Assran et al. *I-JEPA.* CVPR 2023. arXiv:2301.08243
- Gneiting & Raftery. *Strictly Proper Scoring Rules, Prediction, and Estimation.* JASA 2007（energy score）
- McNemar 1947（Psychometrika）；Holm 1979（Scandinavian Journal of Statistics）；Efron & Tibshirani 1993（bootstrap）

---

## 9. 等待中的结果会如何改变文稿

| 结果 | 若通过 | 若不通过 |
|---|---|---|
| E39 示范数量曲线 | 图 3 进正文；摘要加"增益随示范数下降，k=1 时最大" | 改写为"在所测数据量范围内收益稳定"；论点 A 改以宿主类型为主轴 |
| E40 宿主动作预训练 | 5.1 第 3 点由相关改为因果：宿主有了动作知识，SOPT 就不再需要 | 强宿主无收益只作相关性描述 |
| E41 剂量 + 遮蔽 | 5.3 加剂量曲线和遮蔽结果，论点 B 更完整 | 只报告 20° 结果，剂量写作局限 |
| E38b SmolVLA / FastWAM 先验实例复制 | 表 3 写"4 个宿主上可复制" | 注明实例依赖 |
| SmolVLA _jr 重跑 | 主表补 CI；DGX 的 s0 / s1 作为跨平台复制 | — |

---

## 10. 写作注意

- **措辞**：
  - 只说"在没有大规模机器人动作预训练的宿主上显著提升"，不说"提升所有 VLA"；
  - 视角鲁棒性只说"相机视角变化下收益显著放大"，不扩大到所有视觉扰动；
  - 历史长度不说"越长越好"。
- **单 seed 结果不进正文**：E13 全部示范、E14b、E35 ctx192 等。
- **主动报告预注册未通过的检验**：E36a、E35 s3/s4、E25 event / JEPA。这是可信度的来源，放附录并在正文提一句。
- **主表数字统一**：用 17 任务、3 seed 的版本（xlsx T1）；DEV7 的数字只在附录出现，避免同一宿主出现两套数字。
- **先验实例**：主表用 p0。正文在表 3 中给出 3 实例均值，并在表 1 的注释说明"换先验实例约 −1 点，不改变结论"。
