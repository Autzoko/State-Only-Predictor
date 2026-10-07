# 实验计划与日志

计划和判停标准见 `DESIGN.md` §6。每次运行记录：日期、机器、git rev、命令、关键指标、结论。
规则：checkpoint 选择和调参只用 `val`；`test` 只在论文最终数字时使用一次。

## 待办（按顺序，每一步以上一步结果为前提）
- [ ] E0 下载完整 DROID parquet 到 127，运行 `prepare_droid.py`，检查 `summary.json`（building 切分是否均衡，val/test 各约 5%）
- [ ] E0 吞吐测试：S/B 在 A6000 上的 it/s，确定 E1 的 max_steps
- [ ] E1 `ar_regression` / `ar_flow` / `masked` × S（然后是 B），与 zero-vel / const-vel 对比
- [ ] 判停检查（DESIGN §6）：`ar_flow` 的 min10_pos_ade 相比 const_vel 下降 ≥30%？
- [ ] E3 线性 probe（成功/失败、gripper 事件），需要 `sopt/eval/probe.py`
- [ ] E4 下游：LIBERO 少样本策略，需要 `sopt/policy/`（视觉条件 + 预训练 trunk）

## 日志

### 2026-10-07 搭建
- 已核实 DROID schema（`cadene/droid_1.0.1` @ 56b622a）：欧拉角为 extrinsic XYZ，`cartesian_position` 与 Franka FK 的 flange 位姿完全一致；元数据列错位（见 DESIGN §3）。
- 本地 CPU 调试（30 条 episode，40 步）：三种目标函数从训练、评估到保存 checkpoint 全流程跑通。指标没有意义。
