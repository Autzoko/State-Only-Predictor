# 服务器

## 评估（2026-10-07）

| | NYUAIRLAB-127 | NYUAIRLAB-3090 | JUBAIL |
|---|---|---|---|
| GPU | 1× A6000 48GB（XICM 已占用 32GB） | 3× RTX 3090 24GB（空闲） | A100 40/80G、H100、H200（排队） |
| CPU / 内存 | 48 核 / 250 GB | 20 核 / 62 GB | 每个作业 16 核 / 64 GB |
| 磁盘 | 8.8 TB 空闲 | 系统盘 320 GB；data-002 2.9 TB | /scratch 文件数 90% |
| 限制 | 与 XICM 共用 GPU | **被 junyi 预留：启动 GPU 任务前需征得同意** | 每用户最多 2 个作业，目前已有 RoboticsNAS 在用 |

**结论**
- State-only 数据很小（约 3.5 GB，能整体放进内存），模型也小（5–115M）。瓶颈在 dataloader 的 CPU，而不在 GPU 显存。
- **127 是主力**：内存和 CPU 最充足，可以做 E0/E1/E3/E6。S/B 模型只需不到 16 GB 显存，可以和 XICM 共用 GPU。
- **3090 做并行扫参**：3 张卡可以同时跑 E1 的 3 个目标函数。前提是能用，需先确认预留情况。
- **Jubail 做 E2 scaling 的 L 模型和 E4/E5 的 LIBERO 闭环评估**（已有 LIBERO 环境；仿真作业使用 `--gres=gpu:a100:1`）。

## 路径

| 机器 | 代码（GitHub checkout） | 数据（`$SOPT_DATA`） |
|---|---|---|
| 127 | `~/langtian/SOPT` | `~/langtian/SOPT_DATA` |
| 3090 | `~/Desktop/langtian/SOPT` | `~/Desktop/langtian/SOPT_DATA` |
| Jubail | `/scratch/ll5582/SOPT` | `/scratch/ll5582/SOPT_DATA` |

## 流程
1. 本地提交后运行 `scripts/remote/sync.sh 127|3090|jubail|all`：先 push 到 GitHub，再在服务器上 `git pull --ff-only`。服务器只拉代码，不在服务器上直接改代码。
2. 首次部署：`git clone https://github.com/Autzoko/State-Only-Predictor.git SOPT && bash SOPT/scripts/setup_env.sh`。
3. 数据只在 127 上构建（`prepare_droid.py` 流式只取 state，原始 parquet 用完即删），然后把 `processed/droid/` rsync 到其他机器。
4. 训练：`source scripts/remote/env.sh`，然后运行 `GPU=0 scripts/remote/run_bg.sh <run_name> --config ...`（tmux 后台）。
