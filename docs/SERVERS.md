# 服务器

| 机器 | 用途 | 代码 / 数据路径 | 注意事项 |
|---|---|---|---|
| NYUAIRLAB-127（A6000 48GB） | 主开发机；E0/E1/E3/E6 单卡实验 | `~/langtian/SOPT`，`~/langtian/SOPT_DATA` | tmux 后台运行：`scripts/remote/run_bg.sh` |
| JUBAIL（Slurm，A100/H100/H200） | E2 scaling、E4/E5 LIBERO 闭环 | `/scratch/ll5582/SOPT`，`/scratch/ll5582/SOPT_DATA` | **同时最多 2 个作业**，逐步提交；仿真 rollout 用 `--gres=gpu:a100:1`（H100 上 EGL 有问题）；`ssh JUBAIL bash -lc ...`；/scratch 文件数配额紧张，数据用单个大 `.npy` |
| NYUAIRLAB-3090 | 备用 | `~/Desktop/langtian/SOPT` | 目前被他人预留，**启动任何任务前需先确认权限** |

工作流程：本地提交 → `scripts/remote/sync.sh <host>` → 远程 `bash scripts/setup_env.sh`（首次）→ 训练。
数据只需在一台机器上运行 `prepare_droid.py`，然后 rsync `data/processed/droid/{states,actions,features}.npy, episodes.parquet`（约 3 GB）到其他机器。
