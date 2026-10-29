# 计算硬件与软件环境（汇总，2026-10-29）

版本号来自各机器上实际安装的包（Jubail：`site-packages` 下的 `*.dist-info`，只用 `ls` 读取）和代码中固定的 revision。每个结果目录的 `meta.json` 记录了运行时的 git commit；以下为论文附录"计算资源与可复现性"的依据。

## 1. 硬件

| 机器 | GPU | CPU / 内存 | 用途 | 状态 |
|---|---|---|---|---|
| **Jubail**（NYUAD HPC，Slurm，`nvidia` 分区，CentOS Linux 8） | NVIDIA A100 80GB 为主（`--constraint=80g`；分区另有 A100 40GB 节点） | 按作业申请：训练 16–30 核、100–360 GB；评测 24 核、160 GB | 当前全部计算：SOPT 先验预训练、所有宿主训练、闭环评测 | 主力 |
| **NYUAIRLAB-127** | 1× RTX A6000 48GB（与其他项目共用） | 48 核 / 250 GB | DROID 状态数据构建（E0）、DEV7 开发实验（E1–E18）、主先验 p0 预训练 | 2026-10-15 起只作跳板与文件中转，不再计算 |
| **实验室 DGX Spark** | 每节点 1× NVIDIA GB10（aarch64） | — | 2026-10-15 至 10-20：TrunkBC 17 任务（E23）、SmolVLA 主表原始 s0 / s1、E25 部分先验 | 目前不可达；结果作跨平台复制 |
| NYUAIRLAB-3090 | 3× RTX 3090 24GB | 20 核 / 62 GB | 仅复制过数据 | 未使用 |
| 本机（macOS） | — | — | 统计（簇 bootstrap、McNemar、Holm）、表格与图 | 不训练 |

Jubail 作业形态：
- 2026-10-26 前：每个作业 3 卡，每臂一卡并行。
- 2026-10-26 起：全部为单卡作业（`--gres=gpu:a100:1`），同时最多 4 个。
- 典型时长（单臂，训练 + 850 episode 评测）：DP / TrunkBC 1.2–1.4 h；SmolVLA 2.0–2.8 h；FastWAM 更长。
- 闭环评测以 CPU 上的 MuJoCo 仿真与 EGL 渲染为主（每次评测 17 个仿真进程），GPU 利用率约 15%；E41 起同一 GPU 上两次评测并行（`EVAL_PAR=2`）。

## 2. 软件环境

### Jubail（当前主力）
| 组件 | 版本 |
|---|---|
| Python | 3.12 |
| PyTorch / torchvision | 2.7.1+cu118 / 0.22.1+cu118 |
| LeRobot | 0.6.2（SmolVLA、π0.5、X-VLA、GR00T、FastWAM 的实现） |
| transformers / accelerate | 5.5.4 / 1.15.0 |
| diffusers | 0.39.0（`/scratch/ll5582/SOPT/pydeps`） |
| 仿真 | hf-libero 0.1.4、robosuite 1.4.0、MuJoCo 3.8.1，`MUJOCO_GL=egl` / `PYOPENGL_PLATFORM=egl`（PyOpenGL 3.1.10、egl-probe 1.0.2） |
| 数据与工具 | numpy 2.2.6、pandas 2.3.3、pyarrow 25.0.1、scipy 1.18.1、omegaconf 2.3.1、einops 0.8.2、PyAV 15.1.0、safetensors 0.8.0、huggingface_hub 1.33.0 |
| pydeps 额外依赖 | dm-tree 0.1.10、sentencepiece 0.2.2、ftfy 6.3.1 |

- 解释器与主体依赖复用 RoboticsNAS 项目环境 `/scratch/ll5582/RoboticsNAS/env`（只读，不修改）；`PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src`。
- 计算节点离线运行（`HF_HUB_OFFLINE=1`），权重预缓存在 `/scratch/ll5582/SOPT_DATA/hf_cache`。

### DGX Spark（历史）
- VLA 环境：Python 3.12，torch 2.11.0 / torchvision 0.26.0（cu130，aarch64 构建），LeRobot 0.6.2。
- 基础环境（TrunkBC 等）：Python 3.11，torch（cu130）。
- 各节点解包同一个打包好的 venv（`scripts/dgx/common.sh`）。

### 127（历史）
- 仿真栈与 Jubail 相同：hf-libero 0.1.4、robosuite 1.4.0、MuJoCo 3.8.1（EGL），独立 `LIBERO_CONFIG_PATH`，不改共享 `~/.libero`。
- VLA 环境 `SOPT_VLA_ENV`：LeRobot 0.6.2，另补装 diffusers 0.41。

## 3. 数据与模型版本

| 用途 | 仓库 @ revision |
|---|---|
| DROID 状态（只读状态列与元数据，约 1.4 GB，不下载视频） | `cadene/droid_1.0.1_v30` @421fe53b89b073df8c9231d03650afd71f53c79e（95,584 条） |
| LIBERO-90 | `IPEC-COMMUNITY/libero_90_no_noops_lerobot` @70696aef03def70c17917f43c5e79276b7e5fbe7 |
| SmolVLA | `lerobot/smolvla_base` @d9f33c94a60fb382c90dea2164c96845bd955e28；VLM `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` @7b375e1b73b11138ff12fe22c8f2822d8fe03467 |
| π0.5-DROID | `lerobot/pi05_droid` @72824c0a93f00ce5bb8bedb7feb58953ba1da364；PaliGemma tokenizer 本地副本 |
| GR00T N1.7 | `nvidia/GR00T-N1.7-3B` @2fc962b973bccdd5d8ce4f67cc63b264d6886495；processor `nvidia/Cosmos-Reason2-2B` @9ce19a195e423419c349abfc86fd07178b230561 |
| X-VLA | `lerobot/xvla-base` @cdb7964e4fe842935d671bfab5a5ebe00a96648c；tokenizer `facebook/bart-large` @cb48c1365bd826bd521f650dc2e0940aee54720c |
| FastWAM | `lerobot/fastwam_base` @385a4bd1df7a7895b24c6e530482092b7b08a783；`Wan-AI/Wan2.2-TI2V-5B-Diffusers` @b8fff7315c768468a5333511427288870b2e9635；`google/umt5-xxl` @66cb9e7e85526fe440a945569e42c72fb6cbc0ad |
| Diffusion Policy | 本仓库实现：ImageNet ResNet-18（每相机，空间 softmax 关键点）+ 1-D 条件 U-Net（512/1024/2048），DDIM 100 / 10 步；无机器人预训练 |
| DINOv2 基线 | `facebook/dinov2-small` @ed25f3a31f01632728cabb09d1542f84ab7b0056 |
| SOPT 主先验 p0 | `outputs/e2_ar_flow_f100_s`：S 规模（约 9.8M 参数），ar_flow，上下文 96 帧（6.4 s @15 Hz），预测 32 帧（2.1 s），40k 步，100% DROID 训练划分，按 DROID val（UW）选模 |

注：FastWAM 三项 revision 是 Jubail 缓存 `refs/main` 指向的快照（离线运行，所有 FastWAM 结果均用此快照）；`src/sopt/vla/fastwam.py` 中尚未写死 revision，其余宿主均已在代码中固定。

## 4. 各实验运行位置

| 实验 | 硬件 |
|---|---|
| E0 数据构建，E1–E18（DEV7 开发集），p0 预训练 | 127（A6000） |
| E23 TrunkBC 17 任务原版、SmolVLA 主表原始 s0 / s1、E25 部分先验 | DGX Spark（GB10） |
| 主表 DP、FastWAM、π0.5、GR00T、X-VLA；SmolVLA 主表 `_jr` 重跑与 s2；TrunkBC 重跑（E25j） | Jubail A100 |
| E31–E41、E18r / E11r、E38、E39、E40、数据验证 DV1–DV3 | Jubail A100 |
| 统计分析、表格、图 | 本机 |

主表中 TrunkBC 与 SmolVLA 均采用 Jubail 结果；DGX 上的原始结果作为跨平台复制报告（逐 seed 差异在 ±3 点以内，结论不变）。
