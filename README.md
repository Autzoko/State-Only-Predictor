# SOPT: State-Only Pretraining of Robot Motion Priors

只用机械臂本体状态序列（不用视觉、语言）进行自监督预训练，得到可迁移的动作/运动先验。
研究设计见 [`docs/DESIGN.md`](docs/DESIGN.md)，实验计划与记录见 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)，服务器说明见 [`docs/SERVERS.md`](docs/SERVERS.md)。

## 结构

```
configs/           default.yaml + model/{s,b,l}.yaml + experiment/*.yaml（按顺序叠加，可再加 key=value）
docs/              DESIGN / EXPERIMENTS（实验日志）/ SERVERS
scripts/           prepare_droid.py（流式只取 state）, train.py, evaluate.py, setup_env.sh
  remote/          sync.sh（push + 服务器 git pull）, env.sh, run_bg.sh（tmux 后台训练）
  slurm/           train.sbatch（Jubail）
src/sopt/
  data/            droid.py（parquet→数组）, features.py（17 维特征）, splits.py（按 building 切分）,
                   dataset.py（窗口采样 + 速度增广）, augment.py（运动学精确的 yaw 增广）, normalize.py
  models/          transformer.py（RoPE, pre-LN）, heads.py（回归 / rectified flow）, state_prior.py
  eval/            forecast.py（ADE/FDE、minADE@K，zero-vel / const-vel 基线）
  train/           pretrain.py
  utils/           rotation.py, franka_fk.py, config.py, misc.py
tests/             运动学/增广正确性、数据集、三种目标函数的 shape 测试
data/, outputs/    不进 git
```

## 快速开始

```bash
bash scripts/setup_env.sh && source scripts/remote/env.sh   # env 位于 ../SOPT_ENV
# 只通过列投影读取 state 列（约 1.4 GB 网络传输，不下载视频，可断点续传）
python scripts/prepare_droid.py --out data/processed/droid --workers 16
python scripts/train.py --config configs/model/s.yaml --config configs/experiment/e1_ar_flow.yaml run_name=e1_ar_flow_s
python scripts/evaluate.py --ckpt outputs/e1_ar_flow_s/best.pt --split val
pytest
```

本地流水线检查（2 个文件，93 条 episode，CPU，约 1 分钟）：
```bash
python scripts/prepare_droid.py --out data/processed/droid_debug --max-files 2 --workers 2
python scripts/train.py --config configs/experiment/debug.yaml
```
