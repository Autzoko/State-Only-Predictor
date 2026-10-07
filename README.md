# SOPT: State-Only Pretraining of Robot Motion Priors

只用机械臂本体状态序列（不用视觉、语言）进行自监督预训练，得到可迁移的动作/运动先验。
研究设计见 [`docs/DESIGN.md`](docs/DESIGN.md)，实验计划与记录见 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)，服务器说明见 [`docs/SERVERS.md`](docs/SERVERS.md)。

## 结构

```
configs/           default.yaml + model/{s,b,l}.yaml + experiment/*.yaml（按顺序叠加，可再加 key=value）
docs/              DESIGN / EXPERIMENTS（实验日志）/ SERVERS
scripts/           download_droid.py, prepare_droid.py, train.py, evaluate.py, setup_env.sh
  remote/          sync.sh（rsync 到 127/3090/jubail）, run_bg.sh（tmux 后台训练）
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
bash scripts/setup_env.sh && conda activate sopt
python scripts/download_droid.py --dest data/raw/droid_1.0.1           # 仅 parquet，约 12 GB
python scripts/prepare_droid.py --src data/raw/droid_1.0.1 --out data/processed/droid --workers 32
python scripts/train.py --config configs/model/s.yaml --config configs/experiment/e1_ar_flow.yaml run_name=e1_ar_flow_s
python scripts/evaluate.py --ckpt outputs/e1_ar_flow_s/best.pt --split val
pytest
```

本地流水线检查（30 条 episode，CPU，约 1 分钟）：
```bash
python scripts/download_droid.py --dest data/raw/droid_debug --chunks 0 30 60 --max-per-chunk 10
python scripts/prepare_droid.py --src data/raw/droid_debug --out data/processed/droid_debug --workers 4
python scripts/train.py --config configs/experiment/debug.yaml
```
