# ManipArena Operations Guide

## 目录
1. [项目结构](#1-项目结构)
2. [环境准备](#2-环境准备)
3. [数据集](#3-数据集)
4. [pi0 训练与评估](#4-pi0-训练与评估)
5. [pi0.5 训练与评估](#5-pi05-训练与评估)
6. [DynaActVAE 训练与集成](#6-dynaactvae-训练与集成)
7. [开环评估](#7-开环评估)
8. [比赛提交](#8-比赛提交)
9. [常用路径速查](#9-常用路径速查)
10. [Config 速查表](#10-config-速查表)

---

## 1. 项目结构

```
/DATA/disk1/yjb/projects/VLA/
├── openpi/                      # OpenPI (pi0/pi0.5) 训练框架
├── maniparena-repo/             # ManipArena 比赛服务端（serve + eval）
├── maniparena-sim/              # ManipArena 仿真环境（Isaac Lab）
├── action-traj-vae/             # DynaActVAE（Action VAE + JEPA Predictor）
```

---

## 2. 环境准备

```bash
# 激活 openpi 虚拟环境（所有操作通用）
source /DATA/disk1/yjb/projects/VLA/openpi/.venv/bin/activate

# 设置缓存目录（另一台服务器上需要）
export OPENPI_DATA_HOME=/data/yjb/.cache/openpi
```

---

## 3. 数据集

### 下载

```bash
# 全量下载
hf download ManipArena/maniparena-dataset --repo-type dataset

# 只下仿真数据
hf download ManipArena/maniparena-dataset --repo-type dataset --include "sim/*"
```

### 数据位置

```
SNAP=/DATA/disk1/yjb/.cache/huggingface/hub/datasets--ManipArena--maniparena-dataset/snapshots/076e818a76a29d3ac930f840ab8af981d7f71e90
```

### 预赛 5 个任务

| 官网任务名 | 数据集路径 |
|-----------|----------|
| Press buttons in order | `$SNAP/real/semantic_reasoning/press_button_in_order` |
| Place blocks into colors | `$SNAP/real/execution_reasoning/put_blocks_to_color` |
| Classify items as shape | `$SNAP/real/semantic_reasoning/classify_items_as_shape` |
| Place ring on rod | `$SNAP/real/execution_reasoning/put_ring_onto_rod` |
| Put spoon to bowl | `$SNAP/real/execution_reasoning/put_spoon_to_bowl` |

### 仿真 3 个任务

| 数据集名 | 仿真环境名 | 数据集路径 |
|---------|-----------|----------|
| press_button_in_order | buttons_contact | `$SNAP/sim/press_button_in_order` |
| put_blocks_to_color | sort_blocks | `$SNAP/sim/put_blocks_to_color` |
| pick_fruits_into_basket | fruits_to_basket | `$SNAP/sim/pick_fruits_into_basket` |

### 数据格式

- 真机：56D（tabletop）/ 62D（mobile），前14D为EE
- 仿真：28D，前14D为EE
- EE布局：`[left_xyz(3), left_rpy(3), left_grip(1), right_xyz(3), right_rpy(3), right_grip(1)]`
- 图片：3个摄像头 480x640，20Hz
- 格式：LeRobot v2.1（parquet + mp4 video）

---

## 4. pi0 训练与评估

### 4.1 计算归一化统计量

```bash
cd /DATA/disk1/yjb/projects/VLA/openpi && source .venv/bin/activate

# 仿真单任务
python scripts/compute_norm_stats.py --config-name pi0_maniparena_sim

# 真机多任务（快速 parquet 路径，秒级完成）
python scripts/compute_norm_stats.py --config-name pi05_maniparena_real_all
```

### 4.2 训练

```bash
# 仿真 - press_button (5000 steps)
python scripts/train.py pi0_maniparena_sim --exp-name pi0_maniparena_sim_press_button

# 真机 - 单任务 arrange_cup (30000 steps)
python scripts/train.py pi0_maniparena --exp-name pi0_maniparena_arrange_cup
```

### 4.3 Checkpoint 位置

```
/DATA/disk1/yjb/projects/VLA/openpi/checkpoints/{config_name}/{exp_name}/step_XXXX/
```

---

## 5. pi0.5 训练与评估

### 5.1 计算归一化统计量

```bash
cd /DATA/disk1/yjb/projects/VLA/openpi && source .venv/bin/activate

# 预赛5任务
python scripts/compute_norm_stats.py --config-name pi05_maniparena_preliminary

# 全部20任务
python scripts/compute_norm_stats.py --config-name pi05_maniparena_real_all
```

### 5.2 训练

```bash
# 预赛5任务
python scripts/train.py pi05_maniparena_preliminary --exp-name pi05_maniparena_preliminary

# 全部20任务
python scripts/train.py pi05_maniparena_real_all --exp-name pi05_maniparena_real_all
```

### 5.3 关键参数差异 (pi0 vs pi0.5)

| | pi0 | pi0.5 |
|--|-----|-------|
| action_horizon | 50 | 10 |
| delta_actions | True | False |
| base checkpoint | pi0_base | pi05_base |
| batch_size | 32 | 256 |
| lr_schedule | 默认 | CosineDecay (5e-5) |

---

## 6. DynaActVAE 训练与集成

### 6.1 训练 Action VAE（第一阶段）

```bash
cd /DATA/disk1/yjb/projects/VLA/action-traj-vae
source /DATA/disk1/yjb/projects/VLA/openpi/.venv/bin/activate

# 无视频预测版本（不需要 Wan VAE，可立即跑）
python scripts/train_robotwin.py --config configs/train_maniparena_no_pred.yaml --wandb

# 完整版本（需要 Wan 2.2 Video VAE 权重）
python scripts/train_robotwin.py --config configs/train_maniparena.yaml --wandb
```

Checkpoint 输出：`checkpoints/maniparena_vae_no_pred/best.pt`

### 6.2 训练 pi0.5 + DynaActVAE（第二阶段，需要 Action VAE 训完）

```bash
cd /DATA/disk1/yjb/projects/VLA/openpi && source .venv/bin/activate

# 先算 norm stats
python scripts/compute_norm_stats.py --config-name pi05_dynaactvae_preliminary

# 训练
python scripts/train.py pi05_dynaactvae_preliminary --exp-name pi05_dynaactvae_preliminary
```

### 6.3 架构说明

```
训练阶段：
  raw actions → frozen VAE encoder → latent z̄_a (action_dim=16, action_horizon=3)
  pi0.5 的 flow matching 在 latent space 里做

推理阶段：
  pi0.5 生成 latent z̃_a → frozen VAE decoder → raw actions (14D EE)
```

---

## 7. 开环评估

### 7.1 启动模型服务

修改 `maniparena-repo/examples/my_policy.py` 中的 `OPENPI_CONFIG_NAME`：
- pi0 sim: `"pi0_maniparena_sim"`
- pi0.5 预赛: `"pi05_maniparena_preliminary"`
- pi0.5 全任务: `"pi05_maniparena_real_all"`
- pi0.5 + DynaActVAE: `"pi05_dynaactvae_preliminary"`

```bash
cd /DATA/disk1/yjb/projects/VLA/maniparena-repo
source /DATA/disk1/yjb/projects/VLA/openpi/.venv/bin/activate

python serve.py \
    --checkpoint /path/to/checkpoint/step_XXXX \
    --control-mode end_pose \
    --port 8000
```

### 7.2 运行开环评估

```bash
cd /DATA/disk1/yjb/projects/VLA/maniparena-repo
source /DATA/disk1/yjb/projects/VLA/openpi/.venv/bin/activate

SNAP=/DATA/disk1/yjb/.cache/huggingface/hub/datasets--ManipArena--maniparena-dataset/snapshots/076e818a76a29d3ac930f840ab8af981d7f71e90

python scripts/eval_openloop.py \
    --server ws://localhost:8000 \
    --dataset $SNAP/real/semantic_reasoning/press_button_in_order \
    --episode 0 \
    --save-dir openloop_plots/任务名 \
    --action-chunk 10
```

`--action-chunk`: pi0 用 50，pi0.5 用 10

### 7.3 预赛5任务批量开环

```bash
SNAP=/DATA/disk1/yjb/.cache/huggingface/hub/datasets--ManipArena--maniparena-dataset/snapshots/076e818a76a29d3ac930f840ab8af981d7f71e90

# 1. Press buttons in order
python scripts/eval_openloop.py --server ws://localhost:8000 \
    --dataset $SNAP/real/semantic_reasoning/press_button_in_order \
    --episode 0 --save-dir openloop_plots/press_button --action-chunk 10

# 2. Place blocks into colors
python scripts/eval_openloop.py --server ws://localhost:8000 \
    --dataset $SNAP/real/execution_reasoning/put_blocks_to_color \
    --episode 0 --save-dir openloop_plots/put_blocks --action-chunk 10

# 3. Classify items as shape
python scripts/eval_openloop.py --server ws://localhost:8000 \
    --dataset $SNAP/real/semantic_reasoning/classify_items_as_shape \
    --episode 0 --save-dir openloop_plots/classify_items --action-chunk 10

# 4. Place ring on rod
python scripts/eval_openloop.py --server ws://localhost:8000 \
    --dataset $SNAP/real/execution_reasoning/put_ring_onto_rod \
    --episode 0 --save-dir openloop_plots/put_ring --action-chunk 10

# 5. Put spoon to bowl
python scripts/eval_openloop.py --server ws://localhost:8000 \
    --dataset $SNAP/real/execution_reasoning/put_spoon_to_bowl \
    --episode 0 --save-dir openloop_plots/put_spoon --action-chunk 10
```

### 7.4 协议验证（可选）

```bash
# 握手测试
python scripts/mock_ping.py --uri ws://127.0.0.1:8000

# Schema 验证
python scripts/mock_schema_check.py --uri ws://127.0.0.1:8000
```

---

## 8. 比赛提交

### 8.1 提交流程

1. 部署模型服务到有公网IP的机器
2. 生成开环图（eval_openloop.py 输出的 jpg）
3. 在官网填写：
   - 舞台：初赛
   - 型号名称：如 `pi05_maniparena`
   - 模型类型：**端态**
   - 开环图：上传 jpg
   - 连接：`120.92.208.221:8000`（你的公网IP:端口）
4. 点"测试延迟"确认连通
5. 提交

### 8.2 启动服务（提交用）

```bash
python serve.py \
    --checkpoint /path/to/best/checkpoint \
    --control-mode end_pose \
    --port 8000
```

### 8.3 持续更新策略

截止日期前可以随时更新权重：停服务 → 换 checkpoint → 重启。只要评测时服务在线即可。

---

## 9. 常用路径速查

| 项目 | 路径 |
|------|------|
| openpi 项目 | `/DATA/disk1/yjb/projects/VLA/openpi/` |
| openpi venv | `/DATA/disk1/yjb/projects/VLA/openpi/.venv/` |
| maniparena-repo | `/DATA/disk1/yjb/projects/VLA/maniparena-repo/` |
| maniparena-sim | `/DATA/disk1/yjb/projects/VLA/maniparena-sim/` |
| action-traj-vae | `/DATA/disk1/yjb/projects/VLA/action-traj-vae/` |
| 数据集 snapshot | `~/.cache/huggingface/hub/datasets--ManipArena--maniparena-dataset/snapshots/076e...` |
| openpi checkpoints | `/DATA/disk1/yjb/projects/VLA/openpi/checkpoints/` |
| openpi assets (norm stats) | `/DATA/disk1/yjb/projects/VLA/openpi/assets/` |
| my_policy.py | `/DATA/disk1/yjb/projects/VLA/maniparena-repo/examples/my_policy.py` |
| maniparena_policy.py | `/DATA/disk1/yjb/projects/VLA/openpi/src/openpi/policies/maniparena_policy.py` |
| openpi config.py | `/DATA/disk1/yjb/projects/VLA/openpi/src/openpi/training/config.py` |

---

## 10. Config 速查表

| Config 名 | 模型 | 数据 | 步数 | 用途 |
|-----------|------|------|------|------|
| `pi0_maniparena_sim` | pi0 | sim/press_button (60eps) | 5,000 | 仿真快速验证 |
| `pi0_maniparena` | pi0 | real/arrange_cup (528eps) | 30,000 | 真机单任务 |
| `pi05_maniparena_preliminary` | pi0.5 | 预赛5任务 | 30,000 | **预赛提交** |
| `pi05_maniparena_real_all` | pi0.5 | 全部20任务 | 30,000 | 全任务训练 |
| `pi05_maniparena_sim` | pi0.5 | sim/press_button | 5,000 | 仿真快速验证 |
| `pi05_dynaactvae_preliminary` | pi0.5+VAE | 预赛5任务 | 30,000 | DynaActVAE版本 |
| `pi0_maniparena_low_mem` | pi0 LoRA | real/arrange_cup | 30,000 | 低显存版本 |

### 新增 Config 模板

在 `openpi/src/openpi/training/config.py` 中添加，参考现有 ManipArena configs，修改：
- `name`
- `local_root`（数据集路径列表）
- `assets > asset_id`（norm stats 存储 ID，避免覆盖）
