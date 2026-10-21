# PC-anchor-only v1：中心监督匹配对照

这是 PC-MH 的**消融对照**，不是一个新的改进方法。只新增本目录，原 PC-MH、
Stepwise、多锚点、数据集、旧 checkpoint 和官方 CEM 都不修改。

## 目标

保持原 PC-MH 的 TF 与单帧起步的 `1->2->3` factual rollout；把 MH 项替换为
使用**同一因果掩码和归约函数**的 factual-center MSE。

对 H=5、每个动作位置相同探针数，目标为：

```
L = w_tf * L_TF + w_rollout * mean_h(MSE_h)
    + w_mh * sum_h(h * MSE_h) / 15
```

本版读取旧 run 的 1/1/1 权重，不重新调权重。`anchor_objective.center_mask_loss`
直接复用 `_mh_causal_mask` 与 `_mh_masked_mse`，不手写另一套训练归约。
其值与梯度等于把旧 MH 的两个 probe **显式替换为 U0** 后的中心监督。
不是把旧 sampler 的半径改成 0；原 sampler 不允许这个值。

没有扰动动作、teacher response loss、额外查询、残差库、模拟器或排序标签。
原入口仍加载 frozen teacher，以保持初始化路径；本 forward 从不调用 teacher。
没有模型内新层或部署时的额外模块。

此对照保留额外中心监督的系数和多时域权重，不声称与非零扰动损失有相同梯度、
相同优化轨迹或相同 FLOPs。没有探针采样，自然也不复现其随机数消耗；数据划分、
loader、seed、优化器、scheduler、更新步数及冻结设置沿用原入口。

## 代码与检查

- `anchor_objective.py`：中心对照 forward。
- `anchor_runtime.py`：配置派生、原依赖哈希、路径及成品 smoke 校验。
- `train_anchor.py`：调用未修改的 `train_pc_mh_ald.run`，只替换 forward 和审计导出回调。
- `submit.sh`：单节点、多 GPU srun；默认 4090node2，不申请 --mem。
- `test_anchor.py`：37 项 CPU 测试；可额外执行真实仓库函数等价性和 fresh-process 导入检查。

默认节点为 4090node2，允许 `4090node2/3` 或 `3090node1/2/3`。所有模式拒绝
4090node1，临时文件用节点本地短 `/tmp/pcao.*` 路径；数值结果仍在 STABLEWM_HOME。

原来的配置和依赖已发生变化时，源码检查会停止。不修改哈希、不覆盖本地旧文件
来绕过检查。配置读取先导入 stable_pretraining，再设置仅允许算术的 resolver，
避免以前的 eval 重复注册错误；未升级任何依赖。

## 服务器命令

```bash
source /mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh
conda activate lewm
export STABLEWM_HOME=/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data
export BASE_CONFIG="$STABLEWM_HOME/pusht_pc_mh_h5_r01565_seed3072_ep10_ddp4/config.yaml"
cd /mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official

(cd pc_mh_anchor_only_v1 && sha256sum -c SHA256SUMS.txt)
python pc_mh_anchor_only_v1/test_anchor.py --with-repo --base-config "$BASE_CONFIG"
```

应包含：

```
REAL CONFIG -> TRAINING IMPORT PASS: ...
PINNED REAL FORWARD ZERO-PROBE VALUE / GRADIENT EQUIVALENCE PASS
ANCHOR CPU TESTS PASS: 37; real LeWM weights/data/GPU/DDP NOT RUN
```

`--with-repo` 的等价性检查使用真实的旧 forward 和 mask，但模型是微型 Toy，
不是 LeWM checkpoint。导入检查只读 YAML，不读数据集、不创建 Trainer。
没有增加研究性的 gate，不需要重新 prepare 残差库。

### 先做两个优化步骤的 DDP smoke

```bash
NODE=4090node2 bash pc_mh_anchor_only_v1/submit.sh smoke v1
```

从旧配置读取**实际 GPU 数、每卡 batch 和 worker 数**。例如 4 GPU / batch 64
对应 global batch 256；不会因为 smoke 而静默改成单卡。smoke 与正式训练只更改
输出路径及必要的 debug 限制。配置写入后的源码再变化会停止，不复用旧协议。

```
ANCHOR FIT AUDIT PASS
ANCHOR FIRST BACKWARD PASS
SAVED PLAIN JEPA: ...
PC-ANCHOR-ONLY SMOKE PASS: ...
```

确认 Slurm `COMPLETED 0:0`、所有 rank 完成且无失败文件，然后再运行：

```bash
NODE=4090node2 bash pc_mh_anchor_only_v1/submit.sh formal v1
```

正式入口会验证同 tag smoke 的完成记录、源代码、配置、父 checkpoint 和环境。
只从原 `lewm_epoch_10` 开始这一个对照分支；不会在成品 PC-MH 上额外训练十轮。
只在实际完成所需 epoch/batch 并导出最终模型后，生成 `training_complete.json`。
这里通过表示执行正确，不表示优于旧 PC-MH。

## 输出与日志

```
$STABLEWM_HOME/pcmh_anchor_only_v1_smoke/   # 两步 smoke
$STABLEWM_HOME/pcmh_anchor_only_v1/         # 正式训练
$STABLEWM_HOME/pcmh_anchor_logs/smoke_v1_JOBID.out/.err
$STABLEWM_HOME/pcmh_anchor_logs/formal_v1_JOBID.out/.err
```

每个 run 中：`protocol.json` 保存父配置/父权重和旧 PC-MH 比较权重哈希、数据文件
信息、源码/版本、配置差异及 GPU/global batch。Lightning checkpoint 根目录也
隔离到新 run，避免写入代码目录里旧的 checkpoints。

`fit_audit.json`：参数范围、初始化、冻结 visual frame 审计。
`training_progress.csv`：从 forward 的直接数值读取，跨 rank 平均，首个/每 200 batch 写入。
`epoch_metrics.csv`：整个训练 epoch 的均值；不会把缺失 metric 写成空 loss 或零。
`*_epoch_10_object.ckpt`：正式推理模型，原 JEPA 类。
`training_complete.json`：含准确 policy、模型 SHA256、所有 rank 的完成情况。

```bash
OUT="$STABLEWM_HOME/pcmh_anchor_only_v1"
cat "$OUT/training_complete.json"
tail -n 5 "$OUT/epoch_metrics.csv"
tail -n 60 "$STABLEWM_HOME"/pcmh_anchor_logs/formal_v1_*.err
```

`FAILED_rank*.json` 和 `BLOCKED.json` 是执行错误；不要仅凭 squeue 消失判断成功。
不覆盖输出、不自动 resume。若 sbatch 尚未提交成功且 started.json 不存在，保留
生成的 job.sbatch，配置/源码未改变时可重新提交它。若作业已经启动后失败，先检查
异常；需要新尝试时使用新 tag（smoke v2，然后 formal v2），不删除旧证据。

## 对比范围与后续

已有 PC-MH checkpoint 复用。此实验是相同更新步数、非等计算量的消融。
正式训练不会自动运行 evaluation。B1000 主指标、B3000 辅指标应使用同一固定
起点/goal/reset/CEM 配置配对评估旧 PC-MH 与新对照；没有把历史 93.6% 单个均值
当成新的配对实验结果。一个训练 seed 配五个 eval seed 不是五个训练重复。

不使用多锚点 rejected cache 或 diagnosis replay；旧结果全部保留。

## 本地验证范围

37 项 CPU 数学、接口替代和微型训练回归测试已执行通过，Python/Bash 语法检查通过。
涵盖 h=1/3/5 权重、梯度、重复方向、冻结梯度/BN buffers、无 teacher 前向、
进程独立 resolver 导入顺序、原入口替代驱动、原模型导出、非空 loss CSV、
提前结束拒绝和失败后的函数恢复。

**没有在作者环境运行真实 LeWM 权重、PushT 数据、Slurm 或 DDP GPU 训练。**
服务器必须先执行 fresh-process/真实仓库检查与两步 GPU smoke；没有新规划成绩。
