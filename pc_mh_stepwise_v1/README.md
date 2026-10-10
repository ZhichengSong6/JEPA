# PC-MH：逐步观测锚定 target 扩展 v1

基于 `ZhichengSong6/JEPA` 的已核查 commit：
`851b60c06ee0aef6927d59f85763cb03a3aede34`。

这是新增 C 组的实现，不是已有 PC-MH 的重训脚本。本实验保存在独立 GitHub 分支，
不会修改已有 PC-MH 训练代码、checkpoint、评估脚本或 IDM。新增程序放在仓库下的
独立 `pc_mh_stepwise_v1/` 目录。

## 1. 方法定义

对于真实 latent 轨迹 `z[0:H+1]`、真实动作 `u[0:H]` 及局部扰动动作 `u'`：

```text
reference[k] = T(z[max(0,k-2):k+1], u[max(0,k-2):k+1])[-1]
y[0]         = z[0]
y[k+1]       = z[k+1] + T(y-history, u'-history)[-1] - reference[k]
```

每条分支都从一个观测开始，history length 为 `1,2,3,3,...`。
事实 teacher 分支第 k 步只读取 `z[0:k+1]`，不会把 `z[k+1]` 输入 predictor；
`z[k+1]` 只在该步预测之后作为 anchor 使用。扰动 target 分支递归使用之前
构造的 `y`，不是每步直接重置为真实 z。

Student 使用原 `planner_context_mh.planner_context_rollout()`：
只读取 `z[0]`，之后始终用自己的预测。所有 target 计算均 `torch.no_grad()`。

总目标仍为原权重下的：

```text
TF + demonstrated planner-context rollout + causal MH target regression
```

TF 的三帧真实 context、student 的 1→2→3 rollout、原 raw-action normalization、
bounded symmetric probe sampler、radius、每个动作位置的 probe 数、causal mask
及 loss reduction 均沿用原实现。没有加入新的 loss 权重、goal/readout、IDM、
CEM weighting、teacher 更新、扰动状态 clipping 或真实 simulator rollout。

这些 targets 仍是观测约束下的模型推断，不是真实反事实标签；效果需要实际规划评估。
训练过程中不运行 CEM。

**不要沿用旧 PC-MH 的 initial half-response loss=0、response gain=1、
init equivalence ratio=1 检查。** 新构造有意改变多步 target response。
仍应满足：零扰动回到真实轨迹；H=1 时与原目标等价；损失 midpoint/response 分解成立。

## 2. 文件

- `planner_context_stepwise_mh.py`：新 target 和训练 forward。
- `train_pc_stepwise_mh.py`：从旧运行配置派生新实验；调用原训练入口；保护和导出。
- `submit_pc_stepwise_mh.sh`：按旧配置的 GPU/world size 提交 Slurm/srun 作业；不申请 --mem。
- `test_pc_stepwise_mh.py`：CPU 单元、梯度、配置及集成检查。
- `LOCAL_TEST_RESULTS.txt`：本地运行记录，明确区分 fixture 和真实仓库测试。

不需要安装新 Python 包，使用服务器原 `lewm` 环境。

## 3. 从 GitHub 实验分支获取代码

实验分支 `experiment/pc-mh-stepwise-v1` 以 `agent/stage1-bias-calibration`
的 `851b60c` 为父 commit。原分支没有改动。

如果服务器仓库存在未跟踪文件，先确认 `git status --short`，避免同名路径冲突；
不要自动清理或覆盖现有文件：

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official
git fetch origin experiment/pc-mh-stepwise-v1
git switch --track -c experiment/pc-mh-stepwise-v1 origin/experiment/pc-mh-stepwise-v1
```

然后使用下面的测试及训练命令。不要解压本地 tarball 覆盖已检出的文件。

## 4. 真实仓库的 CPU 单元检查

```bash
source /mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh
conda activate lewm
export STABLEWM_HOME=/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data
cd /mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official
python pc_mh_stepwise_v1/test_pc_stepwise_mh.py --with-repo
```

这一检查不读取完整数据集和正式模型，不训练 LeWM；使用合成的微型模型验证代码。
`--with-repo` 会使用服务器上的真实 sampler/rollout，并运行 H=1 的原版/新版
完整 loss 与梯度对照。预期显示 `STEPWISE UNIT TESTS PASS`，且 0 skipped。

本地已执行 25 项测试中的 24 项；1 项依赖服务器真实仓库的对照未在本地执行。
本地集成/runner 测试使用明确的 fixture backend，不等于实际 Lightning、Slurm、
DDP 或真实 checkpoint 训练通过。GPU 端到端路径尚未在本地运行。

## 5. 复用旧 PC-MH 的实际配置

默认基线配置路径：

```text
$STABLEWM_HOME/pusht_pc_mh_h5_r01565_seed3072_ep10_ddp4/config.yaml
```

路径来自已核查的原 PC-MH 配置，但这里没有远程访问服务器验证该文件。
若旧运行实际在其他目录，显式指定：

```bash
export BASE_CONFIG=/实际的旧PC-MH运行目录/config.yaml
```

**必须是旧运行保存的 config.yaml，不是 `config/train/lewm_pc_mh_ald.yaml` 默认模板。**
程序继承旧配置的 init/teacher policy、seed、epochs、global batch、DDP、precision、
train split、sampler 参数、数据窗口、optimizer 等，正式实验仅改变输出标识；
teacher target 的变化由新 forward 实现。

若保存的 `trainer.devices` 是 `auto` 而非确定整数，程序会停止，不会从目录名猜测
GPU 数量。先用旧启动记录确认实际值，避免误把 4 GPU × 每卡 batch 改成 1 GPU。

程序对六个关键旧源码文件检查 Git blob hash。存在差异时停止，不自动 checkout/reset。
源码哈希验证不能验证旧数据文件是否被外部修改；默认沿用原数据和原环境。

## 6. 提交两步 smoke（同原 DDP 设置）

```bash
bash pc_mh_stepwise_v1/submit_pc_stepwise_mh.sh smoke
```

保留原 devices、local batch 等，只将 max_epochs=1、max_steps=2、
limit_train_batches=2、limit_val_batches=1。使用真实数据和旧父 checkpoint，
验证训练路径。不是研究性的额外诊断，更不是重训旧基线。

默认 smoke 输出目录：

```text
$STABLEWM_HOME/pusht_pc_mh_h5_r01565_seed3072_ep10_ddp4_stepwise_v1_smoke/
```

日志位于该目录 `slurm_JOBID.out` 和 `slurm_JOBID.err`。
依次应看到 `PARENT STATE CHECK`、`FIT AUDIT PASS`、`FIRST BACKWARD PASS`、
`SAVED PLAIN JEPA`、`STEPWISE PC-MH SMOKE PASS`。还应检查 Slurm 作业以 0 退出。

## 7. 正式训练：只新增 C 组

smoke 成功后运行：

```bash
bash pc_mh_stepwise_v1/submit_pc_stepwise_mh.sh formal
```

沿用旧配置第二阶段的 epoch 数（当前研究设置应是 10），不是在旧 PC-MH 成品上
再多训 10 个 epochs。初始化仍是旧 PC-MH 的父 LeWM10；teacher 仍是同一个父模型。

默认正式输出目录：

```text
$STABLEWM_HOME/pusht_pc_mh_h5_r01565_seed3072_ep10_ddp4_stepwise_v1/
```

默认新模型文件：

```text
lewm_pc_mh_h5_r01565_ddp4_stepwise_epoch_10_object.ckpt
```

所有具体名称由旧运行配置派生，提交输出会显示实际路径。最终标志是
`STEPWISE PC-MH FORMAL PASS` 和 `training_complete.json`，不是 planning success。

如需指定节点：

```bash
NODE=4090node2 bash pc_mh_stepwise_v1/submit_pc_stepwise_mh.sh formal
```

默认不锁定单个节点，但排除旧记录中有启动故障的 3090node1。
按旧配置申请 GPU 数，可能需要等待整节点对应资源。没有 --mem 请求。
`--time` 仅为调度上限，不是运行时长承诺。

## 8. 安全、记录与对比

新目录有：`execution.json`、`baseline_config.yaml`、`launch_config.yaml`、
`fit_audit.json`、`training_progress.csv`、每 epoch inference objects、
原框架生成的 training weights，以及成功结束后的 `training_complete.json`。

- 初始化时检查 student/teacher 全部参数与 buffers 的一致性。
- teacher 全冻结；encoder/projector 全冻结；仅 action_encoder/predictor/pred_proj 有梯度。
- 每 epoch 验证视觉参数及 buffers、teacher 全部 state 未改变。
- checkpoint 原子保存，保存失败导致作业失败，不误报完成。
- inference object 仅保存原 JEPA 模型，不带新 target 构造器或 teacher。
- 输出目录已存在时拒绝重新 prepare；已启动的目录拒绝重新 run；无自动 resume。
- 提交失败但还没启动时，可直接重新 `sbatch 新目录/job.sbatch`。
- 运行失败后保留日志和已有输出，修复后使用新 tag，如 `formal v2`，不要删除旧实验覆盖结果。

与现有 PC-MH 成品比较时，沿用旧 evaluation 配置和 CEM seeds，仅替换 policy。
本包不替你更改评估任务、预算或指标，也不自动提交完整评估。

同样 epochs/更新步数不代表同样计算量：新 target 多了一条 factual teacher 分支。
`execution.json`/`started.json`/`training_complete.json` 与 Slurm accounting 可用于记录耗时。
这个实现没有保证规划提升；其用途是进行既定的 B（已有）对 C（新增）方法比较。
