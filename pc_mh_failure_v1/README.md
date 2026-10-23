# PC-MH / Stepwise failure-case analysis v1

只读诊断，不训练 LeWM/IDM，不修改 CEM，不把模拟器反事实结果用作训练监督。

## 运行

在现有 `LeWM_official` 仓库中使用。依赖现有诊断及 Stepwise 目录；不要单独放进另一个仓库。

```bash
source /mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh
conda activate lewm
export STABLEWM_HOME=/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data
cd /mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official
python pc_mh_failure_v1/test_failure.py
bash pc_mh_failure_v1/submit.sh v1
```

默认原 PC-MH 配置：
`$STABLEWM_HOME/pusht_pc_mh_h5_r01565_seed3072_ep10_ddp4/config.yaml`

默认 Stepwise：
`$STABLEWM_HOME/pusht_pc_mh_h5_r01565_seed3072_ep10_ddp4_stepwise_v2/training_complete.json`

从以上文件读取实际 policy、teacher 和 checkpoint。验证 Stepwise 是从同一份 baseline run config 准备的，核对 completion SHA256，并要求三模型 encoder/projector 完全一致。不猜测其他 checkpoint。

资源：1 GPU、1 进程，不用训练 DDP；不申请 `--mem`；临时路径为项目目录下 `t${SLURM_JOB_ID}`，避免 AF_UNIX 路径过长。不删除临时目录或旧实验。

默认 42–46 五个 seed，每次始终保留 100 个环境，CEM N/I/K=100/10/10，H=5、action_block=5、execute=25、eval_budget=50。只对固定选出的最多 12 个案例深入回放，两轮 CEM population（迭代 0、9，零起始）。没有增加规划预算。选择配额为双方失败4、Stepwise退步3、救回3、共同成功2；不足按固定规则补齐。不是手挑视频。

## 一次作业的三个阶段

1. **survey**：两个模型各跑完整 cohort，保存所有逐案例成败、reset recipes、native action/state 轨迹。生成 `cases.csv` 和分层随机选择的 `selected_cases.csv`。
2. **trace**：仍用完整 100 环境重跑，但仅被动记录所选案例的 CEM 候选。与 survey 比较全部案例的 success、动作、完整物理轨迹和 reset counts，验证 tracing 没有改变行为。复用现有 dataset/live pixel 来源校验和 prefix 精确回放。任何不一致写 `BLOCKED.json` 并停止，不降低容差。
3. **analyze**：同一 source history 和同一 population 由两个模型交叉评分；回放全部 100 候选及两个模型/encoder 的 elite mean。计算 target 误差分解及少量联合动作探针，生成报告。

**重要：这是新的、显式固定 native reset 的诊断 cohort。** 不能把它自动叫作旧 93.6% / 92.8% 的逐案例重现，也不能用新 cohort 的差值替换 headline benchmark。旧 `.txt` 的 aggregate 数字没有记录候选群、完整 reset/hidden state，不能从它们重建精确反事实。代码没有伪造旧案例身份。数据集抽样方式仍使用仓库原方法。

## 查看结果

```bash
OUT="$STABLEWM_HOME/pcmh_failure_v1"
tail -n 60 "$OUT"/slurm_*.out
tail -n 60 "$OUT"/slurm_*.err
cat "$OUT/COMPLETE.json"
python pc_mh_failure_v1/run.py summarize --output "$OUT"
```

主要文件：

- `REPORT.md`：cohort 结果和解释边界。
- `cases.csv`：完整 500 案例（seed、dataset row、episode/start、两模型结果和组别）。
- `selected_cases.csv`：最多12个深入分析案例。
- `analysis/populations.csv`：同群交叉评分、候选覆盖、encoder mismatch、elite-mean failure、预测/target 误差分解。
- `analysis/joint_response.csv`：按 action-block pair、horizon、模型记录真实有限交互响应和误差。
- `analysis/images/`：实际目标帧、实际回放起点、原模型与 encoder 选中的候选终点、elite mean 终点；不是模型渲染的未来。
- `analysis/populations/*.npz`、`analysis/joint/*.npz`：数值明细。
- `capture/seed*/..._traces.pt`：原始可信本地 traces；不纳入小结果包。

`FAILURE DIAGNOSTIC ALL PASS` 和 `COMPLETE.json` 表示诊断流程完成，**不是方法提升**。`BLOCKED.json` 表示某项协议校验或执行失败。阶段完成记录带 artifact hashes；已完成阶段可校验后复用，半成品阶段不静默重跑/混写。不要再次执行 `submit.sh v1` 覆盖目录；保留日志排错。提交未成功时可重新提交生成的 `job.sbatch`。

无需第三次独立 GPU smoke：本地测试之后，`all` 自带 repeatability/replay gate，不通过就不会进入反事实机制分析。若只做单 seed 先导，可用 `SEEDS="42" MAX_CASES=4 bash pc_mh_failure_v1/submit.sh pilot`，但它不是完整五 seed 结果。

## 指标的严格解释

- `predictor_top1_miss`：同一 population 存在 ever-success 候选，实际 encoder cost 的 argmin 成功，但模型 cost 的 argmin 失败。是局部证据，不是整条 episode 必被救回。
- `encoder_top1_task_mismatch`：有成功候选，但实际 encoder cost 的 argmin 失败。不能用它宣称所有 predictor 训练都无用。
- `no_success_in_recorded_population`：本次记录的 population 没有成功候选，不证明全局不可达或规划器永远找不到。
- `elite_mean_failure`：有成功 elite，但该评分方式的 elite mean 失败。最后一轮另核对 source elite mean 与真正返回动作一致。
- 上述标记可以重叠，不强行分成互斥的根因标签。selected populations 的频率不是总体 failure 占比，不做伪独立显著性检验。
- 所有 encoder costs 使用同一个 goal 和固定第25步 endpoint。另存 ever-success、first success、endpoint success；不能混用。仅诊断时 native replay 继续执行至25步，即使中途成功；不改变 closed-loop 终止规则。

### Target 分解

在一个实际返回动作 `U0` 周围，回放其真实未来（diagnosis only），构造 PC-MH 和 Stepwise 各自的参考 target。对每个候选计算：

`prediction_mse = reference_fit_mse + reference_target_mse + cross_term`。

**这是以 on-policy 返回动作为中心的诊断 reference，不是历史训练 loss，也不是证明 teacher 在原离线训练分布上的质量。** broad CEM candidates 可能超出局部校准半径，故不把低 reference gap 自动解释为拟合成功。两个 target 各自对应各自模型；保存 cross term，不能把两个平方误差当作可相加的占比。也记录 elite-union 子集。

### 联合动作探针

固定 pair `(0,1),(0,4),(2,4)`；每对包括四个单块小探针、四个联合正负角点、四个**等总L2幅度**的单块探针。总半径0.1565，若边界不允许则对称缩小并记录实际半径。不能通过分别裁剪破坏对称性。

混合有限响应为 `(F++ - F+- - F-+ + F--)/4`，不是精确 Hessian。比较 teacher、PC-MH、Stepwise、各自 reference targets 与模拟器回放的响应。在 `causally_joint_affected=True` 的行上解释；先看真实信号大小，近零时相对误差为空。匹配幅度的单块对照用于排除“联合扰动更大”的解释。

脚本不自动宣布“联合响应是瓶颈”，也不自动启动新训练。是否继续需要结合重复出现的决策失败、teacher 是否更有用、matched-single 对照及真实交互信号。

## 验证范围

本地 25 项合成/替代依赖测试通过；包括分组选择、误差分解、matched-radius 探针、因果交互量、只读权重hash、完整100环境 survey 调用、source rescore 不符时先阻断回放。

本容器没有实际 stable-worldmodel/PushT/CUDA，因此**未在真实服务器跑通**。真实协议由运行时 gate 校验。未修改任何原训练/评估脚本；不新增第三方依赖，要求当前 installed stable-worldmodel==0.0.6。
