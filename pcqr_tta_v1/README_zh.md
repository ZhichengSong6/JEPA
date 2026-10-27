# PCQR-TTA v0.1：规划查询响应保持的测试时适应

这是上一轮方案的第一版实现，不是新的有效性结论，也不是重新训练 PC-MH。
代码只新增 `pcqr_tta_v1/`，不修改原 `jepa.py`、`eval.py`、CEM 源码、旧 checkpoint 或旧结果。

## 已实现和验证范围

已实现：真实已完成动作块缓存、多步观察监督、更新前候选响应缓存、三步 Adam 更新、五种匹配模式、单环境 SWM policy/CEM 接入、新的配对 PushT pilot、Slurm smoke 脚本。

本地验证使用 CPU PyTorch 和测试替身。**没有在作者服务器、真实 LeWM checkpoint、真实 SWM 环境或 GPU 上执行闭环实验。**
测试通过只能说明相应代码性质通过，不能说明成功率提高，不能代替服务器 smoke。

运行时额外检查：当前 checkpoint 的原生 `JEPA.rollout` 与新可微 rollout 数值相符；不同方法对应 case 的初始 pixels/goal/state 哈希一致；观测里的上一动作与实际返回动作对应；每个 episode 结束时冻结参数及全部 buffers 未变。检查失败立即停止，不绕过。

## 1. 放到哪里，怎么运行

将 ZIP 解压到现有仓库根目录，使结构为：

```text
LeWM_official/
  eval.py
  jepa.py
  config/
  pcqr_tta_v1/
    core.py
    swm_bridge.py
    run_pusht.py
    tests/
    scripts/
```

不需要 pip install，不升级依赖，不切分支，不 reset 现有工作区。

在服务器提交 smoke：

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official
bash pcqr_tta_v1/scripts/submit_smoke.sh
```

默认使用现有 `lewm` conda 环境、4090node3 的一张 GPU、3 个新起点、B1000，以及 `frozen obs response` 三组。
脚本会在已知 PC-MH 目录中寻找唯一的 `*epoch_10_object.ckpt`；发现多个或没有文件时停止，不凭文件名猜哪一份权重正确。

指定 checkpoint 前缀时：

```bash
export POLICY='你的checkpoint相对STABLEWM_HOME的前缀'
bash pcqr_tta_v1/scripts/submit_smoke.sh
```

`POLICY` 不带 `_object.ckpt` 后缀。默认源目录是交接材料中的：
`$STABLEWM_HOME/pusht_pc_mh_h5_r01565_seed3072_ep10_ddp4`。
此处不编造实际 checkpoint 基名。

集群要求指定 partition/account 时，在提交命令后追加真实集群参数，例如 `--partition=<真实名称>`。脚本没有猜测 partition，也不申请 `--mem`。

在已经分配 GPU 的交互环境里直接运行：

```bash
source /mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh
conda activate lewm
export STABLEWM_HOME=/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data
cd /mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official

python -m unittest discover -s pcqr_tta_v1/tests -v
python -m pcqr_tta_v1.run_pusht \
  --policy-dir "$STABLEWM_HOME/pusht_pc_mh_h5_r01565_seed3072_ep10_ddp4" \
  --out "$STABLEWM_HOME/pcqr_tta_v1_runs/pilot_001" \
  --num-cases 3 --budget 1000 --modes frozen obs response
```

输出目录必须不存在。禁止为了重跑而覆盖旧目录。

## 2. 五种模式究竟比较什么

| mode | 更新目标 |
|---|---|
| `frozen` | 不更新；调用原生 cost/CEM，跳过反馈编码和适应 |
| `obs` | 仅已经观测到的多步转移监督 |
| `l2` | `obs` + 参数相对本轮更新前值的均方偏移 |
| `absolute` | `obs` + 当前查询候选预测相对更新前模型的绝对输出 MSE |
| `response` | `obs` + 当前查询候选之间的中心化响应保持 |

`obs` 只是匹配的简单适应 baseline，**不是 AdaJEPA 的完整复现**。
`l2` 是相对每次更新前参数的信赖域式惩罚，不是对最初 checkpoint 的永久 L2-SP。
`absolute`/`response` 的 teacher 是本轮更新前模型，不是 EMA，也不是原始离线 teacher。

全五组运行：

```bash
MODES='frozen obs l2 absolute response' \
NUM_CASES=10 \
bash pcqr_tta_v1/scripts/submit_smoke.sh
```

每组使用相同 case、同一初始 checkpoint、同一 CEM 配置、同一每-case seed、相同可训练参数、更新次数和梯度裁剪。
但各正则项的数值尺度不同，同一个 lambda 不代表公平的最优调参比较。默认权重只是启动值，不能据此直接宣称胜过调好的 baseline。

## 3. 目标函数与时间对齐

观测 buffer 只保存：

```text
真实当前 z_t --实际执行完的 5 个归一化 raw actions--> 真实 z_(t+5)
```

五个 raw action 按时间顺序展平为 10 维，保持训练/规划同一 action scaler。
没有将 CEM 未执行的 plan suffix 放进训练 buffer；没有使用 dataset 的未来 goal frame 当适应 target。

从 buffer 中的真实起点构造长度最多 H=5 的已完成窗口。未完成 horizon 只有 padding，损失 mask 为 false，不当作零目标。
观测损失为所有有效起点/horizon 的 latent MSE，默认 horizon 权重均为 1。

当前 CEM 第一轮已经生成的候选中，确定性等间隔取最多 Q=32 个；不额外消费 CEM/global RNG，不按 goal 或 elite 筛选。

对于更新前 teacher 输出 T 和当前 student 输出 S（形状 `[Q,H,D]`）：

```python
S_centered = S - S.mean(dim=0, keepdim=True)
T_centered = T - T.mean(dim=0, keepdim=True)
L_response = ((S_centered - T_centered) ** 2).mean()
L = L_observed + strength * L_response
```

中心化维是 **候选 Q**，不是 horizon H，也不是不同环境。
允许每个 horizon 的候选预测共同平移，不强制保持旧的 goal-cost 排名。

Teacher 输出在任何更新之前一次性计算并 detach；全部内层更新共用同一 query/teacher，不能每步刷新 teacher。

**第一步正则梯度为零。** 因为 student 此时与更新前 teacher 完全相同，仅做一步更新时 `response`/`absolute`/`l2` 与 `obs` 相同。默认采用 3 步，测试显式覆盖此性质。

## 4. CEM 与梯度的接入边界

没有重写 CEM。policy 每次请求动作前 arm 一个 hook：

1. 若原 policy 使用缓存动作，不触发 cost，也不适应。
2. 若原 policy 开始重规划，在第一次 `get_cost`、尚未给任何候选评分之前适应。
3. 然后用更新后的模型给**整个第一轮候选群**评分。
4. 同一次 CEM 的全部后续迭代都不再更新模型。

因此不会将更新前后的评分混在同一轮 elites 中，也不在 CEM 第 k 轮偷偷改变模型。

CEM 可能被 `torch.inference_mode()` 包裹。实现显式进入 `inference_mode(False)` + `enable_grad()`，并在退出 inference mode 后 clone 训练输入为普通 tensor。只写 `enable_grad()` 不够。

模型始终 `eval()`，但 action_encoder/predictor/pred_proj 的参数允许梯度。encoder/projector 参数冻结，所有 dropout/BN 保持 eval。
每个 episode 重置全部参数/buffers、Adam moments、buffer、候选/动作缓存。非有限更新整笔回滚；结构/API错误回滚后抛出，不当作正常失败继续统计。

## 5. 默认超参数

```text
H=5, action_block=5, packed_action_dim=10
planner initial observation history=1
predictor rollout history grows 1 -> 2 -> 3
train_parts = action_encoder, predictor, pred_proj
optimizer = Adam(lr=1e-5, weight_decay=0)
steps=3, strength=1, max_grad_norm=1
query_count=32, buffer_blocks=64, batch_starts=16
```

这不是 LoRA/adapter-only 方法；默认更新原 PC-MH 可训练的三个 predictor-side 模块。

`--steps`、`--lr`、`--strength` 可覆盖；其余在新的 `configs/default.json` 副本中固定并用 `--adapt-config` 指定。
不要靠扫描测试集成功率选择参数。

## 6. 新 pilot 与旧 benchmark 的区别

在线适应不能让 100 个评估 episode 共享一个会更新的模型。本版强制 `num_envs=1`，每个 case 重置模型；并行应启动不同进程/模型副本。

本 runner 使用原仓库 `get_dataset`、`img_transform`、action scaler、`AutoCostModel`、原 solver 配置和原 `World.evaluate_from_dataset`。但**新建一个串行配对 pilot**，不是旧的 500 个起点结果复算，也没有自动继承旧 cohort 的诊断排除列表。

因此本版的 `frozen` 对照必须重新测；不能直接把它与历史 90.2% 拼接比较。
代码也未声称已经复现先前另一个分支的 native-reset 自定义补丁。运行时会记录实际源码哈希并校验方法间的初始状态/图像/goal 是否相同。

默认 eval_budget=50 raw steps，receding_horizon=5 blocks 即 25 raw steps；通常只有首次规划和执行 25 步后的重规划，能够实际适应的机会很少。这是沿用原设置的 smoke，不是充分的在线适应效果评估。延长预算属于实验设定变化，应单独固定并报告。

支持复用本版保存的 `cases.json`：

```bash
python -m pcqr_tta_v1.run_pusht \
  --policy-dir "$STABLEWM_HOME/pusht_pc_mh_h5_r01565_seed3072_ep10_ddp4" \
  --cases "$STABLEWM_HOME/pcqr_tta_v1_runs/pilot_001/cases.json" \
  --out "$STABLEWM_HOME/pcqr_tta_v1_runs/pilot_002" \
  --modes frozen obs response --budget 1000
```

`--cases` 只接受明确的 `pcqr-cases-v1` schema，不猜测旧 `cohort.json` 字段。

```json
{
  "schema": "pcqr-cases-v1",
  "dataset": "pusht_expert_train",
  "goal_offset_steps": 25,
  "cases": [{"episode": 123, "start": 20}]
}
```

上面的 episode/start 仅说明格式，不是已确认存在的数据点。
正式评估应从确认的旧 cohort 显式转换；不要把示例当正式数据。

## 7. 输出和判读

- `MANIFEST.json`：checkpoint SHA256、实际本地 jepa/eval/config 哈希、依赖版本、超参数。
- `cases.json`：本次配对 case 列表。
- `*_caseXXXX.jsonl`：首帧指纹、每次适应的逐步 loss/梯度/耗时、每次 policy 调用耗时。
- `*_caseXXXX_result.json`、`results.csv`：原始环境 metrics、replans、cost_calls、真实执行反馈量与实际应用的优化步数。
- `COMPLETE.json` 或 `FAILED.json`：成功完成状态或完整异常；失败不伪装成已完成。

不猜测不同 SWM 版本 `success_rate` 的单位；原始 metrics 原样保存。

`adapt_ms` 是适应本体；`hook_ms` 还包含查询编码及首次 rollout 一致性检查。
`policy_wall_ms` 包含真实反馈编码、适应、原规划过程；`episode_wall_s` 还包含 reset、构造环境及评估开销。
日志写盘也可能影响全局 wall time；固定预算成功率不能代替端到端时间效率评估。

正确 smoke 至少应看到：首次 replan `warmup`；随后适应组出现 `updated`；`applied_steps=3`；同一 replan 只出现一次 adapt；逐步正则第 0 步为零、后续可能非零。
`frozen` 没有适应事件，优化步数为零。

## 8. 明确未包含

未包含自动修改摩擦/质量/控制增益的环境 perturbation，没有编造当前安装环境的物理参数接口；未包含真实机器人控制或多任务结果；未包含原离线 PC-MH 局部探针完整复现作为第六种模式；未复现 AdaJEPA/Sandwich-Residuals；未包含跨训练 seed 的正式统计；没有推送到 GitHub、提交 Slurm 作业或在服务器跑实验。

这版目的：把“已执行反馈 + 当前查询相对响应保持”变成可检查、可对照的最小闭环实现。

## 9. 接口核对来源

代码编写时读取了用户仓库 `ZhichengSong6/JEPA` 的：
- `jepa.py`，main commit `73b6b9caffb0151197eb466d96001b101c01b5a8`。
- 同 commit 的 `eval.py`、`config/eval/pusht.yaml`、`eval_pusht_final_errors.py`。
- `agent/pc-lewm10-eval-v1` 的 `bias_cem.py`（用于检查旧 CEM 行为，不作为替换 solver）。
- upstream SWM policy 源码（不假定与服务器安装版本完全相同，因此有 API/形状/rollout 检查）。

服务器可能是按目录 restore 的混合工作树。仅知道 HEAD 不够；MANIFEST 记录运行时文件哈希，不要求切到 main 或升级 SWM。
