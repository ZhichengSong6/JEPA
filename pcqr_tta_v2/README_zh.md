# PCQR-TTA v2: 高频重规划 + 严格首次适应前配对审计

**这是新协议，不是历史 PushT B1000 / B3000 的复现。** 基于已测试的 `pcqr_tta_v1` core/bridge；不改 checkpoint、encoder、predictor 架构、latent goal cost 或 CEM 优化器。每个 episode 都从同一个 checkpoint 重置，反馈只使用已经执行完毕的观测。需要仓库中同时存在 `pcqr_tta_v1/` 和 `pcqr_tta_v2/`。

## 新增功能

1. `--receding-horizon 1`：每执行 1 个 block = 5 个原始环境步就重新规划；H=5, action_block=5 不变。在 50 步的 episode 中最多 10 次规划、9 次带反馈的适应。**每次** CEM 调用仍使用 B1000（100 samples×10 iterations）；单个 episode 的总 CEM 计算量约为旧 RH5 协议的 5 倍，无法把两者成功率直接比较为等预算。
2. 同一 checkpoint 下 `frozen / obs / l2 / absolute / response` 五组对照；相同 case list、seed、CEM、训练步数和时间记录。
3. **首次更新之前**，逐一校验首轮及首次反馈规划的 raw action prefix、当前 replan pixels、CEM 第一批候选及选中参数的指纹（SHA256）。不一致时默认 `--audit-policy strict` 让实验失败并保留 `FAILED.json`；`record` 模式仅供排障，不能据此声称配对。
4. 输出每个 case 的 `first_plan_audit`、`first_preupdate_audit`、`preupdate_mismatch`，以及完整 JSONL 日志。没有记录未执行候选的真实结果，不需要物理真值。

## 先复用 3 个老起点运行 v2 烟雾测试

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official
export STABLEWM_HOME=/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data
export CASES="$STABLEWM_HOME/pcqr_tta_v1_runs/smoke_18840_20261031_205123/cases.json"
NODE=4090node2 NUM_CASES=3 RH=1 MODES='frozen obs l2 absolute response' \
    bash pcqr_tta_v2/scripts/submit_smoke.sh
```

检查输出：

```bash
python -m pcqr_tta_v2.report --run /完整/结果目录
cat /完整/结果目录/COMPLETE.json
```

预期 `case_results=15`、`first_preupdate_verified=true`、每 case 约 10 个 replans；适应版本每 case 最多 27 次（9×3）的优化步数。**如果严格审计失败，先查 JSONL 中 `type=preupdate_audit`，不要绕过检查，也不要直接扩容。**

## 然后开始新的 24-case 五组 pilot

为确保不复用仅 3 个起点的旧 `cases.json`：

```bash
unset CASES
NODE=4090node2 NUM_CASES=24 RH=1 MODES='frozen obs l2 absolute response' \
  WALLTIME=08:00:00 bash pcqr_tta_v2/scripts/submit_smoke.sh
```

脚本会生成新的 `cases.json`，并把所有实验记录在 `$STABLEWM_HOME/pcqr_tta_v2_runs/` 下。想完全复现实验，之后用新生成的 `cases.json` 作为 CASES 运行，并设置 NUM_CASES=24；不能把三例 smoke 扩成 24 例。GPU 资源不足时任务会排队，脚本不会自行重试。

## 解释边界

- 训练 buffer 的 MSE 下降不是独立模型泛化，更不是规划收益。
- 首次适应前严格匹配不表示后续轨迹必须相同；在线策略导致后续状态分岔本就是被研究现象。
- `regularizer_after > 0` 只说明响应偏差非零，不证明响应约束带来性能收益。
- 本版未加入摩擦/质量等物理扰动，不声称 sim-to-real/generalization。24 例仍属于方法筛选；完整文章需要更多 episode、多独立训练初始化和任务扩展。
- 由于 RH1 相比 RH5 增加了规划频次，应分别报告固定“每次规划预算”和“episode 总规划/更新耗时”。
