# PCQR-TTA v2 | PushT physical-reset audit (experimental protocol)

This branch adds an opt-in PushT reset variant. **It has NOT been validated
on the real server yet.** The intent is to isolate the hidden physics
initialization issue observed in case 0, not to force an outcome or hide it
behind a tolerance.

## What the code changes

The installed legacy SWM PushT `_set_state` sets visible state then calls
`space.step(dt)`. It does not explicitly clear the **block** linear or
angular velocity. A reset first places the goal, then the start state;
the dataset evaluation may call `_set_state` again.

- `PUSHT_RESET_MODE=legacy` (default): original environment and reset
  untouched, but before the first two plans records agent position/velocity,
  block position/angle/**linear velocity/angular velocity/force/torque**,
  space iterations and damping in the JSONL and (if enabled) numerical NPZ.
- `PUSHT_RESET_MODE=fresh_dynamics`: **new evaluation protocol**. During
  the synchronous dataset evaluation only, temporarily overrides the
  individual PushT instance's `_set_state` to place its observed 7D state
  directly, clear block velocity/angular velocity/force/torque, activate and
  reindex the two bodies, and NOT advance `space.step(dt)` during the goal
  or start placement. The next action executes the normal unmodified physics
  simulation. The original method is restored even on exceptions.
  Position/angle/agent velocity are validated after placement; the runner
  refuses to proceed if the override was not executed.

This is a *simulator initialization* change. It does not add physics truth,
new reward, control objective, teacher labels, predictor weights, planner
samples, or a new optimization loss. It is **not the historical PushT
benchmark protocol**; compare all five methods within this protocol only.

## 1. First perform two independent Frozen-vs-Frozen repeats

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official
source /mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh
conda activate lewm

export STABLEWM_HOME=/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data
export CASES="$STABLEWM_HOME/pcqr_tta_v2_runs/diag_case0_fixed_20261112_142148_4027200.json"
test -s "$CASES"

export OUT_DIR="$STABLEWM_HOME/pcqr_tta_v2_runs/reset_frozen_A_$(date +%Y%m%d_%H%M%S)_$$"
NODE=4090node3 NUM_CASES=1 RH=1 BUDGET=1000 EVAL_BUDGET=25 SEED=1729 \
  MODES=frozen TRACE_FIRST_TWO=1 AUDIT_POLICY=strict \
  PUSHT_RESET_MODE=fresh_dynamics \
  bash pcqr_tta_v2/scripts/submit_smoke.sh
echo "Run A: $OUT_DIR"

export OUT_DIR="$STABLEWM_HOME/pcqr_tta_v2_runs/reset_frozen_B_$(date +%Y%m%d_%H%M%S)_$$"
NODE=4090node3 NUM_CASES=1 RH=1 BUDGET=1000 EVAL_BUDGET=25 SEED=1729 \
  MODES=frozen TRACE_FIRST_TWO=1 AUDIT_POLICY=strict \
  PUSHT_RESET_MODE=fresh_dynamics \
  bash pcqr_tta_v2/scripts/submit_smoke.sh
echo "Run B: $OUT_DIR"
```

Both jobs are independent; wait for both Slurm results. Save the printed
paths as `RUN_A` and `RUN_B` in the calling shell. A command inside
a `( ... )` subshell does not leave exported variables in the parent.

```bash
export RUN_A=/path/printed/for/A
export RUN_B=/path/printed/for/B

python - <<'PY'
import os
from pathlib import Path
from pcqr_tta_v2.trace_audit import compare_pair

a, b = (Path(os.environ[k]) for k in ('RUN_A', 'RUN_B'))
for root in (a, b):
    print(root, (root / 'COMPLETE.json').read_text())
for n in (0, 1):
    x = a / 'numeric_traces' / f'frozen_case0000_replan{n}.npz'
    y = b / 'numeric_traces' / f'frozen_case0000_replan{n}.npz'
    stats = compare_pair(x, y)
    print('Replan:', n)
    for k in ('issued_action_prefix', 'env_state', 'env_pixels',
              'solver_pixels', 'physics_block_velocity',
              'physics_block_angular_velocity', 'physics_block_position',
              'physics_block_angle'):
        print(' ', k, stats.get(k))
PY
```

Also verify `reset_patch_calls >= 2` in each `*_result.json`,
and compare both pre-update `physics_hidden_sha256` fields.

- If the first two plans and physics match exactly, this is positive
  evidence (not proof) that the reset was responsible. Next run a 3-case
  five-mode smoke with `PUSHT_RESET_MODE=fresh_dynamics`.
- If not, inspect `physics_block_velocity` and `physics_block_angular_velocity`
  at replan 0 first. Do not disable strict or lower thresholds to force a pass.
- If the simulator or wrapper rejects the new variant, preserve `FAILED.json`
  and the original v2 results. Do not downgrade other dependencies silently.

## 2. After the two Frozen repeats pass

Run 3 cases × 5 methods (not the 24-case pilot):

```bash
export CASES="$STABLEWM_HOME/pcqr_tta_v1_runs/smoke_18840_20261031_205123/cases.json"
unset OUT_DIR
NODE=4090node3 NUM_CASES=3 RH=1 BUDGET=1000 EVAL_BUDGET=50 \
  MODES='frozen obs l2 absolute response' TRACE_FIRST_TWO=1 \
  AUDIT_POLICY=strict PUSHT_RESET_MODE=fresh_dynamics \
  bash pcqr_tta_v2/scripts/submit_smoke.sh
```

Audit success in this new protocol does not establish that response
regularization improves performance, nor that the environment is deterministic
for every initial condition. If five-mode smoke passes, study a larger
pre-registered cohort, independent repetitions and timing with planning budget
per-replan vs per-episode separated.

## Known limits

* Hidden velocities in the original dataset are unknown. Zeroing them is an
  explicit intervention, **not a reconstruction of expert demonstration
  dynamics**. This changes the physical initial condition.
* Exact SHA equality can still fail due to other hidden solver/contact state.
  No claim of full bitwise determinism is made before repeat validation.
* No modifications are made to `stable_worldmodel` on disk; patch exists on
  a single in-memory PushT instance during one episode evaluation only.
* Source/manifest records reset mode, patch invocation count, and hashes.
