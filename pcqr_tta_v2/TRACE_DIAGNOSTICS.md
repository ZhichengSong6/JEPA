# v2 failure 19520: Numeric pre-update trace diagnostic

The existing strict audit failed at case 0, mode l2, first feedback replan
(`current_replan_pixels_sha256`). Frozen/Obs matched at both replans.
L2's action-command prefix, CEM initial population, and selected predictor
parameters matched. A SHA256 mismatch by itself does **not** quantify the
pixel difference; current logs do not retain the pixel arrays.

This diagnostic adds read-only, lossless `numeric_traces/*.npz` snapshots
to the **first two CEM replans** per mode/case, BEFORE any model update.
Included fields (where available): original raw world `state`,
`pixels`, `proprio`, previous `action`; transformed CEM pixels;
original CEM candidate population; and all issued raw actions.
The existing strict SHA256 gate remains enabled and failures still write
`FAILED.json`. Do not mislabel these runs as successful method evaluation.

## Reproduce the suspicious case, not the entire cohort

With `conda activate lewm` and `STABLEWM_HOME` set:

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official

export OLD_CASES="$STABLEWM_HOME/pcqr_tta_v1_runs/smoke_18840_20261031_205123/cases.json"
export DIAG_CASES="$STABLEWM_HOME/pcqr_tta_v2_runs/diag_case0_$(date +%Y%m%d_%H%M%S)_$$.json"

python - <<'PY'
import json, os
from pathlib import Path
original = json.loads(Path(os.environ["OLD_CASES"]).read_text())
one = dict(original)
one["cases"] = original["cases"][:1]
with Path(os.environ["DIAG_CASES"]).open("x") as f:
    json.dump(one, f, indent=2)
PY

export CASES="$DIAG_CASES"
export OUT_DIR="$STABLEWM_HOME/pcqr_tta_v2_runs/diag_trace_$(date +%Y%m%d_%H%M%S)_$$"

NODE=4090node2 NUM_CASES=1 RH=1 BUDGET=1000 EVAL_BUDGET=25 \
  MODES='frozen obs l2' TRACE_FIRST_TWO=1 AUDIT_POLICY=strict \
  bash pcqr_tta_v2/scripts/submit_smoke.sh

echo "$OUT_DIR"
```

After Slurm finishes (including when it exits FAILED):

```bash
python -m pcqr_tta_v2.trace_report --run "$OUT_DIR" --case 0 --modes obs l2
cat "$OUT_DIR/FAILED.json" 2>/dev/null || cat "$OUT_DIR/COMPLETE.json"
```

The report prints exact equality, number of differing elements, MAE,
maximum absolute difference, and L2 norm for each available field.

- `env_state` differs: the physical trajectory itself is not bitwise
  reproducible; compare numeric magnitude before concluding cause.
- `env_state` agrees while `env_pixels` differs: inspect rendering or
  other visual parameters.
- `env_pixels` agrees while `solver_pixels` differs: investigate
  preprocessing/transform nondeterminism.
- All fields agree: prior mismatch was intermittent; confirm recurrence
  before treating exact-match as a guaranteed property.

**No tolerances are applied.** Do not set `AUDIT_POLICY=record` to
inflate a benchmark. Diagnostic traces are environment/control inputs
and are not used to train the world model.
