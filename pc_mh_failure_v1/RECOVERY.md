# Joint-probe domain failure: analysis-only recovery

Job 17695 reached the analysis/replay stage and then wrote:

```
ValueError: Need finite raw [5,10] center in [-1,1] and positive radius
```

The last shown replay was seed 45 / pcmh / case 36 / solve 0. The message combines shape, finiteness, action-domain and radius checks; the submitted log does NOT contain the actual action min/max. Do not infer the numerical excess or claim NaNs from this message alone. The recovery first audits every recorded returned action, using the original decode/normalization path, and records the exact values needed to distinguish these cases.

## Scope of the repair

- `run.py` and `analysis_core.py` are deliberately UNCHANGED, so the existing protocol's code hashes still verify.
- New `recover_analysis.py` checks the original protocol, completed survey and completed trace/repeatability artifact hashes before reading saved trace pickles.
- A separate recovery output links the original survey/capture read-only. The source `BLOCKED.json`, partial analysis, model checkpoints and protocol are untouched.
- No CEM survey or trace capture is rerun. Analysis-stage physical replay is recomputed; partially saved old analysis is NOT merged or blindly reused.
- A finite raw center outside the original bounded probe domain skips ONLY bounded joint-action probes at that center. The case remains in the population scoring, exact replay, elite-mean and reference-target analysis.
- The original returned action is NOT clipped, and the bounds/tolerance are NOT widened. This avoids pretending a changed reference action is the official returned plan.
- Invalid shapes, NaN/Inf, invalid radii, normalization round-trip failures, source-score mismatches, altered artifacts and exact replay failures still abort.
- Skipped joint diagnostics mean 'not evaluated', NOT zero interaction, zero error, success, or evidence against joint-action effects. Eligibility is not a guarantee of nonzero feasible symmetric radius; the original pair-level skip flags also apply.

## Run on the server

Do NOT delete `pcmh_failure_v1`, edit its protocol, or invoke the original `submit.sh v2` to rerun all collection.

```bash
source /mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh
conda activate lewm
export STABLEWM_HOME=/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data
cd /mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official

git fetch origin agent/pc-mh-failure-v1
git restore --source=origin/agent/pc-mh-failure-v1 -- \
  pc_mh_failure_v1/recover_analysis.py \
  pc_mh_failure_v1/test_recovery.py \
  pc_mh_failure_v1/submit_recovery.sh \
  pc_mh_failure_v1/RECOVERY.md

# Keep unit-test temporary files in project storage too.
mkdir -p "$STABLEWM_HOME/recovery_test_tmp"
TMPDIR="$STABLEWM_HOME/recovery_test_tmp" python pc_mh_failure_v1/test_recovery.py

export SOURCE_OUT="$STABLEWM_HOME/pcmh_failure_v1"
bash pc_mh_failure_v1/submit_recovery.sh v1
```

The launcher uses one GPU/process, no --mem, and short per-job TMPDIR. It preserves all output directories. Choose a new recovery tag if a previous recovery attempt has already created its output; investigate the error rather than changing gate tolerances.

Default new results: `$STABLEWM_HOME/pcmh_failure_recovery_v1`.
Logs: `$STABLEWM_HOME/pcmh_failure_recovery_logs/recover_v1_JOBID.out` / `.err`.

```bash
OUT="$STABLEWM_HOME/pcmh_failure_recovery_v1"
cat "$OUT/joint_probe_coverage.json"
cat "$OUT/COMPLETE.json"
cat "$OUT/REPORT.md"
```

`joint_center_audit.csv` supplies the actual failing center's shape, raw minimum/maximum, nonfinite count, bound excess and status. `analysis/populations.csv` retains all analyzed source populations. `analysis/joint_response.csv` includes explicit skipped rows for out-of-domain centers. Original data files linked into the new folder must remain available in the source folder.

A `RECOVERY ANALYSIS PASS` completion with skipped centers is a completed population analysis with partial bounded-joint coverage, not a claim that every joint test ran. Interpret joint statistics only on actually evaluated, causally affected rows and report coverage; do not extrapolate to skipped failures.

## Validation

15 local synthetic/mocked-dependency regression tests passed. They cover action-domain diagnosis, exact preservation of raw/normalized reference actions, unchanged eligible-call delegation, skipped outputs without fabricated zero metrics, NaN/Inf/radius rejection, scoped patch restoration, strict source error/gate checks, output path protection and reading synthetic saved traces without CEM calls. Python and Bash syntax checked. Actual server captures, installed PushT replay and GPU execution have NOT been run in this assistant environment. No planning improvement is claimed.
