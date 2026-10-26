# LeWM10-only supplement to the completed PC-MH / anchor-only comparison

This entrypoint evaluates only the original `lewm_epoch_10` parent. It does not
retrain any model, rerun PC-MH/anchor-only, sample a new cohort, modify CEM, or
perform candidate-population capture/counterfactual replay. Existing results
are read-only. The ordinary closed-loop environment execution is still needed
for the ten missing parent evaluations.

## Fixed protocol

- Source: `$STABLEWM_HOME/pcmh_anchor_compare_v1`, completed job 18299.
- Parent is resolved from the recorded anchor-only training protocol and its
  checksum; it must be the same `lewm_epoch_10` used for initialization.
- Reuse the exact existing `cohort.json` bytes, 500 starts across seeds 142-146.
  Never call the cohort sampler. Existing diagnostic-episode exclusions remain.
- Copy each saved resolved evaluation config. B1000 = N100/I10/K10 is primary;
  B3000 = N300/I10/K30 is secondary. Both run, 100 environments per seed.
- Use the existing `run_one` / native-reset evaluator and `check_pair`, checking
  parent results against both previously evaluated models. State, goal/reset
  recipe, variations and initial render checks are preserved.
- Read old 20 results; execute only ten parent runs (1,000 new closed-loop
  executions, with the same 500 starts shared across budgets).
- Only LeWM10 is evaluated. One existing anchor model is loaded on CPU solely
  to verify its saved state digest and shared encoder/projector frame.
- Default node: **4090node3**, one GPU/process. Allow only 4090node2/3 or
  3090node1/2/3. No `--mem`, no CUDA visibility override, short local TMPDIR.

## Server commands

Run from the LeWM repository in the existing `lewm` environment. Pull only this
new directory; do not replace the previous evaluation or training directories.

```bash
source /mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh
conda activate lewm
export STABLEWM_HOME=/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data
cd /mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official

(cd pc_mh_lewm10_eval_v1 && sha256sum -c SHA256SUMS.txt)
python pc_mh_lewm10_eval_v1/test_parent.py --with-repo

SOURCE_COMPARE="$STABLEWM_HOME/pcmh_anchor_compare_v1" \
NODE=4090node3 bash pc_mh_lewm10_eval_v1/submit.sh run v1
```

No new GPU smoke is required. The CPU tests are bookkeeping/adapter fixtures;
`--with-repo` additionally verifies the actual unchanged source files. Neither
claim replaces a successful real GPU evaluation.

The launcher verifies existing completion/artifact hashes, source versions,
training/parent provenance, cohort identities and configs before submission.
Do not edit hashes or old protocols to bypass a mismatch.

## Progress and results

```bash
export STABLEWM_HOME=/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data
OUT="$STABLEWM_HOME/pcmh_lewm10_compare_v1"
squeue -u "$USER"
tail -n 60 "$OUT"/slurm_*.out
tail -n 60 "$OUT"/slurm_*.err

# After the whole job has completed:
cat "$OUT/COMPLETE.json"
python pc_mh_lewm10_eval_v1/run_parent.py summarize --output "$OUT"
cat "$OUT/summary.csv"
cat "$OUT/pairwise.csv"
```

Success includes `LEWM10 SUPPLEMENT / THREE-MODEL COMPARISON PASS`, a final
Slurm `COMPLETED 0:0`, and `COMPLETE.json` with `new_runs:10`, `reused_runs:20`,
`source_unchanged:true`, and verified initial conditions. A partial run is not
reported as an overall score. Read `STATUS.json` / `errors/` on failure.

Outputs:

- `REPORT.md`: the three-model table and interpretation limits.
- `summary.csv`: per-budget means, evaluation-seed SDs and three differences.
- `pairwise.csv`: anchor-minus-parent, PC-MH-minus-parent, PC-MH-minus-anchor;
  discordant wins/losses and descriptive episode-cluster 95% intervals.
- `per_seed.csv`: all three success rates per seed/budget.
- `paired_cases.csv`: all three binary outcomes for each start/budget.
- `runs/B*_seed*_lewm10.json`: sealed parent outcomes, config and reset records.
- `cohort.json/csv`, `source_COMPLETE.json`, `source_protocol.json`: exact
  read-only-input snapshots in the NEW output. The copied cohort's internal
  binding deliberately remains to the original protocol.

PC-MH/anchor statistics are recomputed from the original saved case outcomes
and must match their existing paired effect and confidence interval. No source
file, model, checkpoint or existing result is written by this entrypoint.

## Interrupted jobs

Only after the prior job has ended, resume with:

```bash
export STABLEWM_HOME=/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data
NODE=4090node3 bash pc_mh_lewm10_eval_v1/submit.sh resume v1
```

Complete parent cells are checksum/provenance/config/reset-checked and reused.
An interrupted cell with no finalized JSON is recomputed; do not delete sealed
cells or manually change their values. Concurrent execution is locked. A
completed run refuses resubmission. A real mismatch needs investigation rather
than repetitive resume attempts. Source/config/dependency changes are blocked.

## Interpretation

These are fixed checkpoints, not repeated training initializations. Intervals
reuse the previous descriptive paired dataset-episode cluster bootstrap with
5,000 draws, are not adjusted for multiple contrasts and do not measure
training-seed uncertainty. Both budgets reuse the same 500 starts; they are not
1,000 independent cases. The cohort is not a held-out-pretraining claim.

LeWM10 versus a Stage-2 model includes the effect of ten additional epochs;
this alone cannot isolate a particular calibration mechanism from extra
training. Adding the parent baseline does not strengthen or change the already
measured PC-MH-minus-anchor effect. Timing includes existing native recording
and reset audits, not clean deployment latency.

## Local validation scope

See `LOCAL_TEST_RESULTS.txt`. CPU tests exercise provenance, rejection of
changed/missing inputs, the ten-parent-only call matrix, no re-sampling,
partial/full resume, mismatch rejection before sealing, model-mutation guards,
frame-check adapter execution, complete three-model merging and node rules.
Real stable_pretraining/stable_worldmodel dependencies, PushT dataset,
LeWM checkpoints, CUDA and Slurm execution were not available for local tests.
