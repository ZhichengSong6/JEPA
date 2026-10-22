# PC-MH vs PC-anchor-only: paired CEM evaluation

Read-only comparison of the existing PC-MH checkpoint against the completed
`pcmh_anchor_only_v3` epoch-10 checkpoint. No new training, teacher queries,
residual bank, candidate-population capture, or counterfactual replay.

## Protocol (fixed before this evaluation)

- B1000 primary: CEM N=100, I=10, K=10.
- B3000 secondary: N=300, I=10, K=30. Both budgets always run.
- Seeds 142–146, 100 starting points each: 20 model/seed/budget runs, 2,000
  closed-loop executions; 500 paired cases per budget, 1,000 paired CSV rows.
- H=5, action block=5, receding horizon=5; goal offset=25, eval budget=50;
  observation history=1; unchanged CEM, transforms and latent cost.
- Uniform sampling from valid dataset rows, without repeated rows across seeds.
  Full episodes appearing in the old `selected_cases.csv` are excluded. No
  success values are read for selection. The manifest is written before either
  model is evaluated. The same manifest is reused for both models/budgets.
- This is a NEW paired native-reset cohort, NOT a reproduction of the historical
  93.6% result and NOT a claim that these episodes were unseen during training.
  Seeds were fixed after model training, before this evaluation.
- Reuse `eval_cem_mh_diagnostics.run_closed(..., set(), set(), False)`.
  `trace_enabled=False` leaves the actual solver and model.get_cost unwrapped.
  Its N=100 population assertions only occur in the inactive trace path.
- Existing NativeRecorder pins otherwise missing reset seeds. Pairing verifies
  native variation contexts, initial renders, full physics, goal/reset setter
  recipes and environment spec. Dataset row/start/goal identity is shared.
  Executed native actions are retained, but no alternative actions are replayed.
- Both models must have identical encoder/projector parameters and buffers.
  Checkpoint checksums come from the anchor training protocol/completion record;
  the comparison checkpoint is NOT guessed from a filename or historical score.
- All trained checkpoints and original scripts are read only. Runtime model
  state is hashed before and after inference. No Lightning Trainer is created.

## Server use

After restoring this isolated directory from GitHub:

```bash
source /mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh
conda activate lewm
export STABLEWM_HOME=/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data
cd /mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official
(cd pc_mh_anchor_eval_v1 && sha256sum -c SHA256SUMS.txt)
python pc_mh_anchor_eval_v1/test_compare.py --with-repo

ANCHOR_RUN="$STABLEWM_HOME/pcmh_anchor_only_v3" \
EXCLUDE_CASES="$STABLEWM_HOME/pcmh_failure_recovery_v1/selected_cases.csv" \
NODE=4090node3 bash pc_mh_anchor_eval_v1/submit.sh run v1
```

This submits ONE single-GPU, single-process job on 4090node3 by default. Do not
submit the training launcher. No extra GPU smoke or four-GPU check is needed.
All job modes reject 4090node1; only 4090node2/3 or 3090node1/2/3 are allowed.
There is no `--mem` request and no CUDA_VISIBLE_DEVICES rewrite. Temporary files
use a short private node-local /tmp directory; reports remain in STABLEWM_HOME.
The 24-hour resource limit is not an estimate of actual evaluation duration.

The excluded-cases file is required. If the recovery directory is unavailable,
set EXCLUDE_CASES to the equivalent original failure run's selected_cases.csv;
do not silently use an empty exclusion list or substitute all the case outcomes.

## Progress and completion

```bash
OUT="$STABLEWM_HOME/pcmh_anchor_compare_v1"
squeue -u "$USER"
tail -n 60 "$OUT"/slurm_*.out
tail -n 60 "$OUT"/slurm_*.err
cat "$OUT/STATUS.json"
```

Each cell prints EVAL / FINISHED; each pair prints PAIRED INITIAL STATE/GOAL/RESET
PASS. Full success prints `PC-MH / ANCHOR PAIRED EVALUATION PASS`. Require the job's
final `COMPLETED 0:0` plus COMPLETE.json, not just an empty squeue.

```bash
cat "$OUT/COMPLETE.json"
cat "$OUT/REPORT.md"
cat "$OUT/summary.csv"
cat "$OUT/per_seed.csv"
python pc_mh_anchor_eval_v1/run_eval.py summarize --output "$OUT"
```

`summary.csv`: both success means, sample SD across five eval seeds, paired wins,
losses, and PC-MH-minus-anchor difference in percentage points. Positive favors
PC-MH. Descriptive 95% CI uses 5,000 paired dataset-episode cluster bootstrap
samples, keeping both model outcomes together. It is NOT training-seed
uncertainty, a guarantee of significance, or evidence of equivalence when zero
is inside the interval. Both budgets share the same sampled cases.

`paired_cases.csv`: 1,000 paired rows; `cohort.csv`: 500 immutable start/goal
identities; `runs/*.json`: sealed cell results, config, reset context and executed
actions; `errors/`: exception histories. `model_audit.json` records runtime model
state checks. Timing includes native recording/audits, not clean planner latency.

## Interruption / resume

Do not delete completed results or resubmit while a previous job is queued or
running. After accounting confirms the previous job ended, explicitly run:

```bash
NODE=4090node3 bash pc_mh_anchor_eval_v1/submit.sh resume v1
```

Source, dependency, input/checkpoint and dataset identities must still match.
Only complete, checksum-valid cells with the same manifest are reused; an
incomplete cell is rerun. The pair checks run again. No aggregate mean is
published from a missing/duplicated 20-cell matrix. Existing COMPLETE.json
prevents accidentally rerunning a finished comparison.

## Validation scope

`LOCAL_TEST_RESULTS.txt` records CPU mathematical/bookkeeping/interface-fixture
checks. Tests cover 100/300-sample forwarding with tracing disabled, outcome
validation, full matrix summarization, selection exclusion/deduplication, goal
mismatch rejection, source/input seals, and Slurm node restrictions. The authoring
environment has no stable_pretraining/stable_worldmodel, real checkpoints,
PushT data or GPUs. Real-server execution is NOT claimed tested or successful.
`--with-repo` checks the unchanged helper source hashes on your server; it does
not itself run a simulator or load model weights.
