# Experimental multi-anchor residual-difference PC-MH (v1)

This is a new, **unvalidated research candidate**, not an established improvement.
Only files in this directory are added. Existing training scripts, checkpoints,
PC-MH/Stepwise/failure-analysis code and the deployed CEM are unchanged.

## Exact objective

For an original training window, frozen teacher T and frozen observation encoder:

```
r_i,h = z_i+h - T_h(z_i, U_i)
Y_old(U) = z_i+h + T_h(z_i,U) - T_h(z_i,U_i)
Y_new(U) = Y_old(U) + r_hat_h(z_i,U) - r_hat_h(z_i,U_i)
L = L_original_PC_MH + 0.5 * masked_MSE(S_h(z_i,U), stop_gradient(Y_new(U)))
```

Teacher and student rollouts start at the current frame with context 1->2->3.
The factual anchor remains exact at U=U_i. A constant estimated residual
reduces to the original target. This is NOT Stepwise recursion, inverse-action
imitation, reward/value learning, or ground-truth counterfactual labeling.

`r_hat` is exact K=8 input-neighbor interpolation (one nearest window from each
of 8 distinct episodes). Query's reference episode is excluded. Query features
are standardized start latent / sqrt(D) and standardized **causal** action
prefix / sqrt(h*A). Statistics are fit ONLY on the bank. Weights are
softmax(-squared_distance / kth_squared_distance), floor 1e-8. There is no
outcome/goal-conditioned neighbor selection and no ANN approximation.

## Frozen preparation protocol

* Recreate the original saved PC-MH config and original `spt.data.random_split`.
  Only indices in its TRAIN subset are eligible for new supervision.
* Episode-partition that training subset 60/20/20 (seed 20261013).
* Sample 100,000 windows TOTAL: 60,000 bank, 20,000 gate calibration, 20,000
  independent gate audit. Nonfinite actions are excluded and counted, not
  converted into labels. More than 5% exclusion aborts.
* Encode the original 8-frame dataset window in FP32/eval mode; retain frame 2
  as start, actions [2:7], and actual futures [3:8]. Do not load IDM caches.
* Normalization of dataset actions is the ORIGINAL baseline normalization.
  Retrieval feature scales use the bank only. Baseline preprocessing itself
  may use full-column statistics, as the original code does: this release does
  not pretend to repair historical train/validation overlap.
* At each horizon choose a different-episode reference i for held-out query j
  using input distance only. Neither i nor j is in the memory. Compare
  `T(x_j)+r_i` with `T(x_j)+r_i+r_hat(x_j)-r_hat(x_i)` against the actual recorded
  next latent. This tests factual residual transfer, NOT controlled physical
  counterfactual equivalence between different states.
* Calibrate thresholds on a FIXED grid q={.5,.75,.9} over max neighbor start
  distance, max action-prefix distance, weighted residual dispersion, pair
  start distance, pair action distance. Both reference and query must pass.
  Choose highest calibration coverage with >=10% MSE reduction, >=20% coverage
  and >=64 accepted pairs; no passing choice disables that horizon.
* Independent audit must have >=10% accepted aggregate reduction, >=10%
  TERMINAL reduction, >=20% TERMINAL coverage, and >=64 terminal pairs.
  These are descriptive development cutoffs, NOT significance tests.
* On audit rejection: save `REJECTED.json` and STOP before expensive query
  generation. Do not weaken thresholds or loop over seeds to force acceptance.
* Generate 4096 query anchors from BANK-partition training windows (not held-out
  audit/calibration). Installed official stable-worldmodel 0.0.6 CEM is called
  in model-only mode: N100/I10/K10/H5. Four distinct actions per anchor: returned
  mean, elite, boundary, random. No factual action is injected into the search.
* The latent cost adapter is checked against real JEPA pixel `get_cost` on an
  actual preprocessed window BEFORE full extraction, and again for the query
  pool. Unexpected installed solver shapes/return mean abort.
* Cache old target, corrected target and mask. Raw out-of-range actions are NOT
  clipped. Unsupported causal prefixes receive no extra query loss. At least
  20% of ALL generated queries must have supported terminal targets.
* No `World` is created and no candidate action is executed. This code never
  reads `pcmh_failure*`, diagnostic replay outcomes, task states or success.
* Write `READY.json` only after full audit and query coverage both pass.
  Input/output/source SHA256 manifests prevent silent reuse after changes.

The memory is frozen after calibration; it is NOT rebuilt with audit data.
Neighbors consistent with each other are not necessarily correct. Partial
observability, contact switches, latent metric limitations and lack of action
coverage can defeat this estimator. Unsupported actions remain a deployment
limitation, not newly labeled failures.

## Files

* `core.py`: retrieval, gate calibration, target difference, masked/DDP loss.
* `runtime.py`: pinned repository/runtime/config/source checks and data adapter.
* `queries.py`: installed CEM adapter + fixed query selection, no simulator.
* `pipeline.py`: cache -> calibrate -> independent audit -> queries -> READY.
* `train.py`: delegates to original training entrypoint with extra query loss.
* `submit.sh`: single-GPU preparation, baseline-matched DDP training, no --mem.
* `test_multianchor.py`: numerical, leakage/causality, interfaces and gradient tests.

## Server commands

Run from the existing repository after fetching this directory. Do not change
branches/reset the working tree or edit any old protocol hashes.

```bash
source /mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh
conda activate lewm
export STABLEWM_HOME=/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data
cd /mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official
export BASE_CONFIG="$STABLEWM_HOME/pusht_pc_mh_h5_r01565_seed3072_ep10_ddp4/config.yaml"

python pc_mh_multianchor_v1/test_multianchor.py --with-repo
bash pc_mh_multianchor_v1/submit.sh prepare v1
```

Preparation runs on ONE GPU with no training optimizer. Logs:
`$STABLEWM_HOME/pcmh_ma_logs/prepare_v1_JOBID.out` and `.err`.
Data: `$STABLEWM_HOME/pcmh_ma_data_v1`.

```
NATIVE CEM / PIXEL COST PREFLIGHT PASS
CACHE ...
CALIBRATION horizon=...
AUDIT h=... ...
QUERY CACHE ...
PREPARATION READY: ...
```

A numerical/program error writes `BLOCKED.json`; a failed research gate writes
`REJECTED.json`, which is not a Python bug. Never interpret job disappearance
or exit 0 alone as method acceptance. Read the gate files:

```bash
DATA="$STABLEWM_HOME/pcmh_ma_data_v1"
cat "$DATA/audit_report.json"
test ! -f "$DATA/query_report.json" || cat "$DATA/query_report.json"
test ! -f "$DATA/READY.json" || cat "$DATA/READY.json"
test ! -f "$DATA/REJECTED.json" || cat "$DATA/REJECTED.json"
test ! -f "$DATA/BLOCKED.json" || cat "$DATA/BLOCKED.json"
```

Only with `READY.json`, run the TWO-STEP baseline-world-size training smoke:

```bash
bash pc_mh_multianchor_v1/submit.sh train-smoke v1
# After MA TRAINING SMOKE PASS and COMPLETED 0:0:
bash pc_mh_multianchor_v1/submit.sh train v1
```

The smoke output is `pcmh_ma_train-smoke_v1`; formal output is
`pcmh_ma_train_v1`. The formal model starts from the SAME LeWM10 parent as
original PC-MH, not from Stepwise/PC-MH output. Original loader batch, devices,
learning-rate schedule and epochs are kept. Additional query batch = 16 per
rank, weight .5, sampled with private per-rank RNG; baseline RNG is unchanged.
The extra masked loss uses the global valid count under DDP. Validation remains
the baseline validation (audit labels are never fed into student validation).
Loss CSV values are written directly from forward outputs, not ambiguous
callback metric aliases. Teacher/visual state and gradients are checked.

Inference checkpoints remain plain `jepa.JEPA`; retrieval memory is NOT a model
submodule. `training_complete.json` contains the policy to pass to the existing
`eval.py`. Do not quote a planning gain before actual paired evaluation.

The same-query comparison exists but is NOT submitted automatically:

```bash
# Only after the primary method warrants this attribution control:
bash pc_mh_multianchor_v1/submit.sh control v1
```

It uses the identical query pool/mask, query weight, batch sizes and initialization,
but old anchored targets. Compare this against corrected-target training to
separate residual information from extra query exposure/compute. Equal epochs
are NOT equal computation compared with the old PC-MH; preparation and additional
query cost must be reported.

## Optional small pipeline smoke

The real pipeline is also available at 1024 total windows and 8 query anchors:

```bash
bash pc_mh_multianchor_v1/submit.sh prepare-smoke mini1
# Once preparation completes (smoke gates are diagnostic only):
DATA_OUT="$STABLEWM_HOME/pcmh_ma_data_mini1_smoke" \
  bash pc_mh_multianchor_v1/submit.sh train-smoke mini1
```

A smoke bank can NEVER authorize formal training. A smoke runs query-generation
interfaces even if statistical gates reject, and is explicitly labeled as such.
Do not use it as efficacy evidence or change the formal protocol from its scores.

## Storage and failure behavior

All numerical artifacts, logs and checkpoints remain under STABLEWM_HOME.
Only short-lived IPC/temp files use a private `mktemp -d /tmp/pcma.XXXXXX` on the
allocated node; there is no recursive deletion trap. This avoids the previous
NFS temporary socket/cleanup issues. NPZ readers are always explicitly closed.
Outputs are fresh directories, no overwrite/auto-resume, no old-result deletion.
On failure preserve artifacts and diagnose before selecting a new tag. A prepared
partial directory is not silently called complete. No server jobs are submitted
by merely importing/running the unit tests.

## Validation status / limitations

36 CPU tests passed in the authoring container, plus a synthetic full 5-horizon
calibration/audit run. Native CEM interface test uses a fixture, not installed
stable-worldmodel. No real LeWM checkpoint, PushT HDF5, CUDA or Slurm/DDP run was
available in the authoring container. The real runtime checks and training smoke
must therefore run on the server. Tests are not evidence of method novelty,
calibration generality, planning improvement, or statistical significance.

No new pip installs or library upgrades are required. The version guard targets
the user's installed stable-worldmodel==0.0.6. Original source hashes match the
project base `851b60c06ee0aef6927d59f85763cb03a3aede34` (unchanged on failure branch).
