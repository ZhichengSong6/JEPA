# Smoke 17896: fit/train metric-cache mismatch

The uploaded log reports FAILED 1:0 for job 17896 after 3m43s. The first
backward reached the custom callback and raised `The new forward did not
execute`. A finite loss is present in the log. This is not a successful smoke
or a completed formal run; it is not evidence that the ablation is ineffective.

## Cause and minimal repair

`anchor_only_forward` saved detached metrics under the raw `stage` value.
`stable_pretraining.Module.training_step` dispatches its forward with `fit`,
while `CheckedExport.on_after_backward` and `on_train_batch_end` looked under
`train`. The finite forward record therefore existed under a different key.

The fix maps `fit` and `train` to `train` in the PRIVATE metric cache only.
The original log_dict prefix, loss tensors, causal weighting, gradients,
optimizer, scheduler, frozen modules, deployment model and node restrictions
are unchanged. Validation/test entries remain separate. The backward guard is
retained and now reports cached keys and the raw stage if records are missing.
No dependency upgrades, site-packages edits, old-result deletion, or gate bypass
are required. The TCPStore/Broken-pipe messages are consistent with workers
exiting after the Python exception; this patch does not change NCCL settings.

## Checks

The original 37 tests plus six stage regressions pass locally (43 total).
The two-update fixture now dispatches `fit`, not a manually assumed `train`.
Tests verify fit/train value and gradient equality, validation separation,
nonempty training CSVs and continued rejection of genuinely missing records.
An old-writer regression reproduces the failure with a finite fit-stage loss.
Fixture module overrides restore only the mocked module names rather than
removing newly lazy-imported PyTorch modules from sys.modules.

`test_anchor.py --with-repo --base-config ...` also launches a fresh process
with the user's INSTALLED `spt.Module.training_step`. It intercepts that actual
method immediately after forward dispatch, then runs CPU backward and the
production callback on the captured result. This is an interface test using a
tiny model, not a complete framework optimizer/Trainer/DDP test. It loads no
LeWM weights or dataset. A later GPU smoke still tests the full execution path.

The installed-framework check and the real server GPU/DDP smoke have NOT been
run in the authoring container (stable_pretraining/stable_worldmodel absent).

## Server recovery

Preserve `pcmh_anchor_only_v1_smoke` and job 17896 logs. Do not resubmit its old
job.sbatch after changing code: its protocol pins the original source hashes.
Fetch the repair on `agent/pc-anchor-only-v1` and restore only the changed
files, then run:

```bash
export STABLEWM_HOME=/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data
export BASE_CONFIG="$STABLEWM_HOME/pusht_pc_mh_h5_r01565_seed3072_ep10_ddp4/config.yaml"
(cd pc_mh_anchor_only_v1 && sha256sum -c SHA256SUMS.txt)
python pc_mh_anchor_only_v1/test_anchor.py --with-repo --base-config "$BASE_CONFIG"
NODE=4090node2 bash pc_mh_anchor_only_v1/submit.sh smoke v2
```

Tests should include `REAL SPT TRAINING_STEP DISPATCH -> BACKWARD AUDIT PASS`
and `ANCHOR CPU TESTS PASS: 43`. The new smoke directory is
`$STABLEWM_HOME/pcmh_anchor_only_v2_smoke`; logs are under
`$STABLEWM_HOME/pcmh_anchor_logs/smoke_v2_JOBID.out` and `.err`.

Only after smoke COMPLETED 0:0, `PC-ANCHOR-ONLY SMOKE PASS`, and the matching
all-rank completion record, explicitly submit:

```bash
NODE=4090node2 bash pc_mh_anchor_only_v1/submit.sh formal v2
```

`v2` is a run tag, not a new objective. The formal directory will be
`$STABLEWM_HOME/pcmh_anchor_only_v2`. No automatic submission is performed.
All job modes continue to allow only 4090node2/3 or 3090node1/2/3, default to
4090node2 and exclude 4090node1. No --mem is added.
