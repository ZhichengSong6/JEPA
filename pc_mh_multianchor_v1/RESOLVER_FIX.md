# Resolver startup repair (job 17795)

The failed `prepare v1` process read its config through `runtime.omega()` before
`train_mh_ald` first imported `stable_pretraining`. The old helper registered
`eval` first; the installed dependency then registered `eval` unconditionally.
OmegaConf raised `ValueError: resolver 'eval' is already registered`.
This happened before loading teacher weights, reading training windows, or
building a residual bank. It is not a rejected scientific acceptance gate.

## Repair

`runtime.omega()` now imports `stable_pretraining` first, then explicitly installs
the existing numeric-only `arithmetic` resolver with `replace=True`. Later normal
imports use the cached module. No global resolver clearing, module reload,
site-packages edits, dependency upgrade, or training/config changes are used.

`test_resolver_startup.py` exercises the old and new orders in fresh subprocesses
using real OmegaConf and a synthetic dependency with the observed import-time
registration. It also checks arithmetic restrictions, repeated calls, other
resolvers, and propagation of dependency import failures. `test_multianchor.py`
now runs these checks; with `--with-repo`, it additionally starts a fresh process
that reads the actual saved config BEFORE importing real training modules.
That real check loads no dataset or checkpoint, creates no Trainer, and does not
require GPU allocation.

## Server commands

After fetching the repaired directory:

```bash
export STABLEWM_HOME=/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data
export BASE_CONFIG="$STABLEWM_HOME/pusht_pc_mh_h5_r01565_seed3072_ep10_ddp4/config.yaml"
(cd pc_mh_multianchor_v1 && sha256sum -c SHA256SUMS.txt)
python pc_mh_multianchor_v1/test_multianchor.py --with-repo --base-config "$BASE_CONFIG"
```

Expect `RESOLVER STARTUP SYNTHETIC TESTS PASS: 6`,
`REAL CONFIG -> TRAINING IMPORT PASS`, and the original 36 numerical tests.
Only after these finish successfully, submit a NEW preparation output:

```bash
DATA_OUT="$STABLEWM_HOME/pcmh_ma_data_v2" \
  bash pc_mh_multianchor_v1/submit.sh prepare v2
```

Keep `pcmh_ma_data_v1` and job 17795 logs. Do not erase the failed run, change its
protocol hashes, or resubmit its generated sbatch against modified source.
`v2` is an output tag, not a method change. The failed run did not reach cache
extraction, so there is no completed residual-bank stage to reuse.
Continue to require `READY.json` before training; `REJECTED.json` and
`BLOCKED.json` retain their original meanings. Do not start `train` or `control`
while preparation is unfinished.

## Validation scope

Locally rerun: 36 numerical/interface-fixture tests and 6 fresh-process resolver
regressions passed with exit status 0; Python compilation and Bash syntax pass.
The new real-repository check is provided for the user's installed environment.
It has NOT been run here: this container lacks stable_pretraining and
stable_worldmodel, and has no server data/checkpoints or GPUs. No LeWM training
or planning improvement is claimed.
