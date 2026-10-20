# Job 18149: diagnose CUDA enumeration before another training attempt

The supplied log shows the saved-config import and real SPT fit-dispatch checks
passed. The new failure is in Lightning Trainer construction, when the installed
`stable_pretraining` CPUOffloadCallback queries logical CUDA device 3:
`AssertionError: Invalid device id`. This is before Trainer.fit or the first
training batch in this attempt. The log does NOT include the process CUDA mask,
the runtime-visible device count, or GPU health information. It does not prove a
physical GPU is broken, nor prove that Slurm exposed all four requested GPUs.

This change adds diagnosis only; it does not claim to repair hardware or a
confirmed allocation bug. The training objective, launcher, source hashes,
callbacks, batch size, precision, checkpoints and old results are unchanged.
Do not replace the old error by silently using 3 GPUs, manually set a guessed
CUDA_VISIBLE_DEVICES, reset GPUs, or disable all dependency callbacks.

## Run once inside an allocation

Fetch the added files on `agent/pc-anchor-only-v1` and run:

```bash
source /mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh
conda activate lewm
export STABLEWM_HOME=/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data
export BASE_CONFIG="$STABLEWM_HOME/pusht_pc_mh_h5_r01565_seed3072_ep10_ddp4/config.yaml"
python pc_mh_anchor_only_v1/test_gpu_check.py
NODE=4090node2 bash pc_mh_anchor_only_v1/submit_gpu_check.sh v1
```

The job requests the saved number of GPUs/tasks (this baseline: four), two CPU
cores per task, no explicit --mem, and a 10-minute TIME LIMIT (not a runtime
estimate). It inherits GPU visibility/binding, as the existing launcher does.
It runs only on 4090node2/3 or 3090node1/2/3. All results go to the NEW directory
`$STABLEWM_HOME/pcmh_anchor_gpucheck_v1`. It does not enter training.

Each rank launches two separate fresh Python processes:

1. Load the actual baseline config and construct an actual Lightning Trainer
   with installed callback factories, but no model, dataset or Trainer.fit.
   Record the installed CPUOffload source, versions and full failure traceback.
2. Inspect CUDA count before initialization, the native runtime count when the
   corresponding PyTorch diagnostic API exists, count after initialization,
   visible device properties/UUIDs, and tiny arithmetic on the rank's device.

The diagnostic never rewrites CUDA_VISIBLE_DEVICES, GPU counts or callbacks.
The two phases cannot pre-initialize each other's CUDA state. Missing rank
records, count mismatch, failed callback initialization, UUID order mismatch
or nonzero child exits lead to a failed summary. No DDP collective test is run.
Worker steps finish recording failures rather than killing peers early; the
batch summary propagates failure as a nonzero final job status.

## Read the results

```bash
OUT="$STABLEWM_HOME/pcmh_anchor_gpucheck_v1"
cat "$OUT/GPU_CHECK.json"
cat "$OUT/nvidia_smi.txt"
tail -n 60 "$OUT"/slurm_*.out
tail -n 60 "$OUT"/slurm_*.err
```

`pre/native/post=4/3/3` would support an enumeration mismatch. Four actual GPUs
with only the Trainer probe failing would instead point to callback/import
compatibility. These are interpretation examples, not measurements from the
server. If this check passes, a fresh full smoke is still required; passing is
not a model-training result. Do not submit formal training until smoke passes.

## Local validation

13 CPU-only synthetic diagnostic tests passed, plus Python/Bash syntax checks.
They cover expected counts, count changes, unsupported ranks, rejecting fewer
GPUs, child failure/missing reports and rank mapping mismatches. They do NOT
test server GPU health, installed SPT factories, real Trainer construction,
LeWM, CUDA/DDP training or Slurm. The original 43 tests were not rerun because
none of their training files are modified by this diagnostic addition.
