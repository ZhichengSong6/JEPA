# Long-job placement and interrupted-cache recovery

User policy: jobs exceeding 30 minutes must run on `4090node2`, `4090node3`,
`3090node1`, `3090node2`, or `3090node3`, never `4090node1`.
All modes in this launcher reserve more than 30 minutes, so all use this rule.
The default is one node, `4090node2`; set `NODE` to ONE of the five permitted
node names to choose another. A list/range or `4090node1` is rejected before
submission. Placement is passed to sbatch on the command line as well as in the
script; a runtime guard also rejects an unexpected host. No memory request or
numerical/training/gate setting was changed. Existing old launchers elsewhere
in the repository and already-generated sbatch files are NOT rewritten.

## Job 17796

The supplied log shows the resolver/import tests and CEM/pixel precheck passed.
The process ran on 4090node1 and reached the third cache group, then Slurm printed
`Killed`. That line alone does not establish whether the cause was cancellation,
a time limit, out-of-memory or another termination. Inspect the accounting:

```bash
sacct -j 17796 --format=JobID,JobName%20,State%22,ExitCode,Elapsed,Timelimit,NodeList
```

Do not delete `pcmh_ma_data_v2`, alter its protocol, or resubmit its old sbatch.
The original extractor saves a whole group only after finishing it. Progress
such as `CACHE 12288/20000` is not a disk checkpoint. Complete `bank.npz` and
`calibration.npz` can be reused if present and valid; an absent audit.npz means
all 20,000 audit windows must be re-extracted, not only the missing suffix.

## New recovery entrypoint

`resume_prepare.py` is specifically for an interrupted FORMAL cache stage with
no cache_complete marker. It verifies original source/config/checkpoint/environment
and dataset metadata using the unchanged runtime verifier. The original dataset
split, normalization, internal episode split and selected clip IDs are reproduced.
Only complete zero-invalid-action groups with the exact expected row IDs are
accepted for reuse. CRC, fields, dtypes, shapes, finiteness and the residual
identity are checked; up to 16 deterministic rows per reused group are re-encoded.

There was no final cache seal in the interrupted run. Recovery records its own
read-time hashes and provenance; it does NOT pretend those are original seals or
that the small re-encoding check proves every number. Use only trusted artifacts
from your own interrupted run. Existing malformed groups block rather than being
silently replaced. Missing groups are recomputed; `.partial` files are never used.

Source files stay untouched. The NEW sibling output contains independent saved
NPZ groups and a new original-format protocol plus cache_recovery.json. Original
pipeline.py, runtime.py, core.py, teacher, objectives and acceptance gates remain
unchanged. Audit/query creation continues through the original functions. There
is no automatic student training, threshold relaxation or simulator-label use.

## Server commands

After fetching the updated submit.sh, resume_prepare.py, test_node_resume.py,
this document and SHA256SUMS.txt from the experiment branch:

```bash
(cd pc_mh_multianchor_v1 && sha256sum -c SHA256SUMS.txt)
python pc_mh_multianchor_v1/test_node_resume.py

# Only after old job 17796 has stopped. No new GPU smoke is required.
SOURCE_OUT="$STABLEWM_HOME/pcmh_ma_data_v2" \
DATA_OUT="$STABLEWM_HOME/pcmh_ma_data_v3" \
NODE=4090node2 \
  bash pc_mh_multianchor_v1/submit.sh prepare-resume v3
```

Expected progress, only if both completed files pass checks:

```
REUSE CACHE bank: 60000 windows; ...
REUSE CACHE calibration: 20000 windows; ...
RECOMPUTE GROUP audit: 20000 selected windows; ...
CACHE RECOVERY COMPLETE: original source preserved
```

Logs: `pcmh_ma_logs/prepare-resume_v3_JOBID.out` and `.err`.
New output: `$STABLEWM_HOME/pcmh_ma_data_v3`.
Inspect cache_recovery_progress.json, audit_report.json, query_report.json (if
reached), READY.json / REJECTED.json / BLOCKED.json. A recovery_complete.json or
zero exit status does not replace the scientific READY gate. Do not train if
BLOCKED.json exists or READY.json is absent. Future training must use DATA_OUT
pointing to v3 and a permitted NODE as well.

## Validation

14 new CPU fixture tests passed locally, including all six launcher modes, all
five allowed nodes, forbidden node/list rejection, unchanged source arrays,
identity/corruption rejection, deterministic re-encoding checks, reuse of two
complete groups and recomputation of a third, and audit rejection stopping query
generation. The original 36 numerical/interface tests also passed locally.
Python compilation and bash syntax passed. No server cache, NFS, real LeWM
checkpoint, CUDA, Slurm allocation or full production recovery was run here.
