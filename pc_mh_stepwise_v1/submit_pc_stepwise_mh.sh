#!/usr/bin/env bash
# bash pc_mh_stepwise_v1/submit_pc_stepwise_mh.sh smoke|formal [tag]
# Optional: BASE_CONFIG=/absolute/path/to/old_run/config.yaml NODE=4090node2
# No --mem: this cluster advertises RealMemory=1 MB.
set -euo pipefail
MODE="${1:-}"
TAG="${2:-v1}"
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Usage: bash $0 {smoke|formal} [tag]" >&2
  exit 2
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
export STABLEWM_HOME="${STABLEWM_HOME:-/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data}"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
BASE_CONFIG="${BASE_CONFIG:-$STABLEWM_HOME/pusht_pc_mh_h5_r01565_seed3072_ep10_ddp4/config.yaml}"
source "$CONDA_SH"
conda activate lewm
cd "$REPO"

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "Missing baseline RUN config: $BASE_CONFIG" >&2
  echo "Set BASE_CONFIG to the actual saved pure PC-MH config. No defaults or batch size will be guessed." >&2
  exit 3
fi

EXECUTION="$(python "$HERE/train_pc_stepwise_mh.py" prepare \
  --base-config "$BASE_CONFIG" --mode "$MODE" --tag "$TAG")"
RUN_DIR="$(dirname "$EXECUTION")"
read -r NGPUS WORKERS EPOCHS LOCAL_BATCH GLOBAL_BATCH < <(
  python - "$EXECUTION" <<'PY'
import json, sys
m=json.load(open(sys.argv[1]))
print(m['expected_world_size'], m['num_workers_per_rank'], m['max_epochs'],
      m['local_batch'], m['global_batch'])
PY
)
# Give each rank enough CPU threads for its unchanged data-loader worker count.
CPUS=$((WORKERS + 2))
if (( CPUS < 2 )); then CPUS=2; fi
JOB="$RUN_DIR/job.sbatch"
{
  echo '#!/usr/bin/env bash'
  echo "#SBATCH --job-name=pcsw_${MODE}"
  echo '#SBATCH --partition=GPU'
  echo '#SBATCH --nodes=1'
  echo "#SBATCH --ntasks-per-node=$NGPUS"
  echo "#SBATCH --gres=gpu:$NGPUS"
  echo "#SBATCH --cpus-per-task=$CPUS"
  echo '#SBATCH --exclude=3090node1'
  # This is a scheduler resource limit, not an estimate of runtime.
  if [[ "$MODE" == "smoke" ]]; then
    echo '#SBATCH --time=01:00:00'
  else
    echo '#SBATCH --time=24:00:00'
  fi
  if [[ -n "${NODE:-}" ]]; then
    if [[ ! "$NODE" =~ ^(3090node[23]|4090node[123])$ ]]; then
      echo "Unsupported NODE=$NODE" >&2; exit 4
    fi
    echo "#SBATCH --nodelist=$NODE"
  fi
  echo "#SBATCH --output=$RUN_DIR/slurm_%j.out"
  echo "#SBATCH --error=$RUN_DIR/slurm_%j.err"
  echo 'set -euo pipefail'
  printf 'source %q\n' "$CONDA_SH"
  echo 'conda activate lewm'
  printf 'cd %q\n' "$REPO"
  printf 'export STABLEWM_HOME=%q\n' "$STABLEWM_HOME"
  # Keep multiprocessing AF_UNIX socket paths short; preserve all artifacts in project storage.
  # This cluster's long run directory exceeds the 108-byte Unix socket path limit.
  echo 'export TMPDIR="$STABLEWM_HOME/t${SLURM_JOB_ID}"'
  echo 'mkdir -p "$TMPDIR"'
  echo 'echo "SLURM TMPDIR=$TMPDIR"'
  echo 'export PYTHONUNBUFFERED=1'
  echo 'export OMP_NUM_THREADS=1'
  echo 'export NCCL_DEBUG=WARN'
  echo 'export MUJOCO_GL=egl'
  printf 'srun --kill-on-bad-exit=1 python -u %q run --execution %q\n' \
    "$HERE/train_pc_stepwise_mh.py" "$EXECUTION"
} > "$JOB"
bash -n "$JOB"
if ! JID="$(sbatch --parsable "$JOB")"; then
  echo "Submission failed; preparation is preserved. After fixing scheduler arguments, retry:" >&2
  printf '  sbatch %q\n' "$JOB" >&2
  exit 5
fi
printf 'Submitted %s job: %s\n' "$MODE" "$JID"
printf 'Base config : %s\nRun directory: %s\n' "$BASE_CONFIG" "$RUN_DIR"
printf 'GPUs=%s local_batch=%s global_batch=%s epochs=%s\n' "$NGPUS" "$LOCAL_BATCH" "$GLOBAL_BATCH" "$EPOCHS"
printf 'Stdout: tail -n 60 %q\n' "$RUN_DIR/slurm_${JID%%;*}.out"
printf 'Stderr: tail -n 60 %q\n' "$RUN_DIR/slurm_${JID%%;*}.err"
squeue -j "${JID%%;*}" -o '%.18i %.12j %.2t %.12M %.24R' || true
