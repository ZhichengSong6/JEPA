#!/usr/bin/env bash
# One GPU, one process. Run from the repository; no training or --mem request.
# Usage: bash pc_mh_failure_v1/submit.sh [tag]
# Optional: SEEDS="42" MAX_CASES=4 (pilot cohort, not a 5-seed benchmark).
set -euo pipefail
TAG="${1:-v1}"
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid output tag' >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
source "$CONDA_SH"
conda activate lewm
export STABLEWM_HOME="${STABLEWM_HOME:-/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data}"
BASE_CONFIG="${BASE_CONFIG:-$STABLEWM_HOME/pusht_pc_mh_h5_r01565_seed3072_ep10_ddp4/config.yaml}"
STEPWISE_RUN="${STEPWISE_RUN:-$STABLEWM_HOME/pusht_pc_mh_h5_r01565_seed3072_ep10_ddp4_stepwise_v2}"
OUT="$STABLEWM_HOME/pcmh_failure_$TAG"
read -r -a SEED_ARRAY <<< "${SEEDS:-42 43 44 45 46}"
cd "$REPO"
python "$HERE/run.py" prepare --output "$OUT" \
  --baseline-config "$BASE_CONFIG" \
  --stepwise-complete "$STEPWISE_RUN/training_complete.json" \
  --seeds "${SEED_ARRAY[@]}" --max-cases "${MAX_CASES:-12}"
JOB="$OUT/job.sbatch"
{
  echo '#!/usr/bin/env bash'
  echo '#SBATCH --job-name=pcmh_failure'
  echo '#SBATCH --partition=GPU'
  echo '#SBATCH --nodes=1'
  echo '#SBATCH --ntasks=1'
  echo '#SBATCH --gres=gpu:1'
  echo '#SBATCH --cpus-per-task=6'
  echo '#SBATCH --exclude=3090node1'
  echo '#SBATCH --time=24:00:00'
  if [[ -n "${NODE:-}" ]]; then
    [[ "$NODE" =~ ^(3090node[23]|4090node[123])$ ]] || { echo 'Invalid NODE' >&2; exit 3; }
    echo "#SBATCH --nodelist=$NODE"
  fi
  echo "#SBATCH --output=$OUT/slurm_%j.out"
  echo "#SBATCH --error=$OUT/slurm_%j.err"
  echo 'set -euo pipefail'
  printf 'source %q\n' "$CONDA_SH"
  echo 'conda activate lewm'
  printf 'cd %q\n' "$REPO"
  printf 'export STABLEWM_HOME=%q\n' "$STABLEWM_HOME"
  echo 'export TMPDIR="$STABLEWM_HOME/t${SLURM_JOB_ID}"'
  echo 'mkdir -p "$TMPDIR"'
  echo 'export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2'
  printf 'srun --kill-on-bad-exit=1 python -u %q run --output %q --stage all\n' "$HERE/run.py" "$OUT"
} > "$JOB"
bash -n "$JOB"
if ! JID="$(sbatch --parsable "$JOB")"; then
  printf 'Submission failed, preparation retained. Retry only this prepared job:\n  sbatch %q\n' "$JOB" >&2
  exit 4
fi
printf 'Submitted failure analysis: %s\nOutput: %s\n' "$JID" "$OUT"
printf 'stdout: tail -n 60 %q\nstderr: tail -n 60 %q\n' \
  "$OUT/slurm_${JID%%;*}.out" "$OUT/slurm_${JID%%;*}.err"
echo 'This job collects a matched diagnostic cohort; it does not train any model.'
