#!/usr/bin/env bash
# bash pc_mh_lewm10_eval_v1/submit.sh run|resume [tag]
set -euo pipefail
MODE="${1:-run}"; TAG="${2:-v1}"; NODE="${NODE:-4090node3}"
[[ "$MODE" == run || "$MODE" == resume ]] || { echo 'Use run|resume [tag]' >&2; exit 2; }
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid tag' >&2; exit 2; }
[[ "$NODE" =~ ^(4090node[23]|3090node[123])$ ]] || { echo 'Allowed: 4090node2-3 / 3090node1-3 only' >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/.." && pwd)"
CONDA_SH=/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh
source "$CONDA_SH"; conda activate lewm
cd "$REPO"
export STABLEWM_HOME="${STABLEWM_HOME:-/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data}"
SOURCE_COMPARE="${SOURCE_COMPARE:-$STABLEWM_HOME/pcmh_anchor_compare_v1}"
OUT="$STABLEWM_HOME/pcmh_lewm10_compare_$TAG"
if [[ "$MODE" == run ]]; then
  python "$HERE/run_parent.py" prepare --source "$SOURCE_COMPARE" --output "$OUT" --node "$NODE"
else
  [[ ! -f "$OUT/COMPLETE.json" ]] || { echo 'Already complete; read REPORT.md' >&2; exit 3; }
  python "$HERE/run_parent.py" verify --output "$OUT"
  python - "$OUT/protocol.json" "$NODE" <<'PY'
import json, sys
with open(sys.argv[1]) as f: meta=json.load(f)
if meta['node'] != sys.argv[2]: raise SystemExit('Resume must use prepared node; preserve protocol')
PY
fi
JOB="$(mktemp "$OUT/parent_XXXXXX.sbatch")"
{
  echo '#!/usr/bin/env bash'
  echo '#SBATCH --partition=GPU'
  echo '#SBATCH --nodes=1'
  echo '#SBATCH --ntasks=1'
  echo '#SBATCH --gres=gpu:1'
  echo '#SBATCH --cpus-per-task=6'
  echo '#SBATCH --time=24:00:00'
  echo '#SBATCH --exclude=4090node1'
  echo "#SBATCH --nodelist=$NODE"
  echo '#SBATCH --job-name=pca_lewm10'
  echo "#SBATCH --output=$OUT/slurm_%j.out"
  echo "#SBATCH --error=$OUT/slurm_%j.err"
  echo 'set -euo pipefail'
  printf 'source %q\n' "$CONDA_SH"; echo 'conda activate lewm'
  printf 'cd %q\n' "$REPO"
  printf 'export STABLEWM_HOME=%q\n' "$STABLEWM_HOME"
  echo 'export TMPDIR="$(mktemp -d /tmp/pcl10.XXXXXX)"'
  echo 'chmod 700 "$TMPDIR"'
  echo 'trap '\''rmdir "$TMPDIR" 2>/dev/null || true'\'' EXIT'
  echo 'export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2'
  printf 'srun --kill-on-bad-exit=1 python -u %q run --output %q\n' "$HERE/run_parent.py" "$OUT"
} > "$JOB"
bash -n "$JOB"
JID="$(sbatch --parsable "$JOB")"
printf 'Submitted LeWM10-only evaluation: %s\nNode: %s; GPU/tasks: 1/1\nOutput: %s\n' "$JID" "$NODE" "$OUT"
printf 'stdout: tail -n 60 %q\nstderr: tail -n 60 %q\n' "$OUT/slurm_${JID%%;*}.out" "$OUT/slurm_${JID%%;*}.err"
echo 'Only 10 parent runs. Existing 20 model runs and cohort are read-only; no training or new smoke.'
