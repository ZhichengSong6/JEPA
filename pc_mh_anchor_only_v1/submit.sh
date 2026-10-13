#!/usr/bin/env bash
# bash pc_mh_anchor_only_v1/submit.sh smoke|formal [tag]
set -euo pipefail
MODE="${1:-}"; TAG="${2:-v1}"; NODE="${NODE:-4090node2}"
case "$MODE" in smoke|formal) ;; *) echo 'Usage: submit.sh smoke|formal [tag]' >&2; exit 2;; esac
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid tag' >&2; exit 2; }
# Apply the long-job policy to every mode, including smoke. One node only.
[[ "$NODE" =~ ^(4090node[23]|3090node[123])$ ]] || { echo 'Allowed: 4090node2-3 / 3090node1-3 only' >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/.." && pwd)"
CONDA_SH='/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh'
source "$CONDA_SH"; conda activate lewm
cd "$REPO"
export STABLEWM_HOME="${STABLEWM_HOME:-/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data}"
BASE_CONFIG="${BASE_CONFIG:-$STABLEWM_HOME/pusht_pc_mh_h5_r01565_seed3072_ep10_ddp4/config.yaml}"
OUT="$STABLEWM_HOME/pcmh_anchor_only_$TAG"
SMOKE_OUT="$STABLEWM_HOME/pcmh_anchor_only_${TAG}_smoke"
LIMIT=24:00:00
EXTRA=()
if [[ "$MODE" == smoke ]]; then OUT="$SMOKE_OUT"; LIMIT=00:30:00
else EXTRA=(--smoke-run "$SMOKE_OUT"); fi
python "$HERE/train_anchor.py" prepare --base-config "$BASE_CONFIG" \
  --output "$OUT" --mode "$MODE" --node "$NODE" "${EXTRA[@]}"
read -r GPUS CPUS < <(python - "$OUT/protocol.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]));print(m['world_size'],m['cpus_per_task'])
PY
)
LOGS="$STABLEWM_HOME/pcmh_anchor_logs"; mkdir -p "$LOGS"
JOB="$OUT/job.sbatch"
{
 echo '#!/usr/bin/env bash'
 echo '#SBATCH --partition=GPU'
 echo '#SBATCH --nodes=1'
 echo "#SBATCH --ntasks-per-node=$GPUS"
 echo "#SBATCH --gres=gpu:$GPUS"
 echo "#SBATCH --cpus-per-task=$CPUS"
 echo "#SBATCH --nodelist=$NODE"
 echo '#SBATCH --exclude=4090node1'
 echo "#SBATCH --time=$LIMIT"
 echo "#SBATCH --job-name=pca_$MODE"
 echo "#SBATCH --output=$LOGS/${MODE}_${TAG}_%j.out"
 echo "#SBATCH --error=$LOGS/${MODE}_${TAG}_%j.err"
 echo 'set -euo pipefail'
 printf 'source %q\n' "$CONDA_SH"; echo 'conda activate lewm'
 printf 'cd %q\n' "$REPO"
 printf 'export STABLEWM_HOME=%q\n' "$STABLEWM_HOME"
 echo 'export TMPDIR="$(mktemp -d /tmp/pcao.XXXXXX)"'
 echo 'chmod 700 "$TMPDIR"'
 echo 'trap '\''rmdir "$TMPDIR" 2>/dev/null || true'\'' EXIT'
 echo 'export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 NCCL_DEBUG=WARN'
 printf 'srun --kill-on-bad-exit=1 python -u %q run --output %q\n' "$HERE/train_anchor.py" "$OUT"
} > "$JOB"
bash -n "$JOB"
JID="$(sbatch --parsable "$JOB")"
printf 'Submitted PC-anchor-only %s: %s\nNode: %s\nOutput: %s\n' "$MODE" "$JID" "$NODE" "$OUT"
printf 'stdout: tail -n 60 %q\nstderr: tail -n 60 %q\n' \
 "$LOGS/${MODE}_${TAG}_${JID%%;*}.out" "$LOGS/${MODE}_${TAG}_${JID%%;*}.err"
echo 'No residual preparation, automatic next stage or planning evaluation was submitted.'
