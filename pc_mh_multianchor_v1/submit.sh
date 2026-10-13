#!/usr/bin/env bash
# bash pc_mh_multianchor_v1/submit.sh prepare|prepare-smoke|train-smoke|train|control [tag]
set -euo pipefail
MODE="${1:-}"; TAG="${2:-v1}"
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid tag' >&2; exit 2; }
case "$MODE" in prepare|prepare-smoke|train-smoke|train|control) ;; *) echo "Usage: $0 prepare|prepare-smoke|train-smoke|train|control [tag]" >&2; exit 2;; esac
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/.." && pwd)"
export STABLEWM_HOME="${STABLEWM_HOME:-/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data}"
CONDA_SH=/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh
source "$CONDA_SH"; conda activate lewm
cd "$REPO"
BASE_CONFIG="${BASE_CONFIG:-$STABLEWM_HOME/pusht_pc_mh_h5_r01565_seed3072_ep10_ddp4/config.yaml}"
DATA_OUT="${DATA_OUT:-$STABLEWM_HOME/pcmh_ma_data_$TAG}"
LOGS="$STABLEWM_HOME/pcmh_ma_logs"; mkdir -p "$LOGS"
GPUS=1; CPUS=6; LIMIT=24:00:00
case "$MODE" in
 prepare|prepare-smoke)
  OUT="$DATA_OUT"; [[ "$MODE" == prepare-smoke ]] && OUT="${DATA_OUT}_smoke"
  CMD=(python -u "$HERE/pipeline.py" --base-config "$BASE_CONFIG" --output "$OUT")
  [[ "$MODE" == prepare-smoke ]] && CMD+=(--smoke)
  ;;
 *)
  [[ -f "$DATA_OUT/queries_complete.json" ]] || { echo "Missing prepared queries: $DATA_OUT" >&2; exit 3; }
  if [[ "$MODE" != train-smoke && ! -f "$DATA_OUT/READY.json" ]]; then
    echo "No READY.json: do not bypass acceptance gates. Inspect audit_report.json / query_report.json." >&2; exit 3
  fi
  read -r GPUS WORKERS < <(python - "$HERE" "$DATA_OUT/baseline_config.yaml" <<'PY'
import sys
sys.path.insert(0,sys.argv[1])
import runtime
c=runtime.config(sys.argv[2])
print(c.trainer.devices,c.loader.num_workers)
PY
)
  CPUS=$((WORKERS+2)); VARIANT=corrected
  [[ "$MODE" == control ]] && VARIANT=same_queries
  OUT="$STABLEWM_HOME/pcmh_ma_${MODE}_${TAG}"
  CMD=(python -u "$HERE/train.py" --prepared "$DATA_OUT" --output "$OUT" --variant "$VARIANT")
  [[ "$MODE" == train-smoke ]] && { CMD+=(--smoke); LIMIT=01:00:00; }
  ;;
esac
[[ ! -e "$OUT" ]] || { echo "Preserve existing output $OUT; no overwrite/auto-resume" >&2; exit 4; }
JOB="$(mktemp "$LOGS/${MODE}_${TAG}_XXXXXX.sbatch")"
{
 echo '#!/usr/bin/env bash'
 echo '#SBATCH --partition=GPU'
 echo '#SBATCH --nodes=1'
 echo "#SBATCH --ntasks-per-node=$GPUS"
 echo "#SBATCH --gres=gpu:$GPUS"
 echo "#SBATCH --cpus-per-task=$CPUS"
 echo "#SBATCH --time=$LIMIT"
 echo '#SBATCH --exclude=3090node1'
 echo "#SBATCH --job-name=pcma_${MODE}"
 echo "#SBATCH --output=$LOGS/${MODE}_${TAG}_%j.out"
 echo "#SBATCH --error=$LOGS/${MODE}_${TAG}_%j.err"
 if [[ -n "${NODE:-}" ]]; then
  [[ "$NODE" =~ ^(3090node[23]|4090node[123])$ ]] || exit 5
  echo "#SBATCH --nodelist=$NODE"
 fi
 echo 'set -euo pipefail'
 printf 'source %q\n' "$CONDA_SH";echo 'conda activate lewm'
 printf 'cd %q\n' "$REPO"
 printf 'export STABLEWM_HOME=%q\n' "$STABLEWM_HOME"
 # Node-local private short directory: avoid NFS socket length/cleanup problems.
 # Numerical artifacts remain exclusively in STABLEWM_HOME.
 echo 'export TMPDIR="$(mktemp -d /tmp/pcma.XXXXXX)"'
 echo 'chmod 700 "$TMPDIR"'
 echo 'trap '\''rmdir "$TMPDIR" 2>/dev/null || true'\'' EXIT'
 echo 'export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 NCCL_DEBUG=WARN'
 printf 'srun --kill-on-bad-exit=1 ';printf '%q ' "${CMD[@]}";echo
} > "$JOB"
bash -n "$JOB"
JID="$(sbatch --parsable "$JOB")"
printf 'Submitted %s: %s\nOutput: %s\n' "$MODE" "$JID" "$OUT"
printf 'stdout: tail -n 80 %q\nstderr: tail -n 80 %q\n' "$LOGS/${MODE}_${TAG}_${JID%%;*}.out" "$LOGS/${MODE}_${TAG}_${JID%%;*}.err"
echo 'No evaluation, oracle, replay or automatic next-stage submission is performed.'
