#!/usr/bin/env bash
# Short allocated-GPU diagnosis only. Never submits training or changes masks.
# NODE=4090node2 bash pc_mh_anchor_only_v1/submit_gpu_check.sh v1
set -euo pipefail
TAG="${1:-v1}"; NODE="${NODE:-4090node2}"
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid tag' >&2; exit 2; }
[[ "$NODE" =~ ^(4090node[23]|3090node[123])$ ]] || { echo 'Allowed: 4090node2-3 / 3090node1-3 only' >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/.." && pwd)"
CONDA_SH='/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh'
source "$CONDA_SH"; conda activate lewm
export STABLEWM_HOME="${STABLEWM_HOME:-/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data}"
BASE_CONFIG="${BASE_CONFIG:-$STABLEWM_HOME/pusht_pc_mh_h5_r01565_seed3072_ep10_ddp4/config.yaml}"
GPUS="$(python - "$HERE" "$BASE_CONFIG" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import anchor_runtime as rt
print(rt.config(sys.argv[2])['trainer']['devices'])
PY
)"
[[ "$GPUS" =~ ^[1-4]$ ]] || { echo "Invalid saved GPU count: $GPUS" >&2; exit 3; }
OUT="$STABLEWM_HOME/pcmh_anchor_gpucheck_$TAG"
[[ ! -e "$OUT" ]] || { echo "Preserve $OUT; choose a new tag" >&2; exit 4; }
mkdir "$OUT"
JOB="$OUT/job.sbatch"
{
 echo '#!/usr/bin/env bash'
 echo '#SBATCH --partition=GPU'
 echo '#SBATCH --nodes=1'
 echo "#SBATCH --ntasks-per-node=$GPUS"
 echo "#SBATCH --gres=gpu:$GPUS"
 echo '#SBATCH --cpus-per-task=2'
 echo "#SBATCH --nodelist=$NODE"
 echo '#SBATCH --exclude=4090node1'
 echo '#SBATCH --time=00:10:00'
 echo '#SBATCH --job-name=pca_gpucheck'
 echo "#SBATCH --output=$OUT/slurm_%j.out"
 echo "#SBATCH --error=$OUT/slurm_%j.err"
 echo 'set -euo pipefail'
 printf 'source %q\n' "$CONDA_SH"; echo 'conda activate lewm'
 printf 'cd %q\n' "$REPO"
 printf 'export STABLEWM_HOME=%q\n' "$STABLEWM_HOME"
 echo 'export TMPDIR="$(mktemp -d /tmp/pcagpu.XXXXXX)"'
 echo 'chmod 700 "$TMPDIR"'
 echo 'trap '\''rmdir "$TMPDIR" 2>/dev/null || true'\'' EXIT'
 echo 'export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 NCCL_DEBUG=WARN'
 # GPU count, visibility and binding are inherited exactly as in the old launcher.
 # No hardcoded CUDA_VISIBLE_DEVICES and no --gpus-per-task/--gpu-bind override.
 printf 'nvidia-smi --query-gpu=index,name,uuid,pci.bus_id --format=csv,noheader > %q 2>&1 || true\n' "$OUT/nvidia_smi.txt"
 echo 'set +e'
 printf 'srun --kill-on-bad-exit=0 python -u %q worker --output %q --expected %q --base-config %q\n' \
   "$HERE/check_gpu_startup.py" "$OUT" "$GPUS" "$BASE_CONFIG"
 echo 'STEP_EXIT=$?'
 echo 'set -e'
 echo 'echo "Probe srun exit=$STEP_EXIT; checking all rank reports"'
 printf 'python -u %q summarize --output %q --expected %q\n' "$HERE/check_gpu_startup.py" "$OUT" "$GPUS"
 echo 'test "$STEP_EXIT" -eq 0'
} > "$JOB"
bash -n "$JOB"
JID="$(sbatch --parsable "$JOB")"
printf 'Submitted GPU-startup check: %s\nNode: %s; GPUs/tasks: %s\nOutput: %s\n' "$JID" "$NODE" "$GPUS" "$OUT"
echo 'No training, data loading, callback patch, GPU reset, or CUDA visibility rewrite.'
