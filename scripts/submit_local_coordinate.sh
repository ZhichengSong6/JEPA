#!/usr/bin/env bash
# Submit nonlinear local-coordinate training.
#
# Usage:
#   NODE=4090node3 MODE=smoke  bash scripts/submit_local_coordinate.sh
#   NODE=4090node3 MODE=formal bash scripts/submit_local_coordinate.sh
set -euo pipefail

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
DATA="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV="lewm"

NODE="${NODE:-4090node3}"
MODE="${MODE:-smoke}"

if [[ "$NODE" != "4090node2" && "$NODE" != "4090node3" ]]; then
  echo "NODE must be 4090node2 or 4090node3" >&2
  exit 2
fi
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "MODE must be smoke or formal" >&2
  exit 2
fi

cd "$REPO"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

python -m py_compile   local_coordinate_geometry.py   train_local_coordinate.py   jepa.py

python - <<'PY'
import torch
from local_coordinate_geometry import LocalCoordinateAdapter

torch.manual_seed(0)
m = LocalCoordinateAdapter(16, hidden_dim=32, num_blocks=4)
x = torch.randn(13, 16)
assert torch.allclose(m(x), x, atol=1e-7, rtol=1e-7)

with torch.no_grad():
    # Make the map non-identity while preserving exact invertibility.
    last = m.blocks[0].shift.net[-1]
    last.weight.normal_(0.0, 0.03)

y = m(x)
xr = m.inverse(y)
err = float((xr - x).abs().max())
assert err < 2e-5, err
print(f"local coordinate sanity checks passed; max_roundtrip_error={err:.3e}")
PY

LOG_DIR="$REPO/logs"
GEN_DIR="$REPO/slurm/generated_local_coordinate"
mkdir -p "$LOG_DIR" "$GEN_DIR"

if [[ "$MODE" == "smoke" ]]; then
  SUBDIR="pusht_local_coordinate_seed3072_smoke"
  STEPS=20
  VAL_INTERVAL=10
  VAL_BATCHES=5
  JOB_NAME="local_coord_smoke"
else
  SUBDIR="pusht_local_coordinate_seed3072"
  STEPS=2000
  VAL_INTERVAL=500
  VAL_BATCHES=20
  JOB_NAME="local_coord"
fi

OUT_DIR="$DATA/$SUBDIR"

if [[ "$MODE" == "smoke" ]]; then
  rm -rf "$OUT_DIR"
else
  if [[ -e "$OUT_DIR" ]]; then
    echo "Formal output already exists: $OUT_DIR" >&2
    echo "Delete it deliberately before rerunning formal." >&2
    exit 2
  fi
fi

SLURM_FILE="$GEN_DIR/${MODE}.slurm"
cat > "$SLURM_FILE" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=$LOG_DIR/local_coordinate_${MODE}_%j.out
#SBATCH --error=$LOG_DIR/local_coordinate_${MODE}_%j.err

set -euo pipefail
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$REPO"

export STABLEWM_HOME="$DATA"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "node=\$(hostname)"
echo "commit=\$(git rev-parse HEAD)"
echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}"
python --version
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

python -u train_local_coordinate.py \
  subdir="$SUBDIR" \
  output_model_name="lewm_local_coordinate" \
  trainer.max_steps=$STEPS \
  trainer.val_check_interval=$VAL_INTERVAL \
  trainer.limit_val_batches=$VAL_BATCHES
EOF

bash -n "$SLURM_FILE"
JID="$(sbatch --parsable "$SLURM_FILE")"

echo "Submitted local-coordinate $MODE."
echo "job     : $JID"
echo "node    : $NODE"
echo "output  : $OUT_DIR"
echo "stdout  : $LOG_DIR/local_coordinate_${MODE}_${JID}.out"
echo "stderr  : $LOG_DIR/local_coordinate_${MODE}_${JID}.err"
echo
echo "squeue -u zsong469 -o '%.18i %.18j %.2t %.10M %.24R'"
echo "tail -f $LOG_DIR/local_coordinate_${MODE}_${JID}.out"
