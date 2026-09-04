#!/usr/bin/env bash
# Planner-side fixed-population gate for the nonlinear Local Coordinate model.
set -euo pipefail

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
DATA="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV="lewm"
NODE="${NODE:-4090node3}"

RUN_DIR="$REPO/outputs/pusht_official_diagnostic/formal_20260903T024414Z_3108082"
LOCAL_POLICY="pusht_local_coordinate_seed3072/lewm_local_coordinate"
MODEL_CKPT="$DATA/pusht_local_coordinate_seed3072/lewm_local_coordinate_object.ckpt"
OUT_DIR="$REPO/outputs/local_coordinate_gate"
LOG_DIR="$REPO/logs"
GEN_DIR="$REPO/slurm/generated_local_coordinate_gate"

# Default project policy: formal/scientific jobs go to 4090node2/3 unless the
# user explicitly asks otherwise.
if [[ "$NODE" != "4090node2" && "$NODE" != "4090node3" ]]; then
  echo "NODE must be 4090node2 or 4090node3" >&2
  exit 2
fi

cd "$REPO"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

python -m py_compile   eval_local_coordinate_gate.py   local_coordinate_geometry.py   jepa.py

[[ -f "$RUN_DIR/run_identity.json" ]] || {
  echo "Missing authoritative official diagnostic: $RUN_DIR" >&2
  exit 2
}
[[ -f "$MODEL_CKPT" ]] || {
  echo "Missing Local Coordinate checkpoint: $MODEL_CKPT" >&2
  exit 2
}

# One authoritative evaluation for this trained checkpoint.
rm -rf "$OUT_DIR"
mkdir -p "$LOG_DIR" "$GEN_DIR"

SLURM_FILE="$GEN_DIR/gate.slurm"
cat > "$SLURM_FILE" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=local_coord_gate
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=$LOG_DIR/local_coordinate_gate_%j.out
#SBATCH --error=$LOG_DIR/local_coordinate_gate_%j.err

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
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

python -u eval_local_coordinate_gate.py \
  +diagnostic.run_dir="$RUN_DIR" \
  +diagnostic.output_dir="$OUT_DIR" \
  +diagnostic.coordinate_policy="$LOCAL_POLICY" \
  +diagnostic.cases="[7,27,53,77]" \
  +diagnostic.sources="[lewm,ald_tf]" \
  +diagnostic.solves="[0,1]" \
  +diagnostic.iterations="[0,3,9,19,29]" \
  +diagnostic.num_matched_controls=2 \
  +diagnostic.model_batch_size=32 \
  +diagnostic.coordinate_audit_tolerance=3e-4

echo
echo "===== SUMMARY ====="
cat "$OUT_DIR/summary.json"
echo
echo "===== AGGREGATE ====="
cat "$OUT_DIR/aggregate_summary.csv"
echo
echo "===== AUDIT MAXIMA ====="
python - <<'PY'
import csv
from pathlib import Path

p = Path("$OUT_DIR") / "audits.csv"
rows = list(csv.DictReader(p.open()))
for key in (
    "replay_state_max_abs",
    "inverse_encoder_max_abs",
    "inverse_predictor_max_abs",
    "inverse_goal_max_abs",
):
    vals = [float(r[key]) for r in rows]
    print(f"{key}: {max(vals):.9g}")
PY
EOF

bash -n "$SLURM_FILE"
JID="$(sbatch --parsable "$SLURM_FILE")"

echo "Submitted Local Coordinate planner-side gate."
echo "job    : $JID"
echo "node   : $NODE"
echo "output : $OUT_DIR"
echo "stdout : $LOG_DIR/local_coordinate_gate_${JID}.out"
echo "stderr : $LOG_DIR/local_coordinate_gate_${JID}.err"
echo
echo "squeue -u zsong469 -o '%.18i %.18j %.2t %.10M %.24R'"
echo "tail -f $LOG_DIR/local_coordinate_gate_${JID}.out"
