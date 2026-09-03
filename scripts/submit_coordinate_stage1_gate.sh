#!/usr/bin/env bash
# Fixed-population life/death gate for Coordinate Stage-I.
set -euo pipefail

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
DATA="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV="lewm"
NODE="${NODE:-4090node3}"

RUN_DIR="$REPO/outputs/pusht_official_diagnostic/formal_20260903T024414Z_3108082"
COORD_POLICY="pusht_coordinate_stage1_seed3072/lewm_coordinate_stage1"
OUT_DIR="$REPO/outputs/coordinate_stage1_gate"
LOG_DIR="$REPO/logs"
GEN_DIR="$REPO/slurm/generated_coordinate_stage1_gate"

if [[ "$NODE" != "4090node2" && "$NODE" != "4090node3" ]]; then
  echo "NODE must be 4090node2 or 4090node3" >&2
  exit 2
fi

cd "$REPO"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

python -m py_compile eval_coordinate_stage1_gate.py

[[ -f "$RUN_DIR/run_identity.json" ]] || {
  echo "Missing official diagnostic: $RUN_DIR" >&2
  exit 2
}
[[ -f "$DATA/pusht_coordinate_stage1_seed3072/lewm_coordinate_stage1_object.ckpt" ]] || {
  echo "Missing Coordinate Stage-I checkpoint" >&2
  exit 2
}

# The gate is deterministic and should have exactly one authoritative output.
rm -rf "$OUT_DIR"
mkdir -p "$LOG_DIR" "$GEN_DIR"

SLURM_FILE="$GEN_DIR/gate.slurm"
cat > "$SLURM_FILE" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=coord_s1_gate
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=$LOG_DIR/coordinate_stage1_gate_%j.out
#SBATCH --error=$LOG_DIR/coordinate_stage1_gate_%j.err

set -euo pipefail
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$REPO"

export STABLEWM_HOME="$DATA"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

echo "node=$(hostname)"
echo "commit=$(git rev-parse HEAD)"
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u eval_coordinate_stage1_gate.py \
  +diagnostic.run_dir="$RUN_DIR" \
  +diagnostic.output_dir="$OUT_DIR" \
  +diagnostic.coordinate_policy="$COORD_POLICY" \
  +diagnostic.cases="[7,27,53,77]" \
  +diagnostic.sources="[lewm,ald_tf]" \
  +diagnostic.solves="[0,1]" \
  +diagnostic.iterations="[0,3,9,19,29]" \
  +diagnostic.num_matched_controls=2 \
  +diagnostic.model_batch_size=64

echo
echo "===== SUMMARY ====="
cat "$OUT_DIR/summary.json"
echo
echo "===== AGGREGATE ====="
cat "$OUT_DIR/aggregate_summary.csv"
EOF

bash -n "$SLURM_FILE"
JID="$(sbatch --parsable "$SLURM_FILE")"

echo "Submitted Coordinate Stage-I gate."
echo "job: $JID"
echo "node: $NODE"
echo "output: $OUT_DIR"
echo "stdout: $LOG_DIR/coordinate_stage1_gate_${JID}.out"
echo "stderr: $LOG_DIR/coordinate_stage1_gate_${JID}.err"
echo
echo "squeue -u zsong469 -o '%.18i %.14j %.2t %.10M %.24R'"
echo "tail -f $LOG_DIR/coordinate_stage1_gate_${JID}.out"
