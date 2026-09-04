#!/usr/bin/env bash
# One-shot oracle geometry-capacity test.
# This is intentionally bounded: one run, fixed protocol, no sweep loop.
set -euo pipefail

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
DATA="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV="lewm"
NODE="${NODE:-4090node3}"

RUN_DIR="$REPO/outputs/pusht_official_diagnostic/formal_20260903T024414Z_3108082"
OUT_DIR="$REPO/outputs/oracle_geometry_capacity"
LOG_DIR="$REPO/logs"
GEN_DIR="$REPO/slurm/generated_oracle_geometry_capacity"

if [[ "$NODE" != "4090node2" && "$NODE" != "4090node3" ]]; then
  echo "NODE must be 4090node2 or 4090node3" >&2
  exit 2
fi

cd "$REPO"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

python -m py_compile eval_oracle_geometry_capacity.py

[[ -f "$RUN_DIR/run_identity.json" ]] || {
  echo "Missing authoritative official diagnostic: $RUN_DIR" >&2
  exit 2
}

# Exactly one authoritative output for this test.
rm -rf "$OUT_DIR"
mkdir -p "$LOG_DIR" "$GEN_DIR"

SLURM_FILE="$GEN_DIR/oracle_geometry_capacity.slurm"
cat > "$SLURM_FILE" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=oracle_geom
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=$LOG_DIR/oracle_geometry_capacity_%j.out
#SBATCH --error=$LOG_DIR/oracle_geometry_capacity_%j.err

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
python --version
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

python -u eval_oracle_geometry_capacity.py \
  +diagnostic.run_dir="$RUN_DIR" \
  +diagnostic.output_dir="$OUT_DIR" \
  +diagnostic.cases="[7,27,53,77]" \
  +diagnostic.num_controls=2 \
  +diagnostic.sources="[lewm,ald_tf]" \
  +diagnostic.solves="[0,1]" \
  +diagnostic.iterations="[0,3,9,19,29]" \
  +diagnostic.model_batch_size=32 \
  +diagnostic.seed=3072 \
  +diagnostic.steps_diag=300 \
  +diagnostic.steps_full=500 \
  +diagnostic.lr_diag=0.03 \
  +diagnostic.lr_full=0.003 \
  +diagnostic.pair_margin_frac=0.02 \
  +diagnostic.pair_bank_size=2048 \
  +diagnostic.populations_per_step=8 \
  +diagnostic.pairs_per_pop_step=256 \
  +diagnostic.temperature=0.25 \
  +diagnostic.full_identity_reg=0.0001

echo
echo "===== SUMMARY ====="
cat "$OUT_DIR/summary.json"
echo
echo "===== AGGREGATE ====="
cat "$OUT_DIR/aggregate_summary.csv"
echo
echo "===== FIT DIAGNOSTICS ====="
cat "$OUT_DIR/fit_diagnostics.csv"
EOF

bash -n "$SLURM_FILE"
JID="$(sbatch --parsable "$SLURM_FILE")"

echo "Submitted bounded oracle geometry-capacity test."
echo "job: $JID"
echo "node: $NODE"
echo "output: $OUT_DIR"
echo "stdout: $LOG_DIR/oracle_geometry_capacity_${JID}.out"
echo "stderr: $LOG_DIR/oracle_geometry_capacity_${JID}.err"
echo
echo "squeue -u zsong469 -o '%.18i %.14j %.2t %.10M %.24R'"
echo "tail -f $LOG_DIR/oracle_geometry_capacity_${JID}.out"
