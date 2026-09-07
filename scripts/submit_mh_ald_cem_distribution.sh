#!/usr/bin/env bash
# Actual CEM-distribution diagnostic: ALD vs MH-ALD at N=100,I=10,K=10.
#
# Usage:
#   NODE=4090node3 bash scripts/submit_mh_ald_cem_distribution.sh smoke
#   NODE=4090node3 bash scripts/submit_mh_ald_cem_distribution.sh formal

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/submit_mh_ald_cem_distribution.sh {smoke|formal}" >&2
  exit 2
fi
MODE="$1"
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Usage: bash scripts/submit_mh_ald_cem_distribution.sh {smoke|formal}" >&2
  exit 2
fi

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
NODE="${NODE:-4090node3}"

ALD_POLICY="pusht_ald_h5_seed3072_ep10_ddp4/lewm_ald_h5_ddp4_epoch_10"
MHALD_POLICY="pusht_mh_ald_h5_seed3072_ep10_ddp4/lewm_mh_ald_h5_ddp4_epoch_10"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="$REPO/outputs/mh_ald_cem_distribution/$RUN_TAG"
ALD_TRACE="$OUT_ROOT/trace_ald"
MH_TRACE="$OUT_ROOT/trace_mh_ald"
ALD_CROSS="$OUT_ROOT/cross_on_ald_cem"
MH_CROSS="$OUT_ROOT/cross_on_mh_ald_cem"
COMPARE="$OUT_ROOT/comparison"
LOG_DIR="$REPO/logs"
GEN_DIR="$REPO/slurm/generated_mh_ald_cem_distribution"
mkdir -p "$OUT_ROOT" "$LOG_DIR" "$GEN_DIR"
ln -sfn "$OUT_ROOT" "$REPO/outputs/mh_ald_cem_distribution_latest"

cd "$REPO"
source "$CONDA_SH"
conda activate lewm
python -m py_compile   traced_cem.py   eval_pusht_cem_population_fidelity.py   scripts/summarize_mh_ald_cem_distribution.py

if [[ "$MODE" == "smoke" ]]; then
  NUM_EVAL=2
  MAX_SOLVES=2
  MAX_CAND=30
  ITERS="0 1"
else
  NUM_EVAL=10
  MAX_SOLVES=20
  MAX_CAND=0
  ITERS="0 1 3 5 9"
fi

FILE="$GEN_DIR/cem_distribution_${MODE}_${RUN_TAG}.slurm"
cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --job-name=mhald_dist
#SBATCH --output=$LOG_DIR/mh_ald_dist_${RUN_TAG}_%j.out
#SBATCH --error=$LOG_DIR/mh_ald_dist_${RUN_TAG}_%j.err

set -euo pipefail
source "$CONDA_SH"
conda activate lewm
cd "$REPO"
export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

ALD_POLICY="$ALD_POLICY"
MHALD_POLICY="$MHALD_POLICY"
ALD_TRACE="$ALD_TRACE"
MH_TRACE="$MH_TRACE"
ALD_CROSS="$ALD_CROSS"
MH_CROSS="$MH_CROSS"
COMPARE="$COMPARE"
NUM_EVAL="$NUM_EVAL"
MAX_SOLVES="$MAX_SOLVES"
MAX_CAND="$MAX_CAND"
ITERS="$ITERS"
OUT_ROOT="$OUT_ROOT"
EOF

cat >> "$FILE" <<'EOF'
rm -rf "$ALD_TRACE" "$MH_TRACE" "$ALD_CROSS" "$MH_CROSS" "$COMPARE"
mkdir -p "$ALD_TRACE" "$MH_TRACE" "$ALD_CROSS" "$MH_CROSS" "$COMPARE"

echo "============================================================"
echo "Actual CEM-distribution diagnostic"
echo "CEM N=100 I=10 K=10 B=1000"
echo "Paired seed=42 starts"
echo "Trace both ALD-CEM and MH-ALD-CEM"
echo "Cross-score ALD and MH-ALD on the exact same populations"
echo "============================================================"

echo "=== TRACE ALD-CEM ==="
JEPA_CEM_TRACE_DIR="$ALD_TRACE" CUDA_VISIBLE_DEVICES=0 python -u eval.py   --config-name=pusht.yaml   solver=traced_cem   solver.save_candidates=true   policy="$ALD_POLICY"   seed=42   solver.num_samples=100   solver.n_steps=10   solver.topk=10   eval.num_eval="$NUM_EVAL"   eval.eval_budget=50   output.filename="$OUT_ROOT/ald_trace_closed_loop.txt" &
P0=$!

echo "=== TRACE MH-ALD-CEM ==="
JEPA_CEM_TRACE_DIR="$MH_TRACE" CUDA_VISIBLE_DEVICES=1 python -u eval.py   --config-name=pusht.yaml   solver=traced_cem   solver.save_candidates=true   policy="$MHALD_POLICY"   seed=42   solver.num_samples=100   solver.n_steps=10   solver.topk=10   eval.num_eval="$NUM_EVAL"   eval.eval_budget=50   output.filename="$OUT_ROOT/mh_trace_closed_loop.txt" &
P1=$!

wait "$P0"
wait "$P1"

echo "=== CROSS-SCORE ON ALD-CEM POPULATIONS ==="
CUDA_VISIBLE_DEVICES=2 python -u eval_pusht_cem_population_fidelity.py   --trace-dir "$ALD_TRACE"   --trace-label ald_cem   --policies "$ALD_POLICY" "$MHALD_POLICY"   --labels ald mh_ald   --iterations $ITERS   --max-solves "$MAX_SOLVES"   --max-candidates "$MAX_CAND"   --device cuda:0   --output-dir "$ALD_CROSS" &
P2=$!

echo "=== CROSS-SCORE ON MH-ALD-CEM POPULATIONS ==="
CUDA_VISIBLE_DEVICES=3 python -u eval_pusht_cem_population_fidelity.py   --trace-dir "$MH_TRACE"   --trace-label mh_ald_cem   --policies "$ALD_POLICY" "$MHALD_POLICY"   --labels ald mh_ald   --iterations $ITERS   --max-solves "$MAX_SOLVES"   --max-candidates "$MAX_CAND"   --device cuda:0   --output-dir "$MH_CROSS" &
P3=$!

wait "$P2"
wait "$P3"

python -u scripts/summarize_mh_ald_cem_distribution.py   --ald-cem "$ALD_CROSS"   --mh-cem "$MH_CROSS"   --output-dir "$COMPARE"

echo "=== CEM DISTRIBUTION DONE ==="
echo "ALD trace: $ALD_TRACE"
echo "MH trace : $MH_TRACE"
echo "Compare  : $COMPARE"
EOF

bash -n "$FILE"
JID=$(sbatch --parsable "$FILE")
echo "Submitted MH-ALD CEM-distribution diagnostic ($MODE): $JID"
echo "node: $NODE"
echo "run_root: $OUT_ROOT"
echo "status: squeue -j $JID"
echo "stdout: tail -f $LOG_DIR/mh_ald_dist_${RUN_TAG}_${JID}.out"
echo "stderr: tail -f $LOG_DIR/mh_ald_dist_${RUN_TAG}_${JID}.err"
