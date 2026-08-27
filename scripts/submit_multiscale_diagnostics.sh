#!/usr/bin/env bash
# Submit next-round multi-scale PushT diagnostics.
#
# Experiments
# -----------
# M1: actual CEM-population fidelity (requires fresh traces with candidates)
# M2: complete CEM center-value trajectory / basin divergence
# M3: near-null physical-vs-encoder-vs-predictor response decomposition
#
# Usage:
#   bash scripts/submit_multiscale_diagnostics.sh smoke
#   NODE=4090node3 bash scripts/submit_multiscale_diagnostics.sh formal
#
# IMPORTANT: M3 uses the previous formal planner-center landscape results to
# identify the suspicious gain-explosion centers. Those directories are kept.
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Usage: bash scripts/submit_multiscale_diagnostics.sh {smoke|formal}" >&2
  exit 2
fi

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
NODE="${NODE:-4090node2}"

LEWM_POLICY="${LEWM_POLICY:-lewm_epoch_10}"
ALD_POLICY="${ALD_POLICY:-pusht_ald_h5_seed3072_ep10_ddp4/lewm_ald_h5_ddp4_epoch_10}"

PREV_LEWM_TRACE="$REPO/outputs/pusht_cem_trace_lewm_formal"
PREV_ALD_TRACE="$REPO/outputs/pusht_cem_trace_ald_formal"

LOG_DIR="$REPO/logs"
GEN_DIR="$REPO/slurm/generated_multiscale_diagnostics"
mkdir -p "$LOG_DIR" "$GEN_DIR"

if [[ "$MODE" == "smoke" ]]; then
  TAG="smoke"
  NUM_EVAL=2
  MAX_SOLVES=4
  MAX_CANDIDATES=64
  NULL_DIRS=16
  NULL_TOP=4
  POP_ITERS="0 5 10 29"
else
  TAG="formal"
  NUM_EVAL=10
  MAX_SOLVES=20
  MAX_CANDIDATES=0
  NULL_DIRS=64
  NULL_TOP=10
  POP_ITERS="0 1 3 5 10 20 29"

  echo "Cleaning this round's validated smoke-only artifacts..."
  rm -rf \
    "$REPO/outputs/pusht_cem_population_trace_lewm_smoke" \
    "$REPO/outputs/pusht_cem_population_trace_ald_smoke" \
    "$PREV_LEWM_TRACE/null_response_decomposition_smoke" \
    "$PREV_ALD_TRACE/null_response_decomposition_smoke"
  rm -f \
    "$LOG_DIR"/multiscale_diag_smoke_*.out \
    "$LOG_DIR"/multiscale_diag_smoke_*.err \
    "$GEN_DIR/multiscale_diag_smoke.slurm"
fi

for d in "$PREV_LEWM_TRACE" "$PREV_ALD_TRACE"; do
  if [[ ! -f "$d/planner_center_landscape/planner_center_metrics.csv" ]]; then
    echo "ERROR: previous formal landscape result missing under $d" >&2
    exit 3
  fi
done

LEWM_TRACE="$REPO/outputs/pusht_cem_population_trace_lewm_$TAG"
ALD_TRACE="$REPO/outputs/pusht_cem_population_trace_ald_$TAG"
SLURM_FILE="$GEN_DIR/multiscale_diag_${TAG}.slurm"

cat > "$SLURM_FILE" <<EOF
#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=ms_$TAG
#SBATCH --output=$LOG_DIR/multiscale_diag_${TAG}_%j.out
#SBATCH --error=$LOG_DIR/multiscale_diag_${TAG}_%j.err

set -euo pipefail

source "$CONDA_SH"
conda activate lewm
cd "$REPO"

export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

python -m py_compile \
  traced_cem.py \
  pusht_trace_eval_utils.py \
  eval_pusht_cem_population_fidelity.py \
  eval_pusht_center_value_trajectory.py \
  eval_pusht_null_response_decomposition.py

LEWM_POLICY="$LEWM_POLICY"
ALD_POLICY="$ALD_POLICY"
LEWM_TRACE="$LEWM_TRACE"
ALD_TRACE="$ALD_TRACE"
PREV_LEWM_TRACE="$PREV_LEWM_TRACE"
PREV_ALD_TRACE="$PREV_ALD_TRACE"

echo "============================================================"
echo "Multi-scale planner diagnostics: $TAG"
echo "node=\$(hostname)"
echo "LeWM population trace=\$LEWM_TRACE"
echo "ALD population trace=\$ALD_TRACE"
echo "============================================================"

echo
echo "=== TRACE-A: LeWM CEM with full populations saved ==="
rm -rf "\$LEWM_TRACE"
mkdir -p "\$LEWM_TRACE"
JEPA_CEM_TRACE_DIR="\$LEWM_TRACE" \
python -u eval.py \
  --config-name=pusht.yaml \
  solver=traced_cem \
  solver.save_candidates=true \
  policy="\$LEWM_POLICY" \
  eval.num_eval=$NUM_EVAL \
  output.filename=multiscale_trace_lewm_$TAG.txt

echo
echo "=== TRACE-B: ALD CEM with full populations saved ==="
rm -rf "\$ALD_TRACE"
mkdir -p "\$ALD_TRACE"
JEPA_CEM_TRACE_DIR="\$ALD_TRACE" \
python -u eval.py \
  --config-name=pusht.yaml \
  solver=traced_cem \
  solver.save_candidates=true \
  policy="\$ALD_POLICY" \
  eval.num_eval=$NUM_EVAL \
  output.filename=multiscale_trace_ald_$TAG.txt

echo
echo "=== M1a: actual population fidelity on LeWM trajectory ==="
python -u eval_pusht_cem_population_fidelity.py \
  --trace-dir "\$LEWM_TRACE" \
  --trace-label lewm_cem \
  --policies "\$LEWM_POLICY" "\$ALD_POLICY" \
  --labels lewm ald \
  --iterations $POP_ITERS \
  --max-solves $MAX_SOLVES \
  --max-candidates $MAX_CANDIDATES

echo
echo "=== M1b: actual population fidelity on ALD trajectory ==="
python -u eval_pusht_cem_population_fidelity.py \
  --trace-dir "\$ALD_TRACE" \
  --trace-label ald_cem \
  --policies "\$LEWM_POLICY" "\$ALD_POLICY" \
  --labels lewm ald \
  --iterations $POP_ITERS \
  --max-solves $MAX_SOLVES \
  --max-candidates $MAX_CANDIDATES

echo
echo "=== M2a: center-value trajectory on LeWM CEM ==="
python -u eval_pusht_center_value_trajectory.py \
  --trace-dir "\$LEWM_TRACE" \
  --trace-label lewm_cem \
  --policies "\$LEWM_POLICY" "\$ALD_POLICY" \
  --labels lewm ald \
  --max-solves $MAX_SOLVES

echo
echo "=== M2b: center-value trajectory on ALD CEM ==="
python -u eval_pusht_center_value_trajectory.py \
  --trace-dir "\$ALD_TRACE" \
  --trace-label ald_cem \
  --policies "\$LEWM_POLICY" "\$ALD_POLICY" \
  --labels lewm ald \
  --max-solves $MAX_SOLVES

echo
echo "=== M3a: near-null decomposition from LeWM formal centers ==="
python -u eval_pusht_null_response_decomposition.py \
  --trace-dir "\$PREV_LEWM_TRACE" \
  --trace-label lewm_cem \
  --policies "\$LEWM_POLICY" "\$ALD_POLICY" \
  --labels lewm ald \
  --gain-threshold 1e4 \
  --top-centers $NULL_TOP \
  --num-directions $NULL_DIRS \
  --output-dir "\$PREV_LEWM_TRACE/null_response_decomposition_$TAG"

echo
echo "=== M3b: near-null decomposition from ALD formal centers ==="
python -u eval_pusht_null_response_decomposition.py \
  --trace-dir "\$PREV_ALD_TRACE" \
  --trace-label ald_cem \
  --policies "\$LEWM_POLICY" "\$ALD_POLICY" \
  --labels lewm ald \
  --gain-threshold 1e4 \
  --top-centers $NULL_TOP \
  --num-directions $NULL_DIRS \
  --output-dir "\$PREV_ALD_TRACE/null_response_decomposition_$TAG"

echo
echo "=== DONE ==="
echo "M1 LeWM: \$LEWM_TRACE/cem_population_fidelity/"
echo "M1 ALD : \$ALD_TRACE/cem_population_fidelity/"
echo "M2 LeWM: \$LEWM_TRACE/center_value_trajectory/"
echo "M2 ALD : \$ALD_TRACE/center_value_trajectory/"
echo "M3 LeWM: \$PREV_LEWM_TRACE/null_response_decomposition_$TAG/"
echo "M3 ALD : \$PREV_ALD_TRACE/null_response_decomposition_$TAG/"
EOF

echo "Submitting $MODE multi-scale diagnostics to $NODE"
echo "Slurm file: $SLURM_FILE"
sbatch "$SLURM_FILE"
